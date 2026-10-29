# Slot Lockup Fixes — Round 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- An index pass no longer blocks the event loop for ~30s (I1).
- Every vault file is searchable under `vault` (I2).
- The orphaned `notes` collection is deleted once its documents have moved.

**Architecture:**
- An index on `documents(collection, file_path)`, and a `get_hash` that requires the collection.
- `index_collection` yields to the event loop every 50 files.
- After startup indexing has finished, a background task prunes collections that `collections.json` no longer lists.

**Tech Stack:** Python 3.12, sqlite + sqlite-vec, FastAPI 0.135.2 / Starlette 1.0.0, pytest 9.0.2 + pytest-asyncio 1.3.0.

**Spec:** `docs/audit/findings.md`, section "Found after the round-2 deploy" (I1, I2). The user decision of 2026-10-28: delete `notes` after its documents have moved.

## Implementer Rules (local model via opencode)

Same as rounds 1 and 2:

1. **Do one task, then stop** and report: the diff summary, the exact test command run, and its pass/fail counts.
2. **Never edit a test to make it pass.** The round-3 tests were validated in advance: they fail against `2adbc71` in the ways each task predicts, and pass against a reference fix. The only sanctioned test edits are the three lines named in Task 1, Step 3.
3. **Stop and report, don't improvise**, if any test fails differently from the plan's "Expected", or passes when the plan says it should fail.
4. **Never** run `sudo`, `systemctl`, `scripts/deploy-manager.sh`, `git push`, `gh`, or anything that talks to `192.168.1.14`, `127.0.0.1:8081-8084` or `:11434`. Never open `/opt/llama/data/skills.db`.
5. Stage files by explicit path only (never `git add -A` or `git add .`), and run `git status --short` before each commit. Never stage `config/manager.env.orig`.
6. Run tests only with the command in Global Constraints. Don't install packages or create venvs.
7. Change only the files listed in the task. If another file seems to need a change, stop and report.

## Global Constraints

- Branch: `fix/slot-lockup-round3`, from `origin/main` @ `2adbc71` (deployed). Line numbers below refer to `2adbc71`.
- No dependency changes, and no new tables. The new index is created with `CREATE INDEX IF NOT EXISTS` inside `init_schema`, so the live database gains it on the next start.
- Run tests with: `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest <target> -q -p no:cacheprovider`

## Review Focus

1. **Pruning must never run on a config that wasn't read.** `load_collections_config` returns `[]` for a *missing* file, so "no entries" must mean "do not prune", not "prune everything". A wrong path in `manager.env` would otherwise wipe the index. Pinned by `test_an_unreadable_or_empty_config_prunes_nothing` and `test_a_missing_config_prunes_nothing`.
2. **Move first, delete after.** Pruning waits for every startup index job (`ReindexRegistry.wait_idle`), so a document is never missing while its replacement is still being embedded. Pinned by the first assertion in `test_orphaned_collection_is_pruned_after_its_documents_move`.
3. **The pending prune is cancelled on shutdown** before the registry and DB close. Pinned by `test_shutdown_with_a_pending_prune_is_prompt`.
4. **An unchanged pass must yield.** With the index the lookups are fast (0.041s for 2,937 on the live data), but a pass with no awaits still holds the loop. Pinned by `test_an_unchanged_index_pass_yields_to_the_event_loop`.
5. **A prune that fails says so, and a failed startup pass blocks it** (Task 2b). Without a handler, the prune task's exception is discarded at shutdown by `gather(return_exceptions=True)`. `wait_idle` waits for jobs to finish, not to succeed, so with the embeddings server down the vault job would end with errors and `notes` would be deleted before its 695 documents moved. Pinned by `test_a_failed_prune_is_logged` and both cases of `test_a_failed_startup_index_job_prunes_nothing`.
6. Pruning deletes vector rows too, so no orphaned `documents_vec` rows are left. Pinned by `test_delete_collection_removes_its_documents_vectors_and_row`.

Validated on a copy of the live DB, with a stub embedder: a full vault pass moved exactly 695 documents to `vault` in 4.4s, leaving 2,937 `vault` rows to match 2,937 files. Pruning `notes` then removed its 443 stale rows, with 0 orphaned vector rows.

---

### Task 0: Land the round-3 tests and docs

**Files:** `tests/test_index_scoping.py` (new), `tests/test_orphan_pruning.py` (new), `docs/audit/findings.md` (I1/I2 and status), this plan.

- [ ] **Step 1:** `git branch --show-current` prints `fix/slot-lockup-round3`, and `git log -1 --format=%h` prints `2adbc71`.
- [ ] **Step 2:** Run `tests/`. Expected: **7 failed, 287 passed**. The 7 failures: the 6 in `test_index_scoping.py`, plus `test_orphaned_collection_is_pruned_after_its_documents_move`.
- [ ] **Step 3:** Stage the four files by path and commit `test: round-3 tests (failing until Task 2); docs: I1, I2 and round-3 plan`.

### Task 1: Indexed, collection-scoped lookups; a yielding index pass (I1, I2)

**Files:** Modify `manager/vectordb.py` (`init_schema:33`, `get_hash:172`, a new `delete_collection` before `list_collections:196`), `manager/collections.py` (loop at `:203`, call at `:208`, imports), `tests/test_vectordb.py` and `tests/test_collections.py` (sanctioned edits only).

**Interfaces:**
- `init_schema` also runs `CREATE INDEX IF NOT EXISTS idx_documents_collection_file_path ON documents(collection, file_path)`.
- `get_hash(self, collection: str, file_path: str) -> str | None` filters on both columns. The collection is a required first argument, so an old one-argument call fails loudly.
- `delete_collection(self, collection: str) -> int` deletes that collection's `documents_vec` rows, its `documents` rows and its `collections` row, commits, and returns the number of documents removed.
- `index_collection` passes `collection_id` to `get_hash`, and calls `await asyncio.sleep(0)` once every 50 files of the loop, whether or not the file changed.

- [ ] **Step 1:** Tests exist: all 6 in `tests/test_index_scoping.py`. Expected:
  - the two index tests: `['collection', 'file_path'] in [['id']]`;
  - `..._yields_to_the_event_loop`: `assert 0 >= 2`;
  - `test_get_hash_is_scoped_to_the_collection`: `TypeError: VectorDB.get_hash() takes 2 positional arguments but 3 were given`;
  - `..._indexed_under_another_collection_is_indexed_here`: `assert (0 == 1)`;
  - `test_delete_collection...`: `AttributeError ... 'delete_collection'`.
- [ ] **Step 2:** Implement per the Interfaces block.
- [ ] **Step 3: Sanctioned test edits** (the signature change; nothing else in these tests):
  - `tests/test_vectordb.py:113`: `db.get_hash("/tmp/test.md")` becomes `db.get_hash("skills", "/tmp/test.md")`;
  - `tests/test_vectordb.py:114`: `db.get_hash("/tmp/nonexistent.md")` becomes `db.get_hash("skills", "/tmp/nonexistent.md")`;
  - `tests/test_collections.py:188`: `db.get_hash(str(Path(notes_dir) / "a.md"))` becomes `db.get_hash("notes", str(Path(notes_dir) / "a.md"))`.
- [ ] **Step 4:** Run `tests/`. Expected: all pass except `test_orphaned_collection_is_pruned_after_its_documents_move`.
- [ ] **Step 5:** Commit `fix(index): indexed, collection-scoped hash lookups; index passes yield`.

### Task 2: Prune unconfigured collections after startup indexing

**Files:** Modify `manager/app.py` (a new module-level coroutine near `_collection_indexer:424`; `lifespan:839-884`), `README.md` (the collections setup note at `:462`).

**Interfaces:**
- `async def _prune_orphaned_collections(server, keep: set[str]) -> None`:
  - `await server.reindex_registry.wait_idle()`;
  - then, for every collection in `server.db.list_collections()` whose `id` isn't in `keep`, call `server.db.delete_collection(id)`;
  - log one `INFO` line per pruned collection, naming it, its document count, and that it isn't in the collections config.
- `lifespan`:
  - Only when `load_collections_config` succeeded **and** returned a non-empty list, start `prune_task = asyncio.create_task(_prune_orphaned_collections(server, {e["id"] for e in entries}))` after starting the index jobs.
  - Otherwise there is no task: a missing, empty or unreadable config prunes nothing.
  - Initialize `prune_task = None` before the collections block.
- `lifespan` teardown: if `prune_task` is set, cancel it and await it (`asyncio.gather(prune_task, return_exceptions=True)`) **before** `await server.reindex_registry.shutdown()`.
- README, at the restart note: collections no longer listed in `collections.json` are deleted from the index after startup indexing finishes; a missing or empty config deletes nothing.

- [ ] **Step 1:** Tests exist: all 5 in `tests/test_orphan_pruning.py`. Expected:
  - `test_orphaned_collection_is_pruned_after_its_documents_move` fails with `assert ['notes', 'skills'] == ['skills']`;
  - the other four **pass** (they pin that nothing is pruned when it shouldn't be).
- [ ] **Step 2:** Implement per the Interfaces block.
- [ ] **Step 3:** Run `tests/`. Expected: **all pass**.
- [ ] **Step 4:** Commit `fix(collections): prune unconfigured collections after startup indexing`.

### Task 2b: A failed prune is logged; a failed startup pass skips the prune

Found reviewing Task 2. Both gaps came from the plan, not from the Task 2 implementation, which matched the plan.

**Files:** Modify `manager/app.py` (`_prune_orphaned_collections` only). Commit `tests/test_orphan_pruning.py` as given.

**Interfaces** (the signature and the lifespan are unchanged):
- After `await server.reindex_registry.wait_idle()`, for every id in `keep`, take `job = server.reindex_registry.get_current(id)`. The startup pass for that id failed if `job` is `None` (it never started), `job.status != "done"`, or `job.stats.get("errors", 0) > 0`.
- If any failed: log **one `WARNING`** from the `manager.app` logger that names every failed collection id and says nothing was pruned, then return without deleting anything.
- Wrap everything after `wait_idle` in `try: ... except Exception: logger.exception(...)`, with a message containing the word `prune`. Catch `Exception` only. `CancelledError` must still propagate, so shutdown can cancel the task.

- [ ] **Step 1:** Commit the test file and this plan as they are in the working tree: `test: a failed prune is logged; a failed startup pass skips the prune`. Then run `tests/`. Expected: **3 failed, 294 passed**:
  - `test_a_failed_prune_is_logged`: `the failed prune left no ERROR record in manager.app`;
  - both cases of `test_a_failed_startup_index_job_prunes_nothing`: `assert 'notes' in ['skills']`.
- [ ] **Step 2:** Implement per the Interfaces block.
- [ ] **Step 3:** Run `tests/`. Expected: **all pass**.
- [ ] **Step 4:** Commit `fix(collections): log a failed prune; skip pruning when a startup pass failed`.

### Task 3: PR, merge, deploy, verify (Claude and the user; not the implementer)

- [ ] **Step 1:** Push, open the PR, user merges.
- [ ] **Step 2:** `git switch main && git pull --ff-only`, then deploy **from `main`**, so `DEPLOYED_FROM` names the merge commit. Run `--check`, then `sudo ./scripts/deploy-manager.sh` with opencode idle.
- [ ] **Step 3:** Verify on the host:
  - provenance: the stamp and every file's sha256 match the merge;
  - "Application startup complete" within a few seconds of the start;
  - `manager.log` shows the vault job with `new` ≈ 695 and `errors: 0`, then `pruned collection 'notes' (443 documents)`;
  - no WARNING that pruning was skipped (if there is one, the vault pass had errors: read them before going further);
  - `GET /collections` lists no `notes`, and `vault`'s `doc_count` equals its file count on disk;
  - both slots pass a smoke request.
