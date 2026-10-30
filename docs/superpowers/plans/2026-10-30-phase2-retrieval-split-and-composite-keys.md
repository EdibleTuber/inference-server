# Phase 2: Retrieval Split and Composite Document Keys — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the collections routes and indexing orchestration out of `manager/app.py` (Q28). Then make document ids unique per collection rather than globally (Q23), migrating the live `skills.db` once, with a backup taken first.

**Architecture:** There are two PRs, as the spec requires ("a pure move in its own PR").

- **PR A** adds a flat module, `manager/retrieval.py`, that holds the six `/collections` routes as an `APIRouter`, along with explicit `start_retrieval` and `stop_retrieval` hooks that `app.py`'s lifespan calls. No behaviour changes.
- **PR B** changes `manager/vectordb.py`:
  - documents get an integer surrogate key and `UNIQUE(collection, id)`;
  - vectors move to a new vec0 table, `doc_vectors`, partitioned by `collection` and keyed by the document's integer key;
  - an existing database (schema v0) is migrated in one transaction at `init_schema()`, after a backup file is written.

**Tech Stack:** Python 3.12.3, FastAPI 0.135.2, Starlette 1.0.0, sqlite-vec 0.1.9 and SQLite 3.45.1. These are the deployed venv's versions, read on 2026-10-30.

**Spec:** `docs/design-review-2026-10-29.md`: "API surface and retrieval" (Q28, Q23) and the phase table (phase 2, review weight Medium).

## Global Constraints

- **Test command:** `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/ -q -p no:cacheprovider`. The suite must stay green after every task.
- **Live ports:** tests never reach live ports 8080–8084 or 11434. The conftest socket guard enforces this; don't weaken it.
- **Flat modules only:** new modules go directly in `manager/`, with no subpackages. `scripts/deploy-manager.sh:143` installs `manager/*.py` flat into `/opt/llama/manager/`, so a subpackage would never be deployed.
- **Client API unchanged:** no HTTP response shape changes in either PR. Document ids in responses stay the relative path (`"Security/Recon/SKILL"`), and clients (agent_core, PAL) don't change.
- **Do not deploy between Tasks 3 and 4.** After Task 3 the code refuses a v0 database, so collection endpoints would return 503 until Task 4 lands. Inference is unaffected either way.
- **Commits:** stage explicit paths and never use `git add -A`. Commit messages end with the repo's attribution lines.

## Implementer Rules (Qwen)

1. Do exactly one task, then stop and report: "Done" plus a **Deviations** list. Write "none" if there are none.
2. Never edit a test except where a step names it as a **sanctioned edit**. If an existing test fails, stop and report it; don't fix it.
3. Stop and report if a step's "Expected" result doesn't happen, including a test that passes when it should fail.
4. Never run sudo, systemctl, deploy scripts, `git push`, or anything that touches ports 8080–8084 or 11434.
5. Code in this plan is a sketch to be verified against the real files. If it is wrong, deviating is correct: say so in Deviations.

## Review Focus

These are the failure modes most likely to reach the live system. Each has a test in the task named after the arrow.

1. A stale `patch("manager.app.EmbeddingsClient")` that silently stops patching after the move. Tests would then build a real embeddings client. → Task 1 `test_app_no_longer_holds_retrieval_names` makes a stale target raise `AttributeError` instead.
2. Shutdown order changes in the move: the DB closes while a reindex job or the prune task is still using it. → Task 1 keeps the order (prune cancel → registry shutdown → embeddings close → DB close → consumers cancelled). The existing `tests/test_orphan_pruning.py` and `tests/test_background_indexing.py` must pass with no edits other than the patch-target and logger-name strings.
3. A migration that half-applies: the new tables are created, then an error occurs, leaving neither schema usable. → Task 4 `test_failed_migration_rolls_back_to_the_legacy_database` injects a failure after the copies.
4. Migration with no backup. → Task 4 `test_migration_backs_up_the_legacy_database_first`.
5. A failed migration taking inference down with it. → Task 4 `test_a_failed_migration_leaves_inference_up`.

Not covered, on purpose: tag filtering still post-filters `max(limit*20, 100)` candidates within a collection (`vectordb.py:117`), so a rare tag in a big collection can still come back short. That's out of scope here; note it in findings.md (Task 4 step 6).

## Findings this plan fixes (verified 2026-10-30)

- **Q23 (pre-existing, latent):** `documents.id` is a global `PRIMARY KEY` (`manager/vectordb.py:43-44`), and `upsert_document` resolves conflicts with `ON CONFLICT(id) DO UPDATE SET collection=excluded.collection` (`vectordb.py:91-96`). If two collections hold the same relative path, the second upsert moves the row into its own collection and overwrites the shared vector. Each reindex then moves it back. There are no collisions on the host today: none of the four configured source dirs share a relative path. They would start as soon as `learnings/` or `relationships/` gained a `README.md`.
- **Search starvation (pre-existing, live):** `search()` takes the global top `k = max(limit*20, 100)` vectors across all collections, then drops the rows from other collections (`vectordb.py:117-135`). `vault` has 2915 docs, so `skills` (2 docs) is usually outside the top 100. Live today: `POST /collections/skills/search {"query":"how to write a plan for skills"}` returns `{"results":[]}`. Fix: the new vec table has `collection TEXT PARTITION KEY`, so the KNN runs inside one collection. This is supported in sqlite-vec 0.1.9, and was verified on a copy of the live DB.
- **`get_children` LIKE wildcards (pre-existing):** `collections.py:158` matches `id LIKE 'A_B/%'`. `_` is a LIKE wildcard, so `AxB/Workflows/y` counts as a child of `A_B/SKILL`. Fix: compare prefixes literally.

---

## PR A — branch `phase2a/retrieval-split`

### Task 1: Move collections routes and indexing orchestration to `manager/retrieval.py`

**Files:**
- Create: `manager/retrieval.py`
- Modify: `manager/app.py`. Remove `_collection_indexer` (`app.py:423-440`), `_prune_orphaned_collections` (`app.py:443-480`), the retrieval block of the lifespan (`app.py:914-966` startup and `app.py:970-981` shutdown), and the six collection routes (`app.py:1138-1285` and `app.py:1363-1384`). Also remove the imports that become unused: `EmbeddingsClient`, `VectorDB`, `index_collection`, `load_collections_config` and `get_children`. `ServerState` keeps the attributes `db`, `embeddings_client` and `reindex_registry` and still constructs `ReindexRegistry()`, so that import stays. The `self.db: VectorDB | None` and `self.embeddings_client: EmbeddingsClient | None` annotations in `ServerState.__init__` (`app.py:88-89`) still name the two classes: import them under `typing.TYPE_CHECKING` in `app.py`, which keeps them off the module at runtime as `test_app_no_longer_holds_retrieval_names` requires.
- Create: `tests/test_retrieval_split.py`
- Sanctioned test edits, patch-target strings only:
  - `tests/conftest.py:207`, `tests/test_background_indexing.py:48`, `tests/test_background_indexing.py:180` and `tests/test_orphan_pruning.py:81`: `"manager.app.EmbeddingsClient"` becomes `"manager.retrieval.EmbeddingsClient"`.
  - `tests/test_orphan_pruning.py:221`: `"manager.app.index_collection"` becomes `"manager.retrieval.index_collection"`.
  - `tests/test_orphan_pruning.py:173,190,225`: the logger name `"manager.app"` becomes `"manager.retrieval"`, because the prune logs now come from the new module. (Added after Task 1: the plan originally missed these, and Qwen was right to change them.)

**Interfaces:**
- Produces, in `manager/retrieval.py`:
  - `def make_router(server: "ServerState") -> APIRouter`: the six routes, with bodies moved verbatim. `GET /collections/{collection_id}/reindex/status` must be registered **before** `GET /collections/{collection_id}/reindex/{job_id}`, which `tests/test_collection_endpoints.py:156` pins.
  - `async def start_retrieval(server: "ServerState", config: ManagerConfig) -> asyncio.Task | None`: the moved startup block. It returns the prune task, or `None`.
  - `async def stop_retrieval(server: "ServerState", prune_task: asyncio.Task | None) -> None`: the moved shutdown block, in the same order (cancel and await prune, `reindex_registry.shutdown()`, close embeddings, close DB).
  - `_collection_indexer` and `_prune_orphaned_collections`, moved unchanged.
- Import `ServerState` only under `typing.TYPE_CHECKING`; `app.py` imports `retrieval`, so a runtime import would be circular.
- `app.py` lifespan order after the move:
  1. startup probe;
  2. queue consumers started;
  3. `prune_task = await start_retrieval(server, config)`;
  4. `yield`;
  5. `await stop_retrieval(server, prune_task)`;
  6. consumers cancelled.

  This is the same order as today.
- `app.include_router(make_router(server))`, placed where the first collection route used to be.

- [ ] **Step 1: Write the failing test** `tests/test_retrieval_split.py`

```python
"""Q28: collections routes and indexing live in manager/retrieval.py.

The absence checks are what make a stale test patch loud: once app.py no
longer has these names, patch("manager.app.EmbeddingsClient") raises
AttributeError instead of silently patching nothing and letting a test build
a real embeddings client.
"""
import manager.app as app_module


def test_app_no_longer_holds_retrieval_names():
    for name in ("EmbeddingsClient", "VectorDB", "index_collection",
                 "load_collections_config", "get_children",
                 "_collection_indexer", "_prune_orphaned_collections"):
        assert not hasattr(app_module, name), name


def test_retrieval_module_provides_the_router_and_hooks():
    from manager import retrieval
    assert callable(retrieval.make_router)
    assert callable(retrieval.start_retrieval)
    assert callable(retrieval.stop_retrieval)


def test_collection_routes_are_unchanged(collection_config):
    from manager.app import create_app
    app = create_app(collection_config)
    routes = [(r.path, sorted(r.methods)) for r in app.routes
              if r.path.startswith("/collections")]
    assert routes == [
        ("/collections", ["GET"]),
        ("/collections/{collection_id}/search", ["POST"]),
        ("/collections/{collection_id}/docs/{doc_id:path}", ["GET"]),
        ("/collections/{collection_id}/reindex", ["POST"]),
        ("/collections/{collection_id}/reindex/status", ["GET"]),
        ("/collections/{collection_id}/reindex/{job_id}", ["GET"]),
    ]
```

- [ ] **Step 2: Run it and check that it fails**

Run: `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/test_retrieval_split.py -q -p no:cacheprovider`
Expected: `test_app_no_longer_holds_retrieval_names` and `test_retrieval_module_provides_the_router_and_hooks` FAIL. `test_collection_routes_are_unchanged` PASSES: it is the before/after invariant.

- [ ] **Step 3: Create `manager/retrieval.py` and cut `app.py` over**, per the Interfaces block. Move the code; don't rewrite it. Comments and docstrings move with their code. Update the "Used by both startup (lifespan) and the POST /reindex endpoint" docstring only if its wording becomes false.

- [ ] **Step 4: Apply the five sanctioned patch-target edits** listed under Files.

- [ ] **Step 5: Run the full suite**

Run: the test command in Global Constraints.
Expected: everything passes, including all three tests in `tests/test_retrieval_split.py`.

- [ ] **Step 6: Commit**

```bash
git add manager/retrieval.py manager/app.py tests/test_retrieval_split.py tests/conftest.py tests/test_background_indexing.py tests/test_orphan_pruning.py
git commit -m "refactor: move collections routes and indexing to manager/retrieval.py (Q28)"
```

### Task 2: Ship PR A (Claude and the user; no Qwen)

- [ ] Claude reviews the move as a move. It diffs each moved function body against its `app.py` original, with whitespace-only and indentation-only differences allowed. It also runs the suite with the live-port probe and expects `LIVE-PORT ATTEMPTS: 1`.
- [ ] Open the PR, merge it, then fast-forward the main checkout.
- [ ] User: `! sudo /mnt/secondary/inference-server/scripts/deploy-manager.sh`. Claude then checks that `deploy-manager.sh --check` reports all current, including `retrieval.py`. It also checks that `GET /collections` matches the doc counts captured before the deploy, and that one `POST /collections/vault/search` returns results.

---

## PR B — branch `phase2b/composite-keys` (from main after PR A)

### Task 3: Composite document keys and a collection-partitioned vector table

**Files:**
- Modify: `manager/vectordb.py`, `manager/collections.py:149-162` (`get_children`)
- Create: `tests/test_composite_keys.py`
- Sanctioned test edit: in `tests/test_index_scoping.py:148`, replace
  `vec_ids = {r[0] for r in db._conn.execute("SELECT id FROM documents_vec").fetchall()}`
  with
  `vec_ids = {r[0] for r in db._conn.execute("SELECT d.id FROM doc_vectors v JOIN documents d ON d.pk = v.rowid").fetchall()}`.
  The assertion that follows it (`== {"k"}`) stays.
  (After review: the join was a mistake in the plan. A leaked vector's document row is already gone, so the join hid the leak. `1a2bcae` replaced it with a raw rowid count. Also, `tests/test_orphan_pruning.py:112` needed the same table rename, and Qwen correctly changed it to count the `notes` partition.)

**Interfaces:**
- Produces, in `manager/vectordb.py`:
  - `SCHEMA_VERSION = 1`, stored in `PRAGMA user_version`.
  - `class SchemaVersionError(RuntimeError)`.
  - Schema v1. **This exact DDL is the contract Task 4 migrates to**:
    ```sql
    CREATE TABLE documents (
        pk          INTEGER PRIMARY KEY,
        collection  TEXT NOT NULL REFERENCES collections(id),
        id          TEXT NOT NULL,
        name        TEXT NOT NULL,
        metadata    TEXT,
        summary     TEXT NOT NULL,
        content     TEXT NOT NULL,
        file_path   TEXT NOT NULL,
        file_hash   TEXT NOT NULL,
        updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(collection, id)
    );
    CREATE VIRTUAL TABLE doc_vectors USING vec0(
        collection TEXT PARTITION KEY,
        embedding  float[768]
    );   -- doc_vectors.rowid = documents.pk
    CREATE INDEX idx_documents_collection_file_path ON documents(collection, file_path);
    ```
    `collections` is unchanged.
  - `init_schema()` dispatches on `user_version` and on whether a `documents` table exists:
    - `0` with no `documents` table: create v1 and set `user_version = 1`.
    - `0` with a `documents` table (legacy): raise `SchemaVersionError("legacy schema v0; migration not implemented")`. Task 4 replaces this branch.
    - `1`: run `CREATE INDEX IF NOT EXISTS` for the index above, which keeps `test_init_schema_adds_the_index_to_an_existing_database` green.
    - anything else: raise `SchemaVersionError`.
  - Public method signatures are unchanged: `upsert_collection`, `upsert_document`, `search`, `get_document`, `get_hash`, `delete_stale`, `delete_collection`, `list_collections`, `close`. Their behaviour:
    - `upsert_document` is one transaction. It runs `INSERT … ON CONFLICT(collection, id) DO UPDATE …` with `RETURNING pk` (SQLite 3.45 supports it), then deletes and reinserts `doc_vectors` at `rowid = pk` with the collection.
    - `search` adds `AND collection = ?` to the KNN query. It keeps `k = max(limit * 20, 100)` because the tag filter still runs afterwards, and fetches each candidate's document by `pk`. The result dict and the score formula are unchanged.
    - `delete_stale` and `delete_collection` delete vectors by `pk`.
  - New: `def children(self, collection: str, skill_doc_id: str) -> list[dict]`. It returns every document in `collection` whose `id` starts **literally** with `skill_doc_id.rsplit("/", 1)[0] + "/"`, other than `skill_doc_id` itself, as `{"id", "name", "summary"}`. Use `substr(id, 1, ?) = ?`, not `LIKE`.
- Modify: `collections.get_children(db, collection, skill_doc_id)` keeps its signature and becomes `return db.children(collection, skill_doc_id)`, so it no longer reaches into `db._conn`.

- [ ] **Step 1: Write the failing tests** `tests/test_composite_keys.py`

```python
"""Q23: document ids are unique per collection, not globally; search runs
inside one collection.

Before: documents.id was a global PRIMARY KEY (vectordb.py:43-44), so two
collections holding the same relative path shared one row and one vector,
and search() took a global top-k before filtering by collection, so a small
collection beside a large one came back empty.
"""
from unittest.mock import AsyncMock

import pytest

from manager.collections import get_children, index_collection
from manager.vectordb import VectorDB

DIM = 768


def unit(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


@pytest.fixture
def db(tmp_path):
    d = VectorDB(str(tmp_path / "ck.db"))
    d.init_schema()
    yield d
    d.close()


def _two_readmes(db):
    db.upsert_collection("a", "/a", "markdown")
    db.upsert_collection("b", "/b", "markdown")
    db.upsert_document("README", "a", "A", {}, "sa", "content a", "/a/README.md", "ha", unit(0))
    db.upsert_document("README", "b", "B", {}, "sb", "content b", "/b/README.md", "hb", unit(1))


def test_same_id_in_two_collections_keeps_both(db):
    _two_readmes(db)
    assert db.get_document("a", "README")["content"] == "content a"
    assert db.get_document("b", "README")["content"] == "content b"
    assert db.get_hash("a", "/a/README.md") == "ha"
    assert db.get_hash("b", "/b/README.md") == "hb"
    assert {c["id"]: c["doc_count"] for c in db.list_collections()} == {"a": 1, "b": 1}


def test_each_collection_searches_its_own_vector(db):
    _two_readmes(db)
    [hit_a] = db.search("a", unit(0), limit=5)
    [hit_b] = db.search("b", unit(1), limit=5)
    assert (hit_a["collection"], hit_a["score"]) == ("a", pytest.approx(1.0))
    assert (hit_b["collection"], hit_b["score"]) == ("b", pytest.approx(1.0))


@pytest.mark.asyncio
async def test_reindexing_two_collections_with_a_shared_path_is_a_no_op(tmp_path, db):
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        (d / "README.md").write_text(f"# {name}\n\nAbout {name}.\n")
    emb = AsyncMock()
    emb.embed_text = AsyncMock(return_value=unit(0))
    for name in ("a", "b"):
        await index_collection(collection_id=name, source_dir=str(tmp_path / name),
                               doc_type="markdown", db=db, embeddings=emb)
    emb.embed_text.reset_mock()

    stats = [await index_collection(collection_id=name, source_dir=str(tmp_path / name),
                                    doc_type="markdown", db=db, embeddings=emb)
             for name in ("a", "b")]

    assert [s["unchanged"] for s in stats] == [1, 1]
    assert emb.embed_text.await_count == 0


def test_a_small_collection_is_searchable_beside_a_large_one(db):
    db.upsert_collection("big", "/big", "markdown")
    db.upsert_collection("small", "/small", "markdown")
    for i in range(150):
        db.upsert_document(f"n{i}", "big", f"N{i}", {}, "s", "c", f"/big/n{i}.md", "h", unit(0))
    db.upsert_document("only", "small", "Only", {}, "s", "c", "/small/only.md", "h", unit(1))

    hits = db.search("small", unit(0), limit=5)

    assert [h["id"] for h in hits] == ["only"]
    assert hits[0]["score"] == pytest.approx(0.0, abs=1e-6)   # orthogonal; float32 distances


def test_children_match_the_prefix_literally(db):
    db.upsert_collection("skills", "/s", "skill")
    for doc_id in ("A_B/SKILL", "A_B/Workflows/x", "AxB/Workflows/y"):
        db.upsert_document(doc_id, "skills", doc_id, {}, "s", "c", f"/s/{doc_id}.md", "h", unit(0))

    assert [c["id"] for c in get_children(db, "skills", "A_B/SKILL")] == ["A_B/Workflows/x"]


def test_a_fresh_database_is_created_at_the_current_schema_version(db):
    from manager.vectordb import SCHEMA_VERSION
    assert SCHEMA_VERSION == 1
    assert db._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_a_newer_schema_version_is_refused(tmp_path):
    from manager.vectordb import SchemaVersionError
    path = str(tmp_path / "future.db")
    first = VectorDB(path)
    first.init_schema()
    first._conn.execute("PRAGMA user_version = 2")
    first._conn.commit()
    first.close()

    again = VectorDB(path)
    with pytest.raises(SchemaVersionError):
        again.init_schema()
    again.close()
```

- [ ] **Step 2: Run them and check that they fail**

Run: `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/test_composite_keys.py -q -p no:cacheprovider`
Expected: all 7 FAIL against the current code. `SCHEMA_VERSION` and `SchemaVersionError` are imported inside their tests on purpose, so the other five fail on behaviour, not on a module-level `ImportError`.

- [ ] **Step 3: Implement schema v1 in `manager/vectordb.py` and `VectorDB.children`, and delegate `collections.get_children`**, per the Interfaces block.

- [ ] **Step 4: Apply the sanctioned edit** to `tests/test_index_scoping.py:148`.

- [ ] **Step 5: Run the full suite**

Expected: everything passes.

- [ ] **Step 6: Commit**

```bash
git add manager/vectordb.py manager/collections.py tests/test_composite_keys.py tests/test_index_scoping.py
git commit -m "feat(vectordb): per-collection document keys; collection-partitioned vectors (Q23)"
```

### Task 4: One-time v0 → v1 migration of an existing database, backed up first

**Files:**
- Modify: `manager/vectordb.py`, plus a `findings.md` status line (step 6)
- Create: `tests/test_schema_migration.py`

**Interfaces:**
- Consumes: Task 3's v1 DDL, `SCHEMA_VERSION` and `SchemaVersionError`.
- Produces, in `manager/vectordb.py`:
  - `class MigrationError(RuntimeError)`.
  - `def _verify_migration(conn: sqlite3.Connection) -> None`, at module level, which tests monkeypatch. It runs after both copies and before anything legacy is dropped. It raises `MigrationError` unless `count(documents_v1) == count(documents)` and `count(doc_vectors) == count(documents_v1)`. The second check means every legacy document got its vector. A document without one would never be re-embedded: its hash still matches, so it would stay unsearchable forever.
  - `VectorDB.__init__` keeps `db_path`. The backup path is `f"{db_path}.pre-v1.bak"`.
- `init_schema()`'s legacy branch (v0 with a `documents` table) now migrates. **The order is the invariant**:
  1. **Backup.** If the `.pre-v1.bak` file doesn't exist, write it with `self._conn.backup(sqlite3.connect(bak_path))`, then close that connection. If it exists, leave it alone. It can only have been written by an earlier attempt that rolled back, so it is the legacy DB. If the backup raises, the error propagates and nothing is migrated.
  2. `BEGIN IMMEDIATE`.
  3. Create `documents_v1` with Task 3's `documents` DDL, then copy every legacy row into it, letting `pk` be assigned.
  4. Create `doc_vectors` with Task 3's DDL. Copy the vectors with `INSERT INTO doc_vectors(rowid, collection, embedding) SELECT d.pk, d.collection, v.embedding FROM documents_v1 d JOIN documents_vec v ON v.id = d.id`.
  5. `_verify_migration(self._conn)`.
  6. Drop `documents_vec`, drop `documents` (its index goes with it), rename `documents_v1` to `documents`, and create `idx_documents_collection_file_path`.
  7. `PRAGMA user_version = 1`, then `COMMIT`.
  8. `logger.info("skills.db schema migrated v0 -> v1: %d documents, %d vectors; backup at %s", …)`. The host step greps for this line.

  Any exception in steps 2–7 runs `ROLLBACK` and re-raises. Verified on a copy of the live DB with sqlite-vec 0.1.9: vec0 `CREATE VIRTUAL TABLE` and `DROP TABLE` roll back cleanly inside a transaction, and the full migration of 2917 documents takes about 1.3 s.
- The connection's transaction handling: Python's `sqlite3` with the default `isolation_level` does **not** open a transaction for DDL. Without an explicit `BEGIN`, every `CREATE`/`DROP` autocommits on its own, and `test_failed_migration_rolls_back_to_the_legacy_database` is the test that catches that. Choose either an explicit `BEGIN IMMEDIATE`/`COMMIT`/`ROLLBACK` on the existing connection, or `isolation_level=None` with explicit transactions in every mutating method. Whichever you choose, `upsert_document`, `delete_stale` and `delete_collection` must each stay a single transaction, and **each must roll back on any exception before re-raising**.
  - This was added after Task 3's review. It fixes a pre-existing bug, verified on the Task 3 commit. A vector insert that raises leaves the document row in an open transaction, and the next write's `commit()` commits it, so the document is stored with its hash and no vector. Every later reindex sees an unchanged hash and skips it, so it can never be found by search. It would also make `_verify_migration` abort a later migration.
  - The test is `test_a_failed_upsert_leaves_no_half_written_document`.
- No change is needed in the lifespan. `start_retrieval` (Task 1) already catches any exception from `init_schema()`, sets `server.db = None`, and logs it, so a failed migration leaves collection endpoints at 503 while inference keeps serving.

- [ ] **Step 1: Write the failing tests** `tests/test_schema_migration.py`

```python
"""Q23: an existing (v0) skills.db is migrated once to schema v1, after a
backup, in one transaction.

LEGACY_DDL is a frozen copy of vectordb.py's v0 init_schema (as of 039b3e6):
it is history, not something to keep in sync.
"""
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import sqlite_vec
from fastapi.testclient import TestClient

from manager.vectordb import VectorDB, _serialize_vec

DIM = 768

LEGACY_DDL = """
CREATE TABLE IF NOT EXISTS collections (
    id          TEXT PRIMARY KEY,
    source_dir  TEXT NOT NULL,
    doc_type    TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    collection  TEXT NOT NULL REFERENCES collections(id),
    name        TEXT NOT NULL,
    metadata    TEXT,
    summary     TEXT NOT NULL,
    content     TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    file_hash   TEXT NOT NULL,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE VIRTUAL TABLE IF NOT EXISTS documents_vec USING vec0(
    id TEXT PRIMARY KEY,
    embedding float[768]
);
CREATE INDEX IF NOT EXISTS idx_documents_collection_file_path
    ON documents(collection, file_path);
"""


def unit(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


DOCS = [
    dict(id="README", collection="vault", file_path="/v/README.md", file_hash="h1",
         content="vault readme", vec=unit(0)),
    dict(id="notes/x", collection="vault", file_path="/v/notes/x.md", file_hash="h2",
         content="note x", vec=unit(1)),
    dict(id="PLAN_SKILL", collection="skills", file_path="/s/PLAN_SKILL.md", file_hash="h3",
         content="plan skill", vec=unit(2)),
]


def _raw(path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def make_legacy_db(path, docs) -> None:
    conn = _raw(path)
    conn.executescript(LEGACY_DDL)
    for coll in sorted({d["collection"] for d in docs}):
        conn.execute("INSERT INTO collections (id, source_dir, doc_type) VALUES (?, ?, 'markdown')",
                     (coll, f"/{coll}"))
    for d in docs:
        conn.execute(
            "INSERT INTO documents (id, collection, name, metadata, summary, content, file_path, file_hash)"
            " VALUES (?, ?, ?, '{}', 's', ?, ?, ?)",
            (d["id"], d["collection"], d["id"], d["content"], d["file_path"], d["file_hash"]))
        if d["vec"] is not None:
            conn.execute("INSERT INTO documents_vec (id, embedding) VALUES (?, ?)",
                         (d["id"], _serialize_vec(d["vec"])))
    conn.commit()
    conn.close()


def tables(path) -> set[str]:
    conn = _raw(path)
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        conn.close()


def user_version(path) -> int:
    conn = _raw(path)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def doc_count(path) -> int:
    conn = _raw(path)
    try:
        return conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    finally:
        conn.close()


@pytest.fixture
def legacy(tmp_path) -> Path:
    path = tmp_path / "skills.db"
    make_legacy_db(path, DOCS)
    return path


def test_legacy_database_migrates_every_document_and_vector(legacy):
    db = VectorDB(str(legacy))
    db.init_schema()
    try:
        for d in DOCS:
            assert db.get_document(d["collection"], d["id"])["content"] == d["content"]
            assert db.get_hash(d["collection"], d["file_path"]) == d["file_hash"]
            [hit] = db.search(d["collection"], d["vec"], limit=1)
            assert (hit["id"], hit["score"]) == (d["id"], pytest.approx(1.0))
        assert {c["id"]: c["doc_count"] for c in db.list_collections()} == {"vault": 2, "skills": 1}
    finally:
        db.close()
    assert user_version(legacy) == 1


def test_migrated_database_accepts_a_shared_id(legacy):
    db = VectorDB(str(legacy))
    db.init_schema()
    try:
        db.upsert_document("README", "skills", "R", {}, "s", "skills readme",
                           "/s/README.md", "h4", unit(3))
        assert db.get_document("vault", "README")["content"] == "vault readme"
        assert db.get_document("skills", "README")["content"] == "skills readme"
    finally:
        db.close()


def test_migration_backs_up_the_legacy_database_first(legacy):
    bak = Path(f"{legacy}.pre-v1.bak")
    db = VectorDB(str(legacy))
    db.init_schema()
    db.close()
    assert bak.exists()
    assert user_version(bak) == 0
    assert "documents_vec" in tables(bak)
    assert doc_count(bak) == len(DOCS)


def test_failed_migration_rolls_back_to_the_legacy_database(legacy, monkeypatch):
    from manager.vectordb import MigrationError
    before = tables(legacy)

    def boom(conn):
        raise MigrationError("injected")

    monkeypatch.setattr("manager.vectordb._verify_migration", boom)
    db = VectorDB(str(legacy))
    with pytest.raises(MigrationError):
        db.init_schema()
    db.close()

    assert user_version(legacy) == 0
    assert tables(legacy) == before
    assert doc_count(legacy) == len(DOCS)

    monkeypatch.undo()                     # a later start retries and succeeds
    db = VectorDB(str(legacy))
    db.init_schema()
    db.close()
    assert user_version(legacy) == 1


def test_a_document_without_a_vector_aborts_the_migration(tmp_path):
    from manager.vectordb import MigrationError
    path = tmp_path / "skills.db"
    make_legacy_db(path, DOCS[:2] + [dict(DOCS[2], vec=None)])
    before = tables(path)
    db = VectorDB(str(path))
    with pytest.raises(MigrationError):
        db.init_schema()
    db.close()
    assert user_version(path) == 0
    assert tables(path) == before


def test_migration_runs_once(legacy):
    bak = Path(f"{legacy}.pre-v1.bak")
    db = VectorDB(str(legacy))
    db.init_schema()
    db.close()
    bak.unlink()

    db = VectorDB(str(legacy))
    db.init_schema()
    db.close()

    assert not bak.exists()
    assert user_version(legacy) == 1
    assert doc_count(legacy) == len(DOCS)


def test_a_failed_migration_leaves_inference_up(collection_config, monkeypatch):
    from manager.vectordb import MigrationError
    make_legacy_db(collection_config.skills_db_path, DOCS)

    def boom(conn):
        raise MigrationError("injected")

    monkeypatch.setattr("manager.vectordb._verify_migration", boom)
    with patch("manager.retrieval.EmbeddingsClient") as emb_cls:
        emb_cls.return_value = AsyncMock()
        from manager.app import create_app
        with TestClient(create_app(collection_config)) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/collections").status_code == 503


def test_a_failed_upsert_leaves_no_half_written_document(tmp_path):
    db = VectorDB(str(tmp_path / "skills.db"))
    db.init_schema()
    try:
        db.upsert_collection("c", "/c", "markdown")
        with pytest.raises(sqlite3.Error):
            db.upsert_document("bad", "c", "B", {}, "s", "c", "/c/bad.md", "hbad",
                               [0.0] * 4)              # wrong dimension: the vector insert raises
        db.upsert_document("good", "c", "G", {}, "s", "c", "/c/good.md", "hgood", unit(0))

        assert db.get_hash("c", "/c/bad.md") is None   # retried on the next reindex
        docs = db._conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        vecs = db._conn.execute("SELECT count(*) FROM doc_vectors").fetchone()[0]
        assert (docs, vecs) == (1, 1)
    finally:
        db.close()
```

- [ ] **Step 2: Run them and check that they fail**

Run: `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/test_schema_migration.py -q -p no:cacheprovider`
Expected:
- The first six FAIL. Task 3's code raises `SchemaVersionError` on a legacy DB, and `MigrationError` doesn't exist yet.
- `test_a_failed_upsert_leaves_no_half_written_document` FAILS on `get_hash` returning `'hbad'`.
- `test_a_failed_migration_leaves_inference_up` errors at `monkeypatch.setattr`, because `_verify_migration` doesn't exist yet. That counts as failing.

- [ ] **Step 3: Implement the migration** in `init_schema()`'s legacy branch, per the Interfaces block, in exactly that order.

- [ ] **Step 4: Run the full suite**

Expected: everything passes.

- [ ] **Step 5: Measure on a copy of the real data** (read-only on the live DB)

```bash
S=$(mktemp -d)
/opt/llama/manager/venv/bin/python -c "import sqlite3; s=sqlite3.connect('file:/opt/llama/data/skills.db?mode=ro', uri=True); d=sqlite3.connect('$S/skills.db'); s.backup(d); d.close()"
PYTHONPATH=. /opt/llama/manager/venv/bin/python -c "
import logging; logging.basicConfig(level=logging.INFO)
from manager.vectordb import VectorDB
db = VectorDB('$S/skills.db'); db.init_schema()
print({c['id']: c['doc_count'] for c in db.list_collections()})
print(db._conn.execute('PRAGMA integrity_check').fetchone()[0])"
```
Expected:
- the `skills.db schema migrated v0 -> v1` log line, where the documents count equals the vectors count;
- per-collection doc counts equal to `GET /collections` on the live server;
- `ok`.

Report the output in Deviations, then `rm -r "$S"`.

- [ ] **Step 6: Add a `findings.md` Status line.** Q23 and the search starvation are fixed in phase 2. Tag filtering is still a post-filter over `max(limit*20, 100)` candidates within a collection, and stays open.

- [ ] **Step 7: Commit**

```bash
git add manager/vectordb.py tests/test_schema_migration.py findings.md
git commit -m "feat(vectordb): migrate an existing skills.db to schema v1, backed up first (Q23)"
```

### Task 5: Host cutover for PR B (the user runs these; Claude verifies)

Prerequisite: PR B is merged and the main checkout is fast-forwarded. Claude records `GET /collections` before starting.

- [ ] **Step 1: Manual backup**, in addition to the in-code one. The read-only half was tested on 2026-10-30, as Claude's user against the live DB into the scratchpad: it reported 2917 documents and integrity `ok`. The `sudo -u` wrapper is untested.
  ```
  ! sudo -u _llama-mgr /opt/llama/manager/venv/bin/python -c "import sqlite3; s=sqlite3.connect('file:/opt/llama/data/skills.db?mode=ro', uri=True); d=sqlite3.connect('/opt/llama/data/skills.db.pre-phase2'); s.backup(d); d.close(); print('ok')"
  ```
- [ ] **Step 2: Deploy.** `! sudo /mnt/secondary/inference-server/scripts/deploy-manager.sh`. The restart runs the migration.
- [ ] **Step 3: Claude verifies:**
  - `grep 'schema migrated v0 -> v1' /var/log/llama/manager.err`, where documents equals vectors;
  - `/opt/llama/data/skills.db.pre-v1.bak` exists;
  - `GET /collections` doc counts equal the ones recorded before;
  - `POST /collections/skills/search {"query":"how to write a plan for skills"}` now returns at least one result (it returned `[]` before);
  - one vault search returns results.
- [ ] **Rollback (untested):**
  1. `sudo systemctl stop llama-manager`;
  2. `sudo -u _llama-mgr cp /opt/llama/data/skills.db.pre-phase2 /opt/llama/data/skills.db`;
  3. check out the PR A merge commit and run `deploy-manager.sh`.

  Old code can't read a v1 DB: `upsert_document`'s `ON CONFLICT(id)` has no matching unique constraint. So the database restore and the code rollback go together.
- [ ] Once a week of normal use has gone by, delete `skills.db.pre-phase2` and `skills.db.pre-v1.bak` (about 50 MB each). The migrated DB keeps the dropped tables' pages as free space. That's harmless, and a later `VACUUM` reclaims it; this plan doesn't run one.
