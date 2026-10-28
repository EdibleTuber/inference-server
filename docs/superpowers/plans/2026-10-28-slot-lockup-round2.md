# Slot Lockup Fixes — Round 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- One malformed document no longer breaks collection indexing.
- A manager restart no longer waits for indexing.
- A streaming client sees the backend's real error status.
- A slow-but-healthy backend is no longer timed out at 120s and treated as sick.

**Architecture:** four independent fixes.
- **Indexing (S1):** `index_collection` isolates each document and counts failures in a new `errors` stat.
- **Timeouts (F4):** the proxy uses a configurable read timeout, and a read/write timeout becomes a 504 `upstream_timeout` that leaves the slot healthy.
- **Streaming (A2):** the consumer opens a streaming backend response itself, so it can pass a 4xx/5xx through as a plain error, and hands a 2xx to a relay that always closes the backend stream.
- **Startup indexing (S3):** startup indexing goes through `ReindexRegistry` as background jobs, which the registry can now wait for and cancel.

**Tech Stack:** Python 3.12, FastAPI 0.135.2 / Starlette 1.0.0, httpx 0.28.1, pytest 9.0.2 + pytest-asyncio 1.3.0. These are the deployed venv's versions; uvicorn 0.42 reports ASGI spec 2.3.

**Spec:** `docs/audit/findings.md`: S1, S2, S3, A2, A4 (the registry part), F4. Deferred to the design review: A1, F5, O4, Q1, Q2.

## Implementer Rules (local model via opencode)

Same workflow as round 1: Qwen 3.8 27B in opencode implements one task per prompt, with a Claude review of each task's diff before the next.

1. **Do one task, then stop** and report: the diff summary, the exact test command run, and its pass/fail counts. Do not start the next task until told to.
2. **Never edit a test to make it pass.** The round 2 tests were written and validated in advance. They fail against `a183d14` in the ways each task predicts, and pass against a reference fix. The only sanctioned test edits are the three named in Tasks 2, 3 and 4, made exactly as written.
3. **Stop and report, don't improvise**, if any test fails differently from the plan's "Expected", or passes when the plan says it should fail.
4. **Never** run `sudo`, `systemctl`, `scripts/deploy-manager.sh`, `git push`, `gh`, or anything that talks to `192.168.1.14`, `127.0.0.1:8081-8084` or `:11434`.
5. Stage files by explicit path only (never `git add -A` or `git add .`), and run `git status --short` before each commit. Never stage `config/manager.env.orig`.
6. Run tests only with the command in Global Constraints. Don't install packages or create venvs.
7. Change only the files listed in the task. If another file seems to need a change, stop and report.

## Global Constraints

- Branch: `fix/slot-lockup-round2`, from `origin/main` @ `a183d14` (the round-1 merge, which is what's deployed). Line numbers below refer to `a183d14`.
- No dependency changes.
- Run tests with: `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest <target> -q -p no:cacheprovider`
- The new tests are hermetic (mock transport, stubbed probes, socket guard). Do not weaken that.
- Concurrency tasks (3, 4) give interfaces and invariants, not bodies. Where this plan does show code, it's a sketch; deviate if it's wrong and say so in the commit.

## Review Focus

1. A **connect timeout** must stay a health signal: an unhealthy slot with a 503 and a reprobe, never `upstream_timeout`. Catching `httpx.TimeoutException` would get this wrong, because it includes `ConnectTimeout`. Pinned by `test_connect_timeout_is_not_treated_as_a_busy_backend` (Task 2).
2. A streaming client that **disconnects before the first chunk** must still close the backend stream. The relay generator never starts in that case, so only `StreamingResponse(background=...)` can close it. Pinned in Task 3.
3. A **document that fails to re-index** must not be stale-deleted, and must be retried next run; don't record its hash. Pinned in Task 1.
4. **Shutdown** must cancel running index jobs *before* closing the DB and the embeddings client. Pinned in Task 4.
5. A `POST /collections/{id}/reindex` **during startup indexing** joins the startup job instead of starting a parallel one. Pinned in Task 4.

---

### Task 0: Land the round-2 tests and docs

**Files:** `tests/test_consumer.py` (round-2 sections appended), `tests/test_indexing_resilience.py` (new), `tests/test_background_indexing.py` (new), `docs/incidents/2026-10-27-main-slot-lockup.md` (status), `docs/audit/findings.md` (status), this plan.

- [ ] **Step 1:** `git branch --show-current` prints `fix/slot-lockup-round2`, and `git log -1 --format=%h` prints `a183d14`.
- [ ] **Step 2:** Run `tests/`. Expected: **24 failed, 256 passed**: 7 for Task 1, 5 each for Tasks 2 and 3, 7 for Task 4. The failures are the tests named under Tasks 1–4 below, minus the ones each task marks "passes" (they pin behavior).
- [ ] **Step 3:** Stage the six files above by path and commit `test: round-2 tests (failing until Task 4); docs: round-1 status, round-2 plan`.

### Task 1: Indexing survives a bad document (S1)

**Files:** Modify `manager/collections.py` (`derive_tags:82`, `index_collection:156`).

**Interfaces:**
- `derive_tags` accepts `frontmatter["tags"]` as a list, a string or anything else:
  - a string `s` is treated as `[s]`;
  - in a list, only `str` items are used (lowercased);
  - `None` or any other type adds nothing.
  - It never raises.
- `index_collection(...) -> dict` returns stats with the keys `new, updated, removed, unchanged, errors`.
  - Each document past the unchanged-hash check is processed under its own `try/except Exception`.
  - On failure: `logger.exception` naming the collection and file path, `stats["errors"] += 1`, the document isn't upserted (so its hash isn't stored and the next run retries it), and the loop continues.
  - The path stays in `seen_paths`, so a failed update never stale-deletes the previous version.

- [ ] **Step 1:** Tests exist: all 7 in `tests/test_indexing_resilience.py`. Run them. Expected:
  - `test_derive_tags_tolerates_empty_tags` and `test_empty_tags_frontmatter_is_indexed`: `TypeError: 'NoneType' object is not iterable`;
  - `test_derive_tags_accepts_a_single_string`: `'solo' in {'area', 'l', 'o', 's', 'topic'}`;
  - `test_derive_tags_skips_non_string_items`: `AttributeError` on `.lower()`;
  - the other three: `RuntimeError: embedding failed for ...` escaping `index_collection`.
- [ ] **Step 2:** Implement per the Interfaces block.
- [ ] **Step 3:** Run `tests/test_indexing_resilience.py tests/test_collections.py`. Expected: all pass.
- [ ] **Step 4:** Commit `fix(collections): one bad document no longer aborts indexing`.

### Task 2: A slow backend is a 504, not a sick slot (F4)

**Files:** Modify `manager/config.py`, `manager/app.py` (non-streaming proxy at `:445-454`, `_decision_from_item_error:513`, `_openai_error_response:635`, `_anthropic_error_response:650`), `README.md` (config table, next to `RESTART_TIMEOUT` at `:371`), `tests/test_consumer.py` (sanctioned edit only).

**Interfaces:**
- `ManagerConfig.proxy_read_timeout: int = 1800`, from env `PROXY_READ_TIMEOUT`, declared after `restart_timeout`.
  - Why 1800: requests run up to `--n-predict 32768`, about 22 minutes at the observed ~25 tok/s, plus up to ~130s of prefill for a 64K prompt.
- The non-streaming proxy keeps the `client.post(backend_url, json=body, timeout=...)` call shape, because existing tests mock exactly that. It passes `httpx.Timeout(config.proxy_read_timeout, connect=10.0)`.
- `except (httpx.ReadTimeout, httpx.WriteTimeout)` sets:
  - `item["error_code"] = "upstream_timeout"`;
  - `item["error"] = f"{slot_name} did not respond within {config.proxy_read_timeout}s"`;
  - and does **not** mark the slot unhealthy or schedule a reprobe.

  Every other exception (including `ConnectTimeout` and `ConnectError`) keeps today's proxy-error path.
- `_decision_from_item_error` maps `"upstream_timeout"` to 504 `upstream_timeout`.
- The OpenAI shaper renders it as `{"type": "upstream_timeout", "message": ...}`, and the Anthropic shaper uses inner type `"api_error"`. Neither sets `Retry-After`, because it is only ever sent with a 503.
- README: add a `PROXY_READ_TIMEOUT` row. Say that it bounds how long a single request may run, not queue wait.

- [ ] **Step 1:** Tests exist: the 7 under `# Round 2, Task 2` in `tests/test_consumer.py`. Expected:
  - `test_proxy_read_timeout_config`: `AttributeError ... proxy_read_timeout`;
  - `..._uses_the_configured_read_timeout`: `assert 120 == 1800`;
  - `..._504_and_leaves_the_slot_healthy`: `assert None == 'upstream_timeout'`;
  - `..._maps_to_504`: `(503, 'queue_error') == (504, 'upstream_timeout')`;
  - `..._error_shapes`: `KeyError: 'upstream_timeout'` in the Anthropic shaper;
  - `test_connect_error_still_marks_the_slot_unhealthy` and `test_connect_timeout_is_not_treated_as_a_busy_backend`: **pass** (they pin behavior).
- [ ] **Step 2:** Implement per the Interfaces block.
- [ ] **Step 3: Sanctioned test edit.** The round-1 test `test_consumer_recovers_from_a_proxy_timeout_without_restarting` asserts a `ReadTimeout` yields "Proxy error", which this task deliberately changes. Replace exactly one line, `tests/test_consumer.py:282`: `    assert "Proxy error" in first["error"]` becomes `    assert first["error_code"] == "upstream_timeout"`. Its other assertions (no restart, next request served) are the property it exists for; leave them.
- [ ] **Step 4:** Run `tests/`. Expected: all pass except the Task 3 and Task 4 tests that are still expected to fail.
- [ ] **Step 5:** Commit `fix(proxy): configurable read timeout; a timeout is a 504, not an unhealthy slot`.

### Task 3: Streaming passes the backend's status through (A2)

**Files:** Modify `manager/app.py` (the streaming branch at `:430-443`; a new module-level `_relay_stream`), `tests/test_endpoints.py` (sanctioned edit only).

Observed 2026-10-28 on the live host: llama-server answered a bad streaming request with `400 application/json`, and the manager relayed it as `200 text/event-stream`.

**Interfaces:**
- The streaming branch opens the backend response **in the consumer**, using `client = httpx.AsyncClient()` (not a context manager) and `ctx = client.stream("POST", backend_url, json=body, timeout=<Task 2's Timeout>)`, entered with `resp = await ctx.__aenter__()`. This keeps the `client.stream(...)` call shape the existing streaming test mocks.
- A single `close()` coroutine closes both: `await ctx.__aexit__(None, None, None)`, then `await client.aclose()`. It is idempotent.
- `resp.status_code >= 400`: `content = await resp.aread()`, then `close()`. `item["response"] = Response(content, status_code=resp.status_code, media_type=<backend content-type>)`. A 5xx also marks the slot unhealthy and schedules a reprobe, same as non-streaming.
- `resp.status_code < 400`: `item["response"] = _relay_stream(resp, close)`.
  - `def _relay_stream(resp, close) -> StreamingResponse` returns a `StreamingResponse` whose generator yields `resp.aiter_bytes()` and calls `close()` in a `finally`.
  - It uses `status_code=resp.status_code`, `media_type=<backend content-type, default "text/event-stream">`, and `background=BackgroundTask(close)` (`from starlette.background import BackgroundTask`).
- If entering `ctx` raises, close the client and re-raise into the existing handlers: `ReadTimeout`/`WriteTimeout` become 504 `upstream_timeout` (Task 2), anything else the proxy-error 503.
- **Invariant:** every backend stream opened is closed, whichever path is taken: error status, normal end, client gone before the first chunk (the generator never starts, so `background` is the only closer), client gone mid-stream, or the backend failing mid-stream.

- [ ] **Step 1:** Tests exist: the 7 under `# Round 2, Task 3`. Expected:
  - `..._400_is_passed_through_as_json`: `assert not True` (got a `StreamingResponse`);
  - `..._before_the_first_chunk`: `backend stream should be opened by the consumer`;
  - `..._connect_error_is_a_503...` and `..._read_timeout_before_headers_is_a_504`: a `StreamingResponse` where `None` was expected;
  - `test_stream_uses_the_configured_read_timeout`: `'read': None`;
  - `..._success_relays_status_headers_and_chunks` and `..._mid_stream`: **pass** (they pin behavior).
- [ ] **Step 2:** Implement per the Interfaces block.
- [ ] **Step 3: Sanctioned test edit.** `tests/test_endpoints.py::test_messages_streaming_passthrough` mocks the backend response with no status code, and reading the status is this task's fix. After `tests/test_endpoints.py:1206` (`        mock_stream_resp.aiter_bytes = fake_aiter_bytes`), insert exactly:
  ```python
          mock_stream_resp.status_code = 200
          mock_stream_resp.headers = {"content-type": "text/event-stream"}
  ```
- [ ] **Step 4:** Run `tests/`. Expected: all pass except the 7 Task 4 tests.
- [ ] **Step 5:** Commit `fix(stream): pass the backend status through; always close the backend stream`.

### Task 4: Startup indexing runs in the background (S3, S2, A4)

**Files:** Modify `manager/reindex_jobs.py`, `manager/app.py` (`lifespan:706-760`, `trigger_reindex:1018`, import at `:44`), `manager/collections.py` (delete `index_all_collections:263`, now unused), `README.md`, `tests/conftest.py` (sanctioned edit only).

**Interfaces:**
- `ReindexRegistry` keeps a reference to every job task in `self._tasks: set[asyncio.Task]`, removed on completion with `add_done_callback`.
  - `async def wait_idle(self) -> None` returns once no job task is running, including jobs started while it waits.
  - `async def shutdown(self) -> None` cancels every running job and waits for them all.
- `_run_job` changes:
  - on `asyncio.CancelledError` it sets `status="error"` and `error="cancelled"`, then re-raises;
  - it merges `"errors"` into `job.stats` when the indexer returns it, and leaves the default stats dict unchanged;
  - on success it logs one `INFO` line with the job id, the collection and its stats.
- `def _collection_indexer(server, entry: dict)` (module level) returns the `async (paths) -> stats` closure over `index_collection`. Both startup and `trigger_reindex` use it, replacing the inline `_indexer`.
- `lifespan` startup, when `skills_db_path` is set:
  - **(a)** create the DB and the embeddings client. On failure: `server.db = None`, and log "collection endpoints will return 503".
  - **(b)** for each entry of `load_collections_config(...)`, call `server.db.upsert_collection(id, source_dir, doc_type)` first (so endpoints recognize the collection immediately), then `await server.reindex_registry.start(collection_id=id, indexer=_collection_indexer(server, entry), paths=None)`. On failure: log "serving the existing index". Neither message claims endpoints return 503 when `db` is set (S2).
  - Startup does not await the jobs.
- `lifespan` teardown: `await server.reindex_registry.shutdown()` **before** closing the embeddings client and the DB.
- README: say startup indexing runs in the background, and that `GET /collections/{id}/reindex/status` shows the startup job.

- [ ] **Step 1:** Tests exist: all 7 in `tests/test_background_indexing.py`. Expected:
  - the three app tests: `{"error":"No reindex job recorded for this collection"}`;
  - the four registry tests: `AttributeError` for `wait_idle` / `shutdown`.
- [ ] **Step 2:** Implement per the Interfaces block.
- [ ] **Step 3: Sanctioned test edit.** `tests/conftest.py:189-191` (`collection_client`): once startup indexing is background, collection tests would race it. Insert one line between `with TestClient(collection_app) as client:` and `yield client`:
  ```python
          client.portal.call(client.app.state.server.reindex_registry.wait_idle)
  ```
- [ ] **Step 4:** Run `tests/`. Expected: **all pass**.
- [ ] **Step 5:** Commit `fix(startup): index collections in the background; cancel jobs on shutdown`.

### Task 5: PR, merge, deploy, verify (Claude and the user; not the implementer)

- [ ] **Step 1:** Push, open the PR, user merges.
- [ ] **Step 2:** Deploy from a clean checkout of the merge, as in round 1: `--check`, then `sudo ./scripts/deploy-manager.sh` with opencode idle. Expected: the 15s health check now passes, because startup no longer waits for indexing (S4 goes away as a side effect).
- [ ] **Step 3:** Verify on the host:
  - `DEPLOYED_FROM` and every file's sha256 match the merge commit;
  - `manager.log` shows each collection's reindex job finishing. Expect `errors: 0` for vault, since `templates/note-template.md` now indexes; any non-zero count names a newly found bad file in the log;
  - `learnings` and `relationships` are indexed for the first time;
  - both slots pass a normal and a streaming smoke request;
  - a streaming request with `"temperature": "hot"` now returns **400**, not 200.
