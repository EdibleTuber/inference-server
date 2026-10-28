# Slot Lockup Fixes — Round 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single proxy timeout or failed restart can no longer lock a slot: the queue consumer survives every per-item failure, chat requests never restart or swap a slot, and an unhealthy slot is reprobed instead of answering 503 forever.

**Architecture:** The queue consumer stops calling `ensure_model_on_slot` (the swap path) and calls a new read-only `ServerState.ready_for`, which can probe but never restart or swap. `ensure_model_on_slot` stays the only swap path (`POST /swap`) and turns any swap exception into an ordinary failed swap. The consumer gets a per-item guard so no exception can end it, and the routing 503 path reprobes the slot first.

**Tech Stack:** Python 3.12, FastAPI 0.135.2 / Starlette 1.0.0, httpx 0.28.1, pytest 9.0.2 + pytest-asyncio 1.3.0 (the deployed venv's versions).

**Spec:** `docs/audit/findings.md` (F1–F8 are in `docs/incidents/2026-10-27-main-slot-lockup.md`). This round fixes **F3, F2, F1, O1, F6**, plus A6/W5 because they ship with the next deploy. Everything else in the findings doc is deferred to round 2.

## Implementer Rules (local model via opencode)

This plan is written to be implemented task by task by Qwen 3.8 27B in opencode, with a Claude review of each task's diff before the next task starts. Rules, all mandatory:

1. **Do one task, then stop** and report: the diff summary, the exact test command run, and its pass/fail counts. Do not start the next task until told to.
2. **Never edit a test to make it pass.** `tests/test_consumer.py` was written and validated in advance: it fails against the current code in the ways each task predicts, and passes against a correct fix. The only sanctioned test edits are the two named in Task 3 Step 3 and Task 5 Step 4, made exactly as written.
3. **Stop and report, don't improvise**, if any test fails differently from the plan's "Expected", or passes when the plan says it should fail. That means the plan or the test is wrong, and a human decides.
4. **Never** run `sudo`, `systemctl`, `scripts/deploy-manager.sh`, `git push`, `gh`, or anything that talks to `192.168.1.14`, `127.0.0.1:8081-8084` or `:11434`. You are served by that manager, and Task 6 is not yours.
5. Stage files by explicit path only (never `git add -A` or `git add .`), and run `git status --short` before each commit. Never stage `config/manager.env.orig`.
6. Run tests only with the command in Global Constraints. Don't install packages or create venvs.
7. Change only the files listed in the task. If another file seems to need a change, stop and report.

## Global Constraints

- Base: `origin/main` @ `bac1b52`. Local `main` is stale; branch from `origin/main`. Line numbers below refer to `bac1b52`.
- No dependency changes this round.
- New tests open no real sockets. Stub `SlotState.probe` and swap/restart, and patch `manager.app.httpx.AsyncClient` with an `httpx.MockTransport` where the consumer proxies. The live main llama-server is on `127.0.0.1:8081`, the same port as `test_config`.
- Every regression test is run and seen **failing** before its fix lands, with the failure the step names. A test that passes before the fix is a defect in the test.
- The consumer swallows `Exception`, never `BaseException`: `asyncio.CancelledError` must propagate so shutdown works.
- Run tests with: `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest <target> -q -p no:cacheprovider`
- Concurrency tasks (3–5) give interfaces, invariants and tests, not bodies. Test code in this plan is a sketch to verify against the real fixtures; deviate where it's wrong and say so in the commit.

## Review Focus

1. A **streaming** request (`"stream": true`) that `ready_for` rejects must get a JSON 409/503, not a 200 event-stream. Test added in Task 3.
2. Model names differing only in case or `.gguf` suffix (`qwen3.8-27b-ud-q6_k.gguf`) must be `ready`, never `wrong_model`; otherwise the fix becomes a 409 storm. Test added in Task 3.
3. A probe that raises inside `ready_for` must release `swap_lock` and answer `unavailable`; a held lock would wedge the slot just like the original bug. Test added in Task 3.
4. Cancelling the consumer during an item (manager shutdown) must cancel the task, not be caught as an item error. Test added in Task 4.
5. Items drained by a failed `POST /swap` keep answering 503 `queue_error` (no `error_code` set). Test added in Task 3's mapping test.

---

### Task 0: Branch and land the audit docs

**Files:**
- Add: `docs/incidents/2026-10-27-main-slot-lockup.md`, `docs/audit/findings.md`, `docs/audit/qwen/prompts/*.md`, `docs/audit/qwen/run_inventory.py`, this plan
- Create: `docs/audit/qwen/.gitignore` containing `out/`

- Add: `tests/test_consumer.py` (already written; Tasks 3–5 tests)

- [ ] **Step 1:** Confirm the branch already exists and is checked out: `git branch --show-current` prints `fix/slot-lockup-round1`, and `git log -1 --format=%h` prints `bac1b52`.
- [ ] **Step 2:** Stage the docs above by explicit path, run `git status --short` and confirm only those paths are staged, then commit `docs: incident 2026-10-27, audit findings, inventory tooling, round-1 plan`.
- [ ] **Step 3:** Run `tests/test_consumer.py`. Expected: **13 failed, 3 passed**; the passing three are `test_case_variant_model_name_is_ready`, `test_cancelling_the_consumer_mid_item_cancels_it` and `test_unhealthy_slot_mid_swap_is_not_probed`. Stage only `tests/test_consumer.py` and commit `test: round-1 consumer, readiness and routing tests (failing until Task 5)`.

### Task 1: Restart timeout that matches systemd (F3)

**Files:**
- Modify: `manager/config.py` (add field + env read), `manager/swap.py:77-97`
- Test: `tests/test_swap.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `ManagerConfig.restart_timeout: int = 150` (env `RESTART_TIMEOUT`), declared after `slots` so existing positional and keyword constructions keep working. `ModelSwapper.restart_llama_server()` raises only `RuntimeError`, including on timeout.

Why 150: both slot units have `TimeoutStopUSec=1min 30s` (`systemctl show`, 2026-10-28), so `systemctl restart` can legitimately take ~90s before the start. The incident stop took 30.06s against a 30s limit.

- [ ] **Step 1: Write the failing tests** (reuse the `swapper` fixture and the `fake_executor` pattern from `test_restart_raises_on_failure`; add `import subprocess` to `tests/test_swap.py`'s imports, which lacks it)

```python
@pytest.mark.asyncio
async def test_restart_timeout_is_a_runtime_error(swapper):
    async def fake_executor(executor, fn):
        return fn()
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["systemctl"], 150)), \
         patch.object(asyncio.get_event_loop(), "run_in_executor", side_effect=fake_executor):
        with pytest.raises(RuntimeError, match="timed out"):
            await swapper.restart_llama_server()

@pytest.mark.asyncio
async def test_restart_uses_configured_timeout(swapper):
    ok = MagicMock(returncode=0)
    async def fake_executor(executor, fn):
        return fn()
    with patch("subprocess.run", return_value=ok) as run, \
         patch.object(asyncio.get_event_loop(), "run_in_executor", side_effect=fake_executor):
        await swapper.restart_llama_server()
    assert run.call_args.kwargs["timeout"] == swapper._config.restart_timeout == 150
```

In `tests/test_config.py`: `RESTART_TIMEOUT` unset → `from_env().restart_timeout == 150`; `RESTART_TIMEOUT=200` → `200` (follow that file's existing env-patching style).

- [ ] **Step 2: Run, verify failure.** Expected: the first fails with `subprocess.TimeoutExpired` (not `RuntimeError`); the second with `AttributeError: 'ManagerConfig' object has no attribute 'restart_timeout'`.
- [ ] **Step 3: Implement.** Add the field and `_int_env("RESTART_TIMEOUT", 150)`. In `restart_llama_server`, pass `timeout=self._config.restart_timeout` and convert `subprocess.TimeoutExpired` into `RuntimeError(f"Restart of {unit} timed out after {n}s")`. Add a `RESTART_TIMEOUT` row to the README config table next to `SWAP_TIMEOUT`.
- [ ] **Step 4:** Run `tests/test_swap.py tests/test_config.py`. Expected: all pass.
- [ ] **Step 5:** Commit `fix(swap): restart timeout configurable, default 150s, always RuntimeError`.

### Task 2: A swap exception is a failed swap, not a crash (F2 part 1, A6, W5)

**Files:**
- Modify: `manager/app.py:239-267` (`ensure_model_on_slot`), `README.md:178, 187, 389, 413`
- Test: `tests/test_endpoints.py`

**Interfaces:**
- Consumes: Task 1 (`restart_llama_server` raises only `RuntimeError`), though this task catches `Exception` regardless.
- Produces: `ensure_model_on_slot(slot_name, model_name) -> bool` **never raises `Exception`**. On any exception from `swap_to` it logs with `logger.exception`, marks the slot unhealthy, drains the queue with a message that includes the exception text, and returns `False`, exactly as for a health timeout.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_ensure_model_on_slot_returns_false_when_swap_raises(test_config):
    from manager.app import ServerState
    server = ServerState(test_config)
    slot = server.slots["main"]
    slot.swapper.swap_to = AsyncMock(side_effect=RuntimeError("restart refused"))
    queued = {"body": {}, "event": asyncio.Event(), "response": None, "error": None}
    await slot.queue.enqueue(queued)

    assert await server.ensure_model_on_slot("main", "test-model-q4") is False
    assert slot.healthy is False
    assert not slot.swap_lock.locked()
    assert slot.queue.depth == 0
    assert queued["event"].is_set() and "restart refused" in queued["error"]

def test_background_swap_failure_is_logged_not_raised(client, monkeypatch, caplog):
    async def boom(self, model_path, sampling_args=""):
        raise RuntimeError("restart refused")
    monkeypatch.setattr("manager.swap.ModelSwapper.swap_to", boom)
    r = client.post("/swap", json={"model": "test-model-q8", "target": "main"})
    assert r.status_code == 202
    assert "background swap failed" in caplog.text
```

- [ ] **Step 2: Run, verify failure.** Expected: the first raises `RuntimeError: restart refused` out of `ensure_model_on_slot`; the second makes `TestClient` re-raise `RuntimeError` from the background task.
- [ ] **Step 3: Implement** the `try/except Exception` around `swap_to` inside the `async with slot.swap_lock`, sharing the failure tail (`mark_unhealthy`, `_drain_slot_queue_with_error`, `return False`) with the timeout case. In the README, replace "returns once the slot is healthy" and "503 if the swap fails" (lines 178, 187, 389, 413) with the 202 + poll `GET /status` (`swapping`, `loaded_model`, `healthy`) contract from the `swap_slot` docstring.
- [ ] **Step 4:** Run `tests/test_endpoints.py`. Expected: all pass.
- [ ] **Step 5:** Commit `fix(swap): treat a raising swap as a failed swap; document 202 /swap`.

### Task 3: Chat requests never swap or restart a slot (O1, F1)

**Files:**
- Modify: `manager/app.py` (new `ready_for`, `_not_loaded_message`, `_decision_from_item_error`; `_queue_consumer:347-360`; `_route_and_wait:488-511`; the 409 builder at `:451-463` reuses `_not_loaded_message`), `README.md:409`
- Test: create `tests/test_consumer.py`

**Interfaces:**
- Produces:
  - `Readiness = Literal["ready", "wrong_model", "unavailable"]` (module level in `app.py`).
  - `async def ServerState.ready_for(self, slot_name: str, model_name: str) -> Readiness`
  - `def _not_loaded_message(model_name: str, slots: dict[str, SlotState]) -> str`, the exact text of today's 409 (`app.py:451-462`).
  - `def _decision_from_item_error(item: dict, slot_name: str) -> _RouteDecision`
  - Queue items gain `"error_code": str | None`, one of `None`, `"queue_error"`, `"slot_unavailable"` or `"model_not_loaded"`.

**Invariants:**
- `ready_for` holds `slot.swap_lock` for its whole body and **never calls `swap_to`, `restart_llama_server` or `update_env_file`**.
- `ready_for` returns `"wrong_model"` iff `not same_model(model_name, slot.loaded_model)`, checked under the lock before anything else.
- Same model and healthy → `"ready"`, with no I/O.
- Same model and unhealthy → exactly one `slot.probe(client)` (its own `httpx.AsyncClient`), then `"ready"` if `slot.healthy and same_model(...)`, else `"unavailable"`. If the probe raises (contract says it doesn't), the lock is still released and the result is `"unavailable"`.
- In the consumer: `"wrong_model"` → `error_code="model_not_loaded"`, `error=_not_loaded_message(...)`. `"unavailable"` → `error_code="slot_unavailable"`, `error=f"{slot_name} slot not ready"`. In both cases set the event and continue with no proxy call.
- `_decision_from_item_error` maps `None`/`"queue_error"` → 503 `queue_error`, `"slot_unavailable"` → 503 `slot_unavailable` (with `slot_name`), and `"model_not_loaded"` → 409 `model_not_loaded`. The existing `_openai_error_response` / `_anthropic_error_response` render these unchanged.

Decision to flag: the request path no longer restarts a backend that is truly down. A crashed llama-server is restarted by systemd (`Restart=on-failure`). A hung one stays 503 until someone acts. Automatic restart of a hung backend belongs to the round-2 supervisor (Q2), with its own policy.

- [ ] **Step 1: Tests already exist.** They're the eight tests under `# Task 3` in `tests/test_consumer.py`, with the hermetic fixtures at the top of that file (mock backend, probe stub, socket guard). Read them; do not edit them.
- [ ] **Step 2: Run, verify failure.** Run `tests/test_consumer.py`. Expected for the Task 3 tests:
  - `test_queued_request_for_a_swapped_out_model_gets_409_not_a_swap`: `['test-model-q8', 'test-model-q4'] == ['test-model-q8']` (the O1 reproduction);
  - `..._probed_not_restarted`, `..._503_without_restart` and `..._proxy_timeout_without_restarting`: `['test-model-q4'] == []` (`swap_to` was called);
  - `..._gets_json_409`: a `StreamingResponse` where `None` was expected;
  - `test_probe_raising_inside_ready_for_releases_lock` / `test_item_error_codes_map_to_statuses`: `AttributeError` for `ready_for` / `_decision_from_item_error`;
  - `test_case_variant_model_name_is_ready` passes (it pins behavior).
- [ ] **Step 3: Implement** against the interfaces and invariants above. In `_route_and_wait`, add `"error_code": None` to the item and replace the `item["error"]` branch with `return _decision_from_item_error(item, slot_name)`. In the README 409 paragraph, add one sentence: a request already queued when its model is swapped out also gets 409.

  **Sanctioned test edit:** `tests/test_endpoints.py::test_messages_queue_error_is_anthropic_503` steers the consumer by stubbing `ensure_model_on_slot`, which the consumer no longer calls. Left alone, its request is **proxied to the real llama-server on :8081** (observed while validating this plan). Change exactly two lines in it:
  - `server.ensure_model_on_slot = AsyncMock(return_value=False)` becomes `server.ready_for = AsyncMock(return_value="unavailable")`;
  - `assert "Model swap failed" in err["message"]` becomes `assert "main slot not ready" in err["message"]`.
- [ ] **Step 4:** Run `tests/`. Expected: everything passes except these three later-task tests in `tests/test_consumer.py`: `test_consumer_survives_a_raising_readiness_check` (Task 4), `test_unhealthy_slot_is_reprobed_and_routed` and `test_unhealthy_slot_still_503s_when_the_probe_fails` (Task 5).
- [ ] **Step 5:** Commit `fix(queue): consumer checks readiness instead of swapping; 409 for stale-routed requests`.

### Task 4: The consumer cannot die (F2 part 2)

**Files:**
- Modify: `manager/app.py` (`_queue_consumer`)
- Test: `tests/test_consumer.py`

**Interfaces:**
- Consumes: Task 3 (`ready_for`, `error_code`, the test helpers).
- Produces: the invariant that **every dequeued item has its event set exactly once**, with `response` or `error` populated, unless the consumer is cancelled. `Exception` from any per-item step (`ready_for`, building the request, the proxy, building the response) gives `error_code="queue_error"`, `error=f"Internal error on slot {slot_name}: {exc}"`, a `logger.exception`, `event.set()`, and the loop continues. `asyncio.CancelledError` propagates.

- [ ] **Step 1: Tests already exist.** They're the two tests under `# Task 4` in `tests/test_consumer.py`. Do not edit them.
- [ ] **Step 2: Run, verify failure.** Expected: `test_consumer_survives_a_raising_readiness_check` fails with `AssertionError: items [1, 2] not answered within 2.0s; consumer died with RuntimeError('boom')`. `test_cancelling_the_consumer_mid_item_cancels_it` passes; it pins the `BaseException` boundary.
- [ ] **Step 3: Implement** the per-item `try/except Exception` in `_queue_consumer`, preserving the existing 5xx and proxy-error handling (`mark_unhealthy` + `_reprobe_for`) inside it.
- [ ] **Step 4:** Run `tests/`. Expected: everything passes except the two Task 5 tests named in Task 3 Step 4. The proxy-error lines (`app.py:381-382, 388-392` at `bac1b52`) are now covered by Task 3's proxy-timeout test and this task's tests: `--cov=manager --cov-report=term-missing` must not list them (needs `pytest-cov`, not in the deployed venv; if absent, skip the check and say so).
- [ ] **Step 5:** Commit `fix(queue): a per-item failure can no longer end the slot's consumer`.

### Task 5: Reprobe before answering 503 (F6)

**Files:**
- Modify: `manager/app.py:465-473` (`_validate_and_route`)
- Test: `tests/test_consumer.py` (it tests routing state, not HTTP)

**Interfaces:**
- Consumes: `ServerState._reprobe_slot_if_unlocked(name, slot, client)` (existing, `app.py:129-140`).
- Produces: in `_validate_and_route`, when the resolved slot is unhealthy and holds the model, call `_reprobe_slot_if_unlocked` once (own `httpx.AsyncClient`), then:
  - healthy and `same_model` → route normally (`None, slot_name`);
  - `loaded_model` changed → the 409 from `_not_loaded_message`;
  - still unhealthy → today's 503 `slot_unavailable`.

  A slot mid-swap (lock held) is skipped, so this path never waits behind a swap.

- [ ] **Step 1: Tests already exist.** They're the three tests under `# Task 5` in `tests/test_consumer.py`. Do not edit them.
- [ ] **Step 2: Run, verify failure.** Expected:
  - `test_unhealthy_slot_is_reprobed_and_routed` fails: a 503 `_RouteDecision` where `None` was expected;
  - `test_unhealthy_slot_still_503s_when_the_probe_fails` fails: `[] == ['main']` (no probe happened);
  - `test_unhealthy_slot_mid_swap_is_not_probed` passes.
- [ ] **Step 3: Implement** per the Interfaces block.
- [ ] **Step 4: Sanctioned test edit, then run `tests/`.** `tests/test_endpoints.py::test_chat_completions_503_on_batch_unhealthy` now reprobes the unhealthy batch slot, which connects to the real port `127.0.0.1:8083` (observed while validating this plan). Change its signature to `(client, monkeypatch)` and insert these lines right after its docstring; change nothing else:

  ```python
      from manager.slots import SlotState

      async def _no_probe(self, probe_client):
          return None

      monkeypatch.setattr(SlotState, "probe", _no_probe)
  ```

  Expected: **all tests pass**.
- [ ] **Step 5:** Commit `fix(routing): reprobe an unhealthy slot before answering 503`.

### Task 6: PR, merge, deploy, verify on the host

No code. Each step's result is checked on the host, not inferred.

- [ ] **Step 1:** Run the full suite once more. Push and open a PR with `gh pr create`; the body lists the findings fixed and links the incident doc. The user merges.
- [ ] **Step 2:** From a clean checkout of the merge commit: `./scripts/deploy-manager.sh --check`. Expected: every file STALE versus 7b8ea97 except those unchanged since then, and `0 modified under manager/`. This deploy also ships PR #13's 202 `/swap`. Confirm `agent_core` clients poll `/status` after a 202 rather than assuming the swap is done (a behavior change for them).
- [ ] **Step 3:** With opencode idle: `sudo ./scripts/deploy-manager.sh`. Startup indexing takes about 36s (S3), longer than the script's 15s check (S4), so a "did not report healthy within 15s" warning is expected. Then wait for `curl -s http://192.168.1.14:11434/health` to return 200.
- [ ] **Step 4: Verify provenance and host fixes.**
  - `head -1 /opt/llama/manager/DEPLOYED_FROM` shows the merge sha.
  - `sha256sum /opt/llama/manager/app.py` matches `git show <merge>:manager/app.py | sha256sum`.
  - `/var/log/llama/manager.log` now exists and is growing (S8 group fix took effect on restart).
  - `GET /status` shows both slots `healthy: true`.
- [ ] **Step 5: Smoke test both slots** with one short non-streaming and one streaming chat request each through `:11434`. Expected: 200, and a `slot=<name>` log line in `manager.log`. Confirm the first `re` swap ever (memory: `re` never manager-swapped) only when the user wants one; it is no longer expected to crash, but it is still untested on the host.
