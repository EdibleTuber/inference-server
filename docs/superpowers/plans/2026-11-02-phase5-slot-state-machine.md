# Phase 5: Slot State Machine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each slot has one explicit state, changed only by its own queue consumer. A swap is a FIFO item in the slot's queue. A slot stays busy until its response is complete, streams included. A failed swap falls back once to the last model that worked. A per-slot supervisor probes and (when enabled) restarts a dead backend. A request whose client has gone is skipped.

**Architecture:**
- **The consumer is the only writer.** `SlotState` stores one state, changed only through a transition table. `healthy` and `swapping` in `/status` are derived from it, so they can never disagree with `state`.
- **Everything that changes a slot goes through its queue or its inbox.** `/swap` enqueues a swap item. Probes that other code wants (`/status`, routing, a 5xx, the supervisor) are requests the consumer serves between items.
- **The supervisor owns no state.** It is a ticker that asks the consumer to probe; the consumer counts failures and decides on a restart.

**Tech Stack:** Python 3.12.3, FastAPI 0.135.2, httpx, asyncio. llama-server build 11136 with `--parallel 1`.

**Spec:** `docs/design-review-2026-10-29.md`, the "Slots" section (Q10, Q12, Q19, Q20, Q21, Q26), the `/status` line of "Observability (Q27)", and the phase 5 row of "Phases and process".

## Global Constraints

- **Test command:** `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/ -q -p no:cacheprovider`.
- **Live ports:** tests never touch live ports 8080–8084 or 11434.
- **Modules:** new modules go flat in `manager/`. `deploy-manager.sh` ships only `manager/*.py`.
- **Client API:** responses may only gain fields. Existing `/status` keys (`loaded_model`, `healthy`, `swapping`, `last_swap_utc`, `queue_depth`, `queue_limit`, `models`, `state`, `in_flight`, `in_flight_age_seconds`, `oldest_queue_wait_seconds`, `last_error`) keep their names. `state` gains the values `probing` and `failed`. agent_core and PAL read `/status`.
- **Phase 4's lines stay exact:** one `manager.requests` line per inference request and per `/swap`, written once.
- **Every queued item is answered exactly once**, its event set exactly once, whatever happens to the consumer short of cancellation.
- **Never edit a test to make it pass.** Claude owns the tests.

## How this phase runs (heaviest)

What changed from phase 4 (light to medium):
- **Code in the plan:** none. Each task gives interfaces with exact signatures, the invariants, the failure modes and the test names. The implementer writes the code against the real files.
- **Tests:** Claude writes each task's tests first, checks each one fails against the code before the task, checks they pass against a scratch reference, and commits them before the task starts. Existing tests that encode behaviour a task changes on purpose are rewritten or removed in the same commit, and the commit message names each one.
- **Review:** adversarial, against concrete interleavings walked through the actual source, at 4 findings or fewer per round. Claude takes over any task that fails review twice.
- **Mutation runs:** every task, run by the real `mutator` subagent against a throwaway detached worktree, never the phase worktree.
- **Deploys:** all six tasks ship in one PR, deployed outside Qwen's sessions (Qwen runs on the `main` slot being modified). Intermediate commits are never deployed, so a task may leave behaviour that a later task completes.
- **Worktree:** branch `phase5/slot-state-machine` in `/mnt/secondary/inference-server-wt-phase5`.

## Implementer Rules (Qwen)

1. Do exactly one task, then stop and report "Done" plus a **Deviations** list.
2. Never edit a test file. If one is wrong, stop and report it.
3. Stop and report if a step's Expected result doesn't happen.
4. Never run sudo, systemctl, deploy scripts or `git push`, and never touch ports 8080–8084 or 11434.
5. If anything in this plan turns out to be wrong, deviating is correct; say so in Deviations.
6. Commits stage explicit paths and end with the repo's attribution lines.

## Review Focus

Each item names the task whose tests pin it:

1. **A `/swap` arriving while requests are queued** must run after them and before anything queued after it, and a request queued behind a swap to another model gets the 409 a fresh request would. → Task 1, `test_a_swap_runs_after_the_items_ahead_of_it_and_before_the_items_behind_it`.
2. **A stream whose response is never sent** (the endpoint fails between the handoff and the send) must not wedge the slot in `serving`. → Task 3, `test_a_stream_nobody_sends_is_closed_by_the_consumer`.
3. **A backend that is loading** (503 for about 60 s after any restart; measured 14:12:04 → 14:13:03 UTC on 2026-11-02) must not count as dead, or the supervisor restarts it in a loop. → Task 5, `test_a_loading_backend_is_not_counted_as_a_failure_until_the_grace_ends`.
4. **`/status` polled while a long request runs** must answer at once and not probe the busy slot. → Task 2, `test_status_answers_at_once_and_does_not_probe_a_busy_slot`.
5. **A failed trial during onboarding** falls back to the old model, so the slot ends healthy on the wrong model; `onboard` must report the trial failed, not wait until it times out. → Task 4, `test_onboard_reports_a_failed_trial_that_fell_back`.

## Decisions made while planning (beyond the spec)

- **States and transitions** (the final table, after Task 5; earlier tasks use a subset, named in each task):

  | From | To |
  |---|---|
  | `idle` | `serving`, `swapping`, `unhealthy` |
  | `serving` | `idle`, `unhealthy` |
  | `swapping` | `idle`, `failed` (Task 4; until then `unhealthy`) |
  | `unhealthy` | `probing`, `swapping` |
  | `probing` | `idle`, `unhealthy` |
  | `failed` | `idle` (a healthy probe), `swapping` (a `/swap`) |

  A slot starts `unhealthy` and the consumer's first act is a probe. Setting the current state again is a no-op. Any other move raises `IllegalTransition` and is a bug. Every real transition logs one INFO line on `manager.slots`: `slot=<name> <from> -> <to>`.
- **`healthy` = state is `idle` or `serving`; `swapping` = state is `swapping`.** Both stay in `/status` for agent_core and `onboard`. During a swap `healthy` is now false (it used to keep its pre-swap value).
- **Probing a `failed` slot (your decision, 2026-11-02):** the supervisor keeps probing it but never restarts it. A healthy probe moves it to `idle`, so a backend fixed by hand is noticed. A `/swap` is also accepted while `failed`.
- **Supervisor restarts ship off (your decision, 2026-11-02):** `SUPERVISOR_RESTART=0` by default. The supervisor still probes, changes states, counts, and logs `would restart` at WARNING, but never runs `systemctl`. Turning it on is one line in `/etc/llama/manager.env`.
- **Supervisor defaults** (each an env var on `ManagerConfig`): probe every 15 s (`SUPERVISOR_PROBE_INTERVAL`) while the slot is `idle`, `unhealthy` or `failed`; restart after 3 consecutive failed probes (`SUPERVISOR_FAILURES`); backoff between restarts 30 s doubling to 600 s (`SUPERVISOR_BACKOFF`, `SUPERVISOR_BACKOFF_MAX`). A backend answering 503 is *loading*, not failed, for up to `SWAP_TIMEOUT` after the slot last left `idle`/`serving` or was restarted; after that a 503 counts as a failure.
- **No probe while `serving` or `swapping`.** Measured 2026-11-02 on egpu: `/health` and `/v1/models` answer in under 0.4 ms throughout a 13.7 s, 600-token generation, so probing during generation would be safe. It is still not done: a request in flight already finds out if the backend dies, and the idle-only rule from Q21 stays. Main runs the same binary but was not measured.
- **`/swap` answers 200 `already_loaded` only when nothing is ahead of it:** the slot is `idle`, its queue is empty, the model is loaded, and `trial` is false. The endpoint and the consumer share one event loop and nothing awaits between that check and the response, so the answer can't go stale. Otherwise it enqueues and answers 202. A full queue refuses it with 503 `queue_full`.
- **A swap item's line is written by the consumer when the swap ends**, with status 202 and outcome `ok`, `already_loaded` (the model was loaded by the time the item ran), `failed`, `fell_back` (Task 4) or `drained`. `duration_ms` runs from the 202 to the end, as today.
- **`/status` per slot gains `last_swap`:** `null`, or `{"id", "model", "trial", "outcome", "message", "at_utc"}` for the last swap item the consumer finished. `id` is the swap's `X-Request-Id`. `onboard` sends its own id and polls for it (Task 4).
- **`/status` gains `supervisor` per slot** (Task 5): `{"restart_enabled", "restarts", "restarts_suppressed", "consecutive_failures"}`. `restarts` counts real restarts only; `restarts_suppressed` counts the ones observe-only mode skipped.
- **`/status` and routing no longer probe directly.** They ask the consumer (`SlotState.refresh`), wait at most 5 s, and report what the slot knows. A slot that is `serving` or `swapping` is not probed, and the call returns at once.
- **Q26 uses Starlette's `Request.is_disconnected()`**, called by the consumer when it dequeues an item. A gone client's item is answered (event set) without a backend call, and its line has status 499 and outcome `client_gone`. Whether `is_disconnected()` is reliable for a body already read is **unverified**; Task 6 measures it with a real uvicorn server before relying on it.

---

## Branch `phase5/slot-state-machine` (worktree `/mnt/secondary/inference-server-wt-phase5`)

### Task 1: `/swap` is a FIFO item in the slot's queue (Q12)

Transition: the swap moves from a background task into the consumer: `idle → swapping → idle`, run by the consumer in queue order.

**Files:**
- Modify: `manager/app.py`: the `/swap` endpoint, `_queue_consumer`, a new `_run_swap_item`.
- Modify: `manager/slots.py`: `last_swap`, in `to_status_dict`.
- Modify: `README.md`: the `/swap` section (202 means queued; 200 only when nothing is ahead) and the `/status` table (`last_swap`).
- Test: `tests/test_swap_queue.py` (new, Claude), plus the existing tests Claude reworks.

**Interfaces:**
- Consumes: `ServerState.ensure_model_on_slot(slot_name, model_name, *, force=False) -> bool` (unchanged signature; from now on called only by the consumer); `log_swap(...)` from `manager.request_log`; `request_context(request)`.
- Produces:
  - A swap item: a dict with `"kind": "swap"`, `"model": str`, `"trial": bool`, `"id": str`, `"client": str | None`, `"user_agent": str | None`, `"accepted_at": float` (`time.monotonic()` when the 202 was built), `"event": asyncio.Event`. A request item gains `"kind": "request"`; an item with no `"kind"` is a request (tests and old code build them that way).
  - `async def _run_swap_item(server: ServerState, slot_name: str, item: dict) -> None` in `manager/app.py`. Runs the swap, sets `slot.last_swap`, writes the swap line, sets the item's event. Never raises except `CancelledError`.
  - `SlotState.last_swap: dict | None`, keys `id`, `model`, `trial`, `outcome`, `message`, `at_utc`; reported by `to_status_dict()["last_swap"]`.

**Invariants:**
1. A swap item runs after every item enqueued before it and before every item enqueued after it.
2. While the consumer runs the swap, `/status` shows `swapping: true` for that slot. Until then the slot keeps serving.
3. Exactly one swap line per `/swap`: refusals at once (unchanged), the 200 at once, a 202 when its item ends.
4. A swap item drained by a failed swap ahead of it (`_drain_slot_queue_with_error`) still gets its line (outcome `drained`) and its event is set. It is finished **after** the swap whose failure drained it, so `last_swap` ends on the latest swap. (`_drain_slot_queue_with_error` runs inside `ensure_model_on_slot`, before that swap's own line, so the drained swap items have to wait somewhere until it is written.) Task 4 replaces this failure tail.
5. No `BackgroundTasks` remains in `manager/app.py`, and the consumer awaits the swap itself: it is not handed to another task.
6. `duration_ms` runs from the 202 (the item's `accepted_at`), so time spent waiting in the queue counts.
7. `last_swap` and the swap line are set in the same synchronous step, with no `await` between them.

**Existing tests reworked in Claude's commit** (each relied on the swap finishing inside the request, through `BackgroundTasks`): in `tests/test_endpoints.py`, `test_swap_valid_main`, `test_swap_valid_egpu`, `test_swap_failure_visible_via_slot_state` (also made non-vacuous: its slot used to start unhealthy, so it passed with no swap at all), `test_background_swap_failure_is_logged_not_raised` (renamed `test_a_raising_swap_is_reported_not_raised`) and `test_swap_echo_is_canonical_after_completion`; in `tests/test_profile_roster.py`, the `roster` fixture (now runs the lifespan), `test_an_untargeted_swap_goes_to_the_profiles_home`, `test_a_trial_swap_loads_an_unverified_pair` and `test_a_trial_swap_restarts_even_when_the_model_is_loaded`; `tests/test_request_lines.py::test_swap_lines`; and `tests/test_slot_status.py::test_status_dict_reports_ages_rounded_and_null_when_empty` (an exact key set became a superset). `tests/conftest.py` gains `wait_for_swap(server, slot, rid)`.

- [ ] **Step 1: Claude's tests are committed** (`tests/test_swap_queue.py` and the reworked tests). Run: `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/test_swap_queue.py -q -p no:cacheprovider`. Expected: FAIL (the swap still runs as a background task).
- [ ] **Step 2: Implement** the swap item, `_run_swap_item`, the consumer dispatch on `kind`, `last_swap`, and the endpoint's 200-only-when-nothing-is-ahead rule.
- [ ] **Step 3: Update `README.md`** (the `/swap` section and the `/status` table).
- [ ] **Step 4: Run the full suite.** Expected: all pass.
- [ ] **Step 5: Commit** `manager/app.py manager/slots.py README.md`.

### Task 2: One state, one writer (Q19)

Transitions: `unhealthy → probing → idle | unhealthy`, `idle | serving → unhealthy`, `idle ↔ serving`, `idle | unhealthy → swapping → idle | unhealthy`. Every write of a slot's state or `loaded_model` happens on that slot's consumer task.

**Files:** `manager/slots.py`, `manager/app.py`, `README.md` (the `/status` `state` row); test `tests/test_state_machine.py` (new, Claude) plus reworked tests.

**Interfaces (produced):**
- `class IllegalTransition(ValueError)` and `TRANSITIONS: dict[str, frozenset[str]]` in `manager/slots.py`.
- `SlotState.state -> str` (read-only property over a stored field); `SlotState._set_state(new: str) -> None` (no-op for the current state, `IllegalTransition` for a move not in `TRANSITIONS`, one INFO line per real move).
- `SlotState.healthy -> bool` (property: state is `idle` or `serving`). It keeps a setter, for tests only: `True` stores `idle`, `False` stores `unhealthy`, bypassing the table. No code in `manager/` assigns `healthy` (a test reads the source to pin it).
- `async def SlotState.refresh(self, timeout: float) -> bool`: ask the consumer to probe; return True once a probe it asked for has finished, False on timeout. Returns False at once, without asking, when the state is `serving` or `swapping`. Callers waiting at the same time share one probe.
- `ServerState.reprobe_all_slots()` and `reprobe_all_slots_concurrently()` keep their names and contracts but call `refresh` (bound: `REFRESH_TIMEOUT = 5.0` in `manager/app.py`).
- `_reprobe_for` is deleted: after a 5xx or a proxy error the consumer moves the slot to `unhealthy` and probes it itself right after the item.
- The startup probe moves into the consumer: the lifespan starts the consumers, then awaits `refresh` on each slot before logging `ready` / `not reachable`.

**Invariants:** every call of `SlotState.probe`, `_set_state`, `mark_swapped` and `mark_unhealthy` runs on the task named `queue_consumer_<slot>` (pinned at runtime by recording `asyncio.current_task().get_name()` across startup, `/status`, routing, a 5xx and a swap); `/status` never waits on a `serving` or `swapping` slot.

### Task 3: A slot is busy until its response completes (Q10)

Transition: `serving → idle` happens only when the response is complete: a stream's closer has run, or a non-streamed response has been built.

**Interfaces:** the consumer awaits the relay's closer before taking the next item. If the relay's generator has not started and its background task has not run within `STREAM_START_TIMEOUT = 30.0` s of the handoff, the consumer runs the closer itself and the line's outcome is `client_disconnected`.

**Invariants:** `in_flight` is at most 1; a client that leaves mid-stream frees the slot as soon as the closer runs; the item's event is set at the handoff, not at the close (the client must get its response to start reading). Claude rewrites `test_two_requests_in_flight_when_a_stream_is_relayed_and_the_next_starts`, which pins today's two-in-flight behaviour on purpose.

### Task 4: A failed swap falls back once; then `failed` (Q20)

Transitions: `swapping → idle` (fell back), `swapping → failed`, `failed → swapping`, `failed → idle`.

**Interfaces:** `SlotState.known_good: str | None`, the model the slot last served healthily (set by a healthy probe and by a successful swap). A failed swap (raised or timed out) swaps once to `known_good` (not a trial, so the already-loaded shortcut can't apply: the backend was just restarted). Success: `idle`, swap outcome `fell_back`, message naming the failure. Failure, or no `known_good`: `failed`. While `failed`, every request item and every queued request is answered at once with 503 `slot_failed`; a swap item still runs. `scripts/onboard.py` sends its own `X-Request-Id` and decides from `last_swap` with that id: landed when its outcome is `ok`; failed for any other outcome.

### Task 5: Supervisor (Q21)

Transitions: drives `idle → unhealthy`, `unhealthy → probing → idle | unhealthy` and `failed → idle` by asking for probes; never touches a `serving` or `swapping` slot.

**Interfaces:** one task per slot, `supervisor_<slot>`, that calls `slot.request_probe()` every `SUPERVISOR_PROBE_INTERVAL` while the state is `idle`, `unhealthy` or `failed`. The consumer counts consecutive failures, treats a 503 as loading within the grace, and after `SUPERVISOR_FAILURES` failures (and not before the backoff allows) restarts the unit with `ModelSwapper.restart_llama_server()`, or logs `would restart` when `SUPERVISOR_RESTART=0`. A `failed` slot is probed but never restarted. `/status` gains `supervisor` per slot.

### Task 6: Skip a request whose client has gone (Q26)

**Interfaces:** a request item carries `"request": starlette.requests.Request`. At dequeue the consumer awaits `request.is_disconnected()`; if true, the item is answered without a backend call (event set; the endpoint finishes the line with 499 `client_gone`), and the slot never leaves `idle`. Step 1 of this task is a measurement: a real uvicorn server, a client that sends a body and disconnects while queued, and a check that `is_disconnected()` turns true. If it doesn't, the task stops and Claude redesigns it.
