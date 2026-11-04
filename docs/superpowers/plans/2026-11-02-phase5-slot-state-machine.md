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

- **States and transitions** (the final table, after Task 5; earlier tasks use a subset, named in each task). The table is over **stored** states:

  | From | To |
  |---|---|
  | `idle` | `swapping`, `unhealthy` |
  | `swapping` | `idle`, `failed` (Task 4; until then `unhealthy`) |
  | `unhealthy` | `probing`, `swapping` |
  | `probing` | `idle`, `unhealthy` |
  | `failed` | `idle` (a healthy probe), `swapping` (a `/swap`) |

  **`serving` is not stored** (changed while writing Task 2's tests, 2026-11-02): the slot reads `serving` when its stored state is `idle` and a request is in flight. A stored `serving` would need the stream's closer, which runs on the response's task, to write the state, or the slot would stay `serving` after a consumer that had already exited; it would also log two INFO lines per request. A stored state other than `idle` shows as itself, so an `unhealthy` slot with a request still in flight reads `unhealthy`. A slot starts `unhealthy`; the lifespan's `refresh` asks for the first probe. Setting the current state again is a no-op. Any other move raises `IllegalTransition` and is a bug. Every real transition logs one INFO line on `manager.slots`: `slot=<name> <from> -> <to>`.
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
- **Q26 uses Starlette's `Request.is_disconnected()`**, called by the consumer when it dequeues an item. A gone client's item is answered (event set) without a backend call, and its line has status 499 and outcome `client_gone`. Measured on 2026-11-04: one check is not reliable for a large body on the production stack; a second check one loop tick later is. Task 6 has the details.

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

Transitions (stored): `unhealthy → probing → idle | unhealthy`, `idle → unhealthy`, `idle | unhealthy → swapping → idle | unhealthy`. A probe of an `idle` slot leaves it `idle` while it runs. Every write of a slot's state or `loaded_model` happens on that slot's consumer task.

**Files:** `manager/slots.py`, `manager/app.py`, `README.md` (the `/status` `state`, `healthy` and `swapping` rows); test `tests/test_state_machine.py` (new, Claude) plus reworked tests.

**Interfaces (produced):**
- `class IllegalTransition(ValueError)` and `TRANSITIONS: dict[str, frozenset[str]]` in `manager/slots.py`, keyed by stored state; for this task exactly the moves above. `serving` appears nowhere in it.
- `SlotState.state -> str`: a read-only property over a stored field, which reads `serving` for a stored `idle` with a request in flight. `SlotState._set_state(new: str) -> None` is a no-op for the current stored state, raises `IllegalTransition` for a move not in `TRANSITIONS` (state unchanged, nothing logged), and logs one INFO line `slot=<name> <from> -> <to>` on `manager.slots` per real move.
- `SlotState.probe` moves an `unhealthy` slot to `probing` before calling the backend, then to `idle` or `unhealthy`; an `idle` slot stays `idle` during the call and moves to `unhealthy` only if it fails. `mark_unhealthy()` moves to `unhealthy`; `mark_swapped(model)` moves `swapping → idle` and raises `IllegalTransition` from anywhere else. `ensure_model_on_slot` moves to `swapping` before the swap runs (after the already-loaded shortcut and the profile checks).
- `SlotState.healthy -> bool`: a property, true when the state is `idle` or `serving`. It keeps a setter for tests only: `True` stores `idle`, `False` stores `unhealthy`, bypassing the table and logging nothing. No code in `manager/` assigns `healthy`, and `healthy` is no longer a dataclass field (a test reads the source to pin both). `to_status_dict()["swapping"]` is `state == "swapping"`, not the lock.
- `async def SlotState.refresh(self, timeout: float) -> bool`: ask the consumer to probe and wake it (`queue_event`); return True once a probe it asked for has finished, False on timeout. Returns False at once, without asking, when the state is `serving` or `swapping`. Callers waiting at the same time share one probe, and one caller timing out does not cancel the probe for the others.
- `ServerState.reprobe_all_slots()` (sequential) and `reprobe_all_slots_concurrently()` (`gather`, `return_exceptions=True`) keep their names, call `refresh` with `REFRESH_TIMEOUT = 5.0` (in `manager/app.py`) and never raise. `_validate_and_route`'s reprobe of an unhealthy slot calls `refresh` too. `_reprobe_slot_if_unlocked` may go.
- `_reprobe_for` is deleted: after a 5xx or a proxy error the consumer moves the slot to `unhealthy` and probes it itself right after the item, without waiting for another item.
- The startup probe moves into the consumer: the lifespan starts the consumers, then awaits `refresh` on each slot before logging `slot=<name> ready with model <model>` / `slot=<name> not reachable at startup` (unchanged text).

**Invariants:** every call of `SlotState.probe`, `_set_state`, `mark_swapped` and `mark_unhealthy` runs on the task named `queue_consumer_<slot>` (pinned at runtime by recording `asyncio.current_task().get_name()` across startup, `/status`, routing including the stale-view reprobe, a 5xx and a swap); `/status` never waits on a `serving` or `swapping` slot. Behaviour change: during a swap `healthy` is false.

**Before Task 3** a handed-off stream still lets the consumer take the next item, so a swap can start while a stream relays (stored `idle → swapping` is legal). Task 3 closes that.

**Existing tests reworked in Claude's commit:** `tests/test_endpoints.py`'s `client` fixture now runs the lifespan (consumers are the only probers), `test_status_endpoint_does_not_block_event_loop` runs the lifespan, `test_the_reprobe_skips_a_slot_mid_swap` uses the `swapping` state instead of the lock, and `test_reprobe_waits_for_swap_lock` is deleted with `_reprobe_for` (replaced by `test_after_a_fault_the_consumer_probes_the_slot_before_its_next_item`); in `tests/test_consumer.py`, `test_unhealthy_slot_is_reprobed_and_routed` and `test_unhealthy_slot_still_503s_when_the_probe_fails` run a consumer, and `test_unhealthy_slot_mid_swap_is_not_probed` uses the `swapping` state; `tests/test_slots.py::test_mark_swapped_sets_all_three_fields` enters `swapping` first; `tests/test_slot_status.py::test_state_priority` builds its states through the machine; `tests/test_profile_roster.py::test_a_slot_with_no_home_models_reports_an_empty_list` runs the lifespan; `tests/test_swap_queue.py`'s `running` starts every slot's consumer, and `test_last_swap_reports_each_finished_swap_by_request_id` reads `/status` only while they run.

- [ ] **Step 1: Claude's tests are committed.** Run: `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/test_state_machine.py -q -p no:cacheprovider`. Expected: FAIL (collection error: no `IllegalTransition`).
- [ ] **Step 2: Implement** the stored state, the table, `_set_state`, the derived `healthy`, `refresh`, the consumer serving probe requests and post-fault probes, the reprobe callers, and the lifespan order.
- [ ] **Step 3: Update `README.md`** (the `/status` `state`, `healthy` and `swapping` rows: `probing` is a new `state` value; `healthy` is false while swapping).
- [ ] **Step 4: Run the full suite** twice. Expected: all pass, and no test takes about 5 s (that is a `refresh` waiting on a consumer that isn't running).
- [ ] **Step 5: Commit** `manager/app.py manager/slots.py README.md`.

### Task 3: A slot is busy until its response completes (Q10)

Transition: the slot reads `serving` (stored `idle`, a request in flight) until the response is complete: a stream's closer has run, or a non-streamed response has been built.

**Interfaces:** the consumer awaits the relay's closer before taking the next item. If the relay's generator has not started and its background task has not run within `STREAM_START_TIMEOUT = 30.0` s of the handoff, the consumer runs the closer itself and the line's outcome is `client_disconnected`.

**Invariants:** `in_flight` is at most 1; a client that leaves mid-stream frees the slot as soon as the closer runs; the item's event is set at the handoff, not at the close (the client must get its response to start reading). Claude rewrites `test_two_requests_in_flight_when_a_stream_is_relayed_and_the_next_starts`, which pins today's two-in-flight behaviour on purpose. The README's `/swap` section (Task 1) already says requests in flight run before a queued swap; for a relayed stream that becomes true here, so Task 3's review checks it. The README's `/status` `in_flight` row says it "can reach 2 until phase 5"; Task 3 rewrites that row to state the property (at most 1).

Added while writing Task 3's tests (2026-11-03):
- The request ends (`end_request`, the line, and the consumer moving on) when the closer runs, from whichever of the three paths runs it: the generator's `finally`, the background task, or the consumer at `STREAM_START_TIMEOUT`. That holds even if closing the backend stream raises. When the close raises in the generator's `finally`, Starlette never reaches the background task (`starlette/responses.py:257-284`), so today's `finish_stream` would never run on that path.
- "Started" means the generator has begun or the background task has run. Once a client is reading, the stream runs as long as the backend sends; `STREAM_START_TIMEOUT` never cuts it off.
- Cancelling the consumer while it waits leaves the stream to its client. Shutdown must not close a response that is being read.
- A closer that runs late (the consumer closed an unstarted stream, then the response's background task ran) ends only its own request. In-flight keys stay per request, never the client's `X-Request-Id`.
- The item's event is still set at the handoff, before the wait.

**Tests (Claude's commit):** `tests/test_stream_busy.py` (7 tests). In `tests/test_slot_status.py`, `test_two_requests_in_flight_when_a_stream_is_relayed_and_the_next_starts` becomes `test_one_request_in_flight_while_a_stream_relays_then_the_next_starts`, and `test_two_in_flight_requests_sharing_a_client_request_id_are_both_counted` becomes `test_a_late_closer_does_not_end_a_later_request_sharing_its_client_id` (with one request in flight, a late closer is the remaining way one request could end another). Tests monkeypatch `manager.app.STREAM_START_TIMEOUT`, so the consumer must read it at call time.

### Task 4: A failed swap falls back once; then `failed` (Q20)

Transitions: `swapping → idle` (fell back), `swapping → failed`, `failed → swapping`, `failed → idle`.

**Interfaces:** `SlotState.known_good: str | None`, the model the slot last served healthily (set by a healthy probe and by a successful swap). A failed swap (raised or timed out) swaps once to `known_good` (not a trial, so the already-loaded shortcut can't apply: the backend was just restarted). Success: `idle`, swap outcome `fell_back`, message naming the failure. Failure, or no `known_good`: `failed`. While `failed`, every request item and every queued request is answered at once with 503 `slot_failed`; a swap item still runs. `scripts/onboard.py` sends its own `X-Request-Id` and decides from `last_swap` with that id: landed when its outcome is `ok`; failed for any other outcome.

Added while writing Task 4's tests (2026-11-03):
- **Table:** `swapping → unhealthy` goes. A failed swap now either falls back or fails, so the final table is the one in "Decisions". `mark_unhealthy` on a `swapping` or `failed` slot raises `IllegalTransition`.
- **`known_good`:** every healthy probe sets it, including one that moves nothing, and so does `mark_swapped`. A failed probe, or one that finds no model, leaves it alone. It starts `None`, and the test-only `healthy` setter does not set it.
- **Probing a `failed` slot:** the slot stays `failed` while the probe runs (no `probing`). A healthy probe moves it to `idle`; a failed probe leaves it `failed` and only records the error.
- **The fallback:** it runs once, straight through `swap_to`, and the slot stays `swapping` throughout. When it lands, the slot is `idle` and the queue keeps running against the known-good model, with nothing drained. `last_swap` has outcome `fell_back`, and its message names the failure and the known-good model. `last_error` keeps the original `swap_failed`.
- **Entering `failed`:** every queued request is answered at once with `slot_failed`, including one behind another queued swap. Queued swap items stay in order and run (a `/swap` recovers the slot), so the `drained` outcome no longer occurs. The `failed` swap's message names the failure, and the known-good model when the fallback tried one.
- **While `failed`:** the consumer answers a request item with `slot_failed` without a readiness check or a backend call. Routing returns 503 `slot_failed` after its `refresh`, so a healthy probe still recovers the slot first. The error type is `slot_failed` on `/v1/chat/completions` and `api_error` on `/v1/messages`. `_decision_from_item_error` maps `slot_failed` to 503. A `/swap` on a `failed` slot is always 202.
- **`onboard`:** each swap sends its own `X-Request-Id`, and the result is judged only from the `last_swap` entry carrying that id. An `ok` outcome means the swap landed; any other outcome fails at once and prints that outcome and its message. The old checks on `healthy`, `swapping` and `last_swap_utc` go.

**Tests (Claude's commit):** `tests/test_fallback.py` (12 tests). In `tests/test_onboard.py`, `FakeManager` now sets `last_swap` keyed by the `X-Request-Id` it received, and there are three new tests: `test_onboard_reports_a_failed_trial_that_fell_back` (Review Focus 5), `test_each_swap_carries_its_own_request_id` and `test_a_last_swap_from_another_request_is_not_taken_for_this_one`; also, `test_a_refused_swap_exits_1_without_polling` now accepts a swap with no pre-read of `/status`. In `tests/test_state_machine.py`, the table tests now cover the final table (`test_the_table_is_exactly_the_final_table`, with `failed` added to `put`), `mark_unhealthy` is split into the legal and illegal cases, and a failed swap with no known-good model ends `failed`. `tests/test_swap_queue.py::test_a_swap_drained_by_a_failed_swap_ahead_still_gets_its_line` becomes `test_a_swap_queued_behind_a_failed_swap_still_runs`.

**Review round 1 (2026-11-03, on `28966f9`):**
- **The 503 names the fault that failed the slot, not the latest probe.** `_slot_failed_message` reads `slot.last_error` when it answers. Routing refreshes a failed slot before answering, and so does `GET /status`. A failed probe records `probe_failed`, so in practice the 503 says "probe failed: ..." instead of naming the swap. Fix: keep the reason on the slot when it enters `failed` (the tail's final message), and build the 503 from that. `last_error` stays "the last fault", unchanged. Test: `test_the_503_names_the_fault_that_failed_the_slot_after_a_failed_probe`.
- **Decided: the fallback goes to a different model.** Qwen's guard (`not same_model(known_good, model_name)`) was not in the plan. Loading the model that just failed again would double the outage for nothing, and a `/swap` retries if wanted, so the guard stays. Its comment should state it as a rule ("a fallback is to a different model"), and the README's `/swap` paragraph should say so. Test: `test_a_failed_swap_whose_known_good_is_the_same_model_fails_without_retrying`.
- **`onboard` prints the outcome and its message.** Already correct; it is now pinned in `test_onboard_reports_a_failed_trial_that_fell_back`.
- **`manager/README.md:78`** still says a timed-out swap marks the slot unhealthy and drains its queue with 503s. Rewrite it to match the main README.
- Mutator survivors judged not to need tests: A13 (pinned in `tests/test_endpoints.py::test_ensure_model_on_slot_skips_swap_if_already_loaded`), A14 (equivalent: without the guard `render_model_env` raises inside the same `try`, and the outcome is unchanged), O5 (the code decides before it checks the deadline; this is only observable on the final poll).

### Task 5: Supervisor (Q21)

Transitions: drives `idle → unhealthy`, `unhealthy → probing → idle | unhealthy` and `failed → idle` by asking for probes; never touches a `serving` or `swapping` slot.

**Interfaces:** one task per slot, `supervisor_<slot>`, that calls `slot.request_probe()` every `SUPERVISOR_PROBE_INTERVAL` while the state is `idle`, `unhealthy` or `failed`. The consumer counts consecutive failures, treats a 503 as loading within the grace, and after `SUPERVISOR_FAILURES` failures (and not before the backoff allows) restarts the unit with `ModelSwapper.restart_llama_server()`, or logs `would restart` when `SUPERVISOR_RESTART=0`. A `failed` slot is probed but never restarted. `/status` gains `supervisor` per slot.

Added while writing Task 5's tests (2026-11-03):
- **New module `manager/supervisor.py`** (it imports nothing from `manager.slots`; `slots.py` imports from it):
  - `PROBE_HEALTHY, PROBE_LOADING, PROBE_FAILED = "healthy", "loading", "failed"`.
  - `@dataclass class Supervision`: `consecutive_failures: int = 0`, `restarts: int = 0`, `restarts_suppressed: int = 0`, plus whatever backoff bookkeeping you need (the tests construct it with no arguments).
  - `judge(sup, result, *, stored_state, grace_started, config, now) -> bool`. It is pure: it reads no clock and touches only `sup`. It returns True when a restart is due.
    - A healthy result resets the count and the backoff.
    - A loading result while `now - grace_started < config.swap_timeout` changes nothing: it neither counts nor resets.
    - Any other result counts one failure.
    - A restart is due when the count is at least `config.supervisor_failures`, `stored_state == "unhealthy"`, and the backoff allows it. The first restart is always allowed. After that, each due restart must come at least the current delay after the previous one; the delay starts at `supervisor_backoff`, doubles each time, and is capped at `supervisor_backoff_max`.
    - When a restart is due, the count resets to 0. This holds in observe-only mode too, so the backoff paces the `would restart` lines.
    - A `failed` slot is counted but never due.
  - `async def supervise(slot, interval: float) -> None`: forever, sleep `interval`, then call `slot.request_probe()` if `slot.state` (the derived state, so `serving` is excluded) is `idle`, `unhealthy` or `failed`. It writes nothing.
  - `supervisor_status(sup, config) -> dict`: exactly `{"restart_enabled", "restarts", "restarts_suppressed", "consecutive_failures"}`.
- **`SlotState`:**
  - `supervision: Supervision` (default factory).
  - `grace_started: float`, which is `time.monotonic()` when the slot is made, so the manager's own start is a grace. `_set_state` resets it on every move out of `idle`. Moves among other states never reset it: each probe of an unhealthy slot goes `unhealthy → probing → unhealthy`, and if that reset the grace a 503 would never count.
  - `probe()` now returns `PROBE_HEALTHY` (a usable model), `PROBE_LOADING` (HTTP 503) or `PROBE_FAILED` (anything else, including no usable model). What it records and how it moves the state are unchanged.
- **The consumer** (`_serve_probe_requests`):
  - After each probe, and after `finish_probe()` (so refresh waiters never wait behind a restart), it calls `judge` with `time.monotonic()`. A probe that raises counts as failed.
  - When a restart is due and `config.supervisor_restart` is true, it calls `slot.swapper.restart_llama_server()` under `slot.swap_lock`, on the consumer's own task. A restart that returns increments `restarts` and sets `grace_started` to now. A restart that raises is logged at ERROR with its exception; it is not counted and the grace is not reset. The consumer carries on.
  - When restarts are off, it increments `restarts_suppressed` and logs one WARNING containing `would restart`, the slot name and its unit.
  - Every probe the consumer runs is judged, whoever asked for it (supervisor, `/status`, routing, a 5xx).
- **Lifespan:** one `asyncio` task per slot named `supervisor_<slot>`, running `supervise(slot, config.supervisor_probe_interval)`, cancelled and awaited at shutdown.
- **`/status`:** each slot gains `supervisor: supervisor_status(...)`.
- **`ManagerConfig`:**
  - `supervisor_probe_interval: float = 15.0` (`SUPERVISOR_PROBE_INTERVAL`), `supervisor_failures: int = 3` (`SUPERVISOR_FAILURES`), `supervisor_backoff: float = 30.0` (`SUPERVISOR_BACKOFF`), `supervisor_backoff_max: float = 600.0` (`SUPERVISOR_BACKOFF_MAX`), `supervisor_restart: bool = False` (`SUPERVISOR_RESTART`, exactly `0` or `1`).
  - A bad value raises `ValueError` naming its key. The floats accept `2.5`.
  - The template `config/manager.env` must not turn restarts on.
- **Known consequence, accepted:** the grace is `SWAP_TIMEOUT`, which the template sets to 600, so a backend stuck answering 503 is restarted only after about 10 minutes.

**Tests (Claude's commit):** `tests/test_supervisor.py`.

**Review round 1 (2026-11-03, on `be6d718`):**
- **A healthy probe resets the delay, not the time since the last restart.** `judge`'s healthy branch clears `_last_due`, so the restart after a healthy probe is allowed at once. A backend that comes up and dies straight away would be restarted again within seconds. The rule stands as written: every due restart comes at least the current delay after the previous one, and a healthy probe only puts that delay back to `supervisor_backoff`. Test: `test_a_healthy_probe_does_not_lift_the_delay_since_the_last_restart`. The README's backoff wording ("resets to the base on a healthy probe") is still true and should stay.
- Accepted deviation: `ready_for`'s own probe predates this task and is not judged. It runs under the held `swap_lock`, so a restart there would deadlock, and the supervisor's tick still covers a dead backend.
- Mutator survivor P10 (sleep before the first ask) is now pinned by `test_the_supervisor_waits_one_interval_before_its_first_ask`. `test_the_supervisor_asks_for_a_probe_each_interval` also gains an upper bound. Qwen's code already passes both.

### Task 6: Skip a request whose client has gone (Q26)

**Interfaces:** a request item carries `"request": starlette.requests.Request`. At dequeue the consumer awaits `request.is_disconnected()`; if true, the item is answered without a backend call (event set; the endpoint finishes the line with 499 `client_gone`), and the slot never leaves `idle`. Step 1 of this task is a measurement: a real uvicorn server, a client that sends a body and disconnects while queued, and a check that `is_disconnected()` turns true. If it doesn't, the task stops and Claude redesigns it.

**Step 1, done by Claude (2026-11-04).** The scratch app ran on uvicorn 0.42.0, Starlette 1.0.0 and httptools 0.7.1, on port 18991. Its endpoint reads the body and waits on a queue, and a separate consumer task calls `is_disconnected()` 0.3–1 s after the client closes.
- **Production stack:** the unit passes no `--http`, so uvicorn chooses httptools. With a 1 KB body, the first check sees the disconnect. With a 2 MB body, the first check said "connected" 17 times in 20, and a second check after one `await asyncio.sleep(0)` was right 20 times in 20.
- **Other stacks:** h11 saw it on the first check. A client still connected never read as gone on any stack.
- **Why:** httptools stops reading the socket once more than 64 KiB of body is buffered (`uvicorn/protocols/http/httptools_impl.py:309-310`). Only `receive()` resumes reading (`:569`), so the client's close is not read and `connection_lost` (`:121-123`) never marks the request disconnected. `Request.is_disconnected()` (`starlette/requests.py:317-329`) calls `receive()` once under a cancel scope that is already cancelled. That resumes reading but returns before the event loop can read the close, so the disconnect shows on the next loop iteration.
- So: **check, yield one loop tick, check again.**

Added while writing Task 6's tests (2026-11-04):
- **`_client_is_gone(slot_name: str, request) -> bool`** (`manager/app.py`):
  - Returns True if `await request.is_disconnected()` is true; otherwise awaits `asyncio.sleep(0)` and returns the second check.
  - Never raises. If the check raises, it logs one WARNING on `manager.app` naming the slot and returns False (connected), so the item is served as before.
  - It must not wait for a disconnect: a connected client is served at once.
- **The check in `_process_request_item`:**
  - It is the first thing inside the outer `try`. That puts it after the `record.slot` and `record.dequeued_at` stamps, so the line has `queue_wait_ms`.
  - It comes before the `failed` branch, `ready_for`, `begin_request` and any backend call. A gone client on a failed slot gets `client_gone`: nobody is there to read `slot_failed`, and the line should say why the request was not served.
  - A gone client's item gets `item["error"]` (a message naming the slot) and `item["error_code"] = "client_gone"`, its event is set, and it returns. The slot's state, `in_flight` and `last_error` are untouched.
  - An item with no `"request"` key is served exactly as before. Read it with `item.get("request")`.
  - Stream and non-stream requests are checked the same way.
  - **Swap items are never checked.** `/swap` has already answered 202 (`app.py`, the swap endpoint), and once a response is complete, uvicorn's `receive()` reports a disconnect (`httptools_impl.py:573`). So a swap item carrying its Request would always read as gone, and every queued swap would be skipped.
- **Wiring:**
  - `_route_and_wait` gains `request: Request | None = None` and sets `item["request"]` when it is given.
  - `/v1/chat/completions` and `/v1/messages` pass their `request`.
  - `count_tokens` and `/swap` do not queue requests and are unchanged.
- **The 499:**
  - `_decision_from_item_error` maps `client_gone` to status 499, code `client_gone`. Without that mapping it falls through to 503 `queue_error`.
  - `_openai_error_response` renders `{"error": {"type": "client_gone", "message": ...}}`.
  - `_anthropic_error_response`'s `type_by_code` gains `"client_gone": "client_gone"`. Anthropic has no equivalent error type, so it keeps the manager's code, as `model_not_loaded` already does. Without the key, the lookup raises `KeyError`: a 500 and no line.
  - No `Retry-After`, because that header is only for 503.
  - The endpoint writes the line, as it does for every other item error: one line, status 499, outcome `client_gone`, `slot` set, `queue_wait_ms` an int, `backend_ms` null. The endpoint is not cancelled when its client goes (measured), so it always gets there.
- **A client gone mid-body:**
  - In those two endpoints, `except ClientDisconnect` (from `starlette.requests`) comes before the existing `except Exception` around `await request.json()`.
  - It answers 499 `client_gone` in the endpoint's error shape and writes the line with `record.finish(499, "client_gone")`. Nothing is queued, so the line's `slot` is null.
  - Until now that disconnect reached `except Exception` and was answered 400 `invalid_body`.
- **Accepted:** a client that leaves after the check is served in full. The check is best effort at dequeue, and the window is one consumer step. A stream whose client leaves mid-relay is still handled by Task 3's closer, with outcome `client_disconnected`. That is a different case and a different name; keep the two apart.

**Files:** `manager/app.py`. In `README.md`, add one sentence after the paragraph about the `manager.requests` line ("View logs"): a request whose client disconnected before it was served is not run, and its line has status 499 and outcome `client_gone`.

**Tests (Claude's commit):**
- `tests/test_client_gone.py`: hermetic consumer tests with fake Requests (gone, connected, gone after a loop tick, raising), and endpoint tests driven as ASGI with a `receive()` that reports the disconnect.
- `tests/test_client_gone_real.py`: real uvicorn with httptools and the real app on ephemeral ports. Four clients with a 1 MB body and one with a small body close while queued behind a held request. All five must be answered 499 `client_gone`, and the fake backend must see only the held request. Against a single-check version of the reference, this test failed 5 times in 5.
- A `client_gone` row in `tests/test_consumer.py::test_item_error_codes_map_to_statuses`.

Run `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/test_client_gone.py tests/test_client_gone_real.py tests/test_consumer.py -q -p no:cacheprovider` (expected: all pass), then the full suite twice.

### Branch review round 1 (2026-11-04)

Claude reviewed the whole branch against `main` after Task 6, looking at how the six tasks meet in `_process_request_item`, the consumer, the failure tail and the supervisor. Nothing was wrong across task boundaries. This round has four findings: one config gap from Task 5 and three README passages that earlier tasks made stale. Claude's tests are in `tests/test_supervisor.py`.

1. **Supervisor settings out of range (Task 5).** `manager/config.py` accepts any float for `SUPERVISOR_PROBE_INTERVAL`, `SUPERVISOR_BACKOFF` and `SUPERVISOR_BACKOFF_MAX`, and any int for `SUPERVISOR_FAILURES`. Measured: `asyncio.sleep(-5)` returns at once and `asyncio.sleep(nan)` never returns. So an interval of 0 or below re-arms a probe on every loop tick (`supervisor.py`, `supervise`: `await asyncio.sleep(interval)`), and nan or inf leaves the supervisor silently dead. A backoff of 0, below 0 or nan never holds a restart back (`judge`: `now - sup._last_due < delay`).
   - **The rule:** in `ManagerConfig.from_env`, the interval, the backoff and the cap must each be finite and > 0. `SUPERVISOR_FAILURES` must be >= 1. `SUPERVISOR_BACKOFF_MAX` must be >= `SUPERVISOR_BACKOFF`.
   - Any violation raises `ValueError` with a message naming the key, like the existing `_int_env` and `_float_env` errors. For the cap rule, that key is `SUPERVISOR_BACKOFF_MAX`.
   - Values at the edge are legal: 0.5, 1, and a base equal to its cap.
   - Give each of the four `SUPERVISOR_*` rows in the README configuration table its range, and add the same ranges to the commented block in `config/manager.env`.
2. **README `/swap` row in the endpoints table** (`README.md`, `| POST | /swap |`). It still says "Returns 200 if already loaded and no trial was requested, 202 when a swap starts". That was true before Task 1. Now 200 comes only when nothing is ahead: the slot is idle, its queue is empty, the model is already loaded and no trial was requested. Otherwise 202 means the swap is queued, and 503 `queue_full` means the queue is at its limit. The poll is `GET /status`, through `swapping`, `loaded_model`, `healthy` and `last_swap`. The rewrite must match the "Switch to a different model" section; don't restate any numbers.
3. **README `SWAP_TIMEOUT` row in the configuration table.** It still says "Exceeding it marks the slot unhealthy." That was true before Task 4. Now a swap that exceeds it has failed and takes the failure tail: one fallback swap to the known-good model, and `failed` when that fails or there is none. Point to the "Switch to a different model" section instead of repeating it. Add one clause: the supervisor uses the same value as its loading grace, in which a 503 is loading, not failed. The supervisor paragraph gives the details.
4. **README supervisor paragraph** ("Each slot has a supervisor…"). It reads as if only the supervisor's own probes count. In fact the consumer judges every probe it runs (`_serve_probe_requests` → `_apply_supervision`), including the ones `/status` and routing ask for through `refresh`. A client that polls `/status` faster than `SUPERVISOR_PROBE_INTERVAL` makes a restart due sooner than the interval alone would. Say that every probe counts, whoever asked for it. The behavior itself is not changed in this round.

**Files:** `manager/config.py`, `README.md`, `config/manager.env`.

Run `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/test_supervisor.py -q -p no:cacheprovider` (expected: all pass), then the full suite twice.
