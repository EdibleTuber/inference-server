# Phase 4: Observability — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every log line lands in the journal. Every inference request and every swap logs one structured JSON line. `/status` shows, per slot, what the slot is doing and the last thing that went wrong, so phase 5's state machine is visible while it beds in.

**Architecture:**
- **Journal only:** the four units stop appending to `/var/log/llama/*`, and the manager drops its `LOG_FILE` handler.
- **Request log:** a new module, `manager/request_log.py`, owns the request record, token extraction, and the one JSON line per request or swap. The queue consumer and the stream relay fill in the record. The endpoint emits the line, or for a relayed stream, the stream's closer does.
- **`/status`:** `SlotState` and `RequestQueue` gain the bookkeeping behind `state`, `in_flight`, `in_flight_age_seconds`, `oldest_queue_wait_seconds` and `last_error`.

**Tech Stack:** Python 3.12.3, FastAPI 0.135.2, httpx, systemd/journald (persistent: `/var/log/journal`, default limits, 1.0 GB in use on 2026-11-01).

**Spec:** `docs/design-review-2026-10-29.md`, the "Observability (Q27)" section and the phase 4 row of "Phases and process".

## Global Constraints

- **Test command:** `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/ -q -p no:cacheprovider`.
- **Live ports:** tests never touch live ports 8080–8084 or 11434.
- **Modules:** new modules go flat in `manager/`. `deploy-manager.sh` ships only `manager/*.py`.
- **Client API:** responses may only gain fields. Existing `/status` keys (`loaded_model`, `healthy`, `swapping`, `last_swap_utc`, `queue_depth`, `queue_limit`, `models`) keep their meaning. agent_core and PAL read them.
- **Relayed bytes are untouched:** a stream reaches the client byte-for-byte as the backend sent it, with no added buffering. Observation never changes what the client receives.
- **Observation never fails a request:** an exception while extracting tokens or writing the log line is caught and logged at WARNING, and the request completes as it would have without it.

## How this phase runs (light to medium)

What changed from phase 3, which was medium-heavy:
- **Tests:** Claude still commits each task's tests before the task starts and checks they fail first. That's cheap and has caught real plan defects. Qwen never edits a test.
- **Mutation runs:** only Task 2's are worth it, because it touches the stream relay, which Q10 and the disconnect tests depend on. Tasks 1 and 3 fail loudly: a missing field or a log line in the wrong place shows up at once.
- **Code in the plan:** none, in any task. Task 2 is the only task near concurrency (the relay and its closer), and there the tests pin the invariants.
- **Worktree:** branch `phase4/observability` in `/mnt/secondary/inference-server-wt-phase4`.

## Implementer Rules (Qwen)

1. Do exactly one task, then stop and report "Done" plus a **Deviations** list.
2. Never edit a test file. If one is wrong, stop and report it.
3. Stop and report if a step's Expected result doesn't happen.
4. Never run sudo, systemctl, deploy scripts or `git push`, and never touch ports 8080–8084 or 11434.
5. If anything in this plan turns out to be wrong, deviating is correct; say so in Deviations.
6. Commits stage explicit paths and end with the repo's attribution lines.

## Review Focus

Each item names the task whose tests pin it:

1. **The `timings` line split across two chunks**, which happens whenever the backend's write boundary falls inside it. Token counts must still come out right. → Task 2, `test_stream_tokens_survive_a_line_split_across_chunks`.
2. **A client that leaves mid-stream** must produce exactly one line, with outcome `client_disconnected`, and `in_flight` must return to 0. → Task 2, `test_a_stream_the_client_leaves_logs_once_as_client_disconnected`; Task 3, `test_in_flight_returns_to_zero_on_every_exit_path`.
3. **A backend body that isn't the JSON we expect** (truncated, an HTML error page, a 2 MB line with no newline) must give null tokens, never an exception, and unchanged bytes. → Task 2, `test_unparseable_bodies_give_null_tokens_and_unchanged_bytes`.
4. **A hostile `X-Request-Id` or `User-Agent`** (newlines, quotes, 4 KB) must not break the one-line-per-request format or reach the log unbounded. → Task 2, `test_hostile_headers_cannot_split_or_bloat_the_line`.
5. **The consumer cancelled or raising mid-request** must not leak an `in_flight` entry. A leaked entry would show a slot as `serving` forever. → Task 3, `test_in_flight_returns_to_zero_on_every_exit_path`.

## Decisions made while planning (beyond the spec)

- **"Journal only" covers all four units**, not just the manager: `llama-manager`, `llama-server`, `llama-server-egpu` and `llama-embeddings`. The manager's Python logging already goes to stderr, but its unit appends stderr to `/var/log/llama/manager.err` (`systemd/llama-manager.service:48`), and `LOG_FILE` adds a second copy (`manager/app.py:349-355`). The logrotate config goes too: nothing will write `/var/log/llama` any more.
- **The log format drops `asctime`**, because journald timestamps every line. It becomes `%(levelname)s %(name)s: %(message)s`.
- **The request line is JSON**, logged by the logger `manager.requests` at INFO, so `journalctl -u llama-manager -o cat | sed -n 's/^INFO manager.requests: //p' | jq` reads it. JSON escaping is also what keeps a hostile header from splitting the line.
- **"One line per request" means per inference request** (`/v1/chat/completions`, `/v1/messages`, `/v1/messages/count_tokens`) **and per `/swap`.** uvicorn's access log still covers every other endpoint, so `/status` polling doesn't flood the structured log.
- **Tokens come from llama-server's own fields**, checked against build 11136 on 2026-11-01:
  - OpenAI shape, streamed or not: the `timings` object, which a stream carries only in its last data chunk.
    - `prompt_tokens = prompt_n + cache_n`, `cached_tokens = cache_n`, `completion_tokens = predicted_n`.
    - A non-streamed body with no `timings` falls back to `usage`: `prompt_tokens`, `prompt_tokens_details.cached_tokens` and `completion_tokens`.
  - Anthropic shape: `usage`. A stream carries it in two places: `message_start.message.usage` and the last `message_delta.usage`.
    - `prompt_tokens = input_tokens + cache_read_input_tokens`, `cached_tokens = cache_read_input_tokens`, `completion_tokens = output_tokens`.
  - `count_tokens`: `prompt_tokens = input_tokens`, and the other two are null.
- **`state` is derived from today's flags**, in this priority order: `swapping` (lock held), then `unhealthy`, then `serving` (`in_flight > 0`), then `idle`. Phase 5 replaces the derivation with the real state machine (Q19) and keeps the field name and these four values; it adds `probing` and `failed`.
- **`in_flight` is a count, not a flag.** Until Q10 lands, the consumer hands a stream to the relay and takes the next item while the stream is still running. Two requests can then be in flight on a `--parallel 1` backend, and the count shows exactly that. Phase 5 makes it at most 1.
- **The supervisor's restart count is deferred to phase 5.** No supervisor exists yet, and systemd's `NRestarts` counts something different: automatic `Restart=on-failure` restarts, not the manager's own. A field whose meaning changes in the next phase would be worse than no field.
- **`last_error` records slot faults, not request faults.** It is set by backend 5xx, upstream timeouts, proxy errors, queue errors, swap failures and probe failures. It is not set by anything the client caused: a 4xx from the manager or the backend, `queue_full`, `model_not_loaded` or `client_disconnected`. It is never cleared, so it shows when the last fault happened.
- **`X-Request-Id`:** a client-supplied value matching `[A-Za-z0-9._:-]{1,64}` is kept, and anything else is replaced by `uuid4().hex`. Every inference and `/swap` response carries it, errors included.

---

## Branch `phase4/observability` (worktree `/mnt/secondary/inference-server-wt-phase4`)

### Task 1: Journal only

**Files:**
- Modify: `manager/config.py`: delete `log_file`, `LOG_FILE` and its docstring line.
- Modify: `manager/app.py:335-355`: delete the `FileHandler` and change the format string.
- Modify: `systemd/llama-manager.service`, `systemd/llama-server.service`, `systemd/llama-server-egpu.service` and `systemd/llama-embeddings.service`: delete every `StandardOutput=`/`StandardError=` line and the comment above it. systemd's default is the journal.
- Delete: `config/llama-logrotate`.
- Modify: `scripts/setup.sh:66,91-92,169`: delete the `/var/log/llama` creation and the logrotate install.
- Modify: `scripts/doctor.sh:118-121`: check 4 becomes "no installed unit logs to a file". It fails if any `/etc/systemd/system/llama-*.service` has a `StandardOutput=` or `StandardError=` line, naming the unit.
- Modify: `config/manager.env:48` (delete `LOG_FILE`), `README.md` (lines 76, 407, 581-591, 715, 740) and `manager/README.md` (lines 126, 139). The Logs section says to read logs with `journalctl -u <unit>`, and shows the `jq` pipeline from Decisions.
- Tests, committed by Claude: `tests/test_repo_consistency.py`, `tests/test_config.py` and `tests/conftest.py` (dropping `log_file=`).

**Interfaces:**
- Produces: `ManagerConfig` without `log_file`. `_setup_logging(config)` installs exactly one handler on the root logger, a `StreamHandler` on stderr, using the format `%(levelname)s %(name)s: %(message)s`.

**Tests (Claude):**
- `test_no_unit_logs_to_a_file`: no `systemd/*.service` has a line starting with `StandardOutput=` or `StandardError=`.
- `test_nothing_names_the_old_log_files[README.md|manager/README.md|config/manager.env|scripts/setup.sh]`: none contains `/var/log/llama`, `LOG_FILE` or `logrotate`.
- `test_logging_goes_to_stderr_only`: after `create_app(test_config)`, the root logger has no `logging.FileHandler`, and its stream handler's format has no `asctime`.
- `test_config.py`: the `LOG_FILE` assertions are removed, and `ManagerConfig` has no `log_file` attribute.

- [ ] **Step 1:** Run the suite. Expected: the new tests fail. `test_config.py` and `conftest.py` will also fail until `log_file` is gone.
- [ ] **Step 2:** Make the changes above.
- [ ] **Step 3:** Run the full suite. Expected: everything passes.
- [ ] **Step 4:** Commit: `feat(ops): every service logs to the journal only (Q27)`.

### Task 2: One structured line per request and per swap

**Files:**
- Create: `manager/request_log.py`.
- Modify: `manager/app.py`: `_route_and_wait`, `_queue_consumer`, `_stream_close`/`_relay_stream`, the three inference endpoints and `swap_slot`.
- Tests, committed by Claude: `tests/test_request_log.py` (unit tests for the module) and `tests/test_request_lines.py` (through the app and the consumer, reusing `tests/test_consumer.py`'s fake SSE backend pattern).

**Interfaces (Produces):**
- `REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,64}")`. `def request_id_from(header: str | None) -> str` returns the header if it fully matches, and `uuid4().hex` otherwise.
- `USER_AGENT_MAX = 120`.
- `@dataclass class Tokens`: `prompt: int | None = None`, `cached: int | None = None`, `completion: int | None = None`.
- `def tokens_from_body(body: bytes, dialect: Literal["openai", "anthropic", "count_tokens"]) -> Tokens`: for a non-streamed body, per the token rules in Decisions. It never raises; anything unparseable gives `Tokens()`.
- `class StreamTokens`: `__init__(self, dialect: Literal["openai", "anthropic"])`, `feed(self, chunk: bytes) -> None` and `tokens: Tokens`.
  - It splits on `\n` and keeps only an incomplete trailing line between calls.
  - It parses only lines that start with `data:` and contain `"timings"` or `"usage"`.
  - If the pending partial line grows past 1 MiB, it drops it and keeps going.
  - `feed` never raises.
- `@dataclass class RequestRecord`:
  - fields: `id: str`, `endpoint: str`, `client: str | None`, `user_agent: str | None` (truncated to `USER_AGENT_MAX`), `model: str | None`, `stream: bool`;
  - set later: `slot: str | None`, `enqueued_at: float | None`, `dequeued_at: float | None` (both `time.monotonic()`), `backend_started_at: float | None`, `tokens: Tokens`;
  - `def finish(self, status: int, outcome: str) -> None`: idempotent. The first call writes the line and later calls do nothing.
  - `@classmethod def from_request(cls, request: Request, endpoint: str, body: dict | None) -> RequestRecord`.
- **The request line:** one INFO record on logger `manager.requests`. The message is `json.dumps(obj, separators=(",", ":"), ensure_ascii=True)` of an object with exactly these keys, in this order:
  `event` (`"request"`), `id`, `endpoint`, `client`, `user_agent`, `slot`, `model`, `stream`, `status`, `outcome`, `queue_wait_ms`, `backend_ms`, `prompt_tokens`, `cached_tokens`, `completion_tokens`.
  - `queue_wait_ms` is `round((dequeued_at - enqueued_at) * 1000)`, or null if the request never queued.
  - `backend_ms` runs from `backend_started_at` to `finish`, or is null if the backend was never called.
- **`outcome`:** `"ok"` (the backend answered below 400 and, for a stream, the relay reached the end); `"backend_error"` (the backend answered 400 or above and the manager passed it through); `"client_disconnected"` (a stream closed before its end); otherwise the manager's own error code, i.e. the `_RouteDecision.code` (`invalid_body`, `missing_model`, `model_not_found`, `model_not_loaded`, `slot_unavailable`, `queue_full`, `queue_error`, `upstream_timeout`).
- **The swap line:** the same logger. The object has keys `event` (`"swap"`), `id`, `client`, `user_agent`, `slot`, `model`, `trial`, `status` (the HTTP status sent), `outcome` and `duration_ms`.
  - A refused swap (400 or 404) or an "already loaded" 200 logs at once, with `outcome` set to the error `type` or `"already_loaded"`, and `duration_ms` null.
  - An accepted swap (202) logs when its background task ends, with `outcome` `"ok"` or `"failed"`, and `duration_ms` measured from the 202 to the end.
- **Who calls `finish`:**
  - the endpoint, for every response that isn't a relayed stream;
  - the stream's closer, for a relayed stream: `ok` when the relay generator got to the end, `client_disconnected` otherwise. The closer is already idempotent, and `finish` is too.
- **`X-Request-Id`:** every inference and `/swap` response carries the record's `id`, errors included.

**Tests (Claude):**
- `test_stream_tokens_survive_a_line_split_across_chunks`: the `timings` line from Decisions, fed split at every byte offset, gives `Tokens(12, 0, 8)` each time (from `prompt_n` 12, `cache_n` 0, `predicted_n` 8).
- `test_openai_tokens_from_timings` and `test_openai_tokens_fall_back_to_usage`: both use the 2026-11-01 sample bodies.
- `test_anthropic_stream_tokens_come_from_message_start_and_the_last_delta`.
- `test_unparseable_bodies_give_null_tokens_and_unchanged_bytes`: truncated JSON, an HTML page, and a 2 MiB line with no newline. Each gives null tokens, the relayed bytes equal the backend's, and nothing raises.
- `test_hostile_headers_cannot_split_or_bloat_the_line`:
  - an `X-Request-Id` with `\n`, a quote, or 65 characters is replaced by 32 hex characters;
  - a 4 KB `User-Agent` containing `\n` becomes one line of at most 120 characters of agent;
  - the line parses as JSON.
- `test_a_good_request_id_is_kept_and_echoed`: `X-Request-Id: abc-123` appears in the response header and in the line.
- `test_one_line_per_request_with_every_key[chat|messages|count_tokens|stream]`: exactly one `manager.requests` record per request. Its keys equal the list above, in order, and `queue_wait_ms`/`backend_ms` are non-negative ints (null for count_tokens' `queue_wait_ms`).
- `test_manager_errors_log_their_code[model_not_found|missing_model|queue_full|model_not_loaded]`.
- `test_a_backend_5xx_logs_backend_error_with_its_status`.
- `test_a_stream_the_client_leaves_logs_once_as_client_disconnected`: driven with `tests/test_consumer.py`'s `drive(..., disconnect=)`.
- `test_a_stream_that_completes_logs_ok_with_its_tokens`.
- `test_swap_lines[refused|already_loaded|accepted_ok|accepted_failed]`.
- `test_a_raising_log_write_does_not_fail_the_request`: `logging.Logger.info` on `manager.requests` is patched to raise, and the request still returns its 200 and its bytes.

- [ ] **Step 1:** Run `tests/test_request_log.py` and `tests/test_request_lines.py`. Expected: they all fail, because `manager.request_log` doesn't exist.
- [ ] **Step 2:** Implement `manager/request_log.py` per the Interfaces block.
- [ ] **Step 3:** Wire it into `app.py`.
  - The endpoints build the record with `RequestRecord.from_request` and set the response header.
  - `_route_and_wait` takes the record, stamps `enqueued_at`, and puts it in the item as `item["record"]`.
  - The consumer stamps `dequeued_at` and `backend_started_at`, feeds `tokens_from_body` for non-streams, and wraps the relay generator so each chunk also goes to a `StreamTokens`.
- [ ] **Step 4:** Run the full suite. Expected: everything passes, including `tests/test_stream_disconnect_real.py` unchanged.
- [ ] **Step 5:** Commit: `feat(observability): one structured line per request and per swap (Q27)`.

### Task 3: `/status` shows what each slot is doing

**Files:**
- Modify: `manager/queue.py`: record a `time.monotonic()` per enqueued item.
- Modify: `manager/slots.py`: add the in-flight table, `last_error` and the new `to_status_dict` keys.
- Modify: `manager/app.py`: the consumer calls `begin_request`/`end_request`; the swap failure tail and `probe` record errors.
- Modify: `README.md`: the `/status` field table.
- Tests, committed by Claude: `tests/test_slot_status.py`.

**Interfaces:**
- Consumes: `RequestRecord.id` from Task 2.
- Produces:
  - `RequestQueue.oldest_wait_seconds(now: float) -> float | None`: the age of the item at the head of the queue, or None if the queue is empty. `drain()` and `dequeue()` keep it consistent: a parallel deque of timestamps, popped in step.
  - `SlotState.begin_request(request_id: str) -> None` and `SlotState.end_request(request_id: str) -> None`. `end_request` is idempotent and ignores an unknown id.
  - `SlotState.in_flight_count: int` and `SlotState.oldest_in_flight_seconds(now: float) -> float | None`.
  - `SlotState.record_error(code: str, message: str) -> None` sets `last_error = {"code", "message", "at_utc"}`, with `at_utc` as an ISO-8601 UTC string. The codes are `backend_error`, `upstream_timeout`, `proxy_error`, `queue_error`, `swap_failed` and `probe_failed`.
  - `SlotState.state: str` is `"swapping"`, `"unhealthy"`, `"serving"` or `"idle"`, in that priority order.
  - `to_status_dict()` gains `state`, `in_flight`, `in_flight_age_seconds`, `oldest_queue_wait_seconds` and `last_error`. The ages are floats rounded to 1 decimal, or null.
- **When a request is in flight:** from just before the consumer calls the backend until the response is complete. For a non-stream, that's the response received or the error caught. For a relayed stream, that's when its closer runs. Every exit path calls `end_request`: the consumer's `finally` for anything it doesn't hand to a relay, and the closer for what it does. `count_tokens` bypasses the queue and is never in flight.

**Tests (Claude):**
- `test_state_priority[swapping_beats_unhealthy|unhealthy_beats_serving|serving|idle]`.
- `test_in_flight_counts_a_held_stream_then_drops_on_close`.
- `test_in_flight_returns_to_zero_on_every_exit_path[ok|backend_5xx|read_timeout|connect_error|readiness_409|consumer_cancelled|client_disconnected]`.
- `test_two_requests_in_flight_when_a_stream_is_relayed_and_the_next_starts`: this pins today's pre-Q10 behavior. Phase 5 changes this test on purpose.
- `test_oldest_queue_wait_tracks_the_head_through_dequeue_and_drain`.
- `test_last_error_is_set_by_slot_faults[backend_5xx|upstream_timeout|proxy_error|swap_failed|swap_raised|probe_failed]` and `test_last_error_is_not_set_by_client_faults[backend_4xx|queue_full|model_not_loaded|client_disconnected]`.
- `test_status_keeps_every_existing_key`: every key from before this phase is present, with its old meaning.

- [ ] **Step 1:** Run `tests/test_slot_status.py`. Expected: it fails.
- [ ] **Step 2:** Implement per the Interfaces block. Then update README's `/status` table, including the note that `in_flight` can reach 2 until phase 5.
- [ ] **Step 3:** Run the full suite. Expected: everything passes.
- [ ] **Step 4:** Commit: `feat(status): each slot's state, in-flight age, oldest queue wait and last error (Q27)`.

### Task 4: Host cutover (the user runs these; Claude verifies each step)

Claude runs the read-only half of each command before handing it over.

1. **Units first:** after the merge, install the four units from the main checkout, then run `daemon-reload`. Nothing restarts yet.
2. **Deploy** from the main checkout: `sudo /mnt/secondary/inference-server/scripts/deploy-manager.sh`. Its manager restart is the only one, and it comes up on the new code and the new unit. Then restart `llama-embeddings`.
   - The two backends pick up their new units on their next restart, which costs a model reload. Do it at a quiet time, or let the next swap do it.
3. **Logrotate:** `sudo rm /etc/logrotate.d/llama`. Old files in `/var/log/llama/` stay where they are until you delete them.
4. **Claude verifies:**
   - `journalctl -u llama-manager` shows the manager's own lines;
   - one short chat produces exactly one `manager.requests` line with tokens;
   - `/status` shows the new keys;
   - doctor check 4 passes.

**Rollback:** reinstall the previous units from `git show <previous merge>:systemd/<unit>`, run `daemon-reload`, then deploy the previous merge commit.
