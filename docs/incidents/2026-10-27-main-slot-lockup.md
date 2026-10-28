# Incident: main slot locked out, 2026-10-27 23:07–23:47 UTC

Status: root-caused, **not fixed**. All defects below are pre-existing in the
deployed code; none was introduced by a proposed change.

## Provenance

Deployed manager: `/opt/llama/manager/DEPLOYED_FROM` = `7b8ea97`. sha256 of
deployed `app.py`/`slots.py` matches `7b8ea97`, not branch HEAD `2fbefeb`; the
only differences from HEAD are `POST /swap` (blocking in deployed) and the
`swapping` field in `/status`. Every line cited below is identical in both.

Evidence sources: `/var/log/llama/manager.err` (manager's Python logging —
`manager.log` is not writable, see F7), `/var/log/llama/llama-server.err`,
`journalctl -u llama-manager -u llama-server`.

## Timeline

| UTC | Event | Source |
|---|---|---|
| 23:07:06 | non-streaming proxy `httpx.ReadTimeout` after 120s; slot marked unhealthy; reprobe returns 200 and restores it; client gets 503 | manager.err:275–340 |
| 23:09:03, 23:12:36 | pairs of 200s ~200ms apart — two clients interleaving | manager.err:341–344 |
| 23:14:37.790 | second non-streaming `ReadTimeout` | manager.err:345 |
| 23:14:37.802 | `slot=main swapping to Qwen3.8-27B-UD-Q6_K` — the model already loaded | manager.err:410 |
| 23:14:37 | `sudo systemctl restart llama-server.service`; an in-flight **stream** dies with `RemoteProtocolError` | journal; manager.err:414–502 |
| 23:15:07.876 | old llama-server stopped after 30.06s; new one starts, model loaded ~11s later | journal; llama-server.err tail |
| 23:15:07 | reprobe fails (server still starting). **No further manager log output for 13 min** — no `restart succeeded`, no `waiting for .../health` | manager.err:503–504 |
| 23:14–23:27 | clients get 503 every ~5s (`Retry-After: 5`) | journal |
| 23:27:57 | operator `GET /status`: both slots probe 200 | manager.err:504–505 |
| 23:46:02 | operator stops services; manager ignores SIGTERM, SIGKILLed at 23:47:32 | journal |

## Causal chain

1. **Streams are not serialized (F5).** For a streaming item, `_queue_consumer`
   builds a `StreamingResponse` and sets the event immediately
   (`app.py:363–376, 394`), then dequeues the next item while the stream is
   still generating. With two clients (opencode + PAL), a non-streaming request
   reaches llama-server (`--parallel 1`) while a long stream occupies its only
   slot, so it waits inside llama-server.
2. **120s non-streaming timeout (F4)** (`app.py:379`) counts that wait plus
   prefill of 22K–64K-token prompts (~500 tok/s → up to ~130s). It fires.
   *Inferred from timing; consistent with llama-server.err showing a task
   cancelled while another was in prefill.*
3. **Timeout ⇒ unhealthy** (`app.py:388–392`): a client-side timeout on a
   healthy, busy backend calls `mark_unhealthy()`.
4. **Unhealthy ⇒ restart the same model (F1).** The next item calls
   `ensure_model_on_slot`; the fast path `if slot.healthy and same_model(...)`
   (`app.py:247`) fails on `healthy` alone, so it "swaps" to the loaded model,
   restarting llama-server and killing the in-flight stream.
5. **Restart raises (F3).** `subprocess.run(..., timeout=30)` (`swap.py:84–89`)
   vs a 30.06s stop. The missing `restart succeeded` log line confirms
   `restart_llama_server` raised; `TimeoutExpired` is inferred from the 30.06s
   timing.
6. **Consumer dies (F2).** `ensure_model_on_slot` is called outside any
   try/except (`app.py:354`). The exception ends the `_queue_consumer` task;
   the task sits in `consumer_tasks`, so the exception is never logged. The
   item being processed never gets `event.set()`; its handler waits forever.
7. **No self-heal (F6).** The slot stays `healthy=False` with `loaded_model`
   set, so `_validate_and_route` returns 503 (`app.py:466–471`) without
   reprobing. Only `/status` and the 409 path reprobe.
8. After `/status` restored `healthy=True`, any newly enqueued request would sit
   in the queue with no consumer. *Not observed: no client requests were
   logged after 23:27:57.*
9. Hung `event.wait()` handlers kept uvicorn waiting for connections to close
   → SIGKILL.

## Findings

| # | Defect | Where | Role |
|---|---|---|---|
| F1 | unhealthy + same model ⇒ full restart instead of probe | `app.py:247` | turned a timeout into a restart |
| F2 | queue consumer has no exception guard around `ensure_model_on_slot`; death is silent and permanent | `app.py:354` | turned a restart into a deadlock |
| F3 | `systemctl restart` timeout 30s ≈ observed stop time under load | `swap.py:88` | raised the fatal exception |
| F4 | fixed 120s non-streaming timeout covers queueing inside llama-server + long prefill | `app.py:379` | trigger |
| F5 | streaming items release the serial queue before the stream ends; "serial GPU access" (`queue.py` docstring) does not hold | `app.py:363–394` | why two clients collided |
| F6 | 503-unhealthy routing path never reprobes | `app.py:466–471` | prevented recovery |
| F7 | `/var/log/llama/manager.log` permission denied; file logging silently off | manager.err:508 | observability |
| F8 | collection indexing failed at the 23:47 startup — cause not yet examined | manager.err:549 | unrelated, open |
