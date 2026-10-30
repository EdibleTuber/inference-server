# Audit findings

## Status

- **Fixed in round 1** (PR #14, merge `a183d14`, deployed 2026-10-28): F1, F2, F3, F6, O1, O2, O3, A6, W3, W5.
- **Fixed on the host**: S8 (`_llama-mgr` added to group `llama`), S9 (`re` env file ownership and sudoers entry); checked on the host 2026-10-30.
- **Fixed in round 2** (PR #15, merge `2adbc71`, deployed 2026-10-28): S1, S2, A2, A4, F4, and S3 (background indexing), which stayed blocked in production on I1.
- **Fixed in PR #16** (merge `fb55b33`, deployed and verified live 2026-10-29): I1 (missing `file_path` index), I2 (orphaned `notes` collection hiding vault files). With I1 fixed, S3's background indexing is **effective in production as of PR #16**.
- **Fixed in phase 1** of the design review: X1, S4–S7, W1, D2, D3 and Q11–Q13 (this audit's numbering) — D2's `re` env file is now `config/llama-server-egpu.env` with a current header, and D3's unit-norm assumption is checked (a sample `:8082` vector has norm 1.0000) — plus the batch-slot removal and the repo side of the `re` → `egpu` rename. Plan: `docs/superpowers/plans/2026-10-30-phase1-hygiene-and-cuts.md`.
- **Fixed when the phase 1 host cutover runs** (plan task 10; not yet run): D5 (the deployed `/etc/llama/llama-server-batch.env`, the batch slot's last host leftover) and the host side of the `re` → `egpu` rename (unit, env file, `SLOTS`, sudoers).
- **Still open:** A1, A3, A5, D1, D4, F5, W2, W4, O4, and findings Q1–Q10, Q14. What comes next, and in which phase, is `docs/design-review-2026-10-29.md` and its phase table — note its Q numbers are the review's, not this audit's.

Inputs: Qwen inventory passes p1–p6 (`docs/audit/qwen/out/`), each lead then
read in source. Code cited is branch HEAD `2fbefeb`; deployed is `7b8ea97`,
identical in every cited region except where noted. Incident findings F1–F8
live in `docs/incidents/2026-10-27-main-slot-lockup.md` and are not repeated.

All findings are **pre-existing** unless marked otherwise.

## Verified in source

| # | Finding | Where | Severity |
|---|---|---|---|
| A1 | Proxying runs outside `swap_lock`: the lock covers only `ensure_model_on_slot` (`app.py:246–267`), and the proxy runs after it (`app.py:361–392`). Any swap — `POST /swap` from another client, or the F1 self-restart — restarts the backend under in-flight requests, streaming or not. The incident's killed stream (manager.err:414–502) is one instance. | `app.py:246, 354–392` | High |
| A2 | Streaming responses ignore the backend's status: `client.stream(...)` status is never read and `StreamingResponse` defaults to 200 (`app.py:366–376`). A llama-server 4xx/5xx (e.g. prompt exceeds context, or 503 while loading) reaches the client as `200 text/event-stream` with a JSON error body. | `app.py:366–376` | High |
| A3 | No client-disconnect handling anywhere (`grep is_disconnected` finds nothing). A request whose client gave up still waits in the queue and then runs on the GPU in full, and `_route_and_wait` waits on `event.wait()` with no timeout (`app.py:504`). | `app.py:476–511` | Medium |
| A4 | Fire-and-forget tasks keep no reference: `create_task(_reprobe_for(slot))` (`app.py:382, 392`) and `create_task(self._run_job(...))` (`reindex_jobs.py:120`). asyncio holds tasks weakly, so they can be collected mid-run, and an exception in a reprobe is never observed. `_run_job` catches its own errors; the reprobes do not. | `app.py:382, 392`; `reindex_jobs.py:120` | Low |
| A5 | `count_tokens` and the `/v1/embeddings` proxy make httpx calls with no `except` (`app.py:768–773, 796–802`). A backend that is down or restarting gives an unshaped 500, not the API's error format. | `app.py:768, 796` | Low |
| A6 | 2fbefeb (merged to `origin/main` as PR #13, **not yet deployed**; deployed is 7b8ea97): `POST /swap` now runs `_do_swap` as a `BackgroundTask` (`app.py:1032–1038`) whose result nobody checks. With F3 unfixed, a `TimeoutExpired` there surfaces only as an ASGI traceback after the 202 response has gone out. Introduced by that change, on top of F3; it ships with the next deploy. | `app.py:1032–1042` (origin/main) | Medium, before next deploy |

## Ordering walk: slot state

Writers of slot state: `probe` (`healthy`, `loaded_model`; under `swap_lock` on
every path except the startup probe, which runs before requests are served),
`mark_swapped` (under the lock), `mark_unhealthy` (the consumer, **outside** the
lock, `app.py:381, 391`). Readers: routing in `_validate_and_route` (no lock) and
`ensure_model_on_slot` (under the lock).

Reproduced = run against the real `ServerState`/`_queue_consumer` with only
`swap_to` stubbed (scratch script, not a committed test). Walked = followed
through source only.

| # | Finding | Evidence | Severity |
|---|---|---|---|
| O1 | **Stale routing swaps the slot back.** Routing reads `loaded_model` with no lock, then the item waits in the queue. If another client swaps the slot X→Y in that window, the consumer calls `ensure_model_on_slot(main, X)` (`app.py:351–354`), which swaps back to X. That contradicts "Implicit swaps live only on POST /swap" (`app.py:448–450`). Interleaved clients on two models thrash the slot, each swap taking ~48s and killing whatever is in flight (A1). | Reproduced: `swap_to` calls `['model-y', 'model-x']` | High |
| O2 | **Consumer death is permanent and silent** (F2). After `swap_to` raises `TimeoutExpired`, the task ends; the item being processed and every later item are never answered, and nothing logs it. | Reproduced: task done with `TimeoutExpired`; both items unanswered; queue depth 1 | Critical |
| O3 | **Unhealthy + same model ⇒ restart** (F1). | Reproduced: `swap_to(['model-x'])` with `model-x` already loaded | Critical |
| O4 | **A loading slot gets restarted by its own clients.** At startup, or after any restart, the probe gets 503 from a loading llama-server, so `healthy=False` and `loaded_model` stays as it was (`None` at startup) (`slots.py:84–87`). Requests get a 409 telling them to `POST /swap` (`app.py:455–463`); `agent_core` clients do that (`app.py:1002–1003`), and by O3 the swap restarts the server mid-load. 23:47:32 shows this state: main probe 503 at startup. | Walked; O3 reproduced | High |
| O5 | **Shutdown cannot complete while a handler is stuck.** uvicorn drains connections before lifespan shutdown; handlers stuck in `event.wait()` (O2/A3) never finish, and the unit sets no `--timeout-graceful-shutdown` (`systemd/llama-manager.service:36–41`), so stop waits the full 90s and SIGKILLs. | Observed 23:46:02–23:47:32 | Medium |

Checked and sound: `queue_event` has no lost wakeup (the consumer `clear()`s
before draining, so a `set()` during processing re-runs the loop). The `/status`
and 409-path reprobes skip a locked slot rather than wait, so they can't stall
behind a swap. `_reprobe_for` waits for the lock, so it can't overwrite
`mark_swapped`. The consumer's unlocked `mark_unhealthy` during a swap is
benign: `mark_swapped` and the queued reprobe both run after it.

## QoL and stability improvements

Not bugs, but each would have shortened or prevented the incident.

| # | Improvement | Grounding |
|---|---|---|
| Q1 | Replace `healthy: bool` with a slot state: `loading / ready / busy / swapping / down`. llama-server's 503 from `/health` means loading, not down. Routing, 409 messages and `/status` can then tell the truth, and O4 disappears. | `slots.py:41`; O4 |
| Q2 | Supervisor: a background loop that reprobes non-ready slots every few seconds and restarts a dead consumer task with an error log. It removes the whole "stuck until someone hits /status" class (F2, F6). | incident 23:15–23:27 |
| Q3 | One structured log line per request: request id, client, model, stream, queue wait, backend time, prompt/gen tokens (llama-server returns `usage`/`timings`), outcome. The incident needed three logs correlated by timestamp to see this. | manager.err vs journal vs llama-server.err |
| Q4 | Fix logging: `manager.log` is unwritable (F7), so Python logs land only in `manager.err` and the journal shows only access lines. Log to one place, ideally the journal. | manager.err:508 |
| Q5 | Bound shutdown with `--timeout-graceful-shutdown`, and cancel waiting handlers on shutdown. | O5 |
| Q6 | `/status` additions: consumer alive, in-flight request age and model, queue wait of the oldest item, last error and when. Every one of these was unknowable during the incident. | `slots.py:164–175` |
| Q7 | Swap policy: wait for or refuse in-flight work before restarting (409 `busy` unless `force`). Fail queued items whose model no longer matches with 409 instead of swapping back. | A1, O1 |
| Q8 | Move the hardcoded timeouts (proxy 120s, restart 30s, probe 3s, count_tokens 30s) into config, next to `SWAP_TIMEOUT`. | `app.py:379, 772`; `swap.py:88`; `slots.py:78` |
| Q9 | State-aware `Retry-After`: it is always 5 (`app.py:525, 544`), so clients retried in lockstep every 5s for 13 minutes. Report the expected wait (swap or load in progress) or back off. | journal 23:14–23:27 |
| Q10 | Index collections in a background task after startup, and skip and log a bad document instead of aborting the run. A manager restart then costs about a second, not the length of an index pass. | S1, S3 |
| Q11 | Pin dependencies with a lockfile generated from the deployed venv, and have `deploy-manager.sh` run the test suite and refuse a dirty tree. | S5–S7 |
| Q12 | A `scripts/doctor.sh` that checks the host against the documented install: group memberships, env-file ownership, sudoers entries per slot unit, log-dir writability, units vs repo. S8 and S9 would have shown up immediately. | S8, S9 |
| Q13 | Hermetic tests: give the test config unused ports (or an httpx mock transport), so the suite never depends on what is running on the host. | Tests |
| Q14 | Require `Content-Type: application/json` on mutating endpoints, and optionally a shared token for `/swap` and `/reindex`. Cheap, and it closes X1 without changing clients that already send JSON. | X1 |

## Config and docs drift

| # | Finding | Severity |
|---|---|---|
| D1 | Runtime `SAMPLING_ARGS` come from the model sidecars (`/opt/llama/models/*.args`), which include `--n-predict 32768`; the repo env files don't. A fresh deploy from the repo runs without the cap until the first swap rewrites the file. | Low |
| D2 | `config/llama-server-re.env` header still says "NOT YET DEPLOYED … leaves `re` out of SLOTS" and "PARE pins gemma-4-26b"; `re` is deployed, in `SLOTS`, and runs Qwen3.6-35B. | Low |
| D3 | `vectordb.py:6` says "cosine distance"; the code comment (`vectordb.py:142–145`) and the formula (`:148`) say it is L2, converted on the assumption that embeddings are unit-normalized. That assumption is **unverified** against the embeddings server. | Low; verify assumption |
| D4 | `config.py:36` docstring says `llama_server_host` is "always localhost"; it is env-overridable (`config.py:99`). | Cosmetic |
| D5 | `/etc/llama/llama-server-batch.env` is still deployed for the decommissioned batch slot. | Cosmetic |

## Startup, indexing, deploy and host setup

| # | Finding | Evidence | Severity |
|---|---|---|---|
| S1 | **F8 root cause:** `derive_tags` iterates `frontmatter["tags"]` without a type check (`collections.py:96–98`). `vault/templates/note-template.md` has an empty `tags:`, which YAML parses as `None`, so `TypeError` is raised. One bad file aborts that collection and every later one: `learnings` and `relationships` are never indexed at startup. It is the only file in the workspace with a non-list `tags`. | manager.err:549–560; scan of 78 tagged files | Medium |
| S2 | The F8 log says "endpoints will return 503", but `server.db` and `embeddings_client` are assigned before indexing (`app.py:610–612`), so the endpoints serve a stale, partial index with no error. | `app.py:609–618` | Low |
| S3 | **Indexing blocks startup.** It runs inside `lifespan` before uvicorn listens, so every manager restart makes inference unavailable for the whole index pass: 36s at 23:47:32 → "Application startup complete" at 23:48:08. | manager.err:509–560 | Medium |
| S4 | `deploy-manager.sh` verifies `/health` for 15s, which is shorter than S3's startup time, so a successful deploy can report failure and exit 1. | `deploy-manager.sh:113–124` | Low |
| S5 | `deploy-manager.sh` stamps `HEAD` as deployed even with uncommitted changes under `manager/`, so the provenance stamp can lie. It also prints "(0 modified…)" when clean, because `${DIRTY:+…}` tests non-empty, not non-zero. | `deploy-manager.sh:56–58, 100` | Medium |
| S6 | `deploy-manager.sh` uses `set -uo pipefail` without `-e` and never checks `install`, so a failed copy still stamps and restarts. | `deploy-manager.sh:18, 90–91` | Medium |
| S7 | `requirements.txt` has only lower bounds. A fresh venv today resolves Starlette 1.10 / FastAPI 0.162 vs deployed 1.0 / 0.135, so `setup.sh` on a new host gets untested versions. | pip resolution in a scratch venv | Medium |
| S8 | **F7 root cause:** `setup.sh` adds `_llama` to group `llama` but not `_llama-mgr` (`setup.sh:51–52`), and `/var/log/llama` is `root:llama 775`, so the manager cannot create `manager.log`. | `id _llama-mgr` | Low |
| S9 | **The `re` slot cannot swap and would kill its own consumer.** `/etc/llama/llama-server-re.env` is `root:root 644`; the README's `chown _llama-mgr` step (README:589) was skipped on this host. Any swap of `re` (including F1's self-restart after one proxy timeout) raises `PermissionError` in `update_env_file` (`swap.py:71`), which is unhandled, so O2 applies: the consumer dies. Even with the env file fixed, the restart would fail: `/etc/sudoers.d/llama-manager` permits `restart llama-server.service` and `llama-server-batch.service` (decommissioned) only, so `sudo systemctl restart llama-server-re.service` is refused, `restart_llama_server` raises `RuntimeError` (`swap.py:92–95`), and the result is the same consumer death. | `ls -l /etc/llama`; sudoers read 2026-10-28; O2 reproduced | High |

## Found after the round-2 deploy (2026-10-28)

| # | Finding | Evidence | Severity |
|---|---|---|---|
| I1 | **No index on `documents.file_path`.** `get_hash` (`vectordb.py:172-177`) is a full table scan, and `index_collection` calls it once per file synchronously, never awaiting for an unchanged file. 2,937 lookups take **30.3s** and block the event loop, so S3's background indexing did not shorten startup in production (startup complete about 33s after start, as before). Any full reindex while serving freezes every endpoint, chat proxying included, for about 30s. Tests used two-file directories, so they could not show it. With `CREATE INDEX ... ON documents(file_path)` on a copy of the live DB: **0.041s**. | timed on a copy of `/opt/llama/data/skills.db` | High |
| I2 | **695 vault files (24%) are invisible to vault search.** An orphaned collection `notes` (not in `collections.json`, same `source_dir`, 1,138 rows) holds them. `get_hash` is not scoped to a collection, so the vault pass sees them as "unchanged" and never indexes them under `vault`. Document ids are global (`id TEXT PRIMARY KEY`, the relative path), so two collections over one directory cannot coexist anyway. Once `get_hash` is scoped, `upsert_document`'s `ON CONFLICT(id)` moves each row to `vault`, which needs about 695 re-embeds. Pre-existing. | DB copy: vault 2,242 rows vs 2,937 files; 695 present only under `notes` | High |

Deploy note: `DEPLOYED_FROM` reads `676737a` (the branch tip) rather than merge `2adbc71`. The trees are identical and every file's sha256 matches the merge, so it is provenance-only. Deploy from `main` next time.

## Tests

229 pass in 18s; 93% line coverage with branches. The ~50 untested lines
include exactly the incident path:

- `app.py:381–382` (backend 5xx → unhealthy + reprobe) and `app.py:388–392`
  (proxy exception → unhealthy + reprobe) are not covered, and no test makes
  `ensure_model_on_slot` raise inside the consumer. A high percentage hides
  that the failure paths are the untested ones.
- `app.py:617–618` (indexing failure) is not covered (S1).
- **Not hermetic:** the suite connects to real ports, 31× to `127.0.0.1:8081`
  (the live main llama-server) and 30× to `:8083`. All are `GET /v1/models`
  probes from lifespan startup and `/status`, so they are harmless. Results were
  identical with the ports blocked and open, so nothing depends on it today. No
  test reached `sudo`. Measured with a connect-blocking pytest plugin and a fake
  `sudo` on `PATH`.

## Streaming and disconnects

Sound. uvicorn 0.42 reports ASGI spec 2.3, so Starlette 1.0's
`StreamingResponse` runs `listen_for_disconnect` alongside the stream and
cancels it on `http.disconnect`. Cancellation unwinds `_stream_gen`'s
`async with client.stream(...)`, closing the backend connection; llama-server
then cancels its task (`srv stop: cancel task` in llama-server.err). This
narrows A3 to **non-streaming** requests and to waiting in the queue.

Re-verified 2026-10-28 against a real socket (`tests/test_stream_disconnect_real.py`
and a scratch probe), because a mocked stream misleads here. A mock whose close
suspends predicts a leak on mid-stream disconnect, but with real httpx/httpcore
the backend connection closes at once on every version tested: when a read is
cancelled, httpcore closes the connection itself. The one path where the
manager's own close is the only closer is a client that leaves before the first
chunk. Round 2's relay covers that with `StreamingResponse(background=...)`, and
the real-socket test fails if that close is removed.

## Docs vs code

| # | Finding | Severity |
|---|---|---|
| W1 | `ARCHITECTURE.md` describes a different system: macOS, launchd, JWT auth proxy, MLX, "all services bind to 127.0.0.1". None of it is this repo. | Medium (misleading on security) |
| W2 | README:50 says each slot "runs one request at a time" and requests are "processed in order"; F5 shows streams are not serialized. | Medium |
| W3 | README:178/409 says "Chat requests never trigger a swap"; O1 and O3 reproduce chat requests triggering swaps. | Medium |
| W4 | README:405 says `healthy` "is not updated while a swap is in progress"; the consumer's `mark_unhealthy` can run mid-swap (benign, but the claim is false). | Cosmetic |
| W5 | 2fbefeb (merged, PR #13) makes `POST /swap` return 202 but leaves README:178, 187, 389 and 413 saying it "returns once the slot is healthy" / "503 if the swap fails". Part of A6. | Medium, before next deploy |

## Security

| # | Finding | Severity |
|---|---|---|
| X1 | The manager binds the LAN IP with no auth (documented: "LAN-only is the trust boundary"). JSON endpoints parse the body regardless of `Content-Type`, so a cross-site `text/plain` POST needs no CORS preflight. **Reproduced:** a `text/plain` body to `POST /swap` returned 202 and called `swap_to`. Any web page opened on a LAN machine can restart backends (or queue GPU work), subject to whatever Private Network Access the browser enforces. | Medium |

## Ruled out

- `queue.py:51–52` `put` then `_depth += 1` looked like a race; `asyncio.Queue.put` on an unbounded queue never suspends, so nothing can interleave.
- Repo systemd units are byte-identical to `/etc/systemd/system/` copies.

## Inventory quality

- Thinking-off runs: `p2`/app.py found every real `except` (13/13) and task site (4/4), no false positives, and flagged `app.py:354` (F2).
- p5 marked 134 claims "consistent". It checks only within one file, so treat that as no evidence either way. Cross-file claims (README, ARCHITECTURE, docstrings that refer to other modules) are not yet reviewed.
- p6 did not link `SLOT_<NAME>_*` keys to their computed reader in `slot_config.py`; read that by hand.

## Not yet covered

The embeddings unit and `download-model.sh`; logrotate `copytruncate` vs the
appending units; whether embeddings are unit-normalized (D3).
