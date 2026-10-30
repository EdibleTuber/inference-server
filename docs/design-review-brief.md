# Design review brief

Starting context for a `/grill-me` design review of the inference server. It covers what the repo docs don't: the live deployment, who calls the server, and how far each doc can be trusted.

Facts were checked on the host on **2026-10-29**, right after round 3 deployed `fb55b33`. They will go stale. Each section names the command that re-derives it, so run that rather than trusting the number.

## How far each doc can be trusted

| Doc | State |
|---|---|
| `docs/audit/findings.md` | **Current** apart from its Status section, which still lists I1 and I2 as open. Both were fixed by PR #16 (`fb55b33`) and verified live on 2026-10-29. This is the index of known problems: A, F, O, S, W, X, D, Q and I. |
| `docs/incidents/2026-10-27-main-slot-lockup.md` | Current. The RCA behind rounds 1–3, with the causal chain F1–F8. |
| `docs/superpowers/plans/2026-10-28-slot-lockup-round{2,3}.md` | Historical. They record what each round changed and why. |
| `docs/superpowers/specs/*.md` | Design docs from March to October 2026, each written for its own phase. Some may be superseded, so verify against the code. |
| `README.md` | Mostly current. Rounds 1–3 corrected the sections they touched. |
| `ARCHITECTURE.md` | **Wrong system** (finding W1). It describes a macOS "llama-cli" with a TUI, an auth proxy and JWTs, none of which exist here. Ignore it. |

## Live topology

Re-derive with `systemctl list-units --all | grep -E 'llama|pal'`, `ss -ltnp`, `nvidia-smi`, `grep -v '^#' /etc/llama/manager.env`.

| Piece | Where | Notes |
|---|---|---|
| `llama-manager.service` (this repo) | `192.168.1.14:11434`, LAN-bound, no auth | Runs from `/opt/llama/manager/` as `_llama-mgr`. `DEPLOYED_FROM` gives the commit. |
| `main` slot: `llama-server.service` | `127.0.0.1:8081`, GPU 0: Tesla PG500-216 (a 32 GB V100-class card) | `Qwen3.8-27B-UD-Q6_K`, about 30 GB of the GPU in use |
| `re` slot: `llama-server-re.service` | `127.0.0.1:8084`, GPU 1: Tesla P40 (24 GB, eGPU) | `Qwen3.6-35B-A3B-UD-IQ4_NL_XL`. Never swapped by the manager in practice. |
| `llama-embeddings.service` | `127.0.0.1:8082`, CPU | nomic-embed-text, used by collection indexing and search |
| `llama-server-batch.service` (Vulkan iGPU, `:8083`) | **disabled** | Decommissioned, but `BATCH_*` keys remain in `manager.env`, batch code remains in `config.py`, `slot_config.py` and `app.py`, and `/etc/llama/llama-server-batch.env` still exists. |
| Collections DB | `/opt/llama/data/skills.db` (sqlite plus sqlite-vec) | Collections come from `/etc/llama/collections.json`: `skills`, `vault`, `learnings`, `relationships`. `learnings` and `relationships` have 0 documents. |
| `:8080` | `0.0.0.0` | SearXNG (container), not part of this repo |

Slot roster in `manager.env`: `SLOTS=main,re`, with one model allowed per slot (`SLOT_MAIN_MODELS`, `SLOT_RE_MODELS`). `QUEUE_LIMIT=50` for main and 20 for `re`, and `SWAP_TIMEOUT=600`.

Host specifics that the design has to respect:

- **Restart rights.** `/etc/sudoers.d/llama-manager` lets `_llama-mgr` restart `llama-server.service` and `llama-server-re.service` only.
- **File and group access.** `_llama-mgr` is in group `llama`. It owns the slot env files, which it rewrites during a swap.
- **Logs.**
  - `/var/log/llama/manager.err` holds the manager's Python logging.
  - `manager.log` is the manager's own log file.
  - `llama-server*.err` are the backend logs. Their timestamps count from process start.
  - **Access logs go only to the journal** (`journalctl -u llama-manager`), because uvicorn's access logger writes to stdout.
- **Deploy.** `sudo ./scripts/deploy-manager.sh`, run from a clean `main`. `--check` compares each deployed file's sha256 with the repo.

## Who actually calls it

Re-derive with:

```
journalctl -u llama-manager --since -7d -o cat | grep -oE '[0-9.]+:[0-9]+ - "[A-Z]+ [^ ?"]+' | sed -E 's/:[0-9]+ - "/ /' | sort | uniq -c | sort -rn
```

Calls by client and endpoint, 2026-10-24 to 2026-10-29 (the full journal window):

| Client | Calls |
|---|---|
| `192.168.1.14` (this host) | 1,033 `POST /v1/chat/completions`, 6 `/status`, 5 `/collections/vault/search`, 2 `/health`, 1 `/collections` |
| `192.168.1.32` | 79 `POST /v1/chat/completions`, 8 `/status`, **2 `POST /swap`**, 2 `/v1/models` |
| `192.168.1.73` | 1 `/status` |

Known local clients:

- **opencode** (`~/.config/opencode/opencode.json`): provider `local` → `http://192.168.1.14:11434/v1`, with both slot models. A second provider, `flashnext` → `127.0.0.1:8090`, isn't listening now.
- **PAL**: `pal-daemon` (user unit) and `pal-discord`.
  - The daemon uses `PAL_INFERENCE_URL=http://192.168.1.14:11434` and `PAL_MODEL=Qwen3.8-27B-UD-Q6_K`.
  - `pal/agent.py:612` reads `/status`, and `:680` calls `/swap`.
  - It writes to the vault at `/mnt/secondary/agent-workspace/vault`, which is the `vault` collection.
- **Unidentified:** `.32` and `.73`. The owner should say what they are. `.32` performs swaps.

Endpoints, from the `@app.` decorators in `manager/app.py`:

- `/health`, `/status`, `/v1/models`
- `/v1/chat/completions`, `/v1/messages`, `/v1/messages/count_tokens`, `/v1/embeddings`
- `/collections`, `/collections/{id}/search`, `/collections/{id}/docs/{doc_id}`
- `/collections/{id}/reindex`, `/reindex/status`, `/reindex/{job_id}`
- `/swap`

`/v1/messages` (Anthropic-shaped) and `/v1/embeddings` were never called in the journal window.

## Code shape

Re-derive with `wc -l manager/*.py`. The suite should be green: run it with `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/ -q -p no:cacheprovider`.

- `manager/app.py` is about 1,400 of the package's roughly 2,950 lines. It holds:
  - the HTTP routes;
  - the per-slot queue consumer;
  - proxying and stream relay;
  - lifespan startup and teardown;
  - collection indexing orchestration.
- Everything else is small modules: `slots.py` (`SlotState`), `queue.py`, `swap.py` (env rewrite plus `sudo systemctl restart`), `slot_config.py` and `config.py`, `routing.py`, `names.py`, `collections.py`, `vectordb.py`, `reindex_jobs.py`, `embeddings.py`, `gpu.py`.
- Tests must never reach the live ports 8081–8084 or 11434. Earlier rounds checked this with a socket guard; about 61 connection attempts to live ports are expected, all from read-only probes, all blocked.
- Stale worktree: `/mnt/secondary/inference-server-wt-nslot` on `feat/manager-n-slot` (`b9fcd31`, 2026-09-23) has **0 commits not on `main`**. Its work was merged and the worktree can go.

## What is verified about current behaviour (rounds 1–3)

Each item is covered by tests, and was checked live after its deploy.

- **The consumer survives failures.** A failed restart or a timeout fails the request, not the consumer. The consumer never swaps models itself: `ServerState.ready_for`, and `_decision_from_item_error`.
- **Timeouts.** The non-streaming proxy read timeout is `PROXY_READ_TIMEOUT=1800`. A read or write timeout returns 504 `upstream_timeout` and leaves the slot healthy. A connect timeout marks the slot unhealthy. Before returning a 503, the manager probes the slot again.
- **Streams.** The consumer opens the upstream stream itself. An upstream status ≥ 400 comes back as a plain error response. A client disconnect closes the upstream.
- **Startup indexing.** It runs in the background, and "startup complete" now comes about 1s after start (it was about 33s). One bad document or a bad config can't stop startup.
- **Collection lookups.** They are indexed and scoped per collection. Pruning collections no longer configured happens only after every startup pass succeeded, and only when the config was read and isn't empty.

## Open design questions already identified

These are from `findings.md`. Don't treat this list as the agenda: the review should start from what the system is for.

- **A1, F5, W2:** is a slot busy until its response completes? Streams are not serialised per slot.
- **O4, Q1:** there is no explicit slot state machine.
- **Q2:** nothing supervises or restarts a hung backend.
- **Document ids** are global (`documents.id TEXT PRIMARY KEY`, the relative path), so two collections over one directory collide.
- **X1:** `POST /swap` accepts `text/plain`, so a browser on the LAN can trigger it cross-site. There is no auth on `:11434` at all.
- **W1:** `ARCHITECTURE.md` describes the wrong system.
- Not yet scheduled: A3, A5, S4–S7, D1–D5, W4, and Q3–Q14.
- Leftovers of the decommissioned batch slot (see the topology table).

## The vault (open for the review)

`/mnt/secondary/agent-workspace/vault` is the `vault` collection, with private remote `EdibleTuber/vault`.

- **Backup is solved.** PAL commits its own writes (`pal/wiki.py:183-204`) but never pushes. Since 2026-10-29, a user timer, `vault-backup.timer`, runs `~/.local/bin/vault-backup` hourly: it commits anything left uncommitted, then pushes. Check it with `journalctl --user -u vault-backup`.
- **Discord channel `history.jsonl` files are ignored and untracked**, because they duplicate Discord. Channel `scratch.md` files stay tracked.
- **Open questions for the review:**
  - the vault's shape, with project notes coming;
  - who owns it: PAL, the owner, or both;
  - whether PAL's web captures should be filtered. A 2026-10-27 research run saved 22 off-topic pages out of 29, including shopping, dating and sign-in pages.
