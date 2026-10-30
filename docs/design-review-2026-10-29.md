# Design review: decisions

A `/grill-me` design review held 2026-10-29 and 2026-10-30, starting from [`design-review-brief.md`](design-review-brief.md). It records what was decided and why, for anyone writing the implementation plans. The code and host facts below were checked during the review against `fb55b33`, the commit that was deployed at the time. Line numbers will drift, so re-read the code before relying on one.

Decision numbers (Q1–Q42) are the review's question numbers, kept so later plans can cite them.

## Purpose and trust boundary

- **Q1: a home inference hub for one owner.** It serves your own clients (PAL, PARE, opencode). It isn't multi-tenant, and it won't be exposed to the internet.
- **Q3/Q9, revised: hot swapping stays a requirement.** Swaps are rare. Onboarding a model is made explicit (see Profiles).
- **Q7: the traffic from `192.168.1.32` was audit and bug-fix testing.** That host (Raspberry Pi vendor prefix, `ubuntu.localdomain`) is the machine the owner SSHes in from. `192.168.1.73` is the owner's Latitude 7420. Neither is a client that needs supporting.
- **Q8: mutating endpoints require `Content-Type: application/json`.** This covers `/swap` and `/collections/*/reindex`, and closes X1, the cross-site `text/plain` POST.
- **Q14: the manager listens on `127.0.0.1` and the Tailscale IP, not the LAN.** Tailscale device identity replaces "anything on the Wi-Fi" as the boundary.
- **Q15: an optional shared bearer token** is checked on every endpoint except `/health` whenever `MANAGER_TOKEN` is set. The reason: `GET /collections/{id}/docs/{doc_id}` returns whole private vault documents (`manager/app.py:1196-1210`).
- **Q16: the rebind rolls out without a cutover.** The order:
  1. The manager listens on the LAN IP, localhost and Tailscale at once.
  2. Clients move one at a time: PAL's unit env, a new PARE inference-URL env, opencode's `baseURL`, and agent_core's default.
  3. The access log confirms nothing still arrives on `192.168.1.14`.
  4. The LAN bind is dropped.

  PARE's env-var name is unverified: it follows from `load_config(..., agent_name="pare")`, and has to be confirmed before relying on it.

## Clients

- **Q17: agent_core stops swapping automatically on `model_not_loaded`.** Swaps happen only through a `/model` command.
  - **A bug that exists today, not introduced by these changes:** `agent_core/inference.py:218` treats only 200 as a successful swap, but the manager answers a swap it actually starts with 202 (`manager/app.py:1345-1348`). So agent_core starts a background swap, then fails the request with the original 409 anyway.
- **Q18: agent_core is the only client of the manager's API.** That API covers the `/status` shape, error types, swap semantics and the token.
  - PAL deletes its own `/model` and `/status` code (`PAL/pal/agent.py:612-689`). That code hardcodes `main`/`batch` and can't reach the second slot.
  - PAL moves to the current agent_core. Its pyproject pins `v1.4.0`, its venv runs 1.6.2, and PARE runs 1.12.1.
  - This repo's README documents the contract.

## Slots

- **Q41: `re` is renamed `egpu`** everywhere: slot name, `SLOT_*` keys, unit (`llama-server-egpu.service`), env file, log files, sudoers, docs, tests and the `/status` key. It happens in phase 1, before profiles exist, because profile sections are keyed by slot name. No client code depends on the name `re`.
- **Q10: a slot is busy until its response completes, streams included.** A client disconnect frees it early.
  - Today the consumer hands off a stream (`app.py:571`), signals done (`app.py:616`) and takes the next item while the stream is still running.
  - Each backend runs `--parallel 1` (`systemd/llama-server.service:96`), so the next request waits inside llama-server, where the manager can't see it.
  - This makes the README's "one request at a time" claim (W2) true.
- **Q19: the slot's consumer is the only code that changes its state.** States are explicit: `idle → serving → idle`, `idle → swapping → idle | failed`, and `any → unhealthy → probing → idle`. Probes and the supervisor send it events instead of writing flags. This covers O4 and finding Q1.
- **Q11: a chat request never triggers a swap.** A request for an unloaded model gets an error naming the slot it can load on.
- **Q12: `/swap` is a FIFO item in the slot's queue.** It runs after everything already queued and before everything queued after it. That fixes A1 by construction. There's no `force` flag.
- **Q20: a failed swap falls back once to the last known-good profile** and fails the swap item with the reason. If the fallback also fails, the slot goes `failed` and every queued item gets an immediate 503.
- **Q21: a per-slot supervisor** probes the backend while the slot is `idle` or `unhealthy`. After repeated probe failures it restarts the unit, with capped backoff. It never restarts a slot that's `serving` or `swapping`, and its restart count goes in `/status`. Unverified: whether llama-server's `/health` stays responsive during generation with `--parallel 1`. Measure that before relaxing the idle-only rule.
- **Q26: a request whose client has gone is skipped when it's dequeued** (A3).

## Profiles (per-model configuration)

The motivating bug is `7b8ea97`. Qwen3.8's `--spec-type draft-mtp` carried over into a gemma swap, and llama-server crash-looped for the full `SWAP_TIMEOUT`. That fix only covers `SAMPLING_ARGS`. The same class of bug is still reachable:

- `CTX_SIZE` and `N_GPU_LAYERS` are set per slot (`/etc/llama/llama-server.env`).
- `--ctx-checkpoints`, `--cache-ram` and `--jinja` are hardcoded in `ExecStart`.
- A missing sidecar silently uses llama.cpp's defaults (`manager/swap.py:31-33`).
- The env file is regex-patched in place, non-atomically (`swap.py:44-69`).

The decisions:

- **Q13: a model is onboarded only when it has a profile.** The roster is derived from the profiles, which replace both the `.args` sidecars and `SLOT_<NAME>_MODELS`. `/v1/models` lists only onboarded models and marks which are loaded. Today it lists every GGUF on disk (`app.py:1000-1012`), including a 0-byte file.
- **Q29: the slot says where a model runs** (host, port, device UUID, unit, queue limit, and workarounds for this llama.cpp build on this GPU such as `GGML_CUDA_DISABLE_GRAPHS`). **The profile says what runs and how**: GGUF path, context size, GPU layers, launch flags, speculative-decoding flags, sampling, and per-model env vars. `ExecStart` holds no flags that depend on the model.
- **Q30: one profile file per model, with a section per slot it may run on.** Each section is verified on its own. `/swap` accepts only verified (model, slot) pairs, and the home slot is the default target.
- **Q31: profiles live in the repo** (`config/models/<stem>.toml`) and deploy to `/etc/llama/models.d/`, where `deploy-manager.sh --check` sha-checks them. GGUFs stay in `/opt/llama/models/`. Measurement notes such as the MTP tok/s table move into the profiles.
- **Q32: every swap regenerates the slot's model env file in full**, writing it atomically (temp file plus rename) into a directory the manager owns. The unit loads it as a second `EnvironmentFile`. Nothing carries over from the previous model.
- **Q33: context size, GPU layers and the full sampling set are required.** A missing key is a validation error at onboarding and at manager startup. Any flag a profile doesn't mention is absent, never inherited.
- **Q25: `onboard <model>` verifies a profile with a trial load.** The steps:
  1. Validate the profile.
  2. Queue a swap to the home slot.
  3. Send one short completion.
  4. Queue a swap back.
  5. Mark the profile verified, with the date and the llama.cpp build.

  There's no VRAM estimator: attention shapes give an 8× spread in KV-cache size, which makes estimates unreliable.
- **Q34: bootstrap.** The two live models get profiles written from today's live env files and sidecars, then run through `onboard` once each.
- **Q42: `gemma-4-26b-a4b-it-q4_k_m` gets `egpu` as its home slot.** It's the regression case for "no flag carried over from the previous model", because it has no MTP layer.

## API surface and retrieval

- **Q24: `/v1/messages` and `/v1/messages/count_tokens` stay.** They're passthroughs to llama-server's native Anthropic API (`7296588`), held to the same queue, token and state rules as chat completions.
- **`/v1/embeddings` is removed.** Nothing calls it: collections call `:8082` directly.
- **Q4: retrieval stays in this server.**
- **Q28: collections routes and indexing orchestration move out of `app.py`** into their own router module with their own startup hook. It's a pure move in its own PR, with no behaviour change.
- **Q23: document ids become `(collection_id, id)`**, with a one-time migration of `skills.db`. Today `documents.id` is a global primary key (`manager/vectordb.py:43-44`).
- **Checked, no action needed:** D3's assumption holds. A sample embedding from `:8082` is unit-norm (768 dimensions, norm 1.0000), so the L2-to-cosine conversion at `vectordb.py:145-148` is valid.

## Observability (Q27)

- Logs go to the journal only.
- One structured line per request: request id, client, slot, model, stream, queue wait, backend time, tokens, outcome.
- `/status` reports each slot's state (Q19), the age of the request in flight, the oldest queue wait, the last error and when it happened, and the supervisor's restart count.

## Housekeeping

- **Q6: the batch slot code and config are removed**: `BATCH_*` keys, code in `config.py`, `slot_config.py` and `app.py`, and `/etc/llama/llama-server-batch.env` (D5).
- **Q36: release hygiene.**
  - `deploy-manager.sh`: `set -e`, a check on every `install`, refusal of a dirty tree, a suite run, and a longer health wait (S4–S6).
  - A lockfile generated from the deployed venv (S7).
  - Hermetic tests (finding Q13).
  - A `doctor.sh` host check (finding Q12).
- **Q37: other cleanups.**
  - Delete `ARCHITECTURE.md` (W1).
  - Fix the Status section of `findings.md`: I1 and I2 are fixed, and so are S8 and S9 on the host.
  - Remove the `-wt-nslot` worktree, which has no commits that aren't on `main`.
  - Delete `config/manager.env.orig`.
- **Q35 and Q39: GGUFs to delete:**
  - the 0-byte `gemma-3-4b-it-q4_k_m`
  - both `Qwen3.5-35B-A3B` variants
  - `gemma-4-E4B-it`
  - `Devstral-Small-2-24B`
  - `Qwen3-Coder-30B-A3B`
  - `CYBER-FROST-3.8` (80 GB, too big for either GPU)

  The models kept are Qwen3.8-27B, Qwen3.6-35B-A3B, gemma-4-26b-a4b and nomic-embed.

## Phases and process (Q38, Q40)

| # | Phase | Review weight |
|---|---|---|
| 1 | Hygiene and cuts: Q36, Q37, the batch and `/v1/embeddings` removal, JSON required on mutating endpoints, `re` → `egpu` | Light: the work is mechanical and failures are loud |
| 2 | Retrieval split, then the composite-key migration (back up `skills.db` first) | Medium |
| 3 | Profiles and `onboard` | Medium-heavy: this is the crash-loop bug class. Regression tests must fail against today's `swap.py`. |
| 4 | Observability: before phase 5, so the riskiest change is visible while it beds in | Light to medium |
| 5 | Slot state machine: Q10, Q12, Q19–Q21, Q26 | Heaviest: concurrency. The plan gives interfaces and invariants, not code. Review is adversarial against concrete interleavings, at 4 findings or fewer per round. |
| 6 | Network and clients: agent_core token, auto-swap removed and localhost default; PAL migrates; the three-way listen, then the LAN dropped | Medium: work in three repos |

Qwen implements and Claude reviews in every phase, one task per prompt. In phases 3 and 5:

- Claude writes the discriminating tests first and checks that each fails against the code before the fix.
- Each phase 5 task covers one state transition.
- Work happens in a worktree.
- Deploys happen outside Qwen's own sessions, because Qwen runs on the `main` slot it's modifying.
- Claude takes over any task that fails review twice.

## Out of scope

These go to separate reviews:

- **The vault**: its shape, who owns it, and filtering PAL's web captures. That's a PAL review.
- **After this plan**: benchmark the onboarded models while tuning their profiles, and enable vision on models that support it.
