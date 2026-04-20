---
title: Phase B Server-Side - Dual-Slot Manager with Vulkan Batch Backend
date: 2026-04-19
status: draft
related:
  - "PAL side (shipped): /home/edible/Projects/PAL/docs/superpowers/specs/2026-04-19-phase-b-dual-backend-batch-model-design.md"
---

# Phase B Server-Side: Dual-Slot Manager with Vulkan Batch Backend

## Context

The inference server currently runs a single llama-server backend on a Tesla P40 (CUDA, port 8081) behind the `llama-manager` FastAPI proxy (port 11434). Chat traffic shares that backend with background jobs PAL fires on every turn (categorize, learning scan, LLM-based PDF TOC detection), which periodically disrupts chat latency and, in one observed case on 2026-04-17, caused `llama-server` to be SIGKILLed under contention.

Phase B adds a second backend running on the host's AMD Cezanne Radeon Vega iGPU via llama.cpp's Vulkan backend, pinned to a small capable model (Gemma 4 E4B IT starting point), and gives the manager a slot-aware routing layer so PAL's background callers land there. Chat continues to own the P40 undisturbed.

The PAL-side integration is already shipped on main (behind `PAL_BATCH_ENABLED=false` by default) — it's a no-op until the batch endpoint exists. This spec covers the server-side work to provide that endpoint.

Current relevant server state:

- `llama-server.service` (CUDA P40, 127.0.0.1:8081)
- `llama-embeddings.service` (CPU, nomic-embed-text, 127.0.0.1:8082)
- `llama-manager.service` (FastAPI, 0.0.0.0:11434) — proxies OpenAI-compatible requests, handles implicit main swaps on incoming-model-name change, serves collection search/reindex endpoints.
- Single-user, LAN-only deployment. No auth. Trust boundary is the network.

## Non-goals

- Replacing or modifying the embeddings service.
- Auth/ACL on any endpoint. LAN boundary is the trust boundary.
- Hot-swap of the llama-server binary (Phase 0 expects a clean rebuild + service restart).
- ROCm support. Phase B standardizes on the Vulkan backend for the iGPU.
- Auto-promotion of unhealthy batch slot (no watchdog that tries to self-heal; admin-triggered swap is the only recovery path).
- Per-slot CTX autotune or device auto-selection. Both are hand-configured.

## Architecture

### Slot model

The manager replaces its flat single-backend state with two logical slots:

| Slot    | Systemd unit                      | Port | Device flag           | Default model               | Swap trigger                 |
|---------|------------------------------------|------|-----------------------|-----------------------------|------------------------------|
| `main`  | `llama-server.service` (existing) | 8081 | `--n-gpu-layers -1`   | Gemma 4 26B-A4B Q4_K_M      | Implicit on unknown-model request + explicit `POST /swap` |
| `batch` | `llama-server-batch.service` (new) | 8083 | `--device Vulkan0`    | Gemma 4 E4B IT Q4_K_M        | Explicit `POST /swap` only   |

Both use `/opt/llama/bin/llama-server`, rebuilt in Phase 0 with CUDA + Vulkan enabled (see Build section).

### Routing

The manager tracks `loaded_model` per slot in-process (a `SlotState` instance per slot). On startup it probes each slot's `/v1/models` once to reconcile. When an inference request arrives:

1. Look up the requested `model` in both slots.
2. If loaded on a slot, route there.
3. If not loaded anywhere, route to main (preserves today's implicit main-swap behavior).
4. If the selected slot is `healthy = False`, return 503 with a typed error (`main_unavailable` or `batch_unavailable`).

The manager never implicitly swaps the batch slot. Any "model not loaded on batch" request either routes to main (if the model is loaded there) or triggers a main swap (if not loaded anywhere). Batch only changes via `POST /swap {target: "batch"}`.

State drift (e.g., external `systemctl restart` of a slot, which might load a different model via its env file) is detected on 5xx from a backend: the manager re-probes that slot's `/v1/models` before marking the slot healthy again and reconciles `loaded_model`.

### Queueing

Each slot has its own FIFO queue (reusing the existing `RequestQueue` class) and its own consumer task. `QUEUE_LIMIT` applies to main (existing 50). `BATCH_QUEUE_LIMIT` applies to batch (default 20, configurable). The two queues are independent — a saturated main queue does not back up batch and vice versa.

### Endpoints

**Existing, modified:**

- `GET /status` — response now includes `slots` section (see Data Shapes).
- `POST /v1/chat/completions` — routes by model name as described above. Returns 503 with typed error for unhealthy target slot instead of 503 with generic server_error.
- `GET /v1/models` — unchanged (returns all GGUFs in `MODELS_DIR`, not per-slot). `/status.slots` is the source of truth for what's currently loaded.

**New:**

- `POST /swap {model, target}` — admin swap endpoint. `target` is `main` or `batch`, defaults to `main` when omitted. `target=main` is equivalent to the existing implicit behavior but usable explicitly; `target=batch` is the only way to change the batch slot. Returns 200 with `{slot, model, status: "ok"}` on success, 404 if model file is missing, 503 if the swap itself failed (systemd error, health timeout).

**Unchanged:**

- `GET /health`
- `/v1/embeddings` proxy
- `/collections/*` (search, docs, reindex)

### Build (Phase 0)

The binary at `/opt/llama/bin/llama-server` must be rebuilt with both CUDA and Vulkan backends enabled. This is a scripted step but not a code change; the plan gates all subsequent tasks on its success.

Prerequisites on the host:
- CUDA toolkit already installed (existing requirement; Tesla P40 is operational).
- Vulkan SDK / loader + headers installed. Distro-specific; for Ubuntu, `vulkan-sdk` or `libvulkan-dev` + `vulkan-headers` + `glslang-tools`.
- `_llama` user in the `video` and `render` groups (iGPU device nodes live at `/dev/dri/*`; access is gated by group membership).

Build invocation (within the llama.cpp source tree):

```
cmake -B build -DGGML_CUDA=ON -DGGML_VULKAN=ON
cmake --build build --config Release -j
sudo install -m 0755 build/bin/llama-server /opt/llama/bin/llama-server
```

Validation before proceeding with the rest of the plan:
- `/opt/llama/bin/llama-server --device list` (or `--list-devices`, depending on llama.cpp version) shows both CUDA0 and Vulkan0.
- As the `_llama` user: `vulkaninfo --summary` lists the Vega iGPU as a Vulkan device.
- One-off load test with Gemma 4 E4B on Vulkan0 (direct `llama-server` invocation, not via systemd) successfully serves `/v1/chat/completions` on a scratch port and returns coherent output. Record tok/s for a realistic prompt as the Vulkan perf baseline.

## Components

### New modules

**`manager/slots.py`** — `SlotState` dataclass encapsulating per-slot state and operations.

```python
@dataclass
class SlotState:
    name: str                           # "main" | "batch"
    host: str
    port: int
    env_file: str                       # path to env file for swap rewrites
    systemd_unit: str                   # e.g. "llama-server-batch.service"
    loaded_model: str | None = None
    healthy: bool = False
    last_swap_utc: str | None = None
    queue: RequestQueue = ...
    swap_lock: asyncio.Lock = ...

    @property
    def url(self) -> str: return f"http://{self.host}:{self.port}"

    async def probe(self, client: httpx.AsyncClient) -> None:
        """Query /v1/models; update loaded_model + healthy accordingly. Never raises."""

    async def reconcile_on_error(self, client: httpx.AsyncClient) -> None:
        """Re-probe after a 5xx from backend; reconcile loaded_model if drifted."""
```

**`manager/routing.py`** — pure function over slot state.

```python
def resolve_slot(model: str, slots: dict[str, SlotState]) -> str:
    """Return 'main' or 'batch' based on what's loaded where.
    Model loaded on main → 'main'. Loaded on batch → 'batch'.
    Loaded on neither → 'main' (implicit main swap preserved)."""
```

Pure, no I/O. Unit-testable in isolation.

### Changed modules

**`manager/app.py`** — `ServerState` holds `slots: dict[str, SlotState]` instead of flat fields (`state`, `current_model`, `loading_model`, `error_message`, `queue`, `swap_lock`). Two consumer tasks, one per slot. `/v1/chat/completions` calls `resolve_slot`, checks slot health, enqueues on slot queue. `POST /swap` handler added. `/status` handler produces the new shape.

**`manager/swap.py`** — `ModelSwapper` gains a `slot: SlotState` parameter (or is instantiated per-slot). `update_env_file` writes to `slot.env_file`. `restart_llama_server` runs `sudo systemctl restart {slot.systemd_unit}`. `wait_for_health` polls `{slot.url}/health`. The `sudoers` entry (currently allows `sudo systemctl restart llama-server.service` for `_llama-mgr`) must be extended to also allow `llama-server-batch.service`.

**`manager/config.py`** — adds:

```python
batch_server_host: str         # "127.0.0.1"
batch_server_port: int         # 8083
batch_server_env: str          # "/etc/llama/llama-server-batch.env"
batch_server_unit: str         # "llama-server-batch.service"
batch_queue_limit: int         # 20
batch_model_default: str       # "gemma-4-E4B-it-Q4_K_M" (informational; no enforcement)
```

All read from `manager.env` with sensible defaults.

### New files (non-code)

**`systemd/llama-server-batch.service`** — parallel to `llama-server.service` with three differences:
- `EnvironmentFile=/etc/llama/llama-server-batch.env`
- `ExecStart` uses `--device ${DEVICE}` instead of `--n-gpu-layers ${N_GPU_LAYERS}`
- `MemoryMax=6G` (bounds iGPU shared-RAM footprint; Gemma 4 E4B Q4 + CTX 16k KV cache fits comfortably in ~3 GB, 6G gives headroom without letting the iGPU starve the host).

**`config/llama-server-batch.env`** — `HOST`, `PORT=8083`, `MODEL_PATH`, `CTX_SIZE=16384`, `DEVICE=Vulkan0`.

**`config/manager.env`** — appended with the new batch-slot vars.

### Files untouched

- `manager/queue.py` (reused as-is for both slots)
- `manager/embeddings.py`, `manager/vectordb.py`, `manager/collections.py`, `manager/reindex_jobs.py`, `manager/gpu.py` (orthogonal)
- `systemd/llama-server.service`, `systemd/llama-manager.service`, `systemd/llama-embeddings.service` (main stays as-is)
- `config/llama-server.env` (main stays as-is)

## Data Shapes

### `GET /status`

```json
{
  "slots": {
    "main": {
      "host": "127.0.0.1",
      "port": 8081,
      "loaded_model": "gemma-4-26b-a4b-it-q4_k_m",
      "healthy": true,
      "last_swap_utc": "2026-04-17T05:37:10+00:00",
      "queue_depth": 0,
      "queue_limit": 50
    },
    "batch": {
      "host": "127.0.0.1",
      "port": 8083,
      "loaded_model": "gemma-4-E4B-it-Q4_K_M",
      "healthy": true,
      "last_swap_utc": "2026-04-19T14:00:00+00:00",
      "queue_depth": 0,
      "queue_limit": 20
    }
  },
  "gpu": { ... },
  "uptime_seconds": 12345
}
```

The top-level `state`, `current_model`, `loading_model`, `error_message`, `queue_depth`, `queue_limit` fields from the existing `/status` are removed. Callers (PAL's `/model`) switch to reading from `slots.main` for equivalent info. This is a breaking change for direct consumers of `/status`, but the only real consumer is PAL which is updated alongside. Note: the existing single-slot state FSM (`loading | ready | swapping | error`) is replaced by a boolean `healthy` plus the presence/absence of `loaded_model`. A slot with `healthy=false, loaded_model=None` is the equivalent of "loading" or "error"; a slot with `healthy=true, loaded_model=<name>` is "ready". Per-slot transient "swapping" state is not exposed (a swap holds the slot's `swap_lock` and blocks the next request until complete — callers don't need a separate state for it).

### `POST /swap`

Request:

```json
{"model": "gemma-4-E4B-it-Q4_K_M", "target": "batch"}
```

`target` optional, defaults to `"main"`.

Response (200):

```json
{"slot": "batch", "model": "gemma-4-E4B-it-Q4_K_M", "status": "ok"}
```

Error (400 invalid target, 404 model file missing, 503 swap failed):

```json
{"error": {"type": "swap_failed", "message": "systemctl restart returned non-zero: ..."}}
```

### `POST /v1/chat/completions` error shape on slot unavailable

```json
{"error": {"type": "batch_unavailable", "message": "batch slot not ready"}}
```

Or `main_unavailable` for main. Status 503, `Retry-After: 5`.

## Build, Deploy, Config

### Rollout order

Phase 0 gates Phases 1+. Do not merge or deploy subsequent tasks until Phase 0 is green.

0. **Build + validate binary** (Phase 0). Rebuild llama-server with CUDA+Vulkan. Validate device enumeration. Manual one-off Vulkan-backed inference against Gemma 4 E4B. Record perf baseline.
1. **Drop Gemma 4 E4B GGUF** in the configured `MODELS_DIR` (on agenthost: `/mnt/secondary/llama-models/gemma-4-E4B-it-Q4_K_M.gguf`). Verify sha256.
2. **Install batch systemd unit and env file**. Start the service. Confirm `curl 127.0.0.1:8083/health` returns 200 and `/v1/models` reports the expected model.
3. **Deploy manager changes** (`slots.py`, `routing.py`, updated `app.py`, `swap.py`, `config.py`, updated `manager.env`, extended sudoers). Restart manager. Confirm `GET /status` shows both slots healthy.
4. **Smoke test from PAL**: `/model` shows both slots. `/model --target batch <name>` swaps batch. `PAL_BATCH_ENABLED=true` on PAL, run a compile, confirm categorizer fires on batch (no P40 activity for that call).
5. **Simulated outage**: `sudo systemctl stop llama-server-batch`. Confirm PAL's `BatchFallbackProposal` UX fires as designed.

### Sudoers extension

Current entry (or equivalent):

```
_llama-mgr ALL=(root) NOPASSWD: /bin/systemctl restart llama-server.service
```

Extend to:

```
_llama-mgr ALL=(root) NOPASSWD: /bin/systemctl restart llama-server.service, /bin/systemctl restart llama-server-batch.service
```

### Model acquisition

Gemma 4 E4B IT Q4_K_M GGUF: pull from HuggingFace (e.g., `bartowski/gemma-3-4b-it-GGUF` or an equivalent reputable GGUF publisher — the exact source is a spec-execution detail, not a design constraint). Verify sha256 against the publisher's listing. Record the source URL and hash in the plan as the task is executed.

## Error Handling

**Build-time (Phase 0):**
- Vulkan SDK missing or CMake can't find headers → install `vulkan-sdk` or distro equivalent. Document in spec's Build section and in the plan.
- Binary builds but `--device Vulkan0` fails at runtime → check `vulkaninfo` as `_llama`; verify `_llama` ∈ `video` + `render` groups.

**Slot startup:**
- Slot unreachable at manager startup → slot marked `healthy = False, loaded_model = None`. Startup proceeds. Inference requests routed to that slot return 503 with typed error. A successful `POST /swap` to that slot clears the error without a manager restart.

**Slot runtime failures:**
- 5xx from backend → response proxied to client as-is; mark slot unhealthy; re-probe asynchronously. If re-probe succeeds with a different `loaded_model`, reconcile and flip to healthy.
- Backend crash (systemd auto-restart fires) → health poll sees it come back up eventually.

**Swap failures:**
- `sudo systemctl restart` non-zero → raise `RuntimeError`. Slot transitions to error. Drain that slot's queue with error (existing `_drain_queue_with_error` pattern, per-slot). Return 503 from `/swap` with the systemd stderr excerpt.
- Health poll timeout → same as today's single-slot behavior, per-slot.
- Model file missing in `MODELS_DIR` → 404 from `/swap` before touching systemd.

**Routing edge cases:**
- Request with a model that's the batch default but batch slot loaded differently → 503 `batch_unavailable`, message hints "call /swap target=batch first".
- Request during a batch swap (swap_lock held) → queue consumer on that slot waits on the lock as today.
- Same model accidentally loaded on both slots → `resolve_slot` returns main (first match). Harmless; underutilizes batch.

## Testing Strategy

### Unit tests

- `manager/routing.py::resolve_slot` — model on main / on batch / on both / on neither / empty model string.
- `manager/slots.py::SlotState` — probe success, probe timeout, probe with unexpected JSON shape. Reconcile after 503 updates `loaded_model` correctly.
- `manager/swap.py::ModelSwapper` — slot-parameterised. Correct env file rewritten for main vs batch, correct unit restarted, health poll targets correct port. Mock `subprocess.run` + `httpx`.
- `manager/app.py::create_app` — `/status` shape has both slots. `POST /swap`: valid main, valid batch, invalid target, missing model field, nonexistent model file. `/v1/chat/completions` routing: main model → main queue, batch model → batch queue, unknown model → main queue.

### Integration tests

- Manager against two fake backend stubs on 8081 and 8083 (each claiming specific models via `/v1/models`). Submit requests targeting each model; assert correct backend receives each, queue depths move correctly.
- Batch backend killed mid-test → 503 with `batch_unavailable`. Main requests continue to succeed.
- `/swap` target=batch → batch stub's env file rewritten (or restart signal logged, depending on stub design).
- State drift: batch stub changes what it returns from `/v1/models` without a `/swap` call. Next request routed to batch triggers re-probe → `loaded_model` reconciled.

### Manual validation (on agenthost)

- Phase 0 gate: curl each slot directly, record Vulkan perf baseline.
- `GET /status` shows both slots healthy with correct models.
- `/model --target batch <name>` from PAL CLI swaps batch successfully.
- `PAL_BATCH_ENABLED=true` on PAL: real compile triggers categorizer on batch. Watch `nvidia-smi` for P40 staying on its chat model; watch iGPU activity (`radeontop` or equivalent).
- `sudo systemctl stop llama-server-batch` mid-compile → PAL's `BatchFallbackProposal` fires with retry/main/skip options.

### Explicitly not tested automatically

- Real Vulkan perf regression (no CI hardware match).
- Driver / kernel crash recovery (not reliably simulatable).

## Success Criteria

- `GET /status` returns both slots with correct `loaded_model` and `healthy = true` after fresh startup.
- `POST /swap {target: "batch", model: <new>}` swaps the batch slot's model without affecting main. `last_swap_utc` updates. `loaded_model` reflects the new model.
- Inference requests route by model name: requests for the main model land on main, requests for the batch model land on batch, requests for an unknown model trigger a main swap.
- `PAL_BATCH_ENABLED=true` on PAL: categorizer, learning scanner, and `detect_from_llm_toc` all fire against the batch slot in real usage, and chat latency remains stable under concurrent background load.
- `systemctl stop llama-server-batch` produces a visible `BatchFallbackProposal` in Discord/CLI with retry/main/skip. "Run on main" completes the affected operation successfully.

## Follow-up work (out of scope)

- Moving `compile_one`, `consolidate`, `summarize_raw_file`, `find_existing_article` to batch — separate per-caller spec once Phase B is measurable.
- `ask_file` delegated sub-call tool on batch — separate spec.
- Per-slot CTX autotune.
- Multi-slot (>2) generalization — the slot dict is already keyed by name, so extension is mechanical, but there's no concrete motivation yet.
- Manager watchdog that attempts self-heal on a crashed slot — for now, admin-triggered `/swap` is the recovery path.
