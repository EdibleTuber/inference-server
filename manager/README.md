# Model Manager

The model manager is a Python FastAPI application that sits in front of llama-server. It is the only component that clients talk to — llama-server is never exposed directly.

---

## What it does

### 1. Request proxying

The manager is an OpenAI-compatible HTTP proxy. Clients send standard `POST /v1/chat/completions` requests; the manager forwards them to llama-server on `localhost:8081` and streams the response back. From the client's perspective, the manager looks like any OpenAI-compatible API.

Why proxy instead of clients talking directly to llama-server? The manager needs to intercept each request to check the `model` field against the roster (the profiles in `config/models/`) and the slots' loaded models, refusing with 404 or 409 before anything is forwarded. It also needs to queue requests so llama-server processes them one at a time. A transparent proxy layer makes both of those things possible without requiring any changes to existing clients.

The manager also exposes the Anthropic API — `POST /v1/messages` and `POST /v1/messages/count_tokens` — so Anthropic-dialect clients (e.g. Claude Code) can point at the same endpoint. These are a byte passthrough: llama-server speaks the Anthropic protocol natively, so the manager never re-translates the body. The Anthropic routes get the same model validation, slot routing, FIFO queueing and serial-GPU semantics as the OpenAI routes; `count_tokens` skips the queue, since it is cheap local tokenization on the backend. Manager-level errors on the Anthropic routes use the Anthropic error envelope: `{"type": "error", "error": {"type": ..., "message": ...}}`.

### 2. Model switching

Chat requests never trigger a swap: a chat for a model that is loaded on no slot gets **409**, which tells the client to load it. Swaps happen only through `POST /swap`: the manager renders the model's profile (the TOML file under `config/models/`) into the slot's generated model env file, restarts the slot's llama-server via systemd, and waits for it to become healthy.

`/swap` returns **202** as soon as the swap has started; the work runs in a background task, so no client connection is held open for the duration. The caller polls `GET /status` (the slot's `swapping`, `loaded_model` and `healthy` fields) until the swap lands.

### 3. FIFO request queuing

Every incoming request goes into a FIFO queue. A background task processes requests one at a time, forwarding each to llama-server and waiting for the response before taking the next. This ensures:

- llama-server (and the GPU) is never overloaded with concurrent inference
- Requests are processed in the order they arrived
- During a model swap, queued requests whose model stays loaded wait for the swap to finish (the readiness check holds the slot's swap lock) and are then processed as usual; a request whose model was swapped out gets the 409 a fresh request would

The queue has a configurable maximum depth (default: 20). Requests beyond the limit receive a 503 response with a `Retry-After` header.

### 4. Status and model listing

- `GET /v1/models` lists the onboarded profiles — models with at least one verified (model, slot) pair — in OpenAI-compatible format, with `home_slot`, `verified_slots` and `loaded_on` alongside the standard fields. It does not scan the models directory: a model appears only after `scripts/onboard.py` has verified it on at least one slot, and the profiles are read once at manager startup, so a new or edited profile needs a deploy and a manager restart to show up.
- `GET /status` returns per-slot health, loaded model, and queue depth, plus a top-level `profile_errors` (profiles that failed validation at startup), GPU VRAM usage and uptime. Useful for health checks and for clients that want to check a slot's state before submitting a long job.
- `GET /health` returns 200 OK. Use for basic uptime monitoring.

---

## How model switching works

When `POST /swap` arrives (body `{"model": ..., "target": <slot>}`; `target` defaults to the profile's home slot), the manager goes through a 5-step process:

```
Request arrives for model "Qwen3.6-35B-A3B-UD-IQ4_NL_XL" on slot "egpu"
                        │
                        ▼
1. Validate the request against the roster
   The model must have a profile (else 404 model_not_found); the target
   must be a slot the profile names (else 400 slot_not_allowed); and the
   (model, slot) pair must carry a verified block in the profile, or
   "trial" must be true (else 400 not_verified). A trial reloads the model
   even when it is already loaded — scripts/onboard.py uses it to verify
   pairs, and a trial that skipped the load would verify nothing.
                        │
                        ▼
2. Hold the slot's swap lock
   A second swap for the same slot waits its turn; nothing is restarted twice at once
                        │
                        ▼
3. Rewrite the slot's generated model env file
   /var/lib/llama-manager/<slot>.env is rewritten whole (atomically) from
   the profile's [slots.<slot>] section — the entire MODEL_ARGS line, so no
   flag can carry over from the previous model. The static env file in
   /etc/llama/ is never touched.
                        │
                        ▼
4. Restart llama-server via systemctl
   sudo systemctl restart llama-server.service
   (this is why _llama-mgr has a sudoers entry)
   llama-server exits, loads the new GGUF into VRAM, starts listening
                        │
                        ▼
5. Poll health endpoint
   GET http://127.0.0.1:8081/health every 2 seconds, up to 120 seconds
   On 200: mark the slot healthy, process queued requests
   On timeout: mark the slot unhealthy, drain its queue with 503 responses
```

**Why rewrite a file and restart instead of a hot-reload API?** llama-server does not have a runtime model-swap API. The env file + systemd restart approach is the supported mechanism. The manager owns the generated model env file under `/var/lib/llama-manager` (the directory belongs to `_llama-mgr`), so it can rewrite it without elevated privileges; the static env file in `/etc/llama/` is never touched. The sudoers entry covers only the `systemctl restart` command — it cannot touch anything else.

**What about queued requests for multiple models?** Every request is validated when it enqueues: a model with no profile gets 404 and a model loaded on no slot gets 409, so the queue only ever holds requests for models that are actually loaded. If a `POST /swap` swaps one of those models out while requests sit in the queue, the readiness check — which holds the slot's swap lock, so it sees the swap's final state — answers the affected requests with the 409 a fresh request would; everything else is served as usual.

---

## Files

| File | Purpose |
|---|---|
| `app.py` | The FastAPI application. Defines all endpoints, the `ServerState` class (holds a `slots` mapping — one `SlotState` per configured backend, each with its own loaded-model tracking, queue, and swap lock — plus the profile roster loaded at startup and the collection-retrieval state), and the background queue consumer task. This is the entry point and the glue between all other modules. |
| `config.py` | Typed configuration dataclass (`ManagerConfig`). Reads all settings from environment variables with sensible defaults. The rest of the app imports config values from here rather than reading `os.environ` directly — centralizes all env var names and type conversions in one place. |
| `profiles.py` | The profile format and roster: `load_profiles` validates the TOML files in the profiles directory (a bad file is reported, never fatal — it lands in `/status`'s `profile_errors`) and `render_model_env` renders one (model, slot) section into the text of the generated model env file. Also a small CLI (`python -m manager.profiles check\|render`) used by `deploy-manager.sh`, `onboard.py` and `doctor.sh`. |
| `queue.py` | The `RequestQueue` class: an async FIFO queue with a configurable size cap. Raises `QueueFullError` when at capacity (the API layer converts this to a 503 response). |
| `swap.py` | The `ModelSwapper` class: executes the concrete steps of a model swap on its slot — atomically writing the rendered model env file, running `systemctl restart`, and polling the health endpoint. Isolated here so it can be tested and mocked independently. |
| `gpu.py` | Queries `nvidia-smi` for GPU name and VRAM usage. Called on every `/status` request since VRAM usage changes as models load. Falls back gracefully if `nvidia-smi` is unavailable (returns `"unknown"` with 0 values), so the manager works normally in development environments without a GPU. |
| `requirements.txt` | Python dependencies: FastAPI, uvicorn, httpx, pytest, pytest-asyncio. |
| `__init__.py` | Empty package marker. Makes `manager/` a Python package so imports work as `from manager.config import ManagerConfig`. |

---

## Running locally for development

The manager can run on a development machine without llama-server or an NVIDIA GPU. It will start in `error` state (no llama-server to connect to), but all endpoints work and can be tested against.

### Setup

```bash
cd manager
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure

Set environment variables to override defaults. For local development, the most important ones are:

```bash
export HOST=127.0.0.1
export PORT=8080
export MODELS_DIR=/tmp/models        # create this and put some .gguf files in it
export PROFILES_DIR=/tmp/models.d    # per-model TOML profiles; copy one from config/models/
export MODEL_ENV_DIR=/tmp/model-env  # generated per-slot model env files; create this
export LLAMA_SERVER_ENV=/tmp/llama-server.env  # the slot's static env file; touch it first
```

Or create a `.env` file and source it:

```bash
# dev.env
HOST=127.0.0.1
PORT=8080
MODELS_DIR=/tmp/models
PROFILES_DIR=/tmp/models.d
MODEL_ENV_DIR=/tmp/model-env
LLAMA_SERVER_ENV=/tmp/llama-server.env
```

```bash
source dev.env
```

### Run

```bash
# From inside manager/
python app.py

# Or with uvicorn directly (auto-reload on file changes, useful during development)
uvicorn manager.app:create_app --factory --reload --host 127.0.0.1 --port 8080
```

The `--factory` flag tells uvicorn to call `create_app()` to get the FastAPI application, which ensures config is loaded from environment on each reload.

### Test endpoints locally

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/status
curl http://127.0.0.1:8080/v1/models

# Anthropic-dialect endpoints (byte passthrough to llama-server)
curl http://127.0.0.1:8080/v1/messages -X POST -H 'content-type: application/json' \
  -d '{"model": "<loaded-model>", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}'
curl http://127.0.0.1:8080/v1/messages/count_tokens -X POST -H 'content-type: application/json' \
  -d '{"model": "<loaded-model>", "messages": [{"role": "user", "content": "hi"}]}'
```

With no llama-server running, each slot in `/status` reports `"healthy": false` (and `"loaded_model": null` if nothing has ever loaded successfully). A chat for a model with no profile gets 404, and a chat for a model no slot has loaded gets 409 — the request never reaches a backend. This is expected: everything up to the point of actually calling llama-server works.

---

## Running tests

Tests live in `../tests/` (the project-level `tests/` directory, not inside `manager/`). They use `pytest` and `pytest-asyncio`.

### Setup

```bash
# From the project root (inference_server/)
python3 -m venv venv
source venv/bin/activate
pip install -r manager/requirements.txt
```

### Run all tests

```bash
pytest tests/
```

### Run specific test files

```bash
pytest tests/test_queue.py      # queue behavior
pytest tests/test_swap.py       # model swap logic (mocks systemctl and llama-server)
pytest tests/test_config.py     # config loading from env vars
pytest tests/test_endpoints.py  # API endpoint integration tests
pytest tests/test_gpu.py        # nvidia-smi output parsing
```

### Run with verbose output

```bash
pytest tests/ -v
```

### What the tests cover

- **`test_config.py`** — Config loads values from env vars; uses sensible defaults when vars are absent; produces correct llama-server URL.
- **`test_queue.py`** — Enqueue/dequeue ordering; `QueueFullError` at capacity; `drain()` clears all items.
- **`test_swap.py`** — the slot's generated model env file is written with the given `MODEL_ARGS` text; `systemctl restart` is called on the slot's unit; health polling returns True on 200 and False on timeout. All subprocess and HTTP calls are mocked — no llama-server or systemd required.
- **`test_endpoints.py`** — `/health`, `/status`, `/v1/models`, and `/v1/chat/completions` integration tests using FastAPI's `TestClient`. `TestClient` runs the app's lifespan, so the startup slot probe — and several individual tests — issue real HTTP requests to whatever is actually listening on the configured slot ports (127.0.0.1:18081 main, 127.0.0.1:18083 egpu in `test_config`), not an in-process mock; those tests are written to pass whether or not a backend is up there. A few tests instead monkeypatch `SlotState.probe` directly, or spin up a local `http.server` instance to control one backend's response precisely.
- **`test_gpu.py`** — `nvidia-smi` output parsing; graceful fallback when `nvidia-smi` is missing.
