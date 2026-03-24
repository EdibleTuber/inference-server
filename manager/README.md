# Model Manager

The model manager is a Python FastAPI application that sits in front of llama-server. It is the only component that clients talk to — llama-server is never exposed directly.

---

## What it does

### 1. Request proxying

The manager is an OpenAI-compatible HTTP proxy. Clients send standard `POST /v1/chat/completions` requests; the manager forwards them to llama-server on `localhost:8081` and streams the response back. From the client's perspective, the manager looks like any OpenAI-compatible API.

Why proxy instead of clients talking directly to llama-server? The manager needs to intercept each request to check the `model` field and trigger a swap if needed. It also needs to queue requests so llama-server processes them one at a time. A transparent proxy layer makes both of those things possible without requiring any changes to existing clients.

### 2. Model switching

The manager watches the `model` field on every chat completions request. If the requested model differs from the currently loaded model, it orchestrates a hot-swap: updates llama-server's config file, restarts the process via systemd, and waits for it to become healthy before proceeding.

The client's HTTP connection stays open during the swap. The manager holds the request in memory while the swap completes (30–60+ seconds), then forwards it to llama-server and returns the response. The client just sees a slow request — no error, no need to retry.

### 3. FIFO request queuing

Every incoming request goes into a FIFO queue. A background task processes requests one at a time, forwarding each to llama-server and waiting for the response before taking the next. This ensures:

- llama-server (and the GPU) is never overloaded with concurrent inference
- Requests are processed in the order they arrived
- During model swaps, incoming requests accumulate in the queue rather than being dropped, and are processed once the swap completes

The queue has a configurable maximum depth (default: 20). Requests beyond the limit receive a 503 response with a `Retry-After` header.

### 4. Status and model listing

- `GET /v1/models` scans `/opt/llama/models/` and returns available GGUF files in OpenAI-compatible format. No config needed — just drop a GGUF in the directory and it appears.
- `GET /status` returns the server state, current model, queue depth, GPU VRAM usage, and uptime. Useful for health checks and for clients that want to check state before submitting a long job.
- `GET /health` returns 200 OK. Use for basic uptime monitoring.

---

## How model switching works

When a request arrives for a different model than the one currently loaded, the manager goes through a 5-step process:

```
Request arrives for model "llama-3-8b-instruct-q5_k_m"
Currently loaded: "qwen2.5-7b-instruct-q4_k_m"
                        │
                        ▼
1. Resolve model name to GGUF path
   "llama-3-8b-instruct-q5_k_m" → /opt/llama/models/llama-3-8b-instruct-q5_k_m.gguf
   (if not found, return 404 immediately — don't start a swap for a nonexistent model)
                        │
                        ▼
2. Set state to "swapping"
   New requests queue up; nothing is forwarded to llama-server during this phase
                        │
                        ▼
3. Update /etc/llama/llama-server.env
   Rewrite MODEL_PATH= line using regex; all other config preserved
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
   On 200: set state to "ready", process queued requests
   On timeout: set state to "error", drain queue with 503 responses
```

**Why update a file and restart instead of a hot-reload API?** llama-server does not have a runtime model-swap API. The env file + systemd restart approach is the supported mechanism. The manager owns `/etc/llama/llama-server.env` (it is the file's owner in the filesystem), so it can update it without elevated privileges. The sudoers entry covers only the `systemctl restart` command — it cannot touch anything else.

**What about queued requests for multiple models?** The queue is strict FIFO. If queued requests reference different models (e.g., request 1 wants model A, request 2 wants model B, request 3 wants model A again), the manager swaps as needed when it reaches each request. Consecutive requests for the same model are batched — a swap only happens when the model actually changes between adjacent queue entries.

---

## Files

| File | Purpose |
|---|---|
| `app.py` | The FastAPI application. Defines all endpoints, the `ServerState` class (holds mutable state: current model, queue, swap lock), and the background queue consumer task. This is the entry point and the glue between all other modules. |
| `config.py` | Typed configuration dataclass (`ManagerConfig`). Reads all settings from environment variables with sensible defaults. The rest of the app imports config values from here rather than reading `os.environ` directly — centralizes all env var names and type conversions in one place. |
| `queue.py` | The `RequestQueue` class: an async FIFO queue with a configurable size cap. Raises `QueueFullError` when at capacity (the API layer converts this to a 503 response). |
| `swap.py` | The `ModelSwapper` class: executes the three concrete steps of a model swap — updating the env file, running `systemctl restart`, and polling the health endpoint. Isolated here so it can be tested and mocked independently. |
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
export LLAMA_SERVER_ENV=/tmp/llama-server.env  # must exist; touch it first
export LOG_FILE=                     # empty = log to console only
```

Or create a `.env` file and source it:

```bash
# dev.env
HOST=127.0.0.1
PORT=8080
MODELS_DIR=/tmp/models
LLAMA_SERVER_ENV=/tmp/llama-server.env
LOG_FILE=
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
```

With no llama-server running, `/status` returns `"state": "error"`. Chat completions will fail with a 503 (no server to forward to). This is expected — everything up to the point of actually calling llama-server works.

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
- **`test_swap.py`** — Env file is updated with correct `MODEL_PATH`; `systemctl restart` is called; health polling returns True on 200 and False on timeout. All subprocess and HTTP calls are mocked — no llama-server or systemd required.
- **`test_endpoints.py`** — `/health`, `/status`, `/v1/models`, and `/v1/chat/completions` integration tests using FastAPI's test client. llama-server is mocked with an in-process HTTP server.
- **`test_gpu.py`** — `nvidia-smi` output parsing; graceful fallback when `nvidia-smi` is missing.
