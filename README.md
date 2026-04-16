# Inference Server

A native llama.cpp inference server with an OpenAI-compatible API, API-driven model switching, and FIFO request queuing. Runs on Ubuntu Server with an NVIDIA GPU.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Ubuntu Server (Headless)                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Model Manager (Python/FastAPI)         LAN_IP:11434    │   │
│  │  user: _llama-mgr                                        │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │ OpenAI-compatible API                              │  │   │
│  │  │ ├─ POST /v1/chat/completions (proxy + model swap)  │  │   │
│  │  │ ├─ GET  /v1/models (list available GGUFs)          │  │   │
│  │  │ ├─ GET  /status (state, model, GPU, queue info)    │  │   │
│  │  │ └─ GET  /health (simple 200 OK)                    │  │   │
│  │  └────────────────────┬───────────────────────────────┘  │   │
│  │                       │                                  │   │
│  │  FIFO Request Queue (max 20, configurable)               │   │
│  └───────────────────────┼──────────────────────────────────┘   │
│                          │ localhost:8081                        │
│  ┌───────────────────────┴──────────────────────────────────┐   │
│  │  llama-server (systemd)              127.0.0.1:8081      │   │
│  │  user: _llama  ·  NVIDIA P40  ·  --n-gpu-layers auto    │   │
│  └───────────────────────┬──────────────────────────────────┘   │
│                          │                                      │
│  ┌───────────────────────┴──────────────────────────────────┐   │
│  │  /opt/llama/models/    (GGUF storage)                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Why this design?

**Two-layer architecture.** Clients talk only to the model manager on port 11434. The model manager talks to llama-server on localhost:8081. llama-server is never directly exposed to the network. This separation means:

- The manager can restart llama-server for model swaps without dropping client connections — it holds the client's HTTP connection open while the swap completes, then proxies the response.
- Privilege separation: llama-server only needs to read model files and access the GPU. The manager only needs to restart llama-server and update one config file. Neither service needs broad system access.

**llama.cpp instead of Ollama.** Running llama-server natively (not in Docker) gives direct GPU access, no container overhead, and access to any GGUF on HuggingFace without waiting for Ollama to support it. The tradeoff is more setup — this repo contains the setup scripts and config templates to make it repeatable.

**FIFO queue instead of parallel inference.** The Tesla P40 has 24GB VRAM. Running one request at a time maximizes throughput per request (the GPU is fully dedicated to each). Parallel inference would split VRAM across requests and slow down each individual one. The queue ensures requests are processed in order, even during model swaps.

**Dedicated system users.** `_llama` runs llama-server with read-only access to model files. `_llama-mgr` runs the manager with write access to one config file and a narrow sudoers entry to restart llama-server. Neither user has a shell or home directory. If either service were compromised, the blast radius is minimal.

---

## Quick Start

### Prerequisites

- Ubuntu Server (tested on 22.04+)
- NVIDIA GPU with CUDA drivers installed
- Python 3.10+
- llama.cpp compiled with CUDA support (see [llama.cpp build docs](https://github.com/ggerganov/llama.cpp))

### Setup

**1. Run the system setup script** (creates users, directories, sudoers entry, installs systemd units):

```bash
sudo bash scripts/setup.sh
```

This script creates:
- System users `_llama` and `_llama-mgr` (no shell, no home directory)
- `/opt/llama/bin/`, `/opt/llama/models/`, `/opt/llama/manager/`
- `/etc/llama/` (config files), `/var/log/llama/` (log files)
- A narrow sudoers entry so `_llama-mgr` can restart `llama-server.service`
- Systemd unit files and enables both services

**2. Install the llama-server binary:**

```bash
sudo cp /path/to/llama-server /opt/llama/bin/llama-server
sudo chmod +x /opt/llama/bin/llama-server
```

**3. Deploy the model manager:**

```bash
sudo cp -r manager/ /opt/llama/manager/
cd /opt/llama/manager
sudo python3 -m venv venv
sudo venv/bin/pip install -r requirements.txt
```

**4. Edit the config files:**

```bash
# Set your LAN IP in the manager config
sudo nano /etc/llama/manager.env

# Set initial model path if you already have a model downloaded
sudo nano /etc/llama/llama-server.env
```

**5. Download a model:**

```bash
sudo bash scripts/download-model.sh \
  https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  qwen2.5-7b-instruct-q4_k_m
```

**6. Start the services:**

```bash
sudo systemctl start llama-server
sudo systemctl start llama-manager
```

**7. Verify everything is up:**

```bash
curl http://localhost:11434/health
curl http://localhost:11434/status
curl http://localhost:11434/v1/models
```

---

## Usage Examples

### Basic chat completion

```bash
curl http://YOUR_LAN_IP:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct-q4_k_m",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming response

```bash
curl http://YOUR_LAN_IP:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct-q4_k_m",
    "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
    "stream": true
  }'
```

### Switch to a different model

Just change the `model` field. The manager detects the change and hot-swaps llama-server automatically. The client connection stays open while the swap completes (30–60+ seconds). Set your HTTP timeout to at least 120 seconds.

```bash
curl http://YOUR_LAN_IP:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b-instruct-q5_k_m",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

### Check server status before sending a request

Useful if you want to avoid waiting through a model swap, or to check queue depth before submitting work.

```bash
curl http://YOUR_LAN_IP:11434/status
```

```json
{
  "state": "ready",
  "current_model": "qwen2.5-7b-instruct-q4_k_m",
  "loading_model": null,
  "error_message": null,
  "queue_depth": 0,
  "queue_limit": 20,
  "uptime_seconds": 3421,
  "gpu": {
    "name": "Tesla P40",
    "vram_total_mb": 24576,
    "vram_used_mb": 18200
  }
}
```

### Using with OpenAI Python client

The API is OpenAI-compatible, so existing clients work without modification:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://YOUR_LAN_IP:11434/v1",
    api_key="not-needed",  # no auth on internal network
)

response = client.chat.completions.create(
    model="qwen2.5-7b-instruct-q4_k_m",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

---

## Model Management

Models are GGUF files stored in `/opt/llama/models/`. The filename (without `.gguf`) is the model name used in API requests.

### Download a model

```bash
sudo bash scripts/download-model.sh \
  https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf \
  llama-3-8b-instruct-q5_k_m
```

### List available models

```bash
curl http://YOUR_LAN_IP:11434/v1/models
```

New models are available immediately after download — no service restart needed. The manager scans the directory on every request to `/v1/models`.

### Manual download

```bash
sudo -u _llama wget -O /opt/llama/models/my-model-q4.gguf \
  "https://huggingface.co/.../model.gguf"
```

Or download as any user in the `llama` group (the directory is group-writable).

### Model naming convention

Use descriptive names that encode the model family, size, and quantization level. Examples:

- `qwen2.5-7b-instruct-q4_k_m` — Qwen 2.5, 7B parameters, instruction-tuned, Q4_K_M quantization
- `llama-3-8b-instruct-q5_k_m` — Llama 3, 8B, instruction-tuned, Q5_K_M quantization
- `deepseek-r1-14b-q6_k` — DeepSeek R1, 14B, Q6_K quantization

The manager uses exact filename matching — the model name in a request must exactly match the filename without `.gguf`.

---

## Configuration Reference

### `/etc/llama/llama-server.env`

Configuration for the llama-server inference backend. **The manager updates `MODEL_PATH` automatically** during model swaps — do not edit this file while the services are running.

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | _(empty)_ | Absolute path to the currently loaded GGUF file. Leave empty on first boot; the manager sets it on the first request. |
| `N_GPU_LAYERS` | `-1` | Number of model layers to offload to GPU. `-1` = auto (fill all available VRAM). Set to a specific number to limit GPU usage. |
| `CTX_SIZE` | `4096` | Context window size in tokens. Larger values use more VRAM. 4096 is safe for large models on the 24GB P40. |
| `HOST` | `127.0.0.1` | Bind address for llama-server. Always localhost — never expose directly. |
| `PORT` | `8081` | Port for llama-server. The manager connects here. |

### `/etc/llama/manager.env`

Configuration for the model manager proxy service.

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address for the manager. Set to your LAN IP to restrict access, or `0.0.0.0` for all interfaces. |
| `PORT` | `8080` | Port the manager listens on. Clients connect here. |
| `LLAMA_SERVER_HOST` | `127.0.0.1` | Address where llama-server is running. Always localhost. |
| `LLAMA_SERVER_PORT` | `8081` | Port where llama-server listens. Must match llama-server.env. |
| `MODELS_DIR` | `/opt/llama/models` | Directory containing GGUF model files. |
| `LLAMA_SERVER_ENV` | `/etc/llama/llama-server.env` | Path to llama-server's env file. The manager writes `MODEL_PATH` here during swaps. |
| `QUEUE_LIMIT` | `20` | Maximum requests to hold in the FIFO queue. Requests beyond this get a 503 response. |
| `SWAP_TIMEOUT` | `120` | Seconds to wait for llama-server health after a model swap. Exceeding this puts the manager in `error` state. |
| `LOG_FILE` | `/var/log/llama/manager.log` | Log file path for the model manager. |
| `EMBEDDINGS_HOST` | `127.0.0.1` | Address where llama-embeddings is running. Always localhost. |
| `EMBEDDINGS_PORT` | `8082` | Port where llama-embeddings listens. |
| `COLLECTIONS_CONFIG` | `/etc/llama/collections.json` | Path to collection definitions JSON file. |
| `SKILLS_DB_PATH` | `/opt/llama/data/skills.db` | Path to the SQLite-vec database for document retrieval. |

---

## API Endpoints

All endpoints are on `LAN_IP:11434`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completions. Reads the `model` field, swaps if needed, queues if busy. Supports streaming (`"stream": true`). |
| `POST` | `/v1/embeddings` | Proxy to llama-embeddings instance. OpenAI-compatible. |
| `GET` | `/v1/models` | Lists available GGUF files as an OpenAI-compatible model list. |
| `GET` | `/status` | Server state, current model, queue depth, GPU VRAM usage, uptime. |
| `GET` | `/health` | Returns `{"status": "ok"}` with HTTP 200. Use for uptime monitors. |
| `GET` | `/collections` | Lists registered document collections with document counts. |
| `POST` | `/collections/{id}/search` | Semantic search within a collection. Returns ranked summaries. |
| `GET` | `/collections/{id}/docs/{doc_id}` | Full document content by ID. |
| `POST` | `/collections/{id}/reindex` | Trigger an incremental reindex. Optional body `{"paths": [...]}` limits scope to specific files; omitted or `null` runs a full scan with stale-deletion. Returns 202 with a job id. |
| `GET` | `/collections/{id}/reindex/status` | Current or most recent reindex job for a collection. 404 if none has run. |
| `GET` | `/collections/{id}/reindex/{job_id}` | State of a specific reindex job (status, stats, error, timestamps). |

### Server states

The `state` field in `/status` tells you what the manager is doing:

| State | Meaning |
|---|---|
| `loading` | llama-server is starting up on boot. No model available yet. |
| `ready` | Accepting and processing requests normally. |
| `swapping` | Actively changing models. Requests are queuing. |
| `error` | llama-server failed to start or health poll timed out. Send a new request with a valid model name to trigger a fresh swap attempt. |

### Client timeout guidance

Model swaps take 30–60+ seconds depending on model size (the GPU must load a new model file into VRAM). During a swap, the manager holds your HTTP connection open. **Set your HTTP client timeout to at least 120 seconds** to avoid timing out while waiting for a swap.

If you prefer not to wait, poll `/status` before sending requests and only proceed when `state` is `"ready"`.

---

## Collection-Based Document Retrieval

The manager includes a vector search system for retrieving documents (skills, notes, etc.)
by semantic similarity. Documents are embedded using a dedicated CPU-only llama.cpp instance
running nomic-embed-text.

### Setup

1. **Download the embedding model:**

```bash
./scripts/download-model.sh nomic-ai/nomic-embed-text-v1.5-GGUF nomic-embed-text-v1.5.Q8_0.gguf
```

2. **Install the embeddings service:**

```bash
sudo cp systemd/llama-embeddings.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-embeddings
```

3. **Configure collections** in `/etc/llama/collections.json`:

```json
[
  { "id": "skills", "source_dir": "/home/edible/.pi/skills", "doc_type": "skill" },
  { "id": "notes", "source_dir": "/home/edible/vault", "doc_type": "markdown" }
]
```

4. **Add env vars** to `/etc/llama/manager.env`:

```bash
EMBEDDINGS_HOST=127.0.0.1
EMBEDDINGS_PORT=8082
COLLECTIONS_CONFIG=/etc/llama/collections.json
SKILLS_DB_PATH=/opt/llama/data/skills.db
```

5. **Restart the manager** — indexing runs automatically on startup.

### API

**Search documents (two-step retrieval):**

```bash
# Step 1: Search for relevant skills
curl -s http://LAN_IP:11434/collections/skills/search \
  -H "Content-Type: application/json" \
  -d '{"query": "network reconnaissance", "limit": 3}' | jq

# Step 2: Get full document content
curl -s http://LAN_IP:11434/collections/skills/docs/Security/Recon/Workflows/PassiveRecon | jq

# List collections
curl -s http://LAN_IP:11434/collections | jq
```

**Trigger a reindex without restarting the server:**

```bash
# Full scan (rescans every file in source_dir; deletes rows for files no longer on disk)
curl -s -X POST http://LAN_IP:11434/collections/vault/reindex \
  -H "Content-Type: application/json" -d '{}' | jq

# Scoped scan (only the listed paths; skips stale-deletion). Useful right after writing
# specific files. Paths must be absolute and under the collection's source_dir.
curl -s -X POST http://LAN_IP:11434/collections/vault/reindex \
  -H "Content-Type: application/json" \
  -d '{"paths": ["/path/to/source_dir/Some/Article.md"]}' | jq

# Both POSTs return 202 with a job_id immediately. Poll status:
curl -s http://LAN_IP:11434/collections/vault/reindex/status | jq
curl -s http://LAN_IP:11434/collections/vault/reindex/<job_id> | jq
```

Concurrent POSTs to the same collection return the in-flight `job_id` rather than stacking duplicate scans. Jobs are in-memory only and wiped on server restart (which already triggers a full reindex).


**Generate embeddings directly:**

```bash
curl -s http://LAN_IP:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "nomic-embed-text", "input": "hello world"}' | jq
```

### Architecture

```
Manager (:11434)
├── /v1/chat/completions → llama-server (:8081, GPU)
├── /v1/embeddings       → llama-embeddings (:8082, CPU)
├── /collections/*/search → SQLite-vec (in-process)
└── /collections/*/docs/* → SQLite-vec (in-process)
```

Documents are indexed on startup with SHA-256 hash-based change detection.
Only new or modified files are re-embedded.

---

## Service Management

### Start / stop / restart

```bash
# Start both services
sudo systemctl start llama-server llama-manager

# Stop both services
sudo systemctl stop llama-manager llama-server

# Restart just the manager (e.g., after config change)
sudo systemctl restart llama-manager

# Restart llama-server (loads fresh model from env file)
sudo systemctl restart llama-server
```

### Check service status

```bash
sudo systemctl status llama-server
sudo systemctl status llama-manager
```

### View logs

```bash
# Live log from systemd journal
sudo journalctl -u llama-server -f
sudo journalctl -u llama-manager -f

# File-based logs (also written for persistence across reboots)
sudo tail -f /var/log/llama/manager.log
sudo tail -f /var/log/llama/llama-server.log
```

### Enable on boot

Both services are enabled during setup. To check or change:

```bash
sudo systemctl enable llama-server llama-manager
sudo systemctl disable llama-server llama-manager
```

### Boot sequence

1. System boots, NVIDIA drivers load
2. `llama-server.service` starts with the model configured in `/etc/llama/llama-server.env`
3. `llama-manager.service` starts (`After=llama-server.service`), connects to llama-server, begins accepting requests

If `MODEL_PATH` is empty or points to a nonexistent file on boot, llama-server fails to start. The manager starts in `error` state and waits. The first API request with a valid model name triggers a swap, which loads the model and transitions to `ready`.

---

## File System Layout

```
/opt/llama/
  ├── bin/
  │   └── llama-server              # compiled llama.cpp binary (CUDA)
  ├── models/                       # GGUF storage
  │   ├── qwen2.5-7b-instruct-q4_k_m.gguf
  │   ├── nomic-embed-text-v1.5.Q8_0.gguf
  │   └── ...
  ├── data/
  │   └── skills.db                 # SQLite-vec database for collections
  └── manager/                      # model manager Python app
      ├── venv/                     # isolated virtualenv
      ├── app.py                    # main FastAPI application
      ├── config.py                 # configuration loading
      ├── queue.py                  # FIFO request queue
      ├── swap.py                   # model swap orchestration
      ├── gpu.py                    # GPU info via nvidia-smi
      ├── embeddings.py             # embeddings client for llama-embeddings
      ├── vectordb.py               # SQLite-vec vector database wrapper
      ├── collections.py            # collection indexing pipeline
      └── requirements.txt

/etc/llama/
  ├── llama-server.env              # runtime config for llama-server (manager writes MODEL_PATH here)
  ├── manager.env                   # runtime config for model manager
  └── collections.json              # collection definitions for document retrieval

/var/log/llama/
  ├── llama-server.log              # inference server stdout/stderr
  ├── manager.log                   # model manager application log
  ├── embeddings.log                # embedding server stdout
  └── embeddings.err                # embedding server stderr

/etc/systemd/system/
  ├── llama-server.service
  ├── llama-manager.service
  └── llama-embeddings.service
```

### This repository

```
inference_server/
├── manager/                        # model manager source (deployed to /opt/llama/manager/)
│   ├── app.py                      # FastAPI app, endpoints, proxy, queue, swap logic
│   ├── config.py                   # configuration loading from env vars
│   ├── queue.py                    # FIFO request queue
│   ├── swap.py                     # model swap orchestration
│   ├── gpu.py                      # GPU info via nvidia-smi
│   ├── embeddings.py               # async client for llama-embeddings instance
│   ├── vectordb.py                 # SQLite-vec vector database wrapper
│   ├── collections.py              # collection indexing pipeline
│   ├── requirements.txt            # Python dependencies
│   └── README.md                   # manager component documentation
├── systemd/                        # systemd unit files (copied to /etc/systemd/system/)
│   ├── llama-server.service
│   ├── llama-manager.service
│   └── llama-embeddings.service    # CPU-only embedding server
├── config/                         # template config files (copied to /etc/llama/)
│   ├── llama-server.env
│   ├── manager.env
│   ├── collections.json            # collection definitions
│   └── llama-logrotate
├── scripts/
│   ├── setup.sh                    # system setup: users, dirs, permissions, sudoers, systemd
│   └── download-model.sh           # GGUF download helper
└── tests/                          # model manager tests
    ├── conftest.py
    ├── test_config.py
    ├── test_queue.py
    ├── test_swap.py
    ├── test_endpoints.py
    ├── test_gpu.py
    ├── test_embeddings.py
    ├── test_vectordb.py
    ├── test_collections.py
    └── test_collection_endpoints.py
```

---

## Security Notes

- **No authentication.** This is intentional for an internal network. Use Tailscale for remote access (encrypted, authenticated at the network layer).
- **llama-server is localhost-only.** It is never exposed to the network. Only the manager can reach it.
- **Dedicated system users.** `_llama` and `_llama-mgr` have no shell, no home directory, and minimal permissions. If a service were compromised, access is tightly scoped.
- **Narrow sudoers.** `_llama-mgr` can only run `systemctl restart llama-server.service`. Nothing else.
- **llama-embeddings is localhost-only.** The embedding server on port 8082 is never exposed to the network.
- **No TLS.** Acceptable on a trusted LAN or Tailscale tunnel. Do not expose port 11434 directly to the internet.
