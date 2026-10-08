# Inference Server

A native llama.cpp inference server that fronts several llama-server backends ("slots") behind one endpoint. It speaks both the OpenAI API and the Anthropic Messages API, routes each request to the slot that has the requested model loaded, swaps models on demand via `POST /swap`, and queues requests FIFO per slot. Runs on Ubuntu Server.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Ubuntu Server (Headless)                        │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Model Manager (Python/FastAPI)              LAN_IP:11434      │  │
│  │  user: _llama-mgr                                              │  │
│  │   OpenAI API     POST /v1/chat/completions, GET /v1/models     │  │
│  │   Anthropic API  POST /v1/messages, /v1/messages/count_tokens  │  │
│  │   Admin          POST /swap, GET /status, GET /health          │  │
│  │   Retrieval      POST /v1/embeddings, /collections/*           │  │
│  │                                                                │  │
│  │   routes by model → one FIFO queue per slot                    │  │
│  └──────┬──────────────────┬──────────────────┬───────────┬───────┘  │
│         │                  │                  │           │          │
│  ┌──────┴───────┐  ┌───────┴──────┐  ┌────────┴─────┐  ┌──┴───────┐  │
│  │ slot: main   │  │ slot: batch  │  │ slot: re     │  │embeddings│  │
│  │ :8081        │  │ :8083        │  │ :8084        │  │ :8082    │  │
│  │ V100 32GB    │  │ Vulkan iGPU  │  │ P40 24GB     │  │ CPU only │  │
│  │ llama-server │  │ llama-server │  │ llama-server │  │          │  │
│  │ .service     │  │ -batch       │  │ -re          │  │          │  │
│  └──────┬───────┘  └───────┬──────┘  └────────┬─────┘  └──┬───────┘  │
│         └──────────────────┴─────────┬────────┴───────────┘          │
│                     user: _llama     │  all bound to 127.0.0.1       │
│  ┌───────────────────────────────────┴────────────────────────────┐  │
│  │  /opt/llama/models/    (GGUF storage)                          │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

Which slots exist, and in what routing order, is set by `SLOTS` in `/etc/llama/manager.env` (see [Configuration Reference](#etcllamamanagerenv)). The hardware labels above describe this host; each GPU slot is pinned to its card by UUID, not by ordinal.

### Why this design?

**Two-layer architecture.** Clients talk only to the model manager on port 11434. The manager talks to each slot's llama-server on localhost (`:8081` main, `:8083` batch, `:8084` re). No llama-server is directly exposed to the network. This separation means:

- The manager can restart a slot's llama-server for a model swap while other slots keep serving, and clients never need to know which backend holds which model.
- Privilege separation: llama-server only needs to read model files and access its GPU. The manager only needs to restart the slot units and update their env files. Neither service needs broad system access.

**llama.cpp instead of Ollama.** Running llama-server natively (not in Docker) gives direct GPU access, no container overhead, and access to any GGUF on HuggingFace without waiting for Ollama to support it. The tradeoff is more setup — this repo contains the setup scripts and config templates to make it repeatable.

**FIFO queue per slot instead of parallel inference.** Each slot runs one request at a time, so its GPU is fully dedicated to that request; parallel inference would split VRAM (and context) across requests and slow each one down. Each slot has its own queue, so a long job on `re` never waits behind `main`. Requests to a slot are processed in order, even during a swap on that slot.

**Dedicated system users.** `_llama` runs every llama-server with read-only access to model files. `_llama-mgr` runs the manager with write access to the slot env files and a narrow sudoers entry to restart the slot units. Neither user has a shell or home directory. If either service were compromised, the blast radius is minimal.

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
- The `llama-server`, `llama-manager` and `llama-embeddings` unit files (installed, not enabled — step 6 starts them)

`setup.sh` covers the `main` slot only. The `batch` and `re` slots' units, env files and sudoers lines are installed by hand — see [Adding the batch and re slots](#adding-the-batch-and-re-slots). The template `config/manager.env` lists all three in `SLOTS`; drop the ones you have not installed, or the manager reports them as permanently unhealthy.

**2. Install the llama-server binary:**

```bash
sudo cp /path/to/llama-server /opt/llama/bin/llama-server
sudo chmod +x /opt/llama/bin/llama-server
```

**3. Deploy the model manager:**

First install only — `scripts/setup.sh` does this as step 7, along with the
users, permissions, sudoers entry and systemd units:

```bash
sudo ./scripts/setup.sh
```

To ship a code change to an existing install, use the deploy script. It reports
drift without privilege, so you can see what would change before changing it:

```bash
./scripts/deploy-manager.sh --check     # read-only, no sudo
sudo ./scripts/deploy-manager.sh        # install, stamp, restart, verify
cat /opt/llama/manager/DEPLOYED_FROM    # which commit is actually running
```

> **Do not use `cp -r manager/ /opt/llama/manager/`.** This README said that for
> a long time, and it is correct exactly once — against a target that does not
> exist yet. With `/opt/llama/manager` already present it creates
> `/opt/llama/manager/manager/`, leaving the new code one level down while the
> service keeps running the old: a silent no-op with no error to notice.
>
> That is not hypothetical. Three modules — `app.py`, `gpu.py` and
> `vectordb.py` — were found behind this repo on the live host, carrying merged
> work that had never shipped, because there was no safe documented way to ship
> it.

**4. Edit the config files:**

```bash
# Set your LAN IP in the manager config
sudo nano /etc/llama/manager.env

# Set initial model path if you already have a model downloaded
sudo nano /etc/llama/llama-server.env
```

**5. Download a model** (arguments are a HuggingFace repo and a filename in it; needs `huggingface-cli`):

```bash
./scripts/download-model.sh \
  bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

**6. Start the services:**

```bash
sudo systemctl enable --now llama-server llama-manager llama-embeddings
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

Chat requests never trigger a swap: a request for a model that is not loaded on any slot gets **409**. Load it explicitly with `POST /swap`, naming the slot as `target` (default: the first slot in `SLOTS`). The call returns once the slot is healthy on the new model — 30 seconds to several minutes for a large cold load — so give it a long client timeout.

```bash
curl -m 900 http://YOUR_LAN_IP:11434/swap \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3-8b-instruct-q5_k_m", "target": "main"}'
# → {"slot": "main", "model": "llama-3-8b-instruct-q5_k_m", "status": "ok"}
```

Errors: 400 for a missing `model` or unknown `target`, 404 if no such GGUF exists, 503 if the swap fails.

### Anthropic Messages API

The same endpoint serves the Anthropic API, so Anthropic-dialect clients (e.g. Claude Code) can point at it. Both routes are a byte passthrough to llama-server's native Anthropic support, with the same model validation, slot routing and queueing as the OpenAI routes. `count_tokens` skips the queue.

```bash
curl http://YOUR_LAN_IP:11434/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-7b-instruct-q4_k_m", "max_tokens": 256,
       "messages": [{"role": "user", "content": "Hello!"}]}'

curl http://YOUR_LAN_IP:11434/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-7b-instruct-q4_k_m",
       "messages": [{"role": "user", "content": "Hello!"}]}'
```

Manager-level errors on these routes use the Anthropic error envelope, `{"type": "error", "error": {"type": ..., "message": ...}}`.

### Check server status before sending a request

Useful if you want to avoid waiting through a model swap, or to check queue depth before submitting work.

```bash
curl http://YOUR_LAN_IP:11434/status
```

This is illustrative — the authoritative shape is whatever `GET /status` actually returns. `gpu.gpus` has one entry per GPU card detected on the host, not a fixed number; the example below shows two cards to illustrate that a multi-GPU host is reported in full, not because any particular host has exactly two:

```json
{
  "slots": {
    "main": {
      "host": "127.0.0.1",
      "port": 8081,
      "loaded_model": "qwen2.5-7b-instruct-q4_k_m",
      "healthy": true,
      "last_swap_utc": "2026-09-23T14:02:11+00:00",
      "queue_depth": 0,
      "queue_limit": 20
    },
    "batch": {
      "host": "127.0.0.1",
      "port": 8083,
      "loaded_model": null,
      "healthy": false,
      "last_swap_utc": null,
      "queue_depth": 0,
      "queue_limit": 20
    },
    "re": {
      "host": "127.0.0.1",
      "port": 8084,
      "loaded_model": "Qwen3.6-35B-A3B-UD-IQ4_NL_XL",
      "healthy": true,
      "last_swap_utc": "2026-10-04T09:12:40+00:00",
      "queue_depth": 1,
      "queue_limit": 20
    }
  },
  "gpu": {
    "name": "Tesla PG500-216",
    "vram_total_mb": 32768,
    "vram_used_mb": 19039,
    "gpus": [
      {
        "index": 0,
        "name": "Tesla PG500-216",
        "vram_total_mb": 32768,
        "vram_used_mb": 19039
      },
      {
        "index": 1,
        "name": "Tesla P40",
        "vram_total_mb": 24576,
        "vram_used_mb": 20710
      }
    ]
  },
  "uptime_seconds": 3421
}
```

`name`, `vram_total_mb`, and `vram_used_mb` at the top of `gpu` always mirror `gpus[0]`, for any consumer that only expects a single GPU.

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
# <huggingface-repo> <filename-in-repo>
./scripts/download-model.sh \
  bartowski/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q5_K_M.gguf
```

The file keeps its upstream name, and that stem is the model name clients use. Rename it afterwards if you want a shorter one.

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

The manager matches model names **case-insensitively** and ignores a `.gguf`
suffix or any leading path; an exact-case filename match wins when present.
The authoritative model identity for clients is `slots.<slot>.loaded_model`
in `GET /status` (the canonical on-disk stem). On-disk GGUF files must use a
lowercase `.gguf` extension. A request for a model loaded on no slot
returns **409**; load it first via `POST /swap`.

---

## Configuration Reference

### `/etc/llama/llama-server.env`

Configuration for the llama-server inference backend. **The manager updates `MODEL_PATH` automatically** during model swaps — do not edit this file while the services are running.

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | _(empty)_ | Absolute path to the currently loaded GGUF file. Leave empty on first boot; the manager sets it on the first request. |
| `DEVICE` | `CUDA0` | `--device` selector passed to llama-server. Names an enumeration position, not a physical card — unambiguous only in combination with `CUDA_VISIBLE_DEVICES` below. |
| `CUDA_VISIBLE_DEVICES` | _(host-specific, no safe default)_ | Restricts this process to exactly one physical GPU by UUID (`GPU-<uuid>`), so `DEVICE=CUDA0` above always names the same card regardless of enumeration order. Get the UUID with `nvidia-smi --query-gpu=uuid,pci.bus_id,name --format=csv`; never use a bare ordinal (`0`) here, since that reintroduces the same enumeration-order ambiguity. See the device-pinning section of [`docs/superpowers/specs/2026-09-23-dual-gpu-three-slot-design.md`](docs/superpowers/specs/2026-09-23-dual-gpu-three-slot-design.md). |
| `SAMPLING_ARGS` | _(model-specific)_ | Extra llama-server flags for the model in `MODEL_PATH` — sampling (`--temp`, `--top-p`, …) and, where the GGUF supports it, speculative decoding. Lives next to `MODEL_PATH` because sampling tuned for one model is wrong for the next: **change both together**. Must stay unbraced (`$SAMPLING_ARGS`) in the unit's `ExecStart` so systemd word-splits it. Empty means llama.cpp's own defaults. |
| `N_GPU_LAYERS` | `-1` | Number of model layers to offload to GPU. `-1` = all. Note that llama.cpp's load-time auto-fit can still move MoE expert tensors to system RAM while reporting every layer offloaded; check `nvidia-smi` after a start (see the comment in `config/llama-server-re.env`). |
| `GGML_CUDA_DISABLE_GRAPHS` | `1` | Works around a CUDA-graph memory leak; see the comment in `config/llama-server.env`. |
| `CTX_SIZE` | _(measured per model)_ | Context window size in tokens. Larger values use more VRAM. Derive this from the `llama_kv_cache` line in llama-server's own log (raise `-lv` until the line appears) after loading the target model — see the comments in `config/llama-server.env` — rather than guessing from a projection. Re-derive after any model change. |
| `HOST` | `127.0.0.1` | Bind address for llama-server. Always localhost — never expose directly. |
| `PORT` | `8081` | Port for llama-server. The manager connects here. |

### `/etc/llama/llama-server-re.env`

Configuration for the `re` slot's llama-server backend (Tesla P40). Same variables and semantics as `/etc/llama/llama-server.env` above, with `PORT=8084` and `CUDA_VISIBLE_DEVICES` set to the P40's UUID. See `config/llama-server-re.env` for the current model choice and how its `CTX_SIZE` was measured.

### `/etc/llama/llama-server-batch.env`

Configuration for the `batch` slot's llama-server backend (Vulkan, on the host's iGPU). `MODEL_PATH`, `CTX_SIZE`, `HOST` and `PORT` (`8083`) as above; `DEVICE` is a Vulkan selector (`Vulkan0`) rather than a CUDA one. See `config/llama-server-batch.env`.

### `/etc/llama/manager.env`

Configuration for the model manager proxy service. The **Default** column is what `manager/config.py` and `manager/slot_config.py` use when a variable is unset; the shipped template `config/manager.env` sets several of these explicitly, and those values are what a `setup.sh` install gets.

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address for the manager. Set to your LAN IP to restrict access, or `0.0.0.0` for all interfaces. |
| `PORT` | `8080` | Port the manager listens on. The template sets `11434`, which is what every example in this README assumes. |
| `LLAMA_SERVER_HOST` / `LLAMA_SERVER_PORT` / `LLAMA_SERVER_ENV` / `LLAMA_SERVER_UNIT` | `127.0.0.1` / `8081` / `/etc/llama/llama-server.env` / `llama-server.service` | The `main` slot's backend address, env file (the manager writes `MODEL_PATH` here during swaps) and systemd unit. |
| `BATCH_SERVER_HOST` / `BATCH_SERVER_PORT` / `BATCH_SERVER_ENV` / `BATCH_SERVER_UNIT` / `BATCH_QUEUE_LIMIT` | `127.0.0.1` / `8083` / `/etc/llama/llama-server-batch.env` / `llama-server-batch.service` / `20` | The same for the `batch` slot. |
| `BATCH_MODEL_DEFAULT` | `gemma-4-E4B-it-Q4_K_M` | Loaded into `ManagerConfig` but not currently read by any other code; the batch slot's model is whatever its env file's `MODEL_PATH` says. |
| `MODELS_DIR` | `/opt/llama/models` | Directory containing GGUF model files. |
| `QUEUE_LIMIT` | `20` | Maximum requests held in the `main` slot's FIFO queue (template: `50`). Requests beyond this get a 503. |
| `SWAP_TIMEOUT` | `120` | Upper bound, in seconds, on waiting for a slot to become healthy after a swap (template: `600`). Health is polled, so a generous value costs nothing on a good load. Exceeding it marks the slot unhealthy. |
| `LOG_FILE` | `/var/log/llama/manager.log` | Log file path for the model manager. |
| `EMBEDDINGS_HOST` / `EMBEDDINGS_PORT` | `127.0.0.1` / `8082` | Where llama-embeddings listens. Always localhost. |
| `COLLECTIONS_CONFIG` | `/etc/llama/collections.json` | Path to collection definitions JSON file. |
| `SKILLS_DB_PATH` | `/opt/llama/data/skills.db` | Path to the SQLite-vec database for document retrieval. |
| `SLOTS` | `main,batch` | Comma-separated slot names the manager fronts, in routing-priority order (template: `main,batch,re`). Every name listed needs a running backend, or that slot reports permanently unhealthy. |
| `SLOT_<NAME>_HOST` / `_PORT` / `_ENV` / `_UNIT` / `_QUEUE_LIMIT` / `_DEVICE` | `127.0.0.1` / _(required)_ / `/etc/llama/llama-server-<name>.env` / `llama-server-<name>.service` / `20` / _(unset)_ | Per-slot configuration, read by `manager/slot_config.py`. For `main` and `batch` a `SLOT_*` variable overrides the legacy `LLAMA_SERVER_*` / `BATCH_*` name; any other slot (e.g. `re`) must set at least `SLOT_<NAME>_PORT` or the manager refuses to start. The template sets `SLOT_RE_PORT=8084`. `SLOT_RE_DEVICE` mirrors the P40's UUID from `CUDA_VISIBLE_DEVICES` in `llama-server-re.env`; `_DEVICE` is not yet read by any code. |

---

## API Endpoints

All endpoints are on `LAN_IP:11434`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI chat completions. Routes by the `model` field to the slot that has it loaded and queues there. 404 if no such GGUF, 409 if it is not loaded on any slot. Supports streaming (`"stream": true`). |
| `POST` | `/v1/messages` | Anthropic Messages API. Byte passthrough to the slot's llama-server, with the same routing, queueing and status codes as chat completions; errors use the Anthropic error envelope. |
| `POST` | `/v1/messages/count_tokens` | Anthropic token counting. Same routing, but bypasses the queue. |
| `POST` | `/swap` | Load a model onto a slot. Body `{"model": ..., "target": <slot>}`; `target` defaults to the first slot in `SLOTS`. Returns when the slot is healthy. |
| `POST` | `/v1/embeddings` | Proxy to llama-embeddings instance. OpenAI-compatible. |
| `GET` | `/v1/models` | Lists available GGUF files as an OpenAI-compatible model list. |
| `GET` | `/status` | Per-slot health, loaded model, and queue depth; GPU VRAM usage; uptime. Re-probes every slot before answering. |
| `GET` | `/health` | Returns `{"status": "ok"}` with HTTP 200. Use for uptime monitors. |
| `GET` | `/collections` | Lists registered document collections with document counts. |
| `POST` | `/collections/{id}/search` | Semantic search within a collection. Returns ranked summaries. |
| `GET` | `/collections/{id}/docs/{doc_id}` | Full document content by ID. |
| `POST` | `/collections/{id}/reindex` | Trigger an incremental reindex. Optional body `{"paths": [...]}` limits scope to specific files; omitted or `null` runs a full scan with stale-deletion. Returns 202 with a job id. |
| `GET` | `/collections/{id}/reindex/status` | Current or most recent reindex job for a collection. 404 if none has run. |
| `GET` | `/collections/{id}/reindex/{job_id}` | State of a specific reindex job (status, stats, error, timestamps). |

### Slot health

There is no single top-level readiness flag. Each entry under `/status`'s `slots` map reports its own state instead:

- `healthy` (bool) — set from the outcome of the slot's most recent probe or swap attempt: `true` after a successful probe or swap, `false` at startup or after a failed one (backend unreachable, bad response, model file not found, or swap timeout). It is not updated while a swap is in progress — a slot that was healthy before a swap started still reports `healthy: true` until that swap resolves.
- `loaded_model` — the model currently loaded on that slot, or `null` if none has loaded successfully yet. It generally lags behind a swap in progress, and is only cleared to `null` when a probe explicitly reports nothing loaded.

A chat request for a model that isn't loaded on any slot returns **409** rather than triggering an implicit swap; load it first via `POST /swap`.

### Client timeout guidance

Model swaps take from 30 seconds to several minutes depending on model size and whether the file is in page cache. `POST /swap` holds the connection open until the slot is healthy or `SWAP_TIMEOUT` expires, so give that call a client timeout longer than `SWAP_TIMEOUT`. Avoid sending chat requests to a slot while you are swapping it.

If you prefer not to wait, poll `/status` before sending requests and check the target slot's `healthy` flag (and `loaded_model`) rather than a top-level state.

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

| Unit | Slot / role |
|---|---|
| `llama-server` | `main` slot |
| `llama-server-batch` | `batch` slot |
| `llama-server-re` | `re` slot |
| `llama-embeddings` | embeddings (CPU) |
| `llama-manager` | the manager on `:11434` |

```bash
# Start everything
sudo systemctl start llama-server llama-server-batch llama-server-re llama-embeddings llama-manager

# Restart just the manager (e.g., after config change)
sudo systemctl restart llama-manager

# Restart one slot (reloads the model named in its env file)
sudo systemctl restart llama-server-re
```

### Check service status

```bash
systemctl status llama-server llama-server-batch llama-server-re llama-embeddings llama-manager
```

### View logs

```bash
# Live log from systemd journal
sudo journalctl -u llama-manager -f

# Each unit appends stdout/stderr to /var/log/llama/<name>.{log,err}.
# llama-server and the manager both log to stderr, so the .err files are
# the ones with content:
tail -f /var/log/llama/llama-server.err
tail -f /var/log/llama/llama-server-re.err
tail -f /var/log/llama/manager.err
```

The manager also tries to write `LOG_FILE` (`/var/log/llama/manager.log`); if `_llama-mgr` cannot create it, it logs a `Could not open log file` warning to `manager.err` and carries on.

Logs rotate weekly via `config/llama-logrotate` (installed to `/etc/logrotate.d/llama`), using `copytruncate` because systemd holds each file open for the life of the process.

### Enable on boot

```bash
sudo systemctl enable llama-server llama-server-batch llama-server-re llama-embeddings llama-manager
```

### Boot sequence

1. System boots, NVIDIA drivers load
2. Each slot unit starts with the model named in its `/etc/llama/llama-server*.env`
3. `llama-manager.service` starts (`After=llama-server.service`), probes every slot in `SLOTS`, and begins accepting requests

If a slot's `MODEL_PATH` is empty or points to a nonexistent file on boot, that llama-server fails to start. The slot reports `"healthy": false` (with `"loaded_model": null`) in `/status`, and requests for it get 409 until you load a model with `POST /swap`.

### Adding the batch and re slots

`scripts/setup.sh` installs only the `main` slot. For each additional slot (`batch` shown; `re` is the same with `-re` names), and see the header comment of each unit file for host prerequisites:

```bash
sudo cp systemd/llama-server-batch.service /etc/systemd/system/
sudo cp config/llama-server-batch.env /etc/llama/llama-server-batch.env
sudo chown _llama-mgr:_llama-mgr /etc/llama/llama-server-batch.env   # the manager rewrites MODEL_PATH on swap
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server-batch
```

Then make sure the slot is in `SLOTS` in `/etc/llama/manager.env`, and that `/etc/sudoers.d/llama-manager` lets `_llama-mgr` restart that unit — `manager/swap.py` runs `sudo systemctl restart <unit>` for whichever slot it swaps, and the entry `setup.sh` writes covers `llama-server.service` only. Edit it with `sudo visudo -f /etc/sudoers.d/llama-manager`. For `re`, also set the P40's UUID in `CUDA_VISIBLE_DEVICES` (see the next section) before starting it.

### Verify each GPU-backed slot is bound to the right physical card

**Why this exists.** `--device CUDA0` names an enumeration position, not a physical card, and CUDA's default enumeration order is not guaranteed to match PCI bus order. Each GPU-backed unit's `CUDA_VISIBLE_DEVICES` (in its `/etc/llama/` env file) pins it to one card by UUID specifically to remove that ambiguity — see the comments in `config/llama-server.env` and `config/llama-server-re.env`, and the device-pinning section of [`docs/superpowers/specs/2026-09-23-dual-gpu-three-slot-design.md`](docs/superpowers/specs/2026-09-23-dual-gpu-three-slot-design.md). The failure mode if the pin is wrong is **silent**: the service starts, loads a model, and serves requests normally, while sitting on the wrong card. Run this procedure whenever a GPU-backed slot is added, after changing any `CUDA_VISIBLE_DEVICES` value, and once after any reboot.

**The invariant this checks:** restarting any GPU-backed slot's service, in any order, with all its cards present, must never place a model on another slot's card.

**How the mapping is read.** `nvidia-smi --query-compute-apps` reports, per running CUDA process, the PID and the **physical card UUID** it is actually using — the invariant's exact question, read directly from the driver rather than inferred from a log line. `ps -o unit=` then maps that PID to the systemd unit that owns it. Together these two commands answer "which physical card is this service on" independently of llama.cpp's version or log verbosity. (An earlier draft of this procedure grepped the log for `ggml_cuda_init`; on this host's current build that string appears only in a CUDA-init *failure* message, never on a successful startup, so the grep silently matched nothing regardless of whether the binding was right or wrong — exactly the failure mode this section exists to catch. Do not resurrect that check.)

This procedure is written generically for however many CUDA-backed slots are active (on this host: `main` on the V100 and `re` on the P40) — substitute the real unit/env names.

1. **Record each card's identity.** With all cards physically present:

   ```bash
   nvidia-smi --query-gpu=index,name,uuid,pci.bus_id --format=csv
   ```

   Write down each row (name, UUID, bus id) somewhere durable — e.g. as a comment in the deployed env file that sets `CUDA_VISIBLE_DEVICES` for that card. The names alone are enough to tell cards apart here (e.g. `Tesla PG500-216` vs `Tesla P40`), but the UUID is what the config actually pins on, so record both.

2. **Confirm each service's actual card, from the driver.** For every GPU-backed unit (e.g. `llama-server`, `llama-server-re`):

   ```bash
   sudo systemctl restart llama-server        # repeat per GPU-backed unit

   # Map each running CUDA process to the physical card it is on:
   nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv

   # Map each PID from that output to the systemd unit that owns it:
   ps -o pid,unit= -p <pid>
   ```

   For each GPU-backed unit, its PID's `gpu_uuid` must be the UUID recorded in step 1 for the card that slot's `CUDA_VISIBLE_DEVICES` names — not the other card's UUID. Seeing the wrong UUID, or no row at all for a unit that should be running, means the pin is wrong or the service isn't actually up; stop and fix the env file before continuing.

   As a secondary, log-based cross-check (not the primary signal — see the note above): `llama_prepare_model_devices: using device CUDA0 (...)` does name the card, but only appears once `-lv` is raised enough to surface it (the exact level has moved before and isn't worth pinning here), which is not something to add to the units permanently just to serve this check. Raise `-lv` on the command temporarily if you want a second line of evidence.

3. **Restart in reverse order and recheck.** Order-dependence is exactly the failure this guards against — a stale CUDA enumeration cached by one service should not leak into another's.

   ```bash
   sudo systemctl restart llama-server-re     # reverse of the order used in step 2
   sudo systemctl restart llama-server
   ```

   Repeat the `nvidia-smi --query-compute-apps` + `ps -o unit=` check from step 2 for each unit. Every service must still map to the same card UUID it did before.

4. **Reboot and recheck.** CUDA enumeration order can differ across a cold boot even when it held across warm restarts.

   ```bash
   sudo reboot
   # after it comes back up:
   nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv
   ps -o pid,unit= -p <pid>   # for each pid from the query above
   ```

   Confirm each unit still maps to the card UUID recorded for it in step 1.

If any step shows a service on the wrong card, the fix is in the relevant `/etc/llama/llama-server*.env`'s `CUDA_VISIBLE_DEVICES` — never in `DEVICE`, which should stay `CUDA0` on every GPU-backed slot.

---

## File System Layout

```
/opt/llama/
  ├── bin/
  │   └── llama-server              # compiled llama.cpp binary (CUDA + Vulkan)
  ├── models/                       # GGUF storage
  ├── data/
  │   └── skills.db                 # SQLite-vec database for collections
  └── manager/                      # model manager Python app (from manager/)
      ├── venv/                     # isolated virtualenv
      ├── DEPLOYED_FROM             # commit stamp written by scripts/deploy-manager.sh
      └── *.py, requirements.txt

/etc/llama/
  ├── manager.env                   # runtime config for the model manager
  ├── llama-server.env              # main slot   (manager rewrites MODEL_PATH on swap)
  ├── llama-server-batch.env        # batch slot  (installed by hand)
  ├── llama-server-re.env           # re slot     (installed by hand)
  └── collections.json              # collection definitions for document retrieval

/var/log/llama/                     # <unit>.log / <unit>.err per service, rotated weekly

/etc/systemd/system/
  ├── llama-manager.service
  ├── llama-server.service          # main
  ├── llama-server-batch.service    # batch (installed by hand)
  ├── llama-server-re.service       # re    (installed by hand)
  └── llama-embeddings.service
```

### This repository

```
inference-server/
├── README.md                       # this file
├── manager/                        # model manager source (deployed to /opt/llama/manager/)
│   ├── app.py                      # FastAPI app: endpoints, proxying, per-slot queue consumers
│   ├── config.py, slot_config.py   # env-var configuration; slot list and per-slot settings
│   ├── slots.py, routing.py        # per-slot state; model → slot resolution
│   ├── names.py                    # model-name normalization
│   ├── queue.py, swap.py           # FIFO queue; env rewrite + systemd restart + health poll
│   ├── gpu.py                      # GPU info via nvidia-smi
│   ├── embeddings.py, vectordb.py, collections.py, reindex_jobs.py   # retrieval
│   └── README.md                   # manager component documentation
├── systemd/                        # unit files (copied to /etc/systemd/system/)
├── config/                         # config templates (copied to /etc/llama/) + logrotate
├── scripts/
│   ├── setup.sh                    # first install: users, dirs, permissions, sudoers, main-slot units
│   ├── deploy-manager.sh           # ship manager code to an existing install (--check is read-only)
│   └── download-model.sh           # GGUF download helper
├── docs/superpowers/               # design specs and implementation plans
└── tests/                          # manager test suite: pip install -r manager/requirements.txt, then pytest tests/
```

---

## Security Notes

- **No authentication.** This is intentional for an internal network. Use Tailscale for remote access (encrypted, authenticated at the network layer).
- **Every llama-server is localhost-only.** No slot backend is exposed to the network. Only the manager can reach them.
- **Dedicated system users.** `_llama` and `_llama-mgr` have no shell, no home directory, and minimal permissions. If a service were compromised, access is tightly scoped.
- **Narrow sudoers.** `_llama-mgr` can only run `systemctl restart` on the slot units. `setup.sh` writes the entry for `llama-server.service`; each additional slot's unit is added by hand (see [Adding the batch and re slots](#adding-the-batch-and-re-slots)). Nothing else.
- **llama-embeddings is localhost-only.** The embedding server on port 8082 is never exposed to the network.
- **No TLS.** Acceptable on a trusted LAN or Tailscale tunnel. Do not expose port 11434 directly to the internet.
