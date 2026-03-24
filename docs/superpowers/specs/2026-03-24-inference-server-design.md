# Inference Server Design Spec

## Overview

Migration from ollama-in-Docker to a native llama.cpp inference server on Ubuntu Server (headless) with an NVIDIA Tesla P40. A lightweight Python model manager sits in front of llama-server, providing OpenAI-compatible API access, API-driven model switching, request queuing, and status reporting.

## Goals

- Run llama.cpp natively for direct GPU access and access to any GGUF model on HuggingFace
- API-driven model hot-swapping with FIFO request queuing during swaps
- OpenAI-compatible API so existing clients work without changes
- Bind to LAN IP for access from workstation, Pi 5 assistant, and Tailscale
- Well-documented, well-commented code throughout for learning purposes
- Clean privilege separation with dedicated system users

## Non-Goals

- Authentication / TLS (internal network, Tailscale for remote)
- Session or conversation history (handled by clients)
- Model aliasing or routing tables (filenames are model names)
- Multiple simultaneous models (single P40, one model at a time)
- Concurrent inference (`--parallel`) — single GPU, serial processing maximizes per-request throughput
- `POST /v1/completions` (non-chat completions) — out of scope for now, can be added later if needed

## Hardware

- **OS:** Ubuntu Server (headless)
- **GPU:** NVIDIA Tesla P40 (24GB VRAM)
- **RAM:** 32GB system memory (available for layer spillover)

## Clients

- Workstation (LAN + Tailscale)
- Pi 5 AI assistant (LAN + Tailscale)
- Multi-agent workflow running locally on the server
- Blender-based workflow (future, from workstation)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Ubuntu Server (Headless)                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Model Manager (Python/FastAPI)          LAN_IP:8080     │   │
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
│  │  /opt/llama/models/    (GGUF storage, read-only for _llama) │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### llama-server (Inference Backend)

**Binary:** llama.cpp's `llama-server`, compiled with CUDA support for the P40.

**Systemd unit:** `llama-server.service`
- `Type=simple` (long-running process)
- Runs as dedicated system user `_llama` (no shell, no home directory)
- Binds to `127.0.0.1:8081` — never directly exposed to the network
- `Restart=on-failure`, `RestartSec=5`
- `After=network-online.target nvidia-persistenced.service`
- Reads runtime config from `/etc/llama/llama-server.env`

**Environment file** (`/etc/llama/llama-server.env`):
- `MODEL_PATH` — path to the currently loaded GGUF
- `N_GPU_LAYERS` — layers to offload to GPU (default: auto-detect based on VRAM)
- `CTX_SIZE` — context window size
- `HOST=127.0.0.1`
- `PORT=8081`

**GPU offloading:** Auto-detects available VRAM and calculates appropriate `--n-gpu-layers`. If a model exceeds 24GB VRAM, remaining layers spill to 32GB system RAM (CPU inference for those layers — slower but functional).

### Model Manager (API Proxy + Orchestrator)

**Runtime:** Python FastAPI application.

**Systemd unit:** `llama-manager.service`
- `Type=simple`
- Runs as dedicated system user `_llama-mgr`
- Binds to `LAN_IP:8080`
- `Restart=on-failure`, `RestartSec=5`
- `After=llama-server.service`, `Wants=llama-server.service`
- Reads config from `/etc/llama/manager.env`

**Privileges:**
- `/etc/llama/llama-server.env` is owned by `_llama-mgr:_llama-mgr` (mode `644`) so the manager can update it directly via filesystem write
- Narrow sudoers entry for `_llama-mgr`: `_llama-mgr ALL=(root) NOPASSWD: /usr/bin/systemctl restart llama-server.service`
- Cannot touch models or anything else

**Responsibilities:**

1. **Request proxying** — Receives OpenAI-compatible requests, forwards to llama-server on `localhost:8081`, streams SSE responses back to the client.

2. **Model switching** — Reads the `model` field from incoming requests. If it differs from the currently loaded model:
   - Resolves model name to a GGUF path in `/opt/llama/models/`
   - Sets state to `"swapping"`
   - Updates `/etc/llama/llama-server.env` with the new `MODEL_PATH`
   - Restarts llama-server via systemd
   - Polls llama-server health endpoint until ready (timeout: 120 seconds)
   - If health poll times out: sets state to `"error"`, drains queue with 503 responses, logs failure
   - On success: sets state to `"ready"`
   - Processes queued requests

3. **Request queuing** — FIFO queue for all incoming requests:
   - Requests are always queued, processed strictly in FIFO order
   - During model swaps, requests accumulate in the queue
   - Max queue depth: 20 (configurable in `manager.env`)
   - Over limit: responds with 503 + `Retry-After` header
   - One request at a time forwarded to llama-server (serial processing, no `--parallel`)
   - **Mixed-model queue policy:** Strict FIFO. If queued requests reference different models, the server swaps as needed when it reaches each request. To prevent thrashing, consecutive same-model requests are batched — the server only swaps when the model actually changes between adjacent queue entries. Clients should use `/status` to check the current model and avoid unnecessary swap triggers.

4. **Model listing** — `GET /v1/models` scans `/opt/llama/models/` and returns available GGUFs as an OpenAI-compatible model list.

5. **Status reporting** — `GET /status` returns:
   ```json
   {
     "state": "ready | loading | swapping | error",
     "current_model": "qwen3.5-35b-a3b-q4_k_m",
     "loading_model": null,
     "error_message": null,
     "uptime_seconds": 3421,
     "queue_depth": 2,
     "queue_limit": 20,
     "gpu": {
       "name": "Tesla P40",
       "vram_total_mb": 24576,
       "vram_used_mb": 18200
     }
   }
   ```

   **State definitions:**
   - `loading` — llama-server is starting up with its configured model on boot (no model was previously loaded)
   - `ready` — accepting and processing requests
   - `swapping` — the manager is actively changing models at runtime
   - `error` — llama-server failed to start or health poll timed out. `error_message` contains details. Recovery: send a new request with a valid model to trigger a fresh swap attempt.

6. **Health check** — `GET /health` returns 200 OK for uptime monitors.

**Model name resolution:** Filenames are model names. A file `qwen3.5-35b-a3b-q4_k_m.gguf` is requested as model `qwen3.5-35b-a3b-q4_k_m`. No alias table — what's in the folder is what's available.

---

## API Surface

All endpoints on `LAN_IP:8080`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completions (streaming SSE or non-streaming). Reads `model` field, swaps if needed, queues if busy. |
| `GET` | `/v1/models` | Lists available GGUFs as OpenAI-compatible model list |
| `GET` | `/status` | Server state, current model, queue depth, GPU info |
| `GET` | `/health` | Simple 200 OK for uptime monitors |

### Request Flow (`/v1/chat/completions`)

```
Request arrives
    │
    ▼
Is model the same as currently loaded?
    ├─ YES → add to FIFO queue → process when turn comes → stream response
    │
    └─ NO → set state to "swapping"
            │
            ▼
         Queue this request + any new arrivals
            │
            ▼
         Update env file, restart llama-server
            │
            ▼
         Poll health until ready
            │
            ▼
         Set state to "ready"
            │
            ▼
         Process queue in order → stream responses
```

---

## File System Layout

```
/opt/llama/
  ├── bin/
  │   └── llama-server              # compiled llama.cpp binary (CUDA)
  ├── models/                       # GGUF storage, owner: _llama:llama, mode: 775
  │   ├── qwen3.5-35b-a3b-q4_k_m.gguf
  │   └── ...
  └── manager/                      # model manager Python app
      ├── venv/                     # isolated virtualenv
      ├── manager.py                # main FastAPI application
      ├── requirements.txt
      └── README.md

/etc/llama/
  ├── llama-server.env              # runtime config for llama-server
  └── manager.env                   # runtime config for model manager

/var/log/llama/
  ├── llama-server.log              # inference server logs
  └── manager.log                   # model manager logs

/etc/systemd/system/
  ├── llama-server.service
  └── llama-manager.service
```

---

## Systemd Integration

### llama-server.service

- `Type=simple`
- `User=_llama`
- `Restart=on-failure`, `RestartSec=5`
- `After=network-online.target nvidia-persistenced.service`
- `EnvironmentFile=/etc/llama/llama-server.env`

### llama-manager.service

- `Type=simple`
- `User=_llama-mgr`
- `Restart=on-failure`, `RestartSec=5`
- `After=llama-server.service`
- `Wants=llama-server.service`
- `EnvironmentFile=/etc/llama/manager.env`

### Boot Sequence

1. System boots → NVIDIA drivers load
2. `llama-server.service` starts with last configured model
3. `llama-manager.service` starts, connects to llama-server, begins accepting requests

**First boot / missing model:** If `MODEL_PATH` in the env file is empty or points to a nonexistent file, llama-server will fail to start. The model manager handles this gracefully — it starts in `"error"` state with no model loaded and accepts requests normally. The first request with a valid model name triggers a swap attempt, which loads the model and transitions to `"ready"`.

### User Privileges

| User | Purpose | Permissions |
|---|---|---|
| `_llama` | Runs llama-server | Read-only model dir, GPU access, no shell, no home |
| `_llama-mgr` | Runs model manager | Sudoers: `systemctl restart llama-server`, write `/etc/llama/llama-server.env`. No model access, no shell, no home |

---

## Logging

- **llama-server:** Logs via systemd journal (`journalctl -u llama-server`). Additionally, `StandardOutput=append:/var/log/llama/llama-server.log` and `StandardError=append:/var/log/llama/llama-server.err` in the unit file for persistent file logs.
- **Model manager:** Application-level logging via Python `logging` module to `/var/log/llama/manager.log`. Logs model swaps, queue state changes, errors, and request metadata (model requested, response time).
- **Log rotation:** Logrotate config at `/etc/logrotate.d/llama` — rotate weekly, keep 4 weeks, compress old logs.

## Network Configuration

The `HOST` value in `manager.env` is the LAN IP to bind to (e.g., `HOST=192.168.1.100`). This is a static configuration value set during initial setup. If the server's IP changes, update `manager.env` and restart the manager service.

Alternatively, set `HOST=0.0.0.0` to bind to all interfaces if IP stability is a concern. Since there's no auth, binding to all interfaces is acceptable on a trusted internal network.

## Client Timeout Behavior During Model Swaps

Model swaps can take 30-60+ seconds. When a request triggers a swap:
- The client connection stays open — the manager holds it while the swap completes, then forwards the request and streams the response.
- Clients should set HTTP timeouts to at least 120 seconds (matching the server-side health poll timeout) to avoid timing out during swaps.
- Clients that prefer not to wait can poll `/status` before sending requests and only send when `state` is `"ready"`.

## Model Management

Models are managed manually by an admin user (not by the service users):
- Download GGUFs to `/opt/llama/models/` via `sudo` or as a user in the `llama` group
- `/opt/llama/models/` is owned by `_llama:llama`, mode `775` — `_llama` can read, admin users in `llama` group can write
- New models are immediately available via `GET /v1/models` (directory scan)
- No restart required to pick up new model files

---

## Security Properties

| Layer | Mechanism |
|---|---|
| **Network** | llama-server on localhost only; model manager on LAN IP only |
| **Process isolation** | Dedicated system users with no shell/home |
| **Privilege separation** | `_llama-mgr` can only restart llama-server and update its env file |
| **Model storage** | Read-only directory for `_llama` user |
| **Remote access** | Tailscale for off-network access (encrypted, authenticated at the network layer) |

---

## Documentation Requirements

- All code must be well-commented explaining the "why" behind decisions
- Each component gets its own README explaining what it does, how to configure it, and how it connects to other components
- Setup scripts should be self-documenting with explanatory comments
- Configuration files should include comments describing each option
