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
│  │   Retrieval      POST /collections/*                           │  │
│  │                                                                │  │
│  │   routes by model → one FIFO queue per slot                    │  │
│  └──────┬─────────────────────────────────────┬───────────┬───────┘  │
│         │                                     │           │          │
│  ┌──────┴───────┐                    ┌────────┴─────┐  ┌──┴───────┐  │
│  │ slot: main   │                    │ slot: egpu   │  │embeddings│  │
│  │ :8081        │                    │ :8084        │  │ :8082    │  │
│  │ V100 32GB    │                    │ P40 24GB     │  │ CPU only │  │
│  │ llama-server │                    │ llama-server │  │          │  │
│  │ .service     │                    │ -egpu        │  │          │  │
│  └──────┬───────┘                    └────────┬─────┘  └──┬───────┘  │
│         └────────────────────────────┬────────┴───────────┘          │
│                     user: _llama     │  all bound to 127.0.0.1       │
│  ┌───────────────────────────────────┴────────────────────────────┐  │
│  │  /opt/llama/models/    (GGUF storage)                          │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

Which slots exist, and in what routing order, is set by `SLOTS` in `/etc/llama/manager.env` (see [Configuration Reference](#etcllamamanagerenv)). The hardware labels above describe this host; each GPU slot is pinned to its card by UUID, not by ordinal.

### Why this design?

**Two-layer architecture.** Clients talk only to the model manager on port 11434. The manager talks to each slot's llama-server on localhost (`:8081` main, `:8084` egpu). No llama-server is directly exposed to the network. This separation means:

- The manager can restart a slot's llama-server for a model swap while other slots keep serving, and clients never need to know which backend holds which model.
- Privilege separation: llama-server only needs to read model files and access its GPU. The manager only needs to restart the slot units and update their env files. Neither service needs broad system access.

**llama.cpp instead of Ollama.** Running llama-server natively (not in Docker) gives direct GPU access, no container overhead, and access to any GGUF on HuggingFace without waiting for Ollama to support it. The tradeoff is more setup — this repo contains the setup scripts and config templates to make it repeatable.

**FIFO queue per slot instead of parallel inference.** Each slot runs one request at a time, so its GPU is fully dedicated to that request; parallel inference would split VRAM (and context) across requests and slow each one down. Each slot has its own queue, so a long job on `egpu` never waits behind `main`. Requests to a slot are processed in order, even during a swap on that slot.

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
- `/etc/llama/` (config files)
- A narrow sudoers entry so `_llama-mgr` can restart `llama-server.service` and `llama-server-egpu.service`
- The `llama-server`, `llama-manager` and `llama-embeddings` unit files (installed, not enabled — step 6 starts them)

`setup.sh` covers the `main` slot only, but writes the sudoers entries for both `llama-server.service` and `llama-server-egpu.service`. The `egpu` slot's unit and env file are installed by hand — see [Adding the egpu slot](#adding-the-egpu-slot). The template `config/manager.env` lists both in `SLOTS`; drop `egpu` if you have not installed it, or the manager reports it as permanently unhealthy.

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

Chat requests never trigger a swap: a request for a model that is not loaded on any slot gets **409**. Load it explicitly with `POST /swap`, naming the slot as `target`. Without a `target`, the model goes to its profile's `home` slot (every profile names exactly one). A swap is a FIFO item in that slot's queue: the call returns **202** as soon as the swap is queued, and the slot's consumer runs it after whatever is already ahead of it — requests in flight and queued on that slot run first, and requests queued behind the swap wait for it to finish (a large cold load takes 30 seconds to several minutes). It returns **200** only when nothing is ahead: the slot is idle (healthy, no request in flight, not mid-swap), its queue is empty, the model is already loaded, and no trial was requested. A queue at its limit refuses the swap **503** `queue_full`. Poll `GET /status` until the target slot's `swapping` is `false` and its `loaded_model` is the new model (`healthy` tells you whether it came up, `last_swap` what the latest finished swap ended with).

```bash
curl http://YOUR_LAN_IP:11434/swap \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.6-35B-A3B-UD-IQ4_NL_XL"}'
# → 202, {"slot": "egpu", "model": "Qwen3.6-35B-A3B-UD-IQ4_NL_XL", "status": "swapping"}
#   (or 200 with "status": "ok" if nothing is ahead — the slot idle, its
#    queue empty, the model already loaded there, and no trial requested)
```

A model may only be swapped to a slot its profile names, and only once that pair has been verified; `"trial": true` relaxes the verification for one swap. That is how `scripts/onboard.py` verifies a new pair — and a trial reloads the model even when it is already loaded, so it really exercises the load. Errors: 400 for a missing `model`, a non-boolean `trial` (`invalid_trial`) or an unknown `target`; 404 `model_not_found` if no profile exists for the model (the message names the `config/models/<model>.toml` to add); 400 `slot_not_allowed` if the profile has no section for the target slot; 400 `not_verified` if the pair is not verified and no trial was requested. A swap that starts returns 202 even if it later fails — a failed swap never surfaces as an HTTP error. When it fails, the slot makes one fallback swap to the model it last served healthily — and a fallback is to a different model, so when the known-good model is the one that just failed there is no fallback to make (reloading it would double the outage for nothing; a `POST /swap` retries if wanted). If the fallback lands, the slot is healthy again on that model and its `last_swap` outcome is `fell_back`. If the fallback fails too, or there is no fallback to make, the slot is `failed`: every request queued on it is answered at once with 503 `slot_failed`, and requests while it stays failed get the same 503 (error `type` `slot_failed` in the OpenAI shape, `api_error` in the Anthropic envelope), naming the fault that failed the slot. A `POST /swap` — or a healthy probe — recovers a failed slot; queued swaps on it still run.

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
      "queue_limit": 20,
      "models": ["qwen2.5-7b-instruct-q4_k_m"],
      "state": "idle",
      "in_flight": 0,
      "in_flight_age_seconds": null,
      "oldest_queue_wait_seconds": null,
      "last_error": null
    },
    "egpu": {
      "host": "127.0.0.1",
      "port": 8084,
      "loaded_model": "Qwen3.6-35B-A3B-UD-IQ4_NL_XL",
      "healthy": true,
      "last_swap_utc": "2026-10-04T09:12:40+00:00",
      "queue_depth": 1,
      "queue_limit": 20,
      "models": ["Qwen3.6-35B-A3B-UD-IQ4_NL_XL"],
      "state": "serving",
      "in_flight": 1,
      "in_flight_age_seconds": 12.4,
      "oldest_queue_wait_seconds": 3.1,
      "last_error": null
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

Per slot (`slots.<name>`):

| Field | Meaning |
|---|---|
| `state` | What the slot is doing right now, most pressing first: `swapping` (a swap is running), then `unhealthy` (the slot is not healthy — see `healthy`), then `failed` (its last swap failed and could not fall back — see `last_error` and `last_swap`), then `probing` (a probe is running), then `serving` (a request is in flight), else `idle`. |
| `in_flight` | Requests the slot is currently serving or relaying: at most one, because the consumer takes the next queued request — a queued swap included — only after a relayed stream has closed, or after the start timeout closed one nobody started. |
| `in_flight_age_seconds` | Age in seconds (rounded to 1 decimal) of the oldest in-flight request; `null` when none. |
| `oldest_queue_wait_seconds` | How long in seconds (rounded to 1 decimal) the request at the head of the queue has been waiting; `null` when the queue is empty. |
| `last_error` | The last slot fault, as `{"code", "message", "at_utc"}` with `code` one of `backend_error` (the backend answered 5xx), `upstream_timeout`, `proxy_error`, `queue_error`, `swap_failed` or `probe_failed`. Set only by slot faults — never by client errors — and never cleared: a later success leaves it in place, so it is the last thing that went wrong, not necessarily something happening now. `null` until a fault. |
| `host`, `port` | The slot's backend address. |
| `loaded_model` | What the backend reports loaded; `null` when nothing is. |
| `healthy` | Whether the slot is healthy: `true` while its state is `idle` or `serving`, `false` while `unhealthy`, `probing`, `swapping` or `failed` — so it drops the moment a swap starts and returns when the swap lands (or its fallback does). A successful probe or swap makes it `true`; a backend 5xx, a proxy error, a failed probe, or a swap that fails without a fallback make it `false`. |
| `swapping` | Whether the slot is in the `swapping` state: `true` from the moment its consumer starts a queued swap until that swap lands or fails. |
| `last_swap_utc` | ISO-8601 UTC time of the slot's last successful swap; `null` if it has never succeeded. |
| `last_swap` | The swap the slot's consumer most recently finished, as `{"id", "model", "trial", "outcome", "message", "at_utc"}`; `outcome` is `ok`, `already_loaded` (the model was loaded by the time the item ran, so nothing swapped), `fell_back` (the swap failed but the slot recovered to the model it last served healthily), or `failed` (the swap failed and the fallback failed too, or there was no known-good model; the slot is now `failed`). `message` is the failure text, `null` otherwise. `null` until the slot's first finished swap. |
| `queue_depth`, `queue_limit` | Requests waiting / the queue's capacity. |
| `models` | The onboarded models whose profile names this slot as their `home`. |

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

Models are GGUF files stored in `/opt/llama/models/`; the filename without `.gguf` is the model name used in API requests. The manager's roster, however, is the set of TOML profiles in `config/models/` (deployed to `/etc/llama/models.d/`): one `config/models/<model>.toml` per model, named for the GGUF's stem. A GGUF on disk without a profile is invisible to the manager. A profile carries the model's `ctx_size`, `n_gpu_layers`, sampling values, extra `flags`, and the slots it may run on.

### Download a model

```bash
# <huggingface-repo> <filename-in-repo>
./scripts/download-model.sh \
  bartowski/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q5_K_M.gguf
```

The file keeps its upstream name, and that stem is the model name clients use. Rename it afterwards if you want a shorter one.

### Add a model to the roster

1. Download the GGUF (above); its stem becomes the model name.
2. Add `config/models/<stem>.toml` — copy an existing profile and change the values. The shipped profiles' comments record what each key means and how the value was measured.
3. Commit and deploy: `scripts/deploy-manager.sh` validates every profile and refuses to ship one that fails (exit 5).
4. Restart the manager, which reads the profiles at startup. The model is now known but verified on no slot, so `POST /swap` refuses it until step 5.
5. Verify it with `scripts/onboard.py <model> --slot <slot>`: a trial load on that slot, a test chat, a swap back to the previous model, and — only if the trial's own `last_swap` entry says `ok` — a `[slots.<slot>.verified]` block appended to the profile. Each swap decides itself from the `last_swap` entry carrying the request id its `POST /swap` sent, so an entry with any other outcome fails the onboarding at once — including `fell_back`, where the slot ends healthy on the previous model and the old health-based check would have read it as a pass. Commit the block and deploy again; afterwards `/swap` accepts the pair without a trial.

### List available models

```bash
curl http://YOUR_LAN_IP:11434/v1/models
```

`GET /v1/models` lists the onboarded profiles — the models with at least one verified (model, slot) pair — in OpenAI-compatible format, with `home_slot`, `verified_slots` and `loaded_on` (the slots whose `loaded_model` matches) alongside the standard fields. It does not scan the models directory: a profile appears only after `scripts/onboard.py` has verified it on at least one slot, and the profiles are read once at manager startup, so a new or edited profile needs a deploy and a manager restart to show up.

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

Static configuration for the `main` slot's llama-server backend. Model-specific settings do not live here: on every swap the manager renders the current model's profile into the generated model env file `/var/lib/llama-manager/main.env` (the whole `MODEL_ARGS` line), which the slot's unit loads **before** this file, so the generated file wins any conflict. The manager never writes this file, so it is safe to edit while the service is running (a restart applies the change).

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `CUDA0` | `--device` selector passed to llama-server. Names an enumeration position, not a physical card — unambiguous only in combination with `CUDA_VISIBLE_DEVICES` below. |
| `CUDA_VISIBLE_DEVICES` | _(host-specific, no safe default)_ | Restricts this process to exactly one physical GPU by UUID (`GPU-<uuid>`), so `DEVICE=CUDA0` above always names the same card regardless of enumeration order. Get the UUID with `nvidia-smi --query-gpu=uuid,pci.bus_id,name --format=csv`; never use a bare ordinal (`0`) here, since that reintroduces the same enumeration-order ambiguity. See the device-pinning section of [`docs/superpowers/specs/2026-09-23-dual-gpu-three-slot-design.md`](docs/superpowers/specs/2026-09-23-dual-gpu-three-slot-design.md). |
| `GGML_CUDA_DISABLE_GRAPHS` | `1` | Works around a CUDA-graph memory leak; see the comment in `config/llama-server.env`. |
| `HOST` | `127.0.0.1` | Bind address for llama-server. Always localhost — never expose directly. |
| `PORT` | `8081` | Port for llama-server. The manager connects here. |

### `config/models/` (the profiles)

One TOML profile per model, named for the GGUF's stem: `config/models/<model>.toml`, deployed to `/etc/llama/models.d/`. The profiles are the model roster: the manager reads them at startup, validates them, and renders the target slot's section into the generated model env file on every swap. A profile that fails validation is left out of the roster, logged at ERROR, and reported in `GET /status`'s top-level `profile_errors`.

```toml
home = "main"                    # required; the slot an untargeted /swap uses

[slots.main]                     # one section per slot this model may run on
ctx_size = 131072                # required, int > 0
n_gpu_layers = -1                # required, int >= -1 (-1 = all)
flags = ["--jinja", ...]         # optional; one token per list item, in order

[slots.main.sampling]            # required; exactly these seven keys
temp = 1.0
top_k = 20
top_p = 0.95
min_p = 0.0
presence_penalty = 0.0
frequency_penalty = 0.0
repeat_penalty = 1.0

[slots.main.verified]            # added by scripts/onboard.py after a trial load
date = "2026-11-01"
llama_cpp_build = 11136
llama_cpp_commit = "057494f93"
```

Every key of `[slots.<slot>]` renders into the `MODEL_ARGS` line of the generated file, so a flag the profile does not mention is absent — never inherited from the previous model. A `flags` item may not repeat a flag the slot or the renderer sets itself (device, context, GPU layers, sampling), and an optional `[slots.<slot>.env]` table may not name `HOST`, `PORT`, `DEVICE`, `CUDA_VISIBLE_DEVICES`, `MODEL_ARGS` or any `LLAMA_ARG_*` variable. The shipped profiles under `config/models/` are worked examples — their comments record how each value was measured.

### `/etc/llama/llama-server-egpu.env`

Configuration for the `egpu` slot's llama-server backend (Tesla P40). Same variables and semantics as `/etc/llama/llama-server.env` above, with `PORT=8084` and `CUDA_VISIBLE_DEVICES` set to the P40's UUID. See `config/llama-server-egpu.env` for the device pinning; the model choice and how its `ctx_size` was measured live in the model's profile under `config/models/`.

### `/etc/llama/manager.env`

Configuration for the model manager proxy service. The **Default** column is what `manager/config.py` and `manager/slot_config.py` use when a variable is unset; the shipped template `config/manager.env` sets several of these explicitly, and those values are what a `setup.sh` install gets.

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address for the manager. Set to your LAN IP to restrict access, or `0.0.0.0` for all interfaces. |
| `PORT` | `8080` | Port the manager listens on. The template sets `11434`, which is what every example in this README assumes. |
| `LLAMA_SERVER_HOST` / `LLAMA_SERVER_PORT` / `LLAMA_SERVER_ENV` / `LLAMA_SERVER_UNIT` | `127.0.0.1` / `8081` / `/etc/llama/llama-server.env` / `llama-server.service` | The `main` slot's backend address, static env file (the manager never writes it — per-model settings go to the generated model env file) and systemd unit. |
| `MODELS_DIR` | `/opt/llama/models` | Directory containing GGUF model files. |
| `PROFILES_DIR` | `/etc/llama/models.d` | Directory holding the per-model TOML profiles (the roster). `setup.sh` and `deploy-manager.sh` install `config/models/*.toml` here. |
| `MODEL_ENV_DIR` | `/var/lib/llama-manager` | Directory holding the generated per-slot model env files (`<slot>.env`), rewritten by the manager on every swap and loaded by the slot's unit before its static env file. Owned by `_llama-mgr`. |
| `QUEUE_LIMIT` | `20` | Maximum requests held in the `main` slot's FIFO queue (template: `50`). Requests beyond this get a 503. |
| `SWAP_TIMEOUT` | `120` | Upper bound, in seconds, on waiting for a slot to become healthy after a swap (template: `600`). Health is polled, so a generous value costs nothing on a good load. Exceeding it fails the swap and takes the failure tail described in "Switch to a different model". The supervisor uses the same value as its loading grace, in which a 503 is loading, not failed; the supervisor paragraph below gives the details. |
| `RESTART_TIMEOUT` | `150` | Upper bound, in seconds, on a `systemctl restart` of a slot's unit during a swap. Both slot units have `TimeoutStopUSec=1min 30s`, so a restart can legitimately take ~90s before the new process starts; 150s covers that. Exceeding it fails the swap. |
| `PROXY_READ_TIMEOUT` | `1800` | The longest, in seconds, the manager waits for the next bytes from the backend. For a non-streaming request that is effectively the whole generation; for a stream it is the longest silence, such as prefill before the first token. A read or write timeout answers 504 `upstream_timeout` and leaves the slot healthy. |
| `SUPERVISOR_PROBE_INTERVAL` | `15` | Seconds between the per-slot supervisor's probes, while the slot is `idle`, `unhealthy` or `failed`. Must be a finite number of seconds greater than 0. |
| `SUPERVISOR_FAILURES` | `3` | Consecutive failed probes that make a supervisor restart due for an unhealthy slot. Must be at least 1. |
| `SUPERVISOR_BACKOFF` | `30` | Base delay, in seconds, between two due supervisor restarts; it doubles after each due restart and resets to the base on a healthy probe. Must be a finite number of seconds greater than 0. |
| `SUPERVISOR_BACKOFF_MAX` | `600` | Cap, in seconds, on the supervisor's restart backoff. Must be a finite number of seconds greater than 0, and at least `SUPERVISOR_BACKOFF`. |
| `SUPERVISOR_RESTART` | `0` | Exactly `0` or `1`. `1` lets a due restart run `systemctl restart` on the slot's unit; `0` (the default) logs one `would restart` warning per due restart instead and counts it in `/status`. The template ships it commented out; enabling it is a deliberate edit to the deployed env file. |
| `EMBEDDINGS_HOST` / `EMBEDDINGS_PORT` | `127.0.0.1` / `8082` | Where llama-embeddings listens. Always localhost. |
| `COLLECTIONS_CONFIG` | `/etc/llama/collections.json` | Path to collection definitions JSON file. |
| `SKILLS_DB_PATH` | `/opt/llama/data/skills.db` | Path to the SQLite-vec database for document retrieval. |
| `SLOTS` | `main` | Comma-separated slot names the manager fronts, in routing-priority order (template: `main,egpu`). Every name listed needs a running backend, or that slot reports permanently unhealthy. |
| `SLOT_<NAME>_HOST` / `_PORT` / `_ENV` / `_UNIT` / `_QUEUE_LIMIT` / `_DEVICE` | `127.0.0.1` / _(required)_ / `/etc/llama/llama-server-<name>.env` / `llama-server-<name>.service` / `20` / _(unset)_ | Per-slot configuration, read by `manager/slot_config.py`. For `main` a `SLOT_*` variable overrides the legacy `LLAMA_SERVER_*` name; any other slot (e.g. `egpu`) must set at least `SLOT_<NAME>_PORT` or the manager refuses to start. The template sets `SLOT_EGPU_PORT=8084`. `SLOT_EGPU_DEVICE` mirrors the P40's UUID from `CUDA_VISIBLE_DEVICES` in `llama-server-egpu.env`; `_DEVICE` is not yet read by any code. |

---

## API Endpoints

All endpoints are on `LAN_IP:11434`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI chat completions. Routes by the `model` field to the slot that has it loaded and queues there. 404 if no profile exists for the model, 409 if no slot has it loaded. Supports streaming (`"stream": true`). |
| `POST` | `/v1/messages` | Anthropic Messages API. Byte passthrough to the slot's llama-server, with the same routing, queueing and status codes as chat completions; errors use the Anthropic error envelope. |
| `POST` | `/v1/messages/count_tokens` | Anthropic token counting. Same routing, but bypasses the queue. |
| `POST` | `/swap` | Load a model onto a slot. Body `{"model": ..., "target": <slot>, "trial": <bool>}`; `target` defaults to the profile's home slot, and the pair must be verified unless `"trial"` is true (a trial reloads even a loaded model). Returns 200 only when nothing is ahead: the slot is idle, its queue is empty, the model is already loaded, and no trial was requested; otherwise 202 means the swap is queued in the slot's FIFO and 503 `queue_full` means the queue is at its limit — see "Switch to a different model". Poll `GET /status` (`swapping`, `loaded_model`, `healthy`, `last_swap`) for the outcome. Errors: `invalid_trial`, 404 `model_not_found` (no profile), `invalid_target`, `slot_not_allowed`, `not_verified`. |
| `GET` | `/v1/models` | Lists the onboarded profiles — models with at least one verified (model, slot) pair — as an OpenAI-compatible model list, each entry with `home_slot`, `verified_slots` and `loaded_on`. |
| `GET` | `/status` | Per-slot state, health, in-flight and queue-wait ages, last fault, loaded model, supervisor state (`restart_enabled`, `restarts`, `restarts_suppressed`, `consecutive_failures`), home-model list and queue depth; top-level `profile_errors`; GPU VRAM usage; uptime. Before answering, asks each idle, unhealthy or failed slot's consumer to probe it; a slot that is serving or swapping is reported as it stands. |
| `GET` | `/health` | Returns `{"status": "ok"}` with HTTP 200. Use for uptime monitors. |
| `GET` | `/collections` | Lists registered document collections with document counts. |
| `POST` | `/collections/{id}/search` | Semantic search within a collection. Returns ranked summaries. |
| `GET` | `/collections/{id}/docs/{doc_id}` | Full document content by ID. |
| `POST` | `/collections/{id}/reindex` | Trigger an incremental reindex. Optional body `{"paths": [...]}` limits scope to specific files; omitted or `null` runs a full scan with stale-deletion. Returns 202 with a job id. |
| `GET` | `/collections/{id}/reindex/status` | Current or most recent reindex job for a collection. 404 if none has run. |
| `GET` | `/collections/{id}/reindex/{job_id}` | State of a specific reindex job (status, stats, error, timestamps). |

### Slot health

There is no single top-level readiness flag. Each entry under `/status`'s `slots` map reports its own state instead:

- `healthy` (bool) — derived from the slot's state: `true` while the slot is `idle` (including while serving), `false` at startup, while `probing`, while `swapping`, while `failed`, or after a failed probe (backend unreachable, bad response, model file not found). A swap that fails and falls back leaves the slot healthy again on the model it fell back to; one whose fallback fails too — or that has no known-good model — leaves it `failed`.
- `loaded_model` — the model currently loaded on that slot, or `null` if none has loaded successfully yet. It generally lags behind a swap in progress, and is only cleared to `null` when a probe explicitly reports nothing loaded.
- `models` — the models whose profile names this slot as `home` (empty if none): the models an untargeted `/swap` sends to this slot. It comes from the profiles, not from any per-slot setting.
- `supervisor` — the per-slot supervisor's view (Q21): `restart_enabled` (whether `SUPERVISOR_RESTART` is `1`), `restarts` (restarts the consumer actually ran), `restarts_suppressed` (restarts that were due while restarts were off) and `consecutive_failures` (failed probes since the last healthy probe or the last due restart).

Each slot has a supervisor that asks its own consumer to probe the backend every `SUPERVISOR_PROBE_INTERVAL` seconds while the slot is `idle`, `unhealthy` or `failed`. Every probe counts, whoever asked for it: the consumer judges every probe it runs, including the ones a `GET /status` or a routed request asks for through a refresh — a client that polls `/status` faster than the interval makes a restart due sooner than the interval alone would. A failed probe (backend unreachable, any other status, a malformed response, or no usable model) counts against the slot; `SUPERVISOR_FAILURES` consecutive failures make a restart due while the slot is `unhealthy` and the backoff allows it — base `SUPERVISOR_BACKOFF`, doubling after each due restart, capped at `SUPERVISOR_BACKOFF_MAX`, reset to the base by a healthy probe. A backend answering 503 is loading, not failed, for `SWAP_TIMEOUT` seconds after the slot last left `idle`, was restarted, or the manager started, so a slow load is not restarted out from under itself. Restarting is opt-in: with `SUPERVISOR_RESTART=0` (the default) a due restart logs one `would restart` warning and counts in `restarts_suppressed`. A `failed` slot is probed but never restarted — its recovery is a `/swap` or a healthy probe.

A chat request for a model that isn't loaded on any slot returns **409** rather than triggering an implicit swap; load it first via `POST /swap`. A request already queued when its model is swapped out gets the same 409, never a swap.

### Client timeout guidance

Model swaps take from 30 seconds to several minutes depending on model size and whether the file is in page cache. `POST /swap` returns 202 immediately and runs the swap in the background, so it needs no long client timeout — poll `GET /status` until the target slot reports `swapping: false` and its `loaded_model` is the new model, and only send requests there once `healthy` is `true` again. Avoid sending chat requests to a slot while you are swapping it.

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

5. **Restart the manager** — indexing starts automatically in the background, so a restart never blocks inference; `GET /collections/{id}/reindex/status` shows the startup job. After startup indexing finishes, collections no longer listed in `collections.json` are deleted from the index; a missing or empty config deletes nothing.

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

### Architecture

```
Manager (:11434)
├── /v1/chat/completions → llama-server (:8081, GPU)
├── /collections/*/search → query vector from llama-embeddings (:8082, CPU) → SQLite-vec (in-process)
└── /collections/*/docs/* → SQLite-vec (in-process)
```

Documents are indexed in the background on startup, with SHA-256 hash-based
change detection. Only new or modified files are re-embedded.

---

## Service Management

### Start / stop / restart

| Unit | Slot / role |
|---|---|
| `llama-server` | `main` slot |
| `llama-server-egpu` | `egpu` slot |
| `llama-embeddings` | embeddings (CPU) |
| `llama-manager` | the manager on `:11434` |

```bash
# Start everything
sudo systemctl start llama-server llama-server-egpu llama-embeddings llama-manager

# Restart just the manager (e.g., after config change)
sudo systemctl restart llama-manager

# Restart one slot (reloads the model named in its env file)
sudo systemctl restart llama-server-egpu
```

### Check service status

```bash
systemctl status llama-server llama-server-egpu llama-embeddings llama-manager
```

### View logs

Every service logs to the journal only; there are no per-service log files.
Read a unit's logs with `journalctl -u <unit>`, adding `-f` to follow:

```bash
journalctl -u llama-server -f
journalctl -u llama-server-egpu -f
journalctl -u llama-embeddings -f
journalctl -u llama-manager -f
```

The manager writes one structured JSON line per inference request and per
`/swap` on the `manager.requests` logger. Extract just those lines with:

```bash
journalctl -u llama-manager -o cat | sed -n 's/^INFO manager.requests: //p' | jq
```

A request whose client disconnected before it was served is not run, and its line has status 499 and outcome `client_gone`.

### Enable on boot

```bash
sudo systemctl enable llama-server llama-server-egpu llama-embeddings llama-manager
```

### Boot sequence

1. System boots, NVIDIA drivers load
2. Each slot unit starts with the model in its generated model env file (`/var/lib/llama-manager/<slot>.env`), the profile rendering the last swap left there
3. `llama-manager.service` starts (`After=llama-server.service`), probes every slot in `SLOTS`, and begins accepting requests

If a slot's generated model env file is missing on boot, or names a GGUF that is not on disk, that llama-server fails to start. The slot reports `"healthy": false` (with `"loaded_model": null`) in `/status`, and requests for it get 409 until you load a model with `POST /swap`.

### Adding the egpu slot

`scripts/setup.sh` installs only the `main` slot. For the `egpu` slot (any further slot is the same with its own names), and see the header comment of the unit file for host prerequisites:

```bash
sudo cp systemd/llama-server-egpu.service /etc/systemd/system/
sudo cp config/llama-server-egpu.env /etc/llama/llama-server-egpu.env
# The unit loads its generated model env file (an EnvironmentFile= without a
# leading "-") before its static env file and refuses to start without it,
# so render the first one from the model's profile. The manager must be
# deployed first (deploy-manager.sh installs profiles.py to
# /opt/llama/manager/), and outside a repo checkout PYTHONPATH=/opt/llama
# puts the package on the path:
PYTHONPATH=/opt/llama /opt/llama/manager/venv/bin/python -m manager.profiles render \
  /etc/llama/models.d Qwen3.6-35B-A3B-UD-IQ4_NL_XL egpu \
  --slots main,egpu --models-dir /opt/llama/models \
  | sudo tee /var/lib/llama-manager/egpu.env >/dev/null
sudo chown _llama-mgr:_llama-mgr /var/lib/llama-manager/egpu.env
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server-egpu
```

The manager never writes the static env file: on every swap it renders the model's profile into the generated one, and the unit loads that first. (`setup.sh` creates `/var/lib/llama-manager` but does not render `main.env` either — the same command above, with `main`'s model and slot, produces it.)

Then make sure the slot is in `SLOTS` in `/etc/llama/manager.env`, and that `/etc/sudoers.d/llama-manager` lets `_llama-mgr` restart that unit — `manager/swap.py` runs `sudo systemctl restart <unit>` for whichever slot it swaps. `setup.sh` writes the entries for `llama-server.service` and `llama-server-egpu.service`; any further slot's entry is added by hand with `sudo visudo -f /etc/sudoers.d/llama-manager`. For `egpu`, also set the P40's UUID in `CUDA_VISIBLE_DEVICES` (see the next section) before starting it.

### Verify each GPU-backed slot is bound to the right physical card

**Why this exists.** `--device CUDA0` names an enumeration position, not a physical card, and CUDA's default enumeration order is not guaranteed to match PCI bus order. Each GPU-backed unit's `CUDA_VISIBLE_DEVICES` (in its `/etc/llama/` env file) pins it to one card by UUID specifically to remove that ambiguity — see the comments in `config/llama-server.env` and `config/llama-server-egpu.env`, and the device-pinning section of [`docs/superpowers/specs/2026-09-23-dual-gpu-three-slot-design.md`](docs/superpowers/specs/2026-09-23-dual-gpu-three-slot-design.md). The failure mode if the pin is wrong is **silent**: the service starts, loads a model, and serves requests normally, while sitting on the wrong card. Run this procedure whenever a GPU-backed slot is added, after changing any `CUDA_VISIBLE_DEVICES` value, and once after any reboot.

**The invariant this checks:** restarting any GPU-backed slot's service, in any order, with all its cards present, must never place a model on another slot's card.

**How the mapping is read.** `nvidia-smi --query-compute-apps` reports, per running CUDA process, the PID and the **physical card UUID** it is actually using — the invariant's exact question, read directly from the driver rather than inferred from a log line. `ps -o unit=` then maps that PID to the systemd unit that owns it. Together these two commands answer "which physical card is this service on" independently of llama.cpp's version or log verbosity. (An earlier draft of this procedure grepped the log for `ggml_cuda_init`; on this host's current build that string appears only in a CUDA-init *failure* message, never on a successful startup, so the grep silently matched nothing regardless of whether the binding was right or wrong — exactly the failure mode this section exists to catch. Do not resurrect that check.)

This procedure is written generically for however many CUDA-backed slots are active (on this host: `main` on the V100 and `egpu` on the P40) — substitute the real unit/env names.

1. **Record each card's identity.** With all cards physically present:

   ```bash
   nvidia-smi --query-gpu=index,name,uuid,pci.bus_id --format=csv
   ```

   Write down each row (name, UUID, bus id) somewhere durable — e.g. as a comment in the deployed env file that sets `CUDA_VISIBLE_DEVICES` for that card. The names alone are enough to tell cards apart here (e.g. `Tesla PG500-216` vs `Tesla P40`), but the UUID is what the config actually pins on, so record both.

2. **Confirm each service's actual card, from the driver.** For every GPU-backed unit (e.g. `llama-server`, `llama-server-egpu`):

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
   sudo systemctl restart llama-server-egpu   # reverse of the order used in step 2
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
  ├── llama-server.env              # main slot, static config only
  ├── llama-server-egpu.env         # egpu slot, static config only (installed by hand)
  ├── models.d/                     # per-model TOML profiles (roster; from config/models/)
  └── collections.json              # collection definitions for document retrieval

/var/lib/llama-manager/             # generated per-slot model env files (<slot>.env),
                                    # rewritten by the manager on every swap; _llama-mgr, 755

/etc/systemd/system/
  ├── llama-manager.service
  ├── llama-server.service          # main
  ├── llama-server-egpu.service     # egpu  (installed by hand)
  └── llama-embeddings.service
```

### This repository

```
inference-server/
├── README.md                       # this file
├── manager/                        # model manager source (deployed to /opt/llama/manager/)
│   ├── app.py                      # FastAPI app: endpoints, proxying, per-slot queue consumers
│   ├── config.py, slot_config.py   # env-var configuration; slot list and per-slot settings
│   ├── profiles.py                 # profile validation + MODEL_ARGS rendering (also a CLI)
│   ├── slots.py, routing.py        # per-slot state; model → slot resolution
│   ├── names.py                    # model-name normalization
│   ├── queue.py, swap.py           # FIFO queue; model env file write + restart + health poll
│   ├── gpu.py                      # GPU info via nvidia-smi
│   ├── embeddings.py, vectordb.py, collections.py, reindex_jobs.py   # retrieval
│   └── README.md                   # manager component documentation
├── systemd/                        # unit files (copied to /etc/systemd/system/)
├── config/                         # config templates (copied to /etc/llama/)
│   └── models/                     # per-model TOML profiles (installed to /etc/llama/models.d/)
├── scripts/
│   ├── setup.sh                    # first install: users, dirs, permissions, sudoers, main-slot units
│   ├── deploy-manager.sh           # ship manager code and profiles (--check is read-only)
│   ├── onboard.py                  # trial-load a model on a slot, then record the pass
│   └── download-model.sh           # GGUF download helper
├── docs/superpowers/               # design specs and implementation plans
└── tests/                          # manager test suite: pip install -r manager/requirements.txt, then pytest tests/
```

---

## Security Notes

- **No authentication.** This is intentional for an internal network. Use Tailscale for remote access (encrypted, authenticated at the network layer).
- **Every llama-server is localhost-only.** No slot backend is exposed to the network. Only the manager can reach them.
- **Dedicated system users.** `_llama` and `_llama-mgr` have no shell, no home directory, and minimal permissions. If a service were compromised, access is tightly scoped.
- **Narrow sudoers.** `_llama-mgr` can only run `systemctl restart` on the slot units. `setup.sh` writes the entries for `llama-server.service` and `llama-server-egpu.service`; each further slot's entry is added by hand (see [Adding the egpu slot](#adding-the-egpu-slot)). Nothing else.
- **llama-embeddings is localhost-only.** The embedding server on port 8082 is never exposed to the network.
- **No TLS.** Acceptable on a trusted LAN or Tailscale tunnel. Do not expose port 11434 directly to the internet.
