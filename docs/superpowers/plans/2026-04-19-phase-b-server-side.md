# Phase B Server-Side Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second inference backend (AMD Vega iGPU / Vulkan) to the inference server, extend `llama-manager` to route across two slots (`main` / `batch`), and expose an explicit `POST /swap` endpoint so PAL's Phase B client code can exercise the batch backend.

**Architecture:** Phase 0 rebuilds `/opt/llama/bin/llama-server` with both CUDA and Vulkan backends and validates Vulkan-backed inference with Gemma 4 E4B. Phase 1 deploys the new `llama-server-batch.service` alongside the existing unit. Phase 2 refactors the manager to hold two `SlotState` instances (routing, queues, swap locks per slot), adds `POST /swap`, updates `/status` to expose both slots. Phase 3 smoke-tests end-to-end against PAL.

**Tech Stack:** Python 3.12, FastAPI, httpx, pytest, pytest-asyncio, systemd, llama.cpp, Vulkan SDK.

**Reference spec:** `docs/superpowers/specs/2026-04-19-phase-b-server-side-design.md`

---

## Phase 0: Rebuild llama-server (gates all subsequent tasks)

### Task 1: Install Vulkan SDK and verify host prerequisites

**Files:** none (host-level changes).

- [ ] **Step 1: Install Vulkan SDK on the host (agenthost, 192.168.1.14)**

Run on the host as a user with sudo:

```bash
sudo apt update
sudo apt install -y libvulkan-dev vulkan-tools glslang-tools spirv-headers
```

(If `apt install` is not the right package manager for your distro, use the equivalent. The required packages are: Vulkan loader, Vulkan headers, `glslangValidator`, SPIR-V headers.)

- [ ] **Step 2: Verify Vulkan tooling and device enumeration**

```bash
vulkaninfo --summary
```

Expected: output lists the AMD Radeon Vega (or "AMD Raven Ridge" / "RADV RENOIR" depending on Mesa version) as a Vulkan device. Exact GPU naming depends on Mesa driver version.

If `vulkaninfo` fails with no devices: install Mesa Vulkan drivers (`mesa-vulkan-drivers` on Debian/Ubuntu) and retry.

- [ ] **Step 3: Ensure `_llama` user can access render nodes**

```bash
groups _llama
```

Expected: output includes `video` and `render`. If not:

```bash
sudo usermod -a -G video,render _llama
```

(Group changes take effect on next process start. Since llama-server is restarted by systemd during swaps, next swap will pick them up. Force immediate effect with `sudo systemctl restart llama-server` if needed.)

- [ ] **Step 4: Verify as `_llama` user**

```bash
sudo -u _llama vulkaninfo --summary | head -30
```

Expected: same output as Step 2. If "permission denied" on `/dev/dri/renderD128`, Step 3 didn't take effect — try `sudo -u _llama -i` in a fresh login shell.

- [ ] **Step 5: No commit**

This task produces host-level changes only. Proceed to Task 2.

---

### Task 2: Rebuild llama.cpp binary with CUDA + Vulkan

**Files:**
- Create: `scripts/build-llama.sh` (new, convenience wrapper documenting the build)

- [ ] **Step 1: Locate the llama.cpp source tree**

Check the existing setup to find how the current binary was built. Likely at `~/src/llama.cpp` or `/opt/llama/src/llama.cpp`. Confirm with:

```bash
ls /opt/llama/bin/llama-server
/opt/llama/bin/llama-server --version
```

The version string identifies the commit. Find or clone the matching source:

```bash
# If existing source tree:
cd <llama.cpp source dir>
git status

# If no existing tree, fresh clone:
git clone https://github.com/ggerganov/llama.cpp.git ~/src/llama.cpp
cd ~/src/llama.cpp
git checkout <tag matching current binary>  # use --version output
```

- [ ] **Step 2: Write the build script**

Create `scripts/build-llama.sh` in the inference_server repo:

```bash
#!/usr/bin/env bash
# scripts/build-llama.sh
#
# Rebuild llama-server with both CUDA and Vulkan backends enabled.
# Run from the llama.cpp source tree root.
#
# Prerequisites:
#   - CUDA toolkit installed (existing requirement for Tesla P40)
#   - Vulkan SDK installed (libvulkan-dev, glslang-tools, spirv-headers)
#   - _llama user in 'video' and 'render' groups
#
# Usage:
#   cd /path/to/llama.cpp
#   bash /path/to/inference_server/scripts/build-llama.sh
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
INSTALL_TARGET="${INSTALL_TARGET:-/opt/llama/bin/llama-server}"

cmake -B "$BUILD_DIR" -DGGML_CUDA=ON -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --config Release --target llama-server -j

echo "Build complete: $BUILD_DIR/bin/llama-server"
echo "Install with:   sudo install -m 0755 $BUILD_DIR/bin/llama-server $INSTALL_TARGET"
echo "Then restart:   sudo systemctl restart llama-server.service"
```

- [ ] **Step 3: Make it executable**

```bash
chmod +x scripts/build-llama.sh
```

- [ ] **Step 4: Run the build**

```bash
cd <llama.cpp source dir>
bash /path/to/inference_server/scripts/build-llama.sh
```

Expected: successful build, `build/bin/llama-server` exists. On failure, CMake will typically say what's missing (Vulkan headers, CUDA toolkit, etc.). Fix and re-run.

- [ ] **Step 5: Install and restart main service**

```bash
sudo install -m 0755 build/bin/llama-server /opt/llama/bin/llama-server
sudo systemctl restart llama-server.service
sudo systemctl status llama-server.service
```

Expected: llama-server.service is `active (running)` after restart.

- [ ] **Step 6: Verify main slot still works**

```bash
curl -s http://127.0.0.1:8081/v1/models | head
```

Expected: 200, returns the currently loaded main model.

- [ ] **Step 7: Verify both device backends compiled in**

```bash
/opt/llama/bin/llama-server --help 2>&1 | grep -iE 'cuda|vulkan|device'
```

Expected: output mentions both CUDA and Vulkan device options. The exact flag names depend on llama.cpp version — common shapes are `--device CUDA0 / Vulkan0` or `--split-mode` + `--tensor-split`. Note the actual flags in a comment in the build script for the next task.

- [ ] **Step 8: Commit the build script**

```bash
git add scripts/build-llama.sh
git commit -m "build: CUDA+Vulkan build script for llama-server"
```

---

### Task 3: Manually validate Vulkan backend (Phase 0 gate)

**Files:** none (manual validation, no code changes).

- [ ] **Step 1: Download Gemma 4 E4B IT Q4_K_M GGUF**

Pull from a reputable publisher (e.g. `bartowski/gemma-3-4b-it-GGUF` on HuggingFace). Place at a temp path first — final install happens in Task 4.

```bash
mkdir -p /tmp/phase-b-validation
cd /tmp/phase-b-validation
# Example; use huggingface-cli or wget with the actual URL from the publisher's model card.
wget -O gemma-4-E4B-it-Q4_K_M.gguf \
    <publisher URL from model card>
```

Note the sha256 for Task 4:

```bash
sha256sum gemma-4-E4B-it-Q4_K_M.gguf
```

- [ ] **Step 2: Run llama-server directly on Vulkan, on a scratch port**

```bash
sudo -u _llama /opt/llama/bin/llama-server \
    --model /tmp/phase-b-validation/gemma-4-E4B-it-Q4_K_M.gguf \
    --host 127.0.0.1 \
    --port 18083 \
    --device Vulkan0 \
    --ctx-size 16384 &
SERVER_PID=$!
sleep 10  # allow model load
```

(If `--device Vulkan0` isn't the right flag per Task 2 Step 7, substitute. Some llama.cpp versions use `-dev Vulkan0` or require `--main-gpu` + `--split-mode none`.)

- [ ] **Step 3: Confirm health and model**

```bash
curl -s http://127.0.0.1:18083/health
curl -s http://127.0.0.1:18083/v1/models
```

Expected: 200 on both. `/v1/models` returns one entry with an ID derived from the GGUF.

- [ ] **Step 4: One-off inference and record tok/s**

```bash
curl -s http://127.0.0.1:18083/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "gemma-4-E4B-it-Q4_K_M",
        "messages": [{"role": "user", "content": "Categorize this text: \"I want to learn about the history of cryptography.\" Respond with exactly one word from this list: Security, Programming, Networking, History, Writing, Unfiled."}],
        "stream": false
    }' | python3 -m json.tool
```

Expected: coherent single-word response ("Security" or "History" are both reasonable; anything sensible passes).

Check the logs for timing. llama-server prints `eval time` (prompt processing) and `generation time` (token generation). Record:

```
Vulkan perf baseline (Gemma 4 E4B IT Q4_K_M):
  prompt processing: _____ tok/s
  generation: _____ tok/s
```

Save these numbers in the server-side spec as a follow-up note, or paste them into the plan execution log.

- [ ] **Step 5: Stop the scratch server**

```bash
kill $SERVER_PID
```

- [ ] **Step 6: Phase 0 gate — decide**

If Steps 1–4 succeeded: Phase 0 is green, proceed to Phase 1.

If Vulkan device fails at runtime, or tok/s is uselessly low (< 5 tok/s generation), STOP. Investigate driver / user group issues or report back that Phase B needs a design revisit. Do not proceed with subsequent tasks.

- [ ] **Step 7: No commit**

---

## Phase 1: Batch service deployment artifacts

### Task 4: Install Gemma 4 E4B GGUF in models dir

**Files:** none (artifact placement).

**Models dir on this server is `/mnt/secondary/llama-models/`** (the `MODELS_DIR` env var in `manager.env` on the host overrides the `/opt/llama/models` default). All references below use the actual deployed path.

- [ ] **Step 1: Verify the GGUF is in place**

If the file was downloaded directly to the final location during Task 3, confirm it's there and has the right ownership:

```bash
ls -la /mnt/secondary/llama-models/gemma-4-E4B-it-Q4_K_M.gguf
```

If the file was downloaded to a temp location, move it:

```bash
sudo install -o _llama -g _llama -m 0644 \
    <source path> \
    /mnt/secondary/llama-models/gemma-4-E4B-it-Q4_K_M.gguf
```

- [ ] **Step 2: Record sha256**

```bash
sha256sum /mnt/secondary/llama-models/gemma-4-E4B-it-Q4_K_M.gguf
```

Paste the hash into the Appendix section of this plan for reproducibility.

- [ ] **Step 3: Verify main-slot `/v1/models` sees it**

```bash
curl -s http://192.168.1.14:11434/v1/models | python3 -m json.tool
```

Expected: the new GGUF appears in the list (manager lists all files in `MODELS_DIR`).

- [ ] **Step 4: No commit**

---

### Task 5: Add batch systemd unit and env file

**Files:**
- Create: `systemd/llama-server-batch.service`
- Create: `config/llama-server-batch.env`

- [ ] **Step 1: Write the systemd unit**

Create `systemd/llama-server-batch.service`:

```ini
# systemd/llama-server-batch.service
#
# Systemd unit for the Vulkan/iGPU batch inference backend.
#
# Parallel to llama-server.service but uses the AMD Vega iGPU via the
# Vulkan backend rather than the Tesla P40 via CUDA. Used for latency-
# tolerant background workloads routed via the llama-manager's batch slot.
#
# Install: sudo cp systemd/llama-server-batch.service /etc/systemd/system/
#          sudo systemctl daemon-reload && sudo systemctl enable llama-server-batch

[Unit]
Description=llama.cpp Batch Inference Server (Vulkan iGPU)
Documentation=https://github.com/ggerganov/llama.cpp

After=network-online.target
Wants=network-online.target

[Service]
Type=simple

# Reuses the same unprivileged user as the main llama-server.
# _llama must be in 'video' and 'render' groups for /dev/dri access.
User=_llama
Group=_llama

EnvironmentFile=/etc/llama/llama-server-batch.env

# Writable shader cache dir (_llama has no $HOME).
# Without this, llama-server logs "Failed to create /home/_llama for shader
# cache" and disables the cache, forcing shader recompile on every restart.
Environment=XDG_CACHE_HOME=/var/cache/llama

# Vulkan device selection via --device flag.
# No --n-gpu-layers: Vulkan backend manages layer placement.
ExecStart=/opt/llama/bin/llama-server \
    --host ${HOST} \
    --port ${PORT} \
    --model ${MODEL_PATH} \
    --device ${DEVICE} \
    --ctx-size ${CTX_SIZE}

# Bound the iGPU's shared-RAM footprint so it cannot starve the host
# under pressure. Gemma 4 E4B Q4 + CTX 16k KV cache fits in ~3 GB;
# 6 GB gives headroom.
MemoryMax=6G

Restart=on-failure
RestartSec=5

StandardOutput=append:/var/log/llama/llama-server-batch.log
StandardError=append:/var/log/llama/llama-server-batch.err

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Write the batch env file**

Create `config/llama-server-batch.env`:

```
# config/llama-server-batch.env
# Configuration for the batch llama-server backend (Vulkan iGPU).
# This file is read by systemd's EnvironmentFile directive.
# The model manager updates MODEL_PATH when swapping the batch slot.

# Path to the GGUF model file for the batch slot.
# Models live at /mnt/secondary/llama-models on this host (MODELS_DIR override).
MODEL_PATH=/mnt/secondary/llama-models/gemma-4-E4B-it-Q4_K_M.gguf

# Vulkan device selector. Vulkan0 is the first enumerated device
# per `vulkaninfo --summary`. On agenthost this is the Radeon Vega iGPU.
DEVICE=Vulkan0

# Context window. Small model, can afford a generous context;
# background callers rarely need >16k.
CTX_SIZE=16384

# Bind address and port. Localhost-only: manager handles external access.
HOST=127.0.0.1
PORT=8083
```

- [ ] **Step 3: Install on host**

```bash
# Create shader cache dir first (referenced by the unit's Environment= directive)
sudo install -d -o _llama -g _llama -m 0755 /var/cache/llama

sudo cp systemd/llama-server-batch.service /etc/systemd/system/
sudo cp config/llama-server-batch.env /etc/llama/llama-server-batch.env
sudo chmod 0644 /etc/llama/llama-server-batch.env
sudo systemctl daemon-reload
sudo systemctl enable llama-server-batch.service
sudo systemctl start llama-server-batch.service
```

- [ ] **Step 4: Verify the service is running**

```bash
sudo systemctl status llama-server-batch.service
curl -s http://127.0.0.1:8083/health
curl -s http://127.0.0.1:8083/v1/models | python3 -m json.tool
```

Expected: service `active (running)`. `/health` returns 200. `/v1/models` returns one entry with the Gemma 4 E4B model ID.

If `--device ${DEVICE}` doesn't work: re-check the flag name per Task 2 Step 7 output. Update `ExecStart` in the unit file, `daemon-reload`, restart.

- [ ] **Step 5: Commit unit + env file**

```bash
git add systemd/llama-server-batch.service config/llama-server-batch.env
git commit -m "feat: llama-server-batch systemd unit for Vulkan iGPU slot"
```

---

### Task 6: Extend sudoers for batch restart

**Files:** none in repo (system config).

- [ ] **Step 1: Identify the existing sudoers entry**

```bash
sudo grep -r _llama-mgr /etc/sudoers /etc/sudoers.d/ 2>/dev/null
```

Expected: one line allowing `_llama-mgr` to run `systemctl restart llama-server.service` via sudo.

- [ ] **Step 2: Edit with `visudo`**

```bash
sudo visudo -f /etc/sudoers.d/llama-mgr
```

(Or whichever file Step 1 identified.)

Change the line to allow both services:

```
_llama-mgr ALL=(root) NOPASSWD: /bin/systemctl restart llama-server.service, /bin/systemctl restart llama-server-batch.service
```

`visudo` will reject the save if the syntax is wrong.

- [ ] **Step 3: Verify as `_llama-mgr`**

```bash
sudo -u _llama-mgr sudo -n systemctl restart llama-server-batch.service
```

Expected: exit 0 (service is actually restarted). If "a terminal is required" or "password required", sudoers wasn't updated correctly.

- [ ] **Step 4: Verify batch slot came back up after the forced restart**

```bash
sleep 8
curl -s http://127.0.0.1:8083/health
```

Expected: 200.

- [ ] **Step 5: No commit** (system config is not tracked in repo).

---

## Phase 2: Manager code changes (TDD)

### Task 7: Add batch fields to ManagerConfig

**Files:**
- Modify: `manager/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Read existing config test patterns**

```bash
head -80 /home/edible/Projects/inference_server/tests/test_config.py
```

Note the existing pattern — mocks env vars with `monkeypatch.setenv`, calls `ManagerConfig.from_env`, asserts on fields.

- [ ] **Step 2: Write failing tests**

Append to `tests/test_config.py`:

```python
def test_from_env_batch_defaults(monkeypatch):
    """Batch-slot fields have sensible defaults when env vars are absent."""
    for var in ("BATCH_SERVER_HOST", "BATCH_SERVER_PORT", "BATCH_SERVER_ENV",
                "BATCH_SERVER_UNIT", "BATCH_QUEUE_LIMIT", "BATCH_MODEL_DEFAULT"):
        monkeypatch.delenv(var, raising=False)
    from manager.config import ManagerConfig
    cfg = ManagerConfig.from_env()
    assert cfg.batch_server_host == "127.0.0.1"
    assert cfg.batch_server_port == 8083
    assert cfg.batch_server_env == "/etc/llama/llama-server-batch.env"
    assert cfg.batch_server_unit == "llama-server-batch.service"
    assert cfg.batch_queue_limit == 20
    assert cfg.batch_model_default == "gemma-4-E4B-it-Q4_K_M"


def test_from_env_batch_overrides(monkeypatch):
    """Env vars override batch-slot defaults."""
    monkeypatch.setenv("BATCH_SERVER_HOST", "127.0.0.2")
    monkeypatch.setenv("BATCH_SERVER_PORT", "9999")
    monkeypatch.setenv("BATCH_SERVER_ENV", "/tmp/custom.env")
    monkeypatch.setenv("BATCH_SERVER_UNIT", "custom.service")
    monkeypatch.setenv("BATCH_QUEUE_LIMIT", "7")
    monkeypatch.setenv("BATCH_MODEL_DEFAULT", "my-model")
    from manager.config import ManagerConfig
    cfg = ManagerConfig.from_env()
    assert cfg.batch_server_host == "127.0.0.2"
    assert cfg.batch_server_port == 9999
    assert cfg.batch_server_env == "/tmp/custom.env"
    assert cfg.batch_server_unit == "custom.service"
    assert cfg.batch_queue_limit == 7
    assert cfg.batch_model_default == "my-model"


def test_batch_server_url_property():
    """batch_server_url composes host and port."""
    from manager.config import ManagerConfig
    cfg = ManagerConfig(
        host="0.0.0.0", port=11434,
        llama_server_host="127.0.0.1", llama_server_port=8081,
        models_dir="/tmp", llama_server_env="/tmp/env",
        queue_limit=50, swap_timeout=60, log_file="/dev/null",
        embeddings_host="127.0.0.1", embeddings_port=8082,
        collections_config="/dev/null", skills_db_path="",
        batch_server_host="127.0.0.1", batch_server_port=8083,
        batch_server_env="/tmp/batch.env",
        batch_server_unit="llama-server-batch.service",
        batch_queue_limit=20, batch_model_default="gemma-4-E4B-it-Q4_K_M",
    )
    assert cfg.batch_server_url == "http://127.0.0.1:8083"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/test_config.py -v -k batch
```

Expected: FAIL with `TypeError: ManagerConfig.__init__() got an unexpected keyword argument 'batch_server_host'` or similar.

- [ ] **Step 4: Add fields + env loading to `ManagerConfig`**

In `manager/config.py`, add to the dataclass (after the existing fields):

```python
    batch_server_host: str
    batch_server_port: int
    batch_server_env: str
    batch_server_unit: str
    batch_queue_limit: int
    batch_model_default: str

    @property
    def batch_server_url(self) -> str:
        """Full URL for connecting to the batch llama-server."""
        return f"http://{self.batch_server_host}:{self.batch_server_port}"
```

Update `from_env` to load them:

```python
            batch_server_host=os.getenv("BATCH_SERVER_HOST", "127.0.0.1"),
            batch_server_port=_int_env("BATCH_SERVER_PORT", 8083),
            batch_server_env=os.getenv("BATCH_SERVER_ENV", "/etc/llama/llama-server-batch.env"),
            batch_server_unit=os.getenv("BATCH_SERVER_UNIT", "llama-server-batch.service"),
            batch_queue_limit=_int_env("BATCH_QUEUE_LIMIT", 20),
            batch_model_default=os.getenv("BATCH_MODEL_DEFAULT", "gemma-4-E4B-it-Q4_K_M"),
```

- [ ] **Step 5: Update `conftest.py` fixtures**

In `tests/conftest.py`, update `test_config` fixture to include the new fields (with test-appropriate defaults):

```python
    return ManagerConfig(
        host="127.0.0.1",
        port=8080,
        llama_server_host="127.0.0.1",
        llama_server_port=8081,
        models_dir=tmp_models_dir,
        llama_server_env=tmp_env_file,
        queue_limit=20,
        swap_timeout=5,
        log_file="/dev/null",
        embeddings_host="127.0.0.1",
        embeddings_port=8082,
        collections_config="/dev/null",
        skills_db_path="",
        batch_server_host="127.0.0.1",
        batch_server_port=8083,
        batch_server_env=str(tmp_path / "llama-server-batch.env"),
        batch_server_unit="llama-server-batch.service",
        batch_queue_limit=20,
        batch_model_default="test-batch-model",
    )
```

Also add a fixture that creates a batch env file analogous to `tmp_env_file`:

```python
@pytest.fixture
def tmp_batch_env_file(tmp_path):
    """Create a temporary llama-server-batch env file for testing batch swaps."""
    env_file = tmp_path / "llama-server-batch.env"
    env_file.write_text(
        "MODEL_PATH=\nDEVICE=Vulkan0\nCTX_SIZE=16384\nHOST=127.0.0.1\nPORT=8083\n"
    )
    return str(env_file)
```

Update `test_config` fixture to use it:

```python
@pytest.fixture
def test_config(tmp_models_dir, tmp_env_file, tmp_batch_env_file):
    ...
    batch_server_env=tmp_batch_env_file,
    ...
```

Same update to `collection_config` fixture (it reconstructs `ManagerConfig` and must pass all required fields).

- [ ] **Step 6: Run all tests**

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/ -q
```

Expected: new config tests pass. Existing tests still pass (fixtures now provide the batch fields).

- [ ] **Step 7: Commit**

```bash
git add manager/config.py tests/test_config.py tests/conftest.py
git commit -m "feat: add batch-slot fields to ManagerConfig"
```

---

### Task 8: Create `manager/slots.py` with `SlotState`

**Files:**
- Create: `manager/slots.py`
- Create: `tests/test_slots.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_slots.py`:

```python
"""Tests for SlotState: per-slot state container and probe helpers."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from manager.slots import SlotState
from manager.queue import RequestQueue


def _make_slot(name="main", port=8081):
    return SlotState(
        name=name,
        host="127.0.0.1",
        port=port,
        env_file="/tmp/env",
        systemd_unit="llama-server.service",
        queue=RequestQueue(max_size=20),
    )


def test_slot_defaults():
    slot = _make_slot()
    assert slot.loaded_model is None
    assert slot.healthy is False
    assert slot.last_swap_utc is None
    assert slot.url == "http://127.0.0.1:8081"
    assert isinstance(slot.swap_lock, asyncio.Lock)
    assert isinstance(slot.queue_event, asyncio.Event)


@pytest.mark.asyncio
async def test_probe_success():
    """Probe hits /v1/models; populates loaded_model and sets healthy True."""
    slot = _make_slot()
    client = MagicMock()
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {
        "object": "list",
        "data": [{"id": "gemma-4-26b-a4b-it-q4_k_m", "object": "model"}],
    }
    client.get = AsyncMock(return_value=mock_response)

    await slot.probe(client)
    client.get.assert_awaited_once_with("http://127.0.0.1:8081/v1/models", timeout=3)
    assert slot.loaded_model == "gemma-4-26b-a4b-it-q4_k_m"
    assert slot.healthy is True


@pytest.mark.asyncio
async def test_probe_empty_data_unhealthy():
    """Probe succeeds but /v1/models returns empty list -> not healthy, no model."""
    slot = _make_slot()
    client = MagicMock()
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"data": []}
    client.get = AsyncMock(return_value=mock_response)

    await slot.probe(client)
    assert slot.loaded_model is None
    assert slot.healthy is False


@pytest.mark.asyncio
async def test_probe_connection_error_unhealthy():
    """Probe catches httpx.ConnectError (or any exception): slot unhealthy, no raise."""
    import httpx
    slot = _make_slot()
    client = MagicMock()
    client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

    await slot.probe(client)
    assert slot.healthy is False
    assert slot.loaded_model is None


@pytest.mark.asyncio
async def test_probe_non_200_unhealthy():
    """Probe with 500 response: unhealthy, no raise."""
    slot = _make_slot()
    client = MagicMock()
    mock_response = MagicMock(status_code=500)
    client.get = AsyncMock(return_value=mock_response)

    await slot.probe(client)
    assert slot.healthy is False


@pytest.mark.asyncio
async def test_reconcile_on_error_reprobes():
    """reconcile_on_error() runs probe() again (short-hand verification)."""
    slot = _make_slot()
    slot.loaded_model = "old-model"
    slot.healthy = True
    client = MagicMock()
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"data": [{"id": "new-model"}]}
    client.get = AsyncMock(return_value=mock_response)

    await slot.reconcile_on_error(client)
    assert slot.loaded_model == "new-model"
    assert slot.healthy is True


def test_mark_unhealthy_clears_flag():
    slot = _make_slot()
    slot.healthy = True
    slot.mark_unhealthy()
    assert slot.healthy is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/test_slots.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'manager.slots'`.

- [ ] **Step 3: Create `manager/slots.py`**

```python
"""
Per-slot state for the dual-slot model manager.

Each inference backend ('main', 'batch') has its own SlotState containing
loaded-model tracking, health, swap lock, queue, and an event the handler
signals when a new item is enqueued. Swap operations and routing decisions
read/write this state in-process.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from manager.queue import RequestQueue

logger = logging.getLogger(__name__)


@dataclass
class SlotState:
    """State container for one inference slot.

    A slot is a backing llama-server process that the manager fronts.
    'loaded_model' reflects what that process currently has loaded;
    'healthy' is a boolean derived from the most recent probe.

    The queue, queue_event, and swap_lock are per-slot so work on one
    slot does not interfere with the other.
    """
    name: str                               # "main" | "batch"
    host: str
    port: int
    env_file: str                           # path for the swap to rewrite
    systemd_unit: str                       # unit name to restart
    queue: RequestQueue
    loaded_model: Optional[str] = None
    healthy: bool = False
    last_swap_utc: Optional[str] = None
    queue_event: asyncio.Event = field(default_factory=asyncio.Event)
    swap_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def probe(self, client) -> None:
        """Query /v1/models on the slot's backend and update state.

        Never raises. On any failure (connection, timeout, non-200,
        unexpected JSON shape), sets healthy=False and leaves loaded_model
        as whatever it was (the caller may want to null it via mark_unhealthy).
        """
        try:
            resp = await client.get(f"{self.url}/v1/models", timeout=3)
        except Exception as exc:
            logger.warning("slot %s probe failed: %s", self.name, exc)
            self.healthy = False
            return

        if resp.status_code != 200:
            logger.warning("slot %s probe returned %s", self.name, resp.status_code)
            self.healthy = False
            return

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("slot %s probe returned non-JSON: %s", self.name, exc)
            self.healthy = False
            return

        entries = data.get("data") or []
        if not entries:
            logger.warning("slot %s /v1/models returned no entries", self.name)
            self.loaded_model = None
            self.healthy = False
            return

        self.loaded_model = entries[0].get("id")
        self.healthy = bool(self.loaded_model)

    async def reconcile_on_error(self, client) -> None:
        """Re-probe after a backend 5xx. Updates loaded_model and healthy."""
        await self.probe(client)

    def mark_unhealthy(self) -> None:
        self.healthy = False

    def mark_swapped(self, model: str) -> None:
        """Record a successful swap: update loaded_model, last_swap_utc, healthy."""
        self.loaded_model = model
        self.last_swap_utc = datetime.now(timezone.utc).isoformat()
        self.healthy = True

    def to_status_dict(self) -> dict:
        """Shape for the /status endpoint's slots section."""
        return {
            "host": self.host,
            "port": self.port,
            "loaded_model": self.loaded_model,
            "healthy": self.healthy,
            "last_swap_utc": self.last_swap_utc,
            "queue_depth": self.queue.depth,
            "queue_limit": self.queue.max_size,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/test_slots.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add manager/slots.py tests/test_slots.py
git commit -m "feat: SlotState container with probe, reconcile, swap tracking"
```

---

### Task 9: Create `manager/routing.py` with `resolve_slot`

**Files:**
- Create: `manager/routing.py`
- Create: `tests/test_routing.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_routing.py`:

```python
"""Tests for manager/routing.py: pure routing function over slot state."""
from manager.routing import resolve_slot
from manager.slots import SlotState
from manager.queue import RequestQueue


def _slot(name: str, loaded_model=None) -> SlotState:
    return SlotState(
        name=name,
        host="127.0.0.1",
        port=8081 if name == "main" else 8083,
        env_file="/tmp/env",
        systemd_unit=f"llama-server-{name}.service" if name == "batch" else "llama-server.service",
        queue=RequestQueue(max_size=10),
        loaded_model=loaded_model,
    )


def test_resolve_model_on_main():
    slots = {"main": _slot("main", "model-A"), "batch": _slot("batch", "model-B")}
    assert resolve_slot("model-A", slots) == "main"


def test_resolve_model_on_batch():
    slots = {"main": _slot("main", "model-A"), "batch": _slot("batch", "model-B")}
    assert resolve_slot("model-B", slots) == "batch"


def test_resolve_model_on_neither_returns_main():
    slots = {"main": _slot("main", "model-A"), "batch": _slot("batch", "model-B")}
    assert resolve_slot("unknown-C", slots) == "main"


def test_resolve_model_on_both_prefers_main():
    """If the same model is somehow loaded on both, main wins (first match)."""
    slots = {"main": _slot("main", "shared"), "batch": _slot("batch", "shared")}
    assert resolve_slot("shared", slots) == "main"


def test_resolve_empty_model_returns_main():
    """Empty model string routes to main (implicit main swap preserved)."""
    slots = {"main": _slot("main", None), "batch": _slot("batch", None)}
    assert resolve_slot("", slots) == "main"


def test_resolve_main_unloaded_but_batch_loaded():
    """Main has nothing loaded, batch has the model -> route to batch."""
    slots = {"main": _slot("main", None), "batch": _slot("batch", "model-B")}
    assert resolve_slot("model-B", slots) == "batch"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/test_routing.py -v
```

Expected: `ModuleNotFoundError: No module named 'manager.routing'`.

- [ ] **Step 3: Create `manager/routing.py`**

```python
"""
Pure routing decisions for the dual-slot model manager.

Given a requested model name and the current per-slot state, returns which
slot should handle the request. Pure function: no I/O, no side effects,
safe to call from anywhere.
"""
from manager.slots import SlotState


def resolve_slot(model: str, slots: dict[str, SlotState]) -> str:
    """Return the slot name that should handle the request.

    Rules:
      1. If the model is loaded on 'main', return 'main'.
      2. Else if the model is loaded on 'batch', return 'batch'.
      3. Else return 'main' (triggers an implicit main swap in the caller,
         preserving the existing single-slot behavior).
    """
    main = slots.get("main")
    if main is not None and main.loaded_model == model:
        return "main"
    batch = slots.get("batch")
    if batch is not None and batch.loaded_model == model:
        return "batch"
    return "main"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/test_routing.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add manager/routing.py tests/test_routing.py
git commit -m "feat: pure resolve_slot routing function"
```

---

### Task 10: Refactor `ModelSwapper` to be slot-parameterized

**Files:**
- Modify: `manager/swap.py`
- Modify: `tests/test_swap.py`

The current `ModelSwapper` hardcodes `llama_server_env` and `llama-server.service`. Refactor to accept a `SlotState` so it can operate on either slot's env file and unit.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_swap.py`:

```python
@pytest.fixture
def batch_slot(tmp_path, tmp_batch_env_file):
    from manager.slots import SlotState
    from manager.queue import RequestQueue
    return SlotState(
        name="batch",
        host="127.0.0.1",
        port=8083,
        env_file=tmp_batch_env_file,
        systemd_unit="llama-server-batch.service",
        queue=RequestQueue(max_size=20),
    )


@pytest.fixture
def main_slot(tmp_env_file):
    from manager.slots import SlotState
    from manager.queue import RequestQueue
    return SlotState(
        name="main",
        host="127.0.0.1",
        port=8081,
        env_file=tmp_env_file,
        systemd_unit="llama-server.service",
        queue=RequestQueue(max_size=50),
    )


def test_swapper_for_main_writes_main_env(test_config, main_slot):
    from manager.swap import ModelSwapper
    swapper = ModelSwapper(test_config, slot=main_slot)
    swapper.update_env_file("/opt/llama/models/new.gguf")
    content = Path(main_slot.env_file).read_text()
    assert "MODEL_PATH=/opt/llama/models/new.gguf" in content


def test_swapper_for_batch_writes_batch_env(test_config, batch_slot):
    from manager.swap import ModelSwapper
    swapper = ModelSwapper(test_config, slot=batch_slot)
    swapper.update_env_file("/opt/llama/models/other.gguf")
    content = Path(batch_slot.env_file).read_text()
    assert "MODEL_PATH=/opt/llama/models/other.gguf" in content


@pytest.mark.asyncio
async def test_swapper_restart_targets_correct_unit(test_config, batch_slot):
    from manager.swap import ModelSwapper
    swapper = ModelSwapper(test_config, slot=batch_slot)
    mock_result = MagicMock()
    mock_result.returncode = 0

    async def fake_executor(executor, fn):
        return fn()

    with patch("subprocess.run", return_value=mock_result) as mock_run, \
         patch.object(asyncio.get_event_loop(), "run_in_executor", side_effect=fake_executor):
        await swapper.restart_llama_server()

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd == ["sudo", "systemctl", "restart", "llama-server-batch.service"]


@pytest.mark.asyncio
async def test_swapper_health_polls_slot_url(test_config, batch_slot):
    """wait_for_health targets the slot's URL, not the hardcoded main URL."""
    from manager.swap import ModelSwapper
    swapper = ModelSwapper(test_config, slot=batch_slot)
    mock_response = MagicMock(status_code=200)
    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.get = AsyncMock(return_value=mock_response)
        ok = await swapper.wait_for_health()
    assert ok is True
    call_url = instance.get.call_args[0][0]
    assert call_url == "http://127.0.0.1:8083/health"
```

Update the existing tests (`test_update_env_file`, `test_update_env_file_overwrites_existing`, `test_restart_llama_server`, `test_restart_raises_on_failure`) to construct `ModelSwapper(test_config, slot=main_slot)` instead of `ModelSwapper(test_config)`. Update the `swapper` fixture accordingly:

```python
@pytest.fixture
def swapper(test_config, main_slot):
    return ModelSwapper(test_config, slot=main_slot)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/test_swap.py -v
```

Expected: FAIL with `TypeError: ModelSwapper.__init__() got an unexpected keyword argument 'slot'`.

- [ ] **Step 3: Refactor `ModelSwapper`**

Replace `manager/swap.py` with:

```python
"""
Model swap orchestration for one slot.

A ModelSwapper instance is bound to a specific slot (main or batch) via
its SlotState and operates on that slot's env file and systemd unit.
The manager instantiates one swapper per slot at startup.
"""
import asyncio
import re
import subprocess
import logging
import httpx

from manager.config import ManagerConfig
from manager.slots import SlotState

logger = logging.getLogger(__name__)


class ModelSwapper:
    """Orchestrates swap for one slot: env rewrite, systemd restart, health poll."""

    def __init__(self, config: ManagerConfig, slot: SlotState):
        self._config = config
        self._slot = slot

    def update_env_file(self, model_path: str) -> None:
        """Rewrite MODEL_PATH in the slot's env file."""
        with open(self._slot.env_file, "r") as f:
            content = f.read()

        content = re.sub(
            r"^MODEL_PATH=.*$",
            f"MODEL_PATH={model_path}",
            content,
            flags=re.MULTILINE,
        )

        with open(self._slot.env_file, "w") as f:
            f.write(content)

        logger.info("slot=%s updated env file MODEL_PATH=%s", self._slot.name, model_path)

    async def restart_llama_server(self) -> None:
        """Restart the slot's systemd unit via sudo."""
        unit = self._slot.systemd_unit
        logger.info("slot=%s restarting %s", self._slot.name, unit)

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                ["sudo", "systemctl", "restart", unit],
                capture_output=True,
                text=True,
                timeout=30,
            ),
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to restart {unit}: {result.stderr}"
            )

        logger.info("slot=%s restart succeeded", self._slot.name)

    async def wait_for_health(self) -> bool:
        """Poll the slot's /health until it responds or we time out."""
        url = f"{self._slot.url}/health"
        timeout = self._config.swap_timeout
        poll_interval = 2

        logger.info("slot=%s waiting for %s (timeout=%ds)", self._slot.name, url, timeout)

        elapsed = 0
        async with httpx.AsyncClient() as client:
            while elapsed < timeout:
                try:
                    response = await client.get(url, timeout=5)
                    if response.status_code == 200:
                        logger.info("slot=%s healthy after %ds", self._slot.name, elapsed)
                        return True
                except (httpx.ConnectError, httpx.TimeoutException, ConnectionError):
                    pass

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

        logger.error("slot=%s health timed out after %ds", self._slot.name, timeout)
        return False

    async def swap_to(self, model_path: str) -> bool:
        """Full swap sequence. Returns True on success, False on health timeout."""
        logger.info("slot=%s starting swap to %s", self._slot.name, model_path)
        self.update_env_file(model_path)
        await self.restart_llama_server()
        healthy = await self.wait_for_health()

        if healthy:
            logger.info("slot=%s swap complete: %s", self._slot.name, model_path)
        else:
            logger.error("slot=%s swap failed (health timeout): %s", self._slot.name, model_path)

        return healthy
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/test_swap.py -v
```

Expected: all swap tests pass (existing + new).

- [ ] **Step 5: Full test suite — existing endpoint tests may break**

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/ -q
```

Expected: `test_endpoints.py` tests fail because `ServerState` constructs `ModelSwapper(config)` without a slot. This is expected — it's fixed in Task 11. For now the endpoint failures are noise from this refactor landing ahead of ServerState.

If the endpoint tests fail with errors at CONSTRUCTION (import-time): fix by temporarily adding a placeholder `ModelSwapper(config, slot=None)` behavior that's later cleaned up. But better: do NOT commit yet if the full suite is broken. Proceed straight to Task 11 in the same commit.

- [ ] **Step 6: Do NOT commit yet**

This refactor is coupled to Task 11. Hold the diff until both compile together.

---

### Task 11: Refactor `ServerState` to hold slots dict; update `/status`

**Files:**
- Modify: `manager/app.py`
- Modify: `tests/test_endpoints.py`

This is the biggest task. It replaces `ServerState`'s flat fields with a `slots: dict[str, SlotState]`, instantiates per-slot swappers, and updates `/status` to the new shape.

- [ ] **Step 1: Rewrite the failing `/status` tests**

Replace `test_status_endpoint` in `tests/test_endpoints.py`:

```python
def test_status_endpoint_has_slots(client):
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "slots" in data
    assert "main" in data["slots"]
    assert "batch" in data["slots"]

    for slot_name in ("main", "batch"):
        slot = data["slots"][slot_name]
        assert set(slot.keys()) == {
            "host", "port", "loaded_model", "healthy",
            "last_swap_utc", "queue_depth", "queue_limit",
        }
        assert isinstance(slot["healthy"], bool)
        assert isinstance(slot["queue_depth"], int)
        assert isinstance(slot["queue_limit"], int)

    assert "gpu" in data
    assert "uptime_seconds" in data


def test_status_no_top_level_flat_fields(client):
    """The old flat fields are removed; consumers must read from slots."""
    data = client.get("/status").json()
    assert "current_model" not in data
    assert "loading_model" not in data
    assert "error_message" not in data
    assert "state" not in data
    assert "queue_depth" not in data
    assert "queue_limit" not in data


def test_status_main_queue_limit_matches_config(client):
    """Slots reflect the config's per-slot queue limits."""
    data = client.get("/status").json()
    assert data["slots"]["main"]["queue_limit"] == 20  # test_config queue_limit
    assert data["slots"]["batch"]["queue_limit"] == 20  # test_config batch_queue_limit
```

Also remove `test_ensure_model_updates_state` — it tested the flat `state` field. Replace with a test that uses the per-slot lock:

```python
@pytest.mark.asyncio
async def test_ensure_model_on_slot_updates_loaded_model(test_config, main_slot):
    from manager.app import ServerState
    server = ServerState(test_config)
    server.slots["main"].swapper.swap_to = AsyncMock(return_value=True)

    ok = await server.ensure_model_on_slot("main", "test-model-q4")
    assert ok is True
    assert server.slots["main"].loaded_model == "test-model-q4"
    assert server.slots["main"].healthy is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/test_endpoints.py -v -k status
```

Expected: FAIL — `/status` returns the old flat shape or 500.

- [ ] **Step 3: Refactor `ServerState` and `/status`**

In `manager/app.py`:

Replace `ServerState.__init__` with:

```python
    def __init__(self, config: ManagerConfig):
        self._config = config
        self.error_message: str | None = None

        # Build slots dict.
        from manager.slots import SlotState
        self.slots: dict[str, SlotState] = {
            "main": SlotState(
                name="main",
                host=config.llama_server_host,
                port=config.llama_server_port,
                env_file=config.llama_server_env,
                systemd_unit="llama-server.service",
                queue=RequestQueue(max_size=config.queue_limit),
            ),
            "batch": SlotState(
                name="batch",
                host=config.batch_server_host,
                port=config.batch_server_port,
                env_file=config.batch_server_env,
                systemd_unit=config.batch_server_unit,
                queue=RequestQueue(max_size=config.batch_queue_limit),
            ),
        }

        # One swapper per slot.
        self.slots["main"].swapper = ModelSwapper(config, slot=self.slots["main"])
        self.slots["batch"].swapper = ModelSwapper(config, slot=self.slots["batch"])

        # Collection retrieval (unchanged).
        self.db: VectorDB | None = None
        self.embeddings_client: EmbeddingsClient | None = None

        from manager.reindex_jobs import ReindexRegistry
        self.reindex_registry = ReindexRegistry()
```

Note: `swapper` is dynamically attached to each slot to avoid a circular import (SlotState doesn't reference ModelSwapper). An alternative is a separate dict keyed by slot name; pick whichever is cleaner after Task 10 lands.

Add:

```python
    def model_path(self, model_name: str) -> str | None:
        """Unchanged: resolves a model name to its file path in MODELS_DIR."""
        name = model_name if model_name.endswith(".gguf") else f"{model_name}.gguf"
        path = Path(self._config.models_dir) / name
        return str(path) if path.exists() else None

    def list_models(self) -> list[str]:
        """Unchanged."""
        models_dir = Path(self._config.models_dir)
        if not models_dir.exists():
            return []
        return sorted(p.stem for p in models_dir.glob("*.gguf"))

    async def ensure_model_on_slot(self, slot_name: str, model_name: str) -> bool:
        """Ensure model_name is loaded on the named slot.

        Holds the slot's swap_lock for the duration. On failure, drains
        the slot's queue with an error.
        """
        slot = self.slots[slot_name]
        async with slot.swap_lock:
            if slot.healthy and slot.loaded_model == model_name:
                return True

            path = self.model_path(model_name)
            if path is None:
                msg = f"Model file not found: {model_name}"
                slot.mark_unhealthy()
                self._drain_slot_queue_with_error(slot_name, msg)
                return False

            logger.info("slot=%s swapping to %s", slot_name, model_name)
            success = await slot.swapper.swap_to(path)
            if success:
                slot.mark_swapped(model_name)
                return True

            msg = f"Model swap timed out on slot {slot_name}: {model_name}"
            slot.mark_unhealthy()
            self._drain_slot_queue_with_error(slot_name, msg)
            return False

    def _drain_slot_queue_with_error(self, slot_name: str, message: str) -> None:
        """Drain the given slot's queue, signaling each item with an error."""
        slot = self.slots[slot_name]
        items = slot.queue.drain()
        for item in items:
            item["error"] = message
            item["event"].set()
        if items:
            logger.warning(
                "slot=%s drained %d queued requests with error: %s",
                slot_name, len(items), message,
            )
```

Delete the old flat fields (`self.state`, `self.current_model`, `self.loading_model`, `self.queue`, `self.swapper`, `self.swap_lock`, `self.queue_event`, `_drain_queue_with_error`, original `ensure_model`).

Replace the `/status` handler body with:

```python
    @app.get("/status")
    async def status():
        gpu = get_gpu_info()
        return {
            "slots": {
                name: slot.to_status_dict()
                for name, slot in server.slots.items()
            },
            "gpu": gpu,
            "uptime_seconds": int(time.time() - _start_time),
        }
```

- [ ] **Step 4: Update lifespan startup to probe each slot**

Replace the existing "Check whether llama-server is already running" block in the `lifespan` function:

```python
        async with httpx.AsyncClient() as probe_client:
            for name, slot in server.slots.items():
                await slot.probe(probe_client)
                if slot.healthy:
                    logger.info(
                        "slot=%s ready with model %s",
                        name, slot.loaded_model,
                    )
                else:
                    logger.info("slot=%s not reachable at startup", name)
```

- [ ] **Step 5: Queue consumer temporarily wired for main only**

The existing `_queue_consumer` processes `server.queue` and `server.queue_event`. Replace with a factory that creates one task per slot (full multi-slot consumer comes in Task 12):

```python
async def _queue_consumer(server: "ServerState", config: ManagerConfig, slot_name: str) -> None:
    """Background task: process one slot's queue serially.

    Waits on the slot's queue_event, dispatches items one-by-one through
    ensure_model_on_slot + proxy to the slot's backend URL.
    """
    slot = server.slots[slot_name]
    backend_url = f"{slot.url}/v1/chat/completions"

    while True:
        await slot.queue_event.wait()
        slot.queue_event.clear()

        while not slot.queue.empty():
            item = await slot.queue.dequeue()
            body: dict = item["body"]
            event: asyncio.Event = item["event"]
            model_name: str = body.get("model", "")

            ok = await server.ensure_model_on_slot(slot_name, model_name)
            if not ok:
                if item["error"] is None:
                    item["error"] = f"Model swap failed on slot {slot_name}"
                    event.set()
                continue

            is_streaming = body.get("stream", False)
            try:
                if is_streaming:
                    async def _stream_gen(request_body=body, url=backend_url):
                        async with httpx.AsyncClient() as client:
                            async with client.stream(
                                "POST", url, json=request_body, timeout=None,
                            ) as resp:
                                async for chunk in resp.aiter_bytes():
                                    yield chunk
                    item["response"] = StreamingResponse(
                        _stream_gen(),
                        media_type="text/event-stream",
                    )
                else:
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(backend_url, json=body, timeout=120)
                    item["response"] = Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        media_type=resp.headers.get("content-type", "application/json"),
                    )
            except Exception as exc:
                logger.exception("slot=%s proxy error: %s", slot_name, exc)
                item["error"] = f"Proxy error: {exc}"

            event.set()
```

And in `lifespan`, start one task per slot:

```python
        consumer_tasks = [
            asyncio.create_task(
                _queue_consumer(server, config, slot_name=name),
                name=f"queue_consumer_{name}",
            )
            for name in server.slots
        ]
```

Replace the existing `consumer_task.cancel()` / await with a loop over `consumer_tasks`.

- [ ] **Step 6: Update `/v1/chat/completions` to enqueue on the correct slot**

Replace the chat_completions handler body. For this task, route everything to main (routing comes in Task 13 — keeping the scope manageable). Use the slot's event:

```python
        event = asyncio.Event()
        item: dict = {"body": body, "event": event, "response": None, "error": None}

        slot_name = "main"  # Task 13 replaces this with resolve_slot(...)
        slot = server.slots[slot_name]
        try:
            await slot.queue.enqueue(item)
        except RequestQueue.QueueFullError:
            return JSONResponse(
                {"error": {"message": "Server busy", "type": "server_error"}},
                status_code=503,
                headers={"Retry-After": "5"},
            )

        slot.queue_event.set()
        await event.wait()

        if item["error"]:
            return JSONResponse(
                {"error": {"message": item["error"], "type": "server_error"}},
                status_code=503,
                headers={"Retry-After": "5"},
            )
        return item["response"]
```

- [ ] **Step 7: Run full test suite**

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/ -q
```

Expected:
- new `/status` tests pass
- swap tests pass
- existing chat-completion tests still pass (routing still goes to main)
- `test_ensure_model_updates_state` renamed/rewritten and passes

If any existing test fails because it references `server.queue` or `server.state` etc., update it to use the slot version.

- [ ] **Step 8: Commit (combined with Task 10's diff)**

```bash
git add manager/swap.py manager/app.py tests/test_swap.py tests/test_endpoints.py
git commit -m "feat: dual-slot ServerState with per-slot swapper and queue consumer"
```

---

### Task 12: Route `/v1/chat/completions` by model name

**Files:**
- Modify: `manager/app.py`
- Modify: `tests/test_endpoints.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_endpoints.py`:

```python
def test_chat_completions_routes_batch_model(client, test_config, monkeypatch):
    """When the batch slot has model X loaded, a request for X enqueues on batch, not main."""
    from manager.app import create_app
    # Use the existing app's server state via the fixture:
    # set main loaded_model to q4, batch loaded_model to q8.
    # (Depends on the fixture; adapt accordingly.)
    app = client.app
    # The app was already created by the client fixture. Grab the ServerState:
    # In create_app, the ServerState isn't exposed. For testability, we'll need
    # a hook. Add it in Step 3 of this task.
    server = app.state.server
    server.slots["main"].loaded_model = "test-model-q4"
    server.slots["main"].healthy = True
    server.slots["batch"].loaded_model = "test-model-q8"
    server.slots["batch"].healthy = True

    # Patch the per-slot consumer to short-circuit (just mark the request OK).
    async def fake_proxy(body, slot_name):
        return Response(content=b'{"id":"test"}', media_type="application/json")

    # Simpler: track which queue the item was enqueued on.
    enqueued_on = []
    original_enqueue_main = server.slots["main"].queue.enqueue
    original_enqueue_batch = server.slots["batch"].queue.enqueue

    async def wrap_main(item):
        enqueued_on.append("main")
        # complete the request immediately so the test doesn't hang
        item["response"] = Response(content=b'{"id":"m"}', media_type="application/json")
        item["event"].set()
    async def wrap_batch(item):
        enqueued_on.append("batch")
        item["response"] = Response(content=b'{"id":"b"}', media_type="application/json")
        item["event"].set()

    server.slots["main"].queue.enqueue = wrap_main
    server.slots["batch"].queue.enqueue = wrap_batch

    try:
        r = client.post("/v1/chat/completions", json={
            "model": "test-model-q8",
            "messages": [{"role": "user", "content": "hi"}],
        })
    finally:
        server.slots["main"].queue.enqueue = original_enqueue_main
        server.slots["batch"].queue.enqueue = original_enqueue_batch

    assert enqueued_on == ["batch"]


def test_chat_completions_routes_main_for_unknown(client):
    """Unknown model routes to main (implicit main-swap path preserved)."""
    app = client.app
    server = app.state.server
    server.slots["main"].loaded_model = "test-model-q4"
    server.slots["main"].healthy = True
    server.slots["batch"].loaded_model = "test-model-q8"
    server.slots["batch"].healthy = True

    enqueued_on = []
    async def wrap_main(item):
        enqueued_on.append("main")
        item["response"] = Response(content=b'{}', media_type="application/json")
        item["event"].set()
    async def wrap_batch(item):
        enqueued_on.append("batch")
        item["response"] = Response(content=b'{}', media_type="application/json")
        item["event"].set()

    server.slots["main"].queue.enqueue = wrap_main
    server.slots["batch"].queue.enqueue = wrap_batch

    # "test-model-q4" exists in MODELS_DIR, loaded on main, routes to main.
    r = client.post("/v1/chat/completions", json={
        "model": "test-model-q4",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert enqueued_on == ["main"]


def test_chat_completions_503_on_batch_unhealthy(client):
    """Request for a model loaded on batch returns 503 if batch unhealthy."""
    app = client.app
    server = app.state.server
    server.slots["main"].loaded_model = "test-model-q4"
    server.slots["main"].healthy = True
    server.slots["batch"].loaded_model = "test-model-q8"
    server.slots["batch"].healthy = False  # unhealthy

    r = client.post("/v1/chat/completions", json={
        "model": "test-model-q8",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 503
    body = r.json()
    assert body["error"]["type"] == "batch_unavailable"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/test_endpoints.py -v -k route
```

Expected: FAIL (`app.state.server` doesn't exist; routing hardcoded to main).

- [ ] **Step 3: Expose `ServerState` on `app.state`**

In `create_app`, after creating `server = ServerState(config)`:

```python
    app.state.server = server  # for test introspection
```

- [ ] **Step 4: Update chat_completions to route via `resolve_slot`**

Replace the hardcoded `slot_name = "main"` line:

```python
        from manager.routing import resolve_slot
        slot_name = resolve_slot(model_name, server.slots)
        slot = server.slots[slot_name]

        if not slot.healthy and slot.loaded_model == model_name:
            # Model IS loaded on this slot but the slot is unhealthy.
            # 503 with a typed error the PAL client recognizes.
            return JSONResponse(
                {"error": {
                    "type": f"{slot_name}_unavailable",
                    "message": f"{slot_name} slot not ready",
                }},
                status_code=503,
                headers={"Retry-After": "5"},
            )
```

The implicit-main-swap case (model loaded nowhere) still routes to main. `ensure_model_on_slot` on main will do the swap via the queue consumer as today.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/test_endpoints.py -v
```

Expected: all endpoint tests pass, including the three new routing tests.

Full suite:

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/ -q
```

Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
git add manager/app.py tests/test_endpoints.py
git commit -m "feat: route chat completions to correct slot by loaded model"
```

---

### Task 13: Add `POST /swap` endpoint

**Files:**
- Modify: `manager/app.py`
- Modify: `tests/test_endpoints.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_endpoints.py`:

```python
def test_swap_valid_main(client, monkeypatch):
    app = client.app
    server = app.state.server
    # Short-circuit the actual swap; ensure_model_on_slot calls mark_swapped on success.
    async def fake_swap(self, model):
        return True
    monkeypatch.setattr("manager.swap.ModelSwapper.swap_to", fake_swap)

    r = client.post("/swap", json={"model": "test-model-q4", "target": "main"})
    assert r.status_code == 200
    body = r.json()
    assert body == {"slot": "main", "model": "test-model-q4", "status": "ok"}
    assert server.slots["main"].loaded_model == "test-model-q4"


def test_swap_valid_batch(client, monkeypatch):
    app = client.app
    server = app.state.server
    async def fake_swap(self, model):
        return True
    monkeypatch.setattr("manager.swap.ModelSwapper.swap_to", fake_swap)

    r = client.post("/swap", json={"model": "test-model-q8", "target": "batch"})
    assert r.status_code == 200
    assert r.json()["slot"] == "batch"
    assert server.slots["batch"].loaded_model == "test-model-q8"


def test_swap_default_target_is_main(client, monkeypatch):
    async def fake_swap(self, model):
        return True
    monkeypatch.setattr("manager.swap.ModelSwapper.swap_to", fake_swap)

    r = client.post("/swap", json={"model": "test-model-q4"})
    assert r.status_code == 200
    assert r.json()["slot"] == "main"


def test_swap_invalid_target(client):
    r = client.post("/swap", json={"model": "test-model-q4", "target": "xxx"})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_target"


def test_swap_missing_model(client):
    r = client.post("/swap", json={"target": "main"})
    assert r.status_code == 400


def test_swap_nonexistent_model_file(client):
    r = client.post("/swap", json={"model": "does-not-exist", "target": "main"})
    assert r.status_code == 404


def test_swap_fails_returns_503(client, monkeypatch):
    async def fake_swap(self, model):
        return False  # health timeout
    monkeypatch.setattr("manager.swap.ModelSwapper.swap_to", fake_swap)

    r = client.post("/swap", json={"model": "test-model-q4", "target": "main"})
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "swap_failed"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/test_endpoints.py -v -k swap
```

Expected: 404 or similar — endpoint doesn't exist yet.

- [ ] **Step 3: Add the `/swap` handler**

In `create_app`, add:

```python
    # ------------------------------------------------------------------
    # POST /swap
    # ------------------------------------------------------------------

    @app.post("/swap")
    async def swap_slot(request: Request):
        """Admin endpoint: swap a slot to a different model.

        Body: {"model": str, "target": "main"|"batch" (optional, default main)}.
        No auth; LAN-only is the trust boundary.
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": {"type": "invalid_body", "message": "Invalid JSON"}},
                status_code=400,
            )

        model_name = body.get("model")
        if not model_name:
            return JSONResponse(
                {"error": {"type": "missing_model", "message": "'model' is required"}},
                status_code=400,
            )

        target = body.get("target", "main")
        if target not in ("main", "batch"):
            return JSONResponse(
                {"error": {"type": "invalid_target",
                           "message": "'target' must be 'main' or 'batch'"}},
                status_code=400,
            )

        if server.model_path(model_name) is None:
            return JSONResponse(
                {"error": {"type": "model_not_found",
                           "message": f"Model file not found: {model_name}"}},
                status_code=404,
            )

        ok = await server.ensure_model_on_slot(target, model_name)
        if not ok:
            return JSONResponse(
                {"error": {"type": "swap_failed",
                           "message": f"swap to {model_name} on {target} failed"}},
                status_code=503,
            )

        return {"slot": target, "model": model_name, "status": "ok"}
```

- [ ] **Step 4: Run tests**

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/test_endpoints.py -v -k swap
```

Expected: all 7 new tests pass.

Full suite:

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/ -q
```

Expected: no regressions.

- [ ] **Step 5: Commit**

```bash
git add manager/app.py tests/test_endpoints.py
git commit -m "feat: POST /swap endpoint for explicit slot swaps"
```

---

### Task 14: Reconcile slot state on backend 5xx

**Files:**
- Modify: `manager/app.py`
- Modify: `tests/test_endpoints.py` (or a new test module)

- [ ] **Step 1: Write failing test**

Append to `tests/test_endpoints.py`:

```python
@pytest.mark.asyncio
async def test_reconcile_on_backend_5xx(test_config):
    """When the queue consumer gets a 5xx from the backend, it re-probes
    the slot and updates loaded_model if it has drifted."""
    from manager.app import ServerState
    server = ServerState(test_config)
    slot = server.slots["main"]
    slot.loaded_model = "old-model"
    slot.healthy = True

    # Fake probe that reports a different model now loaded.
    async def fake_probe(client):
        slot.loaded_model = "new-model"
        slot.healthy = True
    slot.probe = fake_probe

    # Simulate the reconcile call path.
    async with httpx.AsyncClient() as client:
        await slot.reconcile_on_error(client)

    assert slot.loaded_model == "new-model"
    assert slot.healthy is True
```

- [ ] **Step 2: Wire reconcile into the consumer's exception path**

In `_queue_consumer`, find the `except Exception as exc:` block after the proxy call. Extend it:

```python
            except Exception as exc:
                logger.exception("slot=%s proxy error: %s", slot_name, exc)
                item["error"] = f"Proxy error: {exc}"
                slot.mark_unhealthy()
                # Fire-and-forget re-probe; don't block the consumer loop.
                async def _reprobe():
                    async with httpx.AsyncClient() as probe_client:
                        await slot.reconcile_on_error(probe_client)
                asyncio.create_task(_reprobe())
```

Also handle the non-exception 5xx path. In the non-streaming branch, after receiving `resp`, before assigning to `item["response"]`:

```python
                    if resp.status_code >= 500:
                        slot.mark_unhealthy()
                        asyncio.create_task(_reprobe_for(slot))
```

Define `_reprobe_for` as a top-level helper in app.py to avoid closure noise:

```python
async def _reprobe_for(slot) -> None:
    async with httpx.AsyncClient() as probe_client:
        await slot.reconcile_on_error(probe_client)
```

For the streaming branch, reconciliation on stream errors is out of scope (streams can fail for many reasons unrelated to slot health). Document this in a comment.

- [ ] **Step 3: Run tests**

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/test_endpoints.py -v -k reconcile
```

Expected: pass.

Full suite:

```bash
cd /home/edible/Projects/inference_server && python -m pytest tests/ -q
```

Expected: no regressions.

- [ ] **Step 4: Commit**

```bash
git add manager/app.py tests/test_endpoints.py
git commit -m "feat: reconcile slot state after backend 5xx"
```

---

### Task 15: Update `config/manager.env` template

**Files:**
- Modify: `config/manager.env`

- [ ] **Step 1: Append batch-slot vars to the template**

Add to `config/manager.env`:

```
# ---------------------------------------------------------------------------
# Batch inference slot (Vulkan iGPU, Phase B)
# ---------------------------------------------------------------------------
BATCH_SERVER_HOST=127.0.0.1
BATCH_SERVER_PORT=8083
BATCH_SERVER_ENV=/etc/llama/llama-server-batch.env
BATCH_SERVER_UNIT=llama-server-batch.service
BATCH_QUEUE_LIMIT=20
BATCH_MODEL_DEFAULT=gemma-4-E4B-it-Q4_K_M
```

- [ ] **Step 2: Install on host**

```bash
sudo cp config/manager.env /etc/llama/manager.env
```

(Or manually append the new lines to the existing deployed file to preserve local tweaks.)

- [ ] **Step 3: Commit**

```bash
git add config/manager.env
git commit -m "config: manager.env template includes batch slot vars"
```

---

## Phase 3: Deployment and validation

### Task 16: Deploy manager and verify `/status` shape

**Files:** none.

- [ ] **Step 1: Pull latest code onto the host**

The inference_server repo is presumably at `~/Projects/inference_server` on the host or deployed via a symlink. Sync changes:

```bash
# on agenthost:
cd ~/Projects/inference_server
git pull
sudo cp -r manager /opt/llama/
sudo cp config/manager.env /etc/llama/manager.env
sudo systemctl restart llama-manager.service
```

(Adapt path / deploy method to whatever your existing workflow is. If `/opt/llama/manager` is a copy rather than a symlink, a rsync is needed.)

- [ ] **Step 2: Verify manager is running**

```bash
sudo systemctl status llama-manager.service
curl -s http://192.168.1.14:11434/health
```

Expected: 200 on both.

- [ ] **Step 3: Verify new `/status` shape**

```bash
curl -s http://192.168.1.14:11434/status | python3 -m json.tool
```

Expected: `slots.main` and `slots.batch` both present, each with `loaded_model` populated and `healthy: true`.

If batch shows `healthy: false`: check `sudo systemctl status llama-server-batch.service` and `journalctl -u llama-server-batch -n 50`.

- [ ] **Step 4: No commit.**

---

### Task 17: End-to-end smoke test from PAL

**Files:** none.

- [ ] **Step 1: From the PAL CLI (Discord or terminal), run `/model`**

Expected: both slots listed with their loaded models.

- [ ] **Step 2: Swap batch explicitly**

In PAL: `/model --target batch gemma-4-E4B-it-Q4_K_M`

Expected: swap succeeds. `/status` or PAL's own `/model` confirms.

- [ ] **Step 3: Enable Phase B on PAL**

Set `PAL_BATCH_ENABLED=true` in PAL's environment and restart the daemon. The env var is read by `pal/config.py` and stored as `config.batch_enabled`. Simplest approach on the PAL host:

```bash
# in PAL's systemd service override, or the env file the daemon reads:
PAL_BATCH_ENABLED=true
# then restart the PAL daemon
```

- [ ] **Step 4: Run a real compile**

In PAL, trigger a compile (e.g. `/compile` on an existing raw summary, or let the normal flow run by saving a new web fetch).

Expected:
- Categorizer fires against batch slot.
- `nvidia-smi` on the host shows the main model still loaded on the P40; no P40 activity spike for the categorize call.
- `radeontop` (or equivalent AMD utilities) shows iGPU activity during the categorize call.

- [ ] **Step 5: Simulate batch outage**

```bash
sudo systemctl stop llama-server-batch
```

In PAL, trigger another compile.

Expected: PAL's `BatchFallbackProposal` fires with `[r/m/s]` (CLI) or three-button view (Discord). Test each option:
- `[m]` / "Run on main": proceeds successfully via fallback
- `[s]` / "Skip": caller's default applies (e.g. categorize → Unfiled)
- `[r]` / "Retry": fails again after restarting batch

- [ ] **Step 6: Bring batch back up**

```bash
sudo systemctl start llama-server-batch
```

Verify `/status` shows batch healthy again.

- [ ] **Step 7: Record the results**

Update `/home/edible/Projects/PAL/docs/superpowers/runbooks/2026-04-19-phase-b-pal-verification.md` (or create equivalent on the inference_server side) with the actual observations: Vulkan perf numbers, any edge cases hit, rollback notes if anything required it.

- [ ] **Step 8: No commit.**

---

## Appendix: Observations recorded during execution

### Phase 0 findings (2026-04-20)

- **Device flag syntax:** `--device <name>` (comma-separated list supported). `--list-devices` enumerates.
- **Vulkan device mapping** (on agenthost, after adding user to `render` + `video` groups):
  - `CUDA0` = Tesla P40 (for main slot)
  - `Vulkan0` = AMD Radeon Graphics (RADV RENOIR) = Vega iGPU (for batch slot)
  - `Vulkan1` = Tesla P40 via NVIDIA's Vulkan ICD (unused by Phase B)
- **Model substitution:** spec called for Gemma 3 4B IT; bartowski URL 404'd. Substituted Gemma 4 E4B IT Q4_K_M from `unsloth/gemma-4-E4B-it-GGUF`. Same size class, newer family. Model ID when loaded: `gemma-4-E4B-it-Q4_K_M`.
- **Models dir on this server:** `/mnt/secondary/llama-models/` (MODELS_DIR env override of the default `/opt/llama/models`).
- **Shader cache:** `_llama` has no `$HOME`, so llama-server logs "Failed to create /home/_llama for shader cache — disabling." Fixed by `Environment=XDG_CACHE_HOME=/var/cache/llama` in the batch systemd unit + `sudo install -d -o _llama -g _llama /var/cache/llama`.
- **Gemma 4 E4B GGUF sha256:** `dff0ffba4c90b4082d70214d53ce9504a28d4d8d998276dcb3b8881a656c742a` (at `/mnt/secondary/llama-models/gemma-4-E4B-it-Q4_K_M.gguf`).
- **Vulkan device discovery required adding the current user (and `_llama`) to the `render` group.** Without it, RADV silently enumerates zero devices because `/dev/dri/renderD*` is render-group-owned; NVIDIA's ICD still worked because it uses `/dev/nvidia*`.

### PAL-side followups (after Phase B server ships)

- PAL's `config.batch_model` default is `gemma-3-4b-it-q4_k_m`. Update to `gemma-4-E4B-it-Q4_K_M` in `pal/config.py` or override via `PAL_BATCH_MODEL` env var on the PAL host.

### Reserved for future findings

Additional follow-ups identified during Phase 1-3 go here:
- sha256 of Gemma 4 E4B GGUF for reproducibility.
- Observed Vulkan perf baseline numbers.
- Actual `radeontop`-equivalent command that worked on agenthost.
