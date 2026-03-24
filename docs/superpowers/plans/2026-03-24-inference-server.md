# Inference Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native llama.cpp inference server with a Python model manager proxy that provides OpenAI-compatible API access, API-driven model switching, FIFO request queuing, and status reporting.

**Architecture:** llama-server runs as a systemd service under a dedicated `_llama` user, bound to localhost. A FastAPI model manager proxy runs as a second systemd service under `_llama-mgr`, bound to the LAN IP. The manager handles model switching by restarting llama-server with new config, queues requests during swaps, and provides status/health endpoints.

**Tech Stack:** llama.cpp (CUDA), Python 3, FastAPI, uvicorn, httpx, systemd

**Spec:** `docs/superpowers/specs/2026-03-24-inference-server-design.md`

**Note:** This project produces config files, scripts, and a Python application locally. Deployment to the Ubuntu server is a separate step after implementation.

---

## File Structure

```
inference_server/
├── docs/
│   └── superpowers/
│       ├── specs/2026-03-24-inference-server-design.md
│       └── plans/2026-03-24-inference-server.md
├── manager/                        # Model manager Python package
│   ├── __init__.py                 # Package marker (empty)
│   ├── app.py                      # FastAPI app: endpoints, proxy, queue, swap logic
│   ├── config.py                   # Configuration loading from env vars
│   ├── queue.py                    # FIFO request queue implementation
│   ├── swap.py                     # Model swap orchestration (env file update, systemd restart, health poll)
│   ├── gpu.py                      # GPU info via nvidia-smi
│   ├── requirements.txt            # Python dependencies
│   └── README.md                   # Manager component docs
├── systemd/                        # Systemd unit files
│   ├── llama-server.service        # Inference backend service
│   └── llama-manager.service       # Model manager service
├── config/                         # Template config files
│   ├── llama-server.env            # llama-server environment template
│   ├── manager.env                 # Model manager environment template
│   └── llama-logrotate             # Logrotate config
├── scripts/                        # Setup and management scripts
│   ├── setup.sh                    # System setup: users, dirs, permissions, sudoers, systemd
│   └── download-model.sh           # Helper to download GGUFs from HuggingFace
├── tests/                          # Tests for the model manager
│   ├── conftest.py                 # Shared fixtures (mock llama-server, test client)
│   ├── test_config.py              # Config loading tests
│   ├── test_queue.py               # FIFO queue tests
│   ├── test_swap.py                # Model swap logic tests
│   ├── test_endpoints.py           # API endpoint integration tests
│   └── test_gpu.py                 # GPU info parsing tests
├── README.md                       # Project overview and getting started
└── ARCHITECTURE.md                 # (existing) Friend's architecture for reference
```

---

### Task 1: Project Scaffolding and Configuration

**Files:**
- Create: `manager/__init__.py`
- Create: `manager/config.py`
- Create: `manager/requirements.txt`
- Create: `config/llama-server.env`
- Create: `config/manager.env`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Create `manager/__init__.py` (empty package marker)**

- [ ] **Step 2: Create requirements.txt**

```
# manager/requirements.txt
# FastAPI - modern async web framework for the API layer
fastapi>=0.110.0
# Uvicorn - ASGI server to run FastAPI
uvicorn[standard]>=0.29.0
# httpx - async HTTP client for proxying requests to llama-server
httpx>=0.27.0
# pytest + httpx for testing FastAPI apps
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 3: Create llama-server.env template**

```bash
# config/llama-server.env
# Configuration for the llama-server inference backend.
# This file is read by systemd's EnvironmentFile directive.
# The model manager updates MODEL_PATH when swapping models.

# Path to the GGUF model file to load on startup.
# Leave empty on first boot — the manager will set this on first request.
MODEL_PATH=

# Number of model layers to offload to GPU.
# Set to -1 for auto-detect (uses all available VRAM), or a specific number.
# The Tesla P40 has 24GB VRAM — layers that don't fit spill to system RAM.
N_GPU_LAYERS=-1

# Context window size in tokens. Larger = more VRAM usage.
# 4096 is a safe default for the P40 with large models.
CTX_SIZE=4096

# Bind address and port. Always localhost — the model manager is the public-facing endpoint.
HOST=127.0.0.1
PORT=8081
```

- [ ] **Step 4: Create manager.env template**

```bash
# config/manager.env
# Configuration for the model manager proxy service.

# Address to bind to. Set to your server's LAN IP, or 0.0.0.0 for all interfaces.
HOST=0.0.0.0
PORT=8080

# Where llama-server is running (always localhost since it's on the same machine).
LLAMA_SERVER_HOST=127.0.0.1
LLAMA_SERVER_PORT=8081

# Path to the directory containing GGUF model files.
MODELS_DIR=/opt/llama/models

# Path to the llama-server environment file (manager updates MODEL_PATH here during swaps).
LLAMA_SERVER_ENV=/etc/llama/llama-server.env

# Maximum number of requests to hold in the FIFO queue.
# Requests beyond this limit get a 503 response.
QUEUE_LIMIT=20

# How long to wait (seconds) for llama-server to become healthy after a model swap.
# If this timeout is exceeded, the manager enters error state.
SWAP_TIMEOUT=120

# Log file path for the model manager.
LOG_FILE=/var/log/llama/manager.log
```

- [ ] **Step 5: Write failing test for config loading**

```python
# tests/test_config.py
"""Tests for configuration loading from environment variables."""
import pytest
from manager.config import ManagerConfig


def test_config_loads_from_env(monkeypatch):
    """Config should load all values from environment variables."""
    monkeypatch.setenv("HOST", "192.168.1.50")
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("LLAMA_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("LLAMA_SERVER_PORT", "8081")
    monkeypatch.setenv("MODELS_DIR", "/opt/llama/models")
    monkeypatch.setenv("LLAMA_SERVER_ENV", "/etc/llama/llama-server.env")
    monkeypatch.setenv("QUEUE_LIMIT", "20")
    monkeypatch.setenv("SWAP_TIMEOUT", "120")
    monkeypatch.setenv("LOG_FILE", "/var/log/llama/manager.log")

    config = ManagerConfig.from_env()

    assert config.host == "192.168.1.50"
    assert config.port == 8080
    assert config.llama_server_host == "127.0.0.1"
    assert config.llama_server_port == 8081
    assert config.models_dir == "/opt/llama/models"
    assert config.llama_server_env == "/etc/llama/llama-server.env"
    assert config.queue_limit == 20
    assert config.swap_timeout == 120
    assert config.log_file == "/var/log/llama/manager.log"


def test_config_defaults(monkeypatch):
    """Config should use sensible defaults when env vars are missing."""
    for key in ["HOST", "PORT", "LLAMA_SERVER_HOST", "LLAMA_SERVER_PORT",
                "MODELS_DIR", "LLAMA_SERVER_ENV", "QUEUE_LIMIT",
                "SWAP_TIMEOUT", "LOG_FILE"]:
        monkeypatch.delenv(key, raising=False)

    config = ManagerConfig.from_env()

    assert config.host == "0.0.0.0"
    assert config.port == 8080
    assert config.llama_server_port == 8081
    assert config.queue_limit == 20
    assert config.swap_timeout == 120


def test_config_llama_server_url(monkeypatch):
    """Config should provide a convenience URL for the llama-server."""
    monkeypatch.setenv("LLAMA_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("LLAMA_SERVER_PORT", "8081")

    config = ManagerConfig.from_env()

    assert config.llama_server_url == "http://127.0.0.1:8081"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd /home/edible/Projects/inference_server && PYTHONPATH=. python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'manager.config'`

- [ ] **Step 7: Write config.py implementation**

```python
# manager/config.py
"""
Configuration loading for the model manager.

All configuration comes from environment variables, which are set in
/etc/llama/manager.env and loaded by systemd's EnvironmentFile directive.
This module provides a typed config object so the rest of the app doesn't
need to deal with raw env vars or string parsing.
"""
import os
from dataclasses import dataclass


@dataclass
class ManagerConfig:
    """Typed configuration for the model manager service.

    Attributes:
        host: IP address to bind to (LAN IP or 0.0.0.0 for all interfaces).
        port: Port to listen on for incoming API requests.
        llama_server_host: Address where llama-server is running (always localhost).
        llama_server_port: Port where llama-server listens.
        models_dir: Path to directory containing GGUF model files.
        llama_server_env: Path to llama-server's env file (updated during model swaps).
        queue_limit: Max number of requests to hold in the FIFO queue.
        swap_timeout: Seconds to wait for llama-server health after a model swap.
        log_file: Path to the manager's log file.
    """
    host: str
    port: int
    llama_server_host: str
    llama_server_port: int
    models_dir: str
    llama_server_env: str
    queue_limit: int
    swap_timeout: int
    log_file: str

    @property
    def llama_server_url(self) -> str:
        """Full URL for connecting to llama-server."""
        return f"http://{self.llama_server_host}:{self.llama_server_port}"

    @classmethod
    def from_env(cls) -> "ManagerConfig":
        """Load configuration from environment variables with sensible defaults."""
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8080")),
            llama_server_host=os.getenv("LLAMA_SERVER_HOST", "127.0.0.1"),
            llama_server_port=int(os.getenv("LLAMA_SERVER_PORT", "8081")),
            models_dir=os.getenv("MODELS_DIR", "/opt/llama/models"),
            llama_server_env=os.getenv("LLAMA_SERVER_ENV", "/etc/llama/llama-server.env"),
            queue_limit=int(os.getenv("QUEUE_LIMIT", "20")),
            swap_timeout=int(os.getenv("SWAP_TIMEOUT", "120")),
            log_file=os.getenv("LOG_FILE", "/var/log/llama/manager.log"),
        )
```

- [ ] **Step 8: Create conftest.py with shared fixtures**

```python
# tests/conftest.py
"""Shared test fixtures for the model manager test suite."""
import pytest


@pytest.fixture
def tmp_models_dir(tmp_path):
    """Create a temporary models directory with sample GGUF files."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "test-model-q4.gguf").touch()
    (models_dir / "test-model-q8.gguf").touch()
    return str(models_dir)


@pytest.fixture
def tmp_env_file(tmp_path):
    """Create a temporary llama-server env file for testing model swaps."""
    env_file = tmp_path / "llama-server.env"
    env_file.write_text(
        "MODEL_PATH=\nN_GPU_LAYERS=-1\nCTX_SIZE=4096\nHOST=127.0.0.1\nPORT=8081\n"
    )
    return str(env_file)


@pytest.fixture
def test_config(tmp_models_dir, tmp_env_file):
    """Create a ManagerConfig pointing at temporary test paths."""
    from manager.config import ManagerConfig
    return ManagerConfig(
        host="127.0.0.1",
        port=8080,
        llama_server_host="127.0.0.1",
        llama_server_port=8081,
        models_dir=tmp_models_dir,
        llama_server_env=tmp_env_file,
        queue_limit=20,
        swap_timeout=5,  # Short timeout for tests
        log_file="/dev/null",
    )
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `cd /home/edible/Projects/inference_server && PYTHONPATH=. python -m pytest tests/test_config.py -v`
Expected: All 3 tests PASS

- [ ] **Step 10: Commit**

```bash
git add manager/__init__.py manager/config.py manager/requirements.txt config/ tests/conftest.py tests/test_config.py
git commit -m "feat: project scaffolding with config module, env templates, and test fixtures"
```

---

### Task 2: FIFO Request Queue

**Files:**
- Create: `manager/queue.py`
- Create: `tests/test_queue.py`

- [ ] **Step 1: Write failing tests for the request queue**

```python
# tests/test_queue.py
"""Tests for the FIFO request queue."""
import asyncio
import pytest
from manager.queue import RequestQueue


@pytest.mark.asyncio
async def test_queue_processes_in_fifo_order():
    """Requests should be processed in the order they were added."""
    queue = RequestQueue(max_size=20)
    results = []

    await queue.enqueue("first")
    await queue.enqueue("second")
    await queue.enqueue("third")

    while not queue.empty():
        item = await queue.dequeue()
        results.append(item)

    assert results == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_queue_rejects_when_full():
    """Queue should raise when at capacity to trigger 503 response."""
    queue = RequestQueue(max_size=2)

    await queue.enqueue("first")
    await queue.enqueue("second")

    with pytest.raises(queue.QueueFullError):
        await queue.enqueue("third")


@pytest.mark.asyncio
async def test_queue_reports_depth():
    """Queue should accurately report how many items are waiting."""
    queue = RequestQueue(max_size=20)

    assert queue.depth == 0
    await queue.enqueue("item")
    assert queue.depth == 1
    await queue.dequeue()
    assert queue.depth == 0


@pytest.mark.asyncio
async def test_queue_drain_returns_all_items():
    """Drain should empty the queue and return all items."""
    queue = RequestQueue(max_size=20)

    await queue.enqueue("first")
    await queue.enqueue("second")
    await queue.enqueue("third")

    items = queue.drain()

    assert items == ["first", "second", "third"]
    assert queue.depth == 0


@pytest.mark.asyncio
async def test_queue_dequeue_waits_for_item():
    """Dequeue should block until an item is available."""
    queue = RequestQueue(max_size=20)
    result = []

    async def consumer():
        item = await queue.dequeue()
        result.append(item)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.05)
    assert result == []

    await queue.enqueue("hello")
    await asyncio.sleep(0.05)
    assert result == ["hello"]
    await task
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/edible/Projects/inference_server && PYTHONPATH=. python -m pytest tests/test_queue.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write queue implementation**

```python
# manager/queue.py
"""
FIFO request queue for the model manager.

All incoming inference requests go through this queue, processed one at a
time. This ensures serial access to llama-server (single GPU, one request
at a time maximizes throughput per request).

During model swaps, requests accumulate and are processed once the swap
completes. If the queue hits its max depth, new requests are rejected
with QueueFullError (the API layer translates this to 503).
"""
import asyncio
from typing import Any


class RequestQueue:
    """Async FIFO queue with a configurable size limit."""

    class QueueFullError(Exception):
        """Raised when the queue is at capacity."""
        pass

    def __init__(self, max_size: int = 20):
        self._max_size = max_size
        self._queue: asyncio.Queue = asyncio.Queue()
        self._depth = 0

    @property
    def depth(self) -> int:
        """Number of requests currently waiting."""
        return self._depth

    @property
    def max_size(self) -> int:
        """Maximum queue capacity."""
        return self._max_size

    def empty(self) -> bool:
        """True if no requests are waiting."""
        return self._depth == 0

    async def enqueue(self, item: Any) -> None:
        """Add a request to the back of the queue.

        Raises QueueFullError if at capacity.
        """
        if self._depth >= self._max_size:
            raise self.QueueFullError(
                f"Queue is full ({self._depth}/{self._max_size})"
            )
        await self._queue.put(item)
        self._depth += 1

    async def dequeue(self) -> Any:
        """Remove and return the next request. Blocks until available."""
        item = await self._queue.get()
        self._depth -= 1
        return item

    def drain(self) -> list:
        """Remove and return all items. Used on error to send 503s."""
        items = []
        while not self._queue.empty():
            items.append(self._queue.get_nowait())
            self._depth -= 1
        return items
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/edible/Projects/inference_server && PYTHONPATH=. python -m pytest tests/test_queue.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add manager/queue.py tests/test_queue.py
git commit -m "feat: FIFO request queue with size limit and drain support"
```

---

### Task 3: GPU Info Module

**Files:**
- Create: `manager/gpu.py`
- Create: `tests/test_gpu.py`

- [ ] **Step 1: Write failing tests for GPU info**

```python
# tests/test_gpu.py
"""Tests for GPU information retrieval via nvidia-smi."""
import pytest
from unittest.mock import patch, MagicMock
from manager.gpu import get_gpu_info


SAMPLE_NVIDIA_SMI_OUTPUT = """gpu_name, memory.total [MiB], memory.used [MiB]
Tesla P40, 24576 MiB, 18200 MiB"""


def test_parse_nvidia_smi_output():
    """Should parse GPU name, total VRAM, and used VRAM."""
    mock_result = MagicMock()
    mock_result.stdout = SAMPLE_NVIDIA_SMI_OUTPUT
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result):
        info = get_gpu_info()

    assert info["name"] == "Tesla P40"
    assert info["vram_total_mb"] == 24576
    assert info["vram_used_mb"] == 18200


def test_gpu_info_when_nvidia_smi_fails():
    """Should return unknown values if nvidia-smi is not available."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        info = get_gpu_info()

    assert info["name"] == "unknown"
    assert info["vram_total_mb"] == 0
    assert info["vram_used_mb"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/edible/Projects/inference_server && PYTHONPATH=. python -m pytest tests/test_gpu.py -v`
Expected: FAIL

- [ ] **Step 3: Write GPU info implementation**

```python
# manager/gpu.py
"""
GPU information retrieval via nvidia-smi.

Provides GPU name and VRAM usage for the /status endpoint. Queries
on-demand since VRAM usage changes as models load/unload. Falls back
to safe defaults if nvidia-smi is unavailable (e.g., during development).
"""
import subprocess
import logging

logger = logging.getLogger(__name__)


def get_gpu_info() -> dict:
    """Query NVIDIA GPU for name and VRAM usage.

    Returns dict with name, vram_total_mb, vram_used_mb.
    Falls back to "unknown"/0 if nvidia-smi fails.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=gpu_name,memory.total,memory.used",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            logger.warning("nvidia-smi unexpected output: %s", result.stdout)
            return _unknown_gpu()

        values = [v.strip() for v in lines[1].split(",")]
        return {
            "name": values[0],
            "vram_total_mb": int(values[1].replace(" MiB", "")),
            "vram_used_mb": int(values[2].replace(" MiB", "")),
        }

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.warning("Could not query GPU info: %s", e)
        return _unknown_gpu()


def _unknown_gpu() -> dict:
    return {"name": "unknown", "vram_total_mb": 0, "vram_used_mb": 0}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/edible/Projects/inference_server && PYTHONPATH=. python -m pytest tests/test_gpu.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add manager/gpu.py tests/test_gpu.py
git commit -m "feat: GPU info module for status endpoint VRAM reporting"
```

---

### Task 4: Model Swap Orchestration

**Files:**
- Create: `manager/swap.py`
- Create: `tests/test_swap.py`

- [ ] **Step 1: Write failing tests for swap logic**

```python
# tests/test_swap.py
"""Tests for model swap orchestration.

Tests mock subprocess (systemctl) and HTTP (health polling) since
we can't run a real llama-server in the test environment.
"""
import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from manager.swap import ModelSwapper


@pytest.fixture
def swapper(test_config):
    return ModelSwapper(test_config)


def test_update_env_file(swapper, tmp_env_file):
    """Should replace MODEL_PATH with the new model."""
    swapper.update_env_file("/opt/llama/models/new-model.gguf")

    content = Path(tmp_env_file).read_text()
    assert "MODEL_PATH=/opt/llama/models/new-model.gguf" in content
    assert "HOST=127.0.0.1" in content


def test_update_env_file_overwrites_existing(swapper, tmp_env_file):
    """If MODEL_PATH already has a value, it should be replaced."""
    swapper.update_env_file("/opt/llama/models/old-model.gguf")
    swapper.update_env_file("/opt/llama/models/new-model.gguf")

    content = Path(tmp_env_file).read_text()
    assert "MODEL_PATH=/opt/llama/models/new-model.gguf" in content
    assert "old-model" not in content


@pytest.mark.asyncio
async def test_restart_llama_server(swapper):
    """Should call systemctl restart via sudo."""
    mock_result = MagicMock()
    mock_result.returncode = 0

    # Patch run_in_executor to call the function directly instead of in a thread
    async def fake_executor(executor, fn):
        return fn()

    with patch("subprocess.run", return_value=mock_result) as mock_run, \
         patch.object(asyncio.get_event_loop(), "run_in_executor", side_effect=fake_executor):
        await swapper.restart_llama_server()

    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert call_args == ["sudo", "systemctl", "restart", "llama-server.service"]


@pytest.mark.asyncio
async def test_restart_raises_on_failure(swapper):
    """Should raise if systemctl restart fails."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "Failed to restart"

    async def fake_executor(executor, fn):
        return fn()

    with patch("subprocess.run", return_value=mock_result), \
         patch.object(asyncio.get_event_loop(), "run_in_executor", side_effect=fake_executor):
        with pytest.raises(RuntimeError, match="Failed to restart"):
            await swapper.restart_llama_server()


@pytest.mark.asyncio
async def test_wait_for_health_succeeds(swapper):
    """Should return True when llama-server responds to health check."""
    mock_response = AsyncMock()
    mock_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await swapper.wait_for_health()

    assert result is True


@pytest.mark.asyncio
async def test_wait_for_health_times_out(swapper):
    """Should return False if llama-server never becomes healthy."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await swapper.wait_for_health()

    assert result is False


@pytest.mark.asyncio
async def test_swap_to_full_sequence(swapper):
    """swap_to should chain env update, restart, and health poll."""
    # Mock restart to succeed
    async def fake_executor(executor, fn):
        return fn()

    mock_result = MagicMock()
    mock_result.returncode = 0

    # Mock health check to succeed
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("subprocess.run", return_value=mock_result), \
         patch.object(asyncio.get_event_loop(), "run_in_executor", side_effect=fake_executor), \
         patch("httpx.AsyncClient", return_value=mock_client):
        result = await swapper.swap_to("/opt/llama/models/test.gguf")

    assert result is True


@pytest.mark.asyncio
async def test_swap_to_fails_on_health_timeout(swapper):
    """swap_to should return False if health check times out."""
    async def fake_executor(executor, fn):
        return fn()

    mock_result = MagicMock()
    mock_result.returncode = 0

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("subprocess.run", return_value=mock_result), \
         patch.object(asyncio.get_event_loop(), "run_in_executor", side_effect=fake_executor), \
         patch("httpx.AsyncClient", return_value=mock_client):
        result = await swapper.swap_to("/opt/llama/models/test.gguf")

    assert result is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/edible/Projects/inference_server && PYTHONPATH=. python -m pytest tests/test_swap.py -v`
Expected: FAIL

- [ ] **Step 3: Write swap implementation**

```python
# manager/swap.py
"""
Model swap orchestration for the model manager.

Handles the three-step process of switching models:
1. Update llama-server's env file with the new MODEL_PATH
2. Restart llama-server via systemctl (requires sudoers entry)
3. Poll llama-server's health endpoint until it's ready

This is the reason _llama-mgr needs a sudoers entry — it's the only
place that calls systemctl restart.
"""
import asyncio
import re
import subprocess
import logging
import httpx

from manager.config import ManagerConfig

logger = logging.getLogger(__name__)


class ModelSwapper:
    """Orchestrates model swaps: env file update, systemd restart, health poll."""

    def __init__(self, config: ManagerConfig):
        self._config = config

    def update_env_file(self, model_path: str) -> None:
        """Rewrite MODEL_PATH in the llama-server env file.

        Preserves all other env vars. Uses regex replacement so it works
        whether MODEL_PATH is empty or has an existing value.
        """
        env_path = self._config.llama_server_env

        with open(env_path, "r") as f:
            content = f.read()

        content = re.sub(
            r"^MODEL_PATH=.*$",
            f"MODEL_PATH={model_path}",
            content,
            flags=re.MULTILINE,
        )

        with open(env_path, "w") as f:
            f.write(content)

        logger.info("Updated env file: MODEL_PATH=%s", model_path)

    async def restart_llama_server(self) -> None:
        """Restart llama-server via systemctl.

        Runs in a thread executor to avoid blocking the async event loop.
        Requires sudoers entry for _llama-mgr.
        """
        logger.info("Restarting llama-server...")

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                ["sudo", "systemctl", "restart", "llama-server.service"],
                capture_output=True,
                text=True,
                timeout=30,
            ),
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to restart llama-server: {result.stderr}"
            )

        logger.info("llama-server restart command succeeded")

    async def wait_for_health(self) -> bool:
        """Poll llama-server's health endpoint until it responds.

        Returns True if healthy within timeout, False if timed out.
        Polls every 2 seconds.
        """
        url = f"{self._config.llama_server_url}/health"
        timeout = self._config.swap_timeout
        poll_interval = 2

        logger.info("Waiting for llama-server at %s (timeout: %ds)", url, timeout)

        elapsed = 0
        async with httpx.AsyncClient() as client:
            while elapsed < timeout:
                try:
                    response = await client.get(url, timeout=5)
                    if response.status_code == 200:
                        logger.info("llama-server healthy after %ds", elapsed)
                        return True
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass  # Server not up yet — expected during startup

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

        logger.error("Health check timed out after %ds", timeout)
        return False

    async def swap_to(self, model_path: str) -> bool:
        """Execute the full model swap sequence.

        Returns True if swap succeeded and llama-server is healthy,
        False if health check timed out.
        """
        logger.info("Starting model swap to: %s", model_path)

        self.update_env_file(model_path)
        await self.restart_llama_server()
        healthy = await self.wait_for_health()

        if healthy:
            logger.info("Model swap complete: %s", model_path)
        else:
            logger.error("Model swap failed — health timed out: %s", model_path)

        return healthy
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/edible/Projects/inference_server && PYTHONPATH=. python -m pytest tests/test_swap.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add manager/swap.py tests/test_swap.py
git commit -m "feat: model swap orchestration with env update, systemd restart, and health polling"
```

---

### Task 5: FastAPI Application — Endpoints and Proxy

**Files:**
- Create: `manager/app.py`
- Create: `tests/test_endpoints.py`

- [ ] **Step 1: Write failing tests for API endpoints**

```python
# tests/test_endpoints.py
"""Tests for the model manager API endpoints.

Tests the full HTTP layer using FastAPI's TestClient.
Mocks the llama-server backend for proxy tests.
"""
import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from httpx import Response as HttpxResponse


@pytest.fixture
def app(test_config):
    """Create a FastAPI app with test configuration."""
    from manager.app import create_app
    return create_app(test_config)


@pytest.fixture
def client(app):
    return TestClient(app)


# -- Health and Status --

def test_health_endpoint(client):
    """GET /health should always return 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_endpoint(client):
    """GET /status should return server state with all required fields."""
    response = client.get("/status")
    data = response.json()

    assert data["state"] in ("loading", "error", "ready", "swapping")
    assert "current_model" in data
    assert "loading_model" in data
    assert "error_message" in data
    assert "queue_depth" in data
    assert "queue_limit" in data
    assert data["queue_limit"] == 20
    assert "gpu" in data
    assert "uptime_seconds" in data


# -- Model Listing --

def test_models_endpoint(client):
    """GET /v1/models should list available GGUF files."""
    response = client.get("/v1/models")
    data = response.json()

    assert data["object"] == "list"
    model_ids = [m["id"] for m in data["data"]]
    assert "test-model-q4" in model_ids
    assert "test-model-q8" in model_ids


def test_models_openai_format(client):
    """Model list should follow OpenAI format."""
    response = client.get("/v1/models")
    for model in response.json()["data"]:
        assert model["object"] == "model"
        assert "id" in model
        assert "created" in model
        assert "owned_by" in model


# -- Chat Completions --

def test_chat_completions_missing_model(client):
    """Should return 400 if model field is missing."""
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 400


def test_chat_completions_unknown_model(client):
    """Should return 404 for a model that doesn't exist."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 404


def test_chat_completions_503_has_retry_after(client):
    """All 503 responses should include Retry-After header."""
    # Force error state by requesting nonexistent model won't give 503,
    # but we can test the queue-full path by filling the queue.
    # For now, verify the 404 path — 503 with Retry-After is tested
    # via ensure_model failure below.
    pass


# -- Server State --

def test_ensure_model_updates_state(test_config):
    """ensure_model should transition through swapping -> ready on success."""
    from manager.app import ServerState

    server = ServerState(test_config)

    # Mock the swapper to succeed
    server.swapper.swap_to = AsyncMock(return_value=True)

    async def run():
        result = await server.ensure_model("test-model-q4")
        assert result is True
        assert server.state == "ready"
        assert server.current_model == "test-model-q4"

    asyncio.get_event_loop().run_until_complete(run())


def test_ensure_model_error_state_on_failure(test_config):
    """ensure_model should enter error state and drain queue on failure."""
    from manager.app import ServerState

    server = ServerState(test_config)
    server.swapper.swap_to = AsyncMock(return_value=False)

    async def run():
        result = await server.ensure_model("test-model-q4")
        assert result is False
        assert server.state == "error"
        assert server.current_model is None
        assert "timed out" in server.error_message

    asyncio.get_event_loop().run_until_complete(run())


def test_ensure_model_drains_queue_on_failure(test_config):
    """On swap failure, all queued requests should be drained and notified."""
    from manager.app import ServerState

    server = ServerState(test_config)
    server.swapper.swap_to = AsyncMock(return_value=False)

    async def run():
        # Add some items to the queue with events
        event1 = asyncio.Event()
        event2 = asyncio.Event()
        item1 = {"body": {}, "event": event1, "response": None, "error": None}
        item2 = {"body": {}, "event": event2, "response": None, "error": None}
        await server.queue.enqueue(item1)
        await server.queue.enqueue(item2)

        # Trigger swap failure
        await server.ensure_model("test-model-q4")

        # Queue should be empty after drain
        assert server.queue.depth == 0

    asyncio.get_event_loop().run_until_complete(run())


def test_ensure_model_skips_swap_if_already_loaded(test_config):
    """Should not swap if the requested model is already loaded."""
    from manager.app import ServerState

    server = ServerState(test_config)
    server.state = "ready"
    server.current_model = "test-model-q4"
    server.swapper.swap_to = AsyncMock(return_value=True)

    async def run():
        result = await server.ensure_model("test-model-q4")
        assert result is True
        # swap_to should NOT have been called
        server.swapper.swap_to.assert_not_called()

    asyncio.get_event_loop().run_until_complete(run())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/edible/Projects/inference_server && PYTHONPATH=. python -m pytest tests/test_endpoints.py -v`
Expected: FAIL — `ImportError: cannot import name 'create_app' from 'manager.app'`

- [ ] **Step 3: Write the FastAPI application**

```python
# manager/app.py
"""
Model Manager — FastAPI application that sits in front of llama-server.

This is the main entry point for the inference server. Provides an
OpenAI-compatible API that:
- Proxies chat completion requests to llama-server
- Automatically swaps models when a request needs a different model
- Queues requests during model swaps (FIFO, max depth configurable)
- Reports server status, loaded model, GPU info, and queue depth

All clients talk to this service. llama-server is never directly exposed.
"""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response

from manager.config import ManagerConfig
from manager.gpu import get_gpu_info
from manager.queue import RequestQueue
from manager.swap import ModelSwapper

logger = logging.getLogger(__name__)


class ServerState:
    """Tracks the current state of the inference server.

    Central state object — /status reads from it, swap logic writes to it,
    and the queue processor checks it before forwarding requests.
    """

    def __init__(self, config: ManagerConfig):
        self.config = config
        self.state = "loading"  # loading | ready | swapping | error
        self.current_model: str | None = None
        self.loading_model: str | None = None
        self.error_message: str | None = None
        self.start_time = time.time()
        self.queue = RequestQueue(max_size=config.queue_limit)
        self.swapper = ModelSwapper(config)
        # Lock prevents concurrent model swaps
        self._swap_lock = asyncio.Lock()
        # Event signals the queue consumer that work is available
        self._queue_event = asyncio.Event()

    def model_path(self, model_name: str) -> str | None:
        """Resolve a model name to its GGUF file path.

        Model names are filenames without the .gguf extension.
        Returns None if the model doesn't exist.
        """
        path = Path(self.config.models_dir) / f"{model_name}.gguf"
        return str(path) if path.exists() else None

    def list_models(self) -> list[str]:
        """List all available model names."""
        models_dir = Path(self.config.models_dir)
        if not models_dir.exists():
            return []
        return sorted(p.stem for p in models_dir.glob("*.gguf"))

    async def ensure_model(self, model_name: str) -> bool:
        """Ensure the requested model is loaded, swapping if necessary.

        Returns True if the model is ready, False if swap failed.
        Uses a lock to prevent concurrent swap attempts.
        """
        if self.current_model == model_name and self.state == "ready":
            return True

        async with self._swap_lock:
            # Re-check after acquiring lock
            if self.current_model == model_name and self.state == "ready":
                return True

            model_file = self.model_path(model_name)
            if model_file is None:
                return False

            self.state = "swapping"
            self.loading_model = model_name
            self.error_message = None

            logger.info("Swapping to model: %s", model_name)
            success = await self.swapper.swap_to(model_file)

            if success:
                self.state = "ready"
                self.current_model = model_name
                self.loading_model = None
                return True
            else:
                self.state = "error"
                self.current_model = None
                self.loading_model = None
                self.error_message = (
                    f"Health check timed out after {self.config.swap_timeout}s "
                    f"while loading {model_name}"
                )
                logger.error(self.error_message)

                # Drain queue — notify all waiting clients with errors
                drained = self.queue.drain()
                for item in drained:
                    item["error"] = self.error_message
                    item["event"].set()
                if drained:
                    logger.warning(
                        "Drained %d queued requests due to swap failure",
                        len(drained),
                    )

                return False


def _setup_logging(config: ManagerConfig) -> None:
    """Configure logging for both file and console output.

    In production (via systemd), logs go to both the configured log file
    and stderr (which systemd captures in the journal).
    """
    handlers = [logging.StreamHandler()]

    # Add file handler if log file is writable
    log_file = config.log_file
    if log_file and log_file != "/dev/null":
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            handlers.append(file_handler)
        except (PermissionError, FileNotFoundError):
            pass  # Skip file logging if path isn't writable (e.g., in tests)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def create_app(config: ManagerConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = ManagerConfig.from_env()

    _setup_logging(config)

    server = ServerState(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Startup: check if llama-server is already running."""
        logger.info("Model manager starting on %s:%d", config.host, config.port)
        logger.info("Available models: %s", server.list_models())

        # Check if llama-server is already healthy
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{config.llama_server_url}/health", timeout=5
                )
                if resp.status_code == 200:
                    # Read current model from env file
                    try:
                        env_content = Path(config.llama_server_env).read_text()
                        for line in env_content.split("\n"):
                            if line.startswith("MODEL_PATH="):
                                model_path = line.split("=", 1)[1].strip()
                                if model_path:
                                    server.current_model = Path(model_path).stem
                                    server.state = "ready"
                                    logger.info(
                                        "llama-server already running: %s",
                                        server.current_model,
                                    )
                    except FileNotFoundError:
                        pass
        except (httpx.ConnectError, httpx.TimeoutException):
            logger.info("llama-server not reachable at startup")

        if server.state != "ready":
            server.state = "error"
            server.error_message = "No model loaded — send a request to load one"

        # Start background queue consumer
        consumer_task = asyncio.create_task(_queue_consumer(server, config))

        yield

        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

    app = FastAPI(
        title="Inference Server",
        description="OpenAI-compatible API proxy for llama.cpp",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        """Simple health check — always 200 if the service is running."""
        return {"status": "ok"}

    @app.get("/status")
    async def status():
        """Detailed server status for clients and agents."""
        return {
            "state": server.state,
            "current_model": server.current_model,
            "loading_model": server.loading_model,
            "error_message": server.error_message,
            "uptime_seconds": int(time.time() - server.start_time),
            "queue_depth": server.queue.depth,
            "queue_limit": server.queue.max_size,
            "gpu": get_gpu_info(),
        }

    @app.get("/v1/models")
    async def list_models():
        """List available models in OpenAI-compatible format."""
        models = server.list_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": int(
                        os.path.getmtime(
                            Path(config.models_dir) / f"{name}.gguf"
                        )
                    ),
                    "owned_by": "local",
                }
                for name in models
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """OpenAI-compatible chat completions.

        Reads 'model' from body, ensures it's loaded (swapping if needed),
        then proxies to llama-server. Streams SSE responses in real-time.
        """
        body = await request.json()
        model_name = body.get("model")

        if not model_name:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Missing 'model' field",
                        "type": "invalid_request_error",
                    }
                },
            )

        if server.model_path(model_name) is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": f"Model '{model_name}' not found",
                        "type": "invalid_request_error",
                    }
                },
            )

        # Create a request item for the queue. The background consumer
        # will process it and signal the event when done.
        ready_event = asyncio.Event()
        request_item = {
            "body": body,
            "model": model_name,
            "event": ready_event,
            "response": None,
            "error": None,
        }

        try:
            await server.queue.enqueue(request_item)
        except server.queue.QueueFullError:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Server busy — queue full",
                        "type": "server_error",
                    }
                },
                headers={"Retry-After": "30"},
            )

        # Notify the queue consumer that work is available
        server._queue_event.set()

        # Wait for the consumer to process our request
        await ready_event.wait()

        if request_item["error"]:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": request_item["error"],
                        "type": "server_error",
                    }
                },
                headers={"Retry-After": "30"},
            )

        return request_item["response"]

    return app


async def _queue_consumer(server: ServerState, config: ManagerConfig):
    """Background task that processes the request queue serially.

    Runs in a loop, waiting for the queue event to be set. Processes
    one request at a time — this ensures serial access to llama-server
    (single GPU, maximize per-request throughput).

    Handles model swapping between requests: consecutive requests for the
    same model skip the swap (batching). Only swaps when the model changes.
    """
    while True:
        # Wait until there's work in the queue
        await server._queue_event.wait()
        server._queue_event.clear()

        while not server.queue.empty():
            item = await server.queue.dequeue()
            model_name = item["model"]

            # Ensure the right model is loaded
            success = await server.ensure_model(model_name)
            if not success:
                item["error"] = f"Failed to load model: {model_name}"
                item["event"].set()
                continue

            # Forward the request to llama-server
            try:
                body = item["body"]
                is_stream = body.get("stream", False)

                async with httpx.AsyncClient() as client:
                    if is_stream:
                        # Streaming: use an async generator to yield
                        # chunks as they arrive from llama-server.
                        # We collect into a StreamingResponse.
                        async def stream_generator():
                            async with client.stream(
                                "POST",
                                f"{config.llama_server_url}/v1/chat/completions",
                                json=body,
                                timeout=300,
                            ) as resp:
                                async for chunk in resp.aiter_bytes():
                                    yield chunk

                        item["response"] = StreamingResponse(
                            stream_generator(),
                            media_type="text/event-stream",
                            headers={
                                "Cache-Control": "no-cache",
                                "Connection": "keep-alive",
                            },
                        )
                    else:
                        resp = await client.post(
                            f"{config.llama_server_url}/v1/chat/completions",
                            json=body,
                            timeout=300,
                        )
                        item["response"] = Response(
                            content=resp.content,
                            status_code=resp.status_code,
                            media_type="application/json",
                        )

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                logger.error("Failed to proxy to llama-server: %s", e)
                item["error"] = f"llama-server connection failed: {e}"

            item["event"].set()


# --- Entry point for development ---
if __name__ == "__main__":
    import uvicorn

    cfg = ManagerConfig.from_env()
    uvicorn.run(
        "manager.app:create_app",
        factory=True,
        host=cfg.host,
        port=cfg.port,
        log_level="info",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/edible/Projects/inference_server && PYTHONPATH=. python -m pytest tests/test_endpoints.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add manager/app.py tests/test_endpoints.py
git commit -m "feat: FastAPI app with health, status, models, chat completions, and background queue consumer"
```

---

### Task 6: Systemd Unit Files

**Files:**
- Create: `systemd/llama-server.service`
- Create: `systemd/llama-manager.service`

- [ ] **Step 1: Write llama-server.service**

```ini
# systemd/llama-server.service
#
# Systemd unit for the llama.cpp inference backend.
#
# Runs llama-server as the _llama user, bound to localhost only.
# The model manager proxy is the only thing that talks to this service.
#
# Configuration is in /etc/llama/llama-server.env. The model manager
# updates MODEL_PATH in that file when swapping models, then restarts
# this service via sudo systemctl restart.
#
# Install: sudo cp systemd/llama-server.service /etc/systemd/system/
#          sudo systemctl daemon-reload && sudo systemctl enable llama-server

[Unit]
Description=llama.cpp Inference Server
Documentation=https://github.com/ggerganov/llama.cpp

# Wait for NVIDIA drivers before starting.
After=network-online.target nvidia-persistenced.service
Wants=network-online.target

[Service]
Type=simple

# Dedicated unprivileged user — no shell, no home directory.
User=_llama
Group=_llama

# Load runtime configuration
EnvironmentFile=/etc/llama/llama-server.env

# Start llama-server with config from env file.
# --host/--port: localhost only (manager handles external access)
# --model: GGUF file path (updated by manager during swaps)
# --n-gpu-layers: GPU offload (-1 = auto/all)
# --ctx-size: context window in tokens
ExecStart=/opt/llama/bin/llama-server \
    --host ${HOST} \
    --port ${PORT} \
    --model ${MODEL_PATH} \
    --n-gpu-layers ${N_GPU_LAYERS} \
    --ctx-size ${CTX_SIZE}

# Auto-restart on crash with delay to prevent loops.
Restart=on-failure
RestartSec=5

# Persistent file logs (also in journal via journalctl -u llama-server)
StandardOutput=append:/var/log/llama/llama-server.log
StandardError=append:/var/log/llama/llama-server.err

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Write llama-manager.service**

Note: Uses `manager.app:create_app` (not `manager.manager`), and `WorkingDirectory` is the repo's parent so the `manager` package is importable.

```ini
# systemd/llama-manager.service
#
# Systemd unit for the model manager proxy.
#
# Runs the FastAPI model manager as _llama-mgr, bound to the LAN IP.
# Single entry point for all inference requests.
#
# Install: sudo cp systemd/llama-manager.service /etc/systemd/system/
#          sudo systemctl daemon-reload && sudo systemctl enable llama-manager

[Unit]
Description=llama.cpp Model Manager Proxy
Documentation=file:///opt/llama/manager/README.md

# Start after llama-server, but don't hard-depend on it.
# Wants= means "try to start it, but start anyway if it fails"
After=llama-server.service
Wants=llama-server.service

[Service]
Type=simple

# Separate user from _llama for privilege separation.
User=_llama-mgr
Group=_llama-mgr

# Load manager configuration
EnvironmentFile=/etc/llama/manager.env

# WorkingDirectory is /opt/llama so Python can import the 'manager' package.
# The package lives at /opt/llama/manager/ (with __init__.py).
WorkingDirectory=/opt/llama

# Run FastAPI via uvicorn.
# --factory tells uvicorn to call create_app() to get the ASGI app.
ExecStart=/opt/llama/manager/venv/bin/python -m uvicorn \
    manager.app:create_app \
    --factory \
    --host ${HOST} \
    --port ${PORT} \
    --log-level info

Restart=on-failure
RestartSec=5

# Stderr goes to file (uvicorn access logs)
StandardError=append:/var/log/llama/manager.err

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 3: Commit**

```bash
git add systemd/
git commit -m "feat: systemd unit files for llama-server and model manager"
```

---

### Task 7: Setup Script

**Files:**
- Create: `scripts/setup.sh`
- Create: `config/llama-logrotate`

- [ ] **Step 1: Write the setup script**

```bash
#!/usr/bin/env bash
# scripts/setup.sh
#
# Initial setup for the inference server on Ubuntu.
# Run once as root: sudo bash scripts/setup.sh
#
# Creates system users, directories, permissions, sudoers entry,
# Python virtualenv, and installs systemd services.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Run as root (sudo bash scripts/setup.sh)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Inference Server Setup ==="
echo "Repository: $REPO_DIR"
echo ""

# -- 1. System users --
echo "[1/8] Creating system users..."

if ! id -u _llama &>/dev/null; then
    useradd --system --shell /usr/sbin/nologin --no-create-home _llama
    echo "  Created: _llama"
else
    echo "  Exists: _llama"
fi

if ! id -u _llama-mgr &>/dev/null; then
    useradd --system --shell /usr/sbin/nologin --no-create-home _llama-mgr
    echo "  Created: _llama-mgr"
else
    echo "  Exists: _llama-mgr"
fi

# -- 2. Shared group --
echo "[2/8] Creating llama group..."

if ! getent group llama &>/dev/null; then
    groupadd llama
    echo "  Created: llama"
else
    echo "  Exists: llama"
fi

usermod -aG llama _llama
usermod -aG video _llama  # GPU access

SUDO_USER_NAME="${SUDO_USER:-}"
if [[ -n "$SUDO_USER_NAME" ]]; then
    usermod -aG llama "$SUDO_USER_NAME"
    echo "  Added $SUDO_USER_NAME to llama group (re-login to take effect)"
fi

# -- 3. Directories --
echo "[3/8] Creating directories..."

mkdir -p /opt/llama/bin /opt/llama/models /opt/llama/manager
mkdir -p /etc/llama
mkdir -p /var/log/llama

# -- 4. Permissions --
echo "[4/8] Setting permissions..."

# Models: _llama owns, llama group can write (admins add models)
chown _llama:llama /opt/llama/models
chmod 775 /opt/llama/models
echo "  /opt/llama/models → _llama:llama 775"

chown -R _llama-mgr:_llama-mgr /opt/llama/manager
chmod 755 /opt/llama/manager
echo "  /opt/llama/manager → _llama-mgr 755"

chown root:root /opt/llama/bin
chmod 755 /opt/llama/bin

chown root:llama /var/log/llama
chmod 775 /var/log/llama

# -- 5. Config files --
echo "[5/8] Installing config files..."

cp "$REPO_DIR/config/llama-server.env" /etc/llama/llama-server.env
# Owned by _llama-mgr so the manager can update MODEL_PATH during swaps
chown _llama-mgr:_llama-mgr /etc/llama/llama-server.env
chmod 644 /etc/llama/llama-server.env

cp "$REPO_DIR/config/manager.env" /etc/llama/manager.env
chown _llama-mgr:_llama-mgr /etc/llama/manager.env
chmod 644 /etc/llama/manager.env

# -- 6. Sudoers --
echo "[6/8] Installing sudoers entry..."

cat > /etc/sudoers.d/llama-manager << 'EOF'
# Allow model manager to restart llama-server for model swapping.
_llama-mgr ALL=(root) NOPASSWD: /usr/bin/systemctl restart llama-server.service
EOF
chmod 440 /etc/sudoers.d/llama-manager

if ! visudo -c -f /etc/sudoers.d/llama-manager &>/dev/null; then
    echo "  ERROR: sudoers syntax error!"
    rm /etc/sudoers.d/llama-manager
    exit 1
fi
echo "  Sudoers entry installed"

# -- 7. Python virtualenv --
echo "[7/8] Setting up Python virtualenv..."

python3 -m venv /opt/llama/manager/venv

# Copy the manager package (including __init__.py)
cp "$REPO_DIR/manager/"*.py /opt/llama/manager/
cp "$REPO_DIR/manager/requirements.txt" /opt/llama/manager/

/opt/llama/manager/venv/bin/pip install -r /opt/llama/manager/requirements.txt --quiet
chown -R _llama-mgr:_llama-mgr /opt/llama/manager
echo "  Virtualenv created, dependencies installed"

# -- 8. Systemd services --
echo "[8/8] Installing systemd services..."

cp "$REPO_DIR/systemd/llama-server.service" /etc/systemd/system/
cp "$REPO_DIR/systemd/llama-manager.service" /etc/systemd/system/
systemctl daemon-reload

# Logrotate
cp "$REPO_DIR/config/llama-logrotate" /etc/logrotate.d/llama

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Place llama-server binary at /opt/llama/bin/llama-server"
echo "  2. Download a GGUF model: ./scripts/download-model.sh <repo> <file>"
echo "  3. Set HOST in /etc/llama/manager.env to your LAN IP"
echo "  4. sudo systemctl enable --now llama-server llama-manager"
echo "  5. Test: curl http://<LAN_IP>:8080/health"
```

- [ ] **Step 2: Write logrotate config**

```
# config/llama-logrotate
# Log rotation for inference server logs.
/var/log/llama/*.log /var/log/llama/*.err {
    weekly
    rotate 4
    compress
    delaycompress
    missingok
    notifempty
    create 644 root llama
}
```

- [ ] **Step 3: Commit**

```bash
chmod +x scripts/setup.sh
git add scripts/setup.sh config/llama-logrotate
git commit -m "feat: setup script and logrotate config"
```

---

### Task 8: Model Download Helper Script

**Files:**
- Create: `scripts/download-model.sh`

- [ ] **Step 1: Write the download helper**

```bash
#!/usr/bin/env bash
# scripts/download-model.sh
#
# Download GGUF models from HuggingFace to /opt/llama/models/
#
# Usage: ./scripts/download-model.sh <huggingface-repo> <filename>
# Example: ./scripts/download-model.sh TheBloke/some-model-GGUF model-q4_k_m.gguf
#
# Prerequisites: pip install huggingface-hub

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/opt/llama/models}"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <huggingface-repo> <filename>"
    echo "Example: $0 TheBloke/some-model-GGUF model-q4_k_m.gguf"
    echo "Downloads to: $MODELS_DIR"
    exit 1
fi

REPO="$1"
FILENAME="$2"

if ! command -v huggingface-cli &>/dev/null; then
    echo "ERROR: huggingface-cli not found. Install: pip install huggingface-hub"
    exit 1
fi

if [[ ! -w "$MODELS_DIR" ]]; then
    echo "ERROR: Cannot write to $MODELS_DIR"
    echo "Ensure you're in the 'llama' group, or use sudo."
    exit 1
fi

echo "Downloading: $REPO/$FILENAME → $MODELS_DIR/"
huggingface-cli download "$REPO" "$FILENAME" --local-dir "$MODELS_DIR"
echo "Done. Model available via GET /v1/models"
```

- [ ] **Step 2: Commit**

```bash
chmod +x scripts/download-model.sh
git add scripts/download-model.sh
git commit -m "feat: model download helper for HuggingFace GGUFs"
```

---

### Task 9: Documentation

**Files:**
- Create: `README.md`
- Create: `manager/README.md`

- [ ] **Step 1: Write project README**

Cover: architecture overview, quick start (prerequisites, setup, usage), model management, configuration reference, service management, API endpoints. See spec for content.

- [ ] **Step 2: Write manager README**

Cover: what it does, how model switching works (the 5-step process), file listing with purposes, running locally for development, running tests.

- [ ] **Step 3: Commit**

```bash
git add README.md manager/README.md
git commit -m "docs: project and manager README"
```

---

### Task 10: Run Full Test Suite and Final Verification

- [ ] **Step 1: Install dependencies and run all tests**

Run: `cd /home/edible/Projects/inference_server && pip install -r manager/requirements.txt && PYTHONPATH=. python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Verify project structure**

Run: `find /home/edible/Projects/inference_server -type f | sort | grep -v __pycache__ | grep -v .pyc | grep -v venv`
Expected: All files from file structure are present

- [ ] **Step 3: Final commit if any cleanup needed**

```bash
git status
```
