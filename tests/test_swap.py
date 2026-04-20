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


@pytest.fixture
def swapper(test_config, main_slot):
    return ModelSwapper(test_config, slot=main_slot)


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
    async def fake_executor(executor, fn):
        return fn()

    mock_result = MagicMock()
    mock_result.returncode = 0

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
