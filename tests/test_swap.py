# tests/test_swap.py
"""Tests for model swap orchestration.

Tests mock subprocess (systemctl) and HTTP (health polling) since
we can't run a real llama-server in the test environment.
"""
import asyncio
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from manager.swap import ModelSwapper


@pytest.fixture
def egpu_slot(tmp_path, tmp_egpu_env_file, model_env_dir):
    from manager.slots import SlotState
    from manager.queue import RequestQueue
    return SlotState(
        name="egpu",
        host="127.0.0.1",
        port=18083,
        env_file=tmp_egpu_env_file,
        systemd_unit="llama-server-egpu.service",
        queue=RequestQueue(max_size=20),
        model_env_file=str(model_env_dir / "egpu.env"),
    )


@pytest.fixture
def main_slot(tmp_env_file, model_env_dir):
    from manager.slots import SlotState
    from manager.queue import RequestQueue
    return SlotState(
        name="main",
        host="127.0.0.1",
        port=18081,
        env_file=tmp_env_file,
        systemd_unit="llama-server.service",
        queue=RequestQueue(max_size=50),
        model_env_file=str(model_env_dir / "main.env"),
    )


@pytest.fixture
def swapper(test_config, main_slot):
    return ModelSwapper(test_config, slot=main_slot)


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
async def test_restart_timeout_is_a_runtime_error(swapper):
    async def fake_executor(executor, fn):
        return fn()
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["systemctl"], 150)), \
         patch.object(asyncio.get_event_loop(), "run_in_executor", side_effect=fake_executor):
        with pytest.raises(RuntimeError, match="timed out"):
            await swapper.restart_llama_server()

@pytest.mark.asyncio
async def test_restart_uses_configured_timeout(swapper):
    ok = MagicMock(returncode=0)
    async def fake_executor(executor, fn):
        return fn()
    with patch("subprocess.run", return_value=ok) as run, \
         patch.object(asyncio.get_event_loop(), "run_in_executor", side_effect=fake_executor):
        await swapper.restart_llama_server()
    assert run.call_args.kwargs["timeout"] == swapper._config.restart_timeout == 150

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
    """swap_to should chain the model env write, restart, and health poll."""
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
        result = await swapper.swap_to("test", "MODEL_ARGS=--model /opt/llama/models/test.gguf\n")

    assert result is True
    assert Path(swapper._slot.model_env_file).read_text() == "MODEL_ARGS=--model /opt/llama/models/test.gguf\n"


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
        result = await swapper.swap_to("test", "MODEL_ARGS=--model /opt/llama/models/test.gguf\n")

    assert result is False


@pytest.mark.parametrize("slot_fixture", ["main_slot", "egpu_slot"])
@pytest.mark.asyncio
async def test_each_swapper_writes_its_own_slots_model_env(test_config, slot_fixture, request):
    """Each slot's swapper writes that slot's model env file and leaves its slot env file alone."""
    from manager.swap import ModelSwapper
    slot = request.getfixturevalue(slot_fixture)
    slot_env_before = Path(slot.env_file).read_text()
    swapper = ModelSwapper(test_config, slot=slot)
    with patch.object(swapper, "restart_llama_server", AsyncMock()), \
         patch.object(swapper, "wait_for_health", AsyncMock(return_value=True)):
        assert await swapper.swap_to("m", f"MODEL_ARGS=--model /m/{slot.name}.gguf\n") is True
    assert Path(slot.model_env_file).read_text() == f"MODEL_ARGS=--model /m/{slot.name}.gguf\n"
    assert Path(slot.env_file).read_text() == slot_env_before


@pytest.mark.asyncio
async def test_swapper_restart_targets_correct_unit(test_config, egpu_slot):
    from manager.swap import ModelSwapper
    swapper = ModelSwapper(test_config, slot=egpu_slot)
    mock_result = MagicMock()
    mock_result.returncode = 0

    async def fake_executor(executor, fn):
        return fn()

    with patch("subprocess.run", return_value=mock_result) as mock_run, \
         patch.object(asyncio.get_event_loop(), "run_in_executor", side_effect=fake_executor):
        await swapper.restart_llama_server()

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd == ["sudo", "systemctl", "restart", "llama-server-egpu.service"]


@pytest.mark.asyncio
async def test_swapper_health_polls_slot_url(test_config, egpu_slot):
    """wait_for_health targets the slot's URL, not the hardcoded main URL."""
    from manager.swap import ModelSwapper
    swapper = ModelSwapper(test_config, slot=egpu_slot)
    mock_response = MagicMock(status_code=200)
    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.get = AsyncMock(return_value=mock_response)
        ok = await swapper.wait_for_health()
    assert ok is True
    call_url = instance.get.call_args[0][0]
    assert call_url == "http://127.0.0.1:18083/health"
