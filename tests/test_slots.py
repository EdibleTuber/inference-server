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
async def test_probe_strips_gguf_suffix():
    """llama-server returns the GGUF filename with suffix; probe strips .gguf
    so loaded_model matches what the manager's /v1/models and PAL both use."""
    slot = _make_slot()
    client = MagicMock()
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {
        "data": [{"id": "gemma-4-E4B-it-Q4_K_M.gguf", "object": "model"}],
    }
    client.get = AsyncMock(return_value=mock_response)

    await slot.probe(client)
    assert slot.loaded_model == "gemma-4-E4B-it-Q4_K_M"
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
