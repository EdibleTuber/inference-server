# tests/test_endpoints.py
"""Tests for the model manager API endpoints."""
import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def app(test_config):
    from manager.app import create_app
    return create_app(test_config)


@pytest.fixture
def client(app):
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_endpoint(client):
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


def test_models_endpoint(client):
    response = client.get("/v1/models")
    data = response.json()
    assert data["object"] == "list"
    model_ids = [m["id"] for m in data["data"]]
    assert "test-model-q4" in model_ids
    assert "test-model-q8" in model_ids


def test_models_openai_format(client):
    response = client.get("/v1/models")
    for model in response.json()["data"]:
        assert model["object"] == "model"
        assert "id" in model
        assert "created" in model
        assert "owned_by" in model


def test_chat_completions_missing_model(client):
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 400


def test_chat_completions_unknown_model(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "nonexistent-model", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 404


def test_ensure_model_updates_state(test_config):
    from manager.app import ServerState
    server = ServerState(test_config)
    server.swapper.swap_to = AsyncMock(return_value=True)

    async def run():
        result = await server.ensure_model("test-model-q4")
        assert result is True
        assert server.state == "ready"
        assert server.current_model == "test-model-q4"

    asyncio.get_event_loop().run_until_complete(run())


def test_ensure_model_error_state_on_failure(test_config):
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
    from manager.app import ServerState
    server = ServerState(test_config)
    server.swapper.swap_to = AsyncMock(return_value=False)

    async def run():
        event1 = asyncio.Event()
        event2 = asyncio.Event()
        item1 = {"body": {}, "event": event1, "response": None, "error": None}
        item2 = {"body": {}, "event": event2, "response": None, "error": None}
        await server.queue.enqueue(item1)
        await server.queue.enqueue(item2)
        await server.ensure_model("test-model-q4")
        assert server.queue.depth == 0

    asyncio.get_event_loop().run_until_complete(run())


def test_ensure_model_skips_swap_if_already_loaded(test_config):
    from manager.app import ServerState
    server = ServerState(test_config)
    server.state = "ready"
    server.current_model = "test-model-q4"
    server.swapper.swap_to = AsyncMock(return_value=True)

    async def run():
        result = await server.ensure_model("test-model-q4")
        assert result is True
        server.swapper.swap_to.assert_not_called()

    asyncio.get_event_loop().run_until_complete(run())
