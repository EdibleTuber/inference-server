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
    assert data["slots"]["main"]["queue_limit"] == 20
    assert data["slots"]["batch"]["queue_limit"] == 20


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


@pytest.mark.asyncio
async def test_ensure_model_on_slot_updates_loaded_model(test_config):
    from manager.app import ServerState
    server = ServerState(test_config)
    server.slots["main"].swapper.swap_to = AsyncMock(return_value=True)

    ok = await server.ensure_model_on_slot("main", "test-model-q4")
    assert ok is True
    assert server.slots["main"].loaded_model == "test-model-q4"
    assert server.slots["main"].healthy is True


@pytest.mark.asyncio
async def test_ensure_model_on_slot_error_on_failure(test_config):
    from manager.app import ServerState
    server = ServerState(test_config)
    server.slots["main"].swapper.swap_to = AsyncMock(return_value=False)

    ok = await server.ensure_model_on_slot("main", "test-model-q4")
    assert ok is False
    assert server.slots["main"].healthy is False


@pytest.mark.asyncio
async def test_ensure_model_on_slot_drains_queue_on_failure(test_config):
    from manager.app import ServerState
    server = ServerState(test_config)
    server.slots["main"].swapper.swap_to = AsyncMock(return_value=False)

    event1 = asyncio.Event()
    event2 = asyncio.Event()
    item1 = {"body": {}, "event": event1, "response": None, "error": None}
    item2 = {"body": {}, "event": event2, "response": None, "error": None}
    await server.slots["main"].queue.enqueue(item1)
    await server.slots["main"].queue.enqueue(item2)
    await server.ensure_model_on_slot("main", "test-model-q4")
    assert server.slots["main"].queue.depth == 0


@pytest.mark.asyncio
async def test_ensure_model_on_slot_skips_swap_if_already_loaded(test_config):
    from manager.app import ServerState
    server = ServerState(test_config)
    server.slots["main"].healthy = True
    server.slots["main"].loaded_model = "test-model-q4"
    server.slots["main"].swapper.swap_to = AsyncMock(return_value=True)

    result = await server.ensure_model_on_slot("main", "test-model-q4")
    assert result is True
    server.slots["main"].swapper.swap_to.assert_not_called()


def test_chat_completions_routes_batch_model(client):
    """When the batch slot has model X loaded, a request for X enqueues on batch, not main."""
    app = client.app
    server = app.state.server
    server.slots["main"].loaded_model = "test-model-q4"
    server.slots["main"].healthy = True
    server.slots["batch"].loaded_model = "test-model-q8"
    server.slots["batch"].healthy = True

    enqueued_on = []
    original_enqueue_main = server.slots["main"].queue.enqueue
    original_enqueue_batch = server.slots["batch"].queue.enqueue

    async def wrap_main(item):
        enqueued_on.append("main")
        from fastapi.responses import Response
        item["response"] = Response(content=b'{"id":"m"}', media_type="application/json")
        item["event"].set()

    async def wrap_batch(item):
        enqueued_on.append("batch")
        from fastapi.responses import Response
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
    """Model loaded on main routes to main (implicit main-swap path preserved)."""
    app = client.app
    server = app.state.server
    server.slots["main"].loaded_model = "test-model-q4"
    server.slots["main"].healthy = True
    server.slots["batch"].loaded_model = "test-model-q8"
    server.slots["batch"].healthy = True

    enqueued_on = []
    original_enqueue_main = server.slots["main"].queue.enqueue
    original_enqueue_batch = server.slots["batch"].queue.enqueue

    async def wrap_main(item):
        enqueued_on.append("main")
        from fastapi.responses import Response
        item["response"] = Response(content=b'{}', media_type="application/json")
        item["event"].set()

    async def wrap_batch(item):
        enqueued_on.append("batch")
        from fastapi.responses import Response
        item["response"] = Response(content=b'{}', media_type="application/json")
        item["event"].set()

    server.slots["main"].queue.enqueue = wrap_main
    server.slots["batch"].queue.enqueue = wrap_batch

    try:
        r = client.post("/v1/chat/completions", json={
            "model": "test-model-q4",
            "messages": [{"role": "user", "content": "hi"}],
        })
    finally:
        server.slots["main"].queue.enqueue = original_enqueue_main
        server.slots["batch"].queue.enqueue = original_enqueue_batch

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
