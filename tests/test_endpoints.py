# tests/test_endpoints.py
"""Tests for the model manager API endpoints."""
import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

_Path = Path  # alias used by model_path tests (brief requirement)


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


def test_chat_completions_routes_main_when_loaded_on_main(client):
    """Model loaded on main routes to main."""
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


def test_chat_completions_409_when_not_loaded(client):
    """A model that exists on disk but is loaded on neither slot -> 409 (no implicit swap)."""
    app = client.app
    server = app.state.server
    server.slots["main"].loaded_model = "test-model-q4"
    server.slots["main"].healthy = True
    server.slots["batch"].loaded_model = None
    server.slots["batch"].healthy = False

    r = client.post("/v1/chat/completions", json={
        "model": "test-model-q8",  # on disk, not loaded anywhere
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 409
    assert r.json()["error"]["type"] == "model_not_loaded"


def test_chat_completions_cold_start_no_crash(client):
    """Both slots unloaded (loaded_model=None, unhealthy): request must not 500."""
    app = client.app
    server = app.state.server
    server.slots["main"].loaded_model = None
    server.slots["main"].healthy = False
    server.slots["batch"].loaded_model = None
    server.slots["batch"].healthy = False

    r = client.post("/v1/chat/completions", json={
        "model": "test-model-q4",  # exists on disk, not loaded
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 409  # not a 500


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


@pytest.mark.asyncio
async def test_reconcile_on_backend_5xx(test_config):
    """When the queue consumer gets a 5xx from the backend, it re-probes
    the slot and updates loaded_model if it has drifted."""
    from manager.app import ServerState
    import httpx
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


@pytest.mark.asyncio
async def test_mark_swapped_stores_ondisk_stem_not_request_casing(test_config):
    """A swap requested with odd casing/suffix is stored as the real on-disk stem."""
    from manager.app import ServerState
    server = ServerState(test_config)
    server.slots["main"].swapper.swap_to = AsyncMock(return_value=True)

    ok = await server.ensure_model_on_slot("main", "TEST-MODEL-Q4.gguf")
    assert ok is True
    # tmp_models_dir has 'test-model-q4.gguf' (lowercase) -> canonical stem stored.
    assert server.slots["main"].loaded_model == "test-model-q4"


@pytest.mark.asyncio
async def test_ensure_model_skips_swap_on_case_variant(test_config):
    """Already-loaded model requested with different case must NOT re-swap."""
    from manager.app import ServerState
    server = ServerState(test_config)
    server.slots["main"].healthy = True
    server.slots["main"].loaded_model = "test-model-q4"
    server.slots["main"].swapper.swap_to = AsyncMock(return_value=True)

    result = await server.ensure_model_on_slot("main", "TEST-MODEL-Q4")
    assert result is True
    server.slots["main"].swapper.swap_to.assert_not_called()


# ---------------------------------------------------------------------------
# model_path resolution (Task 3)
# ---------------------------------------------------------------------------

def test_model_path_exact_and_suffix(test_config):
    from manager.app import ServerState
    server = ServerState(test_config)
    assert server.model_path("test-model-q4").endswith("test-model-q4.gguf")
    assert server.model_path("test-model-q4.gguf").endswith("test-model-q4.gguf")


def test_model_path_case_insensitive(test_config):
    from manager.app import ServerState
    server = ServerState(test_config)
    assert server.model_path("TEST-MODEL-Q4").endswith("test-model-q4.gguf")


def test_model_path_unknown_is_none(test_config):
    from manager.app import ServerState
    server = ServerState(test_config)
    assert server.model_path("nope") is None
    assert server.model_path("") is None


def test_model_path_collision_is_deterministic_and_warns(test_config, caplog):
    from manager.app import ServerState
    d = test_config.models_dir
    _Path(d, "Dup-Model.gguf").touch()
    _Path(d, "dup-model.gguf").touch()
    server = ServerState(test_config)
    # Exact-case match wins, no warning needed.
    assert server.model_path("Dup-Model").endswith("Dup-Model.gguf")
    assert server.model_path("dup-model").endswith("dup-model.gguf")
    # A folded-only variant resolves to the sorted-first file ('D' < 'd') and warns.
    with caplog.at_level("WARNING"):
        resolved = server.model_path("DUP-MODEL")
    assert resolved.endswith("Dup-Model.gguf")
    assert any("collision" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# reprobe swap_lock gate (Task 5)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reprobe_waits_for_swap_lock(test_config):
    """A post-5xx reprobe must not run while a swap holds the slot lock."""
    from manager.app import ServerState, _reprobe_for
    server = ServerState(test_config)
    slot = server.slots["main"]
    slot.reconcile_on_error = AsyncMock()

    await slot.swap_lock.acquire()          # simulate an in-flight swap
    task = asyncio.create_task(_reprobe_for(slot))
    await asyncio.sleep(0.01)
    slot.reconcile_on_error.assert_not_called()   # blocked on the lock
    slot.swap_lock.release()
    await task
    slot.reconcile_on_error.assert_awaited_once()  # ran after release
