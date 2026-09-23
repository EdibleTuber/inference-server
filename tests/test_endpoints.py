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


@pytest.mark.asyncio
async def test_status_endpoint_does_not_block_event_loop(app):
    """A slow nvidia-smi call must not stall other in-flight requests.

    Regression test: GET /status used to call subprocess.run synchronously
    inside the async handler, freezing the single-threaded event loop (and
    therefore every queued chat request) for as long as nvidia-smi took.
    """
    import time
    import httpx

    def slow_nvidia_smi(*args, **kwargs):
        time.sleep(0.3)
        result = MagicMock()
        result.stdout = (
            "gpu_name, memory.total [MiB], memory.used [MiB]\n"
            "Tesla P40, 24576 MiB, 18200 MiB"
        )
        result.returncode = 0
        return result

    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(10):
            await asyncio.sleep(0.03)
            ticks += 1

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("manager.gpu.subprocess.run", side_effect=slow_nvidia_smi):
            start = time.monotonic()
            await asyncio.gather(client.get("/status"), ticker())
            elapsed = time.monotonic() - start

    assert ticks == 10
    assert elapsed < 0.45  # ~0.6s if /status still blocks the event loop


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


def test_chat_completions_409_when_not_loaded(client, monkeypatch):
    """A model that exists on disk but is loaded on neither slot -> 409 (no implicit swap).

    The probe is stubbed because this test asserts on the 409 MESSAGE, and the
    message is built from slot state this test sets by hand. The chat handler
    now re-probes before declaring a model unloaded (so a stale slot is not
    reported as empty), and test_config points the slots at 127.0.0.1:8081/8083
    -- which on a machine where the real backends are running is not a fixture,
    it is production. Without this stub the probe replaced the fabricated state
    with the live models and the assertion read
    `main='gemma-4-26b-a4b-it-q4_k_m'`.

    Worth knowing more broadly: TestClient runs the lifespan, so the startup
    probe has always reached those ports. This stub makes one test hermetic; the
    suite as a whole still depends on what happens to be listening.
    """
    from manager.slots import SlotState

    async def _no_probe(self, probe_client):
        return None

    monkeypatch.setattr(SlotState, "probe", _no_probe)

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
    # The message must name WHAT is loaded (self-evident mismatch), not just
    # "not loaded" — this is the finicky-error fix.
    msg = r.json()["error"]["message"]
    assert "test-model-q8" in msg          # the requested (missing) model
    assert "test-model-q4" in msg          # what main actually has loaded


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


def test_swap_non_string_target_returns_structured_400_not_500(client):
    """A non-string target (e.g. a list or dict) must fail like any other invalid
    target -- a structured 400 -- not crash the handler.

    `target not in server.slots` is a dict membership test, which hashes its
    operand; an unhashable target (list, dict) raises TypeError from inside the
    handler, and with no custom exception handler registered Starlette turns
    that into a generic 500 -- unlike the old tuple-membership check, which
    compared by equality and never hashed.
    """
    from fastapi.testclient import TestClient
    unraising_client = TestClient(client.app, raise_server_exceptions=False)

    for bad_target in (["main"], {"main": 1}):
        r = unraising_client.post(
            "/swap", json={"model": "test-model-q4", "target": bad_target})
        assert r.status_code == 400, r.text
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


def test_swap_accepts_a_configured_third_slot(client_with_three_slots):
    """A slot beyond the legacy main/batch pair is a valid /swap target when configured."""
    resp = client_with_three_slots.post(
        "/swap", json={"model": "test-model-q4", "target": "re"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["slot"] == "re"


def test_swap_rejects_an_unconfigured_target(client_with_three_slots):
    """The 400 body must name every configured slot, derived from config -- not a
    hardcoded literal list -- so it stays correct as the configured slots change."""
    app = client_with_three_slots.app
    configured_names = list(app.state.server.slots.keys())
    assert len(configured_names) == 3  # sanity: fixture actually wired up 3 slots

    resp = client_with_three_slots.post(
        "/swap", json={"model": "test-model-q4", "target": "nope"})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_target"
    body = resp.json()["error"]["message"]
    for name in configured_names:
        assert name in body, f"400 message should name configured slot {name!r}: {body}"


def test_swap_echo_is_canonical(client, monkeypatch):
    """A swap requested with odd casing/suffix echoes the canonical on-disk stem."""
    app = client.app
    server = app.state.server

    async def fake_swap(self, model):
        return True
    monkeypatch.setattr("manager.swap.ModelSwapper.swap_to", fake_swap)

    r = client.post("/swap", json={"model": "TEST-MODEL-Q4.gguf", "target": "main"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "test-model-q4"                       # canonical
    assert body["model"] == server.slots["main"].loaded_model     # agrees with /status


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


# --- a stale slot view must not be reported as "not loaded" ----------------

def test_a_stale_slot_is_reprobed_before_returning_409(client, monkeypatch):
    """Probing happens at startup and after a 5xx. A backend that became ready
    LATER was invisible forever.

    Observed on the live server: llama-manager and llama-server-batch both
    started at 23:36:45 -- the same second -- so the startup probe ran before
    the iGPU had finished loading a 4B model. :8083 was serving
    gemma-4-E4B-it-Q4_K_M with 4 slots while /status reported
    loaded_model: null, and every request for that model got a 409 telling the
    caller to load a model that was already loaded.
    """
    from manager.slots import SlotState

    server = client.app.state.server
    server.slots["main"].loaded_model = "test-model-q4"
    server.slots["main"].healthy = True
    server.slots["batch"].loaded_model = None       # stale; it IS loaded
    server.slots["batch"].healthy = False

    async def fake_probe(self, probe_client):
        if self.name == "batch":
            self.loaded_model = "test-model-q8"
            self.healthy = True

    monkeypatch.setattr(SlotState, "probe", fake_probe)

    enqueued_on = []

    async def wrap(item, _name=None):
        enqueued_on.append(_name)
        from fastapi.responses import Response
        item["response"] = Response(content=b'{}', media_type="application/json")
        item["event"].set()

    server.slots["main"].queue.enqueue = lambda i: wrap(i, "main")
    server.slots["batch"].queue.enqueue = lambda i: wrap(i, "batch")

    r = client.post("/v1/chat/completions", json={
        "model": "test-model-q8",
        "messages": [{"role": "user", "content": "hi"}],
    })

    assert r.status_code != 409, r.json()
    assert enqueued_on == ["batch"], enqueued_on


def test_a_genuinely_unloaded_model_still_409s_after_the_reprobe(client, monkeypatch):
    """The re-probe must not turn a real 409 into something else. If the probe
    finds nothing, the answer is unchanged."""
    from manager.slots import SlotState

    server = client.app.state.server
    server.slots["main"].loaded_model = "test-model-q4"
    server.slots["main"].healthy = True
    server.slots["batch"].loaded_model = None
    server.slots["batch"].healthy = False

    probed = []

    async def fake_probe(self, probe_client):
        probed.append(self.name)          # finds nothing new

    monkeypatch.setattr(SlotState, "probe", fake_probe)

    r = client.post("/v1/chat/completions", json={
        "model": "test-model-q8",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 409
    assert r.json()["error"]["type"] == "model_not_loaded"
    assert set(probed) == {"main", "batch"}, probed


def test_the_reprobe_skips_a_slot_mid_swap(client, monkeypatch):
    """A swap in flight is authoritative and already producing fresh state.
    Waiting on its lock would stall this request behind a model load, and
    probing past it could clobber a fresh mark_swapped."""
    from manager.slots import SlotState

    server = client.app.state.server
    server.slots["main"].loaded_model = "test-model-q4"
    server.slots["main"].healthy = True
    server.slots["batch"].loaded_model = None
    server.slots["batch"].healthy = False

    probed = []

    async def fake_probe(self, probe_client):
        probed.append(self.name)

    monkeypatch.setattr(SlotState, "probe", fake_probe)

    async def hold_and_request():
        await server.slots["batch"].swap_lock.acquire()
        try:
            return await asyncio.to_thread(
                client.post, "/v1/chat/completions",
                json={"model": "test-model-q8",
                      "messages": [{"role": "user", "content": "hi"}]})
        finally:
            server.slots["batch"].swap_lock.release()

    r = asyncio.run(hold_and_request())
    assert r.status_code == 409
    assert "batch" not in probed, probed
    assert "main" in probed, probed


def test_server_builds_every_configured_slot(test_config):
    """Slot construction follows configuration, including a third slot."""
    from dataclasses import replace
    from manager.slot_config import SlotConfig
    from manager.app import ServerState

    cfg = replace(test_config, slots=(
        SlotConfig("main", "127.0.0.1", 8081, "/tmp/main.env", "main.service", 20),
        SlotConfig("batch", "127.0.0.1", 8083, "/tmp/batch.env", "batch.service", 20),
        SlotConfig("re", "127.0.0.1", 8084, "/tmp/re.env", "re.service", 30),
    ))
    server = ServerState(cfg)

    assert list(server.slots) == ["main", "batch", "re"]
    assert server.slots["re"].port == 8084
    assert server.slots["re"].queue.max_size == 30
    assert server.slots["re"].env_file == "/tmp/re.env"
    assert server.slots["re"].systemd_unit == "re.service"
    assert all(s.swapper is not None for s in server.slots.values())


def test_slot_count_follows_configuration(test_config):
    """A relationship, not a literal: adding a slot must not need a test edit.

    Deliberately uses a 3-slot config (not test_config's own 2) so this
    can't pass by coincidence against code that still hardcodes main/batch.
    """
    from dataclasses import replace
    from manager.slot_config import SlotConfig
    from manager.app import ServerState

    cfg = replace(test_config, slots=(
        SlotConfig("main", "127.0.0.1", 8081, "/tmp/main.env", "main.service", 20),
        SlotConfig("batch", "127.0.0.1", 8083, "/tmp/batch.env", "batch.service", 20),
        SlotConfig("re", "127.0.0.1", 8084, "/tmp/re.env", "re.service", 30),
    ))
    server = ServerState(cfg)
    assert len(server.slots) == len(cfg.slots)


def test_status_reprobes_before_reporting(test_config):
    """F8: /status must not serve stale startup state.

    Fails against the pre-fix code, which reports whatever the startup probe
    left behind until some chat request happens to trigger a reprobe.

    Patches reprobe_all_slots_concurrently, not reprobe_all_slots: /status
    uses the concurrent variant so its worst case is one probe timeout
    rather than one per slot (see ServerState.reprobe_all_slots_concurrently
    and the /status handler for why). reprobe_all_slots itself stays
    sequential, unchanged, for the chat 409 path.
    """
    from unittest.mock import AsyncMock, patch
    from fastapi.testclient import TestClient
    from manager.app import create_app

    with patch("manager.app.ServerState.reprobe_all_slots_concurrently",
               new_callable=AsyncMock) as reprobe:
        app = create_app(test_config)
        with TestClient(app) as client:
            reprobe.reset_mock()          # ignore any startup-time calls
            resp = client.get("/status")
    assert resp.status_code == 200
    reprobe.assert_awaited_once()


def test_status_still_reports_every_slot(test_config):
    from fastapi.testclient import TestClient
    from manager.app import create_app
    app = create_app(test_config)
    with TestClient(app) as client:
        body = client.get("/status").json()
    assert len(body["slots"]) == len(test_config.slots)


class _FakeV1ModelsServer:
    """A real HTTP server for one /v1/models endpoint, fully controlled by
    the test that owns it.

    Narrow and local to test_status_survives_a_raising_slot_probe -- NOT
    the general fake-backend fixture for the whole endpoint suite (that is
    deferred). Needed here specifically because the fixture's slot ports
    otherwise collide with whatever is actually running on this machine
    (see the module's other /status tests, which rely on that real
    backend and are unaffected by this one using a fake instead): a test
    asserting that a *healthy* sibling survives a concurrent reprobe needs
    a backend whose health and response timing it actually controls,
    independent of what happens to be listening on 127.0.0.1:8083 right
    now.
    """

    def __init__(self):
        import http.server
        import threading

        self.body: dict = {"data": []}
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                import json as _json
                if self.path == "/v1/models":
                    payload = _json.dumps(outer.body).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # keep test output quiet

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.host, self.port = self._httpd.server_address
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def shutdown(self):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)


@pytest.fixture
def fake_v1_models_backend():
    server = _FakeV1ModelsServer()
    yield server
    server.shutdown()


def test_status_survives_a_raising_slot_probe(test_config, fake_v1_models_backend):
    """A slot whose probe raises must not 500 /status or corrupt a sibling.

    Fix-round-1/round-2/round-3 regression test. Without
    return_exceptions=True, asyncio.gather propagates the first child
    exception immediately, without awaiting or cancelling siblings -- so
    reprobe_all_slots_concurrently's `async with httpx.AsyncClient()`
    block exits and closes the shared client while another slot's probe
    may still be running. On the real, unforced interleaving (confirmed by
    the round-3 reviewer: 9/9 runs, batch.probe unwrapped, no client-side
    sleep, backend delays of 0/50/300ms), the sibling is NEVER corrupted --
    asyncio.gather's ensure_future queues both tasks' first step before
    the awaiting coroutine suspends, so the sibling always reaches
    client.get(...) and suspends inside connection setup before aclose()
    runs. The 500 IS the real, every-run production symptom (main's
    exception propagates uncaught, past nothing that would catch it, out
    of the /status handler); the sibling-corruption limb is NOT a
    naturally-occurring production race.

    This test therefore does two different things, and is honest about
    the difference:
      - it demonstrates the 500 the way production actually produces it
        (no forcing needed -- any raising slot 500s /status pre-fix);
      - it additionally pins a PROPERTY -- "even if a sibling's request
        happens to still be pending when the client closes, it must not
        be corrupted" -- against a class of wrong fixes (e.g. one that
        adds return_exceptions=True but also cancels siblings, or one
        that closes the client eagerly). To exercise that property at all
        requires deliberately forcing an interleaving production does not
        naturally produce, since gather's task-scheduling order means the
        sibling normally finishes sending before the raise propagates.
        batch's probe is wrapped to `await asyncio.sleep(...)` BEFORE
        calling the real, unmodified SlotState.probe against a real fake
        backend, guaranteeing the shared client is already closed by the
        time batch's real client.get() call happens, so batch
        deterministically hits the client's real
        "if self._state == ClientState.CLOSED: raise RuntimeError(...)"
        check at send() ENTRY (httpx/_client.py) -- this is a request
        that never got dispatched, not one that was in flight and got torn
        down mid-request.

    The 'batch' slot is pointed at fake_v1_models_backend (a real HTTP
    server this test controls) rather than test_config's real port, so
    batch's baseline health/model is deterministic instead of depending on
    "whatever happens to be running on 127.0.0.1:8083 right now" (see the
    other /status tests in this module, which do rely on that real
    backend and are unaffected by this one using a fake instead).

    The final assertion checks loaded_model against a SECOND, distinct
    fake-backend body set just before the /status call (not the same
    "batch-model" the startup probe already saw) -- this closes a
    coverage gap the round-2 version left open: with the STALE body, a
    wrong fix that cancelled the sibling instead of awaiting it (silently
    leaving batch's prior state untouched) would still pass every
    assertion, since "never probed" and "probed successfully" look
    identical when the answer doesn't change. Asserting on the NEW value
    proves the probe actually ran to completion, not merely that nothing
    clobbered its old result.

    slot.probe() itself is supposed to never raise (see its docstring and
    the malformed-payload tests in test_slots.py); this poisons main's
    probe directly to simulate some other bug reaching past that
    guarantee, so the test isolates reprobe_all_slots_concurrently's own
    fan-out behavior rather than re-testing slot.probe.
    """
    import dataclasses
    from manager.slots import SlotState
    from fastapi.testclient import TestClient
    from manager.app import create_app

    fake_v1_models_backend.body = {"data": [{"id": "batch-model.gguf"}]}

    batch_sc = dataclasses.replace(
        next(sc for sc in test_config.slots if sc.name == "batch"),
        host=fake_v1_models_backend.host,
        port=fake_v1_models_backend.port,
    )
    config = dataclasses.replace(
        test_config,
        slots=tuple(batch_sc if sc.name == "batch" else sc for sc in test_config.slots),
    )

    app = create_app(config)

    # raise_server_exceptions=False: an unhandled exception in the /status
    # handler must surface as a real 500 response here, not as a Python
    # exception raised into this test -- so that a broken fix's two
    # symptoms (the 500, and the corrupted sibling) can be asserted on
    # independently instead of one masking the other.
    with TestClient(app, raise_server_exceptions=False) as client:
        server = client.app.state.server
        main = server.slots["main"]
        batch = server.slots["batch"]

        # batch starts healthy from the real startup probe against the
        # fake backend above -- deterministic, not dependent on this
        # machine's state.
        assert batch.healthy is True
        assert batch.loaded_model == "batch-model"

        # Change what the fake backend reports BEFORE triggering the
        # reprobe below, to a value the startup probe never saw. The final
        # assertion checks for THIS value, not "batch-model" again -- so a
        # wrong fix that skips/cancels the sibling's probe instead of
        # awaiting it (leaving batch's old state untouched) fails here,
        # instead of accidentally passing because "never probed" and
        # "probed successfully" look identical when the value doesn't
        # change. See docstring's "coverage gap" paragraph.
        fake_v1_models_backend.body = {"data": [{"id": "batch-model-v2.gguf"}]}

        async def raise_immediately(probe_client):
            raise AttributeError("boom: simulated bug past slot.probe's own guard")

        real_probe = SlotState.probe  # unbound, unmodified

        async def delayed_real_probe(probe_client):
            # Long enough that main's synchronous raise has already
            # propagated through gather and closed the shared client
            # before this even attempts client.get() -- see docstring.
            await asyncio.sleep(0.05)
            await real_probe(batch, probe_client)

        main.probe = AsyncMock(side_effect=raise_immediately)
        batch.probe = AsyncMock(side_effect=delayed_real_probe)

        resp = client.get("/status")

        # main's raise propagates near-instantly, so the /status response
        # above returns well before batch's 0.05s delayed_real_probe has
        # even attempted its client.get() call. That orphaned coroutine
        # (gather without return_exceptions=True does not cancel siblings)
        # keeps running on TestClient's background portal thread after
        # this thread resumes -- a REAL wall-clock sleep here (not
        # asyncio.sleep; this thread isn't in that event loop) gives it
        # time to finish and actually mutate batch's state before this
        # `with` block exits and tears the portal down.
        import time as _time
        _time.sleep(0.2)

    # The limb that matters most, asserted first so it can't be masked by
    # the status-code assertion below: a genuinely healthy sibling must not
    # be fabricated as dead just because another slot's probe raised.
    assert batch.healthy is True
    # The value changed above, not the one the startup probe already saw --
    # this proves the reprobe actually ran to completion rather than a
    # wrong fix silently leaving batch's prior state untouched.
    assert batch.loaded_model == "batch-model-v2"
    assert resp.status_code == 200
