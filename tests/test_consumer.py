"""Queue consumer, readiness and routing tests for the round-1 slot-lockup fixes.

Regression tests for incident 2026-10-27 (docs/incidents/2026-10-27-main-slot-lockup.md)
and audit findings O1, F1, F2, F6 (docs/audit/findings.md).

Hermetic: the backend is an httpx.MockTransport, SlotState.probe is stubbed,
swap_to is an AsyncMock, and any real TCP connect fails the test.

Names introduced by the fix (ServerState.ready_for, _decision_from_item_error)
are looked up inside each test, so tests that don't use them still run against
pre-fix code and fail on their assertions rather than at import.
"""
import asyncio
import contextlib
import socket
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from manager import app as app_mod
from manager.app import ServerState, _queue_consumer, _validate_and_route
from manager.slots import SlotState


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly on any real TCP connect (the live llama-server is on :8081)."""
    real_connect = socket.socket.connect

    def guarded(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError(f"test opened a real socket to {address}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


class ProbeStub:
    """Replaces SlotState.probe. result=None leaves health unchanged."""

    def __init__(self):
        self.calls: list[str] = []
        self.result: bool | None = None
        self.exc: BaseException | None = None


@pytest.fixture(autouse=True)
def probes(monkeypatch):
    stub = ProbeStub()

    async def fake_probe(self, client):
        stub.calls.append(self.name)
        if stub.exc is not None:
            raise stub.exc
        if stub.result is not None:
            self.healthy = stub.result

    monkeypatch.setattr(SlotState, "probe", fake_probe)
    return stub


class Backend:
    """MockTransport handler. `handler(request, n)` where n counts calls from 1."""

    def __init__(self):
        self.calls: list[httpx.Request] = []
        self.handler = lambda request, n: httpx.Response(200, json={"ok": True})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return self.handler(request, len(self.calls))


@pytest.fixture(autouse=True)
def backend(monkeypatch):
    b = Backend()
    real = httpx.AsyncClient

    class MockedAsyncClient(real):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(b)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(app_mod.httpx, "AsyncClient", MockedAsyncClient)
    return b


@pytest.fixture
def server(test_config):
    s = ServerState(test_config)
    for slot in s.slots.values():
        slot.swapper.swap_to = AsyncMock(return_value=True)
    return s


def swapped(slot) -> list[str]:
    """Model stems passed to swap_to, in order."""
    return [Path(c.args[0]).stem for c in slot.swapper.swap_to.call_args_list]


def make_item(model: str, stream: bool = False) -> dict:
    return {
        "body": {"model": model, "stream": stream,
                 "messages": [{"role": "user", "content": "hi"}]},
        "event": asyncio.Event(), "response": None, "error": None,
        "error_code": None, "backend_path": "v1/chat/completions",
    }


async def submit(server, *items, slot: str = "main") -> None:
    s = server.slots[slot]
    for item in items:
        await s.queue.enqueue(item)
    s.queue_event.set()


@contextlib.asynccontextmanager
async def consumer(server, slot: str = "main"):
    task = asyncio.create_task(_queue_consumer(server, server._config, slot_name=slot))
    try:
        yield task
    finally:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task


async def answered(task, *items, timeout: float = 2.0) -> None:
    """Wait until every item's event is set; on timeout, report the consumer's state."""
    try:
        await asyncio.wait_for(
            asyncio.gather(*(item["event"].wait() for item in items)), timeout)
    except asyncio.TimeoutError:
        state = "alive"
        if task.done():
            state = "cancelled" if task.cancelled() else f"died with {task.exception()!r}"
        unanswered = [i for i, item in enumerate(items, 1) if not item["event"].is_set()]
        raise AssertionError(
            f"items {unanswered} not answered within {timeout}s; consumer {state}"
        ) from None


def load(server, model: str, healthy: bool, slot: str = "main"):
    s = server.slots[slot]
    s.loaded_model, s.healthy = model, healthy
    return s


# ---------------------------------------------------------------------------
# Task 3: chat requests never swap or restart a slot (O1, F1)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_queued_request_for_a_swapped_out_model_gets_409_not_a_swap(server):
    slot = load(server, "test-model-q4", healthy=True)
    item = make_item("test-model-q4")
    await slot.queue.enqueue(item)                         # routed while q4 was loaded
    assert await server.ensure_model_on_slot("main", "test-model-q8") is True

    async with consumer(server) as task:
        slot.queue_event.set()
        await answered(task, item)

    assert swapped(slot) == ["test-model-q8"]
    assert item["error_code"] == "model_not_loaded"
    assert "test-model-q8" in item["error"]


@pytest.mark.asyncio
async def test_unhealthy_slot_with_its_own_model_is_probed_not_restarted(server, probes):
    slot = load(server, "test-model-q4", healthy=False)
    probes.result = True
    item = make_item("test-model-q4")

    async with consumer(server) as task:
        await submit(server, item)
        await answered(task, item)

    assert swapped(slot) == []
    assert item["response"] is not None and item["response"].status_code == 200
    assert slot.healthy is True


@pytest.mark.asyncio
async def test_unhealthy_slot_whose_probe_fails_answers_503_without_restart(
        server, probes, backend):
    slot = load(server, "test-model-q4", healthy=False)
    probes.result = False
    item = make_item("test-model-q4")

    async with consumer(server) as task:
        await submit(server, item)
        await answered(task, item)

    assert swapped(slot) == []
    assert item["error_code"] == "slot_unavailable"
    assert item["response"] is None
    assert backend.calls == []


@pytest.mark.asyncio
async def test_case_variant_model_name_is_ready(server):
    slot = load(server, "test-model-q4", healthy=True)
    item = make_item("TEST-MODEL-Q4.gguf")

    async with consumer(server) as task:
        await submit(server, item)
        await answered(task, item)

    assert swapped(slot) == []
    assert item["error_code"] is None
    assert item["response"].status_code == 200


@pytest.mark.asyncio
async def test_streaming_request_rejected_by_readiness_gets_json_409(server):
    slot = load(server, "test-model-q4", healthy=True)
    item = make_item("test-model-q4", stream=True)
    await slot.queue.enqueue(item)
    assert await server.ensure_model_on_slot("main", "test-model-q8") is True

    async with consumer(server) as task:
        slot.queue_event.set()
        await answered(task, item)

    assert item["response"] is None          # not a 200 StreamingResponse
    assert item["error_code"] == "model_not_loaded"


@pytest.mark.asyncio
async def test_probe_raising_inside_ready_for_releases_lock(server, probes):
    slot = load(server, "test-model-q4", healthy=False)
    probes.exc = RuntimeError("probe blew up")

    result = await server.ready_for("main", "test-model-q4")

    assert result == "unavailable"
    assert not slot.swap_lock.locked()
    assert swapped(slot) == []


@pytest.mark.parametrize("code, status, decision_code", [
    (None, 503, "queue_error"),
    ("queue_error", 503, "queue_error"),
    ("slot_unavailable", 503, "slot_unavailable"),
    ("model_not_loaded", 409, "model_not_loaded"),
])
def test_item_error_codes_map_to_statuses(code, status, decision_code):
    decide = app_mod._decision_from_item_error
    d = decide({"error": "the message", "error_code": code}, "main")
    assert (d.status, d.code, d.message) == (status, decision_code, "the message")
    assert d.slot_name == "main"


@pytest.mark.asyncio
async def test_consumer_recovers_from_a_proxy_timeout_without_restarting(
        server, probes, backend):
    """Incident 2026-10-27 at consumer level: a ReadTimeout on one request must
    not restart the backend for the next request on the same model."""
    slot = load(server, "test-model-q4", healthy=True)
    probes.result = True

    def handler(request, n):
        if n == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json={"ok": True})
    backend.handler = handler

    first, second = make_item("test-model-q4"), make_item("test-model-q4")
    async with consumer(server) as task:
        await submit(server, first, second)
        await answered(task, first, second)
        assert not task.done()

    assert "Proxy error" in first["error"]
    assert second["response"] is not None and second["response"].status_code == 200
    assert swapped(slot) == []


@pytest.mark.asyncio
async def test_backend_5xx_is_passed_through_and_does_not_restart(
        server, probes, backend):
    """A backend 5xx marks the slot unhealthy; the next request on the same
    model must be served after a probe, not trigger a restart (F1 via 5xx)."""
    slot = load(server, "test-model-q4", healthy=True)
    probes.result = True

    def handler(request, n):
        if n == 1:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, json={"ok": True})
    backend.handler = handler

    first, second = make_item("test-model-q4"), make_item("test-model-q4")
    async with consumer(server) as task:
        await submit(server, first, second)
        await answered(task, first, second)
        assert not task.done()

    assert first["error"] is None and first["response"].status_code == 503
    assert second["response"] is not None and second["response"].status_code == 200
    assert swapped(slot) == []


# ---------------------------------------------------------------------------
# Task 4: the consumer cannot die (F2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consumer_survives_a_raising_readiness_check(server):
    load(server, "test-model-q4", healthy=True)
    server.ready_for = AsyncMock(side_effect=[RuntimeError("boom"), "ready"])
    first, second = make_item("test-model-q4"), make_item("test-model-q4")

    async with consumer(server) as task:
        await submit(server, first, second)
        await answered(task, first, second)
        assert not task.done()

    assert first["error_code"] == "queue_error"
    assert "boom" in first["error"]
    assert second["response"] is not None and second["response"].status_code == 200


@pytest.mark.asyncio
async def test_cancelling_the_consumer_mid_item_cancels_it(server):
    load(server, "test-model-q4", healthy=True)
    never = asyncio.Event()

    async def hang(slot_name, model_name):
        await never.wait()
    server.ready_for = hang

    task = asyncio.create_task(_queue_consumer(server, server._config, slot_name="main"))
    await submit(server, make_item("test-model-q4"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


# ---------------------------------------------------------------------------
# Task 5: reprobe before answering 503 (F6)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unhealthy_slot_is_reprobed_and_routed(server, probes):
    load(server, "test-model-q4", healthy=False)
    probes.result = True

    decision, slot_name = await _validate_and_route(server, {"model": "test-model-q4"})

    assert decision is None
    assert slot_name == "main"


@pytest.mark.asyncio
async def test_unhealthy_slot_still_503s_when_the_probe_fails(server, probes):
    load(server, "test-model-q4", healthy=False)
    probes.result = False

    decision, _ = await _validate_and_route(server, {"model": "test-model-q4"})

    assert probes.calls == ["main"]
    assert (decision.status, decision.code) == (503, "slot_unavailable")


@pytest.mark.asyncio
async def test_unhealthy_slot_mid_swap_is_not_probed(server, probes):
    slot = load(server, "test-model-q4", healthy=False)
    probes.result = True

    async with slot.swap_lock:
        decision, _ = await _validate_and_route(server, {"model": "test-model-q4"})

    assert probes.calls == []
    assert (decision.status, decision.code) == (503, "slot_unavailable")
