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
import json
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


# ---------------------------------------------------------------------------
# Round 2, Task 2: proxy timeouts (audit F4)
# ---------------------------------------------------------------------------

def test_proxy_read_timeout_config(monkeypatch):
    from manager.config import ManagerConfig
    monkeypatch.delenv("PROXY_READ_TIMEOUT", raising=False)
    assert ManagerConfig.from_env().proxy_read_timeout == 1800
    monkeypatch.setenv("PROXY_READ_TIMEOUT", "60")
    assert ManagerConfig.from_env().proxy_read_timeout == 60


@pytest.mark.asyncio
async def test_nonstream_request_uses_the_configured_read_timeout(server, backend):
    load(server, "test-model-q4", healthy=True)
    seen = []

    def handler(request, n):
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={"ok": True})
    backend.handler = handler

    item = make_item("test-model-q4")
    async with consumer(server) as task:
        await submit(server, item)
        await answered(task, item)

    assert seen[0]["read"] == 1800
    assert seen[0]["connect"] is not None and seen[0]["connect"] <= 30


@pytest.mark.asyncio
async def test_read_timeout_is_a_504_and_leaves_the_slot_healthy(server, probes, backend):
    slot = load(server, "test-model-q4", healthy=True)

    def handler(request, n):
        raise httpx.ReadTimeout("timed out", request=request)
    backend.handler = handler

    item = make_item("test-model-q4")
    async with consumer(server) as task:
        await submit(server, item)
        await answered(task, item)
        await asyncio.sleep(0.05)            # let any fire-and-forget reprobe run

    assert item["error_code"] == "upstream_timeout"
    assert "1800s" in item["error"]
    assert slot.healthy is True
    assert probes.calls == []


@pytest.mark.asyncio
async def test_connect_error_still_marks_the_slot_unhealthy(server, backend):
    slot = load(server, "test-model-q4", healthy=True)

    def handler(request, n):
        raise httpx.ConnectError("refused", request=request)
    backend.handler = handler

    item = make_item("test-model-q4")
    async with consumer(server) as task:
        await submit(server, item)
        await answered(task, item)

    assert "Proxy error" in item["error"]
    assert item["error_code"] in (None, "queue_error")
    assert slot.healthy is False


@pytest.mark.asyncio
async def test_connect_timeout_is_not_treated_as_a_busy_backend(server, backend):
    """Only read/write timeouts mean "busy". A backend that won't accept a
    connection is unhealthy, exactly like a refused connection."""
    slot = load(server, "test-model-q4", healthy=True)

    def handler(request, n):
        raise httpx.ConnectTimeout("connect timed out", request=request)
    backend.handler = handler

    item = make_item("test-model-q4")
    async with consumer(server) as task:
        await submit(server, item)
        await answered(task, item)

    assert item["error_code"] != "upstream_timeout"
    assert slot.healthy is False


def test_upstream_timeout_maps_to_504():
    d = app_mod._decision_from_item_error(
        {"error": "main did not respond within 1800s", "error_code": "upstream_timeout"},
        "main")
    assert (d.status, d.code) == (504, "upstream_timeout")


def test_upstream_timeout_error_shapes():
    d = app_mod._RouteDecision(status=504, code="upstream_timeout",
                               message="main did not respond within 1800s",
                               slot_name="main")
    oa = app_mod._openai_error_response(d)
    an = app_mod._anthropic_error_response(d)
    assert oa.status_code == an.status_code == 504
    assert "retry-after" not in oa.headers and "retry-after" not in an.headers
    assert json.loads(oa.body)["error"]["type"] == "upstream_timeout"
    assert json.loads(an.body)["error"]["type"] == "api_error"


# ---------------------------------------------------------------------------
# Round 2, Task 3: streaming passes the backend status through (audit A2)
# ---------------------------------------------------------------------------

class TrackedStream(httpx.AsyncByteStream):
    """Backend response body that records whether it was closed."""

    def __init__(self, chunks=(), hold: asyncio.Event | None = None):
        self.chunks = list(chunks)
        self.hold = hold
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.hold is not None:
            await self.hold.wait()

    async def aclose(self):
        self.closed = True


def sse_backend(backend, *, chunks=(), hold=None, status=200):
    """Make the backend answer with a tracked SSE stream; returns the list of
    streams opened, so tests can assert every one was closed."""
    opened: list[TrackedStream] = []

    def handler(request, n):
        stream = TrackedStream(chunks, hold)
        opened.append(stream)
        return httpx.Response(status, headers={"content-type": "text/event-stream"},
                              stream=stream)
    backend.handler = handler
    return opened


async def drive(response, *, disconnect: asyncio.Event | None = None,
                slow_start: float = 0.0,
                leave_after_first_chunk: bool = False) -> list[dict]:
    """Run an ASGI response the way uvicorn does (ASGI spec 2.3, so Starlette
    listens for http.disconnect alongside the stream)."""
    sent: list[dict] = []
    body_sent = asyncio.Event()

    async def receive():
        if leave_after_first_chunk:
            await body_sent.wait()
        else:
            await (disconnect or asyncio.Event()).wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start" and slow_start:
            await asyncio.sleep(slow_start)
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            body_sent.set()

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
             "method": "POST", "path": "/v1/chat/completions", "headers": []}
    await asyncio.wait_for(response(scope, receive, send), 3)
    return sent


async def stream_item(server, model="test-model-q4"):
    item = make_item(model, stream=True)
    async with consumer(server) as task:
        await submit(server, item)
        await answered(task, item)
    return item


@pytest.mark.asyncio
async def test_stream_backend_400_is_passed_through_as_json(server, backend):
    load(server, "test-model-q4", healthy=True)
    err = {"error": {"code": 400, "message": "Field 'temperature': type must be number",
                     "type": "invalid_request_error"}}
    opened: list = []

    def handler(request, n):
        stream = TrackedStream([json.dumps(err).encode()])
        opened.append(stream)
        return httpx.Response(400, headers={"content-type": "application/json"},
                              stream=stream)
    backend.handler = handler

    item = await stream_item(server)

    resp = item["response"]
    assert not isinstance(resp, app_mod.StreamingResponse)
    assert resp.status_code == 400
    assert json.loads(resp.body) == err
    assert len(opened) == 1 and opened[0].closed


@pytest.mark.asyncio
async def test_stream_success_relays_status_headers_and_chunks(server, backend):
    load(server, "test-model-q4", healthy=True)
    opened = sse_backend(backend, chunks=[b"data: a\n\n", b"data: [DONE]\n\n"])

    item = await stream_item(server)
    sent = await drive(item["response"])

    start = sent[0]
    assert start["status"] == 200
    assert dict(start["headers"])[b"content-type"].startswith(b"text/event-stream")
    assert b"".join(m.get("body", b"") for m in sent[1:]) == b"data: a\n\ndata: [DONE]\n\n"
    assert len(opened) == 1 and opened[0].closed


@pytest.mark.asyncio
async def test_stream_is_closed_when_the_client_leaves_before_the_first_chunk(
        server, backend):
    load(server, "test-model-q4", healthy=True)
    opened = sse_backend(backend, chunks=[b"data: a\n\n"], hold=asyncio.Event())

    item = await stream_item(server)
    gone = asyncio.Event()
    gone.set()
    await drive(item["response"], disconnect=gone, slow_start=0.05)

    assert len(opened) == 1, "backend stream should be opened by the consumer"
    assert opened[0].closed


@pytest.mark.asyncio
async def test_stream_is_closed_when_the_client_leaves_mid_stream(server, backend):
    load(server, "test-model-q4", healthy=True)
    opened = sse_backend(backend, chunks=[b"data: a\n\n"], hold=asyncio.Event())

    item = await stream_item(server)
    sent = await drive(item["response"], leave_after_first_chunk=True)

    assert b"data: a\n\n" in [m.get("body") for m in sent]
    assert len(opened) == 1 and opened[0].closed


@pytest.mark.asyncio
async def test_stream_connect_error_is_a_503_not_a_broken_stream(server, backend):
    slot = load(server, "test-model-q4", healthy=True)

    def handler(request, n):
        raise httpx.ConnectError("refused", request=request)
    backend.handler = handler

    item = await stream_item(server)

    assert item["response"] is None
    assert "Proxy error" in item["error"]
    assert slot.healthy is False


@pytest.mark.asyncio
async def test_stream_read_timeout_before_headers_is_a_504(server, backend):
    slot = load(server, "test-model-q4", healthy=True)

    def handler(request, n):
        raise httpx.ReadTimeout("timed out", request=request)
    backend.handler = handler

    item = await stream_item(server)

    assert item["response"] is None
    assert item["error_code"] == "upstream_timeout"
    assert slot.healthy is True


@pytest.mark.asyncio
async def test_stream_uses_the_configured_read_timeout(server, backend):
    load(server, "test-model-q4", healthy=True)
    seen = []

    def handler(request, n):
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=TrackedStream([b"data: [DONE]\n\n"]))
    backend.handler = handler

    item = await stream_item(server)
    await drive(item["response"])

    assert seen and seen[0]["read"] == 1800
