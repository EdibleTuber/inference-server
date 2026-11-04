# tests/test_client_gone.py
"""Phase 5 Task 6: a request whose client has gone is skipped when it is
dequeued (Q26).

The endpoint puts its Request on the queued item as item["request"]. When the
consumer dequeues a request item it asks whether that client is still there,
before anything that touches the slot or the backend. A gone client's item is
answered with error_code "client_gone", and the endpoint answers 499 and writes
the line (outcome client_gone).

One check is not enough on the production stack (uvicorn picks httptools). With
a request body over 64 KiB, uvicorn has stopped reading the socket by the time
the body is read (httptools_impl.py:309-310), so it has not seen the client
close. Starlette's is_disconnected() resumes reading but does not wait
(starlette/requests.py:317-329), so the disconnect only shows on a check one
event-loop tick later. Measured on 2026-11-04 with a 2 MB body and a closed
client: the first check said "connected" 17 times in 20, and the check after
one asyncio.sleep(0) was right 20 times in 20. GoneAfterATick below models
this; tests/test_client_gone_real.py runs it on the real stack.

Hermetic like tests/test_consumer.py: an httpx.MockTransport backend, a stubbed
SlotState.probe, and no real TCP at all.
"""
import asyncio
import json
import logging

import pytest
import pytest_asyncio

from manager import app as app_mod
from tests.test_consumer import (  # noqa: F401  (autouse fixtures are used by name)
    answered, backend, consumer, load, make_item, no_network, probes, server,
    submit,
)

MODEL = "test-model-q4"


class Gone:
    """A client that has gone: is_disconnected() is True at once."""

    def __init__(self):
        self.calls = 0

    async def is_disconnected(self) -> bool:
        self.calls += 1
        return True


class Connected:
    """A client that is still there."""

    def __init__(self):
        self.calls = 0

    async def is_disconnected(self) -> bool:
        self.calls += 1
        return False


class GoneAfterATick:
    """A gone client whose disconnect shows only after the event loop runs
    once more. That is uvicorn httptools with a body over 64 KiB: the first
    is_disconnected() resumes reading and answers False; the EOF is read on
    the next loop iteration. Two checks with nothing awaited between them
    both answer False."""

    def __init__(self):
        self.calls = 0
        self._seen = False

    def _read_eof(self) -> None:
        self._seen = True

    async def is_disconnected(self) -> bool:
        self.calls += 1
        if self.calls == 1:
            asyncio.get_running_loop().call_soon(self._read_eof)
            return False
        return self._seen


class Raises:
    """A check that raises: the client counts as connected."""

    async def is_disconnected(self) -> bool:
        raise RuntimeError("receive channel broken")


def gone_item(client, stream: bool = False) -> dict:
    item = make_item(MODEL, stream=stream)
    item["request"] = client
    return item


def watch_begin(slot) -> list:
    """Record begin_request calls (the slot reading `serving`)."""
    begun: list = []
    real = slot.begin_request

    def begin(request_id):
        begun.append(request_id)
        real(request_id)

    slot.begin_request = begin
    return begun


# ---------------------------------------------------------------------------
# The consumer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("healthy", [True, False])
async def test_a_gone_client_is_answered_before_readiness_or_the_backend(
        server, probes, backend, healthy):
    """Healthy: a check after begin_request or the backend call is caught by
    begun/backend.calls. Unhealthy on the same model: ready_for would probe,
    so a check after the readiness check is caught by probes.calls."""
    slot = load(server, MODEL, healthy=healthy)
    state_before = slot.stored_state
    begun = watch_begin(slot)
    item = gone_item(Gone())
    await submit(server, item)
    async with consumer(server) as task:
        await answered(task, item)

    assert item["error_code"] == "client_gone"
    assert "main" in item["error"]
    assert item["response"] is None
    assert backend.calls == []
    assert probes.calls == []
    assert begun == []
    assert slot.in_flight == {}
    assert slot.stored_state == state_before
    assert slot.last_error is None


@pytest.mark.asyncio
async def test_a_disconnect_that_shows_one_loop_tick_after_the_first_check_is_caught(
        server, backend):
    """One check, or two checks with nothing awaited between them, serves this
    client in full: the large-body case on uvicorn httptools."""
    load(server, MODEL, healthy=True)
    client = GoneAfterATick()
    item = gone_item(client)
    await submit(server, item)
    async with consumer(server) as task:
        await answered(task, item)

    assert item["error_code"] == "client_gone"
    assert backend.calls == []
    assert client.calls >= 2


@pytest.mark.asyncio
async def test_a_connected_client_is_served(server, backend):
    """Catches an inverted check, and a check that waits for the disconnect
    (the item would never be answered)."""
    load(server, MODEL, healthy=True)
    client = Connected()
    item = gone_item(client)
    await submit(server, item)
    async with consumer(server) as task:
        await answered(task, item)

    assert item["error"] is None
    assert item["response"] is not None
    assert len(backend.calls) == 1
    assert client.calls >= 1


@pytest.mark.asyncio
async def test_an_item_with_no_request_is_served_as_before(server, backend):
    """Items built without a Request (internal callers, tests) are served
    unchanged. Catches item["request"] (a KeyError, answered as queue_error)."""
    load(server, MODEL, healthy=True)
    item = make_item(MODEL)
    await submit(server, item)
    async with consumer(server) as task:
        await answered(task, item)

    assert item["error"] is None
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_a_check_that_raises_counts_the_client_as_connected(
        server, backend, caplog):
    """Catches a check outside a try (the item answered 503 queue_error, and a
    client-side event recorded as the slot's last_error)."""
    caplog.set_level(logging.WARNING, logger="manager.app")
    slot = load(server, MODEL, healthy=True)
    item = gone_item(Raises())
    await submit(server, item)
    async with consumer(server) as task:
        await answered(task, item)

    assert item["error"] is None
    assert len(backend.calls) == 1
    assert slot.last_error is None
    warnings = [r for r in caplog.records
                if r.name == "manager.app" and r.levelno == logging.WARNING
                and "main" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_a_gone_stream_request_is_skipped_too(server, backend):
    load(server, MODEL, healthy=True)
    item = gone_item(Gone(), stream=True)
    await submit(server, item)
    async with consumer(server) as task:
        await answered(task, item)

    assert item["error_code"] == "client_gone"
    assert item["response"] is None
    assert backend.calls == []


@pytest.mark.asyncio
async def test_a_gone_client_on_a_failed_slot_is_answered_client_gone(server, backend):
    """The skip comes before the failed branch: nobody is there to read a
    slot_failed answer, and the line should say why it was not served."""
    slot = load(server, MODEL, healthy=True)
    slot._set_state("swapping")
    slot._set_state("failed")
    item = gone_item(Gone())
    await submit(server, item)
    async with consumer(server) as task:
        await answered(task, item)

    assert item["error_code"] == "client_gone"
    assert slot.stored_state == "failed"
    assert backend.calls == []


@pytest.mark.asyncio
async def test_the_item_behind_a_gone_client_is_served(server, backend):
    load(server, MODEL, healthy=True)
    gone, live = gone_item(Gone()), gone_item(Connected())
    await submit(server, gone, live)
    async with consumer(server) as task:
        await answered(task, gone, live)

    assert gone["error_code"] == "client_gone"
    assert live["error"] is None
    assert len(backend.calls) == 1


# ---------------------------------------------------------------------------
# The endpoints, driven as ASGI: a receive() that reports the disconnect
# ---------------------------------------------------------------------------

CHAT = "/v1/chat/completions"
MESSAGES = "/v1/messages"
BODIES = {
    CHAT: {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    MESSAGES: {"model": MODEL, "max_tokens": 16,
               "messages": [{"role": "user", "content": "hi"}]},
}


@pytest_asyncio.fixture
async def running(test_config, caplog):
    """The real app with its lifespan running (consumers, startup probes),
    main loaded and healthy."""
    caplog.set_level(logging.INFO, logger="manager.requests")
    app = app_mod.create_app(test_config)
    async with app.router.lifespan_context(app):
        main = app.state.server.slots["main"]
        main.loaded_model, main.healthy = MODEL, True
        yield app


async def call(app, path: str, messages: list[dict]) -> tuple[int, dict]:
    """POST through the ASGI app. receive() hands out `messages`, then reports
    http.disconnect on every later call: the client is gone."""
    pending = list(messages)
    sent: list[dict] = []

    async def receive():
        if pending:
            return pending.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": path, "raw_path": path.encode(), "query_string": b"",
        "root_path": "", "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 18080),
        "headers": [(b"content-type", b"application/json")],
    }
    await asyncio.wait_for(app(scope, receive, send), 5)
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent
                    if m["type"] == "http.response.body")
    return start["status"], json.loads(body)


def request_lines(caplog) -> list[dict]:
    out = []
    for r in caplog.records:
        if r.name == "manager.requests":
            line = json.loads(r.getMessage())
            if line["event"] == "request":
                out.append(line)
    return out


def error_type(path: str, body: dict) -> str:
    if path == MESSAGES:
        assert body["type"] == "error"
    return body["error"]["type"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [CHAT, MESSAGES])
async def test_a_client_gone_while_queued_gets_499_and_one_line(
        running, backend, caplog, path):
    """Catches: no client_gone mapping in _decision_from_item_error (a 503
    queue_error), no client_gone key in the Anthropic renderer's type_by_code
    (a KeyError and a 500 with no line), an endpoint that does not put its
    Request on the item, and a check before the dequeue stamp (queue_wait_ms
    null)."""
    body = json.dumps(BODIES[path]).encode()
    status, answer = await call(running, path, [
        {"type": "http.request", "body": body, "more_body": False},
    ])

    assert status == 499
    assert error_type(path, answer) == "client_gone"
    assert backend.calls == []
    [line] = request_lines(caplog)
    assert (line["status"], line["outcome"]) == (499, "client_gone")
    assert line["slot"] == "main"
    assert isinstance(line["queue_wait_ms"], int)
    assert line["backend_ms"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [CHAT, MESSAGES])
async def test_a_client_gone_mid_body_gets_499_not_400(running, backend, caplog, path):
    """Starlette raises ClientDisconnect from request.json() when the client
    leaves before the body is complete. Before Task 6 the endpoint's
    `except Exception` answered that 400 invalid_body."""
    status, answer = await call(running, path, [
        {"type": "http.request", "body": b'{"model": "test-model-q4", "mes',
         "more_body": True},
    ])

    assert status == 499
    assert error_type(path, answer) == "client_gone"
    assert backend.calls == []
    assert running.state.server.slots["main"].queue.depth == 0
    [line] = request_lines(caplog)
    assert (line["status"], line["outcome"]) == (499, "client_gone")
    assert line["slot"] is None
