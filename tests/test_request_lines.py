# tests/test_request_lines.py
"""One structured line per request and per swap, through the app (phase 4,
Task 2; Q27).

Hermetic like tests/test_consumer.py: the backend is an httpx.MockTransport,
SlotState.probe is stubbed, and swaps are short-circuited.
"""
import asyncio
import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from manager import app as app_mod
from manager.request_log import RequestRecord
from manager.slots import SlotState
from tests.test_consumer import (
    TrackedStream, answered, consumer, drive, load, make_item, sse_backend, submit,
)
from tests.test_request_log import (
    ANTHROPIC_BODY, KEYS, OPENAI_BODY, OPENAI_LAST_CHUNK, OPENAI_STREAM,
)

SWAP_KEYS = ["event", "id", "client", "user_agent", "slot", "model", "trial", "status",
             "outcome", "duration_ms"]
MSGS = [{"role": "user", "content": "hi"}]


class Backend:
    def __init__(self):
        self.handler = lambda request: httpx.Response(200, json=OPENAI_BODY)

    def __call__(self, request):
        return self.handler(request)


@pytest.fixture
def backend(monkeypatch):
    b = Backend()
    real = httpx.AsyncClient

    class Mocked(real):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(b)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(app_mod.httpx, "AsyncClient", Mocked)
    return b


@pytest.fixture(autouse=True)
def no_probe(monkeypatch):
    async def probe(self, client):
        return None
    monkeypatch.setattr(SlotState, "probe", probe)


@pytest.fixture
def client(test_config, backend, caplog):
    caplog.set_level(logging.INFO, logger="manager.requests")
    app = app_mod.create_app(test_config)
    with TestClient(app) as c:
        main = app.state.server.slots["main"]
        main.loaded_model, main.healthy = "test-model-q4", True
        yield c


def lines(caplog, event="request") -> list[dict]:
    out = []
    for r in caplog.records:
        if r.name == "manager.requests":
            line = json.loads(r.getMessage())
            if line["event"] == event:
                out.append(line)
    return out


def _sse(chunks):
    return lambda request: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=TrackedStream(chunks))


# --- one line per request, every key ------------------------------------------

CASES = {
    "chat": ("/v1/chat/completions", {"model": "test-model-q4", "messages": MSGS},
             lambda r: httpx.Response(200, json=OPENAI_BODY), (12, 8, 4)),
    "messages": ("/v1/messages", {"model": "test-model-q4", "max_tokens": 4, "messages": MSGS},
                 lambda r: httpx.Response(200, json=ANTHROPIC_BODY), (12, 8, 4)),
    "count_tokens": ("/v1/messages/count_tokens", {"model": "test-model-q4", "messages": MSGS},
                     lambda r: httpx.Response(200, json={"input_tokens": 12}), (12, None, None)),
    "stream": ("/v1/chat/completions",
               {"model": "test-model-q4", "stream": True, "messages": MSGS},
               _sse([OPENAI_STREAM[:40], OPENAI_STREAM[40:]]), (12, 0, 8)),
}


@pytest.mark.parametrize("case", CASES)
def test_one_line_per_request_with_every_key(client, backend, caplog, case):
    path, body, handler, tokens = CASES[case]
    backend.handler = handler
    r = client.post(path, json=body)
    assert r.status_code == 200
    (line,) = lines(caplog)
    assert list(line) == KEYS
    assert line["endpoint"] == path
    assert (line["slot"], line["model"], line["status"], line["outcome"]) == (
        "main", "test-model-q4", 200, "ok")
    assert line["stream"] is (case == "stream")
    assert line["client"] == "testclient"
    assert line["user_agent"] == "testclient"
    assert (line["prompt_tokens"], line["cached_tokens"], line["completion_tokens"]) == tokens
    assert type(line["backend_ms"]) is int and line["backend_ms"] >= 0
    if case == "count_tokens":
        assert line["queue_wait_ms"] is None
    else:
        assert type(line["queue_wait_ms"]) is int and line["queue_wait_ms"] >= 0
    assert r.headers["x-request-id"] == line["id"]


def test_a_streamed_response_is_relayed_unchanged(client, backend):
    backend.handler = _sse([OPENAI_STREAM[:40], OPENAI_STREAM[40:]])
    r = client.post("/v1/chat/completions",
                    json={"model": "test-model-q4", "stream": True, "messages": MSGS})
    assert r.content == OPENAI_STREAM


def test_a_good_request_id_is_kept_and_echoed(client, caplog):
    r = client.post("/v1/chat/completions", headers={"X-Request-Id": "abc-123"},
                    json={"model": "test-model-q4", "messages": MSGS})
    assert r.headers["x-request-id"] == "abc-123"
    assert lines(caplog)[0]["id"] == "abc-123"


def test_a_malformed_request_id_is_replaced(client, caplog):
    r = client.post("/v1/chat/completions", headers={"X-Request-Id": "has space"},
                    json={"model": "test-model-q4", "messages": MSGS})
    rid = r.headers["x-request-id"]
    assert len(rid) == 32 and lines(caplog)[0]["id"] == rid


# --- outcomes -------------------------------------------------------------------

def _queue_full(client):
    client.app.state.server.slots["main"].queue._max_size = 0


ERRORS = {
    "model_not_found": ({"model": "no-such-model", "messages": MSGS}, None, 404),
    "missing_model": ({"messages": MSGS}, None, 400),
    "queue_full": ({"model": "test-model-q4", "messages": MSGS}, _queue_full, 503),
    "model_not_loaded": ({"model": "test-model-q8", "messages": MSGS}, None, 409),
}


@pytest.mark.parametrize("code", ERRORS)
def test_manager_errors_log_their_code(client, caplog, code):
    body, setup, status = ERRORS[code]
    if setup:
        setup(client)
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == status
    (line,) = lines(caplog)
    assert (line["status"], line["outcome"]) == (status, code)
    assert line["backend_ms"] is None
    assert r.headers["x-request-id"] == line["id"]


def test_an_invalid_json_body_logs_invalid_body(client, caplog):
    r = client.post("/v1/messages", content=b"{not json",
                    headers={"content-type": "application/json"})
    assert r.status_code == 400
    (line,) = lines(caplog)
    assert (line["outcome"], line["model"]) == ("invalid_body", None)


def test_a_backend_5xx_logs_backend_error_with_its_status(client, backend, caplog):
    backend.handler = lambda r: httpx.Response(500, json={"error": "boom"})
    r = client.post("/v1/chat/completions", json={"model": "test-model-q4", "messages": MSGS})
    assert r.status_code == 500
    (line,) = lines(caplog)
    assert (line["status"], line["outcome"]) == (500, "backend_error")


def test_a_backend_4xx_logs_backend_error(client, backend, caplog):
    backend.handler = lambda r: httpx.Response(400, json={"error": "bad temp"})
    client.post("/v1/chat/completions", json={"model": "test-model-q4", "messages": MSGS})
    (line,) = lines(caplog)
    assert (line["status"], line["outcome"]) == (400, "backend_error")


def test_an_upstream_timeout_logs_its_code(client, backend, caplog):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)
    backend.handler = handler
    r = client.post("/v1/chat/completions", json={"model": "test-model-q4", "messages": MSGS})
    assert r.status_code == 504
    (line,) = lines(caplog)
    assert (line["status"], line["outcome"]) == (504, "upstream_timeout")
    assert type(line["backend_ms"]) is int


def test_a_raising_log_write_does_not_fail_the_request(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("journal gone")
    monkeypatch.setattr(logging.getLogger("manager.requests"), "info", boom)
    r = client.post("/v1/chat/completions", json={"model": "test-model-q4", "messages": MSGS})
    assert r.status_code == 200 and r.json() == OPENAI_BODY


# --- streams, at the consumer -------------------------------------------------

def _record(stream=True) -> RequestRecord:
    return RequestRecord(id="r1", endpoint="/v1/chat/completions", client=None,
                         user_agent=None, model="test-model-q4", stream=stream)


@pytest.fixture
def server(test_config):
    from unittest.mock import AsyncMock
    s = app_mod.ServerState(test_config)
    for slot in s.slots.values():
        slot.swapper.swap_to = AsyncMock(return_value=True)
    return s


async def _stream(server, chunks, hold=None):
    item = make_item("test-model-q4", stream=True)
    item["record"] = _record()
    async with consumer(server) as task:
        await submit(server, item)
        await answered(task, item)
    return item


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["mid_stream", "before_first_chunk"])
async def test_a_stream_the_client_leaves_logs_once_as_client_disconnected(
        server, backend, caplog, how):
    caplog.set_level(logging.INFO, logger="manager.requests")
    load(server, "test-model-q4", healthy=True)
    hold = asyncio.Event()
    opened = sse_backend(backend, chunks=[b"data: a\n\n"], hold=hold)
    backend.handler = lambda r, _h=backend.handler: _h(r, 1)
    item = await _stream(server, None)
    if how == "mid_stream":
        await drive(item["response"], leave_after_first_chunk=True)
    else:
        gone = asyncio.Event()
        gone.set()
        await drive(item["response"], disconnect=gone, slow_start=0.05)
    await asyncio.sleep(0)
    assert opened[0].closed
    (line,) = lines(caplog)
    assert (line["id"], line["status"], line["outcome"]) == ("r1", 200, "client_disconnected")


@pytest.mark.asyncio
async def test_a_stream_that_completes_logs_ok_with_its_tokens(server, backend, caplog):
    caplog.set_level(logging.INFO, logger="manager.requests")
    load(server, "test-model-q4", healthy=True)
    sse_backend(backend, chunks=[OPENAI_STREAM[:7], OPENAI_STREAM[7:]])
    backend.handler = lambda r, _h=backend.handler: _h(r, 1)
    item = await _stream(server, None)
    sent = await drive(item["response"])
    assert b"".join(m.get("body", b"") for m in sent[1:]) == OPENAI_STREAM
    (line,) = lines(caplog)
    assert (line["outcome"], line["prompt_tokens"], line["completion_tokens"]) == ("ok", 12, 8)
    assert type(line["backend_ms"]) is int


@pytest.mark.asyncio
async def test_unparseable_bodies_give_null_tokens_and_unchanged_bytes(server, backend, caplog):
    """Review Focus 3, through the relay."""
    caplog.set_level(logging.INFO, logger="manager.requests")
    load(server, "test-model-q4", healthy=True)
    chunks = [b"<html>502</html>\n", b'data: {"timings": {"prompt_n": \n\n',
              b"data: " + b"x" * (2 * 1024 * 1024), b"\n\ndata: [DONE]\n\n"]
    sse_backend(backend, chunks=chunks)
    backend.handler = lambda r, _h=backend.handler: _h(r, 1)
    item = await _stream(server, None)
    sent = await drive(item["response"])
    assert b"".join(m.get("body", b"") for m in sent[1:]) == b"".join(chunks)
    (line,) = lines(caplog)
    assert line["outcome"] == "ok"
    assert (line["prompt_tokens"], line["cached_tokens"], line["completion_tokens"]) == (
        None, None, None)


@pytest.mark.asyncio
async def test_an_item_without_a_record_is_served_as_before(server, backend, caplog):
    """Items built before this phase (and by older tests) carry no record."""
    caplog.set_level(logging.INFO, logger="manager.requests")
    load(server, "test-model-q4", healthy=True)
    item = make_item("test-model-q4")
    async with consumer(server) as task:
        await submit(server, item)
        await answered(task, item)
    assert item["error"] is None and item["response"].status_code == 200
    assert lines(caplog) == []


# --- swaps ----------------------------------------------------------------------

@pytest.fixture
def swap_result(monkeypatch):
    result = {"ok": True}

    async def fake_swap_to(self, model_name, env_text):
        return result["ok"]
    monkeypatch.setattr("manager.swap.ModelSwapper.swap_to", fake_swap_to)
    return result


SWAPS = {
    "refused": ({"model": "test-model-q4", "target": "nope"}, True, 400, "invalid_target"),
    "already_loaded": ({"model": "test-model-q4", "target": "main"}, True, 200, "already_loaded"),
    "accepted_ok": ({"model": "test-model-q8", "target": "main"}, True, 202, "ok"),
    "accepted_failed": ({"model": "test-model-q8", "target": "main"}, False, 202, "failed"),
}


@pytest.mark.parametrize("case", SWAPS)
def test_swap_lines(client, caplog, swap_result, case):
    body, ok, status, outcome = SWAPS[case]
    swap_result["ok"] = ok
    r = client.post("/swap", json=body, headers={"X-Request-Id": f"swap-{case}"})
    assert r.status_code == status
    assert r.headers["x-request-id"] == f"swap-{case}"
    (line,) = lines(caplog, event="swap")
    assert list(line) == SWAP_KEYS
    assert (line["id"], line["status"], line["outcome"], line["model"], line["trial"]) == (
        f"swap-{case}", status, outcome, body["model"], False)
    if status == 202:
        assert line["slot"] == "main"
        assert type(line["duration_ms"]) is int and line["duration_ms"] >= 0
    else:
        assert line["duration_ms"] is None
    assert lines(caplog) == []                  # a swap is not also a request line


# --- review round 1: a line even when the request path raises ---------------

def test_a_count_tokens_backend_failure_still_writes_a_line(test_config, backend, caplog):
    """The endpoint's pre-existing behavior on an unreachable backend (the
    exception escapes, the client gets a 500) is kept; it must still log."""
    caplog.set_level(logging.INFO, logger="manager.requests")

    def handler(request):
        raise httpx.ConnectError("down", request=request)
    backend.handler = handler
    app = app_mod.create_app(test_config)
    with TestClient(app, raise_server_exceptions=False) as c:
        main = app.state.server.slots["main"]
        main.loaded_model, main.healthy = "test-model-q4", True
        r = c.post("/v1/messages/count_tokens", json={"model": "test-model-q4", "messages": MSGS})
    assert r.status_code == 500
    (line,) = lines(caplog)
    assert (line["endpoint"], line["status"], line["outcome"], line["slot"]) == (
        "/v1/messages/count_tokens", 500, "proxy_error", "main")


@pytest.mark.asyncio
async def test_a_raising_stream_closer_still_writes_the_line(caplog):
    """Client gone before the first chunk: the background task is the only
    closer. If closing the backend stream raises, the line is still written."""
    caplog.set_level(logging.INFO, logger="manager.requests")

    async def close():
        raise RuntimeError("aexit failed")
    resp = httpx.Response(200, content=b"data: x\n\n",
                          headers={"content-type": "text/event-stream"})
    record = _record()
    relayed = app_mod._relay_stream(resp, close, record=record)
    try:
        await relayed.background()
    except RuntimeError:
        pass                                    # propagating it is allowed
    (line,) = lines(caplog)
    assert (line["id"], line["outcome"]) == ("r1", "client_disconnected")
