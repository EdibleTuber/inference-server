# tests/test_slot_status.py
"""/status shows what each slot is doing (phase 4, Task 3; Q27).

Hermetic like tests/test_consumer.py, whose autouse fixtures (no real
sockets, stubbed probe, MockTransport backend) are imported here. Time is
controlled by replacing the `time` module that manager.queue and
manager.slots read (see Clock), so ages are exact.
"""
import asyncio
import types
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from manager import app as app_mod
from manager import queue as queue_mod
from manager import slots as slots_mod
from manager.queue import RequestQueue
from manager.request_log import RequestRecord
from manager.slots import SlotState
from tests.test_consumer import (  # noqa: F401  (fixtures are used by name)
    TrackedStream, answered, backend, consumer, drive, load, make_item, no_network,
    probes, server, submit,
)

# Captured before the autouse `probes` fixture replaces it, so the
# probe_failed case can run the real probe against a failing client.
REAL_PROBE = SlotState.probe

EXISTING_SLOT_KEYS = {"host", "port", "loaded_model", "healthy", "swapping",
                      "last_swap_utc", "queue_depth", "queue_limit"}
NEW_SLOT_KEYS = {"state", "in_flight", "in_flight_age_seconds",
                 "oldest_queue_wait_seconds", "last_error"}


class Clock:
    """Stands in for the `time` module in manager.queue and manager.slots.
    Only monotonic() is controlled; anything else falls through to time."""

    def __init__(self, t: float = 100.0):
        self.t = t

    def monotonic(self) -> float:
        return self.t

    def __getattr__(self, name):
        import time
        return getattr(time, name)


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(queue_mod, "time", c)
    monkeypatch.setattr(slots_mod, "time", c)
    return c


def make_slot(max_size: int = 5) -> SlotState:
    return SlotState(name="main", host="127.0.0.1", port=1, env_file="/nonexistent",
                     systemd_unit="none.service", queue=RequestQueue(max_size))


def recorded(model: str = "test-model-q4", stream: bool = False, rid: str = "r1") -> dict:
    item = make_item(model, stream=stream)
    item["record"] = RequestRecord(id=rid, endpoint="/v1/chat/completions",
                                   model=model, stream=stream)
    return item


# --- the queue's head age --------------------------------------------------

@pytest.mark.asyncio
async def test_oldest_queue_wait_tracks_the_head_through_dequeue_and_drain(clock):
    q = RequestQueue(max_size=2)
    assert q.oldest_wait_seconds(100.0) is None

    clock.t = 100.0
    await q.enqueue("a")
    clock.t = 103.0
    await q.enqueue("b")
    clock.t = 105.0
    with pytest.raises(RequestQueue.QueueFullError):
        await q.enqueue("c")                    # a refused item leaves no stamp
    assert q.oldest_wait_seconds(110.0) == pytest.approx(10.0)

    assert await q.dequeue() == "a"
    assert q.oldest_wait_seconds(110.0) == pytest.approx(7.0)
    assert await q.dequeue() == "b"
    assert q.oldest_wait_seconds(110.0) is None

    clock.t = 120.0
    await q.enqueue("d")
    clock.t = 121.0
    await q.enqueue("e")
    assert q.drain() == ["d", "e"]
    assert q.oldest_wait_seconds(130.0) is None
    clock.t = 140.0
    await q.enqueue("f")                        # drain left nothing stale behind
    assert q.oldest_wait_seconds(141.0) == pytest.approx(1.0)


# --- the in-flight table ---------------------------------------------------

def test_in_flight_table_counts_ages_and_ends_idempotently(clock):
    slot = make_slot()
    assert (slot.in_flight_count, slot.oldest_in_flight_seconds(100.0)) == (0, None)

    clock.t = 100.0
    slot.begin_request("r1")
    clock.t = 104.0
    slot.begin_request("r2")
    assert slot.in_flight_count == 2
    assert slot.oldest_in_flight_seconds(110.0) == pytest.approx(10.0)

    slot.end_request("r1")
    assert slot.in_flight_count == 1
    assert slot.oldest_in_flight_seconds(110.0) == pytest.approx(6.0)
    slot.end_request("r1")                      # idempotent
    slot.end_request("never-begun")             # unknown id is ignored
    assert slot.in_flight_count == 1
    slot.end_request("r2")
    assert (slot.in_flight_count, slot.oldest_in_flight_seconds(110.0)) == (0, None)


def test_record_error_sets_code_message_and_utc_time():
    slot = make_slot()
    assert slot.last_error is None
    slot.record_error("proxy_error", "Proxy error: boom")
    err = slot.last_error
    assert set(err) == {"code", "message", "at_utc"}
    assert (err["code"], err["message"]) == ("proxy_error", "Proxy error: boom")
    at = datetime.fromisoformat(err["at_utc"])
    assert at.utcoffset() == timedelta(0)
    slot.record_error("backend_error", "second")  # the latest fault wins
    assert slot.last_error["code"] == "backend_error"


@pytest.mark.asyncio
@pytest.mark.parametrize("swapping, healthy, in_flight, expected", [
    pytest.param(True, False, True, "swapping", id="swapping_beats_unhealthy"),
    pytest.param(False, False, True, "unhealthy", id="unhealthy_beats_serving"),
    pytest.param(False, True, True, "serving", id="serving"),
    pytest.param(False, True, False, "idle", id="idle"),
])
async def test_state_priority(swapping, healthy, in_flight, expected):
    slot = make_slot()
    slot.healthy = healthy
    if in_flight:
        slot.begin_request("r1")
    if swapping:
        await slot.swap_lock.acquire()
    try:
        assert slot.state == expected
        assert slot.to_status_dict()["state"] == expected
    finally:
        if swapping:
            slot.swap_lock.release()


@pytest.mark.asyncio
async def test_status_dict_reports_ages_rounded_and_null_when_empty(clock):
    slot = make_slot()
    d = slot.to_status_dict()
    # A superset: later phases add keys (phase 5: last_swap, supervisor).
    assert EXISTING_SLOT_KEYS | NEW_SLOT_KEYS <= set(d)
    assert (d["in_flight"], d["in_flight_age_seconds"], d["oldest_queue_wait_seconds"],
            d["last_error"]) == (0, None, None, None)

    clock.t = 100.0
    slot.begin_request("r1")
    clock.t = 101.0
    await slot.queue.enqueue("waiting")
    clock.t = 112.36
    d = slot.to_status_dict()
    assert (d["in_flight"], d["in_flight_age_seconds"], d["oldest_queue_wait_seconds"]) == (
        1, 12.4, 11.4)
    assert type(d["in_flight"]) is int


# --- in flight, through the consumer ----------------------------------------

def _observing(backend, slot, respond):
    """Backend handler that notes the slot's in-flight count when called."""
    seen: list[int] = []

    def handler(request, n):
        seen.append(slot.in_flight_count)
        return respond(request, n)
    backend.handler = handler
    return seen


def _raise(exc_type):
    def respond(request, n):
        raise exc_type("boom", request=request)
    return respond


def _stream_response(hold=None, status=200, opened=None):
    def respond(request, n):
        stream = TrackedStream([b"data: a\n\n"], hold)
        if opened is not None:
            opened.append(stream)
        return httpx.Response(status, headers={"content-type": "text/event-stream"},
                              stream=stream)
    return respond


EXIT_PATHS = ["ok", "backend_5xx", "read_timeout", "connect_error", "readiness_409",
              "consumer_cancelled", "client_disconnected", "stream_ok", "stream_backend_5xx"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", EXIT_PATHS)
async def test_in_flight_returns_to_zero_on_every_exit_path(server, backend, path):
    """Review Focus 2 and 5: a leaked entry would show the slot serving forever."""
    slot = load(server, "test-model-q4", healthy=True)
    stream = path in ("client_disconnected", "stream_ok", "stream_backend_5xx")
    item = recorded(stream=stream)
    respond = {
        "ok": lambda r, n: httpx.Response(200, json={"ok": True}),
        "backend_5xx": lambda r, n: httpx.Response(500, json={"error": "boom"}),
        "read_timeout": _raise(httpx.ReadTimeout),
        "connect_error": _raise(httpx.ConnectError),
        "readiness_409": lambda r, n: httpx.Response(200, json={"ok": True}),
        "client_disconnected": _stream_response(hold=asyncio.Event()),
        "stream_ok": _stream_response(),
        "stream_backend_5xx": _stream_response(status=500),
    }.get(path)

    if path == "consumer_cancelled":
        entered, never = asyncio.Event(), asyncio.Event()
        seen: list[int] = []

        async def hang(request, n):
            seen.append(slot.in_flight_count)
            entered.set()
            await never.wait()
        backend.handler = hang
        async with consumer(server):
            await submit(server, item)
            await asyncio.wait_for(entered.wait(), 2)
            assert seen == [1]
        # leaving the block cancelled the consumer mid-request
        assert slot.in_flight_count == 0
        return

    seen = _observing(backend, slot, respond)
    if path == "readiness_409":
        load(server, "test-model-q8", healthy=True)
    async with consumer(server) as task:
        await submit(server, item)
        await answered(task, item)

    if path == "readiness_409":
        assert seen == [] and item["error_code"] == "model_not_loaded"
    else:
        assert seen == [1], "a request is in flight while its backend call runs"
    if path == "client_disconnected":
        gone = asyncio.Event()
        gone.set()
        await drive(item["response"], disconnect=gone, slow_start=0.05)
    elif path == "stream_ok":
        await drive(item["response"])
    await asyncio.sleep(0)
    assert slot.in_flight_count == 0


@pytest.mark.asyncio
async def test_in_flight_counts_a_held_stream_then_drops_on_close(server, backend, clock):
    slot = load(server, "test-model-q4", healthy=True)
    opened: list = []
    _observing(backend, slot, _stream_response(hold=asyncio.Event(), opened=opened))
    item = recorded(stream=True)
    clock.t = 100.0
    async with consumer(server) as task:
        await submit(server, item)
        await answered(task, item)

    assert isinstance(item["response"], app_mod.StreamingResponse)
    assert slot.in_flight_count == 1, "a relayed stream is in flight until it closes"
    assert slot.state == "serving"
    clock.t = 103.0
    assert slot.oldest_in_flight_seconds(103.0) == pytest.approx(3.0)

    await drive(item["response"], leave_after_first_chunk=True)
    await asyncio.sleep(0)
    assert opened[0].closed
    assert slot.in_flight_count == 0
    assert slot.state == "idle"


@pytest.mark.asyncio
async def test_two_requests_in_flight_when_a_stream_is_relayed_and_the_next_starts(
        server, backend):
    """Pins today's pre-Q10 behavior: the consumer takes the next item while a
    relayed stream is still running. Phase 5 changes this test on purpose."""
    slot = load(server, "test-model-q4", healthy=True)
    hold = asyncio.Event()
    stream_respond = _stream_response(hold=hold)

    def respond(request, n):
        if n == 1:
            return stream_respond(request, n)
        return httpx.Response(200, json={"ok": True})
    seen = _observing(backend, slot, respond)

    first, second = recorded(stream=True, rid="r1"), recorded(rid="r2")
    async with consumer(server) as task:
        await submit(server, first, second)
        await answered(task, first, second)

    assert seen == [1, 2]
    assert slot.in_flight_count == 1            # the stream is still open
    gone = asyncio.Event()
    gone.set()
    await drive(first["response"], disconnect=gone, slow_start=0.05)
    await asyncio.sleep(0)
    assert slot.in_flight_count == 0


# --- last_error ---------------------------------------------------------------

class _FailingClient:
    async def get(self, url, timeout=None):
        raise httpx.ConnectError("refused", request=httpx.Request("GET", url))


async def _through_consumer(server, item):
    async with consumer(server) as task:
        await submit(server, item)
        await answered(task, item)
    return item


SLOT_FAULTS = {
    # case: (respond, stream, expected code)
    "backend_5xx": (lambda r, n: httpx.Response(500, json={"error": "boom"}), False,
                    "backend_error"),
    "stream_backend_5xx": (_stream_response(status=503), True, "backend_error"),
    "upstream_timeout": (_raise(httpx.ReadTimeout), False, "upstream_timeout"),
    "proxy_error": (_raise(httpx.ConnectError), False, "proxy_error"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(SLOT_FAULTS) + [
    "queue_error", "swap_failed", "swap_raised", "probe_failed"])
async def test_last_error_is_set_by_slot_faults(server, backend, case):
    slot = load(server, "test-model-q4", healthy=True)
    if case in SLOT_FAULTS:
        respond, stream, code = SLOT_FAULTS[case]
        backend.handler = respond
        await _through_consumer(server, recorded(stream=stream))
    elif case == "queue_error":
        code = "queue_error"

        async def broken(slot_name, model_name):
            raise RuntimeError("readiness exploded")
        server.ready_for = broken
        item = await _through_consumer(server, recorded())
        assert item["error_code"] == "queue_error"
    elif case in ("swap_failed", "swap_raised"):
        code = "swap_failed"
        slot.swapper.swap_to = (AsyncMock(return_value=False) if case == "swap_failed"
                                else AsyncMock(side_effect=RuntimeError("unit gone")))
        assert await server.ensure_model_on_slot("main", "test-model-q8") is False
    else:
        code = "probe_failed"
        await REAL_PROBE(slot, _FailingClient())
        assert slot.healthy is False
    assert slot.last_error is not None, f"{case} must record a slot fault"
    assert slot.last_error["code"] == code
    assert isinstance(slot.last_error["message"], str) and slot.last_error["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [
    "backend_4xx", "queue_full", "model_not_loaded", "client_disconnected", "swap_refused"])
async def test_last_error_is_not_set_by_client_faults(server, backend, case):
    slot = load(server, "test-model-q4", healthy=True)
    if case == "backend_4xx":
        backend.handler = lambda r, n: httpx.Response(400, json={"error": "bad field"})
        item = await _through_consumer(server, recorded())
        assert item["response"].status_code == 400
    elif case == "queue_full":
        for _ in range(slot.queue.max_size):
            await slot.queue.enqueue(make_item("test-model-q4"))
        body = {"model": "test-model-q4", "messages": [{"role": "user", "content": "hi"}]}
        decision = await app_mod._route_and_wait(server, body, "v1/chat/completions")
        assert decision.code == "queue_full"
    elif case == "model_not_loaded":
        load(server, "test-model-q8", healthy=True)
        item = await _through_consumer(server, recorded("test-model-q4"))
        assert item["error_code"] == "model_not_loaded"
    elif case == "client_disconnected":
        backend.handler = _stream_response(hold=asyncio.Event())
        item = await _through_consumer(server, recorded(stream=True))
        gone = asyncio.Event()
        gone.set()
        await drive(item["response"], disconnect=gone, slow_start=0.05)
        await asyncio.sleep(0)
    else:
        # No profile: a refusal, nothing was stopped (ensure_model_on_slot).
        assert await server.ensure_model_on_slot("main", "no-such-model") is False
    assert slot.last_error is None


@pytest.mark.asyncio
async def test_last_error_is_never_cleared_by_a_later_success(server, backend):
    slot = load(server, "test-model-q4", healthy=True)
    backend.handler = _raise(httpx.ReadTimeout)
    await _through_consumer(server, recorded())
    first = dict(slot.last_error)
    backend.handler = lambda r, n: httpx.Response(200, json={"ok": True})
    await _through_consumer(server, recorded(rid="r2"))
    assert slot.last_error == first


# --- /status ------------------------------------------------------------------

def test_status_keeps_every_existing_key(test_config, monkeypatch):
    """Every key from before this phase is present with its old meaning, and
    the new keys ride alongside (agent_core and PAL read the old ones)."""
    async def no_gpu():
        return []
    monkeypatch.setattr(app_mod, "get_gpu_info_async", no_gpu)
    app = app_mod.create_app(test_config)
    with TestClient(app) as c:
        main = app.state.server.slots["main"]
        main.loaded_model, main.healthy = "test-model-q4", True
        main.last_swap_utc = "2026-11-01T00:00:00+00:00"
        body = c.get("/status").json()
    for name, slot in app.state.server.slots.items():
        got = body["slots"][name]
        assert EXISTING_SLOT_KEYS | {"models"} | NEW_SLOT_KEYS <= set(got), name
        assert (got["host"], got["port"]) == (slot.host, slot.port)
        assert (got["queue_depth"], got["queue_limit"]) == (0, slot.queue.max_size)
        assert got["swapping"] is False
    got = body["slots"]["main"]
    assert (got["loaded_model"], got["healthy"], got["last_swap_utc"]) == (
        "test-model-q4", True, "2026-11-01T00:00:00+00:00")
    assert (got["state"], got["in_flight"], got["in_flight_age_seconds"],
            got["oldest_queue_wait_seconds"], got["last_error"]) == (
        "idle", 0, None, None, None)
    assert {"gpu", "uptime_seconds", "profile_errors"} <= set(body)


@pytest.mark.asyncio
async def test_two_in_flight_requests_sharing_a_client_request_id_are_both_counted(
        server, backend):
    """Review round 1: a client-supplied X-Request-Id is kept as given, so two
    requests can carry the same id (a retry, a client with a fixed id). The
    in-flight table must count each request, not each id: ending one must
    not end the other."""
    slot = load(server, "test-model-q4", healthy=True)
    hold = asyncio.Event()
    stream_respond = _stream_response(hold=hold)

    def respond(request, n):
        if n == 1:
            return stream_respond(request, n)
        return httpx.Response(200, json={"ok": True})
    seen = _observing(backend, slot, respond)

    first, second = recorded(stream=True, rid="same"), recorded(rid="same")
    async with consumer(server) as task:
        await submit(server, first, second)
        await answered(task, first, second)

    assert seen == [1, 2], "the second request is in flight alongside the first"
    assert slot.in_flight_count == 1, "the stream is still open after the second ended"
    gone = asyncio.Event()
    gone.set()
    await drive(first["response"], disconnect=gone, slow_start=0.05)
    await asyncio.sleep(0)
    assert slot.in_flight_count == 0
