# tests/test_stream_busy.py
"""A slot is busy until its response completes (phase 5, Task 3; Q10).

The consumer hands a relayed stream to the client (the item's event is set at
the handoff, so the client can start reading), then waits for the stream's
closer before it takes the next item. A stream nobody starts within
STREAM_START_TIMEOUT of the handoff is closed by the consumer itself.

Hermetic like tests/test_consumer.py, whose autouse fixtures (no real
sockets, stubbed probe, MockTransport backend) are imported here. `drive`
runs a response the way uvicorn does (ASGI spec 2.3).
"""
import asyncio
import contextlib
import time

import httpx
import pytest

from manager import app as app_mod
from manager.request_log import RequestRecord
from tests.test_consumer import (  # noqa: F401  (fixtures are used by name)
    TrackedStream, answered, backend, consumer, drive, load, make_item, no_network,
    probes, server, submit, swapped,
)


def recorded(model="test-model-q4", stream=False, rid="r1"):
    """An item carrying a request record whose finish calls are noted on
    item["finished"] (status, outcome); the real, idempotent finish runs too,
    so the first call is the one that writes the line."""
    item = make_item(model, stream=stream)
    record = RequestRecord(id=rid, endpoint="/v1/chat/completions",
                           model=model, stream=stream)
    finished: list[tuple[int, str]] = []
    real = record.finish

    def finish(status, outcome):
        finished.append((status, outcome))
        real(status, outcome)
    record.finish = finish
    item["record"], item["finished"] = record, finished
    return item


def stream_then_json(backend, slot, *, chunks=(b"data: a\n\n",), hold=None,
                     stream_cls=TrackedStream, order=None):
    """Backend: call 1 answers a tracked SSE stream, every later call 200 JSON.
    Returns (opened streams, the in-flight count seen at each backend call)."""
    opened: list = []
    seen: list[int] = []

    def handler(request, n):
        seen.append(slot.in_flight_count)
        if n == 1:
            stream = stream_cls(chunks, hold)
            opened.append(stream)
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  stream=stream)
        if order is not None:
            order.append(f"backend:{n}")
        return httpx.Response(200, json={"ok": True})
    backend.handler = handler
    return opened, seen


def test_the_stream_start_timeout_is_thirty_seconds():
    assert app_mod.STREAM_START_TIMEOUT == 30.0


@pytest.mark.asyncio
async def test_the_next_item_waits_for_a_relayed_stream_to_close(server, backend):
    """in_flight is at most 1: the next item's backend call waits for the
    stream; the stream's item is answered at the handoff, not at the close."""
    slot = load(server, "test-model-q4", healthy=True)
    hold = asyncio.Event()
    opened, seen = stream_then_json(backend, slot, hold=hold)
    first, second = recorded(stream=True, rid="r1"), recorded(rid="r2")

    async with consumer(server) as task:
        await submit(server, first, second)
        await answered(task, first)          # the handoff, while the stream is open
        reading = asyncio.create_task(drive(first["response"]))
        await asyncio.sleep(0.1)
        assert len(backend.calls) == 1, "the next item started while the stream relayed"
        assert not second["event"].is_set()
        assert (slot.in_flight_count, slot.state) == (1, "serving")

        hold.set()
        sent = await reading
        await answered(task, second)

    assert b"data: a\n\n" in [m.get("body") for m in sent]
    assert opened[0].closed
    assert first["finished"][0] == (200, "ok")   # the first call writes the line
    assert seen == [1, 1]
    assert slot.in_flight_count == 0


@pytest.mark.asyncio
async def test_a_client_that_leaves_mid_stream_frees_the_slot(server, backend):
    slot = load(server, "test-model-q4", healthy=True)
    opened, seen = stream_then_json(backend, slot, hold=asyncio.Event())
    first, second = recorded(stream=True, rid="r1"), recorded(rid="r2")

    async with consumer(server) as task:
        await submit(server, first, second)
        await answered(task, first)
        await asyncio.sleep(0.05)
        assert not second["event"].is_set()
        await drive(first["response"], leave_after_first_chunk=True)
        await answered(task, second)

    assert opened[0].closed
    assert first["finished"][0] == (200, "client_disconnected")   # the first call writes the line
    assert seen == [1, 1]
    assert slot.in_flight_count == 0


@pytest.mark.asyncio
async def test_a_stream_nobody_starts_is_closed_by_the_consumer_after_the_start_timeout(
        server, backend, monkeypatch):
    """The endpoint never ran the response (its client is gone, or it raised
    after the handoff): neither the generator nor the background task ever
    runs. The consumer closes the stream itself, the line says
    client_disconnected, and the slot moves on."""
    monkeypatch.setattr(app_mod, "STREAM_START_TIMEOUT", 0.3)
    slot = load(server, "test-model-q4", healthy=True)
    opened, seen = stream_then_json(backend, slot, hold=asyncio.Event())
    first, second = recorded(stream=True, rid="r1"), recorded(rid="r2")

    async with consumer(server) as task:
        await submit(server, first, second)
        await answered(task, first)
        await asyncio.sleep(0.15)
        assert not second["event"].is_set(), "closed before the start timeout"
        assert not opened[0].closed
        await answered(task, second, timeout=2.0)

    assert opened[0].closed
    assert first["finished"][0] == (200, "client_disconnected")   # the first call writes the line
    assert seen == [1, 1]
    assert slot.in_flight_count == 0


@pytest.mark.asyncio
async def test_a_started_stream_is_not_cut_off_by_the_start_timeout(
        server, backend, monkeypatch):
    """The start timeout bounds only a stream nobody starts: once the client
    is reading, the stream runs as long as the backend sends."""
    monkeypatch.setattr(app_mod, "STREAM_START_TIMEOUT", 0.1)
    slot = load(server, "test-model-q4", healthy=True)
    hold = asyncio.Event()
    opened, seen = stream_then_json(backend, slot, hold=hold)
    first, second = recorded(stream=True, rid="r1"), recorded(rid="r2")

    async with consumer(server) as task:
        await submit(server, first, second)
        await answered(task, first)
        reading = asyncio.create_task(drive(first["response"]))
        await asyncio.sleep(0.4)
        assert not opened[0].closed, "a stream the client is reading was closed"
        assert not second["event"].is_set()

        hold.set()
        sent = await reading
        await answered(task, second)

    assert b"data: a\n\n" in [m.get("body") for m in sent]
    assert first["finished"][0] == (200, "ok")   # the first call writes the line
    assert seen == [1, 1]


@pytest.mark.asyncio
async def test_a_queued_swap_waits_for_a_relayed_stream(server, backend):
    """The README's /swap promise: requests in flight run before a queued
    swap -- a relayed stream included."""
    slot = load(server, "test-model-q4", healthy=True)
    hold = asyncio.Event()
    opened, _ = stream_then_json(backend, slot, hold=hold)
    first = recorded(stream=True, rid="r1")
    swap_item = {"kind": "swap", "model": "test-model-q8", "trial": False, "id": "s1",
                 "client": None, "user_agent": None, "accepted_at": time.monotonic(),
                 "event": asyncio.Event()}

    async with consumer(server) as task:
        await submit(server, first, swap_item)
        await answered(task, first)
        reading = asyncio.create_task(drive(first["response"]))
        await asyncio.sleep(0.1)
        assert swapped(slot) == [], "the swap started while a stream relayed"
        assert slot.state == "serving"

        hold.set()
        await reading
        await answered(task, swap_item)

    assert opened[0].closed
    assert swapped(slot) == ["test-model-q8"]
    assert slot.last_swap["outcome"] == "ok"


class RaisingClose(TrackedStream):
    """A backend stream whose close raises (a connection that breaks while
    being torn down). `order` notes when the close was attempted."""
    order: list[str] = []

    async def aclose(self):
        self.closed = True
        RaisingClose.order.append("stream:closed")
        raise RuntimeError("close failed")


@pytest.mark.asyncio
async def test_the_slot_moves_on_when_closing_the_stream_raises(server, backend):
    """The closer runs from the generator's finally when the client leaves
    mid-stream; if closing raises there, Starlette never reaches the
    background task. The request still ends, its line is still written, and
    the next item still runs -- after the close, not before."""
    slot = load(server, "test-model-q4", healthy=True)
    order: list[str] = []
    RaisingClose.order = order
    opened, seen = stream_then_json(backend, slot, hold=asyncio.Event(),
                                    stream_cls=RaisingClose, order=order)
    first, second = recorded(stream=True, rid="r1"), recorded(rid="r2")

    async with consumer(server) as task:
        await submit(server, first, second)
        await answered(task, first)
        with contextlib.suppress(Exception):
            await drive(first["response"], leave_after_first_chunk=True)
        await answered(task, second)

    assert order == ["stream:closed", "backend:2"]
    assert first["finished"][0] == (200, "client_disconnected")   # the line
    assert seen == [1, 1]
    assert slot.in_flight_count == 0


class SlowClose(TrackedStream):
    """A backend stream whose close takes a while (the connection is torn
    down over the network); it finishes when `gate` opens."""
    gate: asyncio.Event | None = None

    async def aclose(self):
        await SlowClose.gate.wait()
        self.closed = True


@pytest.mark.asyncio
async def test_the_next_item_waits_for_the_close_to_finish_not_just_to_begin(
        server, backend):
    """The client leaves before the first chunk, so the background task is
    the closer. Until the backend stream's close has finished, the slot is
    still busy with it: the next request must not reach the backend while
    the old stream is half torn down."""
    slot = load(server, "test-model-q4", healthy=True)
    SlowClose.gate = asyncio.Event()
    opened, seen = stream_then_json(backend, slot, hold=asyncio.Event(),
                                    stream_cls=SlowClose)
    first, second = recorded(stream=True, rid="r1"), recorded(rid="r2")

    async with consumer(server) as task:
        await submit(server, first, second)
        await answered(task, first)
        gone = asyncio.Event()
        gone.set()
        driving = asyncio.create_task(
            drive(first["response"], disconnect=gone, slow_start=0.05))
        await asyncio.sleep(0.2)
        assert not opened[0].closed, "the close should still be waiting on its gate"
        assert len(backend.calls) == 1, "the next request started mid-close"
        assert not second["event"].is_set()

        SlowClose.gate.set()
        await driving
        await answered(task, second)

    assert opened[0].closed
    assert seen == [1, 1]
