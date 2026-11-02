# tests/test_swap_queue.py
"""Phase 5 Task 1: /swap is a FIFO item in the slot's queue (Q12).

The app runs under httpx.ASGITransport, which does not run the lifespan, so
each test decides whether a consumer runs (`running`) and when. The backend
is an httpx.MockTransport whose handler can block on a gate, swap_to is a
fake that records order, and SlotState.probe is a no-op.
"""
import asyncio
import contextlib
import json
import logging
from datetime import datetime

import httpx
import pytest
import pytest_asyncio

from manager import app as app_mod
from manager.slots import SlotState

# Captured before any fixture patches httpx.AsyncClient for the backend.
REAL_ASYNC_CLIENT = httpx.AsyncClient

MSGS = [{"role": "user", "content": "hi"}]
LAST_SWAP_KEYS = {"id", "model", "trial", "outcome", "message", "at_utc"}


@pytest.fixture(autouse=True)
def no_probe(monkeypatch):
    async def probe(self, client):
        return None
    monkeypatch.setattr(SlotState, "probe", probe)


class World:
    """What happened, in order, plus gates the test can hold shut."""

    def __init__(self):
        self.order: list[str] = []
        self.backend_gate: asyncio.Event | None = None
        self.swap_gate: asyncio.Event | None = None
        self.swap_ok = True
        self.backend_calls = 0


@pytest.fixture
def world(monkeypatch):
    w = World()

    async def handler(request: httpx.Request) -> httpx.Response:
        w.backend_calls += 1
        # The consumer forwards only the body, so the tag rides in the message.
        tag = json.loads(request.content)["messages"][0]["content"]
        w.order.append(f"backend:{tag}:start")
        if w.backend_gate is not None:
            await w.backend_gate.wait()
        w.order.append(f"backend:{tag}:end")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    class Mocked(REAL_ASYNC_CLIENT):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(app_mod.httpx, "AsyncClient", Mocked)

    async def fake_swap_to(self, model_name, env_text):
        w.order.append(f"swap:{model_name}:task={asyncio.current_task().get_name()}")
        if w.swap_gate is not None:
            await w.swap_gate.wait()
        return w.swap_ok

    monkeypatch.setattr("manager.swap.ModelSwapper.swap_to", fake_swap_to)
    return w


@pytest.fixture
def app(test_config, world, caplog):
    caplog.set_level(logging.INFO, logger="manager.requests")
    a = app_mod.create_app(test_config)
    main = a.state.server.slots["main"]
    main.loaded_model, main.healthy = "test-model-q4", True
    return a


@pytest_asyncio.fixture
async def client(app):
    async with REAL_ASYNC_CLIENT(transport=httpx.ASGITransport(app=app),
                                 base_url="http://manager") as c:
        yield c


@contextlib.asynccontextmanager
async def running(server, slot="main"):
    """The slot's consumer, named exactly as the lifespan names it."""
    task = asyncio.create_task(
        app_mod._queue_consumer(server, server._config, slot_name=slot),
        name=f"queue_consumer_{slot}")
    try:
        yield task
    finally:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task


async def until(predicate, timeout=2.0, what="condition"):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.005)


def swap_lines(caplog) -> list[dict]:
    out = []
    for r in caplog.records:
        if r.name == "manager.requests":
            line = json.loads(r.getMessage())
            if line["event"] == "swap":
                out.append(line)
    return out


def chat(client, tag, model="test-model-q4"):
    return client.post("/v1/chat/completions",
                       json={"model": model, "messages": [{"role": "user", "content": tag}]})


def swap(client, model, rid, **extra):
    return client.post("/swap", json={"model": model, "target": "main", **extra},
                       headers={"X-Request-Id": rid})


def filler(model="test-model-q4") -> dict:
    """A request item put straight on the queue (no endpoint waits on it)."""
    return {"body": {"model": model, "messages": MSGS}, "event": asyncio.Event(),
            "response": None, "error": None, "error_code": None,
            "backend_path": "v1/chat/completions"}


# --- FIFO ------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_swap_runs_after_the_items_ahead_of_it_and_before_the_items_behind_it(
        app, client, world):
    """Review Focus 1. A is in flight, the swap is queued, then B: A finishes
    on the old model, the swap runs, and B is checked against the new model
    (409, never sent to the backend)."""
    server = app.state.server
    world.backend_gate = asyncio.Event()
    async with running(server):
        a = asyncio.create_task(chat(client, "A"))
        await until(lambda: "backend:A:start" in world.order, what="A at the backend")
        r = await asyncio.wait_for(swap(client, "test-model-q8", "s1"), 2)
        assert r.status_code == 202, r.text
        b = asyncio.create_task(chat(client, "B"))
        await until(lambda: server.slots["main"].queue.depth == 2, what="swap and B queued")
        assert not any(e.startswith("swap:") for e in world.order), world.order
        world.backend_gate.set()
        ra, rb = await asyncio.wait_for(asyncio.gather(a, b), 2)
    assert ra.status_code == 200
    assert rb.status_code == 409 and rb.json()["error"]["type"] == "model_not_loaded"
    assert [e.split(":task=")[0] for e in world.order] == [
        "backend:A:start", "backend:A:end", "swap:test-model-q8"]
    assert server.slots["main"].loaded_model == "test-model-q8"


@pytest.mark.asyncio
async def test_a_swap_answers_202_and_waits_in_the_queue(app, client, world):
    """With no consumer running, the 202 leaves the swap queued: nothing has run."""
    server = app.state.server
    r = await asyncio.wait_for(swap(client, "test-model-q8", "s1"), 2)
    assert r.status_code == 202
    assert r.json() == {"slot": "main", "model": "test-model-q8", "status": "swapping"}
    assert r.headers["x-request-id"] == "s1"
    assert server.slots["main"].queue.depth == 1
    assert world.order == []
    assert server.slots["main"].loaded_model == "test-model-q4"


@pytest.mark.asyncio
async def test_the_swap_runs_on_the_slots_consumer(app, client, world):
    server = app.state.server
    async with running(server):
        r = await asyncio.wait_for(swap(client, "test-model-q8", "s1"), 2)
        assert r.status_code == 202
        await until(lambda: server.slots["main"].last_swap is not None, what="the swap")
    (entry,) = [e for e in world.order if e.startswith("swap:")]
    assert entry == "swap:test-model-q8:task=queue_consumer_main"


def test_no_background_task_runs_a_swap():
    """Invariant 5: the swap is never a BackgroundTask or a bare task."""
    import inspect
    source = inspect.getsource(app_mod)
    assert "BackgroundTasks" not in source
    assert "add_task(" not in source


@pytest.mark.asyncio
async def test_a_full_queue_refuses_a_swap_with_503_queue_full(app, client, world, caplog):
    server = app.state.server
    main = server.slots["main"]
    for _ in range(main.queue.max_size):
        await main.queue.enqueue(filler())
    r = await asyncio.wait_for(swap(client, "test-model-q8", "full-1"), 2)
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "queue_full"
    assert r.headers["x-request-id"] == "full-1"
    assert main.queue.depth == main.queue.max_size
    (line,) = swap_lines(caplog)
    assert (line["id"], line["slot"], line["status"], line["outcome"], line["duration_ms"]) == (
        "full-1", "main", 503, "queue_full", None)
    assert world.order == []


# --- 200 only when nothing is ahead ---------------------------------------------

@pytest.mark.asyncio
async def test_already_loaded_with_nothing_ahead_is_200_and_queues_nothing(app, client, world,
                                                                           caplog):
    server = app.state.server
    r = await asyncio.wait_for(swap(client, "test-model-q4", "al-1"), 2)
    assert r.status_code == 200
    assert r.json() == {"slot": "main", "model": "test-model-q4", "status": "ok"}
    assert server.slots["main"].queue.depth == 0
    (line,) = swap_lines(caplog)
    assert (line["status"], line["outcome"]) == (200, "already_loaded")


@pytest.mark.parametrize("ahead", ["queued_item", "in_flight"])
@pytest.mark.asyncio
async def test_already_loaded_with_something_ahead_is_queued_as_202(app, client, world, ahead):
    """Something ahead could change the slot before this swap's turn, so the
    endpoint can't answer for the slot as it will be: it queues."""
    server = app.state.server
    main = server.slots["main"]
    if ahead == "queued_item":
        await main.queue.enqueue(filler())
    else:
        main.begin_request("held")
    r = await asyncio.wait_for(swap(client, "test-model-q4", "al-2"), 2)
    assert r.status_code == 202, r.text
    assert main.queue.depth == (2 if ahead == "queued_item" else 1)


@pytest.mark.asyncio
async def test_a_queued_swap_whose_model_is_loaded_when_it_runs_does_not_swap(
        app, client, world, caplog):
    server = app.state.server
    main = server.slots["main"]
    ahead = filler()
    await main.queue.enqueue(ahead)
    r = await asyncio.wait_for(swap(client, "test-model-q4", "al-3"), 2)
    assert r.status_code == 202
    async with running(server):
        await until(lambda: main.last_swap is not None, what="the swap item")
    assert not any(e.startswith("swap:") for e in world.order), world.order
    assert main.last_swap["outcome"] == "already_loaded"
    (line,) = swap_lines(caplog)
    assert (line["id"], line["status"], line["outcome"]) == ("al-3", 202, "already_loaded")
    assert type(line["duration_ms"]) is int


# --- last_swap and the line ------------------------------------------------------

@pytest.mark.asyncio
async def test_last_swap_reports_each_finished_swap_by_request_id(app, client, world):
    server = app.state.server
    main = server.slots["main"]
    status = (await client.get("/status")).json()["slots"]["main"]
    assert "last_swap" in status and status["last_swap"] is None

    async with running(server):
        await swap(client, "test-model-q8", "ok-1")
        await until(lambda: main.last_swap is not None, what="the first swap")
        first = (await client.get("/status")).json()["slots"]["main"]["last_swap"]

        world.swap_ok = False
        await swap(client, "test-model-q4", "bad-1")
        await until(lambda: main.last_swap["id"] == "bad-1", what="the second swap")
        second = (await client.get("/status")).json()["slots"]["main"]["last_swap"]

    assert set(first) == LAST_SWAP_KEYS
    assert (first["id"], first["model"], first["trial"], first["outcome"], first["message"]) == (
        "ok-1", "test-model-q8", False, "ok", None)
    assert datetime.fromisoformat(first["at_utc"]).tzinfo is not None
    assert (second["id"], second["outcome"]) == ("bad-1", "failed")
    assert isinstance(second["message"], str) and "test-model-q4" in second["message"]


@pytest.mark.asyncio
async def test_the_swap_line_is_written_when_the_swap_ends(app, client, world, caplog):
    server = app.state.server
    world.swap_gate = asyncio.Event()
    async with running(server):
        r = await asyncio.wait_for(swap(client, "test-model-q8", "end-1"), 2)
        assert r.status_code == 202
        await until(lambda: any(e.startswith("swap:") for e in world.order), what="swap start")
        assert swap_lines(caplog) == []
        await asyncio.sleep(0.05)
        world.swap_gate.set()
        await until(lambda: server.slots["main"].last_swap is not None, what="swap end")
    (line,) = swap_lines(caplog)
    assert (line["id"], line["slot"], line["status"], line["outcome"], line["trial"]) == (
        "end-1", "main", 202, "ok", False)
    assert line["duration_ms"] >= 50


@pytest.mark.asyncio
async def test_status_shows_swapping_while_the_consumer_runs_the_swap(app, client, world):
    server = app.state.server
    world.swap_gate = asyncio.Event()
    async with running(server):
        await asyncio.wait_for(swap(client, "test-model-q8", "sw-1"), 2)
        await until(lambda: any(e.startswith("swap:") for e in world.order), what="swap start")
        during = (await client.get("/status")).json()["slots"]["main"]
        world.swap_gate.set()
        await until(lambda: server.slots["main"].last_swap is not None, what="swap end")
        after = (await client.get("/status")).json()["slots"]["main"]
    assert (during["swapping"], during["state"]) == (True, "swapping")
    assert after["swapping"] is False


@pytest.mark.asyncio
async def test_a_swap_drained_by_a_failed_swap_ahead_still_gets_its_line(
        app, client, world, caplog):
    """Invariant 4. Task 4 changes what a failed swap does to the queue; until
    then the failure tail drains it, and a drained swap must still be
    answered, logged and reported."""
    server = app.state.server
    main = server.slots["main"]
    world.swap_ok = False
    assert (await swap(client, "test-model-q8", "first")).status_code == 202
    assert (await swap(client, "test-model-q4", "second", trial=True)).status_code == 202
    async with running(server):
        await until(lambda: main.last_swap is not None and main.last_swap["id"] == "second",
                    what="the drained swap")
    assert [e.split(":task=")[0] for e in world.order] == ["swap:test-model-q8"]
    assert main.last_swap["outcome"] == "drained"
    by_id = {line["id"]: line for line in swap_lines(caplog)}
    assert set(by_id) == {"first", "second"}
    assert (by_id["first"]["status"], by_id["first"]["outcome"]) == (202, "failed")
    assert (by_id["second"]["status"], by_id["second"]["outcome"], by_id["second"]["trial"]) == (
        202, "drained", True)


@pytest.mark.asyncio
async def test_the_swap_lines_duration_runs_from_the_202(app, client, world, caplog):
    """The wait in the queue counts: duration_ms runs from the 202, not from
    when the consumer picked the item up."""
    server = app.state.server
    assert (await swap(client, "test-model-q8", "dur-1")).status_code == 202
    await asyncio.sleep(0.1)
    async with running(server):
        await until(lambda: server.slots["main"].last_swap is not None, what="the swap")
    (line,) = swap_lines(caplog)
    assert line["duration_ms"] >= 100
