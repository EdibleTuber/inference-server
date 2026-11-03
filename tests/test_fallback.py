# tests/test_fallback.py
"""Phase 5 Task 4: a failed swap falls back once; then `failed` (Q20).

A swap that raises or times out swaps the slot once to its known-good model
(the model it last served healthily). If that lands, the slot is `idle` and
the swap's outcome is `fell_back`. If it fails too, or there is no
known-good model, the slot is `failed`: every queued request is answered at
once with 503 `slot_failed`, and so is every request while it stays failed.
Queued swaps still run; a /swap or a healthy probe recovers the slot.

Built on tests/test_swap_queue.py's world (a MockTransport backend that
counts calls, a stubbed probe); the `swaps` fixture below replaces its
swap_to with per-model results and gates.
"""
import asyncio
import time

import httpx
import pytest

from manager import app as app_mod
from manager.slots import SlotState
from tests.test_state_machine import FakeProbeClient, make_slot, models_response, put
from tests.test_swap_queue import (  # noqa: F401  (fixtures are used by name)
    MSGS, app, client, filler, no_probe, running, swap, swap_lines, until, world,
)

Q4, Q8 = "test-model-q4", "test-model-q8"

# Captured at import, before the autouse no_probe fixture replaces it, so the
# unit tests below run the real probe.
REAL_PROBE = SlotState.probe


class Swaps:
    """swap_to's behaviour per model: results[model] is True (lands), False
    (times out) or an exception to raise; gates[model] holds it until set."""

    def __init__(self):
        self.results: dict = {}
        self.gates: dict[str, asyncio.Event] = {}
        self.order: list[str] = []


@pytest.fixture
def swaps(world, monkeypatch):
    s = Swaps()

    async def fake_swap_to(self, model_name, env_text):
        s.order.append(model_name)
        gate = s.gates.get(model_name)
        if gate is not None:
            await gate.wait()
        result = s.results.get(model_name, True)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr("manager.swap.ModelSwapper.swap_to", fake_swap_to)
    return s


def main_of(app):
    return app.state.server.slots["main"]


async def finished(main, rid):
    await until(lambda: main.last_swap is not None and main.last_swap["id"] == rid,
                what=f"swap {rid} to finish")


def fail(slot, message="Model swap timed out on slot main: test-model-q8"):
    """Put a slot in `failed` through the machine, with the fault recorded."""
    put(slot, "failed")
    slot.record_error("swap_failed", message)
    return slot


# ---------------------------------------------------------------------------
# known_good
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_known_good_follows_a_healthy_probe_and_a_successful_swap():
    slot = make_slot()
    assert slot.known_good is None
    await REAL_PROBE(slot, FakeProbeClient(slot, models_response("foo-Q4")))
    assert slot.known_good == "foo-Q4"
    await REAL_PROBE(slot, FakeProbeClient(slot, exc=httpx.ConnectError("refused")))
    assert slot.known_good == "foo-Q4", "a failed probe must not clear it"
    await REAL_PROBE(slot, FakeProbeClient(slot, httpx.Response(200, json={"data": []})))
    assert slot.known_good == "foo-Q4", "a probe that finds no model must not clear it"
    put(slot, "swapping")
    slot.mark_swapped("bar-Q8")
    assert slot.known_good == "bar-Q8"


# ---------------------------------------------------------------------------
# Falling back
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["timed_out", "raised"])
async def test_a_failed_swap_falls_back_once_to_the_known_good_model(
        app, client, swaps, caplog, how):
    """The fallback reloads the known-good model even though loaded_model
    still names it: the failed swap restarted the backend, so the
    already-loaded shortcut must not apply."""
    main = main_of(app)
    main.known_good = Q4
    swaps.results[Q8] = False if how == "timed_out" else RuntimeError("boom")
    assert (await swap(client, Q8, "s1")).status_code == 202
    async with running(app.state.server):
        await finished(main, "s1")

    assert swaps.order == [Q8, Q4]
    assert (main.state, main.loaded_model, main.healthy) == ("idle", Q4, True)
    last = main.last_swap
    assert last["outcome"] == "fell_back"
    assert Q8 in last["message"] and Q4 in last["message"]
    assert main.last_error["code"] == "swap_failed"
    assert Q8 in main.last_error["message"]
    (line,) = swap_lines(caplog)
    assert (line["id"], line["status"], line["outcome"]) == ("s1", 202, "fell_back")


@pytest.mark.asyncio
async def test_a_slot_that_fell_back_keeps_serving_its_queue(app, client, swaps, world):
    """Falling back is not a failure of the slot: requests queued behind the
    swap run against the known-good model (one for the model that failed to
    load gets the 409 a fresh request would)."""
    main = main_of(app)
    main.known_good = Q4
    swaps.results[Q8] = False
    assert (await swap(client, Q8, "s1")).status_code == 202
    for_q4, for_q8 = filler(Q4), filler(Q8)
    await main.queue.enqueue(for_q4)
    await main.queue.enqueue(for_q8)
    async with running(app.state.server):
        await asyncio.wait_for(asyncio.gather(for_q4["event"].wait(),
                                              for_q8["event"].wait()), 2)
    assert main.last_swap["outcome"] == "fell_back"
    assert for_q4["error"] is None and for_q4["response"].status_code == 200
    assert for_q8["error_code"] == "model_not_loaded"
    assert world.backend_calls == 1


# ---------------------------------------------------------------------------
# Failed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_failed_fallback_leaves_the_slot_failed(app, client, swaps, caplog):
    main = main_of(app)
    main.known_good = Q4
    swaps.results[Q8] = False
    swaps.results[Q4] = RuntimeError("still broken")
    assert (await swap(client, Q8, "s1")).status_code == 202
    async with running(app.state.server):
        await finished(main, "s1")
        status = (await client.get("/status")).json()["slots"]["main"]

    assert swaps.order == [Q8, Q4], "falls back exactly once"
    assert main.state == "failed"
    assert (status["state"], status["healthy"], status["swapping"]) == ("failed", False, False)
    assert main.last_swap["outcome"] == "failed"
    assert Q8 in main.last_swap["message"] and Q4 in main.last_swap["message"]
    assert main.last_error["code"] == "swap_failed"
    (line,) = swap_lines(caplog)
    assert (line["id"], line["outcome"]) == ("s1", "failed")


@pytest.mark.asyncio
async def test_with_no_known_good_model_a_failed_swap_goes_straight_to_failed(
        app, client, swaps):
    main = main_of(app)
    assert main.known_good is None
    swaps.results[Q8] = False
    assert (await swap(client, Q8, "s1")).status_code == 202
    async with running(app.state.server):
        await finished(main, "s1")
    assert swaps.order == [Q8]
    assert main.state == "failed"
    assert main.last_swap["outcome"] == "failed"
    assert Q8 in main.last_swap["message"]


@pytest.mark.asyncio
async def test_entering_failed_answers_every_queued_request_at_once_and_keeps_queued_swaps(
        app, client, swaps, world):
    """The requests queued behind the failed swap are answered when it fails
    -- r2 too, though it sits behind another swap that is still running;
    that swap still runs, and recovers the slot."""
    main = main_of(app)
    swaps.results[Q8] = False
    swaps.gates[Q4] = asyncio.Event()
    assert (await swap(client, Q8, "s1")).status_code == 202
    r1, r2 = filler(Q4), filler(Q4)
    await main.queue.enqueue(r1)
    assert (await swap(client, Q4, "s2", trial=True)).status_code == 202
    await main.queue.enqueue(r2)
    async with running(app.state.server):
        await finished(main, "s1")
        await asyncio.wait_for(asyncio.gather(r1["event"].wait(), r2["event"].wait()), 2)
        assert swaps.order == [Q8, Q4], "the queued swap should be running, held"
        assert not main.queue.depth
        swaps.gates[Q4].set()
        await finished(main, "s2")

    for r in (r1, r2):
        assert r["error_code"] == "slot_failed" and r["response"] is None
    assert world.backend_calls == 0
    assert main.last_swap["outcome"] == "ok"
    assert (main.state, main.loaded_model) == ("idle", Q4)


@pytest.mark.asyncio
@pytest.mark.parametrize("path, error_type", [
    ("/v1/chat/completions", "slot_failed"), ("/v1/messages", "api_error")])
async def test_while_failed_a_request_is_answered_at_once_with_503_slot_failed(
        app, client, swaps, world, path, error_type):
    main = fail(main_of(app))
    body = {"model": Q4, "max_tokens": 8, "messages": MSGS}
    async with running(app.state.server):
        started = time.monotonic()
        r = await client.post(path, json=body)
        elapsed = time.monotonic() - started
        queued = filler(Q4)
        await main.queue.enqueue(queued)
        main.queue_event.set()
        await asyncio.wait_for(queued["event"].wait(), 2)

    assert r.status_code == 503
    assert r.json()["error"]["type"] == error_type
    assert "failed" in r.json()["error"]["message"]
    assert elapsed < 1.0
    assert queued["error_code"] == "slot_failed"
    assert world.backend_calls == 0
    assert main.state == "failed"


@pytest.mark.asyncio
async def test_slot_failed_maps_to_503():
    decision = app_mod._decision_from_item_error(
        {"error": "slot main failed: x", "error_code": "slot_failed"}, "main")
    assert (decision.status, decision.code) == (503, "slot_failed")


# ---------------------------------------------------------------------------
# Leaving failed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_healthy_probe_moves_a_failed_slot_to_idle_and_a_failed_one_leaves_it():
    slot = fail(make_slot())
    bad = FakeProbeClient(slot, exc=httpx.ConnectError("refused"))
    await REAL_PROBE(slot, bad)
    assert (bad.state_during, slot.state) == ("failed", "failed")
    good = FakeProbeClient(slot, models_response("foo-Q4"))
    await REAL_PROBE(slot, good)
    assert (good.state_during, slot.state) == ("failed", "idle")
    assert slot.known_good == "foo-Q4"


@pytest.mark.asyncio
async def test_a_swap_on_a_failed_slot_runs_and_recovers_it(app, client, swaps):
    """A /swap is accepted while failed and always queued (202): the 200
    already-loaded answer needs an idle slot, and a failed one is not."""
    main = fail(main_of(app))
    assert main.loaded_model == Q4
    assert (await swap(client, Q4, "s1")).status_code == 202
    async with running(app.state.server):
        await finished(main, "s1")
    assert swaps.order == [Q4]
    assert main.last_swap["outcome"] == "ok"
    assert (main.state, main.healthy) == ("idle", True)
