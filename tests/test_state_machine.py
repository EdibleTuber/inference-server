# tests/test_state_machine.py
"""Phase 5 Task 2: one state, one writer (Q19).

Each slot stores one state, moved only through manager.slots.TRANSITIONS,
and only by the slot's own queue consumer (the task named
queue_consumer_<slot>). `serving` is not stored: it is `idle` with a request
in flight. `healthy` and `swapping` in /status derive from the state.
/status, routing and startup ask the consumer to probe (SlotState.refresh)
instead of probing themselves.

The backends are one httpx.MockTransport that answers /v1/models per slot
(by port) and chat calls; the real SlotState.probe runs against it.
"""
import asyncio
import contextlib
import json
import logging
import re
import time
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from manager import app as app_mod
from manager import slots as slots_mod
from manager.queue import RequestQueue
from manager.slots import IllegalTransition, SlotState

# Captured before any fixture patches httpx.AsyncClient for the backends.
REAL_ASYNC_CLIENT = httpx.AsyncClient
MANAGER_DIR = Path(app_mod.__file__).parent
MSGS = [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_slot(name: str = "main") -> SlotState:
    return SlotState(name=name, host="127.0.0.1", port=1, env_file="/nonexistent",
                     systemd_unit="none.service", queue=RequestQueue(5))


def put(slot: SlotState, state: str) -> SlotState:
    """Put a slot in `state`: the test-only healthy setter, then legal moves."""
    if state in ("unhealthy", "probing", "swapping"):
        slot.healthy = False
        if state != "unhealthy":
            slot._set_state(state)
    elif state == "idle":
        slot.healthy = True
    else:
        raise ValueError(state)
    assert slot.state == state
    return slot


def slot_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.name == "manager.slots" and re.match(r"slot=\S+ \S+ -> \S+$", r.getMessage())]


async def until(predicate, timeout=2.0, what="condition"):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.005)


class FakeProbeClient:
    """What SlotState.probe is handed: notes the slot's state when called."""

    def __init__(self, slot, response=None, exc=None):
        self.slot, self.response, self.exc = slot, response, exc
        self.state_during: str | None = None

    async def get(self, url, timeout=None):
        self.state_during = self.slot.state
        if self.exc is not None:
            raise self.exc
        return self.response


def models_response(model: str) -> httpx.Response:
    return httpx.Response(200, json={"data": [{"id": f"/models/{model}.gguf"}]})


class Backends:
    """Both slots' backends. models[slot] is what /v1/models reports (None:
    503, still loading). Records probes and chat calls in `order`."""

    def __init__(self, ports: dict[int, str]):
        self.ports = ports
        self.models: dict[str, str | None] = {"main": "test-model-q4", "egpu": None}
        self.order: list[str] = []
        self.probe_gate: asyncio.Event | None = None
        self.chat_gate: asyncio.Event | None = None
        self.chat_status = 200
        self.chat_exc: type | None = None

    @property
    def probes(self) -> list[str]:
        return [e.split(":", 1)[1] for e in self.order if e.startswith("probe:")]

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        slot = self.ports[request.url.port]
        if request.method == "GET" and request.url.path == "/v1/models":
            self.order.append(f"probe:{slot}")
            if self.probe_gate is not None:
                await self.probe_gate.wait()
            model = self.models[slot]
            if model is None:
                return httpx.Response(503, json={"error": {"message": "Loading model"}})
            return models_response(model)
        tag = json.loads(request.content)["messages"][0]["content"]
        self.order.append(f"chat:{slot}:{tag}")
        if self.chat_gate is not None:
            await self.chat_gate.wait()
        if self.chat_exc is not None:
            raise self.chat_exc("boom", request=request)
        return httpx.Response(self.chat_status,
                              json={"choices": [{"message": {"content": "ok"}}]})


@pytest.fixture
def backends(test_config, monkeypatch):
    b = Backends({sc.port: sc.name for sc in test_config.slots})

    class Mocked(REAL_ASYNC_CLIENT):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(b)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(app_mod.httpx, "AsyncClient", Mocked)

    async def fake_swap_to(self, model_name, env_text):
        return True
    monkeypatch.setattr("manager.swap.ModelSwapper.swap_to", fake_swap_to)
    monkeypatch.setattr(app_mod, "get_gpu_info_async", AsyncMock(return_value=[]))
    return b


@pytest.fixture
def writes(monkeypatch):
    """Every call of probe, _set_state, mark_swapped and mark_unhealthy, as
    (method, slot, task name)."""
    seen: list[tuple[str, str, str]] = []

    def wrap(name):
        real = getattr(SlotState, name)
        if asyncio.iscoroutinefunction(real):
            async def wrapper(self, *a, **k):
                seen.append((name, self.name, asyncio.current_task().get_name()))
                return await real(self, *a, **k)
        else:
            def wrapper(self, *a, **k):
                task = asyncio.current_task() if _loop_running() else None
                seen.append((name, self.name, task.get_name() if task else "<no task>"))
                return real(self, *a, **k)
        monkeypatch.setattr(SlotState, name, wrapper)

    for name in ("probe", "_set_state", "mark_swapped", "mark_unhealthy"):
        wrap(name)
    return seen


def _loop_running() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


@pytest.fixture
def server(test_config, backends):
    return app_mod.ServerState(test_config)


@contextlib.asynccontextmanager
async def running(server):
    """Every slot's consumer, named exactly as the lifespan names them."""
    tasks = [asyncio.create_task(
        app_mod._queue_consumer(server, server._config, slot_name=name),
        name=f"queue_consumer_{name}") for name in server.slots]
    try:
        yield tasks
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(BaseException):
                await task


def queued(tag: str, model: str = "test-model-q4") -> dict:
    return {"kind": "request",
            "body": {"model": model, "messages": [{"role": "user", "content": tag}]},
            "event": asyncio.Event(), "response": None, "error": None,
            "error_code": None, "backend_path": "v1/chat/completions"}


async def submit(server, *items, slot="main"):
    s = server.slots[slot]
    for item in items:
        await s.queue.enqueue(item)
    s.queue_event.set()


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

STORED = {"unhealthy", "probing", "idle", "swapping"}

LEGAL = [("unhealthy", "probing"), ("unhealthy", "swapping"),
         ("probing", "idle"), ("probing", "unhealthy"),
         ("idle", "swapping"), ("idle", "unhealthy"),
         ("swapping", "idle"), ("swapping", "unhealthy")]

# Illegal in the final table too (Tasks 4 and 5 add only `failed` moves).
ILLEGAL = [("unhealthy", "idle"), ("idle", "probing"), ("probing", "swapping"),
           ("swapping", "probing"), ("idle", "serving"), ("unhealthy", "serving"),
           ("idle", "bogus")]


def test_a_slot_starts_unhealthy():
    slot = make_slot()
    d = slot.to_status_dict()
    assert (slot.state, slot.healthy) == ("unhealthy", False)
    assert (d["state"], d["healthy"], d["swapping"]) == ("unhealthy", False, False)


def test_the_table_is_over_stored_states_and_serving_is_not_one():
    table = slots_mod.TRANSITIONS
    assert STORED <= set(table)
    for frm, targets in table.items():
        assert isinstance(targets, frozenset)
        assert targets <= set(table), f"{frm} -> {targets - set(table)} leads nowhere"
    assert "serving" not in table
    assert not any("serving" in targets for targets in table.values())
    assert issubclass(IllegalTransition, ValueError)


@pytest.mark.parametrize("frm, to", LEGAL, ids=[f"{a}->{b}" for a, b in LEGAL])
def test_every_task_2_move_is_allowed(frm, to):
    slot = put(make_slot(), frm)
    slot._set_state(to)
    assert slot.state == to
    assert to in slots_mod.TRANSITIONS[frm]


@pytest.mark.parametrize("frm, to", ILLEGAL, ids=[f"{a}->{b}" for a, b in ILLEGAL])
def test_a_move_not_in_the_table_raises_and_changes_nothing(frm, to, caplog):
    caplog.set_level(logging.INFO, logger="manager.slots")
    slot = put(make_slot(), frm)
    caplog.clear()
    with pytest.raises(IllegalTransition):
        slot._set_state(to)
    assert slot.state == frm
    assert slot_lines(caplog) == []


@pytest.mark.parametrize("state", sorted(STORED))
def test_setting_the_current_state_again_is_a_no_op_that_logs_nothing(state, caplog):
    caplog.set_level(logging.INFO, logger="manager.slots")
    slot = put(make_slot(), state)
    caplog.clear()
    slot._set_state(state)
    assert slot.state == state
    assert slot_lines(caplog) == []


def test_each_real_move_logs_one_info_line(caplog):
    caplog.set_level(logging.INFO, logger="manager.slots")
    slot = make_slot("egpu")
    slot._set_state("probing")
    slot._set_state("idle")
    slot._set_state("idle")
    slot._set_state("swapping")
    records = [r for r in caplog.records if r.name == "manager.slots"]
    assert [r.getMessage() for r in records] == [
        "slot=egpu unhealthy -> probing", "slot=egpu probing -> idle",
        "slot=egpu idle -> swapping"]
    assert {r.levelno for r in records} == {logging.INFO}


@pytest.mark.parametrize("state, in_flight, shown, healthy, swapping", [
    ("unhealthy", False, "unhealthy", False, False),
    ("unhealthy", True, "unhealthy", False, False),
    ("probing", False, "probing", False, False),
    ("idle", False, "idle", True, False),
    ("idle", True, "serving", True, False),
    ("swapping", False, "swapping", False, True),
    ("swapping", True, "swapping", False, True),
])
def test_healthy_and_swapping_derive_from_the_state(state, in_flight, shown, healthy, swapping):
    slot = put(make_slot(), state)
    if in_flight:
        slot.begin_request("r1")
    d = slot.to_status_dict()
    assert (slot.state, slot.healthy) == (shown, healthy)
    assert (d["state"], d["healthy"], d["swapping"]) == (shown, healthy, swapping)


def test_the_test_only_healthy_setter_bypasses_the_table(caplog):
    caplog.set_level(logging.INFO, logger="manager.slots")
    slot = make_slot()
    slot.healthy = True                    # unhealthy -> idle: not in the table
    assert slot.state == "idle"
    put(slot, "swapping")
    slot.healthy = False
    assert slot.state == "unhealthy"
    caplog.clear()
    slot.healthy = True
    assert slot_lines(caplog) == []


def test_no_code_in_manager_assigns_healthy():
    sources = {p.name: p.read_text() for p in MANAGER_DIR.glob("*.py")}
    assert {"app.py", "slots.py"} <= set(sources)
    offenders = [f"{name}:{n}" for name, text in sources.items()
                 for n, line in enumerate(text.splitlines(), 1)
                 if re.search(r"\.healthy\s*=(?!=)", line)
                 or re.match(r"\s*healthy\s*:\s*bool\b", line)]
    assert offenders == []


# ---------------------------------------------------------------------------
# The writers: probe, mark_swapped, mark_unhealthy, a swap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("ok", [True, False], ids=["healthy", "failed"])
async def test_a_probe_of_an_unhealthy_slot_passes_through_probing(ok, caplog):
    caplog.set_level(logging.INFO, logger="manager.slots")
    slot = make_slot()
    client = FakeProbeClient(slot, models_response("foo-Q4") if ok
                             else httpx.Response(500, json={}))
    await slot.probe(client)
    assert client.state_during == "probing"
    end = "idle" if ok else "unhealthy"
    assert slot.state == end
    assert slot_lines(caplog) == ["slot=main unhealthy -> probing", f"slot=main probing -> {end}"]
    if ok:
        assert slot.loaded_model == "foo-Q4"


@pytest.mark.asyncio
async def test_a_probe_of_an_idle_slot_stays_idle_while_it_runs(caplog):
    caplog.set_level(logging.INFO, logger="manager.slots")
    slot = put(make_slot(), "idle")
    good = FakeProbeClient(slot, models_response("foo-Q4"))
    await slot.probe(good)
    assert (good.state_during, slot.state) == ("idle", "idle")
    assert slot_lines(caplog) == []

    bad = FakeProbeClient(slot, exc=httpx.ConnectError("refused"))
    await slot.probe(bad)
    assert (bad.state_during, slot.state) == ("idle", "unhealthy")
    assert slot_lines(caplog) == ["slot=main idle -> unhealthy"]


def test_mark_swapped_ends_a_swap_and_is_a_bug_outside_one():
    slot = put(make_slot(), "swapping")
    slot.mark_swapped("foo-Q4")
    assert (slot.state, slot.loaded_model) == ("idle", "foo-Q4")
    with pytest.raises(IllegalTransition):
        put(make_slot(), "unhealthy").mark_swapped("foo-Q4")


@pytest.mark.parametrize("state", ["idle", "probing", "swapping", "unhealthy"])
def test_mark_unhealthy_moves_any_state_to_unhealthy(state):
    slot = put(make_slot(), state)
    slot.mark_unhealthy()
    assert slot.state == "unhealthy"


@pytest.mark.asyncio
@pytest.mark.parametrize("ok", [True, False], ids=["ok", "failed"])
async def test_a_swap_is_the_swapping_state_and_not_healthy_while_it_runs(test_config, ok, caplog):
    """Behaviour change (phase 5): `healthy` used to keep its pre-swap value
    during a swap; now it is false, since the state is `swapping`."""
    caplog.set_level(logging.INFO, logger="manager.slots")
    server = app_mod.ServerState(test_config)
    main = put(server.slots["main"], "idle")
    main.loaded_model = "test-model-q4"
    during: list[dict] = []

    async def swap_to(model_name, env_text):
        during.append(main.to_status_dict())
        return ok
    main.swapper.swap_to = swap_to

    assert await server.ensure_model_on_slot("main", "test-model-q8") is ok
    (d,) = during
    assert (d["state"], d["healthy"], d["swapping"]) == ("swapping", False, True)
    end = "idle" if ok else "unhealthy"
    assert main.state == end
    assert slot_lines(caplog) == ["slot=main idle -> swapping", f"slot=main swapping -> {end}"]


# ---------------------------------------------------------------------------
# refresh: ask the consumer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refresh_asks_the_consumer_and_returns_true_once_its_probe_has_run(
        server, backends, writes):
    main = server.slots["main"]
    async with running(server):
        ok = await asyncio.wait_for(main.refresh(2.0), 3)
    assert ok is True
    assert backends.probes == ["main"]
    assert [w for w in writes if w[0] == "probe"] == [("probe", "main", "queue_consumer_main")]
    assert (main.state, main.loaded_model) == ("idle", "test-model-q4")


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", ["serving", "swapping"])
async def test_refresh_returns_false_at_once_and_asks_nothing_of_a_busy_slot(
        server, backends, busy):
    main = put(server.slots["main"], "idle" if busy == "serving" else "swapping")
    if busy == "serving":
        main.begin_request("r1")
    assert main.state == busy
    async with running(server):
        await asyncio.sleep(0.02)
        t0 = time.monotonic()
        ok = await main.refresh(2.0)
        elapsed = time.monotonic() - t0
        await asyncio.sleep(0.05)
        assert (ok, backends.probes) == (False, [])
        assert elapsed < 0.1
        if busy == "serving":
            main.end_request("r1")       # no longer busy: nothing was left asked
            await asyncio.sleep(0.05)
            assert backends.probes == []


@pytest.mark.asyncio
async def test_callers_refreshing_at_the_same_time_share_one_probe(server, backends):
    main = server.slots["main"]
    backends.probe_gate = asyncio.Event()
    async with running(server):
        callers = [asyncio.create_task(main.refresh(2.0)) for _ in range(3)]
        await until(lambda: backends.probes == ["main"], what="the probe")
        await asyncio.sleep(0.05)
        assert backends.probes == ["main"]
        backends.probe_gate.set()
        results = await asyncio.wait_for(asyncio.gather(*callers), 3)
    assert results == [True, True, True]
    assert backends.probes == ["main"]


@pytest.mark.asyncio
async def test_refresh_gives_up_after_its_timeout_when_no_consumer_answers(server):
    main = server.slots["main"]
    t0 = time.monotonic()
    ok = await asyncio.wait_for(main.refresh(0.1), 3)
    elapsed = time.monotonic() - t0
    assert ok is False
    assert 0.09 <= elapsed < 1.0


@pytest.mark.asyncio
async def test_one_caller_timing_out_does_not_cancel_the_probe_another_shares(server, backends):
    main = server.slots["main"]
    backends.probe_gate = asyncio.Event()
    async with running(server):
        short = asyncio.create_task(main.refresh(0.1))
        long = asyncio.create_task(main.refresh(2.0))
        assert await asyncio.wait_for(short, 3) is False
        backends.probe_gate.set()
        assert await asyncio.wait_for(long, 3) is True
        await until(lambda: main.state == "idle", what="the probe to land")
    assert backends.probes == ["main"]


# ---------------------------------------------------------------------------
# /status, a 5xx, startup
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def app_client(test_config, backends):
    app = app_mod.create_app(test_config)
    async with REAL_ASYNC_CLIENT(transport=httpx.ASGITransport(app=app),
                                 base_url="http://manager") as c:
        yield app, c


@pytest.mark.asyncio
async def test_status_answers_at_once_and_does_not_probe_a_busy_slot(app_client, backends):
    """Review Focus 4: /status polled while a long request runs."""
    app, client = app_client
    server = app.state.server
    main = put(server.slots["main"], "idle")
    main.loaded_model = "test-model-q4"
    backends.chat_gate = asyncio.Event()
    async with running(server):
        chat = asyncio.create_task(client.post(
            "/v1/chat/completions",
            json={"model": "test-model-q4", "messages": [{"role": "user", "content": "long"}]}))
        await until(lambda: main.state == "serving", what="main serving")
        backends.order.clear()
        t0 = time.monotonic()
        r = await asyncio.wait_for(client.get("/status"), 3)
        elapsed = time.monotonic() - t0
        assert "main" not in backends.probes
        backends.chat_gate.set()
        assert (await asyncio.wait_for(chat, 3)).status_code == 200
    assert elapsed < 1.0
    s = r.json()["slots"]["main"]
    assert (s["state"], s["healthy"], s["in_flight"]) == ("serving", True, 1)


@pytest.mark.asyncio
async def test_status_probes_an_idle_or_unhealthy_slot_through_its_consumer(
        app_client, backends, writes):
    app, client = app_client
    server = app.state.server
    backends.models["egpu"] = "test-model-q8"   # came up after our last look
    async with running(server):
        writes.clear()
        r = await asyncio.wait_for(client.get("/status"), 3)
    egpu = r.json()["slots"]["egpu"]
    assert (egpu["loaded_model"], egpu["healthy"], egpu["state"]) == ("test-model-q8", True, "idle")
    probes = {(slot, task) for name, slot, task in writes if name == "probe"}
    assert probes == {("main", "queue_consumer_main"), ("egpu", "queue_consumer_egpu")}


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["backend_5xx", "connect_error"])
async def test_after_a_fault_the_consumer_probes_the_slot_before_its_next_item(
        server, backends, writes, fault, caplog):
    caplog.set_level(logging.INFO, logger="manager.slots")
    main = put(server.slots["main"], "idle")
    main.loaded_model = "test-model-q4"
    if fault == "backend_5xx":
        backends.chat_status = 500
    else:
        backends.chat_exc = httpx.ConnectError
    writes.clear()
    first, second = queued("a"), queued("b")
    async with running(server):
        await submit(server, first)
        await asyncio.wait_for(first["event"].wait(), 3)
        # The consumer probes on its own, with no next item to prompt it (a
        # next item's readiness check would probe an unhealthy slot anyway).
        await until(lambda: backends.probes == ["main"] and main.state == "idle",
                    what="the consumer's own probe after the fault")
        backends.chat_status, backends.chat_exc = 200, None
        await submit(server, second)
        await asyncio.wait_for(second["event"].wait(), 3)
    assert backends.order == ["chat:main:a", "probe:main", "chat:main:b"]
    assert second["response"].status_code == 200
    assert "slot=main idle -> unhealthy" in slot_lines(caplog)
    assert main.state == "idle"
    assert {task for name, slot, task in writes if slot == "main"} == {"queue_consumer_main"}
    assert not hasattr(app_mod, "_reprobe_for")


@pytest.mark.asyncio
async def test_startup_probes_each_slot_on_its_consumer_then_logs_readiness(
        test_config, backends, writes, caplog):
    caplog.set_level(logging.INFO, logger="manager.app")
    app = app_mod.create_app(test_config)
    server = app.state.server
    async with app.router.lifespan_context(app):
        main, egpu = server.slots["main"], server.slots["egpu"]
        assert (main.state, main.loaded_model) == ("idle", "test-model-q4")
        assert egpu.state == "unhealthy"
        messages = [r.getMessage() for r in caplog.records if r.name == "manager.app"]
        assert "slot=main ready with model test-model-q4" in messages
        assert "slot=egpu not reachable at startup" in messages
    probes = {(slot, task) for name, slot, task in writes if name == "probe"}
    assert probes == {("main", "queue_consumer_main"), ("egpu", "queue_consumer_egpu")}


# ---------------------------------------------------------------------------
# The invariant: one writer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_every_state_write_runs_on_the_slots_own_consumer(test_config, backends, writes):
    """Startup, /status, routing (incl. the stale-view reprobe), a 5xx and a
    swap: every probe, _set_state, mark_swapped and mark_unhealthy runs on
    queue_consumer_<slot>."""
    app = app_mod.create_app(test_config)
    server = app.state.server
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app), \
            REAL_ASYNC_CLIENT(transport=transport, base_url="http://manager") as client:
        assert (await client.get("/status")).status_code == 200

        r = await client.post("/v1/chat/completions",
                              json={"model": "test-model-q4", "messages": MSGS})
        assert r.status_code == 200

        backends.models["egpu"] = "test-model-q8"     # our view of egpu is stale
        r = await client.post("/v1/chat/completions",
                              json={"model": "test-model-q8", "messages": MSGS})
        assert r.status_code == 200
        assert "chat:egpu:hi" in backends.order

        backends.chat_status = 500
        r = await client.post("/v1/chat/completions",
                              json={"model": "test-model-q4", "messages": MSGS})
        assert r.status_code == 500
        backends.chat_status = 200
        await until(lambda: server.slots["main"].state == "idle", what="main reprobed")

        r = await client.post("/swap", json={"model": "test-model-q8", "target": "main"},
                              headers={"X-Request-Id": "swap-1"})
        assert r.status_code == 202
        await until(lambda: (server.slots["main"].last_swap or {}).get("id") == "swap-1",
                    what="the swap")
        assert server.slots["main"].last_swap["outcome"] == "ok"

    assert {name for name, _, _ in writes} >= {
        "probe", "_set_state", "mark_swapped", "mark_unhealthy"}
    stray = [w for w in writes if w[2] != f"queue_consumer_{w[1]}"]
    assert stray == []
