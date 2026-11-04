# tests/test_supervisor.py
"""Phase 5 Task 5: a per-slot supervisor (Q21).

The supervisor owns no state. `supervise(slot, interval)` is a ticker that
asks the slot's consumer for a probe (`request_probe`) every interval while
the slot is idle, unhealthy or failed. The consumer classifies each probe it
runs (`SlotState.probe` returns healthy, loading or failed), hands the result
to `judge`, and restarts the unit when `judge` says a restart is due -- or,
with SUPERVISOR_RESTART=0 (the default), logs `would restart` and counts it
as suppressed.

A backend answering 503 is loading, not failed, until SWAP_TIMEOUT after the
slot last left idle, was restarted, or the manager started (Review Focus 3).
"""
import asyncio
import dataclasses
import logging
import time

import httpx
import pytest

from manager import app as app_mod
from manager.config import ManagerConfig
from manager.slots import SlotState
from manager.supervisor import Supervision, judge, supervise
from tests.test_state_machine import (  # noqa: F401  (fixtures are used by name)
    FakeProbeClient, backends, make_slot, models_response, put,
    writes,
)
from tests.test_swap_queue import (  # noqa: F401  (fixtures are used by name)
    app, client, no_probe, running, until, world,
)

# Captured at import, before the autouse no_probe fixture replaces it.
REAL_PROBE = SlotState.probe

SUPERVISOR_KEYS = {"restart_enabled", "restarts", "restarts_suppressed",
                   "consecutive_failures"}


def cfg(test_config, **over):
    base = dict(swap_timeout=60, supervisor_failures=3, supervisor_backoff=30.0,
                supervisor_backoff_max=600.0, supervisor_restart=False)
    base.update(over)
    return dataclasses.replace(test_config, **base)


def run(sup, results, config, *, now, state="unhealthy", grace_started=-1e9):
    """judge each result in turn at `now`; the list of restarts-due."""
    return [judge(sup, r, stored_state=state, grace_started=grace_started,
                  config=config, now=now) for r in results]


# ---------------------------------------------------------------------------
# judge: counting, grace, backoff
# ---------------------------------------------------------------------------

def test_a_restart_is_due_after_the_configured_failures_and_not_before(test_config):
    config, sup = cfg(test_config, supervisor_failures=3), Supervision()
    assert run(sup, ["failed", "failed"], config, now=100) == [False, False]
    assert sup.consecutive_failures == 2
    assert run(sup, ["failed"], config, now=100) == [True]
    assert sup.consecutive_failures == 0, "a restart that is due resets the count"


def test_a_healthy_probe_resets_the_count(test_config):
    config, sup = cfg(test_config), Supervision()
    assert run(sup, ["failed", "failed", "healthy", "failed", "failed"],
               config, now=100) == [False] * 5
    assert sup.consecutive_failures == 2


def test_a_loading_backend_is_not_counted_as_a_failure_until_the_grace_ends(test_config):
    """Review Focus 3: llama-server answers 503 for about 60 s after any
    restart. Within SWAP_TIMEOUT of the grace starting, a 503 is loading."""
    config, sup = cfg(test_config, swap_timeout=60), Supervision()
    for now in (1000, 1020, 1040, 1059):
        assert run(sup, ["loading"] * 3, config, now=now, grace_started=1000) == [False] * 3
    assert sup.consecutive_failures == 0
    assert run(sup, ["loading"] * 3, config, now=1061, grace_started=1000) == [
        False, False, True]


def test_a_loading_probe_within_the_grace_neither_counts_nor_resets(test_config):
    config, sup = cfg(test_config, swap_timeout=60), Supervision()
    assert run(sup, ["failed", "failed", "loading", "failed"], config,
               now=1010, grace_started=1000) == [False, False, False, True]


def test_backoff_doubles_up_to_the_cap_and_a_healthy_probe_resets_it(test_config):
    config = cfg(test_config, supervisor_backoff=30.0, supervisor_backoff_max=100.0)
    sup = Supervision()
    assert run(sup, ["failed"] * 3, config, now=0) == [False, False, True]
    # The next restart may not come before 0 + 30.
    assert run(sup, ["failed"] * 4, config, now=10) == [False] * 4
    assert run(sup, ["failed"], config, now=30) == [True]
    # Then 60 (30 doubled): not at 89, at 90.
    assert run(sup, ["failed"] * 3, config, now=89) == [False] * 3
    assert run(sup, ["failed"], config, now=90) == [True]
    # Then capped at 100, not 120: due at 190, and again at 290.
    assert run(sup, ["failed"] * 3, config, now=189) == [False] * 3
    assert run(sup, ["failed"], config, now=190) == [True]
    assert run(sup, ["failed"] * 3, config, now=290) == [False, False, True]
    # A healthy probe puts the backoff back to its base: due at 390 + 30.
    assert run(sup, ["healthy"] + ["failed"] * 3, config, now=390) == [False] * 3 + [True]
    assert run(sup, ["failed"] * 3, config, now=419) == [False] * 3
    assert run(sup, ["failed"], config, now=420) == [True]


def test_a_healthy_probe_does_not_lift_the_delay_since_the_last_restart(test_config):
    """A healthy probe puts the delay back to its base, but the next
    restart still comes at least that delay after the previous one: a
    backend that comes up and dies at once is not restarted again within
    seconds."""
    config = cfg(test_config, supervisor_backoff=30.0, supervisor_backoff_max=600.0)
    sup = Supervision()
    assert run(sup, ["failed"] * 3, config, now=0) == [False, False, True]
    assert run(sup, ["healthy"] + ["failed"] * 3, config, now=5) == [False] * 4
    assert run(sup, ["failed"], config, now=30) == [True]
    # The delay was back at its base, so the next is due 30 later, not 60.
    assert run(sup, ["failed"] * 3, config, now=59) == [False] * 3
    assert run(sup, ["failed"], config, now=60) == [True]


@pytest.mark.parametrize("state", ["failed", "idle", "probing", "swapping"])
def test_only_an_unhealthy_slot_is_ever_due_a_restart(test_config, state):
    """A `failed` slot is probed but never restarted (its recovery is a
    /swap or a healthy probe); no other state but unhealthy is restarted."""
    config, sup = cfg(test_config), Supervision()
    assert run(sup, ["failed"] * 10, config, now=100, state=state) == [False] * 10
    assert sup.consecutive_failures == 10, "the failures are still counted"


# ---------------------------------------------------------------------------
# The slot: where the grace starts, what a probe reports
# ---------------------------------------------------------------------------

def test_the_grace_starts_when_the_slot_is_made():
    before = time.monotonic()
    slot = make_slot()
    assert before <= slot.grace_started <= time.monotonic()


@pytest.mark.parametrize("to", ["unhealthy", "swapping"])
def test_the_grace_restarts_when_the_slot_leaves_idle(to):
    slot = put(make_slot(), "idle")
    slot.grace_started = 0.0
    before = time.monotonic()
    slot._set_state(to)
    assert slot.grace_started >= before


def test_the_grace_does_not_restart_while_the_slot_is_probed_or_fails():
    """Each probe of an unhealthy slot moves it unhealthy -> probing ->
    unhealthy. If that restarted the grace, a 503 would never count."""
    slot = put(make_slot(), "unhealthy")
    slot.grace_started = 0.0
    slot._set_state("probing")
    slot._set_state("unhealthy")
    slot._set_state("swapping")
    slot._set_state("failed")
    assert slot.grace_started == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("response, exc, expected", [
    (models_response("foo-Q4"), None, "healthy"),
    (httpx.Response(503, json={"error": {"message": "Loading model"}}), None, "loading"),
    (httpx.Response(500), None, "failed"),
    (None, httpx.ConnectError("refused"), "failed"),
    (httpx.Response(200, json={"data": []}), None, "failed"),
    (httpx.Response(200, json={"data": [{"id": ""}]}), None, "failed"),
])
async def test_a_probe_reports_healthy_loading_or_failed(response, exc, expected):
    slot = make_slot()
    assert await REAL_PROBE(slot, FakeProbeClient(slot, response, exc)) == expected


# ---------------------------------------------------------------------------
# The ticker
# ---------------------------------------------------------------------------

async def ticks(slot, monkeypatch, seconds=0.2, interval=0.01):
    calls = []
    real = slot.request_probe
    monkeypatch.setattr(slot, "request_probe", lambda: calls.append(1) or real())
    task = asyncio.create_task(supervise(slot, interval))
    await asyncio.sleep(seconds)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return len(calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["idle", "unhealthy", "failed"])
async def test_the_supervisor_asks_for_a_probe_each_interval(monkeypatch, state):
    slot = put(make_slot(), state)
    n = await ticks(slot, monkeypatch, seconds=0.2, interval=0.01)
    assert 5 <= n <= 25, "about one ask per interval, no more"
    assert slot.state == state, "the supervisor writes nothing itself"


@pytest.mark.asyncio
async def test_the_supervisor_waits_one_interval_before_its_first_ask(monkeypatch):
    """The lifespan has just probed every slot; an ask at once would be a
    second probe for nothing."""
    slot = put(make_slot(), "idle")
    assert await ticks(slot, monkeypatch, seconds=0.05, interval=0.5) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["swapping", "serving"])
async def test_the_supervisor_never_asks_while_serving_or_swapping(monkeypatch, state):
    if state == "serving":
        slot = put(make_slot(), "idle")
        slot.begin_request("r1")
    else:
        slot = put(make_slot(), state)
    assert slot.state == state
    assert await ticks(slot, monkeypatch) == 0


# ---------------------------------------------------------------------------
# The consumer: counting, restarting, observe-only
# ---------------------------------------------------------------------------

class Script:
    """The probe's next results, oldest first; healthy once it runs out."""

    def __init__(self):
        self.results: list[str] = []
        self.ran: list[str] = []


@pytest.fixture
def script(monkeypatch):
    s = Script()

    async def probe(self, client):
        result = s.results.pop(0) if s.results else "healthy"
        s.ran.append(result)
        response, exc = {
            "healthy": (models_response(self.loaded_model or "test-model-q4"), None),
            "loading": (httpx.Response(503, json={"error": {"message": "Loading model"}}),
                        None),
            "failed": (None, httpx.ConnectError("refused")),
        }[result]
        return await REAL_PROBE(self, FakeProbeClient(self, response, exc))

    monkeypatch.setattr(SlotState, "probe", probe)
    return s


class Restarts:
    def __init__(self):
        self.calls: list[tuple[str, str, bool]] = []
        self.gate: asyncio.Event | None = None
        self.exc: BaseException | None = None


@pytest.fixture
def restarts(monkeypatch):
    r = Restarts()

    async def fake_restart(self):
        r.calls.append((self._slot.name, asyncio.current_task().get_name(),
                        self._slot.swap_lock.locked()))
        if r.gate is not None:
            await r.gate.wait()
        if r.exc is not None:
            raise r.exc

    monkeypatch.setattr("manager.swap.ModelSwapper.restart_llama_server", fake_restart)
    return r


def main_of(app):
    return app.state.server.slots["main"]


def tune(app, **over):
    """Change the running config in place (the consumer reads it per probe)."""
    config = app.state.server._config
    for key, value in over.items():
        setattr(config, key, value)
    return config


async def probes(slot, n):
    for _ in range(n):
        assert await asyncio.wait_for(slot.refresh(2.0), 3) is True


@pytest.mark.asyncio
async def test_a_dead_backend_is_restarted_after_three_failed_probes(
        app, client, script, restarts):
    tune(app, supervisor_restart=True, supervisor_failures=3)
    main = main_of(app)
    script.results = ["failed"] * 3
    async with running(app.state.server):
        await probes(main, 2)
        assert restarts.calls == []
        await probes(main, 1)
        await until(lambda: restarts.calls, what="the restart")
        status = (await client.get("/status")).json()["slots"]
    assert restarts.calls == [("main", "queue_consumer_main", True)], \
        "once, on the slot's own consumer, under its swap_lock"
    assert status["main"]["supervisor"] == {
        "restart_enabled": True, "restarts": 1, "restarts_suppressed": 0,
        "consecutive_failures": 0}


@pytest.mark.asyncio
async def test_with_restarts_off_it_logs_would_restart_and_never_restarts(
        app, client, script, restarts, caplog):
    assert app.state.server._config.supervisor_restart is False, "off by default"
    caplog.set_level(logging.WARNING)
    main = main_of(app)
    script.results = ["failed"] * 3
    async with running(app.state.server):
        await probes(main, 3)
        await until(lambda: main.supervision.restarts_suppressed, what="the suppressed restart")
        status = (await client.get("/status")).json()["slots"]["main"]["supervisor"]
    assert restarts.calls == []
    assert status == {"restart_enabled": False, "restarts": 0, "restarts_suppressed": 1,
                      "consecutive_failures": 0}
    lines = [r for r in caplog.records
             if r.levelno == logging.WARNING and "would restart" in r.getMessage()]
    assert len(lines) == 1
    assert "llama-server.service" in lines[0].getMessage()
    assert "main" in lines[0].getMessage()


@pytest.mark.asyncio
async def test_refresh_waiters_are_released_before_a_restart_runs(app, client, script, restarts):
    """A restart can take RESTART_TIMEOUT; the refresh that asked for the
    probe that made it due must not wait behind it."""
    tune(app, supervisor_restart=True)
    main = main_of(app)
    script.results = ["failed"] * 3
    restarts.gate = asyncio.Event()
    async with running(app.state.server):
        await probes(main, 2)
        started = time.monotonic()
        assert await asyncio.wait_for(main.refresh(2.0), 3) is True
        await until(lambda: restarts.calls, what="the restart")
        assert time.monotonic() - started < 1.0
        assert main.supervision.restarts == 0, "not counted until it returns"
        restarts.gate.set()
        await until(lambda: main.supervision.restarts == 1, what="the restart to finish")


@pytest.mark.asyncio
async def test_a_restart_that_raises_is_logged_and_not_counted(
        app, client, script, restarts, caplog):
    tune(app, supervisor_restart=True)
    caplog.set_level(logging.ERROR)
    main = main_of(app)
    script.results = ["failed"] * 3
    restarts.exc = RuntimeError("Failed to restart llama-server.service: denied")
    async with running(app.state.server) as consumer:
        await probes(main, 3)
        await until(lambda: restarts.calls, what="the restart")
        await probes(main, 1)       # the consumer is still serving probes
        assert not consumer.done()
    assert main.supervision.restarts == 0
    assert any("denied" in (r.exc_text or "") + r.getMessage() for r in caplog.records
               if r.levelno >= logging.ERROR)


@pytest.mark.asyncio
async def test_a_loading_backend_after_a_restart_is_not_restarted_again_within_the_grace(
        app, client, script, restarts):
    """After a restart the backend answers 503 while it loads. With no
    backoff at all, and the slot's grace long over before the restart, only
    the grace the restart starts keeps it from restarting again."""
    tune(app, supervisor_restart=True, supervisor_backoff=0.0, supervisor_backoff_max=0.0,
         swap_timeout=60)
    main = put(main_of(app), "unhealthy")
    main.grace_started = -1e9
    script.results = ["failed"] * 3 + ["loading"] * 6
    async with running(app.state.server):
        await probes(main, 3)
        await until(lambda: main.supervision.restarts == 1, what="the restart")
        await probes(main, 6)
    assert len(restarts.calls) == 1
    assert main.supervision.consecutive_failures == 0
    assert main.state == "unhealthy"


@pytest.mark.asyncio
async def test_a_failed_slot_is_probed_but_never_restarted(app, client, script, restarts):
    tune(app, supervisor_restart=True)
    main = put(main_of(app), "failed")
    script.results = ["failed"] * 5
    async with running(app.state.server):
        await probes(main, 5)
        assert (main.state, main.supervision.consecutive_failures) == ("failed", 5)
        await probes(main, 1)        # healthy
    assert restarts.calls == []
    assert (main.state, main.supervision.consecutive_failures) == ("idle", 0)


@pytest.mark.asyncio
async def test_the_ticker_drives_a_restart_through_the_consumer(app, client, script, restarts):
    """End to end: the supervisor's own probes, not refresh, count."""
    tune(app, supervisor_restart=True)
    main = main_of(app)
    script.results = ["failed"] * 3
    async with running(app.state.server):
        ticker = asyncio.create_task(supervise(main, 0.01), name="supervisor_main")
        try:
            await until(lambda: restarts.calls, what="the restart")
        finally:
            ticker.cancel()
    assert restarts.calls == [("main", "queue_consumer_main", True)]
    assert script.ran[:3] == ["failed"] * 3


# ---------------------------------------------------------------------------
# /status and the lifespan
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_status_reports_the_supervisor_for_every_slot(app, client, script):
    main = main_of(app)
    main.supervision.restarts, main.supervision.consecutive_failures = 2, 1
    async with running(app.state.server):
        slots = (await client.get("/status")).json()["slots"]
    for name, slot in slots.items():
        assert set(slot["supervisor"]) == SUPERVISOR_KEYS, name
    assert slots["main"]["supervisor"]["restarts"] == 2
    assert slots["egpu"]["supervisor"] == {
        "restart_enabled": False, "restarts": 0, "restarts_suppressed": 0,
        "consecutive_failures": 0}


@pytest.fixture
def real_probe(monkeypatch):
    """Undo the autouse no_probe stub; listed before `writes` so that it
    wraps the real probe."""
    monkeypatch.setattr(SlotState, "probe", REAL_PROBE)


@pytest.mark.asyncio
async def test_the_lifespan_runs_one_supervisor_per_slot_and_stops_them(
        test_config, backends, real_probe, writes):
    """Named supervisor_<slot>; their probes run on each slot's own
    consumer, and nothing they cause writes from any other task."""
    test_config.supervisor_probe_interval = 0.01
    app = app_mod.create_app(test_config)
    async with app.router.lifespan_context(app):
        names = {t.get_name() for t in asyncio.all_tasks()}
        assert {"supervisor_main", "supervisor_egpu"} <= names
        supervisors = [t for t in asyncio.all_tasks()
                       if t.get_name().startswith("supervisor_")]
        writes.clear()
        await until(lambda: sum(1 for w in writes if w[0] == "probe" and w[1] == "main") >= 3,
                    what="the supervisor's probes")
    assert all(t.done() for t in supervisors)
    stray = [w for w in writes if w[2] != f"queue_consumer_{w[1]}"]
    assert stray == []


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPERVISOR_ENV = ["SUPERVISOR_PROBE_INTERVAL", "SUPERVISOR_FAILURES", "SUPERVISOR_BACKOFF",
                  "SUPERVISOR_BACKOFF_MAX", "SUPERVISOR_RESTART"]


def test_the_supervisor_settings_default(monkeypatch):
    for key in SUPERVISOR_ENV:
        monkeypatch.delenv(key, raising=False)
    c = ManagerConfig.from_env()
    assert (c.supervisor_probe_interval, c.supervisor_failures, c.supervisor_backoff,
            c.supervisor_backoff_max, c.supervisor_restart) == (15.0, 3, 30.0, 600.0, False)


def test_the_supervisor_settings_are_read_from_the_env(monkeypatch):
    for key, value in zip(SUPERVISOR_ENV, ["2.5", "5", "10", "120", "1"]):
        monkeypatch.setenv(key, value)
    c = ManagerConfig.from_env()
    assert (c.supervisor_probe_interval, c.supervisor_failures, c.supervisor_backoff,
            c.supervisor_backoff_max, c.supervisor_restart) == (2.5, 5, 10.0, 120.0, True)


@pytest.mark.parametrize("key, value", [
    ("SUPERVISOR_RESTART", "yes"), ("SUPERVISOR_RESTART", "true"),
    ("SUPERVISOR_PROBE_INTERVAL", "often"), ("SUPERVISOR_FAILURES", "three")])
def test_a_bad_supervisor_setting_names_its_key(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key):
        ManagerConfig.from_env()


@pytest.mark.parametrize("key, value", [
    # supervise() sleeps the interval: 0 or a negative interval re-arms a
    # probe on every loop tick (the consumer probes back to back), and nan
    # or inf never wakes (the supervisor is silently dead). Measured
    # 2026-11-04: asyncio.sleep(-5) returns at once; sleep(nan) never does.
    ("SUPERVISOR_PROBE_INTERVAL", "0"), ("SUPERVISOR_PROBE_INTERVAL", "-1"),
    ("SUPERVISOR_PROBE_INTERVAL", "nan"), ("SUPERVISOR_PROBE_INTERVAL", "inf"),
    # judge compares now - _last_due < delay: 0, negative or nan never
    # holds a restart back (one per failed probe), inf holds it forever.
    ("SUPERVISOR_BACKOFF", "0"), ("SUPERVISOR_BACKOFF", "-30"),
    ("SUPERVISOR_BACKOFF", "nan"), ("SUPERVISOR_BACKOFF", "inf"),
    ("SUPERVISOR_BACKOFF_MAX", "0"), ("SUPERVISOR_BACKOFF_MAX", "nan"),
    ("SUPERVISOR_BACKOFF_MAX", "inf"),
    # Below the base the cap shrinks the delay after the first restart.
    ("SUPERVISOR_BACKOFF_MAX", "29.9"),
    # At least one failed probe before a restart is due.
    ("SUPERVISOR_FAILURES", "0"), ("SUPERVISOR_FAILURES", "-1"),
])
def test_an_out_of_range_supervisor_setting_names_its_key(monkeypatch, key, value):
    """Branch review (2026-11-04). Every other supervisor setting is left at
    its default, so SUPERVISOR_BACKOFF is 30 for the cap's 29.9."""
    for k in SUPERVISOR_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key):
        ManagerConfig.from_env()


@pytest.mark.parametrize("key, value", [
    ("SUPERVISOR_PROBE_INTERVAL", "0.5"), ("SUPERVISOR_FAILURES", "1"),
    ("SUPERVISOR_BACKOFF", "600"), ("SUPERVISOR_BACKOFF_MAX", "30"),
])
def test_the_edge_of_each_supervisor_range_is_accepted(monkeypatch, key, value):
    """Small legal values, and a base equal to its cap (600 and 30, each
    against the other's default)."""
    for k in SUPERVISOR_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(key, value)
    ManagerConfig.from_env()


def test_the_template_ships_with_restarts_off():
    """Turning restarts on is a deliberate one-line change in the deployed
    /etc/llama/manager.env, never something the template does."""
    from tests.test_repo_consistency import _template_env
    assert _template_env().get("SUPERVISOR_RESTART", "0") == "0"
