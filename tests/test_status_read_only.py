"""/status only reads (phase 6 Task 6, design feedback 2026-11-05).

GET /status used to ask every idle, unhealthy or failed slot's consumer to
probe before answering, and every probe the consumer runs is judged for the
supervisor. So reading /status moved supervisor restarts earlier, loaded
the backends, and -- with agent_core's /model polling /status during a swap
-- would have probed the other slot every poll. Since phase 5 the
supervisor's ticker probes those slots every SUPERVISOR_PROBE_INTERVAL, which
covers the case the probe-on-read existed for (a backend that came up after
the manager's startup probe).

/status now reports what the last probe found, and how long ago that was.
"""
import asyncio
import time

import httpx
import pytest

from manager import app as app_mod
from manager import slots as slots_mod
from manager.supervisor import supervise
from tests.test_state_machine import (  # noqa: F401  (fixtures)
    REAL_ASYNC_CLIENT,
    Backends,
    app_client,
    backends,
    make_slot,
    models_response,
    put,
    running,
    until,
    writes,
)


async def _settle():
    """Long enough for a consumer woken by a probe request to run it."""
    await asyncio.sleep(0.05)


# --- /status asks for no probe ---------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["idle", "unhealthy", "failed"])
async def test_status_asks_for_no_probe(app_client, backends, writes, state):
    """The states the old /status probed. The answer is what the slot already
    holds, even when the backend now says something else."""
    app, client = app_client
    server = app.state.server
    egpu = put(server.slots["egpu"], state)
    backends.models["egpu"] = "test-model-q8"   # came up since our last look
    async with running(server):
        writes.clear()
        backends.order.clear()
        r = await asyncio.wait_for(client.get("/status"), 3)
        await _settle()
        assert backends.probes == []
        assert [w for w in writes if w[0] == "probe"] == []
        assert all(not s.probe_pending for s in server.slots.values())
    shown = r.json()["slots"]["egpu"]
    assert shown["state"] == state
    assert shown["loaded_model"] is None, "not re-read from the backend"


@pytest.mark.asyncio
async def test_status_waits_for_no_consumer(app_client, backends):
    """With no consumer running, the old /status waited out REFRESH_TIMEOUT
    for a probe nobody would run. It now answers at once."""
    app, client = app_client
    put(app.state.server.slots["egpu"], "unhealthy")
    t0 = time.monotonic()
    r = await asyncio.wait_for(client.get("/status"), app_mod.REFRESH_TIMEOUT + 5)
    elapsed = time.monotonic() - t0
    assert r.status_code == 200
    assert elapsed < 1.0, f"/status took {elapsed:.2f}s"


class _FailingEgpu(Backends):
    """egpu's /v1/models answers 500: every probe of it counts a failure."""

    async def __call__(self, request):
        if (request.url.path == "/v1/models"
                and self.ports.get(request.url.port) == "egpu"):
            self.order.append("probe:egpu")
            return httpx.Response(500, json={"error": {"message": "boom"}})
        return await super().__call__(request)


@pytest.mark.asyncio
async def test_polling_status_does_not_move_the_supervisor_count(test_config, monkeypatch):
    b = _FailingEgpu({sc.port: sc.name for sc in test_config.slots})

    class Mocked(REAL_ASYNC_CLIENT):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(b)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(app_mod.httpx, "AsyncClient", Mocked)
    from unittest.mock import AsyncMock
    monkeypatch.setattr(app_mod, "get_gpu_info_async", AsyncMock(return_value=[]))
    app = app_mod.create_app(test_config)
    server = app.state.server
    egpu = put(server.slots["egpu"], "unhealthy")
    polls = test_config.supervisor_failures + 2
    async with REAL_ASYNC_CLIENT(transport=httpx.ASGITransport(app=app),
                                 base_url="http://manager") as client:
        async with running(server):
            for _ in range(polls):
                assert (await asyncio.wait_for(client.get("/status"), 3)).status_code == 200
                await _settle()
    assert "egpu" not in b.probes
    assert egpu.supervision.consecutive_failures == 0
    assert egpu.supervision.restarts_suppressed == 0
    assert egpu.supervision.restarts == 0


# --- the age of the last probe ---------------------------------------------------------

_KEYS_BEFORE = {
    "host", "port", "loaded_model", "healthy", "swapping", "last_swap_utc",
    "queue_depth", "queue_limit", "state", "in_flight", "in_flight_age_seconds",
    "oldest_queue_wait_seconds", "last_error", "last_swap", "supervisor", "models",
}


@pytest.mark.asyncio
async def test_status_reports_the_age_of_the_last_probe(app_client, backends):
    app, client = app_client
    server = app.state.server
    main, egpu = server.slots["main"], server.slots["egpu"]
    assert main.last_probe_at is None and egpu.last_probe_at is None

    before = (await client.get("/status")).json()["slots"]
    for name in ("main", "egpu"):
        assert set(before[name]) == _KEYS_BEFORE | {"last_probe_age_seconds"}
        assert before[name]["last_probe_age_seconds"] is None, "null before any probe"

    async with running(server):
        assert await main.refresh(3) is True     # one probe, of main only
    after = (await client.get("/status")).json()["slots"]
    age = after["main"]["last_probe_age_seconds"]
    assert isinstance(age, float) and 0.0 <= age < 5.0
    assert after["egpu"]["last_probe_age_seconds"] is None

    main.last_probe_at = time.monotonic() - 12.34
    shown = (await client.get("/status")).json()["slots"]["main"]
    assert shown["last_probe_age_seconds"] in (12.3, 12.4)   # rounded to 1 decimal


class _Clock:
    """manager.slots' view of `time`: real except monotonic(), which the
    fake backend moves on while the probe's call is in flight."""

    def __init__(self, start: float):
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


class _ProbeClient:
    """Hands SlotState.probe one answer, moving the clock during the call."""

    def __init__(self, clock: _Clock, response=None, exc=None):
        self.clock, self.response, self.exc = clock, response, exc

    async def get(self, url, timeout=None):
        self.clock.now = 200.0          # the call takes time
        if self.exc is not None:
            raise self.exc
        return self.response


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["healthy", "loading", "failed", "raised", "malformed"])
async def test_every_probe_path_stamps_the_probe_time(monkeypatch, path):
    """Stamped when the backend call has returned or failed, on every path:
    the stamp says when the slot was last *looked at*, whatever was seen."""
    clock = _Clock(100.0)
    monkeypatch.setattr(slots_mod, "time", clock)
    slot = put(make_slot("main"), "unhealthy")
    assert slot.last_probe_at is None
    client = {
        "healthy": _ProbeClient(clock, models_response("test-model-q4")),
        "loading": _ProbeClient(clock, httpx.Response(503, json={"error": {"message": "Loading"}})),
        "failed": _ProbeClient(clock, httpx.Response(500, json={})),
        "raised": _ProbeClient(clock, exc=httpx.ConnectError("refused")),
        "malformed": _ProbeClient(clock, httpx.Response(200, json=[])),
    }[path]
    await slot.probe(client)
    assert slot.last_probe_at == 200.0, "stamped after the call, not before it"


# --- the case the probe-on-read existed for ------------------------------------------------

@pytest.mark.asyncio
async def test_a_backend_that_comes_up_later_is_seen_by_the_supervisor(test_config, backends):
    """A backend still loading at startup shows as down until something
    probes it again. That something is now the supervisor's ticker, not a
    /status read. (A guard: this passed before Task 6 too.)"""
    server = app_mod.ServerState(test_config)
    egpu = server.slots["egpu"]
    backends.models["egpu"] = None               # 503: still loading
    async with running(server):
        assert await egpu.refresh(3) is True
        assert egpu.loaded_model is None and egpu.state == "unhealthy"
        ticker = asyncio.create_task(supervise(egpu, 0.01))
        try:
            backends.models["egpu"] = "test-model-q8"
            await until(lambda: egpu.state == "idle", timeout=3,
                        what="the ticker's probe to find egpu up")
        finally:
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)
    assert egpu.loaded_model == "test-model-q8"
