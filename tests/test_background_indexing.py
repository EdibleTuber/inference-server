"""Startup collection indexing runs in the background (audit S3).

Observed 2026-10-27/28: indexing ran inside lifespan before uvicorn listened,
so every manager restart made inference unavailable for 32-36s.

The embeddings mock blocks on a threading.Event (capped, so pre-fix code that
indexes synchronously still finishes instead of hanging the test).
"""
import asyncio
import threading
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from manager.reindex_jobs import ReindexRegistry


def _gated_embeddings(gate: threading.Event, cap: float, record: list):
    async def embed(text):
        deadline = time.monotonic() + cap
        try:
            while not gate.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            record.append("cancelled")
            raise
        return [0.1] * 768

    instance = AsyncMock()
    instance.embed_text = AsyncMock(side_effect=embed)
    instance.embed_batch = AsyncMock(return_value=[[0.1] * 768])
    instance.close = AsyncMock()
    return instance


@pytest.fixture
def gated_app(collection_config, monkeypatch):
    from manager.slots import SlotState

    async def _no_probe(self, probe_client):
        return None

    monkeypatch.setattr(SlotState, "probe", _no_probe)   # startup probes stay off the network
    gate = threading.Event()
    record: list = []
    with patch("manager.app.EmbeddingsClient") as client_cls:
        client_cls.return_value = _gated_embeddings(gate, cap=3.0, record=record)
        from manager.app import create_app
        yield create_app(collection_config), gate, record
    gate.set()


def test_startup_does_not_wait_for_collection_indexing(gated_app):
    app, gate, _ = gated_app
    t0 = time.monotonic()
    with TestClient(app) as client:
        entered_after = time.monotonic() - t0
        status = client.get("/collections/skills/reindex/status")
        assert client.get("/health").status_code == 200
        assert status.status_code == 200, status.text
        assert status.json()["status"] in ("queued", "running")
        assert entered_after < 2.0

        gate.set()
        client.portal.call(app.state.server.reindex_registry.wait_idle)
        done = client.get("/collections/skills/reindex/status").json()

    assert done["status"] == "done"
    assert done["stats"]["new"] == 2


def test_reindex_during_startup_joins_the_startup_job(gated_app):
    app, gate, _ = gated_app
    with TestClient(app) as client:
        startup_job = client.get("/collections/skills/reindex/status")
        assert startup_job.status_code == 200, startup_job.text
        post = client.post("/collections/skills/reindex", json={})
        gate.set()
        client.portal.call(app.state.server.reindex_registry.wait_idle)

    assert post.status_code == 202
    assert post.json()["job_id"] == startup_job.json()["job_id"]


def test_shutdown_cancels_a_running_index_job(gated_app):
    app, _gate, record = gated_app
    with TestClient(app) as client:
        status = client.get("/collections/skills/reindex/status")
        assert status.status_code == 200, status.text
        job_id = status.json()["job_id"]
        t0 = time.monotonic()
    exited_after = time.monotonic() - t0

    job = app.state.server.reindex_registry.get(job_id)
    assert exited_after < 2.0
    assert "cancelled" in record
    assert (job.status, job.error) == ("error", "cancelled")
    assert job.finished_at is not None


# ---------------------------------------------------------------------------
# ReindexRegistry: task references, wait_idle, shutdown (audit A4)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wait_idle_waits_for_running_jobs():
    reg = ReindexRegistry()
    release = asyncio.Event()

    async def indexer(paths):
        await release.wait()
        return {"new": 1}

    job = await reg.start("c", indexer)
    waiter = asyncio.create_task(reg.wait_idle())
    await asyncio.sleep(0.02)
    assert not waiter.done()

    release.set()
    await asyncio.wait_for(waiter, 1)
    assert job.status == "done"


@pytest.mark.asyncio
async def test_wait_idle_with_no_jobs_returns_immediately():
    await asyncio.wait_for(ReindexRegistry().wait_idle(), 0.5)


@pytest.mark.asyncio
async def test_shutdown_cancels_running_jobs_and_marks_them():
    reg = ReindexRegistry()

    async def indexer(paths):
        await asyncio.Event().wait()

    job = await reg.start("c", indexer)
    await asyncio.sleep(0.02)
    await asyncio.wait_for(reg.shutdown(), 1)

    assert (job.status, job.error) == ("error", "cancelled")
    assert job.finished_at is not None


@pytest.mark.asyncio
async def test_a_bad_indexer_result_fails_the_job_and_does_not_block_the_next():
    """A non-dict (or non-integer) result is a failed job, not a job stuck in
    "running" -- start() returns a running job instead of starting a new one,
    so a stuck job would block every later reindex of that collection."""
    reg = ReindexRegistry()

    async def bad(paths):
        return "not a dict"

    async def good(paths):
        return {"new": 1}

    first = await reg.start("c", bad)
    await asyncio.wait_for(reg.wait_idle(), 1)
    second = await reg.start("c", good)
    await asyncio.wait_for(reg.wait_idle(), 1)

    assert first.status == "error" and "TypeError" in first.error
    assert first.finished_at is not None
    assert second.job_id != first.job_id and second.status == "done"


def test_unreadable_collections_config_does_not_stop_startup(
        collection_config, monkeypatch):
    """A typo in collections.json must not take inference down with it."""
    from pathlib import Path
    from manager.slots import SlotState

    async def _no_probe(self, probe_client):
        return None

    monkeypatch.setattr(SlotState, "probe", _no_probe)
    Path(collection_config.collections_config).write_text("{ not json")
    with patch("manager.app.EmbeddingsClient") as client_cls:
        client_cls.return_value = _gated_embeddings(threading.Event(), 0.0, [])
        from manager.app import create_app
        with TestClient(create_app(collection_config)) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/collections").status_code == 200


@pytest.mark.asyncio
async def test_job_stats_carry_the_indexers_error_count():
    reg = ReindexRegistry()

    async def indexer(paths):
        return {"new": 1, "updated": 0, "removed": 0, "unchanged": 0, "errors": 2}

    job = await reg.start("c", indexer)
    await asyncio.wait_for(reg.wait_idle(), 1)

    assert job.stats["errors"] == 2
