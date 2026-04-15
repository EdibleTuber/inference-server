"""Unit tests for the reindex job registry."""
import pytest
from datetime import datetime, timezone


def test_reindex_job_defaults():
    from manager.reindex_jobs import ReindexJob
    job = ReindexJob(
        job_id="j1",
        collection_id="notes",
        paths=None,
    )
    assert job.status == "queued"
    assert job.stats == {"new": 0, "updated": 0, "removed": 0, "unchanged": 0}
    assert job.error is None
    assert job.finished_at is None
    assert isinstance(job.started_at, datetime)
    assert job.started_at.tzinfo == timezone.utc


def test_reindex_job_to_dict_shape():
    from manager.reindex_jobs import ReindexJob
    job = ReindexJob(
        job_id="j1",
        collection_id="notes",
        paths=["a.md", "b.md"],
    )
    d = job.to_dict()
    assert d["job_id"] == "j1"
    assert d["collection_id"] == "notes"
    assert d["status"] == "queued"
    assert d["paths"] == ["a.md", "b.md"]
    assert d["stats"] == {"new": 0, "updated": 0, "removed": 0, "unchanged": 0}
    assert "started_at" in d and isinstance(d["started_at"], str)
    assert d["finished_at"] is None
    assert d["error"] is None


def test_registry_get_missing_returns_none():
    from manager.reindex_jobs import ReindexRegistry
    reg = ReindexRegistry()
    assert reg.get("no-such-id") is None


def test_registry_get_current_missing_returns_none():
    from manager.reindex_jobs import ReindexRegistry
    reg = ReindexRegistry()
    assert reg.get_current("notes") is None


@pytest.mark.asyncio
async def test_registry_start_runs_job_to_completion():
    """start() spawns a task; the job transitions queued -> running -> done."""
    import asyncio
    from manager.reindex_jobs import ReindexRegistry

    called_with = {}

    async def fake_indexer(paths):
        called_with["paths"] = paths
        await asyncio.sleep(0)  # yield
        return {"new": 2, "updated": 1, "removed": 0, "unchanged": 5}

    reg = ReindexRegistry()
    job = await reg.start("notes", fake_indexer, paths=["a.md", "b.md"])
    assert job.status in ("queued", "running")
    assert job.paths == ["a.md", "b.md"]

    # Wait for job completion by polling (max 2s)
    for _ in range(200):
        fresh = reg.get(job.job_id)
        if fresh.status == "done":
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail(f"job did not finish; last status={fresh.status}")

    assert fresh.stats == {"new": 2, "updated": 1, "removed": 0, "unchanged": 5}
    assert fresh.finished_at is not None
    assert fresh.error is None
    assert called_with["paths"] == ["a.md", "b.md"]


@pytest.mark.asyncio
async def test_registry_start_captures_exception():
    """If the indexer raises, status is 'error' and error field is set."""
    import asyncio
    from manager.reindex_jobs import ReindexRegistry

    async def exploding_indexer(paths):
        raise RuntimeError("boom")

    reg = ReindexRegistry()
    job = await reg.start("notes", exploding_indexer, paths=None)

    for _ in range(200):
        fresh = reg.get(job.job_id)
        if fresh.status in ("done", "error"):
            break
        await asyncio.sleep(0.01)

    assert fresh.status == "error"
    assert "boom" in (fresh.error or "")
    assert fresh.finished_at is not None


@pytest.mark.asyncio
async def test_registry_start_returns_existing_job_when_running():
    """A second start() for the same collection returns the in-flight job."""
    import asyncio
    from manager.reindex_jobs import ReindexRegistry

    gate = asyncio.Event()

    async def slow_indexer(paths):
        await gate.wait()
        return {"new": 0, "updated": 0, "removed": 0, "unchanged": 0}

    reg = ReindexRegistry()
    first = await reg.start("notes", slow_indexer, paths=None)
    # Second call while first is still running
    second = await reg.start("notes", slow_indexer, paths=["x.md"])
    assert second.job_id == first.job_id, "should return in-flight job, not start a new one"

    gate.set()
    for _ in range(200):
        fresh = reg.get(first.job_id)
        if fresh.status == "done":
            break
        await asyncio.sleep(0.01)
    assert fresh.status == "done"


@pytest.mark.asyncio
async def test_registry_start_after_completion_starts_fresh():
    """Once a job finishes, start() for the same collection spins a new one."""
    import asyncio
    from manager.reindex_jobs import ReindexRegistry

    async def quick_indexer(paths):
        return {"new": 0, "updated": 0, "removed": 0, "unchanged": 0}

    reg = ReindexRegistry()
    first = await reg.start("notes", quick_indexer, paths=None)
    for _ in range(200):
        if reg.get(first.job_id).status == "done":
            break
        await asyncio.sleep(0.01)

    second = await reg.start("notes", quick_indexer, paths=None)
    assert second.job_id != first.job_id
