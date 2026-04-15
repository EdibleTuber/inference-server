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
