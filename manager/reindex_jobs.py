"""ReindexJob registry — in-memory tracking for async reindex runs.

One job per reindex call. Jobs carry their own lifecycle state
(queued -> running -> done | error). The registry holds a per-collection
asyncio.Lock and a map of currently-running job ids per collection so
a concurrent call can either wait on the existing job or start fresh
if none is running.

Jobs are ephemeral: they live only in process memory. A server restart
wipes them. This is intentional — restarts already trigger a fresh
full indexing pass, so recovering old job state has no value.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "done", "error"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ReindexJob:
    job_id: str
    collection_id: str
    paths: Optional[list[str]]  # None = full scan
    status: JobStatus = "queued"
    started_at: datetime = field(default_factory=_utcnow)
    finished_at: Optional[datetime] = None
    stats: dict = field(default_factory=lambda: {
        "new": 0, "updated": 0, "removed": 0, "unchanged": 0,
    })
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "collection_id": self.collection_id,
            "paths": list(self.paths) if self.paths is not None else None,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "stats": dict(self.stats),
            "error": self.error,
        }


class ReindexRegistry:
    """Per-process registry of reindex jobs and per-collection locks."""

    def __init__(self) -> None:
        self._jobs: dict[str, ReindexJob] = {}
        self._current: dict[str, str] = {}  # collection_id -> job_id of running job
        self._locks: dict[str, asyncio.Lock] = {}

    def get(self, job_id: str) -> Optional[ReindexJob]:
        return self._jobs.get(job_id)

    def get_current(self, collection_id: str) -> Optional[ReindexJob]:
        """Return the currently-running (or most recent) job for a collection."""
        job_id = self._current.get(collection_id)
        if job_id is None:
            return None
        return self._jobs.get(job_id)

    def _lock_for(self, collection_id: str) -> asyncio.Lock:
        lock = self._locks.get(collection_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[collection_id] = lock
        return lock

    async def start(
        self,
        collection_id: str,
        indexer,            # callable: async (paths) -> stats dict
        paths: Optional[list[str]] = None,
    ) -> ReindexJob:
        """Start a reindex job for a collection.

        If a job is already running for this collection, return it and do
        NOT kick off a new one. The caller polls via get() to observe
        completion.

        `indexer` must be an async callable accepting the `paths` argument
        and returning a stats dict with keys new/updated/removed/unchanged.
        """
        lock = self._lock_for(collection_id)
        async with lock:
            existing_id = self._current.get(collection_id)
            if existing_id is not None:
                existing = self._jobs.get(existing_id)
                if existing is not None and existing.status in ("queued", "running"):
                    logger.info(
                        "reindex already running for %s (job=%s), returning existing",
                        collection_id, existing_id,
                    )
                    return existing

            job_id = str(uuid.uuid4())
            job = ReindexJob(
                job_id=job_id,
                collection_id=collection_id,
                paths=list(paths) if paths is not None else None,
            )
            self._jobs[job_id] = job
            self._current[collection_id] = job_id

        # Spawn without holding the lock — the worker updates state via its own
        # handle and swaps status atomically.
        asyncio.create_task(self._run_job(job, indexer))
        return job

    async def _run_job(self, job: ReindexJob, indexer) -> None:
        job.status = "running"
        try:
            stats = await indexer(job.paths)
            if not isinstance(stats, dict):
                raise TypeError(f"indexer returned non-dict: {type(stats).__name__}")
            # Merge only the keys we track
            for key in ("new", "updated", "removed", "unchanged"):
                if key in stats:
                    job.stats[key] = int(stats[key])
            job.status = "done"
        except Exception as exc:
            logger.exception("reindex job %s failed", job.job_id)
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished_at = _utcnow()
