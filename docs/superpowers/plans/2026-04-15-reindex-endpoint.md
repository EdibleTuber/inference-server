# Reindex-on-Demand Endpoint Implementation Plan (server side)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `POST /collections/{id}/reindex` endpoint (plus two read endpoints) so callers can trigger an incremental reindex without restarting the server. Supports scoped reindex (only specific paths) and full rescan.

**Architecture:** Extend the existing `index_collection` pipeline with an optional `paths` parameter. Add an in-memory job registry that tracks async reindex runs, enforces one-job-per-collection concurrency via an asyncio lock, and exposes status through two read endpoints. Jobs are ephemeral — the registry lives for the server's lifetime; jobs vanish on restart. That is acceptable because (a) restarts already trigger a full reindex, and (b) callers (PAL) poll within a single operation.

**Tech Stack:** Python 3.11+, FastAPI, asyncio, sqlite-vec (existing), pytest.

---

## File Structure

**Create:**
- `manager/reindex_jobs.py` — `ReindexJob` dataclass + `ReindexRegistry`. Responsible for job lifecycle (create, track, complete) and per-collection concurrency control. No HTTP, no DB schema.
- `tests/test_reindex_jobs.py` — unit tests for the registry.

**Modify:**
- `manager/collections.py` — add optional `paths: list[str] | None` parameter to `index_collection`. When provided, only process those paths and skip stale-deletion. When `None`, behavior is identical to today.
- `manager/app.py` — wire `ReindexRegistry` onto `ServerState`; add three endpoints: `POST /collections/{id}/reindex`, `GET /collections/{id}/reindex/{job_id}`, `GET /collections/{id}/reindex/status`.
- `tests/test_collections.py` — add coverage for the new `paths` parameter.
- `tests/test_collection_endpoints.py` — add endpoint tests (trigger, status, wrong collection, concurrent call short-circuit).

---

## Task 1: Add `paths` parameter to `index_collection`

**Files:**
- Modify: `manager/collections.py`
- Test: `tests/test_collections.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_collections.py`:

```python
import pytest
from unittest.mock import AsyncMock
from pathlib import Path


@pytest.mark.asyncio
async def test_index_collection_with_paths_only_indexes_given_files(notes_dir, tmp_path):
    """When `paths` is provided, only those files are upserted and stale-delete is skipped."""
    from manager.collections import index_collection
    from manager.vectordb import VectorDB

    # Seed the notes_dir with two files
    (Path(notes_dir) / "a.md").write_text("# A\n\nFirst note.")
    (Path(notes_dir) / "b.md").write_text("# B\n\nSecond note.")

    db_path = str(tmp_path / "idx.db")
    db = VectorDB(db_path)
    db.init_schema()
    embeddings = AsyncMock()
    embeddings.embed_text = AsyncMock(return_value=[0.0] * 768)

    # First: full scan should index both
    stats = await index_collection(
        collection_id="notes",
        source_dir=notes_dir,
        doc_type="markdown",
        db=db,
        embeddings=embeddings,
    )
    assert stats["new"] == 2
    assert stats["removed"] == 0

    # Delete one file on disk — but only ask the reindex to touch the OTHER file
    (Path(notes_dir) / "a.md").unlink()

    stats = await index_collection(
        collection_id="notes",
        source_dir=notes_dir,
        doc_type="markdown",
        db=db,
        embeddings=embeddings,
        paths=[str(Path(notes_dir) / "b.md")],
    )
    # Only b.md was considered. No stale-delete ran — a.md's row is still in the DB.
    assert stats["unchanged"] + stats["updated"] == 1
    assert stats["removed"] == 0
    existing = db.get_hash(str(Path(notes_dir) / "a.md"))
    assert existing is not None, "stale-delete should NOT run when paths is given"

    db._conn.close()


@pytest.mark.asyncio
async def test_index_collection_paths_skips_files_outside_source_dir(notes_dir, tmp_path):
    """Paths that aren't under source_dir are silently ignored (defensive)."""
    from manager.collections import index_collection
    from manager.vectordb import VectorDB

    (Path(notes_dir) / "a.md").write_text("# A")
    db = VectorDB(str(tmp_path / "idx.db"))
    db.init_schema()
    embeddings = AsyncMock()
    embeddings.embed_text = AsyncMock(return_value=[0.0] * 768)

    stats = await index_collection(
        collection_id="notes",
        source_dir=notes_dir,
        doc_type="markdown",
        db=db,
        embeddings=embeddings,
        paths=["/etc/passwd", str(Path(notes_dir) / "a.md")],
    )
    assert stats["new"] == 1  # only a.md counted; /etc/passwd rejected
    db._conn.close()


@pytest.mark.asyncio
async def test_index_collection_paths_handles_missing_files(notes_dir, tmp_path):
    """Paths that don't exist on disk are silently skipped."""
    from manager.collections import index_collection
    from manager.vectordb import VectorDB

    db = VectorDB(str(tmp_path / "idx.db"))
    db.init_schema()
    embeddings = AsyncMock()
    embeddings.embed_text = AsyncMock(return_value=[0.0] * 768)

    stats = await index_collection(
        collection_id="notes",
        source_dir=notes_dir,
        doc_type="markdown",
        db=db,
        embeddings=embeddings,
        paths=[str(Path(notes_dir) / "ghost.md")],
    )
    assert stats["new"] == 0
    assert stats["updated"] == 0
    assert stats["unchanged"] == 0
    db._conn.close()
```

The `notes_dir` fixture already exists in `tests/test_collections.py` and creates a directory; the tests above add files to it as needed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/edible/Projects/inference_server && pytest tests/test_collections.py -v -k "index_collection_with_paths or index_collection_paths"`
Expected: FAIL — `index_collection` does not accept `paths`.

- [ ] **Step 3: Extend `index_collection`**

In `manager/collections.py`, change the signature of `index_collection`:

```python
async def index_collection(
    collection_id: str,
    source_dir: str,
    doc_type: str,
    db: VectorDB,
    embeddings: EmbeddingsClient,
    paths: list[str] | None = None,
) -> dict:
```

Then restructure the body. Keep the existing full-scan path intact for when `paths is None`. Add a scoped path when `paths` is provided. Replace the current body with:

```python
    """Index a collection. When `paths` is provided, only those absolute file paths
    under source_dir are considered and stale-deletion is skipped. When `paths` is
    None, behaviour is unchanged: full rglob scan + stale-deletion.

    Returns stats dict {new, updated, removed, unchanged}.
    """
    stats = {"new": 0, "updated": 0, "removed": 0, "unchanged": 0}

    db.upsert_collection(collection_id, source_dir, doc_type)

    if paths is None:
        files = scan_collection(source_dir)
    else:
        # Scoped mode: only process paths under source_dir that exist on disk.
        source = Path(source_dir).resolve()
        files = []
        for p in paths:
            full = Path(p).resolve()
            if not full.exists():
                continue
            try:
                full.relative_to(source)
            except ValueError:
                continue  # path outside source_dir
            content = full.read_bytes()
            files.append({
                "file_path": str(full),
                "file_hash": hashlib.sha256(content).hexdigest(),
            })

    seen_paths = set()

    for file_info in files:
        file_path = file_info["file_path"]
        file_hash = file_info["file_hash"]
        seen_paths.add(file_path)

        existing_hash = db.get_hash(file_path)
        if existing_hash == file_hash:
            stats["unchanged"] += 1
            continue

        is_new = existing_hash is None

        content = Path(file_path).read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(content)

        doc_id = make_doc_id(file_path, source_dir)
        name = frontmatter.get("name", Path(file_path).stem)
        summary = extract_summary(frontmatter, body, doc_type)
        tags = derive_tags(file_path, source_dir, frontmatter)
        metadata = {"tags": tags}

        if "category" not in metadata and doc_type == "skill":
            parts = Path(file_path).relative_to(source_dir).parts
            if parts:
                metadata["category"] = parts[0]

        embedding = await embeddings.embed_text(summary)

        db.upsert_document(
            doc_id=doc_id,
            collection=collection_id,
            name=name,
            metadata=metadata,
            summary=summary,
            content=content,
            file_path=file_path,
            file_hash=file_hash,
            embedding=embedding,
        )

        if is_new:
            stats["new"] += 1
        else:
            stats["updated"] += 1

    # Stale-deletion only runs on a full scan, never in scoped mode.
    if paths is None:
        stats["removed"] = db.delete_stale(collection_id, seen_paths)

    return stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/edible/Projects/inference_server && pytest tests/test_collections.py -v`
Expected: all existing tests still pass AND the three new tests pass.

- [ ] **Step 5: Commit**

```bash
git add manager/collections.py tests/test_collections.py
git commit -m "feat: scoped paths parameter for index_collection"
```

---

## Task 2: `ReindexJob` dataclass + registry scaffolding

**Files:**
- Create: `manager/reindex_jobs.py`
- Create: `tests/test_reindex_jobs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reindex_jobs.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/edible/Projects/inference_server && pytest tests/test_reindex_jobs.py -v`
Expected: `ModuleNotFoundError: No module named 'manager.reindex_jobs'`.

- [ ] **Step 3: Create the module**

Create `manager/reindex_jobs.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/edible/Projects/inference_server && pytest tests/test_reindex_jobs.py -v`
Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add manager/reindex_jobs.py tests/test_reindex_jobs.py
git commit -m "feat: ReindexJob dataclass and registry scaffolding"
```

---

## Task 3: Registry `start()` — concurrency + async task spawning

**Files:**
- Modify: `manager/reindex_jobs.py`
- Test: `tests/test_reindex_jobs.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reindex_jobs.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/edible/Projects/inference_server && pytest tests/test_reindex_jobs.py -v`
Expected: 4 failures (`AttributeError: 'ReindexRegistry' object has no attribute 'start'`).

- [ ] **Step 3: Implement `start()`**

Append to `manager/reindex_jobs.py` (inside the `ReindexRegistry` class, after `_lock_for`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/edible/Projects/inference_server && pytest tests/test_reindex_jobs.py -v`
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add manager/reindex_jobs.py tests/test_reindex_jobs.py
git commit -m "feat: reindex registry start() with per-collection concurrency"
```

---

## Task 4: `POST /collections/{id}/reindex` endpoint

**Files:**
- Modify: `manager/app.py`
- Test: `tests/test_collection_endpoints.py`

- [ ] **Step 1: Read the existing `test_collection_endpoints.py` fixture pattern**

Before writing tests, read `tests/test_collection_endpoints.py` to identify how the existing collection endpoints (search, get_document) are tested. Mirror that setup (fixture that mounts the FastAPI app with a temp DB and a fake embeddings client). You will reuse that harness.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_collection_endpoints.py`. Reuse the existing fixture (likely named `client` or `app_client`):

```python
def test_reindex_post_rejects_unknown_collection(client):
    resp = client.post("/collections/does-not-exist/reindex", json={})
    assert resp.status_code == 404


def test_reindex_post_returns_job_id(client):
    # Assumes the fixture pre-registers a collection named "notes".
    resp = client.post("/collections/notes/reindex", json={})
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in ("queued", "running", "done")
    assert body["collection_id"] == "notes"


def test_reindex_post_accepts_paths(client):
    resp = client.post("/collections/notes/reindex", json={"paths": ["a.md", "b.md"]})
    assert resp.status_code == 202
    body = resp.json()
    assert body["paths"] == ["a.md", "b.md"]


def test_reindex_post_malformed_body_still_works(client):
    """Missing body = full scan."""
    resp = client.post("/collections/notes/reindex")
    assert resp.status_code == 202


def test_reindex_post_paths_must_be_list(client):
    resp = client.post("/collections/notes/reindex", json={"paths": "not a list"})
    assert resp.status_code == 400


def test_reindex_post_concurrent_returns_same_job(client):
    # Fire two back-to-back requests; second should reuse the first's job_id.
    first = client.post("/collections/notes/reindex", json={})
    second = client.post("/collections/notes/reindex", json={})
    assert first.status_code == 202 and second.status_code == 202
    # Don't require equality (the first might have finished by now),
    # but if the first is still running, second must match it.
    if first.json()["status"] == "running":
        assert first.json()["job_id"] == second.json()["job_id"]
```

If the existing test file's fixture does not register a "notes" collection, add one to the fixture or extend the test to register one inline. Read `tests/test_collection_endpoints.py` in full before writing — don't guess at the fixture surface.

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/edible/Projects/inference_server && pytest tests/test_collection_endpoints.py -v -k reindex`
Expected: FAIL (404 on the endpoint, or the endpoint doesn't exist).

- [ ] **Step 4: Wire the registry onto `ServerState`**

In `manager/app.py`, in the `ServerState.__init__` method, after `self.embeddings_client: EmbeddingsClient | None = None`, add:

```python
        # Reindex jobs
        from manager.reindex_jobs import ReindexRegistry
        self.reindex_registry = ReindexRegistry()
```

- [ ] **Step 5: Add the POST endpoint**

In `manager/app.py`, after the existing `get_document` endpoint (the last of the collection endpoints), add:

```python
    # ------------------------------------------------------------------
    # POST /collections/{collection_id}/reindex
    # ------------------------------------------------------------------

    @app.post("/collections/{collection_id}/reindex")
    async def trigger_reindex(collection_id: str, request: Request):
        """Trigger an incremental reindex. Optional body {paths: [...]} limits scope."""
        if server.db is None or server.embeddings_client is None:
            return JSONResponse(
                {"error": "Collection retrieval not configured"}, status_code=503
            )

        # Resolve the collection entry from the active config.
        from manager.collections import load_collections_config, index_collection
        all_collections = server.db.list_collections()
        entry = next((c for c in all_collections if c["id"] == collection_id), None)
        if entry is None:
            return JSONResponse(
                {"error": f"Collection not found: {collection_id}"}, status_code=404
            )

        paths: list[str] | None = None
        if request.headers.get("content-length", "0") != "0":
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
            raw_paths = body.get("paths")
            if raw_paths is not None:
                if not isinstance(raw_paths, list) or not all(isinstance(p, str) for p in raw_paths):
                    return JSONResponse(
                        {"error": "'paths' must be a list of strings"}, status_code=400,
                    )
                paths = list(raw_paths)

        async def _indexer(job_paths):
            return await index_collection(
                collection_id=entry["id"],
                source_dir=entry["source_dir"],
                doc_type=entry["doc_type"],
                db=server.db,
                embeddings=server.embeddings_client,
                paths=job_paths,
            )

        job = await server.reindex_registry.start(
            collection_id=collection_id,
            indexer=_indexer,
            paths=paths,
        )
        return JSONResponse(job.to_dict(), status_code=202)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /home/edible/Projects/inference_server && pytest tests/test_collection_endpoints.py -v -k reindex`
Expected: all 6 reindex tests pass.

Also run the full suite:

```bash
cd /home/edible/Projects/inference_server && pytest tests/ -q 2>&1 | tail -5
```

Expected: no regressions.

- [ ] **Step 7: Commit**

```bash
git add manager/app.py tests/test_collection_endpoints.py
git commit -m "feat: POST /collections/{id}/reindex endpoint"
```

---

## Task 5: `GET /collections/{id}/reindex/{job_id}` endpoint

**Files:**
- Modify: `manager/app.py`
- Test: `tests/test_collection_endpoints.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_collection_endpoints.py`:

```python
def test_reindex_get_by_job_id_returns_status(client):
    """GET /reindex/{job_id} returns the job state."""
    post = client.post("/collections/notes/reindex", json={})
    job_id = post.json()["job_id"]
    resp = client.get(f"/collections/notes/reindex/{job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job_id
    assert body["status"] in ("queued", "running", "done", "error")


def test_reindex_get_by_unknown_job_id_is_404(client):
    resp = client.get("/collections/notes/reindex/not-a-real-job-id")
    assert resp.status_code == 404


def test_reindex_get_wrong_collection_is_404(client):
    """A valid job_id on the wrong collection is still 404."""
    post = client.post("/collections/notes/reindex", json={})
    job_id = post.json()["job_id"]
    resp = client.get(f"/collections/skills/reindex/{job_id}")
    assert resp.status_code == 404
```

If the fixture doesn't register a "skills" collection, the third test can create one or skip the cross-collection check and instead test with a plausibly-unregistered id using a 404 on the parent collection.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/edible/Projects/inference_server && pytest tests/test_collection_endpoints.py -v -k "reindex_get"`
Expected: FAIL.

- [ ] **Step 3: Add the endpoint**

In `manager/app.py`, after the `trigger_reindex` endpoint, add:

```python
    # ------------------------------------------------------------------
    # GET /collections/{collection_id}/reindex/{job_id}
    # ------------------------------------------------------------------

    @app.get("/collections/{collection_id}/reindex/{job_id}")
    async def get_reindex_job(collection_id: str, job_id: str):
        """Return a specific reindex job's state, or 404."""
        if server.db is None:
            return JSONResponse(
                {"error": "Collection retrieval not configured"}, status_code=503
            )
        all_collections = server.db.list_collections()
        if not any(c["id"] == collection_id for c in all_collections):
            return JSONResponse(
                {"error": f"Collection not found: {collection_id}"}, status_code=404
            )
        job = server.reindex_registry.get(job_id)
        if job is None or job.collection_id != collection_id:
            return JSONResponse(
                {"error": f"Job not found: {job_id}"}, status_code=404
            )
        return job.to_dict()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/edible/Projects/inference_server && pytest tests/test_collection_endpoints.py -v -k reindex`
Expected: all reindex tests pass.

- [ ] **Step 5: Commit**

```bash
git add manager/app.py tests/test_collection_endpoints.py
git commit -m "feat: GET /collections/{id}/reindex/{job_id} endpoint"
```

---

## Task 6: `GET /collections/{id}/reindex/status` endpoint

**Files:**
- Modify: `manager/app.py`
- Test: `tests/test_collection_endpoints.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_collection_endpoints.py`:

```python
def test_reindex_status_no_jobs_returns_404(client):
    """Before any reindex is triggered, /status is 404."""
    resp = client.get("/collections/notes/reindex/status")
    assert resp.status_code == 404


def test_reindex_status_after_trigger(client):
    client.post("/collections/notes/reindex", json={})
    resp = client.get("/collections/notes/reindex/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["collection_id"] == "notes"
    assert body["status"] in ("queued", "running", "done", "error")


def test_reindex_status_unknown_collection_is_404(client):
    resp = client.get("/collections/does-not-exist/reindex/status")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/edible/Projects/inference_server && pytest tests/test_collection_endpoints.py -v -k reindex_status`
Expected: FAIL.

- [ ] **Step 3: Add the endpoint**

In `manager/app.py`, after the `get_reindex_job` endpoint, add:

```python
    # ------------------------------------------------------------------
    # GET /collections/{collection_id}/reindex/status
    # ------------------------------------------------------------------

    @app.get("/collections/{collection_id}/reindex/status")
    async def get_reindex_status(collection_id: str):
        """Return the current (or most recent) reindex job for a collection."""
        if server.db is None:
            return JSONResponse(
                {"error": "Collection retrieval not configured"}, status_code=503
            )
        all_collections = server.db.list_collections()
        if not any(c["id"] == collection_id for c in all_collections):
            return JSONResponse(
                {"error": f"Collection not found: {collection_id}"}, status_code=404
            )
        job = server.reindex_registry.get_current(collection_id)
        if job is None:
            return JSONResponse(
                {"error": "No reindex job recorded for this collection"}, status_code=404,
            )
        return job.to_dict()
```

Note on path ordering: FastAPI resolves routes in declaration order. Declaring `/reindex/{job_id}` BEFORE `/reindex/status` means a literal GET on `/reindex/status` would match the `{job_id}` route with `job_id="status"`. **Reorder so `/reindex/status` is declared BEFORE `/reindex/{job_id}` in `app.py`**, or use a regex constraint. Reordering is the minimal change: move the Task 6 endpoint block ABOVE the Task 5 endpoint block in the source file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/edible/Projects/inference_server && pytest tests/test_collection_endpoints.py -v -k reindex`
Expected: all reindex tests pass (including the Task 5 ones still).

Also run: `pytest tests/ -q 2>&1 | tail -5` — no regressions.

- [ ] **Step 5: Commit**

```bash
git add manager/app.py tests/test_collection_endpoints.py
git commit -m "feat: GET /collections/{id}/reindex/status endpoint"
```

---

## Task 7: End-to-end integration smoke test

**Files:**
- Create: `tests/test_reindex_integration.py`

- [ ] **Step 1: Write the integration test**

Create `tests/test_reindex_integration.py`:

```python
"""End-to-end: POST reindex -> poll status -> docs indexed."""
import time
from pathlib import Path

import pytest


def _wait_done(client, collection_id, job_id, timeout_s=5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.get(f"/collections/{collection_id}/reindex/{job_id}")
        if resp.status_code == 200 and resp.json()["status"] in ("done", "error"):
            return resp.json()
        time.sleep(0.05)
    raise AssertionError(f"reindex did not finish within {timeout_s}s")


def test_full_reindex_indexes_new_files(client, notes_source_dir):
    """
    After a file is written into source_dir, POSTing a full reindex causes
    it to appear in the documents table.
    """
    (Path(notes_source_dir) / "fresh.md").write_text("# Fresh\n\nBrand new note.")

    post = client.post("/collections/notes/reindex", json={})
    assert post.status_code == 202
    final = _wait_done(client, "notes", post.json()["job_id"])
    assert final["status"] == "done", final
    assert final["stats"]["new"] >= 1

    # Document is now retrievable
    resp = client.get("/collections/notes/docs/fresh")
    assert resp.status_code == 200


def test_scoped_reindex_only_indexes_given_path(client, notes_source_dir):
    """Writing two files but only scoping the reindex to one means only that one is indexed."""
    (Path(notes_source_dir) / "alpha.md").write_text("# Alpha")
    (Path(notes_source_dir) / "beta.md").write_text("# Beta")

    alpha_path = str(Path(notes_source_dir) / "alpha.md")
    post = client.post(
        "/collections/notes/reindex",
        json={"paths": [alpha_path]},
    )
    assert post.status_code == 202
    final = _wait_done(client, "notes", post.json()["job_id"])
    assert final["status"] == "done"
    assert final["stats"]["new"] == 1

    assert client.get("/collections/notes/docs/alpha").status_code == 200
    # beta.md was not in the scope, so it's not indexed yet
    assert client.get("/collections/notes/docs/beta").status_code == 404
```

This test requires a `notes_source_dir` fixture that points at the source_dir the `client` fixture registers for the "notes" collection. If that fixture does not already exist, add it to `tests/conftest.py` (or the local file) — it must return the absolute path of the on-disk source_dir the running app is configured to scan.

- [ ] **Step 2: Run the integration test**

Run: `cd /home/edible/Projects/inference_server && pytest tests/test_reindex_integration.py -v`
Expected: PASS.

- [ ] **Step 3: Run the full suite**

Run: `cd /home/edible/Projects/inference_server && pytest tests/ -q 2>&1 | tail -5`
Expected: PASS. No regressions from earlier tasks.

- [ ] **Step 4: Commit**

```bash
git add tests/test_reindex_integration.py
git commit -m "test: end-to-end reindex trigger + status polling"
```

---

## Task 8: Live smoke test against the deployed server

**Files:** none.

- [ ] **Step 1: Deploy the changes to the running server**

The user deploys to `/mnt/secondary/inference_server/` on 192.168.1.14 and restarts the systemd unit. Do not attempt this yourself — tell the user the commits are on the local branch and hand off.

- [ ] **Step 2: Trigger a full reindex against the live server**

```bash
curl -s -X POST http://192.168.1.14:11434/collections/notes/reindex -H 'Content-Type: application/json' -d '{}' | python3 -m json.tool
```

Expected: 202 response with a `job_id`.

- [ ] **Step 3: Poll status until done**

```bash
curl -s http://192.168.1.14:11434/collections/notes/reindex/status | python3 -m json.tool
```

Expected: eventually returns `status: done` with stats.

- [ ] **Step 4: Trigger a scoped reindex**

Pick a file that was recently written into `/mnt/secondary/agent-workspace/vault/`:

```bash
curl -s -X POST http://192.168.1.14:11434/collections/notes/reindex \
  -H 'Content-Type: application/json' \
  -d '{"paths": ["/mnt/secondary/agent-workspace/vault/Reverse-Engineering/Master-Guide-AI-Assisted-RE.md"]}' \
  | python3 -m json.tool
```

Expected: 202 with a job_id, quickly transitions to done, stats show 1 updated or unchanged.

---

## Notes on scope explicitly excluded from this plan

- **Auth.** The reindex endpoints inherit whatever auth (none, currently) the existing collection endpoints have. Adding auth is a separate concern.
- **Job persistence.** Jobs live in memory only. A server restart wipes the registry. This is fine because restarts trigger a fresh full scan, and PAL polls within a single operation.
- **Per-path deletion.** The scoped mode (`paths=[...]`) does NOT delete stale rows. If a caller wants to remove a document from the index, they must trigger a full reindex (which runs stale-delete) or add a separate delete endpoint. PAL currently has no delete flow, so this is deferred.
- **Rate limiting / abuse protection.** Not addressed. The per-collection lock already prevents duplicate work; that's the only protection this layer provides.
