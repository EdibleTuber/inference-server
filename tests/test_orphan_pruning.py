"""Collections that are no longer configured are pruned after startup indexing (audit I2).

Live state found 2026-10-28: an orphaned collection `notes` over the same
directory as `vault` held 695 vault files' only rows. Round 3 scopes lookups
so the vault pass moves those rows to `vault`, then prunes collections absent
from collections.json. Pruning waits for startup indexing (move first, delete
after), and never runs when the config is unreadable, missing or empty.
"""
import asyncio
import dataclasses
import hashlib
import logging
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from manager.collections import make_doc_id
from manager.vectordb import VectorDB

SKILL_ID = "Security/Recon/SKILL"


def _gated_embeddings(gate: threading.Event, cap: float):
    async def embed(text):
        deadline = time.monotonic() + cap
        while not gate.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        return [0.1] * 768

    instance = AsyncMock()
    instance.embed_text = AsyncMock(side_effect=embed)
    instance.embed_batch = AsyncMock(return_value=[[0.1] * 768])
    instance.close = AsyncMock()
    return instance


def _seed_orphan(db_path: str, skills_dir: str) -> None:
    """An orphaned `notes` collection over the skills directory: one row for a
    file that still exists (with its CURRENT hash, as in the live database, so
    an unscoped lookup reports it unchanged) and one for a file that is gone."""
    skill = (Path(skills_dir) / "Security" / "Recon" / "SKILL.md").resolve()
    db = VectorDB(db_path)
    db.init_schema()
    db.upsert_collection("notes", skills_dir, "skill")
    db.upsert_document(make_doc_id(str(skill), skills_dir), "notes", "Recon", {},
                       "Recon skill", skill.read_text(), str(skill),
                       hashlib.sha256(skill.read_bytes()).hexdigest(), [0.1] * 768)
    db.upsert_document("gone/doc", "notes", "Gone", {}, "gone", "gone",
                       "/nonexistent/gone.md", "h", [0.1] * 768)
    db.close()


@pytest.fixture
def no_probe(monkeypatch):
    from manager.slots import SlotState

    async def _no_probe(self, probe_client):
        return None

    monkeypatch.setattr(SlotState, "probe", _no_probe)


def _ids(client) -> list[str]:
    return [c["id"] for c in client.get("/collections").json()["collections"]]


def _failing_embeddings():
    instance = AsyncMock()
    instance.embed_text = AsyncMock(side_effect=RuntimeError("embeddings server down"))
    instance.embed_batch = AsyncMock(side_effect=RuntimeError("embeddings server down"))
    instance.close = AsyncMock()
    return instance


def _start(config, gate: threading.Event, cap: float = 3.0, embeddings=None):
    patcher = patch("manager.app.EmbeddingsClient")
    client_cls = patcher.start()
    client_cls.return_value = embeddings or _gated_embeddings(gate, cap)
    from manager.app import create_app
    return create_app(config), patcher


def test_orphaned_collection_is_pruned_after_its_documents_move(
        collection_config, skills_dir, no_probe):
    _seed_orphan(collection_config.skills_db_path, skills_dir)
    gate = threading.Event()
    app, patcher = _start(collection_config, gate)
    try:
        with TestClient(app) as client:
            assert "notes" in _ids(client), "pruned before startup indexing finished"
            gate.set()
            client.portal.call(app.state.server.reindex_registry.wait_idle)
            deadline = time.monotonic() + 3
            while "notes" in _ids(client) and time.monotonic() < deadline:
                time.sleep(0.05)
            final_ids = _ids(client)
            skill_doc = client.get(f"/collections/skills/docs/{SKILL_ID}")
    finally:
        patcher.stop()

    assert final_ids == ["skills"]
    assert skill_doc.status_code == 200, "the moved document should now be in skills"
    db = VectorDB(collection_config.skills_db_path)
    try:
        assert db._conn.execute(
            "SELECT COUNT(*) FROM documents WHERE collection = 'notes'").fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM documents_vec WHERE id = 'gone/doc'").fetchone()[0] == 0
    finally:
        db.close()


@pytest.mark.parametrize("config_text", ["{ not json", "[]"])
def test_an_unreadable_or_empty_config_prunes_nothing(
        collection_config, skills_dir, no_probe, config_text):
    _seed_orphan(collection_config.skills_db_path, skills_dir)
    Path(collection_config.collections_config).write_text(config_text)
    gate = threading.Event()
    gate.set()
    app, patcher = _start(collection_config, gate)
    try:
        with TestClient(app) as client:
            client.portal.call(app.state.server.reindex_registry.wait_idle)
            time.sleep(0.3)
            ids = _ids(client)
    finally:
        patcher.stop()

    assert "notes" in ids


def test_a_missing_config_prunes_nothing(collection_config, skills_dir, no_probe, tmp_path):
    _seed_orphan(collection_config.skills_db_path, skills_dir)
    config = dataclasses.replace(collection_config,
                                 collections_config=str(tmp_path / "absent.json"))
    gate = threading.Event()
    gate.set()
    app, patcher = _start(config, gate)
    try:
        with TestClient(app) as client:
            client.portal.call(app.state.server.reindex_registry.wait_idle)
            time.sleep(0.3)
            ids = _ids(client)
    finally:
        patcher.stop()

    assert "notes" in ids


def test_shutdown_with_a_pending_prune_is_prompt(collection_config, skills_dir, no_probe):
    _seed_orphan(collection_config.skills_db_path, skills_dir)
    gate = threading.Event()            # never opened: indexing, and so the prune, stay pending
    app, patcher = _start(collection_config, gate, cap=5.0)
    try:
        with TestClient(app) as client:
            assert "notes" in _ids(client)
            t0 = time.monotonic()
        exited_after = time.monotonic() - t0
    finally:
        patcher.stop()
        gate.set()

    assert exited_after < 2.0


def _app_records(caplog, level: int, needle: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records
            if r.name == "manager.app" and r.levelno >= level
            and needle in r.getMessage().lower()]


def test_a_failed_prune_is_logged(collection_config, skills_dir, no_probe, caplog, monkeypatch):
    """A prune that raises must say so: unlogged, its exception sits on the
    task until shutdown, where gather(return_exceptions=True) discards it."""
    _seed_orphan(collection_config.skills_db_path, skills_dir)

    def locked(self, collection):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(VectorDB, "delete_collection", locked)
    gate = threading.Event()
    gate.set()
    app, patcher = _start(collection_config, gate)
    try:
        with caplog.at_level(logging.INFO, logger="manager.app"):
            with TestClient(app) as client:
                client.portal.call(app.state.server.reindex_registry.wait_idle)
                deadline = time.monotonic() + 3
                while (not _app_records(caplog, logging.ERROR, "prune")
                       and time.monotonic() < deadline):
                    time.sleep(0.05)
                logged = _app_records(caplog, logging.ERROR, "prune")
    finally:
        patcher.stop()

    assert logged, "the failed prune left no ERROR record in manager.app"
    assert logged[0].exc_info is not None
    assert "database is locked" in str(logged[0].exc_info[1])


@pytest.mark.parametrize("failure", ["document_errors", "job_error"])
def test_a_failed_startup_index_job_prunes_nothing(
        collection_config, skills_dir, no_probe, caplog, failure):
    """wait_idle waits for jobs to finish, not to succeed. If a configured
    collection's startup pass failed, its documents may still sit only under
    the orphan (the embeddings server down: 695 vault files), so the prune
    must be skipped, and must say why, naming the collection."""
    _seed_orphan(collection_config.skills_db_path, skills_dir)
    gate = threading.Event()
    gate.set()
    if failure == "document_errors":     # every document fails: job done, errors > 0
        app, patcher = _start(collection_config, gate, embeddings=_failing_embeddings())
        job_patch = None
    else:                                # the pass itself raises: job status "error"
        app, patcher = _start(collection_config, gate)
        job_patch = patch("manager.app.index_collection",
                          AsyncMock(side_effect=RuntimeError("walk failed")))
        job_patch.start()
    try:
        with caplog.at_level(logging.INFO, logger="manager.app"):
            with TestClient(app) as client:
                client.portal.call(app.state.server.reindex_registry.wait_idle)
                time.sleep(0.3)
                ids = _ids(client)
                job = client.get("/collections/skills/reindex/status").json()
    finally:
        if job_patch is not None:
            job_patch.stop()
        patcher.stop()

    if failure == "document_errors":     # the setup produced the failure it claims
        assert job["status"] == "done" and job["stats"]["errors"] > 0, job
    else:
        assert job["status"] == "error", job
    assert "notes" in ids
    assert _app_records(caplog, logging.WARNING, "skills"), \
        "the skipped prune should log a WARNING naming the failed collection"
