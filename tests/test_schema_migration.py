"""Q23: an existing (v0) skills.db is migrated once to schema v1, after a
backup, in one transaction.

LEGACY_DDL is a frozen copy of vectordb.py's v0 init_schema (as of 039b3e6):
it is history, not something to keep in sync.
"""
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import sqlite_vec
from fastapi.testclient import TestClient

from manager.vectordb import VectorDB, _serialize_vec

DIM = 768

LEGACY_DDL = """
CREATE TABLE IF NOT EXISTS collections (
    id          TEXT PRIMARY KEY,
    source_dir  TEXT NOT NULL,
    doc_type    TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    collection  TEXT NOT NULL REFERENCES collections(id),
    name        TEXT NOT NULL,
    metadata    TEXT,
    summary     TEXT NOT NULL,
    content     TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    file_hash   TEXT NOT NULL,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE VIRTUAL TABLE IF NOT EXISTS documents_vec USING vec0(
    id TEXT PRIMARY KEY,
    embedding float[768]
);
CREATE INDEX IF NOT EXISTS idx_documents_collection_file_path
    ON documents(collection, file_path);
"""


def unit(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


DOCS = [
    dict(id="README", collection="vault", file_path="/v/README.md", file_hash="h1",
         content="vault readme", vec=unit(0)),
    dict(id="notes/x", collection="vault", file_path="/v/notes/x.md", file_hash="h2",
         content="note x", vec=unit(1)),
    dict(id="PLAN_SKILL", collection="skills", file_path="/s/PLAN_SKILL.md", file_hash="h3",
         content="plan skill", vec=unit(2)),
]


def _raw(path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def make_legacy_db(path, docs) -> None:
    conn = _raw(path)
    conn.executescript(LEGACY_DDL)
    for coll in sorted({d["collection"] for d in docs}):
        conn.execute("INSERT INTO collections (id, source_dir, doc_type) VALUES (?, ?, 'markdown')",
                     (coll, f"/{coll}"))
    for d in docs:
        conn.execute(
            "INSERT INTO documents (id, collection, name, metadata, summary, content, file_path, file_hash)"
            " VALUES (?, ?, ?, '{}', 's', ?, ?, ?)",
            (d["id"], d["collection"], d["id"], d["content"], d["file_path"], d["file_hash"]))
        if d["vec"] is not None:
            conn.execute("INSERT INTO documents_vec (id, embedding) VALUES (?, ?)",
                         (d["id"], _serialize_vec(d["vec"])))
    conn.commit()
    conn.close()


def tables(path) -> set[str]:
    conn = _raw(path)
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        conn.close()


def user_version(path) -> int:
    conn = _raw(path)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def doc_count(path) -> int:
    conn = _raw(path)
    try:
        return conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    finally:
        conn.close()


@pytest.fixture
def legacy(tmp_path) -> Path:
    path = tmp_path / "skills.db"
    make_legacy_db(path, DOCS)
    return path


def test_legacy_database_migrates_every_document_and_vector(legacy):
    db = VectorDB(str(legacy))
    db.init_schema()
    try:
        for d in DOCS:
            assert db.get_document(d["collection"], d["id"])["content"] == d["content"]
            assert db.get_hash(d["collection"], d["file_path"]) == d["file_hash"]
            [hit] = db.search(d["collection"], d["vec"], limit=1)
            assert (hit["id"], hit["score"]) == (d["id"], pytest.approx(1.0))
        assert {c["id"]: c["doc_count"] for c in db.list_collections()} == {"vault": 2, "skills": 1}
    finally:
        db.close()
    assert user_version(legacy) == 1


def test_migrated_database_accepts_a_shared_id(legacy):
    db = VectorDB(str(legacy))
    db.init_schema()
    try:
        db.upsert_document("README", "skills", "R", {}, "s", "skills readme",
                           "/s/README.md", "h4", unit(3))
        assert db.get_document("vault", "README")["content"] == "vault readme"
        assert db.get_document("skills", "README")["content"] == "skills readme"
    finally:
        db.close()


def test_migration_backs_up_the_legacy_database_first(legacy):
    bak = Path(f"{legacy}.pre-v1.bak")
    db = VectorDB(str(legacy))
    db.init_schema()
    db.close()
    assert bak.exists()
    assert user_version(bak) == 0
    assert "documents_vec" in tables(bak)
    assert doc_count(bak) == len(DOCS)


def test_failed_migration_rolls_back_to_the_legacy_database(legacy, monkeypatch):
    from manager.vectordb import MigrationError
    before = tables(legacy)

    def boom(conn):
        raise MigrationError("injected")

    monkeypatch.setattr("manager.vectordb._verify_migration", boom)
    db = VectorDB(str(legacy))
    with pytest.raises(MigrationError):
        db.init_schema()
    db.close()

    assert user_version(legacy) == 0
    assert tables(legacy) == before
    assert doc_count(legacy) == len(DOCS)

    monkeypatch.undo()                     # a later start retries and succeeds
    db = VectorDB(str(legacy))
    db.init_schema()
    db.close()
    assert user_version(legacy) == 1


def test_a_document_without_a_vector_aborts_the_migration(tmp_path):
    from manager.vectordb import MigrationError
    path = tmp_path / "skills.db"
    make_legacy_db(path, DOCS[:2] + [dict(DOCS[2], vec=None)])
    before = tables(path)
    db = VectorDB(str(path))
    with pytest.raises(MigrationError):
        db.init_schema()
    db.close()
    assert user_version(path) == 0
    assert tables(path) == before


def test_migration_runs_once(legacy):
    bak = Path(f"{legacy}.pre-v1.bak")
    db = VectorDB(str(legacy))
    db.init_schema()
    db.close()
    bak.unlink()

    db = VectorDB(str(legacy))
    db.init_schema()
    db.close()

    assert not bak.exists()
    assert user_version(legacy) == 1
    assert doc_count(legacy) == len(DOCS)


def test_a_failed_migration_leaves_inference_up(collection_config, monkeypatch):
    from manager.vectordb import MigrationError
    make_legacy_db(collection_config.skills_db_path, DOCS)

    def boom(conn):
        raise MigrationError("injected")

    monkeypatch.setattr("manager.vectordb._verify_migration", boom)
    with patch("manager.retrieval.EmbeddingsClient") as emb_cls:
        emb_cls.return_value = AsyncMock()
        from manager.app import create_app
        with TestClient(create_app(collection_config)) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/collections").status_code == 503


def test_a_failed_upsert_leaves_no_half_written_document(tmp_path):
    db = VectorDB(str(tmp_path / "skills.db"))
    db.init_schema()
    try:
        db.upsert_collection("c", "/c", "markdown")
        with pytest.raises(sqlite3.Error):
            db.upsert_document("bad", "c", "B", {}, "s", "c", "/c/bad.md", "hbad",
                               [0.0] * 4)              # wrong dimension: the vector insert raises
        db.upsert_document("good", "c", "G", {}, "s", "c", "/c/good.md", "hgood", unit(0))

        assert db.get_hash("c", "/c/bad.md") is None   # retried on the next reindex
        docs = db._conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        vecs = db._conn.execute("SELECT count(*) FROM doc_vectors").fetchone()[0]
        assert (docs, vecs) == (1, 1)
    finally:
        db.close()


# Added after the Task 4 mutation run (T8, T11, T14 survived).

def _schema(path) -> tuple:
    conn = _raw(path)
    try:
        names = {(r[0], r[1]) for r in conn.execute("SELECT type, name FROM sqlite_master")}
        columns = [tuple(r) for r in conn.execute("PRAGMA table_info(documents)")]
        return names, columns
    finally:
        conn.close()


def test_a_migrated_database_has_the_same_schema_as_a_fresh_one(legacy, tmp_path):
    """The migration repeats the v1 DDL; this keeps it from drifting from
    _create_v1, and catches a legacy table left behind or the
    (collection, file_path) index missing (audit I1: 30s event-loop stall)."""
    fresh = tmp_path / "fresh.db"
    for path in (legacy, fresh):
        db = VectorDB(str(path))
        db.init_schema()
        db.close()
    assert _schema(legacy) == _schema(fresh)


def test_a_failed_migration_leaves_no_transaction_open(legacy, monkeypatch):
    """start_retrieval drops a VectorDB whose init_schema raised without
    closing it, so the rollback must not depend on close()."""
    from manager.vectordb import MigrationError

    def boom(conn):
        raise MigrationError("injected")

    monkeypatch.setattr("manager.vectordb._verify_migration", boom)
    db = VectorDB(str(legacy))
    with pytest.raises(MigrationError):
        db.init_schema()
    try:
        assert not db._conn.in_transaction
    finally:
        db.close()
