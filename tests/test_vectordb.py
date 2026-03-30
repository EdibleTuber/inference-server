# tests/test_vectordb.py
"""Tests for the SQLite-vec vector database wrapper."""
import pytest


@pytest.fixture
def db(tmp_path):
    from manager.vectordb import VectorDB
    db_path = str(tmp_path / "test.db")
    db = VectorDB(db_path)
    db.init_schema()
    return db


def test_init_schema_creates_tables(db):
    """init_schema creates collections, documents, and documents_vec tables."""
    cursor = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]
    assert "collections" in tables
    assert "documents" in tables


def test_upsert_collection(db):
    """upsert_collection inserts and updates collection rows."""
    db.upsert_collection("skills", "/home/user/skills", "skill")
    row = db._conn.execute("SELECT * FROM collections WHERE id='skills'").fetchone()
    assert row is not None
    db.upsert_collection("skills", "/new/path", "skill")
    row = db._conn.execute("SELECT source_dir FROM collections WHERE id='skills'").fetchone()
    assert row[0] == "/new/path"


def test_upsert_and_search_document(db):
    """Upsert a document with embedding, then search for it."""
    db.upsert_collection("skills", "/tmp/skills", "skill")
    embedding = [0.1] * 768
    db.upsert_document(
        doc_id="Security/Recon",
        collection="skills",
        name="Recon",
        metadata={"tags": ["security"], "category": "Security"},
        summary="Network reconnaissance and scanning",
        content="# Recon\n\nFull workflow content here.",
        file_path="/tmp/skills/Security/Recon/SKILL.md",
        file_hash="abc123",
        embedding=embedding,
    )
    results = db.search("skills", [0.1] * 768, limit=5)
    assert len(results) == 1
    assert results[0]["id"] == "Security/Recon"
    assert results[0]["name"] == "Recon"
    assert results[0]["summary"] == "Network reconnaissance and scanning"
    assert "score" in results[0]


def test_search_with_tag_filter(db):
    """Search filters by tags when provided."""
    db.upsert_collection("skills", "/tmp/skills", "skill")
    embedding = [0.1] * 768
    db.upsert_document("Security/Recon", "skills", "Recon", {"tags": ["security", "recon"]},
        "Network recon", "content", "/tmp/a.md", "a", embedding)
    db.upsert_document("Media/Art", "skills", "Art", {"tags": ["media", "art"]},
        "Image generation", "content", "/tmp/b.md", "b", embedding)
    results = db.search("skills", embedding, limit=5, tags=["security"])
    assert len(results) == 1
    assert results[0]["id"] == "Security/Recon"


def test_get_document(db):
    """get_document returns full content by ID."""
    db.upsert_collection("skills", "/tmp/skills", "skill")
    db.upsert_document("Security/Recon", "skills", "Recon", {"tags": ["security"]},
        "Network recon", "# Full markdown content", "/tmp/a.md", "a", [0.1] * 768)
    doc = db.get_document("skills", "Security/Recon")
    assert doc is not None
    assert doc["content"] == "# Full markdown content"
    assert doc["name"] == "Recon"


def test_get_document_not_found(db):
    """get_document returns None for missing documents."""
    doc = db.get_document("skills", "nonexistent")
    assert doc is None


def test_get_hash(db):
    """get_hash returns stored hash for a file path."""
    db.upsert_collection("skills", "/tmp", "skill")
    db.upsert_document("test", "skills", "Test", {}, "test", "test",
        "/tmp/test.md", "sha256abc", [0.1] * 768)
    assert db.get_hash("/tmp/test.md") == "sha256abc"
    assert db.get_hash("/tmp/nonexistent.md") is None


def test_delete_stale_removes_missing_files(db):
    """delete_stale removes documents whose file_path is not in the keep set."""
    db.upsert_collection("skills", "/tmp", "skill")
    db.upsert_document("a", "skills", "A", {}, "a", "a", "/tmp/a.md", "h1", [0.1] * 768)
    db.upsert_document("b", "skills", "B", {}, "b", "b", "/tmp/b.md", "h2", [0.1] * 768)
    removed = db.delete_stale("skills", {"/tmp/a.md"})
    assert removed == 1
    assert db.get_document("skills", "a") is not None
    assert db.get_document("skills", "b") is None


def test_list_collections(db):
    """list_collections returns all registered collections with doc counts."""
    db.upsert_collection("skills", "/tmp/skills", "skill")
    db.upsert_collection("notes", "/tmp/notes", "markdown")
    db.upsert_document("s1", "skills", "S", {}, "s", "s", "/tmp/s.md", "h", [0.1] * 768)
    cols = db.list_collections()
    assert len(cols) == 2
    skills_col = next(c for c in cols if c["id"] == "skills")
    assert skills_col["doc_count"] == 1
    notes_col = next(c for c in cols if c["id"] == "notes")
    assert notes_col["doc_count"] == 0
