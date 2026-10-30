"""Collection-scoped, indexed hash lookups and an index pass that yields (audit I1, I2).

Found after the round-2 deploy (2026-10-28):
- I1: no index on documents.file_path. 2,937 get_hash lookups took 30.3s
  and blocked the event loop (0.041s with an index).
- I2: get_hash ignored the collection, so 695 vault files indexed under an
  orphaned `notes` collection were skipped as "unchanged" and never appeared
  in vault search.
"""
import asyncio
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from manager.collections import index_collection
from manager.vectordb import VectorDB


@pytest.fixture
def db(tmp_path):
    d = VectorDB(str(tmp_path / "idx.db"))
    d.init_schema()
    yield d
    d.close()


def _embeddings():
    e = AsyncMock()
    e.embed_text = AsyncMock(return_value=[0.0] * 768)
    return e


def _index_columns(db) -> list[list[str]]:
    names = [r[1] for r in db._conn.execute("PRAGMA index_list(documents)").fetchall()]
    return [[c[2] for c in db._conn.execute(f"PRAGMA index_info('{n}')").fetchall()]
            for n in names]


# ---------------------------------------------------------------------------
# I1: an index serves the per-file lookup, including on an existing database
# ---------------------------------------------------------------------------

def test_documents_have_a_collection_file_path_index(db):
    assert ["collection", "file_path"] in _index_columns(db)


def test_init_schema_adds_the_index_to_an_existing_database(tmp_path):
    path = str(tmp_path / "old.db")
    first = VectorDB(path)
    first.init_schema()
    for row in first._conn.execute("PRAGMA index_list(documents)").fetchall():
        name = row[1]
        cols = [c[2] for c in first._conn.execute(f"PRAGMA index_info('{name}')")]
        if cols == ["collection", "file_path"]:
            first._conn.execute(f"DROP INDEX {name}")   # simulate a pre-index database
    first._conn.commit()
    first.close()

    again = VectorDB(path)
    again.init_schema()
    assert ["collection", "file_path"] in _index_columns(again)
    again.close()


@pytest.mark.asyncio
async def test_an_unchanged_index_pass_yields_to_the_event_loop(tmp_path, db):
    """A pass over unchanged files does no awaits of its own, so without an
    explicit yield it holds the event loop for the whole pass."""
    src = tmp_path / "notes"
    src.mkdir()
    for i in range(120):
        (src / f"n{i}.md").write_text(f"# N{i}\n\nNote {i}.\n")
    embeddings = _embeddings()
    await index_collection(collection_id="notes", source_dir=str(src),
                           doc_type="markdown", db=db, embeddings=embeddings)

    ticks = 0
    stop = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0)

    t = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    before = ticks
    stats = await index_collection(collection_id="notes", source_dir=str(src),
                                   doc_type="markdown", db=db, embeddings=embeddings)
    during = ticks - before
    stop.set()
    await t

    assert stats["unchanged"] == 120
    assert during >= 2


# ---------------------------------------------------------------------------
# I2: lookups are scoped to the collection being indexed
# ---------------------------------------------------------------------------

def test_get_hash_is_scoped_to_the_collection(db):
    db.upsert_collection("notes", "/src", "markdown")
    db.upsert_document("a", "notes", "A", {}, "s", "c", "/src/a.md", "h1", [0.0] * 768)

    assert db.get_hash("notes", "/src/a.md") == "h1"
    assert db.get_hash("vault", "/src/a.md") is None


@pytest.mark.asyncio
async def test_a_file_indexed_under_another_collection_is_indexed_here(tmp_path, db):
    """The live bug: a file already indexed (same content) under an orphaned
    collection over the same directory must still be indexed for this one."""
    src = tmp_path / "vault"
    src.mkdir()
    note = src / "a.md"
    note.write_text("# A\n\nShared note.\n")
    digest = hashlib.sha256(note.read_bytes()).hexdigest()
    db.upsert_collection("notes", str(src), "markdown")
    db.upsert_document("a", "notes", "A", {}, "Shared note.", note.read_text(),
                       str(note.resolve()), digest, [0.0] * 768)

    stats = await index_collection(collection_id="vault", source_dir=str(src),
                                   doc_type="markdown", db=db, embeddings=_embeddings())

    assert stats["new"] == 1 and stats["unchanged"] == 0
    assert db.get_document("vault", "a") is not None


# ---------------------------------------------------------------------------
# Pruning a collection
# ---------------------------------------------------------------------------

def test_delete_collection_removes_its_documents_vectors_and_row(db):
    db.upsert_collection("notes", "/src", "markdown")
    db.upsert_collection("keep", "/k", "markdown")
    db.upsert_document("a", "notes", "A", {}, "s", "c", "/src/a.md", "h", [0.0] * 768)
    db.upsert_document("b", "notes", "B", {}, "s", "c", "/src/b.md", "h", [0.0] * 768)
    db.upsert_document("k", "keep", "K", {}, "s", "c", "/k/k.md", "h", [0.0] * 768)

    removed = db.delete_collection("notes")

    assert removed == 2
    assert [c["id"] for c in db.list_collections()] == ["keep"]
    vec_ids = {r[0] for r in db._conn.execute("SELECT d.id FROM doc_vectors v JOIN documents d ON d.pk = v.rowid").fetchall()}
    assert vec_ids == {"k"}
    assert db.get_document("keep", "k") is not None
