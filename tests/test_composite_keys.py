"""Q23: document ids are unique per collection, not globally; search runs
inside one collection.

Before: documents.id was a global PRIMARY KEY (vectordb.py:43-44), so two
collections holding the same relative path shared one row and one vector,
and search() took a global top-k before filtering by collection, so a small
collection beside a large one came back empty.
"""
from unittest.mock import AsyncMock

import pytest

from manager.collections import get_children, index_collection
from manager.vectordb import VectorDB

DIM = 768


def unit(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


@pytest.fixture
def db(tmp_path):
    d = VectorDB(str(tmp_path / "ck.db"))
    d.init_schema()
    yield d
    d.close()


def _two_readmes(db):
    db.upsert_collection("a", "/a", "markdown")
    db.upsert_collection("b", "/b", "markdown")
    db.upsert_document("README", "a", "A", {}, "sa", "content a", "/a/README.md", "ha", unit(0))
    db.upsert_document("README", "b", "B", {}, "sb", "content b", "/b/README.md", "hb", unit(1))


def test_same_id_in_two_collections_keeps_both(db):
    _two_readmes(db)
    assert db.get_document("a", "README")["content"] == "content a"
    assert db.get_document("b", "README")["content"] == "content b"
    assert db.get_hash("a", "/a/README.md") == "ha"
    assert db.get_hash("b", "/b/README.md") == "hb"
    assert {c["id"]: c["doc_count"] for c in db.list_collections()} == {"a": 1, "b": 1}


def test_each_collection_searches_its_own_vector(db):
    _two_readmes(db)
    [hit_a] = db.search("a", unit(0), limit=5)
    [hit_b] = db.search("b", unit(1), limit=5)
    assert (hit_a["collection"], hit_a["score"]) == ("a", pytest.approx(1.0))
    assert (hit_b["collection"], hit_b["score"]) == ("b", pytest.approx(1.0))


@pytest.mark.asyncio
async def test_reindexing_two_collections_with_a_shared_path_is_a_no_op(tmp_path, db):
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        (d / "README.md").write_text(f"# {name}\n\nAbout {name}.\n")
    emb = AsyncMock()
    emb.embed_text = AsyncMock(return_value=unit(0))
    for name in ("a", "b"):
        await index_collection(collection_id=name, source_dir=str(tmp_path / name),
                               doc_type="markdown", db=db, embeddings=emb)
    emb.embed_text.reset_mock()

    stats = [await index_collection(collection_id=name, source_dir=str(tmp_path / name),
                                    doc_type="markdown", db=db, embeddings=emb)
             for name in ("a", "b")]

    assert [s["unchanged"] for s in stats] == [1, 1]
    assert emb.embed_text.await_count == 0


def test_a_small_collection_is_searchable_beside_a_large_one(db):
    db.upsert_collection("big", "/big", "markdown")
    db.upsert_collection("small", "/small", "markdown")
    for i in range(150):
        db.upsert_document(f"n{i}", "big", f"N{i}", {}, "s", "c", f"/big/n{i}.md", "h", unit(0))
    db.upsert_document("only", "small", "Only", {}, "s", "c", "/small/only.md", "h", unit(1))

    hits = db.search("small", unit(0), limit=5)

    assert [h["id"] for h in hits] == ["only"]
    # Orthogonal unit vectors: the true score is 0.0, but the distance is
    # float32, so 1.0 - L2**2/2 lands at ~3.4e-08, not 0.0.
    assert hits[0]["score"] == pytest.approx(0.0, abs=1e-6)


def test_children_match_the_prefix_literally(db):
    db.upsert_collection("skills", "/s", "skill")
    for doc_id in ("A_B/SKILL", "A_B/Workflows/x", "AxB/Workflows/y"):
        db.upsert_document(doc_id, "skills", doc_id, {}, "s", "c", f"/s/{doc_id}.md", "h", unit(0))

    assert [c["id"] for c in get_children(db, "skills", "A_B/SKILL")] == ["A_B/Workflows/x"]


def test_a_fresh_database_is_created_at_the_current_schema_version(db):
    from manager.vectordb import SCHEMA_VERSION
    assert SCHEMA_VERSION == 1
    assert db._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_a_newer_schema_version_is_refused(tmp_path):
    from manager.vectordb import SchemaVersionError
    path = str(tmp_path / "future.db")
    first = VectorDB(path)
    first.init_schema()
    first._conn.execute("PRAGMA user_version = 2")
    first._conn.commit()
    first.close()

    again = VectorDB(path)
    with pytest.raises(SchemaVersionError):
        again.init_schema()
    again.close()
