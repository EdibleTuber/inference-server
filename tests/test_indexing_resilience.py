"""Collection indexing survives malformed or failing documents (audit S1).

Incident 2026-10-27: an empty `tags:` in vault/templates/note-template.md made
derive_tags raise, which aborted indexing of that collection and every later one.
"""
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from manager.collections import derive_tags, index_collection
from manager.vectordb import VectorDB


def _tags(frontmatter: dict) -> set:
    return set(derive_tags(
        file_path="/src/Area/Topic/note.md",
        source_dir="/src",
        frontmatter=frontmatter,
    ))


def test_derive_tags_tolerates_empty_tags():
    assert _tags({"tags": None}) == {"area", "topic"}


def test_derive_tags_accepts_a_single_string():
    tags = _tags({"tags": "Solo"})
    assert "solo" in tags
    assert not {"s", "o", "l"} & tags


def test_derive_tags_skips_non_string_items():
    tags = _tags({"tags": ["Good", 7, None, {"x": 1}]})
    assert "good" in tags
    assert tags == {"area", "topic", "good"}


@pytest.fixture
def db(tmp_path):
    d = VectorDB(str(tmp_path / "idx.db"))
    d.init_schema()
    yield d
    d.close()


def _embeddings(fail_on: str | None = None):
    async def embed(text):
        if fail_on is not None and fail_on in text:
            raise RuntimeError(f"embedding failed for {text!r}")
        return [0.0] * 768
    e = AsyncMock()
    e.embed_text = AsyncMock(side_effect=embed)
    return e


async def _index(source: Path, db, embeddings):
    return await index_collection(
        collection_id="notes", source_dir=str(source), doc_type="markdown",
        db=db, embeddings=embeddings,
    )


@pytest.mark.asyncio
async def test_empty_tags_frontmatter_is_indexed(tmp_path, db):
    src = tmp_path / "notes"
    src.mkdir()
    (src / "template.md").write_text("---\ntitle: T\ntags:\n---\n# T\n\nTemplate body.\n")

    stats = await _index(src, db, _embeddings())

    assert stats["new"] == 1
    assert stats["errors"] == 0
    assert db.get_document("notes", "template") is not None


@pytest.mark.asyncio
async def test_one_bad_document_does_not_abort_the_collection(tmp_path, db):
    src = tmp_path / "notes"
    src.mkdir()
    (src / "a.md").write_text("# A\n\nGood note.\n")
    (src / "b.md").write_text("# B\n\nBAD note.\n")
    (src / "c.md").write_text("# C\n\nAnother good note.\n")

    stats = await _index(src, db, _embeddings(fail_on="BAD"))

    assert stats["new"] == 2
    assert stats["errors"] == 1
    assert db.get_document("notes", "a") is not None
    assert db.get_document("notes", "c") is not None
    assert db.get_document("notes", "b") is None


@pytest.mark.asyncio
async def test_a_failed_document_is_retried_on_the_next_run(tmp_path, db):
    src = tmp_path / "notes"
    src.mkdir()
    (src / "b.md").write_text("# B\n\nBAD note.\n")

    first = await _index(src, db, _embeddings(fail_on="BAD"))
    second = await _index(src, db, _embeddings())

    assert (first["errors"], first["new"]) == (1, 0)
    assert (second["errors"], second["new"]) == (0, 1)


@pytest.mark.asyncio
async def test_a_failed_update_keeps_the_previous_version(tmp_path, db):
    """A document that fails to re-index must not be stale-deleted."""
    src = tmp_path / "notes"
    src.mkdir()
    note = src / "n.md"
    note.write_text("# N\n\nOriginal text.\n")
    await _index(src, db, _embeddings())

    note.write_text("# N\n\nBAD rewrite.\n")
    stats = await _index(src, db, _embeddings(fail_on="BAD"))

    assert stats["errors"] == 1
    assert stats["removed"] == 0
    doc = db.get_document("notes", "n")
    assert doc is not None and "Original text." in doc["content"]
