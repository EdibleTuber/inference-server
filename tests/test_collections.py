# tests/test_collections.py
"""Tests for the collection indexing pipeline."""
import pytest
import json
from pathlib import Path


@pytest.fixture
def skills_dir(tmp_path):
    """Create a temporary skills directory with sample SKILL.md and workflow files."""
    skill_dir = tmp_path / "Security" / "Recon"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: Recon\ndescription: Security reconnaissance. USE WHEN recon, bug bounty, attack surface.\n---\n\n# Recon\n\nReconnaissance skill for security assessment.\n"
    )
    wf_dir = skill_dir / "Workflows"
    wf_dir.mkdir()
    (wf_dir / "PassiveRecon.md").write_text(
        "---\nworkflow: passive-recon\npurpose: Non-intrusive reconnaissance\n---\n\n# Passive Recon\n\n## Purpose\n\nGather information without touching the target.\n\n## Steps\n\n1. DNS lookup\n2. WHOIS\n"
    )
    return str(tmp_path)


@pytest.fixture
def notes_dir(tmp_path):
    """Create a temporary notes directory."""
    projects = tmp_path / "notes" / "Projects" / "pi-mono"
    projects.mkdir(parents=True)
    (projects / "context-management.md").write_text(
        "# Context Management\n\nNotes on how Mother handles context window limits.\n\n## Key insight\n\nStrip old tool results.\n"
    )
    return str(tmp_path / "notes")


def test_parse_frontmatter():
    """parse_frontmatter extracts YAML frontmatter from markdown."""
    from manager.collections import parse_frontmatter
    content = "---\nname: Test\ndescription: A test.\ntags: [a, b]\n---\n\n# Content"
    meta, body = parse_frontmatter(content)
    assert meta["name"] == "Test"
    assert meta["tags"] == ["a", "b"]
    assert body.strip() == "# Content"


def test_parse_frontmatter_none():
    """parse_frontmatter returns empty dict if no frontmatter."""
    from manager.collections import parse_frontmatter
    meta, body = parse_frontmatter("# No frontmatter\n\nJust content.")
    assert meta == {}
    assert "No frontmatter" in body


def test_extract_summary_skill():
    """extract_summary for skill doc_type uses frontmatter description."""
    from manager.collections import extract_summary
    meta = {"description": "Security reconnaissance. USE WHEN recon, bug bounty."}
    body = "# Recon\n\nSome content."
    summary = extract_summary(meta, body, "skill")
    assert "Security reconnaissance" in summary


def test_extract_summary_markdown():
    """extract_summary for markdown doc_type uses first paragraph."""
    from manager.collections import extract_summary
    body = "# Title\n\nThis is the first paragraph with key information.\n\n## Section\n\nMore content."
    summary = extract_summary({}, body, "markdown")
    assert "first paragraph" in summary


def test_derive_tags_from_path():
    """derive_tags extracts lowercased directory segments from relative path."""
    from manager.collections import derive_tags
    tags = derive_tags(
        file_path="/home/user/skills/Security/Recon/SKILL.md",
        source_dir="/home/user/skills",
        frontmatter={},
    )
    assert "security" in tags
    assert "recon" in tags


def test_derive_tags_merges_frontmatter():
    """derive_tags merges frontmatter tags with path-derived tags."""
    from manager.collections import derive_tags
    tags = derive_tags(
        file_path="/home/user/skills/Security/Recon/SKILL.md",
        source_dir="/home/user/skills",
        frontmatter={"tags": ["scanning", "enumeration"]},
    )
    assert "security" in tags
    assert "scanning" in tags
    assert "enumeration" in tags


def test_derive_tags_parses_use_when():
    """derive_tags extracts trigger keywords from USE WHEN pattern in description."""
    from manager.collections import derive_tags
    tags = derive_tags(
        file_path="/tmp/skills/Recon/SKILL.md",
        source_dir="/tmp/skills",
        frontmatter={"description": "Recon skill. USE WHEN recon, bug bounty, attack surface."},
    )
    assert "recon" in tags
    assert "bug bounty" in tags
    assert "attack surface" in tags


def test_scan_collection_finds_markdown_files(skills_dir):
    """scan_collection returns all .md files with hashes."""
    from manager.collections import scan_collection
    files = scan_collection(skills_dir)
    assert len(files) == 2
    paths = {f["file_path"] for f in files}
    assert any("SKILL.md" in p for p in paths)
    assert any("PassiveRecon.md" in p for p in paths)
    assert all("file_hash" in f for f in files)


def test_make_doc_id():
    """make_doc_id strips source_dir prefix and .md suffix."""
    from manager.collections import make_doc_id
    doc_id = make_doc_id(
        file_path="/home/user/skills/Security/Recon/SKILL.md",
        source_dir="/home/user/skills",
    )
    assert doc_id == "Security/Recon/SKILL"


def test_get_children_for_skill(skills_dir):
    """get_children returns workflow documents that share a skill's path prefix."""
    from manager.collections import get_children
    from manager.vectordb import VectorDB

    db = VectorDB(":memory:")
    db.init_schema()
    db.upsert_collection("skills", skills_dir, "skill")
    db.upsert_document("Security/Recon/SKILL", "skills", "Recon", {}, "Recon skill", "c", "/a", "h1", [0.1]*768)
    db.upsert_document("Security/Recon/Workflows/PassiveRecon", "skills", "PassiveRecon", {}, "Passive recon", "c", "/b", "h2", [0.1]*768)

    children = get_children(db, "skills", "Security/Recon/SKILL")
    assert len(children) == 1
    assert children[0]["id"] == "Security/Recon/Workflows/PassiveRecon"


import pytest
from unittest.mock import AsyncMock


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

    # First: full scan should index all files (notes_dir fixture pre-creates one file,
    # plus a.md and b.md = 3 total).
    stats = await index_collection(
        collection_id="notes",
        source_dir=notes_dir,
        doc_type="markdown",
        db=db,
        embeddings=embeddings,
    )
    assert stats["new"] == 3
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
