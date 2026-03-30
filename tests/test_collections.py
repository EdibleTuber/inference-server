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
