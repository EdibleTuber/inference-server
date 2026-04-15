# manager/collections.py
"""
Collection indexing pipeline.

Scans configured directories for markdown files, extracts summaries and tags,
generates embeddings, and stores documents in the vector database.
"""
import hashlib
import json
import logging
import re
from pathlib import Path

import yaml

from manager.embeddings import EmbeddingsClient
from manager.vectordb import VectorDB

logger = logging.getLogger(__name__)


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content.

    Returns (metadata_dict, body_without_frontmatter).
    """
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}, content

    return meta, parts[2]


def extract_summary(frontmatter: dict, body: str, doc_type: str) -> str:
    """Extract a summary string based on document type.

    For skills: uses frontmatter description.
    For workflows: uses first ## Purpose section or first paragraph.
    For markdown/telos: uses first paragraph after any heading.
    """
    if doc_type == "skill" and "description" in frontmatter:
        return frontmatter["description"]

    if doc_type == "skill" and "purpose" in frontmatter:
        return frontmatter["purpose"]

    # Look for a ## Purpose section
    purpose_match = re.search(
        r"^## Purpose\s*\n+(.+?)(?=\n##|\n\n\n|\Z)",
        body,
        re.MULTILINE | re.DOTALL,
    )
    if purpose_match:
        text = purpose_match.group(1).strip()
        words = text.split()
        return " ".join(words[:200])

    # Fall back to first paragraph after any heading
    para_match = re.search(
        r"^#[^\n]*\n+([^\n#].+?)(?=\n\n|\n#|\Z)",
        body,
        re.MULTILINE | re.DOTALL,
    )
    if para_match:
        text = para_match.group(1).strip()
        words = text.split()
        return " ".join(words[:200])

    # Last resort: first 200 words
    words = body.split()
    return " ".join(words[:200])


def derive_tags(file_path: str, source_dir: str, frontmatter: dict) -> list[str]:
    """Derive tags from directory path and frontmatter.

    Path segments become lowercase tags. Frontmatter 'tags' field is merged.
    USE WHEN patterns in description are parsed for trigger keywords.
    """
    tags = set()

    # Tags from path segments (between source_dir and filename)
    rel = Path(file_path).relative_to(source_dir)
    for part in rel.parent.parts:
        tags.add(part.lower())

    # Tags from frontmatter
    if "tags" in frontmatter:
        for tag in frontmatter["tags"]:
            tags.add(tag.lower())

    # Parse USE WHEN pattern from description
    desc = frontmatter.get("description", "")
    use_when_match = re.search(r"USE WHEN\s+(.+?)(?:\.|$)", desc, re.IGNORECASE)
    if use_when_match:
        triggers = [t.strip().lower() for t in use_when_match.group(1).split(",")]
        tags.update(triggers)

    # Remove empty strings
    tags.discard("")

    return sorted(tags)


def make_doc_id(file_path: str, source_dir: str) -> str:
    """Generate document ID from file path relative to source directory."""
    rel = Path(file_path).relative_to(source_dir)
    return str(rel.with_suffix(""))


def scan_collection(source_dir: str) -> list[dict]:
    """Scan a directory for .md files. Returns list of {file_path, file_hash}."""
    results = []
    source = Path(source_dir).resolve()
    if not source.exists():
        logger.warning("Collection source_dir does not exist: %s", source_dir)
        return results

    for md_file in sorted(source.rglob("*.md")):
        content = md_file.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        results.append({
            "file_path": str(md_file.resolve()),
            "file_hash": file_hash,
        })

    return results


def get_children(db: VectorDB, collection: str, skill_doc_id: str) -> list[dict]:
    """Get child workflow documents for a SKILL.md document.

    A child is any document in the same collection whose ID shares the
    skill's parent path prefix (e.g., Security/Recon/SKILL ->
    Security/Recon/Workflows/*).
    """
    # Get the parent directory from the skill doc ID
    parent_prefix = skill_doc_id.rsplit("/", 1)[0] + "/"

    rows = db._conn.execute(
        "SELECT id, name, summary FROM documents WHERE collection = ? AND id LIKE ? AND id != ?",
        (collection, f"{parent_prefix}%", skill_doc_id),
    ).fetchall()

    return [{"id": r["id"], "name": r["name"], "summary": r["summary"]} for r in rows]


async def index_collection(
    collection_id: str,
    source_dir: str,
    doc_type: str,
    db: VectorDB,
    embeddings: EmbeddingsClient,
    paths: list[str] | None = None,
) -> dict:
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
            if not full.is_file():
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


def load_collections_config(config_path: str) -> list[dict]:
    """Load collection definitions from JSON config file.

    Returns list of {id, source_dir, doc_type} dicts.
    Returns empty list if file doesn't exist.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("Collections config not found: %s", config_path)
        return []

    with open(path) as f:
        return json.load(f)


async def index_all_collections(
    config_path: str,
    db: VectorDB,
    embeddings: EmbeddingsClient,
) -> None:
    """Index all collections defined in the config file."""
    collections = load_collections_config(config_path)
    if not collections:
        logger.info("No collections configured, skipping indexing")
        return

    for col in collections:
        logger.info("Indexing collection '%s' from %s", col["id"], col["source_dir"])
        stats = await index_collection(
            collection_id=col["id"],
            source_dir=col["source_dir"],
            doc_type=col["doc_type"],
            db=db,
            embeddings=embeddings,
        )
        logger.info(
            "Collection '%s': %d new, %d updated, %d removed, %d unchanged",
            col["id"], stats["new"], stats["updated"], stats["removed"], stats["unchanged"],
        )
