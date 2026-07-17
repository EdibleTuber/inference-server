# manager/vectordb.py
"""
SQLite-vec vector database wrapper.

Manages schema, document upserts, similarity search, and stale cleanup.
Uses sqlite-vec extension for vector similarity search with cosine distance.
"""
import json
import logging
import sqlite3
import struct

import sqlite_vec

logger = logging.getLogger(__name__)


def _serialize_vec(vec: list[float]) -> bytes:
    """Serialize float vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


class VectorDB:
    """SQLite-vec backed vector database for document retrieval."""

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

    def init_schema(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript("""
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
        """)
        self._conn.commit()

    def upsert_collection(self, id: str, source_dir: str, doc_type: str) -> None:
        """Insert or update a collection."""
        self._conn.execute(
            """INSERT INTO collections (id, source_dir, doc_type)
               VALUES (?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET source_dir=excluded.source_dir, doc_type=excluded.doc_type""",
            (id, source_dir, doc_type),
        )
        self._conn.commit()

    def upsert_document(
        self,
        doc_id: str,
        collection: str,
        name: str,
        metadata: dict,
        summary: str,
        content: str,
        file_path: str,
        file_hash: str,
        embedding: list[float],
    ) -> None:
        """Insert or update a document and its embedding."""
        meta_json = json.dumps(metadata)
        self._conn.execute(
            """INSERT INTO documents (id, collection, name, metadata, summary, content, file_path, file_hash, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(id) DO UPDATE SET
                 collection=excluded.collection, name=excluded.name, metadata=excluded.metadata,
                 summary=excluded.summary, content=excluded.content, file_path=excluded.file_path,
                 file_hash=excluded.file_hash, updated_at=CURRENT_TIMESTAMP""",
            (doc_id, collection, name, meta_json, summary, content, file_path, file_hash),
        )
        vec_bytes = _serialize_vec(embedding)
        self._conn.execute("DELETE FROM documents_vec WHERE id = ?", (doc_id,))
        self._conn.execute(
            "INSERT INTO documents_vec (id, embedding) VALUES (?, ?)",
            (doc_id, vec_bytes),
        )
        self._conn.commit()

    def search(
        self,
        collection: str,
        query_embedding: list[float],
        limit: int = 5,
        tags: list[str] | None = None,
    ) -> list[dict]:
        """Search for similar documents within a collection."""
        vec_bytes = _serialize_vec(query_embedding)
        # sqlite-vec requires k= or LIMIT directly on the vec0 table — JOINs with WHERE
        # clauses confuse the query planner. Fetch a broad candidate set then filter in Python.
        k = max(limit * 20, 100)
        vec_rows = self._conn.execute(
            """SELECT id, distance
               FROM documents_vec
               WHERE embedding MATCH ?
               AND k = ?
               ORDER BY distance""",
            (vec_bytes, k),
        ).fetchall()

        results = []
        for row in vec_rows:
            doc = self._conn.execute(
                "SELECT id, name, collection, summary, metadata FROM documents WHERE id = ? AND collection = ?",
                (row["id"], collection),
            ).fetchone()
            if doc is None:
                continue
            meta = json.loads(doc["metadata"]) if doc["metadata"] else {}
            doc_tags = meta.get("tags", [])
            if tags and not set(tags).intersection(set(doc_tags)):
                continue
            results.append({
                "id": doc["id"],
                "name": doc["name"],
                "collection": doc["collection"],
                "summary": doc["summary"],
                "tags": doc_tags,
                # documents_vec uses sqlite-vec's DEFAULT L2 (Euclidean) metric, so
                # row["distance"] is an L2 distance, not cosine distance. Embeddings are
                # unit-normalized, for which cosine = 1 - L2^2/2. Recover that so score
                # is a true cosine similarity in [-1, 1]. (The old `1 - distance` treated
                # L2 as cosine distance, compressing ~0.85-cosine hits to ~0.25 and going
                # negative below 0.5 cosine.)
                "score": 1.0 - (row["distance"] ** 2) / 2.0,
            })
            if len(results) >= limit:
                break
        return results

    def get_document(self, collection: str, doc_id: str) -> dict | None:
        """Get full document by collection and ID."""
        row = self._conn.execute(
            "SELECT id, name, collection, summary, content, metadata FROM documents WHERE collection = ? AND id = ?",
            (collection, doc_id),
        ).fetchone()
        if row is None:
            return None
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        return {
            "id": row["id"],
            "name": row["name"],
            "collection": row["collection"],
            "summary": row["summary"],
            "content": row["content"],
            "metadata": meta,
        }

    def get_hash(self, file_path: str) -> str | None:
        """Get stored hash for a file path, or None if not indexed."""
        row = self._conn.execute(
            "SELECT file_hash FROM documents WHERE file_path = ?",
            (file_path,),
        ).fetchone()
        return row["file_hash"] if row else None

    def delete_stale(self, collection: str, keep_paths: set[str]) -> int:
        """Delete documents whose file_path is not in keep_paths. Returns count removed."""
        rows = self._conn.execute(
            "SELECT id, file_path FROM documents WHERE collection = ?",
            (collection,),
        ).fetchall()
        removed = 0
        for row in rows:
            if row["file_path"] not in keep_paths:
                self._conn.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
                self._conn.execute("DELETE FROM documents_vec WHERE id = ?", (row["id"],))
                removed += 1
        if removed:
            self._conn.commit()
        return removed

    def list_collections(self) -> list[dict]:
        """List all collections with document counts."""
        rows = self._conn.execute(
            """SELECT c.id, c.source_dir, c.doc_type,
                      (SELECT COUNT(*) FROM documents d WHERE d.collection = c.id) as doc_count
               FROM collections c ORDER BY c.id"""
        ).fetchall()
        return [
            {"id": r["id"], "source_dir": r["source_dir"], "doc_type": r["doc_type"], "doc_count": r["doc_count"]}
            for r in rows
        ]

    def close(self):
        """Close the database connection."""
        self._conn.close()
