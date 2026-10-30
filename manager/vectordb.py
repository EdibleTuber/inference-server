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
from pathlib import Path

import sqlite_vec

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class SchemaVersionError(RuntimeError):
    """Raised when the database schema version cannot be handled by this code."""


class MigrationError(RuntimeError):
    """Raised when a v0 -> v1 migration cannot be completed or verified."""


def _serialize_vec(vec: list[float]) -> bytes:
    """Serialize float vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def _verify_migration(conn: sqlite3.Connection) -> None:
    """Verify a v0 -> v1 migration before anything legacy is dropped.

    Runs after both copies and before the legacy tables are dropped. Raises
    MigrationError unless count(documents_v1) == count(documents) and
    count(doc_vectors) == count(documents_v1). The second check means every
    legacy document got its vector: a document without one would never be
    re-embedded (its hash still matches), so it would stay unsearchable
    forever.
    """
    n_legacy = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    n_v1 = conn.execute("SELECT count(*) FROM documents_v1").fetchone()[0]
    n_vecs = conn.execute("SELECT count(*) FROM doc_vectors").fetchone()[0]
    if n_v1 != n_legacy or n_vecs != n_v1:
        raise MigrationError(
            f"v0 -> v1 migration check failed: {n_legacy} legacy documents, "
            f"{n_v1} copied to documents_v1, {n_vecs} vectors"
        )


class VectorDB:
    """SQLite-vec backed vector database for document retrieval."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

    def init_schema(self) -> None:
        """Create or verify the schema, dispatching on the stored version.

        A fresh database is created at SCHEMA_VERSION. A v1 database gets its
        index re-asserted. A legacy v0 database (one that predates the version
        stamp) is migrated once to v1, after a backup, in one transaction. An
        unknown newer version is refused outright.
        """
        has_documents = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'documents'"
        ).fetchone() is not None
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]

        if version == 0 and not has_documents:
            self._create_v1()
        elif version == 0:
            self._migrate_v0_to_v1()
        elif version == SCHEMA_VERSION:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_collection_file_path"
                " ON documents(collection, file_path)"
            )
            self._conn.commit()
        else:
            raise SchemaVersionError(f"unknown schema version {version}")

    def _create_v1(self) -> None:
        """Create the v1 schema. This DDL is the contract the v0 -> v1
        migration copies into place (see the phase 2 plan, Task 4).

        Documents are keyed by (collection, id): the same relative path can
        exist in several collections. doc_vectors.rowid = documents.pk — the
        vec rowid space is shared across partitions, so it must be globally
        unique, which documents.pk (a rowid alias) is while documents.id is
        only unique per collection.
        """
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS collections (
                id          TEXT PRIMARY KEY,
                source_dir  TEXT NOT NULL,
                doc_type    TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE documents (
                pk          INTEGER PRIMARY KEY,
                collection  TEXT NOT NULL REFERENCES collections(id),
                id          TEXT NOT NULL,
                name        TEXT NOT NULL,
                metadata    TEXT,
                summary     TEXT NOT NULL,
                content     TEXT NOT NULL,
                file_path   TEXT NOT NULL,
                file_hash   TEXT NOT NULL,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(collection, id)
            );

            CREATE VIRTUAL TABLE doc_vectors USING vec0(
                collection TEXT PARTITION KEY,
                embedding  float[768]
            );

            CREATE INDEX idx_documents_collection_file_path
                ON documents(collection, file_path);
        """)
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.commit()

    def _migrate_v0_to_v1(self) -> None:
        """Migrate a legacy v0 database to schema v1, backed up first.

        The order is the invariant:
        1. Back up the legacy DB to {db_path}.pre-v1.bak. If the backup file
           already exists, leave it alone: it can only have been written by an
           earlier attempt that rolled back, so it is the legacy DB. If the
           backup raises, the error propagates and nothing is migrated.
        2. Run the migration in one transaction. An explicit BEGIN IMMEDIATE
           is required: Python's sqlite3 does not open a transaction for DDL,
           so without it every CREATE/DROP would autocommit on its own.
        3. Any exception in the transaction runs ROLLBACK and re-raises,
           leaving the legacy database untouched.

        The v1 DDL copied into place is _create_v1's, per the phase 2 plan.
        """
        bak_path = f"{self._db_path}.pre-v1.bak"
        if not Path(bak_path).exists():
            bak_conn = sqlite3.connect(bak_path)
            try:
                self._conn.backup(bak_conn)
            finally:
                bak_conn.close()

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            # New documents table (v1 DDL), then copy every legacy row,
            # letting pk be assigned.
            self._conn.execute("""
                CREATE TABLE documents_v1 (
                    pk          INTEGER PRIMARY KEY,
                    collection  TEXT NOT NULL REFERENCES collections(id),
                    id          TEXT NOT NULL,
                    name        TEXT NOT NULL,
                    metadata    TEXT,
                    summary     TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    file_path   TEXT NOT NULL,
                    file_hash   TEXT NOT NULL,
                    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(collection, id)
                )
            """)
            self._conn.execute("""
                INSERT INTO documents_v1
                    (collection, id, name, metadata, summary, content,
                     file_path, file_hash, updated_at)
                SELECT collection, id, name, metadata, summary, content,
                       file_path, file_hash, updated_at
                FROM documents
            """)
            # New vec table (v1 DDL), then copy the vectors keyed by pk.
            self._conn.execute("""
                CREATE VIRTUAL TABLE doc_vectors USING vec0(
                    collection TEXT PARTITION KEY,
                    embedding  float[768]
                )
            """)
            self._conn.execute("""
                INSERT INTO doc_vectors (rowid, collection, embedding)
                SELECT d.pk, d.collection, v.embedding
                FROM documents_v1 d JOIN documents_vec v ON v.id = d.id
            """)
            # Verify before anything legacy is dropped.
            _verify_migration(self._conn)
            # Drop the legacy tables (documents' index goes with it), rename,
            # and create the index.
            self._conn.execute("DROP TABLE documents_vec")
            self._conn.execute("DROP TABLE documents")
            self._conn.execute("ALTER TABLE documents_v1 RENAME TO documents")
            self._conn.execute(
                "CREATE INDEX idx_documents_collection_file_path"
                " ON documents(collection, file_path)"
            )
            # Stamp the version, then commit.
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        n_docs = self._conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        n_vecs = self._conn.execute("SELECT count(*) FROM doc_vectors").fetchone()[0]
        logger.info(
            "skills.db schema migrated v0 -> v1: %d documents, %d vectors; backup at %s",
            n_docs, n_vecs, bak_path,
        )

    def upsert_collection(self, id: str, source_dir: str, doc_type: str) -> None:
        """Insert or update a collection, rolling back on any exception."""
        try:
            self._conn.execute(
                """INSERT INTO collections (id, source_dir, doc_type)
                   VALUES (?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET source_dir=excluded.source_dir, doc_type=excluded.doc_type""",
                (id, source_dir, doc_type),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

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
        """Insert or update a document and its embedding, in one transaction.

        Rolls back on any exception before re-raising: a vector insert that
        fails must not leave the document row in an open transaction, where the
        next write's commit() would store it with its hash and no vector —
        unfindable by search forever (its hash would never change, so no
        reindex would re-embed it).
        """
        meta_json = json.dumps(metadata)
        try:
            pk = self._conn.execute(
                """INSERT INTO documents (collection, id, name, metadata, summary, content, file_path, file_hash, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(collection, id) DO UPDATE SET
                     name=excluded.name, metadata=excluded.metadata,
                     summary=excluded.summary, content=excluded.content, file_path=excluded.file_path,
                     file_hash=excluded.file_hash, updated_at=CURRENT_TIMESTAMP
                   RETURNING pk""",
                (collection, doc_id, name, meta_json, summary, content, file_path, file_hash),
            ).fetchone()[0]
            vec_bytes = _serialize_vec(embedding)
            self._conn.execute(
                "DELETE FROM doc_vectors WHERE collection = ? AND rowid = ?",
                (collection, pk),
            )
            self._conn.execute(
                "INSERT INTO doc_vectors (rowid, collection, embedding) VALUES (?, ?, ?)",
                (pk, collection, vec_bytes),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

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
        # clauses confuse the query planner. The collection partition key keeps the KNN
        # inside one collection, but the tag filter still runs afterwards, so we keep a
        # broad candidate set.
        k = max(limit * 20, 100)
        vec_rows = self._conn.execute(
            """SELECT rowid, distance
               FROM doc_vectors
               WHERE embedding MATCH ?
               AND k = ?
               AND collection = ?
               ORDER BY distance""",
            (vec_bytes, k, collection),
        ).fetchall()

        results = []
        for row in vec_rows:
            doc = self._conn.execute(
                "SELECT id, name, collection, summary, metadata FROM documents WHERE pk = ? AND collection = ?",
                (row["rowid"], collection),
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
                # doc_vectors uses sqlite-vec's DEFAULT L2 (Euclidean) metric, so
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

    def get_hash(self, collection: str, file_path: str) -> str | None:
        """Get stored hash for a file path within a collection, or None if not indexed."""
        row = self._conn.execute(
            "SELECT file_hash FROM documents WHERE collection = ? AND file_path = ?",
            (collection, file_path),
        ).fetchone()
        return row["file_hash"] if row else None

    def children(self, collection: str, skill_doc_id: str) -> list[dict]:
        """Get child workflow documents for a SKILL.md document.

        A child is any document in the same collection whose ID literally
        starts with the skill's parent path prefix (e.g., Security/Recon/SKILL
        -> Security/Recon/Workflows/*). The prefix is compared literally, not
        with LIKE, so a '_' in a directory name is not a wildcard.
        """
        parent_prefix = skill_doc_id.rsplit("/", 1)[0] + "/"
        rows = self._conn.execute(
            """SELECT id, name, summary FROM documents
               WHERE collection = ? AND substr(id, 1, ?) = ? AND id != ?""",
            (collection, len(parent_prefix), parent_prefix, skill_doc_id),
        ).fetchall()
        return [{"id": r["id"], "name": r["name"], "summary": r["summary"]} for r in rows]

    def delete_stale(self, collection: str, keep_paths: set[str]) -> int:
        """Delete documents whose file_path is not in keep_paths. Returns count removed.

        One transaction, rolled back on any exception before re-raising.
        """
        rows = self._conn.execute(
            "SELECT pk, file_path FROM documents WHERE collection = ?",
            (collection,),
        ).fetchall()
        removed = 0
        try:
            for row in rows:
                if row["file_path"] not in keep_paths:
                    self._conn.execute(
                        "DELETE FROM doc_vectors WHERE collection = ? AND rowid = ?",
                        (collection, row["pk"]),
                    )
                    self._conn.execute("DELETE FROM documents WHERE pk = ?", (row["pk"],))
                    removed += 1
            if removed:
                self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return removed

    def delete_collection(self, collection: str) -> int:
        """Delete a collection's vector rows, document rows and its own row.

        Returns the number of documents removed. One transaction, rolled back
        on any exception before re-raising.
        """
        rows = self._conn.execute(
            "SELECT pk FROM documents WHERE collection = ?",
            (collection,),
        ).fetchall()
        try:
            for row in rows:
                self._conn.execute(
                    "DELETE FROM doc_vectors WHERE collection = ? AND rowid = ?",
                    (collection, row["pk"]),
                )
                self._conn.execute("DELETE FROM documents WHERE pk = ?", (row["pk"],))
            self._conn.execute("DELETE FROM collections WHERE id = ?", (collection,))
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return len(rows)

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
