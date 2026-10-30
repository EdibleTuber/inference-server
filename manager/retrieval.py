# manager/retrieval.py
"""
Collection retrieval (Q28: moved out of app.py so a pure move is reviewable).

Holds the six /collections routes (make_router) and the indexing
orchestration that app.py's lifespan used to own: start_retrieval runs the
startup block (initialize DB and embeddings client, start the background
index jobs, launch the orphaned-collection prune task) and stop_retrieval
runs the shutdown block in the same order as before.
"""
import asyncio
import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from manager.collections import index_collection, load_collections_config, get_children
from manager.config import ManagerConfig
from manager.embeddings import EmbeddingsClient
from manager.vectordb import VectorDB

if TYPE_CHECKING:
    from manager.app import ServerState

logger = logging.getLogger(__name__)


def _collection_indexer(server, entry: dict):
    """Build the `async (paths) -> stats` indexer for one collection entry.

    Used by both startup (lifespan) and the POST /reindex endpoint, so the
    two paths share one definition.
    """

    async def indexer(paths):
        return await index_collection(
            collection_id=entry["id"],
            source_dir=entry["source_dir"],
            doc_type=entry["doc_type"],
            db=server.db,
            embeddings=server.embeddings_client,
            paths=paths,
        )

    return indexer


async def _prune_orphaned_collections(server, keep: set[str]) -> None:
    """Delete collections the collections config no longer lists.

    Waits for every startup index job first (move first, delete after): a
    document is only removed once its replacement under a still-configured
    collection has been indexed. wait_idle waits for jobs to finish, not to
    succeed, so if any configured collection's startup pass failed, its
    documents may still sit only under the orphan: the prune is then skipped
    and logged. A missing, empty or unreadable config never starts this task,
    so it can never prune everything.
    """
    await server.reindex_registry.wait_idle()
    try:
        failed = []
        for cid in keep:
            job = server.reindex_registry.get_current(cid)
            if (
                job is None
                or job.status != "done"
                or job.stats.get("errors", 0) > 0
            ):
                failed.append(cid)
        if failed:
            logger.warning(
                "prune skipped: startup index pass failed for %s — nothing was pruned",
                ", ".join(sorted(failed)),
            )
            return
        for c in server.db.list_collections():
            if c["id"] not in keep:
                removed = server.db.delete_collection(c["id"])
                logger.info(
                    "pruned collection '%s' (%d documents) — not in the collections config",
                    c["id"], removed,
                )
    except Exception:
        logger.exception("collection prune failed")


async def start_retrieval(server: "ServerState", config: ManagerConfig) -> asyncio.Task | None:
    """Initialize collection retrieval (moved from app.py's lifespan, Q28).

    Returns the orphaned-collection prune task, or None.
    """
    # Initialize collection retrieval if configured. Indexing runs in the
    # background so a restart never blocks inference startup (S3); the
    # startup job shows up in GET /collections/{id}/reindex/status.
    prune_task = None
    if config.skills_db_path:
        try:
            server.db = VectorDB(config.skills_db_path)
            server.db.init_schema()
            server.embeddings_client = EmbeddingsClient(config.embeddings_url)
        except Exception:
            server.db = None
            logger.exception(
                "Collection retrieval initialization failed — "
                "collection endpoints will return 503"
            )
        if server.db is not None:
            config_read = False
            try:
                entries = load_collections_config(config.collections_config)
                config_read = True
            except Exception:
                logger.exception(
                    "Could not read the collections config — "
                    "serving the existing index"
                )
                entries = []
            for entry in entries:
                try:
                    # Register the collection first so the endpoints
                    # recognize it immediately, even before indexing
                    # finishes.
                    server.db.upsert_collection(
                        entry["id"], entry["source_dir"], entry["doc_type"],
                    )
                    await server.reindex_registry.start(
                        collection_id=entry["id"],
                        indexer=_collection_indexer(server, entry),
                        paths=None,
                    )
                except Exception:
                    logger.exception(
                        "Could not start indexing collection '%s' — "
                        "serving the existing index", entry["id"],
                    )
            # Prune collections the config no longer lists, but only when
            # the config was read successfully and names at least one
            # collection: a missing, empty or unreadable config must
            # leave the index untouched.
            if config_read and entries:
                prune_task = asyncio.create_task(
                    _prune_orphaned_collections(server, {e["id"] for e in entries}),
                    name="prune_orphaned_collections",
                )
    return prune_task


async def stop_retrieval(server: "ServerState", prune_task: asyncio.Task | None) -> None:
    """Close collection retrieval (moved from app.py's lifespan, Q28)."""
    # A pending prune waits on the reindex registry, so cancel and await
    # it before that registry (and the DB) closes.
    if prune_task is not None:
        prune_task.cancel()
        await asyncio.gather(prune_task, return_exceptions=True)

    # Cancel any running reindex jobs before closing what they use.
    await server.reindex_registry.shutdown()
    if server.embeddings_client:
        await server.embeddings_client.close()
    if server.db:
        server.db.close()


def make_router(server: "ServerState") -> APIRouter:
    """The six /collections routes (moved from app.py, Q28).

    Route order is load-bearing: /reindex/status must be registered before
    /reindex/{job_id}, which tests/test_collection_endpoints.py pins.
    """
    router = APIRouter()

    # ------------------------------------------------------------------
    # GET /collections
    # ------------------------------------------------------------------

    @router.get("/collections")
    async def list_collections_endpoint():
        """List all registered document collections."""
        if server.db is None:
            return JSONResponse(
                {"error": "Collection retrieval not configured"}, status_code=503
            )
        collections = server.db.list_collections()
        return {"collections": collections}

    # ------------------------------------------------------------------
    # POST /collections/{collection_id}/search
    # ------------------------------------------------------------------

    @router.post("/collections/{collection_id}/search")
    async def search_collection(collection_id: str, request: Request):
        """Semantic search within a collection."""
        if server.db is None or server.embeddings_client is None:
            return JSONResponse(
                {"error": "Collection retrieval not configured"}, status_code=503
            )

        # Verify collection exists
        all_collections = server.db.list_collections()
        if not any(c["id"] == collection_id for c in all_collections):
            return JSONResponse(
                {"error": f"Collection not found: {collection_id}"}, status_code=404
            )

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        query = body.get("query", "")
        if not query:
            return JSONResponse({"error": "query field is required"}, status_code=400)

        limit = body.get("limit", 5)
        tags = body.get("tags")

        try:
            query_embedding = await server.embeddings_client.embed_text(query)
        except Exception as exc:
            logger.exception("Failed to embed query: %s", exc)
            return JSONResponse(
                {"error": "Failed to generate query embedding"}, status_code=503
            )

        results = server.db.search(collection_id, query_embedding, limit=limit, tags=tags)

        # For SKILL documents, attach children (workflows)
        for result in results:
            doc_id = result["id"]
            if doc_id.endswith("/SKILL"):
                result["children"] = get_children(server.db, collection_id, doc_id)

        return {"results": results}

    # ------------------------------------------------------------------
    # GET /collections/{collection_id}/docs/{doc_id:path}
    # ------------------------------------------------------------------

    @router.get("/collections/{collection_id}/docs/{doc_id:path}")
    async def get_document(collection_id: str, doc_id: str):
        """Get full document content by collection and document ID."""
        if server.db is None:
            return JSONResponse(
                {"error": "Collection retrieval not configured"}, status_code=503
            )

        doc = server.db.get_document(collection_id, doc_id)
        if doc is None:
            return JSONResponse(
                {"error": f"Document not found: {doc_id}"}, status_code=404
            )

        return doc

    # ------------------------------------------------------------------
    # POST /collections/{collection_id}/reindex
    # ------------------------------------------------------------------

    @router.post("/collections/{collection_id}/reindex")
    async def trigger_reindex(collection_id: str, request: Request):
        """Trigger an incremental reindex. Optional body {paths: [...]} limits scope."""
        if server.db is None or server.embeddings_client is None:
            return JSONResponse(
                {"error": "Collection retrieval not configured"}, status_code=503
            )

        all_collections = server.db.list_collections()
        entry = next((c for c in all_collections if c["id"] == collection_id), None)
        if entry is None:
            return JSONResponse(
                {"error": f"Collection not found: {collection_id}"}, status_code=404
            )

        paths: list[str] | None = None
        body_bytes = await request.body()
        if body_bytes:
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
            raw_paths = body.get("paths")
            if raw_paths is not None:
                if not isinstance(raw_paths, list) or not all(isinstance(p, str) for p in raw_paths):
                    return JSONResponse(
                        {"error": "'paths' must be a list of strings"}, status_code=400,
                    )
                paths = list(raw_paths)

        job = await server.reindex_registry.start(
            collection_id=collection_id,
            indexer=_collection_indexer(server, entry),
            paths=paths,
        )
        return JSONResponse(job.to_dict(), status_code=202)

    # ------------------------------------------------------------------
    # GET /collections/{collection_id}/reindex/status
    # ------------------------------------------------------------------

    @router.get("/collections/{collection_id}/reindex/status")
    async def get_reindex_status(collection_id: str):
        """Return the current (or most recent) reindex job for a collection."""
        if server.db is None:
            return JSONResponse(
                {"error": "Collection retrieval not configured"}, status_code=503
            )
        all_collections = server.db.list_collections()
        if not any(c["id"] == collection_id for c in all_collections):
            return JSONResponse(
                {"error": f"Collection not found: {collection_id}"}, status_code=404
            )
        job = server.reindex_registry.get_current(collection_id)
        if job is None:
            return JSONResponse(
                {"error": "No reindex job recorded for this collection"}, status_code=404,
            )
        return job.to_dict()

    # ------------------------------------------------------------------
    # GET /collections/{collection_id}/reindex/{job_id}
    # ------------------------------------------------------------------

    @router.get("/collections/{collection_id}/reindex/{job_id}")
    async def get_reindex_job(collection_id: str, job_id: str):
        """Return a specific reindex job's state, or 404."""
        if server.db is None:
            return JSONResponse(
                {"error": "Collection retrieval not configured"}, status_code=503
            )
        all_collections = server.db.list_collections()
        if not any(c["id"] == collection_id for c in all_collections):
            return JSONResponse(
                {"error": f"Collection not found: {collection_id}"}, status_code=404
            )
        job = server.reindex_registry.get(job_id)
        if job is None or job.collection_id != collection_id:
            return JSONResponse(
                {"error": f"Job not found: {job_id}"}, status_code=404
            )
        return job.to_dict()

    return router
