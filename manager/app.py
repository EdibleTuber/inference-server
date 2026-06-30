# manager/app.py
"""
FastAPI application for the model manager.

Provides HTTP endpoints to proxy inference requests to llama-server,
with automatic model swapping and a FIFO queue for serial GPU access.

Endpoints:
    GET  /health                              - Always 200, liveness check
    GET  /status                              - Server state, model, GPU, queue info
    GET  /v1/models                           - List available GGUFs in OpenAI format
    POST /v1/chat/completions                 - Proxy to llama-server (with model swap)
    POST /v1/embeddings                       - Proxy to embeddings server
    GET  /collections                         - List registered collections
    POST /collections/{collection_id}/search  - Semantic search within a collection
    GET  /collections/{collection_id}/docs/{doc_id:path} - Get full document by ID
    POST /collections/{collection_id}/reindex               - Trigger an incremental reindex (optional body {paths: [...]} limits scope)
    GET  /collections/{collection_id}/reindex/status        - Current or most recent reindex job for a collection
    GET  /collections/{collection_id}/reindex/{job_id}      - Specific reindex job state
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from manager.config import ManagerConfig
from manager.embeddings import EmbeddingsClient
from manager.gpu import get_gpu_info
from manager.names import display_name, match_key, same_model
from manager.queue import RequestQueue
from manager.reindex_jobs import ReindexRegistry
from manager.slots import SlotState
from manager.swap import ModelSwapper
from manager.vectordb import VectorDB
from manager.collections import index_all_collections, index_collection, get_children
from manager.routing import resolve_slot

logger = logging.getLogger(__name__)

_start_time = time.time()


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

class ServerState:
    """Holds all mutable state for the model manager."""

    def __init__(self, config: ManagerConfig):
        self._config = config

        # Build slots dict.
        self.slots: dict[str, SlotState] = {
            "main": SlotState(
                name="main",
                host=config.llama_server_host,
                port=config.llama_server_port,
                env_file=config.llama_server_env,
                systemd_unit=config.llama_server_unit,
                queue=RequestQueue(max_size=config.queue_limit),
            ),
            "batch": SlotState(
                name="batch",
                host=config.batch_server_host,
                port=config.batch_server_port,
                env_file=config.batch_server_env,
                systemd_unit=config.batch_server_unit,
                queue=RequestQueue(max_size=config.batch_queue_limit),
            ),
        }

        # One swapper per slot. Attach dynamically to avoid circular import
        # (SlotState does not reference ModelSwapper).
        self.slots["main"].swapper = ModelSwapper(config, slot=self.slots["main"])
        self.slots["batch"].swapper = ModelSwapper(config, slot=self.slots["batch"])

        # Collection retrieval (unchanged).
        self.db: VectorDB | None = None
        self.embeddings_client: EmbeddingsClient | None = None

        self.reindex_registry = ReindexRegistry()

    # ------------------------------------------------------------------
    # Model path helpers
    # ------------------------------------------------------------------

    def model_path(self, model_name: str) -> str | None:
        """Resolve a model name to its file path in MODELS_DIR.

        Case- and .gguf-suffix-insensitive. An exact-case stem match wins;
        otherwise a case-folded (match_key) match is used. NOTE: the glob is
        case-sensitive, so on-disk files MUST use a lowercase '.gguf' extension.
        """
        requested = display_name(model_name)
        if requested is None:
            return None
        models_dir = Path(self._config.models_dir)
        if not models_dir.exists():
            return None
        by_key: dict[str, Path] = {}
        for p in sorted(models_dir.glob("*.gguf")):
            if p.stem == requested:                 # exact-case match wins
                return str(p)
            key = match_key(p.stem)
            if key in by_key:
                logger.warning(
                    "model_path: case-collision on %r; keeping %s, ignoring %s",
                    key, by_key[key].name, p.name,
                )
                continue
            by_key[key] = p
        match = by_key.get(match_key(model_name))
        return str(match) if match is not None else None

    def list_models(self) -> list[str]:
        models_dir = Path(self._config.models_dir)
        if not models_dir.exists():
            return []
        return sorted(p.stem for p in models_dir.glob("*.gguf"))

    # ------------------------------------------------------------------
    # Model swap
    # ------------------------------------------------------------------

    async def ensure_model_on_slot(self, slot_name: str, model_name: str) -> bool:
        """Ensure model_name is loaded on the named slot.

        Holds the slot's swap_lock for the duration. On failure, drains
        the slot's queue with an error.
        """
        slot = self.slots[slot_name]
        async with slot.swap_lock:
            if slot.healthy and same_model(model_name, slot.loaded_model):
                return True

            path = self.model_path(model_name)
            if path is None:
                msg = f"Model file not found: {model_name}"
                slot.mark_unhealthy()
                self._drain_slot_queue_with_error(slot_name, msg)
                return False

            logger.info("slot=%s swapping to %s", slot_name, model_name)
            success = await slot.swapper.swap_to(path)
            if success:
                slot.mark_swapped(display_name(path))   # store the real on-disk stem
                return True

            msg = f"Model swap timed out on slot {slot_name}: {model_name}"
            slot.mark_unhealthy()
            self._drain_slot_queue_with_error(slot_name, msg)
            return False

    def _drain_slot_queue_with_error(self, slot_name: str, message: str) -> None:
        """Drain the given slot's queue, signaling each item with an error."""
        slot = self.slots[slot_name]
        items = slot.queue.drain()
        for item in items:
            item["error"] = message
            item["event"].set()
        if items:
            logger.warning(
                "slot=%s drained %d queued requests with error: %s",
                slot_name, len(items), message,
            )


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(config: ManagerConfig) -> None:
    """Configure console + file logging."""
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # File handler (skip if /dev/null — used in tests)
    if config.log_file:
        try:
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except (OSError, PermissionError) as exc:
            logger.warning("Could not open log file %s: %s", config.log_file, exc)


# ---------------------------------------------------------------------------
# Background queue consumer helpers
# ---------------------------------------------------------------------------

async def _reprobe_for(slot) -> None:
    """Fire-and-forget re-probe used after a backend 5xx.
    Called via asyncio.create_task so the consumer loop is not blocked.
    """
    async with httpx.AsyncClient() as probe_client:
        await slot.reconcile_on_error(probe_client)


# ---------------------------------------------------------------------------
# Background queue consumer
# ---------------------------------------------------------------------------

async def _queue_consumer(server: "ServerState", config: ManagerConfig, slot_name: str) -> None:
    """Background task: process one slot's queue serially.

    Waits on the slot's queue_event, dispatches items one-by-one through
    ensure_model_on_slot + proxy to the slot's backend URL.
    """
    slot = server.slots[slot_name]
    backend_url = f"{slot.url}/v1/chat/completions"

    while True:
        await slot.queue_event.wait()
        slot.queue_event.clear()

        while not slot.queue.empty():
            item = await slot.queue.dequeue()
            body: dict = item["body"]
            event: asyncio.Event = item["event"]
            model_name: str = body.get("model", "")

            ok = await server.ensure_model_on_slot(slot_name, model_name)
            if not ok:
                if item["error"] is None:
                    item["error"] = f"Model swap failed on slot {slot_name}"
                    event.set()
                continue

            is_streaming = body.get("stream", False)
            try:
                if is_streaming:
                    # Note: streaming errors are out of scope for reconcile;
                    # stream failures have too many non-health-related causes.
                    async def _stream_gen(request_body=body, url=backend_url):
                        async with httpx.AsyncClient() as client:
                            async with client.stream(
                                "POST", url, json=request_body, timeout=None,
                            ) as resp:
                                async for chunk in resp.aiter_bytes():
                                    yield chunk
                    item["response"] = StreamingResponse(
                        _stream_gen(),
                        media_type="text/event-stream",
                    )
                else:
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(backend_url, json=body, timeout=120)
                    if resp.status_code >= 500:
                        slot.mark_unhealthy()
                        asyncio.create_task(_reprobe_for(slot))
                    item["response"] = Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        media_type=resp.headers.get("content-type", "application/json"),
                    )
            except Exception as exc:
                logger.exception("slot=%s proxy error: %s", slot_name, exc)
                item["error"] = f"Proxy error: {exc}"
                slot.mark_unhealthy()
                asyncio.create_task(_reprobe_for(slot))

            event.set()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config: ManagerConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Configuration object. If None, loads from environment.

    Returns:
        Configured FastAPI application instance.
    """
    if config is None:
        config = ManagerConfig.from_env()

    _setup_logging(config)

    server = ServerState(config)

    # ------------------------------------------------------------------
    # Lifespan
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Probe each slot to see if it's already running.
        async with httpx.AsyncClient() as probe_client:
            for name, slot in server.slots.items():
                await slot.probe(probe_client)
                if slot.healthy:
                    logger.info(
                        "slot=%s ready with model %s",
                        name, slot.loaded_model,
                    )
                else:
                    logger.info("slot=%s not reachable at startup", name)

        # Start one background queue consumer per slot.
        consumer_tasks = [
            asyncio.create_task(
                _queue_consumer(server, config, slot_name=name),
                name=f"queue_consumer_{name}",
            )
            for name in server.slots
        ]

        # Initialize collection retrieval if configured.
        if config.skills_db_path:
            try:
                server.db = VectorDB(config.skills_db_path)
                server.db.init_schema()
                server.embeddings_client = EmbeddingsClient(config.embeddings_url)
                await index_all_collections(
                    config.collections_config, server.db, server.embeddings_client,
                )
                logger.info("Collection indexing complete")
            except Exception:
                logger.exception("Collection indexing failed — endpoints will return 503")

        yield

        if server.embeddings_client:
            await server.embeddings_client.close()
        if server.db:
            server.db.close()

        for task in consumer_tasks:
            task.cancel()
        for task in consumer_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # App
    # ------------------------------------------------------------------

    app = FastAPI(title="llama-mgr", lifespan=lifespan)
    app.state.server = server

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # GET /status
    # ------------------------------------------------------------------

    @app.get("/status")
    async def status():
        gpu = get_gpu_info()
        return {
            "slots": {
                name: slot.to_status_dict()
                for name, slot in server.slots.items()
            },
            "gpu": gpu,
            "uptime_seconds": int(time.time() - _start_time),
        }

    # ------------------------------------------------------------------
    # GET /v1/models
    # ------------------------------------------------------------------

    @app.get("/v1/models")
    async def list_models():
        names = server.list_models()
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": now,
                    "owned_by": "local",
                }
                for name in names
            ],
        }

    # ------------------------------------------------------------------
    # POST /v1/chat/completions
    # ------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": "Invalid JSON body"}, status_code=400
            )

        model_name = body.get("model")
        if not model_name:
            return JSONResponse(
                {"error": {"message": "model field is required", "type": "invalid_request_error"}},
                status_code=400,
            )

        # Validate model exists.
        if server.model_path(model_name) is None:
            return JSONResponse(
                {
                    "error": {
                        "message": f"Model not found: {model_name}",
                        "type": "invalid_request_error",
                    }
                },
                status_code=404,
            )

        event = asyncio.Event()
        item: dict = {"body": body, "event": event, "response": None, "error": None}

        slot_name = resolve_slot(model_name, server.slots)
        slot = server.slots[slot_name]

        if not slot.healthy and slot.loaded_model == model_name:
            # Model IS loaded on this slot but the slot is unhealthy.
            # 503 with a typed error the PAL client recognizes.
            #
            # Narrow by design: only fires for the "loaded-but-sick" case.
            # The "not-loaded-anywhere" path (resolve_slot returns 'main'
            # with a different loaded_model) falls through to the queue
            # so the main consumer's ensure_model_on_slot can trigger an
            # implicit swap. That preserves pre-Phase-B behavior.
            return JSONResponse(
                {"error": {
                    "type": f"{slot_name}_unavailable",
                    "message": f"{slot_name} slot not ready",
                }},
                status_code=503,
                headers={"Retry-After": "5"},
            )

        try:
            await slot.queue.enqueue(item)
        except RequestQueue.QueueFullError:
            return JSONResponse(
                {"error": {"message": "Server busy", "type": "server_error"}},
                status_code=503,
                headers={"Retry-After": "5"},
            )

        slot.queue_event.set()
        await event.wait()

        if item["error"]:
            return JSONResponse(
                {"error": {"message": item["error"], "type": "server_error"}},
                status_code=503,
                headers={"Retry-After": "5"},
            )
        return item["response"]

    # ------------------------------------------------------------------
    # POST /v1/embeddings
    # ------------------------------------------------------------------

    @app.post("/v1/embeddings")
    async def embeddings_proxy(request: Request):
        """Proxy embeddings requests to the dedicated embeddings server."""
        if not config.embeddings_url:
            return JSONResponse(
                {"error": "Embeddings server not configured"}, status_code=503
            )
        try:
            body = await request.body()
        except Exception:
            return JSONResponse({"error": "Failed to read request body"}, status_code=400)

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{config.embeddings_url}/v1/embeddings",
                content=body,
                headers={"content-type": request.headers.get("content-type", "application/json")},
                timeout=60,
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    # ------------------------------------------------------------------
    # GET /collections
    # ------------------------------------------------------------------

    @app.get("/collections")
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

    @app.post("/collections/{collection_id}/search")
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

    @app.get("/collections/{collection_id}/docs/{doc_id:path}")
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

    @app.post("/collections/{collection_id}/reindex")
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

        async def _indexer(job_paths):
            return await index_collection(
                collection_id=entry["id"],
                source_dir=entry["source_dir"],
                doc_type=entry["doc_type"],
                db=server.db,
                embeddings=server.embeddings_client,
                paths=job_paths,
            )

        job = await server.reindex_registry.start(
            collection_id=collection_id,
            indexer=_indexer,
            paths=paths,
        )
        return JSONResponse(job.to_dict(), status_code=202)

    # ------------------------------------------------------------------
    # GET /collections/{collection_id}/reindex/status
    # ------------------------------------------------------------------

    @app.get("/collections/{collection_id}/reindex/status")
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
    # POST /swap
    # ------------------------------------------------------------------

    @app.post("/swap")
    async def swap_slot(request: Request):
        """Admin endpoint: swap a slot to a different model.

        Body: {"model": str, "target": "main"|"batch" (optional, default main)}.
        No auth; LAN-only is the trust boundary.
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": {"type": "invalid_body", "message": "Invalid JSON"}},
                status_code=400,
            )

        model_name = body.get("model")
        if not model_name:
            return JSONResponse(
                {"error": {"type": "missing_model", "message": "'model' is required"}},
                status_code=400,
            )

        target = body.get("target", "main")
        if target not in ("main", "batch"):
            return JSONResponse(
                {"error": {"type": "invalid_target",
                           "message": "'target' must be 'main' or 'batch'"}},
                status_code=400,
            )

        if server.model_path(model_name) is None:
            return JSONResponse(
                {"error": {"type": "model_not_found",
                           "message": f"Model file not found: {model_name}"}},
                status_code=404,
            )

        ok = await server.ensure_model_on_slot(target, model_name)
        if not ok:
            return JSONResponse(
                {"error": {"type": "swap_failed",
                           "message": f"swap to {model_name} on {target} failed"}},
                status_code=503,
            )

        return {"slot": target, "model": model_name, "status": "ok"}

    # ------------------------------------------------------------------
    # GET /collections/{collection_id}/reindex/{job_id}
    # ------------------------------------------------------------------

    @app.get("/collections/{collection_id}/reindex/{job_id}")
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

    return app


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    cfg = ManagerConfig.from_env()
    application = create_app(cfg)
    uvicorn.run(application, host=cfg.host, port=cfg.port)
