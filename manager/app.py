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
from manager.queue import RequestQueue
from manager.swap import ModelSwapper
from manager.vectordb import VectorDB
from manager.collections import index_all_collections, index_collection, get_children

logger = logging.getLogger(__name__)

_start_time = time.time()


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

class ServerState:
    """Holds all mutable state for the model manager."""

    def __init__(self, config: ManagerConfig):
        self._config = config
        self.state: str = "loading"           # loading | ready | swapping | error
        self.current_model: str | None = None  # name (without .gguf)
        self.loading_model: str | None = None  # name being swapped to
        self.error_message: str | None = None

        self.queue = RequestQueue(max_size=config.queue_limit)
        self.swapper = ModelSwapper(config)
        self.swap_lock = asyncio.Lock()
        self.queue_event = asyncio.Event()

        # Collection retrieval
        self.db: VectorDB | None = None
        self.embeddings_client: EmbeddingsClient | None = None

        # Reindex jobs
        from manager.reindex_jobs import ReindexRegistry
        self.reindex_registry = ReindexRegistry()

    # ------------------------------------------------------------------
    # Model path helpers
    # ------------------------------------------------------------------

    def model_path(self, model_name: str) -> str | None:
        """Resolve a model name to its full GGUF path.

        Accepts names with or without the .gguf suffix.
        Returns None if the file does not exist in models_dir.
        """
        name = model_name if model_name.endswith(".gguf") else f"{model_name}.gguf"
        path = Path(self._config.models_dir) / name
        return str(path) if path.exists() else None

    def list_models(self) -> list[str]:
        """Return sorted list of model names (without .gguf) in models_dir."""
        models_dir = Path(self._config.models_dir)
        if not models_dir.exists():
            return []
        return sorted(p.stem for p in models_dir.glob("*.gguf"))

    # ------------------------------------------------------------------
    # Model swap
    # ------------------------------------------------------------------

    async def ensure_model(self, model_name: str) -> bool:
        """Ensure *model_name* is loaded in llama-server.

        Skips the swap if the requested model is already loaded.
        On failure: transitions to error state, drains the queue,
        notifies all waiting clients with an error, and returns False.
        """
        async with self.swap_lock:
            # Fast path — already loaded.
            if self.state == "ready" and self.current_model == model_name:
                return True

            path = self.model_path(model_name)
            if path is None:
                self.state = "error"
                self.error_message = f"Model file not found: {model_name}"
                self.current_model = None
                self._drain_queue_with_error(self.error_message)
                return False

            logger.info("Swapping to model: %s", model_name)
            self.state = "swapping"
            self.loading_model = model_name
            self.error_message = None

            success = await self.swapper.swap_to(path)

            if success:
                self.state = "ready"
                self.current_model = model_name
                self.loading_model = None
                logger.info("Model ready: %s", model_name)
                return True
            else:
                msg = f"Model swap timed out: {model_name}"
                self.state = "error"
                self.current_model = None
                self.loading_model = None
                self.error_message = msg
                logger.error(msg)
                self._drain_queue_with_error(msg)
                return False

    def _drain_queue_with_error(self, message: str) -> None:
        """Drain all queued items and signal them with an error."""
        items = self.queue.drain()
        for item in items:
            item["error"] = message
            item["event"].set()
        if items:
            logger.warning("Drained %d queued requests with error: %s", len(items), message)


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
# Background queue consumer
# ---------------------------------------------------------------------------

async def _queue_consumer(server: ServerState, config: ManagerConfig) -> None:
    """Background task: process queued inference requests one at a time.

    Waits on queue_event, then drains and processes items serially.
    Streaming responses yield chunks in real-time via an async generator.
    """
    llama_url = f"{config.llama_server_url}/v1/chat/completions"

    while True:
        # Wait until something is enqueued.
        await server.queue_event.wait()
        server.queue_event.clear()

        while not server.queue.empty():
            item = await server.queue.dequeue()
            body: dict = item["body"]
            event: asyncio.Event = item["event"]
            model_name: str = body.get("model", "")

            # Ensure the right model is loaded.
            ok = await server.ensure_model(model_name)
            if not ok:
                # ensure_model already set item["error"] and signalled if it
                # drained. But if this specific item was dequeued before the
                # drain, handle it here.
                if item["error"] is None:
                    item["error"] = server.error_message or "Model swap failed"
                    event.set()
                continue

            is_streaming = body.get("stream", False)

            try:
                if is_streaming:
                    # Build the streaming response now; the generator will be
                    # consumed by FastAPI when the response is sent.
                    async def _stream_gen(request_body=body):
                        async with httpx.AsyncClient() as client:
                            async with client.stream(
                                "POST",
                                llama_url,
                                json=request_body,
                                timeout=None,
                            ) as resp:
                                async for chunk in resp.aiter_bytes():
                                    yield chunk

                    item["response"] = StreamingResponse(
                        _stream_gen(),
                        media_type="text/event-stream",
                    )
                else:
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(llama_url, json=body, timeout=120)
                    item["response"] = Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        media_type=resp.headers.get("content-type", "application/json"),
                    )

            except Exception as exc:
                logger.exception("Error proxying request: %s", exc)
                item["error"] = f"Proxy error: {exc}"

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
        # Check whether llama-server is already running.
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{config.llama_server_url}/health", timeout=3
                )
            if resp.status_code == 200:
                logger.info("llama-server is already running")
                server.state = "ready"
        except Exception:
            logger.info("llama-server not yet reachable at startup")

        # Start background queue consumer.
        consumer_task = asyncio.create_task(
            _queue_consumer(server, config),
            name="queue_consumer",
        )

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

        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # App
    # ------------------------------------------------------------------

    app = FastAPI(title="llama-mgr", lifespan=lifespan)

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
            "state": server.state,
            "current_model": server.current_model,
            "loading_model": server.loading_model,
            "error_message": server.error_message,
            "queue_depth": server.queue.depth,
            "queue_limit": server.queue.max_size,
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

        # Check queue capacity.
        event = asyncio.Event()
        item: dict = {"body": body, "event": event, "response": None, "error": None}

        try:
            await server.queue.enqueue(item)
        except RequestQueue.QueueFullError:
            return JSONResponse(
                {"error": {"message": "Server busy", "type": "server_error"}},
                status_code=503,
                headers={"Retry-After": "5"},
            )

        # Signal the consumer and wait for the result.
        server.queue_event.set()
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
