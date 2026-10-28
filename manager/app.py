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
    POST /v1/messages                         - Anthropic API passthrough (same queue/swap semantics)
    POST /v1/messages/count_tokens            - Anthropic token counting passthrough
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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from manager.config import ManagerConfig
from manager.embeddings import EmbeddingsClient
from manager.gpu import get_gpu_info_async
from manager.names import display_name, match_key, same_model
from manager.queue import RequestQueue
from manager.reindex_jobs import ReindexRegistry
from manager.slots import SlotState
from manager.swap import ModelSwapper, sampling_args_for_model
from manager.vectordb import VectorDB
from manager.collections import index_all_collections, index_collection, get_children
from manager.routing import home_slot, resolve_slot

logger = logging.getLogger(__name__)

_start_time = time.time()

# What ServerState.ready_for can answer. The consumer is the only caller, and
# the only thing it is allowed to do with the answer is proxy (ready) or
# answer an error (wrong_model, unavailable) -- never swap.
Readiness = Literal["ready", "wrong_model", "unavailable"]


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

class ServerState:
    """Holds all mutable state for the model manager."""

    def __init__(self, config: ManagerConfig):
        self._config = config

        # Build slots dict from configuration, in configured order. Order is
        # significant -- it is the routing priority resolve_slot walks --
        # and dict preserves insertion order, so iterating config.slots here
        # is what makes server.slots ordered the same way.
        self.slots: dict[str, SlotState] = {}
        for sc in config.slots:
            self.slots[sc.name] = SlotState(
                name=sc.name,
                host=sc.host,
                port=sc.port,
                env_file=sc.env_file,
                systemd_unit=sc.systemd_unit,
                queue=RequestQueue(max_size=sc.queue_limit),
            )

        # One swapper per slot. Attach dynamically to avoid circular import
        # (SlotState does not reference ModelSwapper).
        for slot in self.slots.values():
            slot.swapper = ModelSwapper(config, slot=slot)

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
    # Readiness (read-only; the queue consumer's only gate)
    # ------------------------------------------------------------------

    async def ready_for(self, slot_name: str, model_name: str) -> Readiness:
        """Whether the named slot can serve model_name right now.

        Read-only with respect to the backend: never swaps, restarts or
        rewrites anything -- that is ensure_model_on_slot's job, reachable
        only from POST /swap. A chat request must not be able to restart a
        slot out from under another request (O1, incident 2026-10-27).

        Holds the slot's swap_lock for the whole body so the
        same_model/healthy decision cannot race an in-flight swap.
        """
        slot = self.slots[slot_name]
        async with slot.swap_lock:
            if not same_model(model_name, slot.loaded_model):
                return "wrong_model"
            if slot.healthy:
                return "ready"
            # Same model but unhealthy: exactly one probe, on a client we own
            # (the lock is ours, so the client never outlives this method).
            try:
                async with httpx.AsyncClient() as probe_client:
                    await slot.probe(probe_client)
            except Exception:
                # The probe contract says it never raises; if it somehow
                # does, this method's swap_lock context releases the lock and
                # the answer is unavailable -- never a wedged slot.
                logger.exception("slot=%s readiness probe raised", slot_name)
            if slot.healthy and same_model(model_name, slot.loaded_model):
                return "ready"
            return "unavailable"

    # ------------------------------------------------------------------
    # Model swap
    # ------------------------------------------------------------------

    async def _reprobe_slot_if_unlocked(self, name: str, slot, probe_client) -> None:
        """Probe one slot unless a swap already holds its lock.

        Shared by reprobe_all_slots and reprobe_all_slots_concurrently so both
        keep exactly the same skip-if-mid-swap rule; only the fan-out (loop vs
        gather) differs between them.
        """
        if slot.swap_lock.locked():
            logger.debug("slot=%s mid-swap, skipping reprobe", name)
            return
        async with slot.swap_lock:
            await slot.probe(probe_client)

    async def reprobe_all_slots(self) -> None:
        """Refresh every slot's view of its backend, one at a time. Never raises.

        Probing otherwise happens only at startup and after a backend 5xx, so a
        backend that became ready LATER is invisible for the life of the
        process. Observed on the live server: llama-manager and
        llama-server-batch both started at 23:36:45 -- the same second -- so the
        startup probe ran before the iGPU had finished loading a 4B model. The
        batch backend was serving a model this manager reported as
        loaded_model: null, and every request for it got a 409 telling the
        caller to load a model that was already loaded.

        Takes each slot's swap_lock, matching _reprobe_for: between the two
        writers of loaded_model, a swap is authoritative and a probe must never
        clobber a fresh mark_swapped.

        A slot whose lock is already held is SKIPPED rather than waited on. A
        swap in flight is both authoritative and already producing fresh state,
        and blocking here would stall the caller behind a whole model load.

        Sequential: worst case is N slots times the per-slot probe timeout
        (see slot.probe) if every slot is unreachable. That is deliberate
        and unchanged here -- this is the chat 409 path's reprobe, called
        for one request already in flight that is about to fail anyway, so
        a few extra seconds of serialized, easy-to-reason-about probing
        costs little. GET /status calls the concurrent variant below
        instead; see its docstring for why.
        """
        async with httpx.AsyncClient() as probe_client:
            for name, slot in self.slots.items():
                await self._reprobe_slot_if_unlocked(name, slot, probe_client)

    async def reprobe_all_slots_concurrently(self) -> None:
        """Refresh every slot's view of its backend, all at once. Never raises.

        Used only by GET /status. reprobe_all_slots (above) is sequential by
        design for the chat 409 path, where that's fine. /status is a
        monitoring endpoint: its worst case matters most exactly when slots
        are down, since that's when someone is anxiously checking it or an
        automated health check is polling it on a tight timeout. Sequential
        probing there would cost up to N * (the per-slot probe timeout) if
        every slot is unreachable -- multiplying the outage's visible cost
        instead of just reporting it. Probing concurrently bounds /status's
        worst case to one probe timeout regardless of slot count.

        Same skip-if-mid-swap rule as reprobe_all_slots, via
        _reprobe_slot_if_unlocked; only the fan-out is different. Slots are
        probed with one shared AsyncClient but each against its own
        swap_lock, so there is no cross-slot contention from running them
        concurrently.

        return_exceptions=True is load-bearing, not decorative. Without it,
        asyncio.gather re-raises the first child exception immediately,
        without awaiting or cancelling its siblings, and that exception
        would propagate straight out of this method -- past nothing that
        would catch it -- and 500 /status. That is the real, every-run
        production symptom of dropping this flag: confirmed by testing
        the un-forced, naturally-occurring interleaving (main.probe
        raising, batch.probe untouched, real backend delays of 0/50/300ms):
        9/9 runs gave /status a 500, with batch's own state left
        untouched (asyncio.gather's ensure_future queues both slots' first
        step before either completes, so in practice a sibling already
        reaches client.get() and suspends inside connection setup before
        this method's `async with httpx.AsyncClient()` block would exit
        and close the shared client -- so the natural race here does not
        corrupt a sibling's health, only crashes the endpoint).

        Guarding against a sibling being corrupted anyway is still the
        right design, not overkill: it is one `async with` block away from
        happening (e.g. if a slower probe start, more slots, or a
        different scheduler ever changes that interleaving), and
        `tests/test_endpoints.py::test_status_survives_a_raising_slot_probe`
        deliberately forces that interleaving to pin it as a property,
        rather than relying on it never mattering. slot.probe() itself
        never raises (see its docstring), so return_exceptions=True is
        defense in depth against a probe somehow still raising either
        way -- gather waits for every child regardless, so the client is
        never closed while a sibling might still be using it, and any
        exception a child does produce is logged and otherwise ignored
        rather than propagated to the /status caller.
        """
        async with httpx.AsyncClient() as probe_client:
            results = await asyncio.gather(
                *(
                    self._reprobe_slot_if_unlocked(name, slot, probe_client)
                    for name, slot in self.slots.items()
                ),
                return_exceptions=True,
            )
            for name, result in zip(self.slots, results):
                if isinstance(result, BaseException):
                    logger.warning(
                        "slot=%s reprobe raised during concurrent /status "
                        "reprobe (should not happen; slot.probe is expected "
                        "to never raise): %s", name, result,
                    )

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

            sampling_args = sampling_args_for_model(path)
            logger.info("slot=%s swapping to %s", slot_name, model_name)
            msg: str | None = None
            try:
                success = await slot.swapper.swap_to(path, sampling_args)
            except Exception as exc:
                # A raising swap is a failed swap, not a crash: the same
                # failure tail as the timeout case (unhealthy, drained
                # queue, False) so no caller ever sees an exception here.
                logger.exception("slot=%s swap to %s raised", slot_name, model_name)
                success = False
                msg = f"Model swap failed on slot {slot_name}: {model_name}: {exc}"
            if success:
                slot.mark_swapped(display_name(path))   # store the real on-disk stem
                return True

            if msg is None:
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

    Acquires the slot's swap_lock first, so the reprobe serializes AFTER any
    in-flight/queued swap and can never clobber a fresh mark_swapped. Between
    the two writers of loaded_model, the swap is authoritative.
    """
    async with slot.swap_lock:
        async with httpx.AsyncClient() as probe_client:
            await slot.reconcile_on_error(probe_client)


def _stream_close(ctx, client):
    """Build the single idempotent closer for an opened backend stream.

    Exits the stream context (closes the backend stream), then closes the
    client that owns it. Idempotent because the relay calls it from BOTH the
    generator's finally and its background task.
    """
    state = {"closed": False}

    async def close() -> None:
        if state["closed"]:
            return
        state["closed"] = True
        try:
            await ctx.__aexit__(None, None, None)
        finally:
            await client.aclose()

    return close


def _relay_stream(resp, close) -> StreamingResponse:
    """Relay an already-opened backend stream (status < 400) to the client.

    `close` runs from the generator's finally and again from the background
    task; it is idempotent, so every path closes the backend stream: normal
    end, client gone mid-stream (generator aborted -> finally), and client
    gone before the first chunk (the generator never starts, so the
    background task is the only closer).
    """

    async def gen():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await close()

    return StreamingResponse(
        gen(),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "text/event-stream"),
        background=BackgroundTask(close),
    )


# ---------------------------------------------------------------------------
# Background queue consumer
# ---------------------------------------------------------------------------

async def _queue_consumer(server: "ServerState", config: ManagerConfig, slot_name: str) -> None:
    """Background task: process one slot's queue serially.

    Waits on the slot's queue_event, dispatches items one-by-one through
    a readiness check (ready_for -- read-only, never swaps or restarts)
    + proxy to the slot's backend URL. A request whose model was swapped
    out while it sat in the queue gets the 409 a fresh request would
    (O1: chat requests never swap a slot).

    Every dequeued item is answered exactly once (response or error) and
    its event set exactly once: an Exception on any per-item step is
    answered as a 503 queue_error for that item and the loop continues,
    while cancellation (asyncio.CancelledError) still propagates and
    cancels the consumer.

    The backend PATH is per-item: each endpoint stamps its own
    item["backend_path"] when enqueuing, since OpenAI chat and Anthropic
    messages are proxied by the same consumer to different backend
    endpoints. The default keeps any un-stamped item on the OpenAI path.
    """
    slot = server.slots[slot_name]

    while True:
        await slot.queue_event.wait()
        slot.queue_event.clear()

        while not slot.queue.empty():
            item = await slot.queue.dequeue()
            event: asyncio.Event = item["event"]

            try:
                body: dict = item["body"]
                model_name: str = body.get("model", "")
                backend_url = f"{slot.url}/{item.get('backend_path', 'v1/chat/completions')}"

                readiness = await server.ready_for(slot_name, model_name)
                if readiness == "wrong_model":
                    item["error"] = _not_loaded_message(model_name, server.slots)
                    item["error_code"] = "model_not_loaded"
                    event.set()
                    continue
                if readiness == "unavailable":
                    item["error"] = f"{slot_name} slot not ready"
                    item["error_code"] = "slot_unavailable"
                    event.set()
                    continue

                is_streaming = body.get("stream", False)
                try:
                    if is_streaming:
                        # The consumer opens the backend stream itself so it
                        # can see the status before the first byte: a 4xx/5xx
                        # is passed through as a plain error response, only a
                        # <400 is relayed (A2). Streaming errors stay out of
                        # scope for reconcile, except 5xx like non-streaming.
                        client = httpx.AsyncClient()
                        try:
                            ctx = client.stream(
                                "POST", backend_url, json=body,
                                timeout=httpx.Timeout(
                                    config.proxy_read_timeout, connect=10.0,
                                ),
                            )
                            resp = await ctx.__aenter__()
                        except Exception:
                            await client.aclose()
                            raise
                        close = _stream_close(ctx, client)
                        if resp.status_code >= 400:
                            try:
                                content = await resp.aread()
                            finally:
                                await close()
                            if resp.status_code >= 500:
                                slot.mark_unhealthy()
                                asyncio.create_task(_reprobe_for(slot))
                            item["response"] = Response(
                                content=content,
                                status_code=resp.status_code,
                                media_type=resp.headers.get(
                                    "content-type", "application/json",
                                ),
                            )
                        else:
                            item["response"] = _relay_stream(resp, close)
                    else:
                        async with httpx.AsyncClient() as client:
                            resp = await client.post(
                                backend_url, json=body,
                                timeout=httpx.Timeout(
                                    config.proxy_read_timeout, connect=10.0,
                                ),
                            )
                        if resp.status_code >= 500:
                            slot.mark_unhealthy()
                            asyncio.create_task(_reprobe_for(slot))
                        item["response"] = Response(
                            content=resp.content,
                            status_code=resp.status_code,
                            media_type=resp.headers.get("content-type", "application/json"),
                        )
                except (httpx.ReadTimeout, httpx.WriteTimeout) as exc:
                    # A slow but healthy backend: 504 upstream_timeout, the
                    # slot stays healthy and no reprobe is scheduled (F4).
                    # ConnectTimeout is deliberately NOT caught here: a
                    # backend that won't accept connections is unhealthy.
                    logger.warning("slot=%s upstream timeout: %s", slot_name, exc)
                    item["error"] = (
                        f"{slot_name} did not respond within "
                        f"{config.proxy_read_timeout}s"
                    )
                    item["error_code"] = "upstream_timeout"
                except Exception as exc:
                    logger.exception("slot=%s proxy error: %s", slot_name, exc)
                    item["error"] = f"Proxy error: {exc}"
                    slot.mark_unhealthy()
                    asyncio.create_task(_reprobe_for(slot))

            except Exception as exc:
                # A per-item failure must never end the consumer (F2 part 2):
                # answer this one item as a 503 queue_error and move on.
                # asyncio.CancelledError is a BaseException, so a cancelled
                # consumer still propagates and dies.
                logger.exception("slot=%s internal error: %s", slot_name, exc)
                item["error"] = f"Internal error on slot {slot_name}: {exc}"
                item["error_code"] = "queue_error"
                event.set()
                continue

            event.set()


# ---------------------------------------------------------------------------
# Shared routing/enqueue core for the inference endpoints
# ---------------------------------------------------------------------------

@dataclass
class _RouteDecision:
    """Outcome of validating, routing and (for queued endpoints) waiting on a
    request.

    On success `response` holds the backend response built by the queue
    consumer and `status` is None. On failure `status`/`code`/`message`
    describe the error dialect-neutrally and the calling endpoint renders it
    in its own API's error shape (see the two error shapers below), so the
    OpenAI and Anthropic paths reject the same requests for the same reasons.
    """
    status: int | None = None
    code: str | None = None
    message: str = ""
    slot_name: str | None = None
    response: Response | None = None


def _not_loaded_message(model_name: str, slots: dict[str, SlotState]) -> str:
    """The 409 'not loaded on any slot' message, shared by the routing path
    (a fresh request) and the queue consumer (a request routed before its
    model was swapped out): WHAT IS loaded where, so the mismatch is
    self-evident, not just 'not loaded', plus the remedy."""
    loaded_desc = ", ".join(
        f"{name}='{s.loaded_model or 'none'}'"
        for name, s in slots.items()
    )
    return (
        f"model '{model_name}' not loaded on any slot "
        f"(loaded: {loaded_desc}); POST /swap to load it, "
        f"or request the model that is already loaded"
    )


def _decision_from_item_error(item: dict, slot_name: str) -> _RouteDecision:
    """Map a queued item's error onto a _RouteDecision.

    `error_code` None or "queue_error" (e.g. items drained by a failed
    swap) is a 503 queue_error; "slot_unavailable" is the typed 503;
    "upstream_timeout" is the 504 for a slow-but-healthy backend (F4);
    "model_not_loaded" is the 409 for a request whose model was swapped
    out while it was queued. The error shapers render these unchanged.
    """
    code = item.get("error_code")
    if code == "model_not_loaded":
        status, code = 409, "model_not_loaded"
    elif code == "slot_unavailable":
        status, code = 503, "slot_unavailable"
    elif code == "upstream_timeout":
        status, code = 504, "upstream_timeout"
    else:  # None or "queue_error"
        status, code = 503, "queue_error"
    return _RouteDecision(
        status=status, code=code, message=item["error"], slot_name=slot_name,
    )


async def _validate_and_route(
    server: "ServerState", body: dict,
) -> tuple[_RouteDecision | None, str | None]:
    """Validate the body and resolve the target slot, dialect-neutrally.

    Shared by every inference endpoint (queued or not). Returns
    (decision, None) on failure or (None, slot_name) on success.
    """
    model_name = body.get("model")
    if not model_name:
        return _RouteDecision(
            status=400, code="missing_model", message="model field is required"
        ), None

    # Validate model exists.
    if server.model_path(model_name) is None:
        return _RouteDecision(
            status=404, code="model_not_found",
            message=f"Model not found: {model_name}",
        ), None

    slot_name = resolve_slot(model_name, server.slots)
    if slot_name is None:
        # Our view of the slots may simply be stale -- see
        # reprobe_all_slots. Ask the backends before telling the caller to
        # load something that may already be loaded.
        await server.reprobe_all_slots()
        slot_name = resolve_slot(model_name, server.slots)
    if slot_name is None:
        # Loaded on neither slot. Do NOT implicitly restart main; tell the
        # caller WHAT IS loaded so the mismatch is self-evident, not just
        # "not loaded". (Implicit swaps live only on POST /swap.)
        return _RouteDecision(
            status=409,
            code="model_not_loaded",
            message=_not_loaded_message(model_name, server.slots),
        ), None

    slot = server.slots[slot_name]
    if not slot.healthy and same_model(model_name, slot.loaded_model):
        # Model IS loaded on this slot but the slot is unhealthy. Our view
        # may be stale -- the backend may have recovered, or even loaded a
        # different model, since the last probe (F6) -- so reprobe once
        # before answering. A slot mid-swap (lock held) is skipped by
        # _reprobe_slot_if_unlocked, so this path never waits behind a swap.
        async with httpx.AsyncClient() as probe_client:
            await server._reprobe_slot_if_unlocked(slot_name, slot, probe_client)
        if slot.healthy and same_model(model_name, slot.loaded_model):
            # The probe confirms it: route normally.
            return None, slot_name
        if not same_model(model_name, slot.loaded_model):
            # The probe reports a different loaded model: 409, exactly like a
            # fresh request for a model that isn't loaded.
            return _RouteDecision(
                status=409,
                code="model_not_loaded",
                message=_not_loaded_message(model_name, server.slots),
            ), None
        # Still unhealthy on the same model: typed 503.
        return _RouteDecision(
            status=503, code="slot_unavailable",
            message=f"{slot_name} slot not ready", slot_name=slot_name,
        ), None

    return None, slot_name


async def _route_and_wait(
    server: "ServerState", body: dict, backend_path: str,
) -> _RouteDecision:
    """Validate, route, enqueue and wait -- the core of the queued endpoints.

    `backend_path` is stamped on the queued item so the slot's consumer
    proxies it to the right backend endpoint (see _queue_consumer).
    """
    decision, slot_name = await _validate_and_route(server, body)
    if decision is not None:
        return decision

    event = asyncio.Event()
    item: dict = {
        "body": body, "event": event, "response": None, "error": None,
        "error_code": None, "backend_path": backend_path,
    }
    slot = server.slots[slot_name]

    try:
        await slot.queue.enqueue(item)
    except RequestQueue.QueueFullError:
        return _RouteDecision(
            status=503, code="queue_full", message="Server busy",
            slot_name=slot_name,
        )

    slot.queue_event.set()
    await event.wait()

    if item["error"]:
        return _decision_from_item_error(item, slot_name)
    return _RouteDecision(response=item["response"])


def _openai_error_response(decision: _RouteDecision) -> JSONResponse:
    """Render a _RouteDecision failure in /v1/chat/completions's error shape
    (pre-existing, unchanged)."""
    if decision.code == "slot_unavailable":
        err = {"type": f"{decision.slot_name}_unavailable", "message": decision.message}
    elif decision.code in ("queue_full", "queue_error"):
        err = {"message": decision.message, "type": "server_error"}
    elif decision.code == "model_not_loaded":
        err = {"type": "model_not_loaded", "message": decision.message}
    elif decision.code == "upstream_timeout":
        err = {"type": "upstream_timeout", "message": decision.message}
    else:  # missing_model, model_not_found
        err = {"message": decision.message, "type": "invalid_request_error"}
    headers = {"Retry-After": "5"} if decision.status == 503 else None
    return JSONResponse({"error": err}, status_code=decision.status, headers=headers)


def _anthropic_error_response(decision: _RouteDecision) -> JSONResponse:
    """Render a _RouteDecision failure in the Anthropic error shape:
    {"type": "error", "error": {"type": ..., "message": ...}}, with the inner
    type taken from Anthropic's error taxonomy where one exists (409 "not
    loaded on any slot" has no Anthropic analog and keeps its manager type).
    """
    type_by_code = {
        "invalid_body": "invalid_request_error",
        "missing_model": "invalid_request_error",
        "model_not_found": "not_found_error",
        "model_not_loaded": "model_not_loaded",
        "slot_unavailable": "api_error",
        "queue_full": "overloaded_error",
        "queue_error": "api_error",
        "upstream_timeout": "api_error",
    }
    headers = {"Retry-After": "5"} if decision.status == 503 else None
    return JSONResponse(
        {
            "type": "error",
            "error": {
                "type": type_by_code[decision.code],
                "message": decision.message,
            },
        },
        status_code=decision.status,
        headers=headers,
    )


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
    # Each slot's SLOT_<NAME>_MODELS list, for /status. Static: config is frozen.
    home_models = {sc.name: list(sc.models) for sc in config.slots}

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
        # Refresh before reporting. The startup probe runs once, and until this
        # was added the only other reprobe was on the chat 409 path -- so a
        # backend that became ready later showed here as a total outage for the
        # life of the process, while requests to it succeeded. Uses the
        # concurrent variant (see ServerState.reprobe_all_slots_concurrently)
        # so this endpoint's worst case is one probe timeout, not one per
        # slot; slots mid-swap are still skipped, so this cannot stall behind
        # a model load either way.
        await server.reprobe_all_slots_concurrently()
        gpu = await get_gpu_info_async()
        return {
            "slots": {
                name: {**slot.to_status_dict(), "models": home_models.get(name, [])}
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

        decision = await _route_and_wait(
            server, body, backend_path="v1/chat/completions"
        )
        if decision.response is not None:
            return decision.response
        return _openai_error_response(decision)

    # ------------------------------------------------------------------
    # POST /v1/messages
    # ------------------------------------------------------------------

    @app.post("/v1/messages")
    async def messages(request: Request):
        """Anthropic API passthrough.

        Same validation, routing, queueing and serial-GPU semantics as
        /v1/chat/completions, but the body is proxied to the backend's native
        /v1/messages endpoint (llama-server speaks Anthropic directly) and
        manager-level errors use the Anthropic error shape.
        """
        try:
            body = await request.json()
        except Exception:
            return _anthropic_error_response(
                _RouteDecision(status=400, code="invalid_body",
                               message="Invalid JSON body")
            )

        decision = await _route_and_wait(server, body, backend_path="v1/messages")
        if decision.response is not None:
            return decision.response
        return _anthropic_error_response(decision)

    # ------------------------------------------------------------------
    # POST /v1/messages/count_tokens
    # ------------------------------------------------------------------

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        """Anthropic token counting, direct passthrough.

        Same validation and slot routing as /v1/messages (counting a prompt
        requires the tokenizer of the model that would serve it), but it
        bypasses the GPU queue entirely: it is a cheap local computation on
        the backend, so it never queues, swaps, or blocks behind completions.
        """
        try:
            body = await request.json()
        except Exception:
            return _anthropic_error_response(
                _RouteDecision(status=400, code="invalid_body",
                               message="Invalid JSON body")
            )

        decision, slot_name = await _validate_and_route(server, body)
        if decision is not None:
            return _anthropic_error_response(decision)

        slot = server.slots[slot_name]
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{slot.url}/v1/messages/count_tokens",
                json=body,
                timeout=30,
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

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
    async def swap_slot(request: Request, background_tasks: BackgroundTasks):
        """Admin endpoint: swap a slot to a different model.

        Body: {"model": str, "target": <slot name> (optional)}.
        Without a target (absent or null) the model goes to its home slot from
        SLOT_<NAME>_MODELS; a model with no home slot is refused with 400.
        No auth; LAN-only is the trust boundary.

        Returns 200 if the model is already loaded, 202 if a swap was started.
        Callers poll GET /status (the slot's ``swapping`` and ``loaded_model``
        fields) to know when the swap lands.
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

        valid = ", ".join(repr(n) for n in server.slots)
        target = body.get("target")
        if target is None:
            # Untargeted: the operator's map decides, never "the first slot".
            # agent_core clients send exactly this after a model_not_loaded
            # 409; guessing a slot here is what used to evict main.
            target = home_slot(model_name, config.slots)
            if target is None:
                return JSONResponse(
                    {"error": {"type": "no_home_slot",
                               "message": (
                                   f"model '{model_name}' has no home slot; pass "
                                   f"'target' (one of: {valid}) or list it in "
                                   f"SLOT_<NAME>_MODELS")}},
                    status_code=400,
                )
        elif not isinstance(target, str) or target not in server.slots:
            return JSONResponse(
                {"error": {"type": "invalid_target",
                           "message": f"'target' must be one of: {valid}"}},
                status_code=400,
            )

        if server.model_path(model_name) is None:
            return JSONResponse(
                {"error": {"type": "model_not_found",
                           "message": f"Model file not found: {model_name}"}},
                status_code=404,
            )

        slot = server.slots[target]
        if slot.healthy and same_model(model_name, slot.loaded_model):
            return {"slot": target, "model": slot.loaded_model, "status": "ok"}

        async def _do_swap():
            ok = await server.ensure_model_on_slot(target, model_name)
            if not ok:
                logger.error("background swap failed: %s on slot %s",
                             model_name, target)

        background_tasks.add_task(_do_swap)
        return JSONResponse(
            {"slot": target, "model": model_name, "status": "swapping"},
            status_code=202,
        )

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
