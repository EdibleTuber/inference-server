# manager/app.py
"""
FastAPI application for the model manager.

Provides HTTP endpoints to proxy inference requests to llama-server,
with automatic model swapping and a FIFO queue for serial GPU access.

Endpoints:
    GET  /health                              - Always 200, liveness check
    GET  /status                              - Server state, model, GPU, queue info
    GET  /v1/models                           - List onboarded profiles in OpenAI format
    POST /v1/chat/completions                 - Proxy to llama-server (with model swap)
    POST /v1/messages                         - Anthropic API passthrough (same queue/swap semantics)
    POST /v1/messages/count_tokens            - Anthropic token counting passthrough
    GET  /collections                         - List registered collections
    POST /collections/{collection_id}/search  - Semantic search within a collection
    GET  /collections/{collection_id}/docs/{doc_id:path} - Get full document by ID
    POST /collections/{collection_id}/reindex               - Trigger an incremental reindex (optional body {paths: [...]} limits scope)
    GET  /collections/{collection_id}/reindex/status        - Current or most recent reindex job for a collection
    GET  /collections/{collection_id}/reindex/{job_id}      - Specific reindex job state
"""
import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import httpx
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from manager import retrieval
from manager.config import ManagerConfig
from manager.request_log import (
    RequestRecord,
    StreamTokens,
    log_swap,
    request_context,
    request_id_from,
    tokens_from_body,
)
from manager.gpu import get_gpu_info_async
from manager.names import same_model
from manager.profiles import ProfileSet, load_profiles, render_model_env
from manager.queue import RequestQueue
from manager.reindex_jobs import ReindexRegistry
from manager.slots import SlotState
from manager.swap import ModelSwapper
from manager.routing import resolve_slot

if TYPE_CHECKING:
    # Named only in ServerState.__init__ annotations; the retrieval module
    # owns the runtime imports (Q28).
    from manager.embeddings import EmbeddingsClient
    from manager.vectordb import VectorDB

logger = logging.getLogger(__name__)

_start_time = time.time()

# What ServerState.ready_for can answer. The consumer is the only caller, and
# the only thing it is allowed to do with the answer is proxy (ready) or
# answer an error (wrong_model, unavailable) -- never swap.
Readiness = Literal["ready", "wrong_model", "unavailable"]

# Token dialect for each backend path the queue consumer proxies, so the
# request line's token fields come from the backend's own body fields.
# count_tokens never queues, so it is not here.
_STREAM_DIALECTS = {
    "v1/chat/completions": "openai",
    "v1/messages": "anthropic",
}


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

class ServerState:
    """Holds all mutable state for the model manager."""

    def __init__(self, config: ManagerConfig, profiles: ProfileSet | None = None):
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
                model_env_file=sc.model_env_file,
            )

        # One swapper per slot. Attach dynamically to avoid circular import
        # (SlotState does not reference ModelSwapper).
        for slot in self.slots.values():
            slot.swapper = ModelSwapper(config, slot=slot)

        # Per-model profiles: the roster a swap renders from and /v1/models
        # lists (Q13, Q29, Q32). A ProfileSet passed in is used as-is;
        # otherwise load_profiles runs and never raises -- a bad file goes
        # into self.profiles.errors and is logged at ERROR, not fatal: one
        # bad profile must not take inference down.
        if profiles is None:
            profiles = load_profiles(
                config.profiles_dir, [sc.name for sc in config.slots], config.models_dir)
        self.profiles = profiles
        for error in self.profiles.errors.values():
            logger.error("profiles: %s", error)

        # Collection retrieval (unchanged).
        self.db: VectorDB | None = None
        self.embeddings_client: EmbeddingsClient | None = None

        self.reindex_registry = ReindexRegistry()

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

    async def ensure_model_on_slot(self, slot_name: str, model_name: str,
                                   *, force: bool = False) -> bool:
        """Ensure model_name is loaded on the named slot.

        Holds the slot's swap_lock for the duration. The model's profile
        section for the slot is rendered into the slot's generated model
        env file, so no flag can carry over from the previous model. A
        healthy slot already holding the model needs no swap unless
        force=True: a trial must really reload the model to verify it.

        A model the profile set cannot render -- no profile, or no section
        for this slot -- is a refusal, not a failed swap: nothing was
        stopped, so the slot's health, loaded model and queue are left
        alone and the old model keeps serving. A swap that starts and then
        fails (raises or times out) took the backend down, so it still
        takes the failure tail: unhealthy slot, drained queue, False.
        """
        slot = self.slots[slot_name]
        async with slot.swap_lock:
            if not force and slot.healthy and same_model(model_name, slot.loaded_model):
                return True

            profile = self.profiles.get(model_name)
            if profile is None:
                logger.error(
                    "No profile for model %s in %s; add a <name>.toml there",
                    model_name, self._config.profiles_dir)
                return False

            section = profile.section(slot_name)
            if section is None:
                logger.error(
                    "Model %s has no [slots.%s] section; it may run on: %s",
                    profile.name, slot_name, ", ".join(profile.sections))
                return False

            logger.info("slot=%s swapping to %s", slot_name, profile.name)
            msg: str | None = None
            try:
                env_text = render_model_env(profile, section, self._config.models_dir)
                success = await slot.swapper.swap_to(profile.name, env_text)
            except Exception as exc:
                # A raising swap is a failed swap, not a crash: the same
                # failure tail as the timeout case (unhealthy, drained
                # queue, False) so no caller ever sees an exception here.
                logger.exception("slot=%s swap to %s raised", slot_name, profile.name)
                success = False
                msg = f"Model swap failed on slot {slot_name}: {profile.name}: {exc}"
            if success:
                slot.mark_swapped(profile.name)
                return True

            if msg is None:
                msg = f"Model swap timed out on slot {slot_name}: {model_name}"
            slot.mark_unhealthy()
            slot.record_error("swap_failed", msg)
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
    """Configure logging: exactly one handler, on stderr, which systemd
    ships to the journal (Q27). journald timestamps every line, so the
    format carries no asctime."""
    formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root.addHandler(handler)


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


def _relay_stream(resp, close, record=None, stream_tokens=None,
                  on_close=None) -> StreamingResponse:
    """Relay an already-opened backend stream (status < 400) to the client.

    `close` runs from the generator's finally and again from the background
    task; it is idempotent, so every path closes the backend stream: normal
    end, client gone mid-stream (generator aborted -> finally), and client
    gone before the first chunk (the generator never starts, so the
    background task is the only closer).

    When the item carries a request record (phase 4, Q27), each relayed
    chunk is also fed, unchanged, to stream_tokens -- the sniffing never
    touches or buffers the bytes -- and the background task, the stream's
    closer, writes the line: ok when the relay reached the end of the
    stream, client_disconnected otherwise. finish is idempotent, so the
    closer and the endpoint can never double-write a relayed stream's line.

    `on_close`, when given, runs from the same background task as the
    closer: a relayed stream stays in flight on its slot until that moment,
    never before the stream closes.
    """
    state = {"completed": False}

    async def gen():
        try:
            async for chunk in resp.aiter_bytes():
                if stream_tokens is not None:
                    stream_tokens.feed(chunk)
                yield chunk
            state["completed"] = True
        finally:
            await close()

    async def finish_stream() -> None:
        try:
            await close()
        finally:
            # The request the relay carries ends here, on every exit path,
            # even if close() raises; the line is written the same way and
            # the exception propagates afterwards.
            if on_close is not None:
                on_close()
            if record is not None:
                if stream_tokens is not None:
                    # The sniffed counts live on the StreamTokens until the
                    # stream ends; hand them to the record so the line has them.
                    record.tokens = stream_tokens.tokens
                record.finish(
                    resp.status_code,
                    "ok" if state["completed"] else "client_disconnected",
                )

    return StreamingResponse(
        gen(),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "text/event-stream"),
        background=BackgroundTask(finish_stream),
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
            record = item.get("record")
            if record is not None:
                # The endpoint stamped enqueued_at; the consumer stamps the
                # dequeue (queue_wait_ms) and, below, backend_started_at.
                record.slot = slot_name
                record.dequeued_at = time.monotonic()

            try:
                body: dict = item["body"]
                model_name: str = body.get("model", "")
                backend_path = item.get("backend_path", "v1/chat/completions")
                backend_url = f"{slot.url}/{backend_path}"
                # Token dialect to sniff the backend body with, keyed off
                # the path the endpoint stamped (count_tokens never
                # queues). An unknown path sniffs nothing.
                dialect = _STREAM_DIALECTS.get(backend_path)

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
                # Q27: the request is in flight from just before the backend
                # call until the response is complete; a request refused
                # before the call (wrong_model, slot_unavailable) never gets
                # here. The record's id when there is one, a generated id
                # otherwise -- an item without a record is still counted and
                # ended.
                request_id = (
                    record.id if record is not None else request_id_from(None)
                )
                slot.begin_request(request_id)
                handed_off = False
                try:
                    if record is not None:
                        # backend_ms runs from the backend call to the line.
                        record.backend_started_at = time.monotonic()
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
                                slot.record_error(
                                    "backend_error",
                                    f"backend returned {resp.status_code} "
                                    f"on {backend_path}",
                                )
                            if record is not None and dialect is not None:
                                record.tokens = tokens_from_body(content, dialect)
                            item["response"] = Response(
                                content=content,
                                status_code=resp.status_code,
                                media_type=resp.headers.get(
                                    "content-type", "application/json",
                                ),
                            )
                        else:
                            stream_tokens = (
                                StreamTokens(dialect)
                                if record is not None and dialect is not None
                                else None
                            )
                            # The relay owns this request's end: its closer
                            # runs when the stream closes, never before. The
                            # default argument pins request_id to THIS item
                            # -- the loop rebinds the name for the next item.
                            item["response"] = _relay_stream(
                                resp, close, record=record,
                                stream_tokens=stream_tokens,
                                on_close=lambda rid=request_id:
                                    slot.end_request(rid),
                            )
                            handed_off = True
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
                            slot.record_error(
                                "backend_error",
                                f"backend returned {resp.status_code} "
                                f"on {backend_path}",
                            )
                        if record is not None and dialect is not None:
                            record.tokens = tokens_from_body(resp.content, dialect)
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
                    slot.record_error("upstream_timeout", item["error"])
                except Exception as exc:
                    logger.exception("slot=%s proxy error: %s", slot_name, exc)
                    item["error"] = f"Proxy error: {exc}"
                    slot.mark_unhealthy()
                    asyncio.create_task(_reprobe_for(slot))
                    slot.record_error("proxy_error", item["error"])
                finally:
                    # Q27: every exit path ends the request, cancellation
                    # included -- CancelledError skips the except clauses
                    # above but not a finally, so a cancelled consumer
                    # cannot leak an entry that would show the slot serving
                    # forever. The one exception is a stream handed to a
                    # relay (handed_off): its closer ends it, never before
                    # the stream closes.
                    if not handed_off:
                        slot.end_request(request_id)

            except Exception as exc:
                # A per-item failure must never end the consumer (F2 part 2):
                # answer this one item as a 503 queue_error and move on.
                # asyncio.CancelledError is a BaseException, so a cancelled
                # consumer still propagates and dies.
                logger.exception("slot=%s internal error: %s", slot_name, exc)
                item["error"] = f"Internal error on slot {slot_name}: {exc}"
                item["error_code"] = "queue_error"
                slot.record_error("queue_error", item["error"])
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

    # Validate the model is in the roster: a GGUF with no profile is not a
    # model this manager knows (Q13), so refuse it before routing.
    if server.profiles.get(model_name) is None:
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
    record: RequestRecord | None = None,
) -> _RouteDecision:
    """Validate, route, enqueue and wait -- the core of the queued endpoints.

    `backend_path` is stamped on the queued item so the slot's consumer
    proxies it to the right backend endpoint (see _queue_consumer).

    When the endpoint passes a request record (phase 4, Q27), it is stamped
    with enqueued_at and travels with the item as item["record"]; the
    consumer stamps the rest and the closer or the endpoint finishes it.
    """
    decision, slot_name = await _validate_and_route(server, body)
    if decision is not None:
        return decision

    event = asyncio.Event()
    item: dict = {
        "body": body, "event": event, "response": None, "error": None,
        "error_code": None, "backend_path": backend_path,
    }
    if record is not None:
        record.enqueued_at = time.monotonic()
        item["record"] = record
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


class JsonOnlyMiddleware:
    """Gate: every POST must declare Content-Type application/json (Q8/X1).

    Deliberately plain ASGI rather than @app.middleware("http"):
    BaseHTTPMiddleware wraps `receive` and re-streams every response,
    which can break how a client disconnect closes the upstream stream,
    and no app-level test would catch that. This gate only inspects
    headers, then either short-circuits with 415 or forwards
    (scope, receive, send) completely unwrapped.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        header = next(
            (v for k, v in scope["headers"] if k == b"content-type"), None
        )
        if header is None:
            media_type = b""
        else:
            media_type = header.split(b";", 1)[0].strip().lower()
        if media_type == b"application/json":
            await self.app(scope, receive, send)
            return
        await JSONResponse(
            {"error": {
                "type": "unsupported_media_type",
                "message": "Content-Type must be application/json",
            }},
            status_code=415,
        )(scope, receive, send)


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
    # Each slot's home models, for /status. Read once at startup from the
    # roster: the ProfileSet is the config's view, and a profile edited
    # later is picked up on restart.
    home_models = server.profiles.home_models()

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

        # Collection retrieval (Q28): the startup block (initialization,
        # background indexing, the orphaned-collection prune task) and the
        # shutdown block both live in manager/retrieval.py.
        prune_task = await retrieval.start_retrieval(server, config)

        yield

        await retrieval.stop_retrieval(server, prune_task)

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
    # Q8 (X1): refuse every POST whose Content-Type is not application/json.
    app.add_middleware(JsonOnlyMiddleware)

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
            "profile_errors": server.profiles.errors,
        }

    # ------------------------------------------------------------------
    # GET /v1/models
    # ------------------------------------------------------------------

    @app.get("/v1/models")
    async def list_models():
        # The onboarded profiles are the model list (Q13): a profile makes
        # the manager aware of a model; a GGUF on disk without one is
        # invisible. loaded_on names the slots whose loaded_model matches.
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": profile.name,
                    "object": "model",
                    "created": now,
                    "owned_by": "local",
                    "home_slot": profile.home,
                    "verified_slots": list(profile.verified_slots()),
                    "loaded_on": [
                        name for name, slot in server.slots.items()
                        if same_model(profile.name, slot.loaded_model)
                    ],
                }
                for profile in server.profiles.onboarded()
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
            body = None
        record = RequestRecord.from_request(request, "/v1/chat/completions", body)
        if body is None:
            response = JSONResponse(
                {"error": "Invalid JSON body"}, status_code=400,
                headers={"x-request-id": record.id},
            )
            record.finish(400, "invalid_body")
            return response

        decision = await _route_and_wait(
            server, body, backend_path="v1/chat/completions", record=record
        )
        if decision.response is not None:
            response = decision.response
            response.headers["x-request-id"] = record.id
            # The endpoint finishes every response except a relayed stream,
            # whose line the stream's closer writes (phase 4, Q27).
            if not isinstance(response, StreamingResponse):
                record.finish(
                    response.status_code,
                    "ok" if response.status_code < 400 else "backend_error",
                )
            return response
        response = _openai_error_response(decision)
        response.headers["x-request-id"] = record.id
        record.finish(decision.status, decision.code)
        return response

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
            body = None
        record = RequestRecord.from_request(request, "/v1/messages", body)
        if body is None:
            response = _anthropic_error_response(
                _RouteDecision(status=400, code="invalid_body",
                               message="Invalid JSON body")
            )
            response.headers["x-request-id"] = record.id
            record.finish(400, "invalid_body")
            return response

        decision = await _route_and_wait(server, body, backend_path="v1/messages",
                                         record=record)
        if decision.response is not None:
            response = decision.response
            response.headers["x-request-id"] = record.id
            # The endpoint finishes every response except a relayed stream,
            # whose line the stream's closer writes (phase 4, Q27).
            if not isinstance(response, StreamingResponse):
                record.finish(
                    response.status_code,
                    "ok" if response.status_code < 400 else "backend_error",
                )
            return response
        response = _anthropic_error_response(decision)
        response.headers["x-request-id"] = record.id
        record.finish(decision.status, decision.code)
        return response

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
            body = None
        record = RequestRecord.from_request(
            request, "/v1/messages/count_tokens", body,
        )
        if body is None:
            response = _anthropic_error_response(
                _RouteDecision(status=400, code="invalid_body",
                               message="Invalid JSON body")
            )
            response.headers["x-request-id"] = record.id
            record.finish(400, "invalid_body")
            return response

        decision, slot_name = await _validate_and_route(server, body)
        if decision is not None:
            response = _anthropic_error_response(decision)
            response.headers["x-request-id"] = record.id
            record.finish(decision.status, decision.code)
            return response

        slot = server.slots[slot_name]
        # count_tokens never queues: the endpoint stamps the slot and the
        # backend call itself, so queue_wait_ms stays null on the line.
        record.slot = slot_name
        record.backend_started_at = time.monotonic()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{slot.url}/v1/messages/count_tokens",
                    json=body,
                    timeout=30,
                )
        except Exception:
            # The exception still escapes (the client gets a 500, exactly as
            # before); the line is the only new thing here.
            record.finish(500, "proxy_error")
            raise
        record.tokens = tokens_from_body(resp.content, "count_tokens")
        response = Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
            headers={"x-request-id": record.id},
        )
        record.finish(
            resp.status_code,
            "ok" if resp.status_code < 400 else "backend_error",
        )
        return response

    # ------------------------------------------------------------------
    # Collection retrieval routes (Q28: defined in manager/retrieval.py)
    # ------------------------------------------------------------------

    app.include_router(retrieval.make_router(server))

    # ------------------------------------------------------------------
    # POST /swap
    # ------------------------------------------------------------------

    @app.post("/swap")
    async def swap_slot(request: Request, background_tasks: BackgroundTasks):
        """Admin endpoint: swap a slot to a different model.

        Body: {"model": str, "target": <slot name> (optional), "trial": bool
        (optional, default false)}. Without a target (absent or null) the
        model goes to its profile's home slot. A (model, slot) pair is
        accepted only if the model's profile has a section for the slot,
        and the section must be verified unless trial is true: trial is how
        scripts/onboard.py verifies a pair, and it reloads even a model
        that is already loaded, because a trial that skips the load would
        verify nothing. No auth; LAN-only is the trust boundary.

        Returns 200 if the model is already loaded and no trial was
        requested, 202 if a swap was started. Callers poll GET /status
        (the slot's ``swapping`` and ``loaded_model`` fields) to know when
        the swap lands.

        Every response -- refusal, the already-loaded 200, and the 202 --
        carries X-Request-Id and writes one swap line (never a request
        line) to the manager.requests logger (phase 4, Q27).
        """
        rid, client_host, user_agent = request_context(request)
        headers = {"x-request-id": rid}

        # The swap line's trial field: False until the body says otherwise.
        # A non-bool trial is refused before it is read, so the line can
        # never carry a non-bool.
        trial = False

        def refuse(status: int, err_type: str, message: str,
                   model: str | None, slot_name: str | None) -> JSONResponse:
            # A refused swap logs its line at once; nothing ran, so
            # duration_ms is null.
            log_swap(rid, client_host, user_agent, slot_name, model, trial,
                     status, err_type, None)
            return JSONResponse(
                {"error": {"type": err_type, "message": message}},
                status_code=status, headers=headers,
            )

        try:
            body = await request.json()
        except Exception:
            return refuse(400, "invalid_body", "Invalid JSON", None, None)

        model_name = body.get("model")
        if not model_name:
            return refuse(400, "missing_model", "'model' is required",
                          None, None)

        # Checks run in this order: trial's type, the profile, the target,
        # the section, the verification.
        if "trial" in body and not isinstance(body["trial"], bool):
            return refuse(400, "invalid_trial", "'trial' must be a boolean",
                          model_name, None)
        trial = body.get("trial", False)

        profile = server.profiles.get(model_name)
        if profile is None:
            # A profile that failed validation is not in the roster either;
            # GET /status's profile_errors says why.
            return refuse(
                404, "model_not_found",
                (f"no profile for model '{model_name}'; "
                 f"add config/models/{model_name}.toml"),
                model_name, None,
            )

        valid = ", ".join(repr(n) for n in server.slots)
        target = body.get("target")
        if target is None:
            # Untargeted: the profile's home slot decides, never "the first
            # slot". agent_core clients send exactly this after a
            # model_not_loaded 409; guessing a slot here is what used to
            # evict main.
            target = profile.home
        if not isinstance(target, str) or target not in server.slots:
            return refuse(400, "invalid_target", f"'target' must be one of: {valid}",
                          model_name, None)

        section = profile.section(target)
        if section is None:
            return refuse(
                400, "slot_not_allowed",
                (f"model '{profile.name}' has no [slots.{target}] section; "
                 f"it may run on: {', '.join(profile.sections)}"),
                model_name, target,
            )

        if section.verified is None and not trial:
            return refuse(
                400, "not_verified",
                (f"model '{profile.name}' is not verified on slot '{target}'; "
                 f"run scripts/onboard.py {profile.name} --slot {target}"),
                model_name, target,
            )

        slot = server.slots[target]
        if not trial and slot.healthy and same_model(model_name, slot.loaded_model):
            # Already loaded and no trial: answer now and log at once. A
            # JSONResponse (not a plain dict) so the X-Request-Id header
            # rides along.
            log_swap(rid, client_host, user_agent, target, model_name, trial,
                     200, "already_loaded", None)
            return JSONResponse(
                {"slot": target, "model": slot.loaded_model, "status": "ok"},
                status_code=200, headers=headers,
            )

        started = time.monotonic()

        async def _do_swap():
            ok = await server.ensure_model_on_slot(target, model_name, force=trial)
            if not ok:
                logger.error("background swap failed: %s on slot %s",
                             model_name, target)
            # The swap line lands when the background task ends; the
            # duration runs from the 202 to that end.
            log_swap(rid, client_host, user_agent, target, model_name, trial,
                     202, "ok" if ok else "failed",
                     round((time.monotonic() - started) * 1000))

        background_tasks.add_task(_do_swap)
        return JSONResponse(
            {"slot": target, "model": model_name, "status": "swapping"},
            status_code=202, headers=headers,
        )

    return app


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    cfg = ManagerConfig.from_env()
    application = create_app(cfg)
    uvicorn.run(application, host=cfg.host, port=cfg.port)
