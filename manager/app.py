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
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from starlette.requests import ClientDisconnect

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
from manager.supervisor import (
    PROBE_FAILED, judge, supervise, supervisor_status,
)
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

# How long a caller (startup, /status, routing) waits for the slot's
# consumer to get to the probe it asked for (phase 5, Q19). The probe
# itself carries its own 3 s client timeout, so this bounds the wait for
# the consumer's turn, not the backend call.
REFRESH_TIMEOUT = 5.0

# How long a handed-off relayed stream may go unstarted before the
# consumer closes it itself (phase 5, Q10): its client never came to
# read it. A stream the client is reading is never cut off by this: once
# the relay's generator or background task has run, the consumer waits
# for the close without a timeout.
STREAM_START_TIMEOUT = 30.0


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

    async def reprobe_all_slots(self) -> None:
        """Ask every slot's consumer to probe, one at a time. Never raises.

        Probing otherwise happens at startup and after a slot fault, so a
        backend that became ready LATER is invisible for the life of the
        process. Observed on the live server: llama-manager and
        llama-server-batch both started at 23:36:45 -- the same second -- so
        the startup probe ran before the iGPU had finished loading a 4B
        model. The batch backend was serving a model this manager reported
        as loaded_model: null, and every request for it got a 409 telling
        the caller to load a model that was already loaded.

        Phase 5 (Q19): the probe runs on each slot's own consumer, asked
        for through refresh. A slot that is serving or swapping is not
        probed -- refresh returns at once for it, so this path never waits
        behind a model load and can never clobber a fresh mark_swapped.

        Sequential: worst case is N slots times REFRESH_TIMEOUT if every
        slot's consumer is slow to get to its probe. That is deliberate
        and unchanged here -- this is the chat 409 path's reprobe, called
        for one request already in flight that is about to fail anyway, so
        a few extra seconds of serialized, easy-to-reason-about refresh
        costs little. GET /status calls the concurrent variant below
        instead; see its docstring for why.
        """
        for name, slot in self.slots.items():
            try:
                await slot.refresh(REFRESH_TIMEOUT)
            except Exception:
                logger.exception(
                    "slot=%s refresh raised (should not happen; refresh is "
                    "expected to never raise)", name,
                )

    async def reprobe_all_slots_concurrently(self) -> None:
        """Ask every slot's consumer to probe, all at once. Never raises.

        Used only by GET /status. reprobe_all_slots (above) is sequential by
        design for the chat 409 path, where that's fine. /status is a
        monitoring endpoint: its worst case matters most exactly when slots
        are down, since that's when someone is anxiously checking it or an
        automated health check is polling it on a tight timeout. Sequential
        refresh there would cost up to N * REFRESH_TIMEOUT if every slot's
        consumer is slow to get to its probe -- multiplying the outage's
        visible cost instead of just reporting it. Refreshing concurrently
        bounds /status's worst case to one refresh timeout regardless of
        slot count.

        Phase 5 (Q19): each refresh runs on that slot's own consumer, so
        there is no cross-slot contention to manage, and a slot that is
        serving or swapping answers at once and is not probed.

        return_exceptions=True is load-bearing, not decorative. Without it,
        asyncio.gather re-raises the first child exception immediately,
        without awaiting or cancelling its siblings, and that exception
        would propagate straight out of this method -- past nothing that
        would catch it -- and 500 /status. slot.refresh is expected to
        never raise (the probe itself is contained on the consumer), so
        this is defense in depth; gather waits for every child regardless,
        and any exception a child does produce is logged and otherwise
        ignored rather than propagated to the /status caller.
        """
        results = await asyncio.gather(
            *(
                slot.refresh(REFRESH_TIMEOUT)
                for name, slot in self.slots.items()
            ),
            return_exceptions=True,
        )
        for name, result in zip(self.slots, results):
            if isinstance(result, BaseException):
                logger.warning(
                    "slot=%s refresh raised during concurrent /status "
                    "reprobe (should not happen; refresh is expected "
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
        fails (raises or times out) took the backend down, so it takes the
        failure tail (Task 4, Q20): one fallback swap to the slot's
        known-good model, and -- when the fallback lands -- the slot is
        idle again; when the fallback fails or there is no known-good
        model, the slot is `failed` and its queued requests are answered
        at once with 503 slot_failed.
        """
        slot = self.slots[slot_name]
        async with slot.swap_lock:
            outcome, _ = await self._ensure_model_on_slot_locked(
                slot_name, model_name, force=force)
        return outcome in ("ok", "already_loaded")

    async def _ensure_model_on_slot_locked(self, slot_name: str, model_name: str,
                                           *, force: bool = False
                                           ) -> tuple[str, str | None]:
        """The body of ensure_model_on_slot, with the slot's swap_lock
        already held by the caller.

        The public wrapper takes the lock; the slot's queue consumer holds
        it itself, so its already-loaded decision and the swap run are one
        atomic step a probe cannot land between and change the answer.

        Returns (outcome, message), outcome one of: "already_loaded" (the
        shortcut), "refused" (the profile set cannot render the model;
        nothing was stopped, the slot is untouched), "ok" (the swap
        landed), "fell_back" (the swap failed but one fallback swap
        landed; the message names the failure and the known-good model)
        or "failed" (the swap failed and the fallback failed or was
        unavailable; the slot is now `failed` and its queued requests
        were answered at once with slot_failed -- Task 4, Q20).
        """
        slot = self.slots[slot_name]
        if not force and slot.healthy and same_model(model_name, slot.loaded_model):
            return "already_loaded", None

        profile = self.profiles.get(model_name)
        if profile is None:
            logger.error(
                "No profile for model %s in %s; add a <name>.toml there",
                model_name, self._config.profiles_dir)
            return "refused", None

        section = profile.section(slot_name)
        if section is None:
            logger.error(
                "Model %s has no [slots.%s] section; it may run on: %s",
                profile.name, slot_name, ", ".join(profile.sections))
            return "refused", None

        # From here the swap is real: the slot shows swapping -- and with
        # it, not healthy -- for its whole duration, so /status and a
        # concurrent request see the truth (phase 5, Q19). The move is
        # legal from wherever a swap can start: the slot was idle holding
        # a different model, unhealthy, or failed -- a /swap is the
        # recovery from failed (Task 4, Q20).
        slot._set_state("swapping")

        logger.info("slot=%s swapping to %s", slot_name, profile.name)
        msg: str | None = None
        try:
            env_text = render_model_env(profile, section, self._config.models_dir)
            success = await slot.swapper.swap_to(profile.name, env_text)
        except Exception as exc:
            # A raising swap is a failed swap, not a crash: the same
            # failure tail as the timeout case (Task 4, Q20) so no caller
            # ever sees an exception here.
            logger.exception("slot=%s swap to %s raised", slot_name, profile.name)
            success = False
            msg = f"Model swap failed on slot {slot_name}: {profile.name}: {exc}"
        if success:
            slot.mark_swapped(profile.name)
            return "ok", None

        if msg is None:
            msg = f"Model swap timed out on slot {slot_name}: {model_name}"
        # Record the fault BEFORE the fallback, so a failed fallback keeps
        # the original last_error (Task 4, Q20).
        slot.record_error("swap_failed", msg)
        # One fallback swap to the slot's known-good model -- the rule:
        # a fallback is to a DIFFERENT model (a healthy probe or a
        # successful swap set it). Reloading the model that just failed
        # would double the outage for nothing, and a /swap retries if
        # wanted -- so when the known-good model is the one that failed,
        # there is no fallback and the slot goes `failed` below. One
        # direct swap_to under this already-held swap_lock -- not a
        # second call into this method, and never the already-loaded
        # shortcut: a fallback must really load the other model.
        known_good = slot.known_good
        if known_good is not None and not same_model(known_good, model_name):
            fb_profile = self.profiles.get(known_good)
            fb_section = (fb_profile.section(slot_name)
                          if fb_profile is not None else None)
            if fb_section is not None:
                try:
                    fb_env = render_model_env(
                        fb_profile, fb_section, self._config.models_dir)
                    fb_ok = await slot.swapper.swap_to(known_good, fb_env)
                except Exception:
                    logger.exception(
                        "slot=%s fallback swap to known-good %s raised",
                        slot_name, known_good)
                    fb_ok = False
                if fb_ok:
                    slot.mark_swapped(known_good)
                    logger.warning(
                        "slot=%s swap to %s failed; fell back to known-good "
                        "%s: %s",
                        slot_name, profile.name, known_good, msg)
                    return "fell_back", f"{msg}; fell back to {known_good}"
                msg = f"{msg}; fallback to {known_good} also failed"
        slot._set_state("failed")
        # Keep the reason on the slot: this tail's final message is what
        # failed it, and the slot_failed 503s that follow must name it --
        # not a later fault (a failed probe while failed records
        # probe_failed as the new last_error; Task 4, review round 1).
        slot.failed_reason = msg
        await self._fail_slot_queue(slot_name)
        return "failed", msg

    async def _fail_slot_queue(self, slot_name: str) -> None:
        """Answer the given slot's queue after the slot moved to `failed`
        (Task 4, Q20). Runs in the failure tail, under the slot's held
        swap_lock.

        Request items -- including items without a kind -- are answered
        with slot_failed at once: their endpoints are waiting on the
        item's event and render the 503. Swap items are not answered
        here: they are put back in the queue, in order, where the
        consumer's next iterations take them -- a swap on a failed slot
        is its recovery, so queued swaps still run. The queue is empty
        after the drain and a put below capacity never yields, so the
        put-back loop runs to completion without interleaving.
        """
        slot = self.slots[slot_name]
        items = slot.queue.drain()
        swaps: list[dict] = []
        answered = 0
        for item in items:
            if item.get("kind") == "swap":
                swaps.append(item)
            else:
                item["error"] = _slot_failed_message(slot_name, slot)
                item["error_code"] = "slot_failed"
                item["event"].set()
                answered += 1
        for item in swaps:
            await slot.queue.enqueue(item)
        if answered:
            logger.warning(
                "slot=%s failed; answered %d queued request(s) with "
                "slot_failed: %s",
                slot_name, answered, _slot_failed_message(slot_name, slot))
        if swaps:
            logger.warning(
                "slot=%s leaving %d queued swap item(s) to the consumer: %s",
                slot_name, len(swaps), _slot_failed_message(slot_name, slot))


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

async def _serve_probe_requests(slot, config: ManagerConfig) -> None:
    """Run this slot's pending probes, if any (phase 5, Q19), judging each
    for the supervisor (Task 5, Q21).

    The consumer is the only probe writer: startup, /status and routing ask
    for a probe through refresh, which arms the handshake and wakes the
    consumer on the queue event. A probe is contained here -- the contract
    says it never raises, but a bug past that guarantee must not end the
    consumer or strand the refresh waiters, so finish_probe always runs.
    The queue event is deliberately NOT cleared here: a refresh that joins
    mid-drain wakes the consumer on this handshake, not on the event, and
    an extra empty pass costs nothing.

    After each probe, and after finish_probe() -- so the refresh waiters
    are released before a restart, which can take RESTART_TIMEOUT, begins --
    the result goes to judge. A probe that raised counts as failed (the
    result stays PROBE_FAILED). A due restart runs under the slot's
    swap_lock on this task (see _apply_supervision).
    """
    while slot.probe_pending:
        result = PROBE_FAILED
        try:
            async with httpx.AsyncClient() as probe_client:
                result = await slot.probe(probe_client)
        except Exception:
            logger.exception(
                "slot=%s probe raised (should not happen; slot.probe is "
                "expected to never raise)", slot.name,
            )
        finally:
            slot.finish_probe()
        await _apply_supervision(slot, config, result)


async def _apply_supervision(slot, config: ManagerConfig, result: str) -> None:
    """The supervisor's half of a finished probe (Task 5, Q21): judge the
    result on this slot's Supervision, and act when a restart is due.

    Runs on the slot's queue consumer -- the only writer of the slot's
    state and supervision -- after finish_probe(), so a restart never
    holds up the refresh waiters. A probe that raised (a bug past its
    never-raises contract) arrives here as PROBE_FAILED and counts.

    A due restart with config.supervisor_restart calls
    slot.swapper.restart_llama_server() under slot.swap_lock on this task;
    one that returns increments restarts and starts a fresh loading grace,
    one that raises is logged with its exception -- not counted, the grace
    untouched -- and the consumer carries on. With restarts off (the
    default) a due restart increments restarts_suppressed and logs one
    WARNING; the judge already reset the count, so the backoff paces those
    lines.
    """
    due = judge(
        slot.supervision, result,
        stored_state=slot.stored_state,
        grace_started=slot.grace_started,
        config=config,
        now=time.monotonic(),
    )
    if not due:
        return
    if not config.supervisor_restart:
        slot.supervision.restarts_suppressed += 1
        logger.warning(
            "slot=%s would restart %s after %d consecutive failed probes "
            "(supervisor restarts are disabled; set SUPERVISOR_RESTART=1 "
            "to enable)",
            slot.name, slot.systemd_unit, config.supervisor_failures,
        )
        return
    try:
        async with slot.swap_lock:
            await slot.swapper.restart_llama_server()
    except Exception:
        logger.exception(
            "slot=%s supervisor restart of %s failed; not counted, the "
            "loading grace is unchanged",
            slot.name, slot.systemd_unit,
        )
        return
    slot.supervision.restarts += 1
    slot.grace_started = time.monotonic()


def _stream_close(ctx, client):
    """Build the single idempotent closer for an opened backend stream.

    Exits the stream context (closes the backend stream), then closes the
    client that owns it. Idempotent: the relay's finish -- run by exactly
    one of the generator's finally, the background task, or the consumer --
    calls it, but a second call can never hurt.
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


class _Relay(StreamingResponse):
    """The handed-off relayed stream, with the consumer's handle on it
    (phase 5, Q10).

    It is the StreamingResponse the endpoint gives the client, plus what
    the consumer needs to hold the slot for it: `started` is set when the
    relay's generator begins or its background task runs; from then on
    the stream is live (or already ran) and the wait on it carries no
    timeout. `closed` is set once the stream's closer has run, from
    whichever path ran it. `finish` runs that closer and ends the request
    it carries; the consumer calls it only when nobody started the stream
    within STREAM_START_TIMEOUT.
    """

    def __init__(self, gen, resp, started, closed, finish, finish_stream):
        super().__init__(
            gen,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "text/event-stream"),
            background=BackgroundTask(finish_stream),
        )
        self.started = started
        self.closed = closed
        self.finish = finish


def _relay_stream(resp, close, record=None, stream_tokens=None,
                  on_close=None) -> _Relay:
    """Relay an already-opened backend stream (status < 400) to the client.

    The closer runs from the generator's finally, from the background
    task, and -- for a stream nobody starts within STREAM_START_TIMEOUT --
    from the consumer itself. All three paths call the same `finish`, and
    its synchronous claim makes them one: the first to run it closes the
    stream and ends the request it carries; a close that raises still ends
    the request before the exception propagates (on that path Starlette
    never reaches the background task, so the request must not depend on
    it).

    The request ends when the closer runs: on_close (the slot's
    end_request) and the line are written from the closer, on every exit
    path -- ok when the relay reached the end of the stream,
    client_disconnected otherwise -- so the slot stays busy until the
    stream closes and never before (phase 5, Q10).

    When the item carries a request record (phase 4, Q27), each relayed
    chunk is also fed, unchanged, to stream_tokens -- the sniffing never
    touches or buffers the bytes -- and the closer writes the line with
    the sniffed counts. finish is idempotent, so the paths can never
    double-write a relayed stream's line.
    """
    state = {"completed": False, "closed": False}
    started = asyncio.Event()
    closed = asyncio.Event()

    def end_request_line() -> None:
        # The request ends here, on every exit path, even if close()
        # raised; all three calls are idempotent, so a path that runs
        # after another is a no-op.
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
        closed.set()

    async def finish() -> None:
        # The one closer for this stream. The claim is synchronous, so
        # exactly one path -- the generator's finally, the background
        # task, or the consumer at the start timeout -- runs it; a close
        # that raises still ends the request (finally below), and the
        # exception propagates afterwards.
        if state["closed"]:
            return
        state["closed"] = True
        try:
            await close()
        finally:
            end_request_line()

    async def gen():
        if state["closed"]:
            # The consumer closed this stream as unstarted before the
            # response was driven, or the endpoint drove it late: end it
            # empty instead of reading a stream that is gone.
            return
        started.set()
        try:
            async for chunk in resp.aiter_bytes():
                if stream_tokens is not None:
                    stream_tokens.feed(chunk)
                yield chunk
            state["completed"] = True
        finally:
            await finish()

    async def finish_stream() -> None:
        # Starlette runs this after the stream, on every path that
        # reaches it. The generator's finally already ran finish on every
        # one of those, so this is the safety net that keeps the old
        # guarantee: the closer runs even when the generator never
        # started (the client left before the first chunk).
        started.set()
        await finish()

    return _Relay(gen(), resp, started, closed, finish, finish_stream)


async def _wait_for_relay(relay: "_Relay", slot_name: str) -> None:
    """Hold the consumer until a handed-off stream's closer has run (Q10).

    The item was already answered at the handoff (its event is set before
    this wait), so the client can start reading; what waits is the slot's
    next item -- a queued swap included -- and any probe the consumer
    would otherwise serve now.

    The start timeout touches only a stream nobody starts: if neither the
    relay's generator nor its background task has run within
    STREAM_START_TIMEOUT of the handoff, the consumer runs the closer
    itself and the line ends client_disconnected. A stream the client is
    reading is never cut off: once started, this wait has no timeout.

    Cancelling the consumer here does not close a started stream (the
    client may still be reading); the cancellation propagates as is. A
    close that raises at the start timeout has already ended the request
    inside finish; it is logged, not swallowed into the consumer.
    """
    try:
        await asyncio.wait_for(relay.started.wait(),
                               timeout=STREAM_START_TIMEOUT)
    except asyncio.TimeoutError:
        if relay.started.is_set():
            # The generator or background task started in the same instant
            # the timeout fired: treat the stream as live.
            await relay.closed.wait()
        else:
            try:
                await relay.finish()
            except Exception:
                logger.exception(
                    "slot=%s closing an unstarted stream raised; the "
                    "request ended anyway", slot_name,
                )
    else:
        await relay.closed.wait()


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

    Before each pass over the queue, and after every item, it runs the
    slot's pending probes (phase 5, Q19): startup, /status and routing ask
    for a probe through refresh, and this consumer is the only thing that
    runs one.

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

        # A probe may have been asked for since the last pass (startup,
        # /status or routing, through refresh): run it before the queue so
        # the next item's readiness is checked against fresh state.
        await _serve_probe_requests(slot, config)

        while not slot.queue.empty():
            item = await slot.queue.dequeue()
            if item.get("kind") == "swap":
                # A queued swap (phase 5, Q12): the consumer runs it itself,
                # after whatever was ahead of it in the queue. It never
                # raises except CancelledError and always finishes the item
                # (last_swap, the swap line, the event). Items with no
                # "kind" are requests, as are those with "kind": "request".
                await _run_swap_item(server, slot_name, item)
            else:
                await _process_request_item(server, config, slot_name, item)
            # A fault answered above may have asked for a probe: run it now,
            # without waiting for another item (phase 5, Q19).
            await _serve_probe_requests(slot, config)


async def _client_is_gone(slot_name: str, request) -> bool:
    """Whether the client that sent a queued request is still connected
    (phase 5, Q26).

    One is_disconnected() check is not reliable on the production stack
    (the unit passes no --http, so uvicorn picks httptools): with a body
    over 64 KiB, uvicorn has paused reading the socket by the time the
    body is read, so it has not seen the client close. Starlette's check
    (starlette/requests.py) resumes the read but returns before it
    completes, so the close shows only on a check one loop tick later.
    Hence: check, yield one loop tick, check again. Neither check waits
    for a disconnect, so a connected client is served at once.

    Never raises: a check that raises counts the client as connected
    (one warning naming the slot), so the item is served as before.
    """
    try:
        if await request.is_disconnected():
            return True
        await asyncio.sleep(0)
        return await request.is_disconnected()
    except Exception as exc:
        logger.warning(
            "slot=%s client-gone check failed; serving the request: %s",
            slot_name, exc,
        )
        return False


async def _process_request_item(server: "ServerState", config: ManagerConfig,
                                slot_name: str, item: dict) -> None:
    """Process one queued request item for the named slot.

    The consumer's per-request body, extracted so the consumer itself stays
    readable: first a check that the client is still connected -- a request
    whose client has gone is answered at once with client_gone, no
    readiness check, no backend call, the slot's state and in_flight
    untouched (Task 6, Q26); then a slot that failed while the item waited
    is answered at once with slot_failed -- no readiness check, no backend
    call (Task 4, Q20); otherwise readiness first (ready_for -- read-only,
    never swaps or restarts), then the proxy to the slot's backend URL.
    Early exits return; the item is always answered (response or error) and
    its event set exactly once. A relayed stream is answered at the handoff and
    then held: the function waits for the stream's closer before
    returning (phase 5, Q10), so the consumer takes no next item, serves
    no probe, and runs no queued swap while the stream relays. A per-item
    exception is answered as a 503 queue_error and swallowed (F2 part
    2); cancellation (asyncio.CancelledError) is a BaseException and
    still propagates and kills the consumer.

    The backend PATH is per-item: each endpoint stamps its own
    item["backend_path"] when enqueuing, since OpenAI chat and Anthropic
    messages are proxied by the same consumer to different backend
    endpoints. The default keeps any un-stamped item on the OpenAI path.
    """
    slot = server.slots[slot_name]
    event: asyncio.Event = item["event"]
    record = item.get("record")
    if record is not None:
        # The endpoint stamped enqueued_at; the consumer stamps the
        # dequeue (queue_wait_ms) and, below, backend_started_at.
        record.slot = slot_name
        record.dequeued_at = time.monotonic()

    try:
        # Q26: skip a request whose client has gone. First thing inside the
        # try, after the record's slot and dequeue stamps above (so the
        # line has queue_wait_ms) and before the failed branch, ready_for,
        # begin_request and any backend call: nobody is there to read the
        # answer, and the line should say why the request was not served.
        # The endpoint puts its Request on the item as item["request"];
        # an item with no "request" (internal callers, tests) is served
        # exactly as before. Swap items never reach here: they are
        # dispatched to _run_swap_item, and a swap's 202 is already
        # answered, so its client would always read as gone.
        request = item.get("request")
        if request is not None and await _client_is_gone(slot_name, request):
            item["error"] = (
                f"client for {slot_name} request disconnected before "
                f"the request was served"
            )
            item["error_code"] = "client_gone"
            event.set()
            return
        if slot.stored_state == "failed":
            # The slot failed before this item was processed -- a failed
            # swap's tail answered most queued requests at once (Task 4,
            # Q20), but an item can still be dequeued into a failed slot
            # (enqueued ahead of the failed swap's failure tail drain, or
            # by a direct enqueue). Answer it at once with slot_failed:
            # no readiness check, no backend call.
            item["error"] = _slot_failed_message(slot_name, slot)
            item["error_code"] = "slot_failed"
            event.set()
            return
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
            return
        if readiness == "unavailable":
            item["error"] = f"{slot_name} slot not ready"
            item["error_code"] = "slot_unavailable"
            event.set()
            return

        is_streaming = body.get("stream", False)
        # Q27: the request is in flight from just before the backend
        # call until the response is complete; a request refused
        # before the call (wrong_model, slot_unavailable) never gets
        # here. Every request gets a FRESH key, record or not: a
        # client-supplied X-Request-Id is kept as given (record.id),
        # so two requests can share it -- a retry, a client with a
        # fixed id -- and keying by the record's id would let the
        # second overwrite the first and let ending either end both.
        # The closer and the finally below end this same key.
        request_id = request_id_from(None)
        slot.begin_request(request_id)
        handed_off = False
        relay = None
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
                        slot.record_error(
                            "backend_error",
                            f"backend returned {resp.status_code} "
                            f"on {backend_path}",
                        )
                        slot.request_probe()
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
                    relay = _relay_stream(
                        resp, close, record=record,
                        stream_tokens=stream_tokens,
                        on_close=lambda rid=request_id:
                            slot.end_request(rid),
                    )
                    item["response"] = relay
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
                    slot.record_error(
                        "backend_error",
                        f"backend returned {resp.status_code} "
                        f"on {backend_path}",
                    )
                    slot.request_probe()
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
            slot.record_error("proxy_error", item["error"])
            slot.request_probe()
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
        return

    event.set()

    if handed_off:
        # Q10: the slot is busy until the relayed stream closes. The
        # event was set at the handoff, so the client can start reading
        # now; what waits is this consumer -- its next item, its probes,
        # a queued swap -- which must not move on while the stream runs.
        await _wait_for_relay(relay, slot_name)


async def _run_swap_item(server: "ServerState", slot_name: str, item: dict) -> None:
    """Run one queued swap item (phase 5, Q12): the swap itself, then the
    slot's last_swap entry, the swap's line, and the item's event.

    The swap holds the slot's swap_lock for its duration, so /status shows
    the slot swapping while it runs. The already-loaded decision is made
    under that same held lock, one atomic step with the swap run, so a
    probe cannot land between the two and change the answer.

    Never raises except CancelledError: an unexpected exception still ends
    the item (a failed entry, a failed line, the event set), so the
    consumer's every-dequeued-item-is-answered-once guarantee holds.
    """
    slot = server.slots[slot_name]
    model_name: str = item["model"]
    trial: bool = item["trial"]
    rid: str = item["id"]
    try:
        async with slot.swap_lock:
            if not trial and slot.healthy and same_model(
                    model_name, slot.loaded_model):
                # The model was loaded while this item waited (something
                # ahead of it in the queue swapped it in): nothing to
                # run. Same condition as _ensure_model_on_slot_locked's
                # shortcut, checked under the same held lock, so the
                # two cannot disagree.
                outcome, message = "already_loaded", None
            else:
                outcome, message = await server._ensure_model_on_slot_locked(
                    slot_name, model_name, force=trial)
                if outcome == "refused":
                    # Unreachable for a queued item -- the endpoint
                    # validated the profile and section before enqueuing.
                    # If it happens anyway, answer it as a failed swap
                    # with a plain message: nothing was stopped, so the
                    # slot's state and queue were left alone.
                    outcome = "failed"
                    message = (f"Model swap failed on slot {slot_name}: "
                               f"{model_name}")
        slot.last_swap = {
            "id": rid,
            "model": model_name,
            "trial": trial,
            "outcome": outcome,
            "message": message,
            "at_utc": datetime.now(timezone.utc).isoformat(),
        }
        log_swap(rid, item["client"], item["user_agent"], slot_name,
                 model_name, trial, 202, outcome,
                 round((time.monotonic() - item["accepted_at"]) * 1000))
        item["event"].set()
    except Exception:
        # The consumer must survive whatever a swap item throws; the item
        # still gets exactly one answer. last_swap and the line are set in
        # the same synchronous step, as in the normal path.
        logger.exception("slot=%s swap item %s failed internally",
                         slot_name, rid)
        slot.last_swap = {
            "id": rid,
            "model": model_name,
            "trial": trial,
            "outcome": "failed",
            "message": "internal error while running the swap "
                       "(see manager logs)",
            "at_utc": datetime.now(timezone.utc).isoformat(),
        }
        log_swap(rid, item["client"], item["user_agent"], slot_name,
                 model_name, trial, 202, "failed",
                 round((time.monotonic() - item["accepted_at"]) * 1000))
        item["event"].set()


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


def _slot_failed_message(slot_name: str, slot: SlotState) -> str:
    """The 503 slot_failed message, shared by the routing path, the queue
    consumer and the failure tail (Task 4, Q20): the slot and the fault
    that failed it -- the reason kept on the slot when it entered
    `failed` (review round 1) -- so the answer names that fault even
    after a later one (a failed probe) has become the last_error."""
    detail = slot.failed_reason or "no fault recorded"
    return f"slot {slot_name} failed: {detail}"


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

    `error_code` None or "queue_error" (an exception while the consumer
    processed the item) is a 503 queue_error; "slot_failed" is the typed
    503 for a request queued on a slot that failed while it waited
    (Task 4, Q20); "slot_unavailable" is the typed 503; "upstream_timeout"
    is the 504 for a slow-but-healthy backend (F4); "model_not_loaded" is
    the 409 for a request whose model was swapped out while it was queued;
    "client_gone" is the 499 for a request whose client disconnected
    before the consumer served it (Task 6, Q26). Without that mapping it
    would fall through to a 503 queue_error.
    The error shapers render these unchanged.
    """
    code = item.get("error_code")
    if code == "model_not_loaded":
        status, code = 409, "model_not_loaded"
    elif code == "client_gone":
        status, code = 499, "client_gone"
    elif code == "slot_failed":
        status, code = 503, "slot_failed"
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
        # different model, since the last probe (F6) -- so ask the slot's
        # consumer for one fresh probe before answering (phase 5, Q19). A
        # busy slot (serving or swapping) is refused at once, so this path
        # never waits behind a swap or an in-flight request.
        await slot.refresh(REFRESH_TIMEOUT)
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
        if slot.stored_state == "failed":
            # The slot failed with the model still loaded (a failed swap
            # and its failed fallback; Task 4, Q20). A failed slot cannot
            # take requests until a /swap or a healthy probe recovers it,
            # so answer now instead of enqueuing a request that would only
            # be answered from the queue.
            return _RouteDecision(
                status=503, code="slot_failed",
                message=_slot_failed_message(slot_name, slot),
                slot_name=slot_name,
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
    request: Request | None = None,
) -> _RouteDecision:
    """Validate, route, enqueue and wait -- the core of the queued endpoints.

    `backend_path` is stamped on the queued item so the slot's consumer
    proxies it to the right backend endpoint (see _queue_consumer).

    When the endpoint passes a request record (phase 4, Q27), it is stamped
    with enqueued_at and travels with the item as item["record"]; the
    consumer stamps the rest and the closer or the endpoint finishes it.
    When the endpoint passes its Request (phase 5, Q26), it travels with the
    item as item["request"], so the consumer can ask at dequeue whether the
    client is still connected.
    """
    decision, slot_name = await _validate_and_route(server, body)
    if decision is not None:
        return decision

    event = asyncio.Event()
    item: dict = {
        "kind": "request",
        "body": body, "event": event, "response": None, "error": None,
        "error_code": None, "backend_path": backend_path,
    }
    if record is not None:
        record.enqueued_at = time.monotonic()
        item["record"] = record
    if request is not None:
        item["request"] = request
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
    elif decision.code == "slot_failed":
        err = {"type": "slot_failed", "message": decision.message}
    elif decision.code in ("queue_full", "queue_error"):
        err = {"message": decision.message, "type": "server_error"}
    elif decision.code == "model_not_loaded":
        err = {"type": "model_not_loaded", "message": decision.message}
    elif decision.code == "upstream_timeout":
        err = {"type": "upstream_timeout", "message": decision.message}
    elif decision.code == "client_gone":
        err = {"type": "client_gone", "message": decision.message}
    else:  # missing_model, model_not_found
        err = {"message": decision.message, "type": "invalid_request_error"}
    headers = {"Retry-After": "5"} if decision.status == 503 else None
    return JSONResponse({"error": err}, status_code=decision.status, headers=headers)


def _anthropic_error_response(decision: _RouteDecision) -> JSONResponse:
    """Render a _RouteDecision failure in the Anthropic error shape:
    {"type": "error", "error": {"type": ..., "message": ...}}, with the inner
    type taken from Anthropic's error taxonomy where one exists (409 "not
    loaded on any slot" and 499 "client gone" have no Anthropic analog and
    keep their manager types).
    """
    # client_gone has no Anthropic equivalent either, so it keeps the
    # manager's code, as model_not_loaded already does. Without the key
    # the lookup below raises KeyError: a 500 and no line.
    type_by_code = {
        "invalid_body": "invalid_request_error",
        "missing_model": "invalid_request_error",
        "model_not_found": "not_found_error",
        "model_not_loaded": "model_not_loaded",
        "client_gone": "client_gone",
        "slot_unavailable": "api_error",
        "slot_failed": "api_error",
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
        # Start one background queue consumer per slot FIRST: the startup
        # probes below run on each slot's own consumer (phase 5, Q19) --
        # the single writer of that slot's state.
        consumer_tasks = [
            asyncio.create_task(
                _queue_consumer(server, config, slot_name=name),
                name=f"queue_consumer_{name}",
            )
            for name in server.slots
        ]

        # One supervisor per slot (Task 5, Q21): a ticker that asks the
        # slot's own consumer to probe while the slot is idle, unhealthy or
        # failed. It writes nothing -- the probe, its judgment and any
        # restart run on the consumer.
        supervisor_tasks = [
            asyncio.create_task(
                supervise(server.slots[name], config.supervisor_probe_interval),
                name=f"supervisor_{name}",
            )
            for name in server.slots
        ]

        # Ask each slot to probe itself through its consumer to see if the
        # backend is already running.
        for name, slot in server.slots.items():
            await slot.refresh(REFRESH_TIMEOUT)
            if slot.healthy:
                logger.info(
                    "slot=%s ready with model %s",
                    name, slot.loaded_model,
                )
            else:
                logger.info("slot=%s not reachable at startup", name)

        # Collection retrieval (Q28): the startup block (initialization,
        # background indexing, the orphaned-collection prune task) and the
        # shutdown block both live in manager/retrieval.py.
        prune_task = await retrieval.start_retrieval(server, config)

        yield

        await retrieval.stop_retrieval(server, prune_task)

        # Stop the askers before the writers: a supervisor cancelled after
        # its consumer could only arm probes nothing runs anymore.
        for task in supervisor_tasks:
            task.cancel()
        for task in supervisor_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

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
                name: {
                    **slot.to_status_dict(),
                    "supervisor": supervisor_status(slot.supervision, config),
                    "models": home_models.get(name, []),
                }
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
        except ClientDisconnect:
            # Q26: the client left before the body was complete. Starlette
            # raises ClientDisconnect from request.json(); before this
            # handler it reached the except below and was answered 400
            # invalid_body. Answer 499 client_gone and write the line here:
            # nothing is queued, so the line's slot stays null.
            record = RequestRecord.from_request(request, "/v1/chat/completions", None)
            response = _openai_error_response(_RouteDecision(
                status=499, code="client_gone",
                message="client disconnected before the request body was complete",
            ))
            response.headers["x-request-id"] = record.id
            record.finish(499, "client_gone")
            return response
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
            server, body, backend_path="v1/chat/completions", record=record,
            request=request,
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
        except ClientDisconnect:
            # Q26: the client left before the body was complete (same
            # handler as /v1/chat/completions, the Anthropic shape): 499
            # client_gone and the line here, nothing queued.
            record = RequestRecord.from_request(request, "/v1/messages", None)
            response = _anthropic_error_response(_RouteDecision(
                status=499, code="client_gone",
                message="client disconnected before the request body was complete",
            ))
            response.headers["x-request-id"] = record.id
            record.finish(499, "client_gone")
            return response
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
                                         record=record, request=request)
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
    async def swap_slot(request: Request):
        """Admin endpoint: swap a slot to a different model.

        Body: {"model": str, "target": <slot name> (optional), "trial": bool
        (optional, default false)}. Without a target (absent or null) the
        model goes to its profile's home slot. A (model, slot) pair is
        accepted only if the model's profile has a section for the slot,
        and the section must be verified unless trial is true: trial is how
        scripts/onboard.py verifies a pair, and it reloads even a model
        that is already loaded, because a trial that skips the load would
        verify nothing. No auth; LAN-only is the trust boundary.

        The swap is a FIFO item in the target slot's queue (phase 5, Q12):
        the slot's consumer runs it after whatever is ahead of it. 200 is
        answered only when nothing is ahead -- the slot is idle (healthy,
        unlocked, nothing in flight), its queue is empty, the model is
        already loaded and no trial was requested. Everything else,
        including an already-loaded model with something ahead, is 202:
        the swap waits its turn in the queue. A full queue is refused 503
        queue_full. Callers poll GET /status (the slot's ``swapping``,
        ``state`` and ``last_swap`` fields) to know when the swap lands.

        Every response -- refusal, the already-loaded 200, and the 202 --
        carries X-Request-Id and writes one swap line (never a request
        line) to the manager.requests logger (phase 4, Q27): refusals and
        the 200 at once, a 202 when the slot's consumer ends the swap.
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
        if (not trial and slot.state == "idle" and slot.queue.empty()
                and same_model(model_name, slot.loaded_model)):
            # Already loaded, nothing ahead of it (the slot idle: healthy,
            # unlocked, nothing in flight; the queue empty) and no trial:
            # answer now and log at once. A JSONResponse (not a plain dict)
            # so the X-Request-Id header rides along.
            log_swap(rid, client_host, user_agent, target, model_name, trial,
                     200, "already_loaded", None)
            return JSONResponse(
                {"slot": target, "model": slot.loaded_model, "status": "ok"},
                status_code=200, headers=headers,
            )

        # Otherwise the swap is a FIFO item in the target slot's queue: the
        # slot's consumer runs it after whatever is ahead of it and writes
        # its line when the swap ends; the 202's duration runs from here.
        event = asyncio.Event()
        item: dict = {
            "kind": "swap",
            "model": model_name,
            "trial": trial,
            "id": rid,
            "client": client_host,
            "user_agent": user_agent,
            "accepted_at": time.monotonic(),
            "event": event,
        }
        try:
            await slot.queue.enqueue(item)
        except RequestQueue.QueueFullError:
            return refuse(503, "queue_full", "Server busy", model_name, target)
        slot.queue_event.set()
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
