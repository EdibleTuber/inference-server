"""
Per-slot state for the model manager.

Each inference backend has its own SlotState containing loaded-model
tracking, the slot's one stored state, a swap lock, a queue, and an event
the handler signals when a new item is enqueued. Which backends exist is
configuration, not code. Swap operations and routing decisions read/write
this state in-process.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from manager.names import display_name
from manager.queue import RequestQueue
from manager.supervisor import (
    PROBE_FAILED, PROBE_HEALTHY, PROBE_LOADING, Supervision,
)

logger = logging.getLogger(__name__)


class IllegalTransition(ValueError):
    """A state move that is not in TRANSITIONS. A bug: the stored state
    moves only through the table, and only on the slot's own queue
    consumer."""


# The only legal moves, over the stored states (phase 5, Q19). `serving` is
# not a stored state -- it is `idle` with a request in flight -- and
# appears nowhere in the table. `failed` (Task 4, Q20): a swap that failed
# and had no known-good model to fall back to, or whose fallback failed too;
# a /swap (failed -> swapping) or a healthy probe (failed -> idle) recovers
# it. Nothing moves a swapping or a failed slot to unhealthy: a failed swap
# falls back or fails, and a failed slot recovers by loading a model, not by
# going stale behind a fault.
TRANSITIONS: dict[str, frozenset[str]] = {
    "unhealthy": frozenset({"probing", "swapping"}),
    "probing": frozenset({"idle", "unhealthy"}),
    "idle": frozenset({"swapping", "unhealthy"}),
    "swapping": frozenset({"idle", "failed"}),
    "failed": frozenset({"idle", "swapping"}),
}


@dataclass
class SlotState:
    """State container for one inference slot.

    A slot is a backing llama-server process that the manager fronts.
    'loaded_model' reflects what that process currently has loaded;
    'stored_state' is the one state the machine holds for the slot
    (unhealthy, probing, idle, swapping, failed). 'state' derives
    'serving' from an idle slot with a request in flight, and 'healthy'
    derives from the state.

    The queue, queue_event, and swap_lock are per-slot so work on one
    slot does not interfere with the other.
    """
    name: str                               # configured slot identifier
    host: str
    port: int
    env_file: str                           # path for the swap to rewrite
    systemd_unit: str                       # unit name to restart
    queue: RequestQueue
    loaded_model: Optional[str] = None
    # The model this slot last served healthily (Task 4, Q20): set by
    # every healthy probe and by mark_swapped, never cleared -- a failed
    # probe or a no-model probe leaves it alone. None until the first
    # such evidence; the fallback target of a failed swap.
    known_good: Optional[str] = None
    # The one stored state (phase 5, Q19), moved only through TRANSITIONS
    # and only by the slot's own queue consumer. `serving` is never stored
    # (it reads idle with a request in flight). Tests reach the states
    # through the `healthy` setter below, which bypasses the table.
    stored_state: str = "unhealthy"
    last_swap_utc: Optional[str] = None
    queue_event: asyncio.Event = field(default_factory=asyncio.Event)
    swap_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    model_env_file: Optional[str] = None   # generated model env file; the swap's write target
    last_error: Optional[dict] = None      # {"code","message","at_utc"}; set only by slot fault
    # The fault that made the slot `failed` -- the failure tail's final
    # message, kept on the slot when it enters `failed` (Task 4, Q20,
    # review round 1): the slot_failed 503s name this, not a later fault
    # (a failed probe while the slot is failed records probe_failed as
    # the new last_error, and the answer must still name the swap that
    # failed the slot). Re-set on every entry into `failed`; read only
    # while the slot is `failed`.
    failed_reason: Optional[str] = None
    # The swap item the consumer finished most recently:
    # {"id","model","trial","outcome","message","at_utc"}, outcome one of
    # ok, already_loaded, fell_back (the swap failed and the slot fell
    # back to its known-good model), failed. None until the first
    # finished swap; set only by the slot's queue consumer.
    last_swap: Optional[dict] = None
    # The supervisor's bookkeeping for this slot (Task 5, Q21): the
    # consecutive-failure count, the restart counters and the backoff.
    # Written only by the slot's own queue consumer (judge, after every
    # probe it runs, and the restart it triggers); the supervisor task
    # itself writes nothing -- it only asks for probes.
    supervision: Supervision = field(default_factory=Supervision)
    # When the current loading grace started, in time.monotonic(): the
    # slot's creation (the manager's own start is a grace) and every move
    # out of `idle` (see _set_state); a supervisor restart that returns
    # starts a fresh one (manager/app.py). While now - grace_started <
    # SWAP_TIMEOUT a 503 is a backend still loading its model, not a
    # failed probe (Task 5, Q21, Review Focus 3).
    grace_started: float = field(default_factory=time.monotonic)
    # When the slot's last probe's backend call returned or failed, in
    # time.monotonic(): stamped by probe after the call, on every return
    # path, whoever asked (phase 6, Task 6) -- the stamp says when the slot
    # was last looked at, whatever was seen. None until the first probe;
    # /status reports its age as last_probe_age_seconds.
    last_probe_at: Optional[float] = None
    # request_id -> time.monotonic() begin. A request is in flight from just
    # before the consumer calls the backend until the response is complete; a
    # relayed stream stays in flight until its closer runs.
    in_flight: dict = field(default_factory=dict)
    # The probe handshake with this slot's consumer (phase 5, Q19):
    # request_probe() arms it and wakes the consumer; finish_probe() is the
    # consumer's half of the protocol and the only thing that ends the wait.
    _probe_pending: bool = field(default=False, init=False, repr=False)
    _probe_done: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def in_flight_count(self) -> int:
        """Number of requests currently in flight on this slot."""
        return len(self.in_flight)

    def oldest_in_flight_seconds(self, now: float) -> float | None:
        """Age of the oldest in-flight request, or None if none are.
        `now` is a `time.monotonic()` reading."""
        if not self.in_flight:
            return None
        return now - min(self.in_flight.values())

    def begin_request(self, request_id: str) -> None:
        """Mark a request as in flight, from just before the backend call."""
        self.in_flight[request_id] = time.monotonic()

    def end_request(self, request_id: str) -> None:
        """End an in-flight request. Idempotent; an unknown id is ignored."""
        self.in_flight.pop(request_id, None)

    def record_error(self, code: str, message: str) -> None:
        """Record a slot fault. Set only by slot faults, never cleared."""
        self.last_error = {
            "code": code,
            "message": message,
            "at_utc": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # The one state, one writer (phase 5, Q19)
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """What the slot is doing now, most pressing first: the stored
        state, except a stored `idle` with a request in flight reads
        `serving`. `serving` is never stored: a relayed stream's closer
        runs on the response's task, not the consumer's, so a slot that
        stored `serving` could strand there after a consumer restart. A
        stored state other than idle (unhealthy, probing, swapping,
        failed) reads as itself, even with requests still in flight."""
        if self.stored_state == "idle" and self.in_flight:
            return "serving"
        return self.stored_state

    @property
    def healthy(self) -> bool:
        """True when the state is idle or serving -- the states in which
        the slot can take a request for its loaded model."""
        return self.stored_state == "idle"

    @healthy.setter
    def healthy(self, value: bool) -> None:
        """TEST-ONLY shortcut; no code in manager/ assigns it (a test
        reads the source to pin both). True stores `idle`, False stores
        `unhealthy`; it bypasses TRANSITIONS and logs nothing, so a test
        can put a slot in a state without moving it through the machine."""
        self.stored_state = "idle" if value else "unhealthy"

    def _set_state(self, new: str) -> None:
        """The only way the stored state moves. Setting the current state
        again is a no-op (no log). A move not in TRANSITIONS raises
        IllegalTransition with the state unchanged and nothing logged --
        it is a bug, not an expected path. Every real move logs one INFO
        line, `slot=<name> <from> -> <to>`, on this module's logger. A move
        out of `idle` (idle -> unhealthy or idle -> swapping) also starts a
        fresh loading grace (grace_started); moves among the other states
        never do (Task 5, Q21)."""
        if new == self.stored_state:
            return
        if new not in TRANSITIONS.get(self.stored_state, frozenset()):
            raise IllegalTransition(
                f"slot={self.name} cannot move {self.stored_state} -> {new}")
        logger.info("slot=%s %s -> %s", self.name, self.stored_state, new)
        if self.stored_state == "idle":
            # Leaving `idle` starts a fresh loading grace (Task 5, Q21):
            # from now, a 503 is a backend still loading, not a failed
            # probe, until SWAP_TIMEOUT. Moves among the other states
            # never restart it: each probe of an unhealthy slot goes
            # unhealthy -> probing -> unhealthy, and if that re-armed the
            # grace a 503 would never count.
            self.grace_started = time.monotonic()
        self.stored_state = new

    def mark_unhealthy(self) -> None:
        """Move the stored state to unhealthy (legal from idle, probing and
        unhealthy; a no-op if it is already there). From swapping or failed
        it raises IllegalTransition: a failed swap falls back or fails, and
        neither state may be marked stale behind a fault (Task 4, Q20)."""
        self._set_state("unhealthy")

    def mark_swapped(self, model: str) -> None:
        """Record a successful swap: move `swapping` -> `idle` (raising
        IllegalTransition from anywhere else -- a swap that ended without
        running from `swapping` is a bug), then record the new model, the
        swap time and the new known-good model: the slot just loaded the
        model and is about to serve it, so it is the model the slot last
        served healthily until proven otherwise (Task 4, Q20)."""
        self._set_state("idle")
        self.loaded_model = model
        self.known_good = model
        self.last_swap_utc = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Asking the consumer to probe (phase 5, Q19)
    # ------------------------------------------------------------------

    @property
    def probe_pending(self) -> bool:
        """Whether a probe has been asked for and not yet finished (the
        consumer's serve loop runs while this is true)."""
        return self._probe_pending

    def request_probe(self) -> bool:
        """Ask this slot's queue consumer to probe, waking it if needed.

        Callers that ask at the same time share one probe: the caller that
        finds none pending arms the handshake and wakes the consumer; a
        caller that finds one pending joins it instead of starting a
        second probe -- True, nothing armed, the done flag left alone, so
        the consumer's finish_probe wakes every waiter with one probe."""
        if self._probe_pending:
            return True
        self._probe_done.clear()
        self._probe_pending = True
        self.queue_event.set()
        return True

    def finish_probe(self) -> None:
        """The consumer's half of the handshake, after its probe returned
        -- success or failure; the probe is contained, and this still
        runs.

        The done flag is set BEFORE the pending flag clears: a caller
        arriving in the gap finds the probe pending and joins a probe
        that is already done, instead of a cleared flag letting it wake a
        fresh one it does not need."""
        self._probe_done.set()
        self._probe_pending = False

    async def refresh(self, timeout: float) -> bool:
        """Ask the consumer to probe this slot and wait for that probe.

        Returns False at once, without asking, when the slot is busy
        (`serving` or `swapping`: a probe there would race the in-flight
        request or the swap, and a 503 mid-load is loading, not a
        fault). Otherwise True once the probe this call asked for has
        run; False after `timeout` when the consumer never got to it.
        Callers waiting at the same time share one probe, and one caller
        timing out cancels only its own waiter -- never the probe the
        others still need.
        """
        if self.state in ("serving", "swapping"):
            return False
        self.request_probe()
        try:
            await asyncio.wait_for(self._probe_done.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        return True

    # ------------------------------------------------------------------
    # Probing the backend
    # ------------------------------------------------------------------

    async def probe(self, client) -> str:
        """Query /v1/models on the slot's backend and update the state.

        Never raises. On any failure (connection, timeout, non-200, non-JSON
        body, or a 200 whose JSON is not shaped like the /v1/models response
        we expect -- a non-object body, a non-list or missing "data" field, a
        non-object first entry, or a first entry whose "id" is present and
        truthy but not a string -- e.g. some unrelated HTTP service answering
        on a collided port), records a probe_failed error and moves the
        stored state to unhealthy -- unless it is already `failed`, which
        stays `failed` (Task 4, Q20) -- leaving loaded_model as whatever it
        was — the last-known loaded model is still useful for status
        reporting until a successful probe or swap updates it.

        Two branches intentionally do NOT follow that "leave loaded_model
        alone" rule, and null it instead:
          - a well-formed empty "data" list -- the backend is actively
            telling us nothing is loaded, which is different from a
            malformed response we can't interpret at all;
          - a "data" field that is present but not a list, or missing
            entirely (falsy) -- pre-fix, `data.get("data") or []` funnelled
            {"data": null}/a missing "data" key/{"data": {}} into the
            empty-data branch above, nulling loaded_model. Post-fix this is
            a DELIBERATE classification change: those payloads are treated
            as malformed (unrecognizable), not as "backend says nothing
            loaded", so loaded_model is now preserved for them instead.
            Flagging this explicitly since the chat 409 path asks for the
            same probe through refresh and would see the same behavior.

        State moves (phase 5, Q19): an unhealthy slot moves to probing
        before the call and to idle (a usable model) or back to unhealthy
        after it; a failed slot (Task 4, Q20) stays failed while the call
        runs -- a healthy probe moves it to idle, a failed probe leaves it
        failed; an idle slot stays idle while the call runs and moves to
        unhealthy only if it fails, so a routine refresh of a healthy slot
        logs nothing.

        Returns one of the supervisor's probe results (Task 5, Q21):
        PROBE_HEALTHY when a usable model is reported, PROBE_LOADING for an
        HTTP 503 (the backend is still loading its model), and
        PROBE_FAILED for anything else -- a connection failure, any other
        status, a body the guards above reject, or a 200 with no usable
        model id. The consumer hands the result to judge
        (manager/supervisor.py); what the probe records and how it moves
        the state are unchanged by the return value.

        Stamps last_probe_at (phase 6, Task 6) with time.monotonic() read
        after the backend call has returned or raised, on every return
        path, whoever asked for the probe.
        """
        if self.stored_state == "unhealthy":
            self._set_state("probing")
        try:
            resp = await client.get(f"{self.url}/v1/models", timeout=3)
        except Exception as exc:
            self.last_probe_at = time.monotonic()
            logger.warning("slot %s probe failed: %s", self.name, exc)
            self._probe_failed(f"probe failed: {exc}")
            return PROBE_FAILED

        self.last_probe_at = time.monotonic()
        if resp.status_code != 200:
            logger.warning("slot %s probe returned %s", self.name, resp.status_code)
            self._probe_failed(f"probe returned {resp.status_code}")
            # A 503 is the backend still loading its model (Task 5, Q21):
            # the judge exempts it while the loading grace runs, and only
            # then does it count as a failure.
            return PROBE_LOADING if resp.status_code == 503 else PROBE_FAILED

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("slot %s probe returned non-JSON: %s", self.name, exc)
            self._probe_failed(f"probe returned non-JSON: {exc}")
            return PROBE_FAILED

        # A 200 with a JSON body that isn't shaped like /v1/models must not
        # raise -- guard each shape assumption instead of indexing/`.get`-ing
        # a value that might not be a dict or list (e.g. a different service
        # answering on this port, or a future backend API change).
        if not isinstance(data, dict):
            logger.warning(
                "slot %s /v1/models returned malformed payload (not an object): %s",
                self.name, type(data).__name__,
            )
            self._probe_failed(
                f"/v1/models returned malformed payload "
                f"(not an object): {type(data).__name__}",
            )
            return PROBE_FAILED

        entries = data.get("data")
        if not isinstance(entries, list):
            logger.warning(
                "slot %s /v1/models 'data' field is malformed (not a list): %s",
                self.name, type(entries).__name__,
            )
            self._probe_failed(
                f"/v1/models 'data' field is malformed "
                f"(not a list): {type(entries).__name__}",
            )
            return PROBE_FAILED

        if not entries:
            logger.warning("slot %s /v1/models returned no entries", self.name)
            self.loaded_model = None
            self._probe_failed("/v1/models returned no entries")
            return PROBE_FAILED

        first = entries[0]
        if not isinstance(first, dict):
            logger.warning(
                "slot %s /v1/models first entry is malformed (not an object): %s",
                self.name, type(first).__name__,
            )
            self._probe_failed(
                f"/v1/models first entry is malformed "
                f"(not an object): {type(first).__name__}",
            )
            return PROBE_FAILED

        raw_id = first.get("id")
        if raw_id and not isinstance(raw_id, str):
            # display_name() unconditionally calls raw.rsplit("/", 1)
            # (manager/names.py) -- a truthy non-string id would raise
            # there. Guard here rather than in display_name(), whose
            # string-in contract other callers rely on.
            logger.warning(
                "slot %s /v1/models first entry 'id' is malformed (not a string): %s",
                self.name, type(raw_id).__name__,
            )
            self._probe_failed(
                f"/v1/models first entry 'id' is malformed "
                f"(not a string): {type(raw_id).__name__}",
            )
            return PROBE_FAILED

        # Normalize to the clean display form (basename, no .gguf, original case).
        # display_name also collapses a full path to its stem, falsy input
        # (missing/empty id), and "" -> None.
        self.loaded_model = display_name(raw_id or "")
        if self.loaded_model:
            # Every healthy probe records the model as the slot's
            # known-good (Task 4, Q20): the fallback target of a later
            # failed swap.
            self.known_good = self.loaded_model
            if self.stored_state in ("probing", "failed"):
                # The slot was unhealthy or failed when the probe ran; a
                # usable model puts it back. An idle slot stays idle here,
                # so a routine refresh of a healthy slot logs nothing.
                self._set_state("idle")
            return PROBE_HEALTHY
        else:
            # The backend answered but has no model we can name (missing or
            # empty id): the same treatment as the empty-data branch above.
            logger.warning(
                "slot %s /v1/models returned no usable model id", self.name)
            self._probe_failed("/v1/models returned no usable model id")
            return PROBE_FAILED

    def _probe_failed(self, message: str) -> None:
        """Record a failed probe and move the state (Task 4, Q20): a failed
        slot stays failed -- its recovery is a /swap or a healthy probe, not
        a fresh fault -- while every other state moves to unhealthy."""
        self.record_error("probe_failed", message)
        if self.stored_state != "failed":
            self.mark_unhealthy()

    async def reconcile_on_error(self, client) -> None:
        """Re-probe after a backend 5xx. Updates loaded_model and state."""
        await self.probe(client)

    def to_status_dict(self) -> dict:
        """Shape for the /status endpoint's slots section."""
        now = time.monotonic()
        return {
            "host": self.host,
            "port": self.port,
            "loaded_model": self.loaded_model,
            "healthy": self.healthy,
            "swapping": self.state == "swapping",
            "last_swap_utc": self.last_swap_utc,
            "queue_depth": self.queue.depth,
            "queue_limit": self.queue.max_size,
            "state": self.state,
            "in_flight": self.in_flight_count,
            "in_flight_age_seconds": _round_1(self.oldest_in_flight_seconds(now)),
            "oldest_queue_wait_seconds": _round_1(
                self.queue.oldest_wait_seconds(now)),
            # How long ago the last probe's backend call returned or failed
            # (phase 6, Task 6); None until the slot's first probe. /status
            # no longer probes, so this is the endpoint's honest answer to
            # "how fresh is that state".
            "last_probe_age_seconds": _round_1(
                None if self.last_probe_at is None
                else now - self.last_probe_at),
            "last_error": self.last_error,
            "last_swap": self.last_swap,
        }


def _round_1(seconds: float | None) -> float | None:
    """Round an age to 1 decimal; None (nothing to measure) stays None."""
    return round(seconds, 1) if seconds is not None else None
