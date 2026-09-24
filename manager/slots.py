"""
Per-slot state for the model manager.

Each inference backend has its own SlotState containing loaded-model
tracking, health, swap lock, queue, and an event the handler signals when a
new item is enqueued. Which backends exist is configuration, not code. Swap
operations and routing decisions read/write this state in-process.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from manager.names import display_name
from manager.queue import RequestQueue

logger = logging.getLogger(__name__)


@dataclass
class SlotState:
    """State container for one inference slot.

    A slot is a backing llama-server process that the manager fronts.
    'loaded_model' reflects what that process currently has loaded;
    'healthy' is a boolean derived from the most recent probe.

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
    healthy: bool = False
    last_swap_utc: Optional[str] = None
    queue_event: asyncio.Event = field(default_factory=asyncio.Event)
    swap_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def probe(self, client) -> None:
        """Query /v1/models on the slot's backend and update state.

        Never raises. On any failure (connection, timeout, non-200, non-JSON
        body, or a 200 whose JSON is not shaped like the /v1/models response
        we expect -- a non-object body, a non-list or missing "data" field, a
        non-object first entry, or a first entry whose "id" is present and
        truthy but not a string -- e.g. some unrelated HTTP service answering
        on a collided port), sets healthy=False and leaves loaded_model as
        whatever it was — the last-known loaded model is still useful for
        status reporting until a successful probe or swap updates it.

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
            Flagging this explicitly since it's shared with the chat 409
            path's reprobe and changes prior behavior.
        """
        try:
            resp = await client.get(f"{self.url}/v1/models", timeout=3)
        except Exception as exc:
            logger.warning("slot %s probe failed: %s", self.name, exc)
            self.healthy = False
            return

        if resp.status_code != 200:
            logger.warning("slot %s probe returned %s", self.name, resp.status_code)
            self.healthy = False
            return

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("slot %s probe returned non-JSON: %s", self.name, exc)
            self.healthy = False
            return

        # A 200 with a JSON body that isn't shaped like /v1/models must not
        # raise -- guard each shape assumption instead of indexing/`.get`-ing
        # a value that might not be a dict or list (e.g. a different service
        # answering on this port, or a future backend API change).
        if not isinstance(data, dict):
            logger.warning(
                "slot %s /v1/models returned malformed payload (not an object): %s",
                self.name, type(data).__name__,
            )
            self.healthy = False
            return

        entries = data.get("data")
        if not isinstance(entries, list):
            logger.warning(
                "slot %s /v1/models 'data' field is malformed (not a list): %s",
                self.name, type(entries).__name__,
            )
            self.healthy = False
            return

        if not entries:
            logger.warning("slot %s /v1/models returned no entries", self.name)
            self.loaded_model = None
            self.healthy = False
            return

        first = entries[0]
        if not isinstance(first, dict):
            logger.warning(
                "slot %s /v1/models first entry is malformed (not an object): %s",
                self.name, type(first).__name__,
            )
            self.healthy = False
            return

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
            self.healthy = False
            return

        # Normalize to the clean display form (basename, no .gguf, original case).
        # display_name also collapses a full path to its stem, falsy input
        # (missing/empty id), and "" -> None.
        self.loaded_model = display_name(raw_id or "")
        self.healthy = bool(self.loaded_model)

    async def reconcile_on_error(self, client) -> None:
        """Re-probe after a backend 5xx. Updates loaded_model and healthy."""
        await self.probe(client)

    def mark_unhealthy(self) -> None:
        self.healthy = False

    def mark_swapped(self, model: str) -> None:
        """Record a successful swap: update loaded_model, last_swap_utc, healthy."""
        self.loaded_model = model
        self.last_swap_utc = datetime.now(timezone.utc).isoformat()
        self.healthy = True

    def to_status_dict(self) -> dict:
        """Shape for the /status endpoint's slots section."""
        return {
            "host": self.host,
            "port": self.port,
            "loaded_model": self.loaded_model,
            "healthy": self.healthy,
            "last_swap_utc": self.last_swap_utc,
            "queue_depth": self.queue.depth,
            "queue_limit": self.queue.max_size,
        }
