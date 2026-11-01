"""
FIFO request queue for the model manager.

All incoming inference requests go through this queue, processed one at a
time. This ensures serial access to llama-server (single GPU, one request
at a time maximizes throughput per request).

During model swaps, requests accumulate and are processed once the swap
completes. If the queue hits its max depth, new requests are rejected
with QueueFullError (the API layer translates this to 503).
"""
import asyncio
import time
from collections import deque
from typing import Any


class RequestQueue:
    """Async FIFO queue with a configurable size limit."""

    class QueueFullError(Exception):
        """Raised when the queue is at capacity."""
        pass

    def __init__(self, max_size: int = 20):
        self._max_size = max_size
        self._queue: asyncio.Queue = asyncio.Queue()
        self._depth = 0
        # Enqueue time of every queued item, in step with _queue: the head's
        # age (oldest_wait_seconds) comes from its front.
        self._stamps: deque[float] = deque()

    @property
    def depth(self) -> int:
        """Number of requests currently waiting."""
        return self._depth

    @property
    def max_size(self) -> int:
        """Maximum queue capacity."""
        return self._max_size

    def empty(self) -> bool:
        """True if no requests are waiting."""
        return self._depth == 0

    def oldest_wait_seconds(self, now: float) -> float | None:
        """How long the item at the head of the queue has been waiting, or
        None if the queue is empty. `now` is a `time.monotonic()` reading."""
        if not self._stamps:
            return None
        return now - self._stamps[0]

    async def enqueue(self, item: Any) -> None:
        """Add a request to the back of the queue.

        Raises QueueFullError if at capacity.
        """
        if self._depth >= self._max_size:
            raise self.QueueFullError(
                f"Queue is full ({self._depth}/{self._max_size})"
            )
        await self._queue.put(item)
        self._depth += 1
        # Stamp only on success: an item refused above leaves no stamp.
        self._stamps.append(time.monotonic())

    async def dequeue(self) -> Any:
        """Remove and return the next request. Blocks until available."""
        item = await self._queue.get()
        self._depth -= 1
        self._stamps.popleft()
        return item

    def drain(self) -> list:
        """Remove and return all items. Used on error to send 503s."""
        items = []
        while not self._queue.empty():
            items.append(self._queue.get_nowait())
            self._depth -= 1
        # Clear every stamp in step with the items so nothing stale survives.
        self._stamps.clear()
        return items
