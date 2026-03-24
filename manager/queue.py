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

    async def dequeue(self) -> Any:
        """Remove and return the next request. Blocks until available."""
        item = await self._queue.get()
        self._depth -= 1
        return item

    def drain(self) -> list:
        """Remove and return all items. Used on error to send 503s."""
        items = []
        while not self._queue.empty():
            items.append(self._queue.get_nowait())
            self._depth -= 1
        return items
