"""Tests for the FIFO request queue."""
import asyncio
import pytest
from manager.queue import RequestQueue


@pytest.mark.asyncio
async def test_queue_processes_in_fifo_order():
    """Requests should be processed in the order they were added."""
    queue = RequestQueue(max_size=20)
    results = []

    await queue.enqueue("first")
    await queue.enqueue("second")
    await queue.enqueue("third")

    while not queue.empty():
        item = await queue.dequeue()
        results.append(item)

    assert results == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_queue_rejects_when_full():
    """Queue should raise when at capacity to trigger 503 response."""
    queue = RequestQueue(max_size=2)

    await queue.enqueue("first")
    await queue.enqueue("second")

    with pytest.raises(queue.QueueFullError):
        await queue.enqueue("third")


@pytest.mark.asyncio
async def test_queue_reports_depth():
    """Queue should accurately report how many items are waiting."""
    queue = RequestQueue(max_size=20)

    assert queue.depth == 0
    await queue.enqueue("item")
    assert queue.depth == 1
    await queue.dequeue()
    assert queue.depth == 0


@pytest.mark.asyncio
async def test_queue_drain_returns_all_items():
    """Drain should empty the queue and return all items."""
    queue = RequestQueue(max_size=20)

    await queue.enqueue("first")
    await queue.enqueue("second")
    await queue.enqueue("third")

    items = queue.drain()

    assert items == ["first", "second", "third"]
    assert queue.depth == 0


@pytest.mark.asyncio
async def test_queue_dequeue_waits_for_item():
    """Dequeue should block until an item is available."""
    queue = RequestQueue(max_size=20)
    result = []

    async def consumer():
        item = await queue.dequeue()
        result.append(item)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.05)
    assert result == []

    await queue.enqueue("hello")
    await asyncio.sleep(0.05)
    assert result == ["hello"]
    await task
