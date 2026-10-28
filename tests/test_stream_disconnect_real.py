"""A client that leaves before the first chunk still closes the REAL backend connection.

Local SSE server on an ephemeral 127.0.0.1 port (never a live slot port), real
httpx. This is the one disconnect path where the manager's own close is the
only closer: the relay generator never starts, so no httpx read is cancelled
and httpcore never closes the connection by itself (when a read IS cancelled
mid-stream, httpcore closes the connection on its own, so that path cannot
show whether the manager's close works). Without StreamingResponse's
background close, the connection stays open and llama-server keeps generating.
"""
import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from manager.app import ServerState, _queue_consumer


@pytest.mark.asyncio
async def test_client_leaving_before_the_first_chunk_closes_the_backend(test_config):
    closed_at: list[float] = []

    async def sse_backend(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                     b"transfer-encoding: chunked\r\n\r\n")
        await writer.drain()
        while await reader.read(4096):   # drain the request; b"" means the peer closed
            pass
        closed_at.append(time.monotonic())
        writer.close()

    srv = await asyncio.start_server(sse_backend, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    try:
        server = ServerState(test_config)
        slot = server.slots["main"]
        slot.port = port
        slot.loaded_model, slot.healthy = "test-model-q4", True
        slot.swapper.swap_to = AsyncMock(return_value=True)

        item = {"body": {"model": "test-model-q4", "stream": True, "messages": []},
                "event": asyncio.Event(), "response": None, "error": None,
                "error_code": None, "backend_path": "v1/chat/completions"}
        await slot.queue.enqueue(item)
        task = asyncio.create_task(_queue_consumer(server, server._config, slot_name="main"))
        slot.queue_event.set()
        await asyncio.wait_for(item["event"].wait(), 5)

        async def receive():
            return {"type": "http.disconnect"}      # client is already gone

        async def send(message):
            if message["type"] == "http.response.start":
                await asyncio.sleep(0.05)           # disconnect wins before iteration

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
                 "method": "POST", "path": "/v1/chat/completions", "headers": []}
        await asyncio.wait_for(item["response"](scope, receive, send), 5)
        left_at = time.monotonic()

        for _ in range(200):
            if closed_at:
                break
            await asyncio.sleep(0.01)
        task.cancel()
    finally:
        srv.close()

    assert closed_at, "backend connection still open 2s after the client left"
    assert closed_at[0] - left_at < 1.0
