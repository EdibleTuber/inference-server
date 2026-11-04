# tests/test_client_gone_real.py
"""Phase 5 Task 6 (Q26) on the real stack: real uvicorn with httptools (what
the deployed unit gets, since it passes no --http and httptools is
installed), real sockets, the real app and lifespan, and a fake llama-server
on an ephemeral 127.0.0.1 port (never a live slot port).

A request for main is held at the fake backend. Behind it, clients send
requests and close their sockets while queued: four with a 1 MB body (over
uvicorn's 64 KiB read limit, so reading is paused when the client closes)
and one with a small body. Once the held request is answered, every queued
request must be skipped: 499 client_gone, and the backend sees only the one
request it held.

Against a single is_disconnected() check, the large bodies are served: a
single check misses them about 17 times in 20 (measured 2026-11-04), so all
four are caught by one check only about once in 2000 runs.
"""
import asyncio
import contextlib
import json
import logging

import httpx
import pytest
import uvicorn

from manager import app as app_mod

MODEL = "test-model-q4"
BIG = 1_000_000
GONE = [("gone-big-1", BIG), ("gone-big-2", BIG), ("gone-big-3", BIG),
        ("gone-big-4", BIG), ("gone-small", 1_000)]


class FakeLlama:
    """GET /v1/models names MODEL; the first chat POST is held until
    `release` is set; every chat POST is counted."""

    def __init__(self):
        self.chat_posts = 0
        self.release = asyncio.Event()

    async def handle(self, reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            method, path = head.split(b" ", 2)[:2]
            length = 0
            for h in head.split(b"\r\n")[1:]:
                if h.lower().startswith(b"content-length:"):
                    length = int(h.split(b":", 1)[1])
            if length:
                await reader.readexactly(length)
            if path.startswith(b"/v1/models"):
                payload = {"object": "list", "data": [{"id": MODEL}]}
            elif method == b"POST" and path.startswith(b"/v1/chat/completions"):
                self.chat_posts += 1
                if self.chat_posts == 1:
                    await self.release.wait()
                payload = {"choices": [{"index": 0, "finish_reason": "stop",
                                        "message": {"role": "assistant",
                                                    "content": "hi"}}]}
            else:
                payload = {}
            data = json.dumps(payload).encode()
            writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                         b"connection: close\r\ncontent-length: %d\r\n\r\n" % len(data)
                         + data)
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()


def chat_request(rid: str, pad: int) -> bytes:
    body = json.dumps({"model": MODEL, "messages": [
        {"role": "user", "content": "x" * pad}]}).encode()
    return (b"POST /v1/chat/completions HTTP/1.1\r\nhost: 127.0.0.1\r\n"
            b"content-type: application/json\r\nx-request-id: %s\r\n"
            b"content-length: %d\r\n\r\n" % (rid.encode(), len(body)) + body)


async def until(cond, timeout: float, what: str) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not cond():
        if loop.time() > deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what}")
        await asyncio.sleep(0.02)


def lines_by_id(caplog) -> dict[str, dict]:
    out = {}
    for r in caplog.records:
        if r.name == "manager.requests":
            line = json.loads(r.getMessage())
            if line["event"] == "request":
                out[line["id"]] = line
    return out


@pytest.mark.asyncio
async def test_clients_that_leave_while_queued_are_skipped_on_the_real_stack(
        test_config, caplog):
    caplog.set_level(logging.INFO, logger="manager.requests")
    llama = FakeLlama()
    backend = await asyncio.start_server(llama.handle, "127.0.0.1", 0)
    app = app_mod.create_app(test_config)
    main = app.state.server.slots["main"]
    main.port = backend.sockets[0].getsockname()[1]

    srv = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=0, http="httptools", ws="none",
        lifespan="on", log_level="warning"))
    serving = asyncio.create_task(srv.serve())
    writers = []
    try:
        await until(lambda: srv.started, 10, "uvicorn to start")
        port = srv.servers[0].sockets[0].getsockname()[1]
        await until(lambda: main.stored_state == "idle", 5,
                    "the startup probe to bring main up")

        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}",
                                     timeout=20) as client:
            held = asyncio.create_task(client.post(
                "/v1/chat/completions", headers={"x-request-id": "held"},
                json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}))
            await until(lambda: llama.chat_posts == 1, 5, "the held request")

            for rid, pad in GONE:
                _, w = await asyncio.open_connection("127.0.0.1", port)
                w.write(chat_request(rid, pad))
                await w.drain()
                writers.append(w)
            await until(lambda: main.queue.depth == len(GONE), 10,
                        "every gone client to be queued")
            for w in writers:
                w.close()
            for w in writers:
                with contextlib.suppress(ConnectionError):
                    await w.wait_closed()
            await asyncio.sleep(0.3)

            llama.release.set()
            r = await held
            assert r.status_code == 200

            ids = [rid for rid, _ in GONE]
            await until(lambda: all(i in lines_by_id(caplog) for i in ids), 10,
                        "a line for every gone client")
    finally:
        llama.release.set()
        srv.should_exit = True
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(serving, 15)
        backend.close()

    lines = lines_by_id(caplog)
    assert {i: (lines[i]["status"], lines[i]["outcome"]) for i in ids} == {
        i: (499, "client_gone") for i in ids}
    assert all(lines[i]["backend_ms"] is None for i in ids)
    assert llama.chat_posts == 1
    assert main.in_flight == {}
