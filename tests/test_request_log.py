# tests/test_request_log.py
"""Unit tests for manager/request_log.py (phase 4, Task 2; Q27).

Sample bodies are llama.cpp build 11136's own output, captured through the
manager on 2026-11-01 (see the plan's Decisions for the token rules).
"""
import json
import logging

import pytest
from starlette.requests import Request

from manager.request_log import (
    USER_AGENT_MAX, RequestRecord, StreamTokens, Tokens, request_id_from,
    tokens_from_body,
)

KEYS = ["event", "id", "endpoint", "client", "user_agent", "slot", "model", "stream",
        "status", "outcome", "queue_wait_ms", "backend_ms", "prompt_tokens",
        "cached_tokens", "completion_tokens"]

OPENAI_BODY = {
    "choices": [{"finish_reason": "length", "index": 0,
                 "message": {"role": "assistant", "content": "Hi"}}],
    "usage": {"completion_tokens": 4, "prompt_tokens": 12, "total_tokens": 16,
              "prompt_tokens_details": {"cached_tokens": 8}},
    "timings": {"cache_n": 8, "prompt_n": 4, "prompt_ms": 161.381, "predicted_n": 4,
                "predicted_ms": 65.343},
}
OPENAI_LAST_CHUNK = (
    b'data: {"choices":[{"finish_reason":"length","index":0,"delta":{}}],'
    b'"object":"chat.completion.chunk","timings":{"cache_n":0,"prompt_n":12,'
    b'"prompt_ms":243.189,"predicted_n":8,"predicted_ms":159.759}}\n\n'
)
OPENAI_STREAM = (
    b'data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"Hi"}}],'
    b'"object":"chat.completion.chunk"}\n\n'
    + OPENAI_LAST_CHUNK + b"data: [DONE]\n\n"
)
ANTHROPIC_BODY = {
    "type": "message", "role": "assistant", "content": [{"type": "text", "text": "Hi"}],
    "stop_reason": "max_tokens",
    "usage": {"cache_read_input_tokens": 8, "input_tokens": 4, "output_tokens": 4},
}
ANTHROPIC_STREAM = (
    b'event: message_start\ndata: {"type":"message_start","message":{"type":"message",'
    b'"role":"assistant","content":[],"usage":{"cache_read_input_tokens":8,'
    b'"input_tokens":4,"output_tokens":0}}}\n\n'
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"Hi"}}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":'
    b'"max_tokens"},"usage":{"output_tokens":2}}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":'
    b'"max_tokens"},"usage":{"output_tokens":4}}\n\n'
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


def _tokens(t: Tokens) -> tuple:
    return (t.prompt, t.cached, t.completion)


# --- request ids ----------------------------------------------------------------

@pytest.mark.parametrize("good", ["abc-123", "A.b_c:9", "x" * 64])
def test_a_well_formed_request_id_is_kept(good):
    assert request_id_from(good) == good


@pytest.mark.parametrize("bad", [None, "", "a b", "a\nb", 'a"b', "x" * 65, "ü"])
def test_anything_else_gets_a_fresh_hex_id(bad):
    rid = request_id_from(bad)
    assert len(rid) == 32 and all(c in "0123456789abcdef" for c in rid)


def test_fresh_ids_differ():
    assert request_id_from(None) != request_id_from(None)


# --- tokens from a whole body ---------------------------------------------------

def test_openai_tokens_from_timings():
    """timings, not usage: prompt = prompt_n + cache_n."""
    assert _tokens(tokens_from_body(json.dumps(OPENAI_BODY).encode(), "openai")) == (12, 8, 4)


def test_openai_tokens_fall_back_to_usage():
    body = {k: v for k, v in OPENAI_BODY.items() if k != "timings"}
    body["usage"] = {"completion_tokens": 5, "prompt_tokens": 30,
                     "prompt_tokens_details": {"cached_tokens": 10}}
    assert _tokens(tokens_from_body(json.dumps(body).encode(), "openai")) == (30, 10, 5)


def test_anthropic_tokens_from_usage():
    """prompt = input_tokens + cache_read_input_tokens."""
    assert _tokens(tokens_from_body(json.dumps(ANTHROPIC_BODY).encode(), "anthropic")) == (12, 8, 4)


def test_anthropic_usage_without_cache_reads_counts_zero_cached():
    body = dict(ANTHROPIC_BODY, usage={"input_tokens": 7, "output_tokens": 3})
    assert _tokens(tokens_from_body(json.dumps(body).encode(), "anthropic")) == (7, 0, 3)


def test_count_tokens_gives_only_the_prompt():
    assert _tokens(tokens_from_body(b'{"input_tokens":12}', "count_tokens")) == (12, None, None)


@pytest.mark.parametrize("body", [
    b'{"usage": {"prompt_tokens": 1',                       # truncated
    b"<html><body>502 Bad Gateway</body></html>",
    b"",
    b"[1, 2, 3]",
    b'{"timings": "nope", "usage": [1]}',
    b'{"timings": {"prompt_n": "12", "cache_n": null, "predicted_n": 1.5}}',
    b'{"usage": {"input_tokens": true, "output_tokens": 3}}',
])
@pytest.mark.parametrize("dialect", ["openai", "anthropic", "count_tokens"])
def test_unparseable_bodies_give_null_tokens(body, dialect):
    t = tokens_from_body(body, dialect)
    for value in _tokens(t):
        assert value is None or (type(value) is int and value >= 0)


# --- tokens from a stream -------------------------------------------------------

def test_openai_stream_tokens_come_from_the_last_chunks_timings():
    st = StreamTokens("openai")
    st.feed(OPENAI_STREAM)
    assert _tokens(st.tokens) == (12, 0, 8)


def test_stream_tokens_survive_a_line_split_across_chunks():
    """Review Focus 1: the backend's write boundary can fall anywhere,
    including inside the one line that carries the counts."""
    for cut in range(1, len(OPENAI_STREAM)):
        st = StreamTokens("openai")
        st.feed(OPENAI_STREAM[:cut])
        st.feed(OPENAI_STREAM[cut:])
        assert _tokens(st.tokens) == (12, 0, 8), f"split at byte {cut}"


def test_stream_tokens_survive_one_byte_at_a_time():
    st = StreamTokens("openai")
    for i in range(len(OPENAI_STREAM)):
        st.feed(OPENAI_STREAM[i:i + 1])
    assert _tokens(st.tokens) == (12, 0, 8)


def test_anthropic_stream_tokens_come_from_message_start_and_the_last_delta():
    st = StreamTokens("anthropic")
    st.feed(ANTHROPIC_STREAM)
    assert _tokens(st.tokens) == (12, 8, 4)


def test_anthropic_stream_split_at_every_offset():
    for cut in range(1, len(ANTHROPIC_STREAM)):
        st = StreamTokens("anthropic")
        st.feed(ANTHROPIC_STREAM[:cut])
        st.feed(ANTHROPIC_STREAM[cut:])
        assert _tokens(st.tokens) == (12, 8, 4), f"split at byte {cut}"


def test_a_stream_without_counts_gives_null_tokens():
    st = StreamTokens("openai")
    st.feed(b'data: {"choices":[]}\n\ndata: [DONE]\n\n')
    assert _tokens(st.tokens) == (None, None, None)


def test_an_oversized_line_is_dropped_and_parsing_resumes():
    """Review Focus 3: a 2 MiB line with no newline must not grow the buffer
    without bound, and the counts on a later line still come through."""
    st = StreamTokens("openai")
    st.feed(b'data: {"timings": "' + b"x" * (2 * 1024 * 1024))
    st.feed(b'"}\n\n' + OPENAI_LAST_CHUNK)
    assert _tokens(st.tokens) == (12, 0, 8)


def test_a_line_with_no_newline_cannot_grow_memory_without_bound():
    """Review Focus 3: 32 MiB with no newline must not be held in memory."""
    import tracemalloc
    st = StreamTokens("openai")
    block = b"y" * (1024 * 1024)
    tracemalloc.start()
    try:
        for _ in range(32):
            st.feed(block)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * 1024 * 1024, f"peak {peak / 2**20:.1f} MiB"


@pytest.mark.parametrize("junk", [
    b'data: {"timings": {"prompt_n": \n\n',
    b"data: <html>usage</html>\n\n",
    b'data: {"usage": {"output_tokens": "4"}}\n\n',
    b"\xff\xfe\x00timings\n\n",
])
def test_stream_junk_never_raises(junk):
    st = StreamTokens("anthropic")
    st.feed(junk)
    st = StreamTokens("openai")
    st.feed(junk)
    assert _tokens(st.tokens) == (None, None, None)


# --- the record and its line ----------------------------------------------------

def _request(headers: list[tuple[bytes, bytes]], client=("192.168.1.32", 40000)):
    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions",
             "headers": headers, "client": client, "query_string": b""}
    return Request(scope)


def _lines(caplog) -> list[dict]:
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == "manager.requests"]


def test_from_request_reads_the_client_the_agent_and_the_body(caplog):
    req = _request([(b"user-agent", b"agent_core/1.12.1"), (b"x-request-id", b"abc-123")])
    rec = RequestRecord.from_request(req, "/v1/chat/completions",
                                     {"model": "test-model-q4", "stream": True})
    assert (rec.id, rec.client, rec.user_agent, rec.model, rec.stream) == (
        "abc-123", "192.168.1.32", "agent_core/1.12.1", "test-model-q4", True)


def test_from_request_tolerates_a_missing_body_client_and_agent():
    rec = RequestRecord.from_request(_request([], client=None), "/v1/messages", None)
    assert (rec.client, rec.user_agent, rec.model, rec.stream) == (None, None, None, False)
    assert len(rec.id) == 32


def test_a_non_string_model_is_not_logged_as_one():
    rec = RequestRecord.from_request(_request([]), "/v1/chat/completions",
                                     {"model": {"x": 1}, "stream": "yes"})
    assert rec.model is None and rec.stream is False


def test_finish_writes_one_line_with_every_key_in_order(caplog):
    caplog.set_level(logging.INFO, logger="manager.requests")
    rec = RequestRecord.from_request(_request([]), "/v1/chat/completions",
                                     {"model": "m", "stream": False})
    rec.finish(200, "ok")
    rec.finish(500, "backend_error")              # idempotent: ignored
    (line,) = _lines(caplog)
    assert list(line) == KEYS
    assert (line["event"], line["status"], line["outcome"]) == ("request", 200, "ok")
    assert line["queue_wait_ms"] is None and line["backend_ms"] is None


def test_the_line_is_compact_ascii_json(caplog):
    caplog.set_level(logging.INFO, logger="manager.requests")
    rec = RequestRecord.from_request(_request([(b"user-agent", "café".encode())]),
                                     "/v1/chat/completions", {"model": "m"})
    rec.finish(200, "ok")
    (record,) = [r for r in caplog.records if r.name == "manager.requests"]
    msg = record.getMessage()
    assert record.levelno == logging.INFO
    assert msg.isascii() and ", " not in msg and '": ' not in msg


def test_hostile_headers_cannot_split_or_bloat_the_line(caplog):
    """Review Focus 4. The user agent is truncated; JSON escapes the rest."""
    caplog.set_level(logging.INFO, logger="manager.requests")
    ua = b'evil\n"agent\r\n' + b"A" * 4096
    req = _request([(b"user-agent", ua), (b"x-request-id", b'x"\ny')])
    rec = RequestRecord.from_request(req, "/v1/chat/completions", {"model": "m"})
    rec.finish(200, "ok")
    (record,) = [r for r in caplog.records if r.name == "manager.requests"]
    msg = record.getMessage()
    assert "\n" not in msg and "\r" not in msg
    line = json.loads(msg)
    assert len(line["user_agent"]) <= USER_AGENT_MAX == 120
    assert line["user_agent"].startswith('evil\n"agent')
    assert len(line["id"]) == 32 and line["id"] != 'x"\ny'


def test_timings_become_whole_milliseconds(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="manager.requests")
    import manager.request_log as rl
    rec = RequestRecord.from_request(_request([]), "/v1/chat/completions", {"model": "m"})
    rec.enqueued_at, rec.dequeued_at, rec.backend_started_at = 100.0, 100.25, 100.5
    monkeypatch.setattr(rl.time, "monotonic", lambda: 101.7504)
    rec.tokens = Tokens(12, 0, 8)
    rec.finish(200, "ok")
    (line,) = _lines(caplog)
    assert (line["queue_wait_ms"], line["backend_ms"]) == (250, 1250)
    assert (line["prompt_tokens"], line["cached_tokens"], line["completion_tokens"]) == (12, 0, 8)


def test_a_raising_log_write_does_not_raise_from_finish(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("journal gone")
    monkeypatch.setattr(logging.getLogger("manager.requests"), "info", boom)
    rec = RequestRecord.from_request(_request([]), "/v1/chat/completions", {"model": "m"})
    rec.finish(200, "ok")                          # must not raise
