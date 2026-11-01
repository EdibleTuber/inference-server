# manager/request_log.py
"""One structured line per inference request and per swap (phase 4, Q27).

The lines are INFO records on the dedicated `manager.requests` logger, so
the journal can be sliced per event type:

    journalctl -u llama-manager -o cat | sed -n 's/^INFO manager.requests: //p' | jq

Everything in this module is observation only: nothing here may raise into
the request path. Token sniffing only reads bytes the caller already owns
(relayed chunks pass through unchanged), every parse is wrapped, and a
failing log write is swallowed (with a warning) rather than propagated.
"""
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

if TYPE_CHECKING:
    from starlette.requests import Request

REQUEST_LOGGER = logging.getLogger("manager.requests")

REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,64}")

# A 4 KB hostile User-Agent must not be able to bloat a line. The rest of
# the line stays one line because JSON escapes it; truncation bounds it.
USER_AGENT_MAX = 120

# A pending SSE line longer than this is dropped: a hostile or broken
# backend must not be able to grow the per-stream buffer without bound.
_MAX_PENDING = 1024 * 1024


def request_id_from(header: str | None) -> str:
    """The caller's X-Request-Id if it is safe to log and echo, else a fresh one.

    A kept id must survive a JSON string, a log line and a response header
    verbatim, so only [A-Za-z0-9._:-]{1,64} passes.
    """
    if header is not None and REQUEST_ID_RE.fullmatch(header):
        return header
    return uuid4().hex


@dataclass
class Tokens:
    """Token counts from the backend's own fields; None means unknown."""

    prompt: int | None = None
    cached: int | None = None
    completion: int | None = None


def _as_count(value: object) -> int | None:
    """A non-negative int, or None. A bool, str, float or null is not a count."""
    if type(value) is int and value >= 0:
        return value
    return None


def _openai_tokens(data: dict) -> Tokens:
    """llama-server's own fields: `timings` (prompt = prompt_n + cache_n),
    with the standard `usage` block as the fallback."""
    timings = data.get("timings")
    if isinstance(timings, dict):
        prompt_n = _as_count(timings.get("prompt_n"))
        cache_n = _as_count(timings.get("cache_n"))
        prompt = (
            prompt_n + cache_n
            if prompt_n is not None and cache_n is not None
            else None
        )
        return Tokens(prompt, cache_n, _as_count(timings.get("predicted_n")))
    usage = data.get("usage")
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
        cached = (
            _as_count(details.get("cached_tokens"))
            if isinstance(details, dict)
            else None
        )
        return Tokens(
            _as_count(usage.get("prompt_tokens")),
            cached,
            _as_count(usage.get("completion_tokens")),
        )
    return Tokens()


def _anthropic_prompt_cached(usage: dict) -> tuple[int | None, int | None]:
    """(prompt, cached) from an Anthropic usage block.

    prompt = input_tokens + cache_read_input_tokens. A usage block that
    counts an input but reports no cache reads counts cached as 0; with no
    input count the prompt -- and so its sub-count -- is unknown, hence null.
    """
    input_tokens = _as_count(usage.get("input_tokens"))
    if input_tokens is None:
        return None, None
    cached = _as_count(usage.get("cache_read_input_tokens"))
    if cached is None:
        cached = 0
    return input_tokens + cached, cached


def _anthropic_tokens(data: dict) -> Tokens:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return Tokens()
    prompt, cached = _anthropic_prompt_cached(usage)
    return Tokens(prompt, cached, _as_count(usage.get("output_tokens")))


def _count_tokens_tokens(data: dict) -> Tokens:
    return Tokens(_as_count(data.get("input_tokens")))


def tokens_from_body(
    body: bytes, dialect: Literal["openai", "anthropic", "count_tokens"],
) -> Tokens:
    """Tokens from a whole (non-streamed) backend body, per the dialect.

    Never raises: anything unparseable, or with no recognizable fields,
    gives Tokens() (all null).
    """
    try:
        data = json.loads(body)
        if not isinstance(data, dict):
            return Tokens()
        if dialect == "openai":
            return _openai_tokens(data)
        if dialect == "anthropic":
            return _anthropic_tokens(data)
        return _count_tokens_tokens(data)
    except Exception:
        return Tokens()


class StreamTokens:
    """Sniff token counts out of a relayed stream, chunk by chunk.

    Splits on \\n and keeps only the incomplete trailing line between
    feeds, so a line -- including the one that carries the counts --
    survives any write boundary. Only `data:` lines containing "timings"
    or "usage" are parsed. A pending line longer than 1 MiB is dropped and
    parsing resumes at the next newline, so a stream with no newline at all
    cannot grow memory without bound. feed never raises.
    """

    def __init__(self, dialect: Literal["openai", "anthropic"]) -> None:
        self.dialect = dialect
        self.tokens = Tokens()
        self._pending = b""
        # openai: the timings line wins; usage lines only count until it shows up.
        self._saw_timings = False

    def feed(self, chunk: bytes) -> None:
        try:
            self._feed(chunk)
        except Exception:
            pass

    def _feed(self, chunk: bytes) -> None:
        parts = (self._pending + chunk).split(b"\n")
        self._pending = parts.pop()
        for line in parts:
            self._line(line)
        if len(self._pending) > _MAX_PENDING:
            self._pending = b""

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        if b'"timings"' not in line and b'"usage"' not in line:
            return
        try:
            data = json.loads(line[5:].lstrip(b" "))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        if self.dialect == "openai":
            if isinstance(data.get("timings"), dict):
                self._saw_timings = True
                self.tokens = _openai_tokens(data)
            elif not self._saw_timings and isinstance(data.get("usage"), dict):
                self.tokens = _openai_tokens(data)
        else:
            # message_delta carries the usage at the top level; message_start
            # nests it under "message".
            usage = data.get("usage")
            if not isinstance(usage, dict):
                message = data.get("message")
                if isinstance(message, dict):
                    usage = message.get("usage")
            if isinstance(usage, dict):
                self._merge_anthropic_usage(usage)

    def _merge_anthropic_usage(self, usage: dict) -> None:
        # message_start carries the input counts; the last message_delta
        # carries the final output_tokens. Each valid field wins as it comes.
        prompt, cached = _anthropic_prompt_cached(usage)
        if prompt is not None:
            self.tokens.prompt = prompt
            self.tokens.cached = cached
        completion = _as_count(usage.get("output_tokens"))
        if completion is not None:
            self.tokens.completion = completion


def request_context(request: "Request") -> tuple[str, str | None, str | None]:
    """(request id, client host, user agent) for a line.

    The id is validated by request_id_from; the client host is None when the
    ASGI scope has no usable client; the user agent is truncated to
    USER_AGENT_MAX (a hostile agent cannot bloat the line; JSON escaping
    keeps it on one line).
    """
    addr = request.client
    client = addr.host if addr is not None else None
    if client == "":
        client = None
    user_agent = request.headers.get("user-agent")
    if user_agent is not None:
        user_agent = user_agent[:USER_AGENT_MAX]
    return request_id_from(request.headers.get("x-request-id")), client, user_agent


@dataclass
class RequestRecord:
    """One inference request, from its endpoint to its finished line.

    The endpoint builds it (from_request), _route_and_wait stamps
    enqueued_at and puts it on the item, the consumer stamps slot,
    dequeued_at, backend_started_at and the tokens, and exactly one side
    finishes it: the endpoint for every response that is not a relayed
    stream, the stream's closer for a relayed stream. finish is idempotent,
    so a race between the two can never double-write.
    """

    id: str
    endpoint: str
    client: str | None = None
    user_agent: str | None = None
    model: str | None = None
    stream: bool = False
    # Set later, by the queue path or the direct (count_tokens) path.
    slot: str | None = None
    enqueued_at: float | None = None
    dequeued_at: float | None = None
    backend_started_at: float | None = None
    tokens: Tokens = field(default_factory=Tokens)
    _finished: bool = field(default=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.user_agent, str):
            self.user_agent = self.user_agent[:USER_AGENT_MAX]

    @classmethod
    def from_request(
        cls, request: "Request", endpoint: str, body: dict | None,
    ) -> "RequestRecord":
        rid, client, user_agent = request_context(request)
        model: str | None = None
        stream = False
        if isinstance(body, dict):
            candidate = body.get("model")
            if isinstance(candidate, str):
                model = candidate
            # The backend treats the body as streaming only when stream is
            # literally true; the line must not log anything else as one.
            stream = body.get("stream") is True
        return cls(
            id=rid,
            endpoint=endpoint,
            client=client,
            user_agent=user_agent,
            model=model,
            stream=stream,
        )

    def finish(self, status: int, outcome: str) -> None:
        """Write the request line. Idempotent: the first call wins."""
        if self._finished:
            return
        self._finished = True
        queue_wait_ms: int | None = None
        if self.enqueued_at is not None and self.dequeued_at is not None:
            queue_wait_ms = round((self.dequeued_at - self.enqueued_at) * 1000)
        backend_ms: int | None = None
        if self.backend_started_at is not None:
            backend_ms = round((time.monotonic() - self.backend_started_at) * 1000)
        _log_line(
            {
                "event": "request",
                "id": self.id,
                "endpoint": self.endpoint,
                "client": self.client,
                "user_agent": self.user_agent,
                "slot": self.slot,
                "model": self.model,
                "stream": self.stream,
                "status": status,
                "outcome": outcome,
                "queue_wait_ms": queue_wait_ms,
                "backend_ms": backend_ms,
                "prompt_tokens": self.tokens.prompt,
                "cached_tokens": self.tokens.cached,
                "completion_tokens": self.tokens.completion,
            }
        )


def log_swap(
    id: str,
    client: str | None,
    user_agent: str | None,
    slot: str | None,
    model: str | None,
    trial: bool,
    status: int,
    outcome: str,
    duration_ms: int | None,
) -> None:
    """Write the /swap line (same logger as the request lines). Never raises."""
    if isinstance(user_agent, str):
        user_agent = user_agent[:USER_AGENT_MAX]
    _log_line(
        {
            "event": "swap",
            "id": id,
            "client": client,
            "user_agent": user_agent,
            "slot": slot,
            "model": model,
            "trial": trial,
            "status": status,
            "outcome": outcome,
            "duration_ms": duration_ms,
        }
    )


def _log_line(obj: dict) -> None:
    """Write one compact ASCII JSON line.

    Observation must never fail a request: a failing write is swallowed
    (with a warning, itself guarded so a broken logger cannot raise twice).
    """
    try:
        REQUEST_LOGGER.info(json.dumps(obj, separators=(",", ":"), ensure_ascii=True))
    except Exception as exc:
        try:
            REQUEST_LOGGER.warning("request log line failed to write: %s", exc)
        except Exception:
            pass
