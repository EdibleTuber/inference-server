# tests/test_manager_token.py
"""The manager token (phase 6, Q15).

With MANAGER_TOKEN set, every request except exactly GET/any /health needs
`Authorization: Bearer <token>` or `x-api-key: <token>`; anything else is a
401 before any other check. Unset, nothing is checked. A token that is
empty or carries whitespace stops the manager at startup, so a stray space
in an env file can never quietly become a different token, or none.

The token reaches the scripts from root-only files or the caller's own
environment, never argv: a command line is readable by every user in ps.
"""
import dataclasses
import json
import logging
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from manager.config import ManagerConfig

REPO = Path(__file__).resolve().parent.parent
TOKEN = "tok-3f9a2c-manager"
WRONG = "tok-wrong-value"


def _app(test_config, token=TOKEN):
    from manager.app import create_app
    return create_app(dataclasses.replace(test_config, manager_token=token))


@pytest.fixture
def guarded(test_config):
    """A token-guarded app, lifespan not started: nothing past the
    middleware may run for a refused request anyway."""
    return TestClient(_app(test_config))


def _bearer(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "x", path)


def _every_route(app) -> list[tuple[str, str]]:
    """(method, path) for every route the app declares except /health,
    derived from app.routes so a route added later is covered too."""
    out = []
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if path is None or path == "/health":
            continue
        for method in sorted(methods):
            out.append((method, _concrete(path)))
    return out


def _route_ids(test_config):
    return [f"{m} {p}" for m, p in _every_route(_app(test_config))]


# --- no token: nothing changes ---------------------------------------------------

def test_without_a_token_nothing_is_checked(test_config):
    with TestClient(_app(test_config, token=None)) as c:
        assert c.get("/status").status_code == 200
        assert c.get("/v1/models").status_code == 200
        # The JSON gate still answers first for a bad Content-Type.
        r = c.post("/v1/chat/completions", content=b"x",
                   headers={"content-type": "text/plain"})
        assert r.status_code == 415


# --- the 401 -------------------------------------------------------------------------

def test_a_request_without_the_token_gets_401_on_every_route(test_config):
    app = _app(test_config)
    routes = _every_route(app)
    assert len(routes) >= 10, routes  # the derivation found the real routes
    assert ("POST", "/swap") in routes and ("GET", "/status") in routes
    routes += [("GET", "/no-such-path"), ("POST", "/no-such-path")]
    c = TestClient(app)
    failures = []
    for method, path in routes:
        r = c.request(method, path, json={},
                      headers={"content-type": "application/json"})
        if r.status_code != 401 or r.headers.get("www-authenticate") != "Bearer":
            failures.append((method, path, r.status_code, r.headers.get("www-authenticate")))
    assert failures == []


def test_the_token_is_accepted_as_bearer_or_x_api_key(test_config):
    with TestClient(_app(test_config)) as c:
        for headers in (_bearer(), {"Authorization": f"bearer {TOKEN}"},
                        {"Authorization": f"BEARER {TOKEN}"}, {"x-api-key": TOKEN}):
            assert c.get("/status", headers=headers).status_code == 200, headers
            assert c.get("/v1/models", headers=headers).status_code == 200, headers
        # Past the middleware, the app's own answers come back as before.
        assert c.get("/no-such-path", headers=_bearer()).status_code == 404


@pytest.mark.parametrize("headers", [
    {"Authorization": f"Bearer {WRONG}"},
    {"Authorization": f"Bearer {TOKEN[:-1]}"},          # a prefix of the right token
    {"Authorization": f"Bearer {TOKEN}x"},              # the right token, extended
    {"Authorization": f"Bearer {TOKEN} "},              # with a trailing space
    {"Authorization": f"Bearer  {TOKEN}"},              # two spaces: not the token
    {"Authorization": "Bearer "},
    {"Authorization": "Bearer"},
    {"Authorization": TOKEN},                           # no scheme
    {"Authorization": "Basic " + __import__("base64").b64encode(
        f"user:{TOKEN}".encode()).decode()},
    {"x-api-key": WRONG},
    {"x-api-key": ""},
    {"x-api-key": f"Bearer {TOKEN}"},                   # x-api-key carries the bare token
], ids=lambda h: repr(h)[:40])
def test_a_wrong_token_gets_401(guarded, headers):
    r = guarded.get("/status", headers=headers)
    assert r.status_code == 401
    assert r.headers.get("www-authenticate") == "Bearer"


def test_the_401_never_echoes_what_was_sent(guarded):
    r = guarded.get("/status", headers={"Authorization": f"Bearer {WRONG}"})
    assert r.status_code == 401
    assert WRONG not in r.text
    assert TOKEN not in r.text
    assert r.json() == {"error": {"type": "unauthorized", "message": r.json()["error"]["message"]}}
    assert r.json()["error"]["message"]


def _raw_status(app, method: str, path: str, headers=()) -> int:
    """Drive the ASGI app with this exact scope["path"]: an HTTP client would
    rewrite "//health" (a host named health) and "/a/../b" before sending."""
    import asyncio
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": method, "scheme": "http", "path": path,
             "raw_path": path.encode(), "root_path": "", "query_string": b"",
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
             "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 11434)}
    asyncio.run(app(scope, receive, send))
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


@pytest.mark.parametrize("path,exempt", [
    ("/health", True),
    ("/health/", False),
    ("/HEALTH", False),
    ("//health", False),
    ("/healthz", False),
    ("/health/../status", False),
    ("/health%2F", False),
    (" /health", False),
])
def test_only_the_exact_health_path_is_exempt(test_config, path, exempt):
    app = _app(test_config)
    status = _raw_status(app, "GET", path)
    if exempt:
        assert status == 200
    else:
        assert status == 401, (path, status)
    # With the token every one of them gets past the middleware.
    assert _raw_status(app, "GET", path, [("authorization", f"Bearer {TOKEN}")]) != 401


def test_health_is_exempt_whatever_the_method(guarded):
    # The route answers GET only; the point is that the middleware passes it.
    assert guarded.post("/health", json={}).status_code != 401
    assert guarded.request("HEAD", "/health").status_code != 401


def test_the_401_comes_before_the_json_check(guarded):
    r = guarded.post("/v1/chat/completions", content=b"x",
                     headers={"content-type": "text/plain"})
    assert r.status_code == 401
    r = guarded.post("/v1/chat/completions", content=b"x",
                     headers={"content-type": "text/plain", **_bearer()})
    assert r.status_code == 415


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens"])
def test_the_401_uses_the_anthropic_shape_on_v1_messages(guarded, path):
    r = guarded.post(path, json={})
    assert r.status_code == 401
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "authentication_error"
    assert body["error"]["message"]


def test_the_401_uses_the_manager_shape_elsewhere(guarded):
    for method, path in (("POST", "/v1/chat/completions"), ("GET", "/status"),
                         ("POST", "/swap")):
        body = guarded.request(method, path, json={}).json()
        assert "type" not in body, (path, body)
        assert body["error"]["type"] == "unauthorized", (path, body)


def test_the_lifespan_still_runs_with_a_token(test_config):
    """The middleware checks HTTP scopes only; a lifespan scope refused would
    leave the manager with no consumers."""
    app = _app(test_config)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/status", headers=_bearer()).status_code == 200


# --- configuration -------------------------------------------------------------------

def test_the_token_is_read_from_manager_token(monkeypatch):
    monkeypatch.delenv("MANAGER_TOKEN", raising=False)
    assert ManagerConfig.from_env().manager_token is None
    monkeypatch.setenv("MANAGER_TOKEN", TOKEN)
    assert ManagerConfig.from_env().manager_token == TOKEN


@pytest.mark.parametrize("raw", ["", " ", "\t", f"{TOKEN} ", f" {TOKEN}",
                                 f"{TOKEN}\t", "two words", f"{TOKEN}\n"],
                         ids=repr)
def test_an_empty_or_whitespace_token_refuses_to_start(monkeypatch, raw):
    monkeypatch.setenv("MANAGER_TOKEN", raw)
    with pytest.raises(ValueError) as e:
        ManagerConfig.from_env()
    assert "MANAGER_TOKEN" in str(e.value)
    if raw.strip():
        # The error goes to the journal: it must not carry the token.
        assert raw.strip() not in str(e.value)


@pytest.mark.parametrize("raw", ["", " ", f"{TOKEN} "], ids=repr)
def test_create_app_refuses_an_empty_or_whitespace_token_too(test_config, raw):
    """from_env is not the only way in: a config built directly with an empty
    token would make `x-api-key:` (empty) match it."""
    with pytest.raises(ValueError) as e:
        _app(test_config, token=raw)
    assert "MANAGER_TOKEN" in str(e.value)


def test_the_token_is_not_in_the_config_repr_or_any_log_line(test_config, caplog):
    cfg = dataclasses.replace(test_config, manager_token=TOKEN)
    assert TOKEN not in repr(cfg)
    assert TOKEN not in str(cfg)

    caplog.set_level(logging.DEBUG)
    from manager.app import create_app
    with TestClient(create_app(cfg)) as c:
        c.get("/status", headers=_bearer())
        c.get("/status", headers={"Authorization": f"Bearer {WRONG}"})
        c.post("/v1/chat/completions", json={"model": "m", "messages": []},
               headers={"x-api-key": TOKEN})
    records = caplog.records
    assert any(r.name.startswith("manager") for r in records), \
        "no manager log line was captured, so this test would prove nothing"
    for r in records:
        line = r.getMessage()
        assert TOKEN not in line, (r.name, line)
        assert WRONG not in line, (r.name, line)


# --- the unit and the env template -------------------------------------------------

def test_the_unit_loads_the_token_file_optionally_after_manager_env():
    from tests.systemd_expand import service_directives
    unit = (REPO / "systemd" / "llama-manager.service").read_text(encoding="utf-8")
    files = [v for k, v in service_directives(unit) if k == "EnvironmentFile"]
    assert files == ["/etc/llama/manager.env", "-/etc/llama/manager-token.env"]


def test_the_manager_env_template_carries_no_token():
    """manager.env is not the token's file: the token lives in the 0640
    manager-token.env, readable by the manager's group only."""
    text = (REPO / "config" / "manager.env").read_text(encoding="utf-8")
    assert not re.search(r"^#?\s*MANAGER_TOKEN=", text, re.M)


# --- scripts/onboard.py --------------------------------------------------------------

def _load_onboard():
    """scripts/ is not a package: load onboard.py by path, as test_onboard does."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("onboard_token_test",
                                                  REPO / "scripts" / "onboard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _onboard_main(monkeypatch, handler, argv):
    onboard_mod = _load_onboard()
    real = httpx.Client
    monkeypatch.setattr(onboard_mod.httpx, "Client", lambda *a, **kw: real(
        *a, **{**kw, "transport": httpx.MockTransport(handler)}))
    return onboard_mod.main(argv)


def test_onboard_sends_the_token_from_its_environment(monkeypatch, capsys):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"slots": {"main": {"loaded_model": None}}})

    monkeypatch.setenv("MANAGER_TOKEN", TOKEN)
    # --slot nope: onboard stops (exit 2) right after its first /status call.
    code = _onboard_main(monkeypatch, handler,
                         ["gemma-4-26b-a4b-it-q4_k_m", "--slot", "nope",
                          "--manager", "http://manager"])
    assert code == 2
    assert seen
    assert all(r.headers.get("authorization") == f"Bearer {TOKEN}" for r in seen)


def test_onboard_without_the_token_sends_no_header(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"slots": {"main": {"loaded_model": None}}})

    monkeypatch.delenv("MANAGER_TOKEN", raising=False)
    _onboard_main(monkeypatch, handler,
                  ["gemma-4-26b-a4b-it-q4_k_m", "--slot", "nope", "--manager", "http://manager"])
    assert seen
    assert all("authorization" not in r.headers for r in seen)


def test_onboard_on_a_401_exits_1_saying_to_export_the_token(monkeypatch, capsys):
    profile = REPO / "config" / "models" / "gemma-4-26b-a4b-it-q4_k_m.toml"
    before = profile.read_bytes()

    def handler(request):
        return httpx.Response(401, headers={"www-authenticate": "Bearer"},
                              json={"error": {"type": "unauthorized", "message": "token required"}})

    monkeypatch.delenv("MANAGER_TOKEN", raising=False)
    code = _onboard_main(monkeypatch, handler,
                         ["gemma-4-26b-a4b-it-q4_k_m", "--manager", "http://manager"])
    out = capsys.readouterr()
    assert code == 1
    text = out.out + out.err
    assert "MANAGER_TOKEN" in text and "export" in text
    assert "Traceback" not in text
    assert profile.read_bytes() == before


def test_onboard_takes_no_token_argument(monkeypatch):
    onboard_mod = _load_onboard()
    with pytest.raises(SystemExit) as e:
        onboard_mod.main(["gemma-4-26b-a4b-it-q4_k_m", "--token", TOKEN])
    assert e.value.code == 2


# --- scripts/deploy-manager.sh --------------------------------------------------------

FAKE_CURL = r'''#!/usr/bin/env python3
"""Stands in for curl: logs each call's argv and the headers it would send
(-H value, or the lines of -H @file / -H @-), answers /health with 200 and
/status with one slot."""
import json, os, sys
args = sys.argv[1:]
headers, url = [], None
i = 0
while i < len(args):
    a = args[i]
    if a in ("-H", "--header"):
        v = args[i + 1]; i += 1
        if v.startswith("@"):
            src = sys.stdin.read() if v == "@-" else open(v[1:]).read()
            headers += [ln for ln in src.splitlines() if ln.strip()]
        else:
            headers.append(v)
    elif a.startswith("http://") or a.startswith("https://"):
        url = a
    i += 1
with open(os.environ["FAKE_CURL_LOG"], "a") as f:
    f.write(json.dumps({"argv": args, "headers": headers, "url": url}) + "\n")
if url and url.endswith("/health"):
    sys.stdout.write("200")
elif url and url.endswith("/status"):
    sys.stdout.write(json.dumps({"slots": {"main": {"loaded_model": "M", "healthy": True}}}))
'''


def _deploy_verify(tmp_path, token_file_text):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(FAKE_CURL)
    curl.chmod(curl.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "manager.env").write_text("HOST=127.0.0.1\nPORT=11434\n")
    token_env = tmp_path / "manager-token.env"
    if token_file_text is not None:
        token_env.write_text(token_file_text)
    log = tmp_path / "curl.log"
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin",
           "DEPLOY_MGR_ENV": str(tmp_path / "manager.env"),
           "DEPLOY_MGR_TOKEN_ENV": str(token_env),
           "DEPLOY_DEST": str(tmp_path / "dest"),
           "HEALTH_WAIT": "3",
           "FAKE_CURL_LOG": str(log)}
    proc = subprocess.run(["bash", str(REPO / "scripts" / "deploy-manager.sh"), "--verify"],
                          capture_output=True, text=True, env=env, cwd=tmp_path)
    calls = [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []
    return proc, calls


def test_deploy_sends_the_token_on_its_status_call(tmp_path):
    proc, calls = _deploy_verify(tmp_path, f"MANAGER_TOKEN={TOKEN}\n")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    status = [c for c in calls if (c["url"] or "").endswith("/status")]
    health = [c for c in calls if (c["url"] or "").endswith("/health")]
    assert status and health, calls
    assert all(f"Authorization: Bearer {TOKEN}" in c["headers"] for c in status), status
    # /health is exempt and stays as it was.
    assert all(not any(h.lower().startswith("authorization") for h in c["headers"])
               for c in health), health
    # Never on a command line: root's argv is readable by every user.
    assert all(TOKEN not in " ".join(c["argv"]) for c in calls), calls
    assert "slot main" in proc.stdout
    assert TOKEN not in proc.stdout + proc.stderr


def test_deploy_without_a_token_file_sends_no_header(tmp_path):
    proc, calls = _deploy_verify(tmp_path, None)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert calls
    assert all(not any(h.lower().startswith("authorization") for h in c["headers"])
               for c in calls), calls
