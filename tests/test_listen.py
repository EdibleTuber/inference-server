"""Several addresses, one process (phase 6, Q14 and Q16).

The manager binds every address in LISTEN_HOSTS itself, in one process, so
one set of slots, queues and request ids serves loopback, the LAN and the
tailnet alike. uvicorn's own --host takes one address, so manager/serve.py
binds the sockets and hands them to uvicorn.

The serve tests run manager.serve.main() in-process with uvicorn.Server
replaced by a recorder. A subprocess would be closer to the unit, but a
mutant that let it start serving would run create_app's lifespan against
the live slot ports, outside conftest's block_live_ports guard.
"""
import errno
import importlib.util
import json
import socket
import stat
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from manager.config import ManagerConfig
from tests.live_ports import LIVE_PORTS
from tests.test_manager_token import FAKE_CURL

REPO = Path(__file__).resolve().parent.parent
IP_FREEBIND = 15     # <linux/in.h>; Python 3.12's socket module has no constant
IPV6_FREEBIND = 78   # <linux/in6.h>


def _serve():
    import manager.serve
    return manager.serve


def _free_port() -> int:
    """A port free on 127.0.0.1 and 127.0.0.2 that is not a live one."""
    for _ in range(20):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        if port in LIVE_PORTS:
            continue
        try:
            with socket.socket() as other:
                other.bind(("127.0.0.2", port))
        except OSError:
            continue
        return port
    raise RuntimeError("no free port")


@pytest.fixture()
def closing():
    """Close every socket a test collects, pass or fail."""
    socks: list[socket.socket] = []
    yield socks
    for s in socks:
        s.close()


# --- LISTEN_HOSTS ------------------------------------------------------------------

def test_listen_hosts_default_to_host(monkeypatch):
    """An env file without the setting behaves as today."""
    monkeypatch.delenv("LISTEN_HOSTS", raising=False)
    monkeypatch.setenv("HOST", "192.168.1.14")
    assert ManagerConfig.from_env().listen_hosts == ("192.168.1.14",)


def test_listen_hosts_are_parsed_and_stripped(monkeypatch):
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("LISTEN_HOSTS", " 127.0.0.1 , 192.168.1.14,100.82.222.92 ")
    assert ManagerConfig.from_env().listen_hosts == (
        "127.0.0.1", "192.168.1.14", "100.82.222.92")


def test_listen_hosts_take_ipv6_and_a_lone_wildcard(monkeypatch):
    for raw, want in (("::1,127.0.0.1", ("::1", "127.0.0.1")),
                      ("0.0.0.0", ("0.0.0.0",)),
                      ("::", ("::",))):
        monkeypatch.setenv("LISTEN_HOSTS", raw)
        assert ManagerConfig.from_env().listen_hosts == want


@pytest.mark.parametrize("raw", [
    pytest.param("localhost", id="not-an-ip"),
    pytest.param("127.0.0.1,127.0.0.1", id="duplicate"),
    pytest.param("::1,0:0:0:0:0:0:0:1", id="duplicate-spelled-differently"),
    pytest.param("0.0.0.0,127.0.0.1", id="wildcard-with-another"),
    pytest.param("127.0.0.1,::", id="v6-wildcard-with-another"),
    pytest.param("0.0.0.0,::", id="two-wildcards"),
    pytest.param("127.0.0.1,,192.168.1.14", id="empty-item"),
    pytest.param("", id="empty"),
    pytest.param("  ", id="blank"),
])
def test_a_bad_listen_hosts_names_the_key(monkeypatch, raw):
    monkeypatch.setenv("LISTEN_HOSTS", raw)
    with pytest.raises(ValueError, match="LISTEN_HOSTS"):
        ManagerConfig.from_env()


def test_a_bad_host_without_listen_hosts_names_host(monkeypatch):
    """The default (HOST,) is checked the same way, and the error names the
    key the operator actually set."""
    monkeypatch.delenv("LISTEN_HOSTS", raising=False)
    monkeypatch.setenv("HOST", "localhost")
    with pytest.raises(ValueError, match=r"\bHOST\b"):
        ManagerConfig.from_env()


# --- bind_sockets --------------------------------------------------------------------

def test_each_socket_is_bound_listening_nonblocking_and_freebind(closing):
    port = _free_port()
    socks = _serve().bind_sockets(["127.0.0.1", "127.0.0.2", "::1"], port)
    closing.extend(socks)
    assert [s.getsockname()[:2] for s in socks] == [
        ("127.0.0.1", port), ("127.0.0.2", port), ("::1", port)]
    for s in socks:
        assert s.type == socket.SOCK_STREAM
        assert s.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1, "listening"
        assert s.getblocking() is False
        assert s.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) != 0
    v4, v6 = socks[0], socks[2]
    assert v4.family == socket.AF_INET and v6.family == socket.AF_INET6
    assert v4.getsockopt(socket.IPPROTO_IP, IP_FREEBIND) != 0
    assert v6.getsockopt(socket.IPPROTO_IPV6, IPV6_FREEBIND) != 0


@pytest.mark.parametrize("host", ["100.64.0.99", "fd00::99"])
def test_an_address_the_host_does_not_have_yet_still_binds(closing, host):
    """The tailnet address appears only once tailscaled is up, which can be
    after the manager starts; IP_FREEBIND lets the bind succeed anyway."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as plain:
        try:
            plain.bind((host, 0))
        except OSError as exc:
            assert exc.errno == errno.EADDRNOTAVAIL
        else:
            pytest.skip(f"this host already lets a plain socket bind {host}")
    port = _free_port()
    socks = _serve().bind_sockets([host], port)
    closing.extend(socks)
    assert socks[0].getsockname()[:2] == (host, port)


def test_a_port_already_taken_raises_naming_the_host(closing):
    port = _free_port()
    taken = socket.socket()
    closing.append(taken)
    taken.bind(("127.0.0.2", port))
    taken.listen()
    with pytest.raises(OSError) as info:
        closing.extend(_serve().bind_sockets(["127.0.0.1", "127.0.0.2"], port) or [])
    assert "127.0.0.2" in str(info.value)
    # What it bound before the failure is closed, not leaked: 127.0.0.1 is free.
    with socket.socket() as again:
        again.bind(("127.0.0.1", port))


def test_one_server_answers_on_every_bound_address(closing):
    """One uvicorn.Server on several sockets: the same process, so the same
    state, answers on each address."""
    calls = []

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while (await receive())["type"] != "lifespan.shutdown":
                await send({"type": "lifespan.startup.complete"})
            await send({"type": "lifespan.shutdown.complete"})
            return
        calls.append(scope["server"][0])
        body = json.dumps({"calls": len(calls)}).encode()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})

    port = _free_port()
    socks = _serve().bind_sockets(["127.0.0.1", "127.0.0.2", "::1"], port)
    closing.extend(socks)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", ws="none"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": socks}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "uvicorn never started"
            time.sleep(0.02)
        seen = [httpx.get(f"http://{h}:{port}/", timeout=5).json()["calls"]
                for h in ("127.0.0.1", "127.0.0.2", "[::1]")]
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    assert seen == [1, 2, 3], "one counter: one process"
    assert calls == ["127.0.0.1", "127.0.0.2", "::1"]


# --- main ----------------------------------------------------------------------------

class _Recorder:
    """Stands in for uvicorn.Server: records the config and sockets, serves nothing."""
    instances: list["_Recorder"] = []

    def __init__(self, config):
        self.config = config
        self.sockets = None
        _Recorder.instances.append(self)

    def run(self, sockets=None):
        self.sockets = sockets


@pytest.fixture()
def recorder(monkeypatch, closing):
    _Recorder.instances = []
    monkeypatch.setattr(uvicorn, "Server", _Recorder)
    yield _Recorder.instances
    for r in _Recorder.instances:
        closing.extend(r.sockets or [])


def test_main_runs_one_uvicorn_on_every_listen_host(monkeypatch, recorder):
    port = _free_port()
    monkeypatch.setenv("LISTEN_HOSTS", "127.0.0.1,127.0.0.2")
    monkeypatch.setenv("PORT", str(port))
    _serve().main()
    assert len(recorder) == 1
    run = recorder[0]
    assert [s.getsockname()[:2] for s in run.sockets] == [
        ("127.0.0.1", port), ("127.0.0.2", port)]
    cfg = run.config
    assert cfg.app == "manager.app:create_app"
    assert cfg.factory is True
    assert cfg.timeout_keep_alive == 120
    assert cfg.log_level == "info"
    # Phase 5's client-gone check relies on what http="auto" picks.
    assert cfg.http == "auto"


def test_main_exits_nonzero_naming_the_host_when_a_bind_fails(monkeypatch, recorder,
                                                              closing, capsys):
    port = _free_port()
    taken = socket.socket()
    closing.append(taken)
    taken.bind(("127.0.0.2", port))
    taken.listen()
    monkeypatch.setenv("LISTEN_HOSTS", "127.0.0.1,127.0.0.2")
    monkeypatch.setenv("PORT", str(port))
    with pytest.raises(SystemExit) as info:
        _serve().main()
    assert info.value.code not in (0, None)
    said = f"{info.value.code}\n{capsys.readouterr().err}"
    assert "127.0.0.2" in said
    assert "Address already in use" in said
    assert recorder == [] or all(r.sockets is None for r in recorder), "never served"


# --- the unit and the template -----------------------------------------------------

def _exec_start() -> str:
    text = (REPO / "systemd" / "llama-manager.service").read_text()
    joined = text.replace("\\\n", " ")
    lines = [ln for ln in joined.splitlines() if ln.startswith("ExecStart=")]
    assert len(lines) == 1, lines
    return " ".join(lines[0].removeprefix("ExecStart=").split())


def test_the_unit_runs_manager_serve():
    assert _exec_start() == "/opt/llama/manager/venv/bin/python -m manager.serve"


def test_the_template_listens_on_loopback_lan_and_tailnet():
    values = [ln.split("=", 1)[1] for ln in
              (REPO / "config" / "manager.env").read_text().splitlines()
              if ln.startswith("LISTEN_HOSTS=")]
    assert values == ["127.0.0.1,192.168.1.14,100.82.222.92"]


# --- the scripts' base URL ---------------------------------------------------------

_URL_CASES = [
    pytest.param("HOST=192.168.1.14\nLISTEN_HOSTS=127.0.0.1,192.168.1.14,100.82.222.92\n",
                 "http://127.0.0.1:{port}", id="loopback-first"),
    pytest.param("HOST=192.168.1.14\nLISTEN_HOSTS=192.168.1.14,127.0.0.1\n",
                 "http://127.0.0.1:{port}", id="loopback-later"),
    pytest.param("LISTEN_HOSTS= 192.168.1.14 , 127.0.0.2 \n",
                 "http://127.0.0.2:{port}", id="any-127-is-loopback"),
    pytest.param("LISTEN_HOSTS=192.168.1.14,::1\n",
                 "http://[::1]:{port}", id="v6-loopback-bracketed"),
    pytest.param("HOST=127.0.0.1\nLISTEN_HOSTS=100.82.222.92,192.168.1.14\n",
                 "http://100.82.222.92:{port}", id="no-loopback-first-host"),
    pytest.param("HOST=192.168.1.14\nLISTEN_HOSTS=0.0.0.0\n", "http://127.0.0.1:{port}",
                 id="v4-wildcard"),
    pytest.param("LISTEN_HOSTS=::\n", "http://[::1]:{port}", id="v6-wildcard"),
    pytest.param("HOST=192.168.1.14\n", "http://192.168.1.14:{port}", id="unset-uses-host"),
    pytest.param("HOST=0.0.0.0\n", "http://127.0.0.1:{port}", id="unset-wildcard-host"),
]


def _onboard():
    spec = importlib.util.spec_from_file_location("onboard_listen", REPO / "scripts" / "onboard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("env_text, url", _URL_CASES)
def test_onboard_uses_a_loopback_listen_host(env_text, url):
    assert _onboard().manager_url_from_env(env_text + "PORT=9000\n") == url.format(port=9000)


@pytest.mark.parametrize("env_text, url", _URL_CASES)
def test_deploy_uses_a_loopback_listen_host(tmp_path, env_text, url):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(FAKE_CURL)
    curl.chmod(curl.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "manager.env").write_text(env_text + "PORT=9000\n")
    log = tmp_path / "curl.log"
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin",
           "DEPLOY_MGR_ENV": str(tmp_path / "manager.env"),
           "DEPLOY_MGR_TOKEN_ENV": str(tmp_path / "no-token.env"),
           "DEPLOY_DEST": str(tmp_path / "dest"),
           "HEALTH_WAIT": "3",
           "FAKE_CURL_LOG": str(log)}
    proc = subprocess.run(["bash", str(REPO / "scripts" / "deploy-manager.sh"), "--verify"],
                          capture_output=True, text=True, env=env, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    urls = [json.loads(ln)["url"] for ln in log.read_text().splitlines()]
    base = url.format(port=9000)
    assert urls and all(u.startswith(base + "/") for u in urls), urls
    assert f"{base}/health" in urls
