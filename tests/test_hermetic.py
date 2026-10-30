import socket
import pytest
from tests.live_ports import LIVE_PORTS, LivePortBlocked

def test_connect_to_a_live_port_is_blocked_by_the_guard():
    # 8083 is the decommissioned batch port: nothing listens there, so
    # before the guard exists this raises a plain ConnectionRefusedError,
    # not LivePortBlocked, and the test fails without touching a live service.
    s = socket.socket()
    try:
        with pytest.raises(LivePortBlocked):
            s.connect(("127.0.0.1", 8083))
    finally:
        s.close()

def test_test_config_uses_no_live_port(test_config):
    ports = {test_config.llama_server_port, test_config.embeddings_port}
    ports |= {s.port for s in test_config.slots}
    assert not ports & LIVE_PORTS
