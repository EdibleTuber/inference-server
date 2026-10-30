# tests/live_ports.py
"""Shared names for the suite-wide live-port guard.

LIVE_PORTS and LivePortBlocked live here, not in conftest.py, because
tests/ has no __init__.py: pytest loads conftest.py as a top-level module
named `conftest`, so a test doing `from tests.conftest import ...` would
get a *second* copy of it, with a different LivePortBlocked class, and
pytest.raises would never match. Both conftest.py and the tests import
from tests.live_ports (the suite runs with PYTHONPATH=.).
"""

# Ports in use on the host: SearXNG (8080), the llama-server slots and the
# embeddings server (8081-8084), and the manager itself (11434). Tests must
# never reach them.
LIVE_PORTS: frozenset[int] = frozenset({8080, 8081, 8082, 8083, 8084, 11434})


class LivePortBlocked(ConnectionRefusedError):
    """Raised by the suite-wide guard (tests/conftest.py) when a test
    connects to a live manager slot port, on any host."""
