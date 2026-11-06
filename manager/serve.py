# manager/serve.py
"""
Entry point that binds several addresses in one process (phase 6, Q14, Q16).

The manager answers on every address in LISTEN_HOSTS -- loopback, the LAN
and the tailnet -- so one process, one set of slots, queues and request
ids, serves them all. uvicorn's --host takes a single address, so this
module binds the sockets itself and hands them to uvicorn, which accepts
on all of them.
"""
import socket
import sys
from collections.abc import Sequence

import uvicorn

from manager.config import ManagerConfig

# <linux/in.h> / <linux/in6.h>; this Python's socket module has no
# constants for them.
IP_FREEBIND = getattr(socket, "IP_FREEBIND", 15)
IPV6_FREEBIND = getattr(socket, "IPV6_FREEBIND", 78)


def bind_sockets(hosts: Sequence[str], port: int) -> list[socket.socket]:
    """One listening TCP socket per host, non-blocking, in input order.

    IP_FREEBIND lets an address the machine does not have yet bind anyway:
    the tailnet address appears only once tailscaled is up, which can be
    after the manager starts. If any address fails, everything bound so
    far is closed and an OSError naming the host and the error is raised.
    """
    socks: list[socket.socket] = []
    try:
        for host in hosts:
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            sock = socket.socket(family, socket.SOCK_STREAM)
            socks.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, IPV6_FREEBIND, 1)
            else:
                sock.setsockopt(socket.IPPROTO_IP, IP_FREEBIND, 1)
            sock.bind((host, port))
            sock.listen()
            sock.setblocking(False)
    except OSError as exc:
        for sock in socks:
            sock.close()
        raise OSError(
            f"cannot bind {host}:{port}: {exc.strerror or exc}"
        ) from exc
    return socks


def main() -> None:
    """Bind every LISTEN_HOSTS address and serve the app on all of them."""
    config = ManagerConfig.from_env()
    try:
        sockets = bind_sockets(config.listen_hosts, config.port)
    except OSError as exc:
        print(f"manager.serve: {exc}", file=sys.stderr)
        sys.exit(1)
    server = uvicorn.Server(
        uvicorn.Config(
            "manager.app:create_app",
            factory=True,
            timeout_keep_alive=120,
            log_level="info",
        )
    )
    server.run(sockets=sockets)


if __name__ == "__main__":
    main()
