# manager/config.py
"""
Configuration loading for the model manager.

All configuration comes from environment variables, which are set in
/etc/llama/manager.env and loaded by systemd's EnvironmentFile directive.
This module provides a typed config object so the rest of the app doesn't
need to deal with raw env vars or string parsing.
"""
import math
import os
import re
from dataclasses import dataclass, field

from manager.slot_config import SlotConfig, build_slots


def _int_env(key: str, default: int) -> int:
    """Read an integer from an env var with a clear error on bad values."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"Configuration error: {key}={raw!r} is not a valid integer"
        ) from None


def _float_env(key: str, default: float) -> float:
    """Read a float from an env var with a clear error on bad values."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"Configuration error: {key}={raw!r} is not a valid float"
        ) from None


def _positive_finite_env(key: str, default: float) -> float:
    """Read a float that must be a finite number of seconds greater than zero.

    A supervisor interval of 0 or below re-arms a probe on every loop tick
    (asyncio.sleep(0/-5) returns at once) and nan or inf never wakes it
    (asyncio.sleep(nan) never returns), so both are refused.
    """
    value = _float_env(key, default)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"Configuration error: {key}={value!r} must be a finite "
            f"number of seconds greater than 0"
        )
    return value


def _bool01_env(key: str, default: bool) -> bool:
    """Read a boolean from an env var that must be exactly 0 or 1, with a
    clear error naming the key on any other value."""
    raw = os.getenv(key)
    if raw is None:
        return default
    if raw not in ("0", "1"):
        raise ValueError(
            f"Configuration error: {key}={raw!r} must be 0 or 1"
        )
    return raw == "1"


def validate_manager_token(value: str | None) -> str | None:
    """Validate MANAGER_TOKEN (phase 6, Q15).

    None (unset) means no token and no check. An empty value, or one that
    carries any whitespace, is refused: a stray space in an env file must
    never quietly become a different token, or no token. The error goes to
    the journal, so it names the key and never the value.
    """
    if value is None:
        return None
    if not value or re.search(r"\s", value):
        raise ValueError(
            "Configuration error: MANAGER_TOKEN is empty or contains "
            "whitespace; it must be a single unbroken string"
        )
    return value


@dataclass
class ManagerConfig:
    """Typed configuration for the model manager service.

    Attributes:
        host: IP address to bind to (LAN IP or 0.0.0.0 for all interfaces).
        port: Port to listen on for incoming API requests.
        llama_server_host: Address where llama-server is running (always localhost).
        llama_server_port: Port where llama-server listens.
        models_dir: Path to directory containing GGUF model files.
        llama_server_env: Path to llama-server's env file (updated during model swaps).
        llama_server_unit: systemd unit name for the main llama-server.
        queue_limit: Max number of requests to hold in the FIFO queue.
        swap_timeout: Seconds to wait for llama-server health after a model swap.
        embeddings_host: Address where the embeddings server is running.
        embeddings_port: Port where the embeddings server listens.
        collections_config: Path to the collections configuration JSON file.
        skills_db_path: Path to the skills SQLite database file.
        slots: Ordered slot list built from SLOTS and SLOT_<NAME>_* (or legacy) env vars.
        restart_timeout: Seconds to allow a systemctl restart before it fails.
        proxy_read_timeout: Seconds a proxied request may run before the
            manager answers 504 upstream_timeout (audit F4).
        profiles_dir: Directory of per-model TOML profiles (the roster). Each
            swap renders its profile's section for the slot into the slot's
            generated model env file.
        supervisor_probe_interval: Seconds between the per-slot supervisor's
            probes, while the slot is idle, unhealthy or failed (Task 5, Q21).
        supervisor_failures: Consecutive failed probes that make a restart due
            for an unhealthy slot.
        supervisor_backoff: Base delay, in seconds, between two supervisor
            restarts (the start of the capped exponential backoff).
        supervisor_backoff_max: Cap, in seconds, on the supervisor's restart
            backoff.
        supervisor_restart: Whether a due restart actually restarts the slot's
            unit (env value exactly 0 or 1). Off by default: a due restart is
            logged as "would restart" and counted in /status.
        manager_token: Bearer token every client must send (phase 6, Q15).
            None (unset) means no check at all. Excluded from repr: the token
            must never appear in logs, argv or error messages.
    """
    host: str
    port: int
    llama_server_host: str
    llama_server_port: int
    models_dir: str
    llama_server_env: str
    llama_server_unit: str
    queue_limit: int
    swap_timeout: int
    embeddings_host: str
    embeddings_port: int
    collections_config: str
    skills_db_path: str
    slots: tuple[SlotConfig, ...] = ()
    restart_timeout: int = 150
    proxy_read_timeout: int = 1800
    profiles_dir: str = "/etc/llama/models.d"
    # The per-slot supervisor (Task 5, Q21).
    supervisor_probe_interval: float = 15.0
    supervisor_failures: int = 3
    supervisor_backoff: float = 30.0
    supervisor_backoff_max: float = 600.0
    supervisor_restart: bool = False
    manager_token: str | None = field(default=None, repr=False)

    @property
    def llama_server_url(self) -> str:
        """Full URL for connecting to llama-server."""
        return f"http://{self.llama_server_host}:{self.llama_server_port}"

    @property
    def embeddings_url(self) -> str:
        """Full URL for connecting to the embeddings server."""
        return f"http://{self.embeddings_host}:{self.embeddings_port}"

    @classmethod
    def from_env(cls) -> "ManagerConfig":
        """Load configuration from environment variables with sensible defaults.

        The supervisor settings are range-checked (branch review round 1):
        the interval, the backoff and the cap must each be finite seconds
        greater than 0, the failure count at least 1, and the cap not below
        the base; a violation raises ValueError naming the key.
        """
        failures = _int_env("SUPERVISOR_FAILURES", 3)
        if failures < 1:
            raise ValueError(
                f"Configuration error: SUPERVISOR_FAILURES={failures!r} "
                f"must be an integer of at least 1"
            )
        backoff = _positive_finite_env("SUPERVISOR_BACKOFF", 30.0)
        backoff_max = _positive_finite_env("SUPERVISOR_BACKOFF_MAX", 600.0)
        if backoff_max < backoff:
            raise ValueError(
                f"Configuration error: SUPERVISOR_BACKOFF_MAX={backoff_max!r} "
                f"must be at least SUPERVISOR_BACKOFF={backoff!r}"
            )
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=_int_env("PORT", 8080),
            llama_server_host=os.getenv("LLAMA_SERVER_HOST", "127.0.0.1"),
            llama_server_port=_int_env("LLAMA_SERVER_PORT", 8081),
            models_dir=os.getenv("MODELS_DIR", "/opt/llama/models"),
            llama_server_env=os.getenv("LLAMA_SERVER_ENV", "/etc/llama/llama-server.env"),
            llama_server_unit=os.getenv("LLAMA_SERVER_UNIT", "llama-server.service"),
            queue_limit=_int_env("QUEUE_LIMIT", 20),
            swap_timeout=_int_env("SWAP_TIMEOUT", 120),
            embeddings_host=os.getenv("EMBEDDINGS_HOST", "127.0.0.1"),
            embeddings_port=_int_env("EMBEDDINGS_PORT", 8082),
            collections_config=os.getenv("COLLECTIONS_CONFIG", "/etc/llama/collections.json"),
            skills_db_path=os.getenv("SKILLS_DB_PATH", "/opt/llama/data/skills.db"),
            slots=build_slots(),
            restart_timeout=_int_env("RESTART_TIMEOUT", 150),
            proxy_read_timeout=_int_env("PROXY_READ_TIMEOUT", 1800),
            profiles_dir=os.getenv("PROFILES_DIR", "/etc/llama/models.d"),
            supervisor_probe_interval=_positive_finite_env(
                "SUPERVISOR_PROBE_INTERVAL", 15.0
            ),
            supervisor_failures=failures,
            supervisor_backoff=backoff,
            supervisor_backoff_max=backoff_max,
            supervisor_restart=_bool01_env("SUPERVISOR_RESTART", False),
            manager_token=validate_manager_token(os.getenv("MANAGER_TOKEN")),
        )
