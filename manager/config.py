# manager/config.py
"""
Configuration loading for the model manager.

All configuration comes from environment variables, which are set in
/etc/llama/manager.env and loaded by systemd's EnvironmentFile directive.
This module provides a typed config object so the rest of the app doesn't
need to deal with raw env vars or string parsing.
"""
import os
from dataclasses import dataclass


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
        queue_limit: Max number of requests to hold in the FIFO queue.
        swap_timeout: Seconds to wait for llama-server health after a model swap.
        log_file: Path to the manager's log file.
        embeddings_host: Address where the embeddings server is running.
        embeddings_port: Port where the embeddings server listens.
        collections_config: Path to the collections configuration JSON file.
        skills_db_path: Path to the skills SQLite database file.
    """
    host: str
    port: int
    llama_server_host: str
    llama_server_port: int
    models_dir: str
    llama_server_env: str
    queue_limit: int
    swap_timeout: int
    log_file: str
    embeddings_host: str
    embeddings_port: int
    collections_config: str
    skills_db_path: str

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
        """Load configuration from environment variables with sensible defaults."""
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=_int_env("PORT", 8080),
            llama_server_host=os.getenv("LLAMA_SERVER_HOST", "127.0.0.1"),
            llama_server_port=_int_env("LLAMA_SERVER_PORT", 8081),
            models_dir=os.getenv("MODELS_DIR", "/opt/llama/models"),
            llama_server_env=os.getenv("LLAMA_SERVER_ENV", "/etc/llama/llama-server.env"),
            queue_limit=_int_env("QUEUE_LIMIT", 20),
            swap_timeout=_int_env("SWAP_TIMEOUT", 120),
            log_file=os.getenv("LOG_FILE", "/var/log/llama/manager.log"),
            embeddings_host=os.getenv("EMBEDDINGS_HOST", "127.0.0.1"),
            embeddings_port=_int_env("EMBEDDINGS_PORT", 8082),
            collections_config=os.getenv("COLLECTIONS_CONFIG", "/etc/llama/collections.json"),
            skills_db_path=os.getenv("SKILLS_DB_PATH", "/opt/llama/data/skills.db"),
        )
