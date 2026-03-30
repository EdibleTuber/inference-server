# tests/test_config.py
"""Tests for configuration loading from environment variables."""
import pytest
from manager.config import ManagerConfig


def test_config_loads_from_env(monkeypatch):
    """Config should load all values from environment variables."""
    monkeypatch.setenv("HOST", "192.168.1.50")
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("LLAMA_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("LLAMA_SERVER_PORT", "8081")
    monkeypatch.setenv("MODELS_DIR", "/opt/llama/models")
    monkeypatch.setenv("LLAMA_SERVER_ENV", "/etc/llama/llama-server.env")
    monkeypatch.setenv("QUEUE_LIMIT", "20")
    monkeypatch.setenv("SWAP_TIMEOUT", "120")
    monkeypatch.setenv("LOG_FILE", "/var/log/llama/manager.log")

    config = ManagerConfig.from_env()

    assert config.host == "192.168.1.50"
    assert config.port == 8080
    assert config.llama_server_host == "127.0.0.1"
    assert config.llama_server_port == 8081
    assert config.models_dir == "/opt/llama/models"
    assert config.llama_server_env == "/etc/llama/llama-server.env"
    assert config.queue_limit == 20
    assert config.swap_timeout == 120
    assert config.log_file == "/var/log/llama/manager.log"


def test_config_defaults(monkeypatch):
    """Config should use sensible defaults when env vars are missing."""
    for key in ["HOST", "PORT", "LLAMA_SERVER_HOST", "LLAMA_SERVER_PORT",
                "MODELS_DIR", "LLAMA_SERVER_ENV", "QUEUE_LIMIT",
                "SWAP_TIMEOUT", "LOG_FILE"]:
        monkeypatch.delenv(key, raising=False)

    config = ManagerConfig.from_env()

    assert config.host == "0.0.0.0"
    assert config.port == 8080
    assert config.llama_server_host == "127.0.0.1"
    assert config.llama_server_port == 8081
    assert config.models_dir == "/opt/llama/models"
    assert config.llama_server_env == "/etc/llama/llama-server.env"
    assert config.queue_limit == 20
    assert config.swap_timeout == 120
    assert config.log_file == "/var/log/llama/manager.log"


def test_config_llama_server_url(monkeypatch):
    """Config should provide a convenience URL for the llama-server."""
    monkeypatch.setenv("LLAMA_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("LLAMA_SERVER_PORT", "8081")

    config = ManagerConfig.from_env()

    assert config.llama_server_url == "http://127.0.0.1:8081"


def test_config_embeddings_defaults(monkeypatch):
    """New embedding/collection config fields have sensible defaults."""
    monkeypatch.delenv("EMBEDDINGS_HOST", raising=False)
    monkeypatch.delenv("EMBEDDINGS_PORT", raising=False)
    monkeypatch.delenv("COLLECTIONS_CONFIG", raising=False)
    monkeypatch.delenv("SKILLS_DB_PATH", raising=False)
    config = ManagerConfig.from_env()
    assert config.embeddings_host == "127.0.0.1"
    assert config.embeddings_port == 8082
    assert config.collections_config == "/etc/llama/collections.json"
    assert config.skills_db_path == "/opt/llama/data/skills.db"


def test_config_embeddings_from_env(monkeypatch):
    """Embedding/collection config reads from env vars."""
    monkeypatch.setenv("EMBEDDINGS_HOST", "10.0.0.5")
    monkeypatch.setenv("EMBEDDINGS_PORT", "9090")
    monkeypatch.setenv("COLLECTIONS_CONFIG", "/tmp/cols.json")
    monkeypatch.setenv("SKILLS_DB_PATH", "/tmp/skills.db")
    config = ManagerConfig.from_env()
    assert config.embeddings_host == "10.0.0.5"
    assert config.embeddings_port == 9090
    assert config.collections_config == "/tmp/cols.json"
    assert config.skills_db_path == "/tmp/skills.db"


def test_config_embeddings_url(monkeypatch):
    """embeddings_url property builds correct URL."""
    monkeypatch.delenv("EMBEDDINGS_HOST", raising=False)
    monkeypatch.delenv("EMBEDDINGS_PORT", raising=False)
    config = ManagerConfig.from_env()
    assert config.embeddings_url == "http://127.0.0.1:8082"
