# tests/test_config.py
"""Tests for configuration loading from environment variables."""
import dataclasses
import pytest
from manager.config import ManagerConfig


def test_config_loads_from_env(monkeypatch):
    """Config should load all values from environment variables."""
    monkeypatch.setenv("HOST", "192.168.1.50")
    # Values deliberately differ from the from_env defaults below, so a
    # broken env read cannot hide behind the fallback.
    monkeypatch.setenv("PORT", "18080")
    monkeypatch.setenv("LLAMA_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("LLAMA_SERVER_PORT", "18081")
    monkeypatch.setenv("MODELS_DIR", "/opt/llama/models")
    monkeypatch.setenv("LLAMA_SERVER_ENV", "/etc/llama/llama-server.env")
    monkeypatch.setenv("QUEUE_LIMIT", "20")
    monkeypatch.setenv("SWAP_TIMEOUT", "120")
    monkeypatch.setenv("LOG_FILE", "/var/log/llama/manager.log")

    config = ManagerConfig.from_env()

    assert config.host == "192.168.1.50"
    assert config.port == 18080
    assert config.llama_server_host == "127.0.0.1"
    assert config.llama_server_port == 18081
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
    monkeypatch.setenv("LLAMA_SERVER_PORT", "18081")

    config = ManagerConfig.from_env()

    assert config.llama_server_url == "http://127.0.0.1:18081"


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


def test_from_env_llama_server_unit_default(monkeypatch):
    monkeypatch.delenv("LLAMA_SERVER_UNIT", raising=False)
    from manager.config import ManagerConfig
    cfg = ManagerConfig.from_env()
    assert cfg.llama_server_unit == "llama-server.service"


def test_from_env_restart_timeout_default(monkeypatch):
    monkeypatch.delenv("RESTART_TIMEOUT", raising=False)
    from manager.config import ManagerConfig
    cfg = ManagerConfig.from_env()
    assert cfg.restart_timeout == 150


def test_from_env_restart_timeout_override(monkeypatch):
    monkeypatch.setenv("RESTART_TIMEOUT", "200")
    from manager.config import ManagerConfig
    cfg = ManagerConfig.from_env()
    assert cfg.restart_timeout == 200


from manager.slot_config import SlotConfig


def test_from_env_populates_slots_in_order(monkeypatch):
    # SLOT_AUX_PORT and SLOT_EGPU_PORT are required: only 'main' has
    # built-in defaults, so each extra slot needs an explicit port
    # (see test_slot_config.py).
    monkeypatch.setenv("SLOTS", "main,aux,egpu")
    monkeypatch.setenv("SLOT_AUX_PORT", "18084")
    monkeypatch.setenv("SLOT_EGPU_PORT", "18085")
    from manager.config import ManagerConfig
    cfg = ManagerConfig.from_env()
    assert [s.name for s in cfg.slots] == ["main", "aux", "egpu"]


def test_from_env_defaults_to_main_only(monkeypatch):
    monkeypatch.delenv("SLOTS", raising=False)
    from manager.config import ManagerConfig
    cfg = ManagerConfig.from_env()
    assert [s.name for s in cfg.slots] == ["main"]


def test_legacy_ports_flow_into_slots(monkeypatch):
    monkeypatch.delenv("SLOTS", raising=False)
    monkeypatch.setenv("LLAMA_SERVER_PORT", "18081")
    from manager.config import ManagerConfig
    cfg = ManagerConfig.from_env()
    by_name = {s.name: s for s in cfg.slots}
    assert by_name["main"].port == 18081


def test_test_config_fixture_slots_match_its_own_ports_and_env_files(test_config):
    """test_config's slots match the fields the fixture also sets, so
    ServerState(test_config) has a real two-slot config to build from --
    not just an empty tuple that happens to type-check. The egpu slot has no
    top-level field anymore, so it is pinned to its own fixture values."""
    assert [s.name for s in test_config.slots] == ["main", "egpu"]
    by_name = {s.name: s for s in test_config.slots}
    assert by_name["main"].port == test_config.llama_server_port
    assert by_name["main"].env_file == test_config.llama_server_env
    assert by_name["egpu"].port == 18083
    assert by_name["egpu"].systemd_unit == "llama-server-egpu.service"


def test_manager_config_has_no_batch_fields():
    names = {f.name for f in dataclasses.fields(ManagerConfig)}
    names |= {n for n in dir(ManagerConfig) if not n.startswith("_")}
    assert not any(n.startswith("batch_") for n in names)


