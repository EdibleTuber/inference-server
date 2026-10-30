# tests/conftest.py
"""Shared test fixtures for the model manager test suite."""
import json
import socket
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from tests.live_ports import LIVE_PORTS, LivePortBlocked


@pytest.fixture(autouse=True)
def block_live_ports(monkeypatch):
    """Block every connect to a live manager slot port, suite-wide (Q13).

    Any connect to a port in LIVE_PORTS, on any host, raises
    LivePortBlocked; all other connects pass through (AF_UNIX is left
    alone). tests/test_consumer.py's no_network fixture is stricter and
    stays in place: it fails on ANY real TCP connect.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _live_port(address):
        # Address tuples are (host, port, ...); AF_UNIX is a path string
        # or an abstract (b"\0name",) pair with no port at index 1 that
        # could match.
        return address[1] if isinstance(address, tuple) and len(address) > 1 else None

    def _check(self, address):
        port = _live_port(address)
        if self.family in (socket.AF_INET, socket.AF_INET6) and port in LIVE_PORTS:
            raise LivePortBlocked(f"test connected to a live port ({address[0]}:{port})")

    def guarded_connect(self, address):
        _check(self, address)
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        _check(self, address)
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)


@pytest.fixture
def tmp_models_dir(tmp_path):
    """Create a temporary models directory with sample GGUF files."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "test-model-q4.gguf").touch()
    (models_dir / "test-model-q8.gguf").touch()
    return str(models_dir)


@pytest.fixture
def tmp_env_file(tmp_path):
    """Create a temporary llama-server env file for testing model swaps."""
    env_file = tmp_path / "llama-server.env"
    env_file.write_text(
        "MODEL_PATH=\nSAMPLING_ARGS=\nN_GPU_LAYERS=-1\nCTX_SIZE=4096\nHOST=127.0.0.1\nPORT=18081\n"
    )
    return str(env_file)


@pytest.fixture
def tmp_egpu_env_file(tmp_path):
    """Create a temporary llama-server-egpu env file for testing egpu swaps."""
    env_file = tmp_path / "llama-server-egpu.env"
    env_file.write_text(
        "MODEL_PATH=\nSAMPLING_ARGS=\nDEVICE=Vulkan0\nCTX_SIZE=16384\nHOST=127.0.0.1\nPORT=18083\n"
    )
    return str(env_file)


@pytest.fixture
def test_config(tmp_models_dir, tmp_env_file, tmp_egpu_env_file):
    """Create a ManagerConfig pointing at temporary test paths."""
    from manager.config import ManagerConfig
    from manager.slot_config import SlotConfig
    return ManagerConfig(
        host="127.0.0.1",
        port=18080,
        llama_server_host="127.0.0.1",
        llama_server_port=18081,
        models_dir=tmp_models_dir,
        llama_server_env=tmp_env_file,
        llama_server_unit="llama-server.service",
        queue_limit=20,
        swap_timeout=5,
        log_file="/dev/null",
        embeddings_host="127.0.0.1",
        embeddings_port=18082,
        collections_config="/dev/null",
        skills_db_path="",
        slots=(
            SlotConfig(
                name="main", host="127.0.0.1", port=18081,
                env_file=tmp_env_file, systemd_unit="llama-server.service",
                queue_limit=20,
            ),
            SlotConfig(
                name="egpu", host="127.0.0.1", port=18083,
                env_file=tmp_egpu_env_file, systemd_unit="llama-server-egpu.service",
                queue_limit=20,
            ),
        ),
    )


@pytest.fixture
def tmp_aux_env_file(tmp_path):
    """Create a temporary env file for a third ('aux') slot, used to test N-slot
    configs beyond the 'main' slot (the only one with built-in defaults)."""
    env_file = tmp_path / "llama-server-aux.env"
    env_file.write_text(
        "MODEL_PATH=\nCTX_SIZE=8192\nHOST=127.0.0.1\nPORT=8085\n"
    )
    return str(env_file)


@pytest.fixture
def three_slot_config(test_config, tmp_aux_env_file):
    """test_config extended with a third configured slot ('aux').

    A slot without built-in defaults (only 'main' has any) has no fallback
    port (see slot_config.build_slots), so it is given one explicitly here.
    """
    import dataclasses
    from manager.slot_config import SlotConfig

    aux_slot = SlotConfig(
        name="aux", host="127.0.0.1", port=8085,
        env_file=tmp_aux_env_file, systemd_unit="llama-server-aux.service",
        queue_limit=20,
    )
    return dataclasses.replace(test_config, slots=test_config.slots + (aux_slot,))


@pytest.fixture
def client_with_three_slots(three_slot_config, monkeypatch):
    """Client for a three-slot config, with the swap itself short-circuited.

    Mirrors the existing /swap tests in tests/test_endpoints.py (e.g.
    test_swap_valid_main), which stub manager.swap.ModelSwapper.swap_to
    directly rather than mocking subprocess/HTTP at a lower level -- that
    method already encapsulates the systemctl restart and health poll that
    tests/test_swap.py mocks when it tests ModelSwapper in isolation.
    """
    from manager.app import create_app

    async def fake_swap_to(self, model_path, sampling_args=""):
        return True

    monkeypatch.setattr("manager.swap.ModelSwapper.swap_to", fake_swap_to)
    app = create_app(three_slot_config)
    return TestClient(app)


@pytest.fixture
def skills_dir(tmp_path):
    """Create a skills directory with one skill and one workflow."""
    skill_dir = tmp_path / "skills" / "Security" / "Recon"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: Recon\ndescription: Security reconnaissance. USE WHEN recon, bug bounty.\n---\n\n# Recon\n"
    )
    wf_dir = skill_dir / "Workflows"
    wf_dir.mkdir()
    (wf_dir / "PassiveRecon.md").write_text(
        "# Passive Recon\n\n## Purpose\n\nGather info without touching target.\n"
    )
    return str(tmp_path / "skills")


@pytest.fixture
def collections_config(tmp_path, skills_dir):
    """Create a collections.json config file."""
    config = [{"id": "skills", "source_dir": skills_dir, "doc_type": "skill"}]
    config_path = tmp_path / "collections.json"
    config_path.write_text(json.dumps(config))
    return str(config_path)


@pytest.fixture
def collection_config(test_config, tmp_path, collections_config):
    """Extend test_config with collection settings.

    Built via dataclasses.replace on test_config (as three_slot_config above
    does) rather than a manual field-by-field copy, so it inherits
    test_config.slots -- a hand-copy that predates the slot work previously
    omitted slots=, producing a ManagerConfig with zero slots that could
    never serve a chat request.
    """
    import dataclasses
    return dataclasses.replace(
        test_config,
        collections_config=collections_config,
        skills_db_path=str(tmp_path / "test.db"),
    )


@pytest.fixture
def collection_app(collection_config):
    """Create app with collection support and mocked embeddings."""
    with patch("manager.app.EmbeddingsClient") as MockEmbClient:
        mock_instance = AsyncMock()
        mock_instance.embed_text = AsyncMock(return_value=[0.1] * 768)
        mock_instance.embed_batch = AsyncMock(return_value=[[0.1] * 768])
        mock_instance.close = AsyncMock()
        MockEmbClient.return_value = mock_instance

        from manager.app import create_app
        app = create_app(collection_config)
        yield app


@pytest.fixture
def collection_client(collection_app):
    with TestClient(collection_app) as client:
        client.portal.call(client.app.state.server.reindex_registry.wait_idle)
        yield client
