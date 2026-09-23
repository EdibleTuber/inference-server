# tests/conftest.py
"""Shared test fixtures for the model manager test suite."""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


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
        "MODEL_PATH=\nN_GPU_LAYERS=-1\nCTX_SIZE=4096\nHOST=127.0.0.1\nPORT=8081\n"
    )
    return str(env_file)


@pytest.fixture
def tmp_batch_env_file(tmp_path):
    """Create a temporary llama-server-batch env file for testing batch swaps."""
    env_file = tmp_path / "llama-server-batch.env"
    env_file.write_text(
        "MODEL_PATH=\nDEVICE=Vulkan0\nCTX_SIZE=16384\nHOST=127.0.0.1\nPORT=8083\n"
    )
    return str(env_file)


@pytest.fixture
def test_config(tmp_models_dir, tmp_env_file, tmp_batch_env_file):
    """Create a ManagerConfig pointing at temporary test paths."""
    from manager.config import ManagerConfig
    from manager.slot_config import SlotConfig
    return ManagerConfig(
        host="127.0.0.1",
        port=8080,
        llama_server_host="127.0.0.1",
        llama_server_port=8081,
        models_dir=tmp_models_dir,
        llama_server_env=tmp_env_file,
        llama_server_unit="llama-server.service",
        queue_limit=20,
        swap_timeout=5,
        log_file="/dev/null",
        embeddings_host="127.0.0.1",
        embeddings_port=8082,
        collections_config="/dev/null",
        skills_db_path="",
        batch_server_host="127.0.0.1",
        batch_server_port=8083,
        batch_server_env=tmp_batch_env_file,
        batch_server_unit="llama-server-batch.service",
        batch_queue_limit=20,
        batch_model_default="test-batch-model",
        slots=(
            SlotConfig(
                name="main", host="127.0.0.1", port=8081,
                env_file=tmp_env_file, systemd_unit="llama-server.service",
                queue_limit=20,
            ),
            SlotConfig(
                name="batch", host="127.0.0.1", port=8083,
                env_file=tmp_batch_env_file, systemd_unit="llama-server-batch.service",
                queue_limit=20,
            ),
        ),
    )


@pytest.fixture
def tmp_re_env_file(tmp_path):
    """Create a temporary env file for a third ('re') slot, used to test N-slot
    configs beyond the legacy main/batch pair."""
    env_file = tmp_path / "llama-server-re.env"
    env_file.write_text(
        "MODEL_PATH=\nCTX_SIZE=8192\nHOST=127.0.0.1\nPORT=8085\n"
    )
    return str(env_file)


@pytest.fixture
def three_slot_config(test_config, tmp_re_env_file):
    """test_config extended with a third configured slot ('re').

    A slot outside the legacy main/batch pair has no fallback port (see
    slot_config.build_slots), so it is given one explicitly here.
    """
    import dataclasses
    from manager.slot_config import SlotConfig

    re_slot = SlotConfig(
        name="re", host="127.0.0.1", port=8085,
        env_file=tmp_re_env_file, systemd_unit="llama-server-re.service",
        queue_limit=20,
    )
    return dataclasses.replace(test_config, slots=test_config.slots + (re_slot,))


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

    async def fake_swap_to(self, model_path):
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
    """Extend test_config with collection settings."""
    from manager.config import ManagerConfig
    return ManagerConfig(
        host=test_config.host,
        port=test_config.port,
        llama_server_host=test_config.llama_server_host,
        llama_server_port=test_config.llama_server_port,
        models_dir=test_config.models_dir,
        llama_server_env=test_config.llama_server_env,
        llama_server_unit=test_config.llama_server_unit,
        queue_limit=test_config.queue_limit,
        swap_timeout=test_config.swap_timeout,
        log_file=test_config.log_file,
        embeddings_host="127.0.0.1",
        embeddings_port=8082,
        collections_config=collections_config,
        skills_db_path=str(tmp_path / "test.db"),
        batch_server_host=test_config.batch_server_host,
        batch_server_port=test_config.batch_server_port,
        batch_server_env=test_config.batch_server_env,
        batch_server_unit=test_config.batch_server_unit,
        batch_queue_limit=test_config.batch_queue_limit,
        batch_model_default=test_config.batch_model_default,
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
        yield client
