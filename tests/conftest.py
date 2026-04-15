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
def test_config(tmp_models_dir, tmp_env_file):
    """Create a ManagerConfig pointing at temporary test paths."""
    from manager.config import ManagerConfig
    return ManagerConfig(
        host="127.0.0.1",
        port=8080,
        llama_server_host="127.0.0.1",
        llama_server_port=8081,
        models_dir=tmp_models_dir,
        llama_server_env=tmp_env_file,
        queue_limit=20,
        swap_timeout=5,
        log_file="/dev/null",
        embeddings_host="127.0.0.1",
        embeddings_port=8082,
        collections_config="/dev/null",
        skills_db_path="",
    )


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
        queue_limit=test_config.queue_limit,
        swap_timeout=test_config.swap_timeout,
        log_file=test_config.log_file,
        embeddings_host="127.0.0.1",
        embeddings_port=8082,
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
        yield client
