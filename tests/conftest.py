# tests/conftest.py
"""Shared test fixtures for the model manager test suite."""
import pytest


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
