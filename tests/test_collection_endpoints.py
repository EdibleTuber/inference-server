# tests/test_collection_endpoints.py
"""Tests for the collection and embeddings API endpoints."""
import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


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


def test_list_collections(collection_client):
    """GET /collections returns registered collections."""
    response = collection_client.get("/collections")
    assert response.status_code == 200
    data = response.json()
    assert "collections" in data
    assert len(data["collections"]) == 1
    assert data["collections"][0]["id"] == "skills"


def test_search_collection(collection_client):
    """POST /collections/skills/search returns results."""
    response = collection_client.post(
        "/collections/skills/search",
        json={"query": "network reconnaissance"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0


def test_search_collection_not_found(collection_client):
    """POST /collections/nonexistent/search returns 404."""
    response = collection_client.post(
        "/collections/nonexistent/search",
        json={"query": "test"},
    )
    assert response.status_code == 404


def test_get_document(collection_client):
    """GET /collections/skills/docs/{id} returns full document."""
    search_resp = collection_client.post(
        "/collections/skills/search",
        json={"query": "recon"},
    )
    results = search_resp.json()["results"]
    assert len(results) > 0
    doc_id = results[0]["id"]
    response = collection_client.get(f"/collections/skills/docs/{doc_id}")
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert data["id"] == doc_id


def test_get_document_not_found(collection_client):
    """GET /collections/skills/docs/nonexistent returns 404."""
    response = collection_client.get("/collections/skills/docs/nonexistent")
    assert response.status_code == 404


def test_embeddings_proxy(collection_client):
    """POST /v1/embeddings proxies to embedding server."""
    with patch("manager.app.httpx.AsyncClient") as MockClient:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "nomic-embed-text",
        }).encode()
        mock_resp.headers = {"content-type": "application/json"}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = mock_client

        response = collection_client.post(
            "/v1/embeddings",
            json={"model": "nomic-embed-text", "input": "hello"},
        )
        assert response.status_code == 200


def test_reindex_post_rejects_unknown_collection(collection_client):
    resp = collection_client.post("/collections/does-not-exist/reindex", json={})
    assert resp.status_code == 404


def test_reindex_post_returns_job_id(collection_client):
    resp = collection_client.post("/collections/skills/reindex", json={})
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in ("queued", "running", "done")
    assert body["collection_id"] == "skills"


def test_reindex_post_accepts_paths(collection_client):
    resp = collection_client.post(
        "/collections/skills/reindex",
        json={"paths": ["/tmp/a.md", "/tmp/b.md"]},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["paths"] == ["/tmp/a.md", "/tmp/b.md"]


def test_reindex_post_missing_body_defaults_to_full_scan(collection_client):
    resp = collection_client.post("/collections/skills/reindex")
    assert resp.status_code == 202
    body = resp.json()
    assert body["paths"] is None


def test_reindex_post_paths_must_be_list(collection_client):
    resp = collection_client.post(
        "/collections/skills/reindex",
        json={"paths": "not a list"},
    )
    assert resp.status_code == 400


def test_reindex_post_paths_must_be_strings(collection_client):
    resp = collection_client.post(
        "/collections/skills/reindex",
        json={"paths": [1, 2, 3]},
    )
    assert resp.status_code == 400
