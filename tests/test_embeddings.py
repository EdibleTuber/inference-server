# tests/test_embeddings.py
"""Tests for the embeddings client."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx


@pytest.fixture
def embeddings_url():
    return "http://127.0.0.1:8082"


@pytest.mark.asyncio
async def test_embed_text_returns_vector(embeddings_url):
    """embed_text returns a list of floats from the embeddings server."""
    from manager.embeddings import EmbeddingsClient

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}],
    }
    mock_response.raise_for_status = MagicMock()

    client = EmbeddingsClient(embeddings_url)
    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await client.embed_text("hello world")

    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_text_sends_correct_payload(embeddings_url):
    """embed_text sends model and input in OpenAI format."""
    from manager.embeddings import EmbeddingsClient

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1]}],
    }
    mock_response.raise_for_status = MagicMock()

    client = EmbeddingsClient(embeddings_url)
    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
        await client.embed_text("test input")

    mock_post.assert_called_once_with(
        "/v1/embeddings",
        json={"model": "nomic-embed-text", "input": "test input"},
        timeout=30,
    )


@pytest.mark.asyncio
async def test_embed_text_raises_on_server_error(embeddings_url):
    """embed_text raises on non-200 response."""
    from manager.embeddings import EmbeddingsClient

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=mock_response
    )

    client = EmbeddingsClient(embeddings_url)
    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response):
        with pytest.raises(httpx.HTTPStatusError):
            await client.embed_text("hello")


@pytest.mark.asyncio
async def test_embed_batch_returns_multiple_vectors(embeddings_url):
    """embed_batch returns one vector per input text."""
    from manager.embeddings import EmbeddingsClient

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ],
    }
    mock_response.raise_for_status = MagicMock()

    client = EmbeddingsClient(embeddings_url)
    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await client.embed_batch(["hello", "world"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2]
    assert result[1] == [0.3, 0.4]
