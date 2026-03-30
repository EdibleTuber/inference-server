# manager/embeddings.py
"""
Async client for the llama.cpp embeddings server.

Wraps the OpenAI-compatible /v1/embeddings endpoint exposed by a
dedicated llama.cpp instance running with --embedding flag.
"""
import httpx
import logging

logger = logging.getLogger(__name__)


class EmbeddingsClient:
    """Async HTTP client for generating text embeddings."""

    def __init__(self, base_url: str, model: str = "nomic-embed-text"):
        self._client = httpx.AsyncClient(base_url=base_url)
        self._model = model

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string. Returns a float vector."""
        response = await self._client.post(
            "/v1/embeddings",
            json={"model": self._model, "input": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one call. Returns list of float vectors."""
        response = await self._client.post(
            "/v1/embeddings",
            json={"model": self._model, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in data]

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()
