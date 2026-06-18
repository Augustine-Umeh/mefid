import httpx
from typing import Dict, List

from exports.schema.constants import EMBEDDER_SERVICE
from exports.schema.models import (
    EmbedImageItem,
    EmbedImagesRequest,
    EmbedImagesResponse,
    EmbedTextBatchRequest,
    EmbedTextBatchResponse,
    EmbedTextRequest,
    EmbedTextResponse,
    EmbeddingResult,
    EmbedTextItem,
    TextEmbeddingResult,
)


class EmbedderClient:
    """Client for the Embedder service.

    Supports two usage patterns:
      * Async context manager (``async with EmbedderClient() as c: ...``).
      * Long-lived client owned by FastAPI (``await client.connect()`` at
        startup, ``await client.close()`` at shutdown).
    """

    def __init__(self, base_url: str = EMBEDDER_SERVICE):
        self.base_url = base_url
        self.client: httpx.AsyncClient | None = None

    async def connect(self) -> "EmbedderClient":
        if self.client is None:
            self.client = httpx.AsyncClient(base_url=self.base_url, timeout=300.0)
        return self

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def __aenter__(self):
        return await self.connect()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def embed_images(
        self, frames: List[EmbedImageItem]
    ) -> List[EmbeddingResult]:
        """Embed a batch of base64-encoded JPEG frames.

        Returns one ``EmbeddingResult`` per input frame, keyed by
        ``frame_id`` so the indexer can attribute vectors to rows.
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        payload = EmbedImagesRequest(frames=frames).model_dump(mode="json")
        response = await self.client.post("/embed/images/", json=payload)
        response.raise_for_status()
        return EmbedImagesResponse(**response.json()).embeddings

    async def embed_text(self, text: str) -> List[float]:
        """Embed a search query string into a single CLIP vector."""
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        payload = EmbedTextRequest(text=text).model_dump(mode="json")
        response = await self.client.post("/embed/text/", json=payload)
        response.raise_for_status()
        return EmbedTextResponse(**response.json()).embedding

    async def embed_texts(self, items: List[EmbedTextItem]) -> List[TextEmbeddingResult]:
        """Embed a batch of transcript strings into CLIP vectors."""
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")
        if not items:
            return []

        payload = EmbedTextBatchRequest(texts=items).model_dump(mode="json")
        response = await self.client.post("/embed/text/batch", json=payload)
        response.raise_for_status()
        return EmbedTextBatchResponse(**response.json()).embeddings

    async def health_check(self) -> Dict[str, str]:
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        response = await self.client.get("/")
        response.raise_for_status()
        return response.json()
