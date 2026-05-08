import httpx
from typing import List, Dict, Any
from exports.schema.constants import EMBEDDER_SERVICE


class EmbedderClient:
    """Client for the Embedder service.

    Supports two usage patterns:
      * As an async context manager (`async with EmbedderClient() as c: ...`)
      * As a long-lived client owned by the FastAPI app
        (`await client.connect()` at startup, `await client.close()` at shutdown).
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
        self,
        frames: List[Dict[str, Any]],
        media_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for video frames.

        Args:
            frames: List of {"timestamp": float, "frame_data": str (base64)}
            media_id: Media identifier these frames belong to

        Returns:
            List of {"timestamp": float, "embedding": List[float]}
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        response = await self.client.post(
            "/embed/images/",
            json={
                "frames": frames,
                "media_id": media_id,
            },
        )
        response.raise_for_status()
        return response.json()

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a text query.

        Args:
            text: Search query text

        Returns:
            Embedding vector (e.g. 512-d for CLIP ViT-B/32)
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        response = await self.client.post(
            "/embed/text/",
            json={"text": text},
        )
        response.raise_for_status()
        data = response.json()
        return data["embedding"]

    async def health_check(self) -> Dict[str, str]:
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        response = await self.client.get("/")
        response.raise_for_status()
        return response.json()
