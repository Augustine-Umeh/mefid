import httpx
from typing import List, Dict, Any
from exports.schema.constants import INDEXER_SERVICE


class IndexerClient:
    """Client for the Indexer service.

    Supports two usage patterns:
      * As an async context manager (`async with IndexerClient() as c: ...`)
      * As a long-lived client owned by the FastAPI app
        (`await client.connect()` at startup, `await client.close()` at shutdown).
    """

    def __init__(self, base_url: str = INDEXER_SERVICE):
        self.base_url = base_url
        self.client: httpx.AsyncClient | None = None

    async def connect(self) -> "IndexerClient":
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

    async def add_vectors(
        self,
        media_id: str,
        embeddings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Add embeddings to the FAISS index.

        Args:
            media_id: Media identifier (matches the Supabase `media.id`)
            embeddings: List of {"timestamp": float, "embedding": List[float]}

        Returns:
            {"start_index": int, "count": int}
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        response = await self.client.post(
            "/add/",
            json={
                "media_id": media_id,
                "embeddings": embeddings,
            },
        )
        response.raise_for_status()
        return response.json()

    async def query_vectors(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> Dict[str, Any]:
        """
        Search the FAISS index for similar embeddings.

        Args:
            query_embedding: Query vector (e.g. 512-d for CLIP ViT-B/32)
            top_k: Number of results to return

        Returns:
            {"distances": List[float], "indices": List[int]}
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        response = await self.client.post(
            "/search/",
            json={
                "query_embedding": query_embedding,
                "top_k": top_k,
            },
        )
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> Dict[str, Any]:
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        response = await self.client.get("/")
        response.raise_for_status()
        return response.json()
