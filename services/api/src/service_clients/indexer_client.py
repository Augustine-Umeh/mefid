import httpx
from typing import Dict, List, Union
from uuid import UUID

from exports.schema.constants import INDEXER_SERVICE
from exports.schema.models import (
    AddVectorItem,
    AddVectorsRequest,
    AddVectorsResponse,
    IndexerVectorHit,
    SearchVectorsRequest,
    SearchVectorsResponse,
)


IdLike = Union[str, UUID]


def _as_uuid(value: IdLike) -> UUID:
    return value if isinstance(value, UUID) else UUID(value)


class IndexerClient:
    """Client for the Indexer service.

    Supports two usage patterns:
      * Async context manager (``async with IndexerClient() as c: ...``).
      * Long-lived client owned by FastAPI (``await client.connect()`` at
        startup, ``await client.close()`` at shutdown).
    """

    def __init__(self, base_url: str = INDEXER_SERVICE):
        self.base_url = base_url
        self.client: httpx.AsyncClient | None = None

    async def connect(self) -> "IndexerClient":
        if self.client is None:
            self.client = httpx.AsyncClient(base_url=self.base_url, timeout=900.0)
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
        media_id: IdLike,
        vectors: List[AddVectorItem],
    ) -> AddVectorsResponse:
        """Add embeddings to FAISS and persist `embeddings` rows.

        The indexer service is responsible for both the FAISS write and the
        Supabase insert (it's the only place that knows the resulting
        ``faiss_index_id`` for each vector).
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        payload = AddVectorsRequest(
            media_id=_as_uuid(media_id),
            vectors=vectors,
        ).model_dump(mode="json")
        response = await self.client.post("vectors/add/", json=payload)
        response.raise_for_status()
        return AddVectorsResponse(**response.json())

    async def search_vectors(
        self,
        embedding: List[float],
        top_k: int,
    ) -> List[IndexerVectorHit]:
        """Nearest-neighbour search against the FAISS index."""
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        payload = SearchVectorsRequest(
            embedding=embedding,
            top_k=top_k,
        ).model_dump(mode="json")
        response = await self.client.post("vectors/search/", json=payload)
        response.raise_for_status()
        return SearchVectorsResponse(**response.json()).hits

    async def health_check(self) -> Dict[str, str]:
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        response = await self.client.get("/")
        response.raise_for_status()
        return response.json()
