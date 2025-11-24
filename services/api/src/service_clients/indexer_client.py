import httpx
from typing import List, Dict, Any
from exports.schema.constants import INDEXER_SERVICE

class IndexerClient:
    """Client for Indexer service"""
    
    def __init__(self, base_url: str = INDEXER_SERVICE):
        self.base_url = base_url
        self.client: httpx.AsyncClient | None = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=300.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
            
    async def add_vectors(
        self,
        video_id: str,
        embeddings: List[Dict[str, Any]]
    ) -> int:
        """
        Add embeddings to FAISS index.
        
        Args:
            video_id: Video identifier
            embeddings: List of {"timestamp": float, "embedding": List[float]}
            
        Returns:
            {"start_index": int, "count": int}
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")
        
        response = await self.client.post(
            "/add/",
            json={
                "video_id": video_id,
                "embeddings": embeddings
            }
        )
        response.raise_for_status()
        return response.json()
        
    async def query_vectors(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> Dict[str, Any]:
        """
        Search FAISS index for similar embeddings.
        
        Args:
            query_embedding: Query vector (512-dim)
            top_k: Number of results to return
            
        Returns:
            {
                "distances": List[float],
                "indices": List[int]
            }
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")
        
        response = await self.client.post(
            "/search/",
            json={
                "query_embedding": query_embedding,
                "top_k": top_k
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the Indexer service.
        
        Returns:
            Health status as a dictionary.
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")
        
        response = await self.client.get("/")
        response.raise_for_status()
        return response.json()