import httpx
from typing import List, Dict, Any
from exports.schema.constants import EMBEDDER_SERVICE


class EmbedderClient:
    """Client for Embedder service"""
    
    def __init__(self, base_url: str = EMBEDDER_SERVICE):
        self.base_url = base_url
        self.client: httpx.AsyncClient | None = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=300.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def embed_images(
        self,
        frames: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for video frames.
        
        Args:
            frames: List of {"timestamp": float, "frame_data": str (base64)}
            
        Returns:
            List of {"timestamp": float, "embedding": List[float]}
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")
        
        response = await self.client.post(
            "/embed/images/",
            json={"frames": frames}
        )
        response.raise_for_status()
        return response.json()
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text query.
        
        Args:
            text: Search query text
            
        Returns:
            Embedding vector (512-dim for CLIP)
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")
        
        response = await self.client.post(
            "/embed/text/",
            json={"text": text}
        )
        response.raise_for_status()
        data = response.json()
        return data["embedding"]
    
    async def health_check(self) -> Dict[str, str]:
        """Check if Embedder service is healthy"""
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")
        
        response = await self.client.get("/")
        response.raise_for_status()
        return response.json()
