import httpx
from typing import List, Dict, Any
from exports.schema.constants import MEDIA_PROCESSOR_SERVICE
from exports.utils.logger import get_logger

logger = get_logger()


class MediaProcessorClient:
    """Client for Media Processor service"""
    
    def __init__(self, base_url: str = MEDIA_PROCESSOR_SERVICE):
        self.base_url = base_url
        self.client: httpx.AsyncClient | None = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=300.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def extract_frames_fixed_interval(
        self, 
        video_id: str,
        interval_seconds: float
    ) -> List[Dict[str, Any]]:
        """
        Extract frames at fixed intervals.
        
        Args:
            video_id: ID of video in MinIO
            interval_seconds: Seconds between frames
            
        Returns:
            List of {"timestamp": float, "frame_data": str (base64)}
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")
        

        response = await self.client.post(
            "/extract/fixed_interval",
            json={
                "video_id": video_id,
                "interval_seconds": interval_seconds
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def extract_frames_scene_detect(
        self,
        video_id: str,
        threshold: int
    ) -> List[Dict[str, Any]]:
        """
        Extract frames using PySceneDetect.
        
        Args:
            video_id: ID of video in MinIO
            threshold: Scene detection threshold
            
        Returns:
            List of {"timestamp": float, "frame_data": str (base64)}
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")
        
        response = await self.client.post(
            "/extract/scene_detect",
            json={
                "video_id": video_id,
                "threshold": threshold
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def health_check(self) -> Dict[str, str]:
        """Check if Media Processor service is healthy"""
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")
        
        response = await self.client.get("/")
        response.raise_for_status()
        
        return response.json()