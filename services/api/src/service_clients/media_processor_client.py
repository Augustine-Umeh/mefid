import httpx
from typing import List, Dict, Any
from fastapi import UploadFile
from exports.schema.constants import MEDIA_PROCESSOR_SERVICE
from exports.utils.logger import get_logger
from exports.schema.models import MediaCreate, ExtractFramesResponse

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
        video_path: str,
        interval_seconds: float,
        source_url: str,
        minio_path_url: str,
        file_name: str
    ) -> ExtractFramesResponse:
        """
        Extract frames at fixed intervals.
        
        Args:
            video_path: Minio path to the video file
            interval_seconds: Seconds between frames
            
        Returns:
            
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")
        

        response = await self.client.post(
            "/extract",
            params={
                "strategy": "fixed_interval",
                "interval_seconds": interval_seconds,
            },
            json={
                "video_path": video_path,
                "source_url": source_url,
                "minio_path_url": minio_path_url,
                "file_name": file_name,
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def extract_frames_scene_detect(
        self,
        video_path: str,
        threshold: int,
        source_url: str,
        minio_path_url: str,
        file_name: str
    ) -> ExtractFramesResponse:
        """
        Extract frames using PySceneDetect.
        
        Args:
            video_path: Minio path to the video file
            threshold: Scene detection threshold
            
        Returns:
            List of {"timestamp": float, "frame_data": str (base64)}
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")
        
        response = await self.client.post(
            "/extract",
            params={
                "strategy": "fixed_interval",
                "threshold": threshold,
            },
            json={
                "video_path": video_path,
                "source_url": source_url,
                "minio_path_url": minio_path_url,
                "file_name": file_name,
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