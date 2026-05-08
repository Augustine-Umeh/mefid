import httpx
from typing import Dict, Any
from exports.schema.constants import MEDIA_PROCESSOR_SERVICE
from exports.utils.logger import get_logger
from exports.schema.models import ExtractFramesResponse

logger = get_logger()


class MediaProcessorClient:
    """Client for the Media Processor service.

    Supports two usage patterns:
      * As an async context manager (`async with MediaProcessorClient() as c: ...`)
      * As a long-lived client owned by the FastAPI app
        (`await client.connect()` at startup, `await client.close()` at shutdown).
    """

    def __init__(self, base_url: str = MEDIA_PROCESSOR_SERVICE):
        self.base_url = base_url
        self.client: httpx.AsyncClient | None = None

    async def connect(self) -> "MediaProcessorClient":
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

    async def extract_frames_fixed_interval(
        self,
        video_path: str,
        interval_seconds: float,
        source_url: str,
        minio_path_url: str,
        file_name: str,
    ) -> ExtractFramesResponse:
        """
        Extract frames at fixed time intervals.

        Args:
            video_path: MinIO path to the video file
            interval_seconds: Seconds between frames
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        response = await self.client.post(
            "/extract/",
            params={
                "strategy": "fixed_interval",
                "interval_seconds": interval_seconds,
            },
            json={
                "video_path": video_path,
                "source_url": source_url,
                "minio_path_url": minio_path_url,
                "file_name": file_name,
            },
        )
        response.raise_for_status()
        return ExtractFramesResponse(**response.json())

    async def extract_frames_scene_detect(
        self,
        video_path: str,
        threshold: int,
        source_url: str,
        minio_path_url: str,
        file_name: str,
    ) -> ExtractFramesResponse:
        """
        Extract frames using PySceneDetect.

        Args:
            video_path: MinIO path to the video file
            threshold: Scene detection threshold (lower = more sensitive)
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        response = await self.client.post(
            "/extract/",
            params={
                "strategy": "scene_detect",
                "threshold": threshold,
            },
            json={
                "video_path": video_path,
                "source_url": source_url,
                "minio_path_url": minio_path_url,
                "file_name": file_name,
            },
        )
        response.raise_for_status()
        return ExtractFramesResponse(**response.json())

    async def health_check(self) -> Dict[str, str]:
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        response = await self.client.get("/")
        response.raise_for_status()
        return response.json()
