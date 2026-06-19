import httpx
from typing import Dict

from exports.schema.constants import EMBEDDER_SERVICE  # noqa: F401  (kept for parity import patterns)
from exports.schema.constants import MEDIA_PROCESSOR_SERVICE
from exports.schema.constants import ExtractionStrategy
from exports.schema.models import ExtractFramesResponse
from exports.utils.logger import get_logger

logger = get_logger()


class MediaProcessorClient:
    """Client for the Media Processor service.

    Supports two usage patterns:
      * Async context manager (``async with MediaProcessorClient() as c: ...``).
      * Long-lived client owned by FastAPI (``await client.connect()`` at
        startup, ``await client.close()`` at shutdown).
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

    async def _extract(
        self,
        *,
        strategy: ExtractionStrategy,
        params_extra: Dict[str, float | int],
        video_object_key: str,
        file_url: str,
        file_name: str,
    ) -> ExtractFramesResponse:
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        params: Dict[str, str | float | int] = {"strategy": strategy.value}
        params.update(params_extra)

        response = await self.client.post(
            "/extract/",
            params=params,
            json={
                "video_object_key": video_object_key,
                "file_url": file_url,
                "file_name": file_name,
            },
        )
        response.raise_for_status()
        return ExtractFramesResponse(**response.json())

    async def extract_frames_hybrid(
        self,
        *,
        video_object_key: str,
        file_url: str,
        file_name: str,
        threshold: int,
    ) -> ExtractFramesResponse:
        return await self._extract(
            strategy=ExtractionStrategy.HYBRID,
            params_extra={"threshold": threshold},
            video_object_key=video_object_key,
            file_url=file_url,
            file_name=file_name,
        )

    async def health_check(self) -> Dict[str, str]:
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        response = await self.client.get("/")
        response.raise_for_status()
        return response.json()
