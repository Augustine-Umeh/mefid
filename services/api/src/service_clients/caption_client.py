from uuid import UUID
import httpx

from exports.schema.constants import CAPTION_SERVICE
from exports.schema.models import CaptionRequest, CaptionResponse
from exports.utils.logger import get_logger

logger = get_logger()


class CaptionClient:
    """Client for the Caption service."""

    def __init__(self, base_url: str = CAPTION_SERVICE):
        self.base_url = base_url
        self.client: httpx.AsyncClient | None = None

    async def connect(self) -> "CaptionClient":
        if self.client is None:
            self.client = httpx.AsyncClient(base_url=self.base_url, timeout=1800.0)
        return self

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def __aenter__(self):
        return await self.connect()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def caption(
        self,
        *,
        video_object_key: str,
        media_id: UUID,
        file_name: str,
    ) -> CaptionResponse:
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        payload = CaptionRequest(
            video_object_key=video_object_key,
            media_id=media_id,
            file_name=file_name,
        ).model_dump(mode="json")
        response = await self.client.post("/caption/", json=payload)
        response.raise_for_status()
        return CaptionResponse(**response.json())

    async def health_check(self) -> dict:
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        response = await self.client.get("/")
        response.raise_for_status()
        return response.json()
