from uuid import UUID
import httpx

from exports.schema.constants import TRANSCRIBE_SERVICE
from exports.schema.models import TranscribeRequest, TranscribeResponse
from exports.utils.logger import get_logger

logger = get_logger()


class TranscribeClient:
    """Client for the Transcribe service."""

    def __init__(self, base_url: str = TRANSCRIBE_SERVICE):
        self.base_url = base_url
        self.client: httpx.AsyncClient | None = None

    async def connect(self) -> "TranscribeClient":
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

    async def transcribe(
        self,
        *,
        video_object_key: str,
        media_id: UUID,
        file_name: str,
    ) -> TranscribeResponse:
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        payload = TranscribeRequest(
            video_object_key=video_object_key,
            media_id=media_id,
            file_name=file_name,
        ).model_dump(mode="json")
        response = await self.client.post("/transcribe/", json=payload)
        response.raise_for_status()
        return TranscribeResponse(**response.json())

    async def health_check(self) -> dict:
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        response = await self.client.get("/")
        response.raise_for_status()
        return response.json()

    async def get_ready(self) -> dict:
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")

        response = await self.client.get("/ready")
        response.raise_for_status()
        return response.json()
