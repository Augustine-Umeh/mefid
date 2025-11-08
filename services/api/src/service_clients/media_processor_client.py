import httpx
from services.exports.src.schema.constants import MEDIA_PROCESSOR_SERVICE
from services.exports.src.schema.models import UploadRequest

class MediaProcessorClient:
    def __init__(self, base_url: str = MEDIA_PROCESSOR_SERVICE):
        self.base_url = base_url
        self.client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(base_url=self.base_url)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    # Send a request to medaia processor service
    async def process_media(self, media_data: UploadRequest) -> httpx.Response:
        """
        Send media processing request to Media Processor service.
        """
        if not self.client:
            raise RuntimeError("HTTP client is not initialized.")
        
        response = await self.client.post(
            "/process_media",
            json=media_data.model_dump()
        )
        response.raise_for_status()
        return response
