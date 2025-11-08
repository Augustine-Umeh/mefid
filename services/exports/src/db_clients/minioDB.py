from urllib.parse import urljoin
from services.exports.src.schema.constants import MINIO_BUCKET_NAME, MINIO_ENDPOINT
from typing import BinaryIO
from miniopy_async import Minio


class MinioDB:
    def __init__(self, minio_client: Minio):
        self.client = minio_client

    async def upload_file(self, object_name: str, file_data: BinaryIO, content_type: str = "application/octet-stream") -> str:
        """Upload a file stream to MinIO and return the object URL."""
        result = await self.client.put_object(
            bucket_name=MINIO_BUCKET_NAME,
            object_name=object_name,
            data=file_data,
            length=-1,  # Unknown length for streaming
            part_size=10 * 1024 * 1024,  # 10MB part size
            content_type=content_type
        )

        # Construct and return accessible URL
        return urljoin(f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET_NAME}/", object_name)
    
    async def download_file(self, object_name: str) -> bytes:
        """Download a file from MinIO and return its raw bytes."""
        response = await self.client.get_object(
            bucket_name=MINIO_BUCKET_NAME,
            object_name=object_name
        )

        try:
            data = await response.read()  # Read the bytes fully
            return data
        finally:
            await response.release()  # Always release connection

    async def delete_file(self, object_name: str) -> None:
        """Delete a file from the bucket."""
        await self.client.remove_object(
            bucket_name=MINIO_BUCKET_NAME,
            object_name=object_name
        )