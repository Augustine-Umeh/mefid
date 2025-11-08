from supabase import acreate_client, AsyncClient
from miniopy_async import Minio
from services.exports.src.schema.constants import (
    SUPABASE_DB_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_BUCKET_NAME,
    MINIO_USE_SSL
)

async def create_minio() -> Minio:
    minio_client: Minio = Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_USE_SSL
    )
    
    if not await minio_client.bucket_exists(MINIO_BUCKET_NAME):
        await minio_client.make_bucket(MINIO_BUCKET_NAME)
    return minio_client

async def create_supabase() -> AsyncClient:
    supabase_client: AsyncClient = await acreate_client(SUPABASE_DB_URL, SUPABASE_SERVICE_ROLE_KEY)
    return supabase_client