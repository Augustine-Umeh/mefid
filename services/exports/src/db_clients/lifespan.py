from contextlib import asynccontextmanager
from fastapi import FastAPI
from miniopy_async import Minio
from services.exports.src.db_clients.db_client import create_supabase, create_minio
from services.exports.src.db_clients.supabaseDB import SupabaseDB
from services.exports.src.db_clients.minioDB import MinioDB
from services.exports.src.utils.logger import get_logger

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize SupabaseDB and MinioDB clients for any FastAPI service.
    Assigns to `app.state.db` and `app.state.minio`.
    """
    try:
        logger.info("Connecting to Supabase...")
        supabase_client = await create_supabase()
        app.state.db = SupabaseDB(supabase_client)
        logger.info("✅ Supabase connected.")

        logger.info("Connecting to MinIO...")
        minio_client = await create_minio()
        app.state.minio = MinioDB(minio_client)
        logger.info("✅ MinIO connected.")

        yield  # FastAPI handles requests after this

    finally:
        logger.info("Cleaning up DB and MinIO clients...")
        app.state.db = None
        app.state.minio = None
        logger.info("✅ Cleanup complete.")
