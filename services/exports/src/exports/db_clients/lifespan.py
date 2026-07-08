import asyncio
import importlib
from contextlib import asynccontextmanager

from fastapi import FastAPI

from exports.db_clients.db_client import create_minio, create_supabase
from exports.db_clients.minioDB import MinioDB
from exports.db_clients.supabaseDB import SupabaseDB
from exports.faiss_store import FaissIndexRegistry, faiss_index_dir
from exports.schema.constants import (
    CLIP_DIMENSION,
    EXPORTS_LIFESPAN_EMBEDDER,
    EXPORTS_LIFESPAN_FAISS,
    FAISS_INDEX_PATH,
)
from exports.utils.logger import get_logger

logger = get_logger()


def _should_init_faiss() -> bool:
    """Indexer service loads FAISS when EXPORTS_LIFESPAN_FAISS is set."""
    if not EXPORTS_LIFESPAN_FAISS:
        return False
    return bool(str(FAISS_INDEX_PATH or "").strip())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Shared startup for Mefid FastAPI apps.

    * **Default:** Supabase + MinIO on ``app.state``; optionally FAISS when
      ``EXPORTS_LIFESPAN_FAISS`` and ``FAISS_INDEX_PATH`` are set (indexer).
    * **Embedder:** set ``EXPORTS_LIFESPAN_EMBEDDER`` (see ``constants``); skips
      database clients and loads CLIP onto ``app.state.clip_engine``.
    """
    faiss_store = None
    clip_engine = None
    supabase_client = None
    minio_client = None
    
    try:
        if EXPORTS_LIFESPAN_EMBEDDER:
            app.state.supabase = None
            app.state.minio = None
            app.state.indexer_write_lock = None
            app.state.faiss = None
            app.state.clip_engine = None

            logger.info("Embedder lifespan: loading CLIP (Supabase/MinIO skipped)...")
            clip_module = importlib.import_module("src.clip_service")
            ClipEmbeddingEngine = clip_module.ClipEmbeddingEngine
            clip_engine = await asyncio.to_thread(ClipEmbeddingEngine.load)
            app.state.clip_engine = clip_engine
            logger.info("Embedder lifespan: CLIP loaded.")
        else:
            app.state.clip_engine = None

            logger.info("Connecting to Supabase...")
            supabase_client = await create_supabase()
            app.state.supabase = SupabaseDB(supabase_client)
            logger.info("✅ Supabase connected.")


            if _should_init_faiss():
                base_dir = faiss_index_dir()
                app.state.indexer_write_lock = asyncio.Lock()
                faiss_store = FaissIndexRegistry(base_dir)
                logger.info("Loading FAISS indexes from %s ...", faiss_store.base_dir)
                await asyncio.to_thread(faiss_store.load)
                app.state.faiss = faiss_store
                logger.info(
                    "✅ FAISS ready (dim=%s, ntotal=%s, path=%s)",
                    CLIP_DIMENSION,
                    faiss_store.ntotal,
                    faiss_store.base_dir,
                )
            else:
                app.state.indexer_write_lock = None
                app.state.faiss = None

                logger.info("Connecting to MinIO...")
                minio_client = await create_minio()
                app.state.minio = MinioDB(minio_client)
                logger.info("✅ MinIO connected.")


        yield  # FastAPI handles requests after this

    finally:
        if faiss_store is not None:
            try:
                logger.info("Saving FAISS index on shutdown...")
                await asyncio.to_thread(faiss_store.save)
                logger.info("✅ FAISS index saved.")
            except Exception:
                logger.exception("FAISS shutdown save failed")
            app.state.faiss = None
            app.state.indexer_write_lock = None

        if clip_engine is not None:
            app.state.clip_engine = None
            logger.info("Embedder lifespan: CLIP released.")

        if supabase_client is not None:
            app.state.supabase = None
            logger.info("Supabase lifespan: Supabase released.")

        if minio_client is not None:
            app.state.minio = None
            logger.info("MinIO lifespan: MinIO released.")

        logger.info("✅ Cleanup complete.")
