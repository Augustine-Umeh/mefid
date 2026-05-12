"""Embedder-only startup: load CLIP weights (no Supabase / MinIO)."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from exports.utils.logger import get_logger

from .clip_service import ClipEmbeddingEngine

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Embedder lifespan: loading CLIP…")
    engine = ClipEmbeddingEngine.load()
    app.state.clip_engine = engine
    logger.info("Embedder lifespan: CLIP loaded.")
    try:
        yield
    finally:
        app.state.clip_engine = None
        logger.info("Embedder lifespan: shutdown complete.")
