import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from exports.db_clients.lifespan import lifespan as shared_lifespan
from exports.utils.logger import get_logger

from .routes.transcribe import router as transcribe_router
from .whisper_service import WhisperEngine

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Whisper on top of the shared MinIO/Supabase lifespan."""
    async with shared_lifespan(app):
        logger.info("Transcribe lifespan: loading Whisper...")
        engine = await asyncio.to_thread(WhisperEngine.load)
        app.state.whisper_engine = engine
        logger.info("Transcribe lifespan: Whisper ready.")
        try:
            yield
        finally:
            app.state.whisper_engine = None
            logger.info("Transcribe lifespan: Whisper released.")


app = FastAPI(
    title="Mefid Transcribe Service",
    description=(
        "Runs Whisper on uploaded videos and stores timestamped transcript "
        "chunks for multimodal search in Mefid."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(transcribe_router, prefix="/transcribe", tags=["Transcribe"])


@app.get("/")
async def root():
    return {"message": "Mefid Transcribe Service is running."}
