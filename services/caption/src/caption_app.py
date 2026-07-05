import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from exports.db_clients.lifespan import lifespan as shared_lifespan
from exports.utils.logger import get_logger

from .caption_engine import CaptionEngine
from .routes.caption import router as caption_router

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Qwen2-VL on top of the shared MinIO/Supabase lifespan."""
    async with shared_lifespan(app):
        logger.info("Caption lifespan: loading caption engine...")
        engine = await asyncio.to_thread(CaptionEngine.load)
        app.state.caption_engine = engine
        logger.info("Caption lifespan: caption engine ready.")
        try:
            yield
        finally:
            app.state.caption_engine = None
            logger.info("Caption lifespan: caption engine released.")


app = FastAPI(
    title="Mefid Caption Service",
    description=(
        "Generates visual captions from raw video using Qwen2-VL and stores "
        "time-bounded caption windows for multimodal search in Mefid."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(caption_router, prefix="/caption", tags=["Caption"])


@app.get("/")
async def root():
    return {"message": "Mefid Caption Service is running."}
