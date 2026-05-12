import asyncio

from fastapi import APIRouter, HTTPException, Request
from exports.schema.models import (
    EmbedImagesRequest,
    EmbedImagesResponse,
    EmbeddingResult,
)
from exports.utils.logger import get_logger

from ..clip_service import ClipEmbeddingEngine

router = APIRouter()
logger = get_logger()


def _get_engine(request: Request) -> ClipEmbeddingEngine:
    engine = getattr(request.app.state, "clip_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="CLIP engine not initialized")
    return engine


@router.post("/")
async def embed_images(request: Request, body: EmbedImagesRequest) -> EmbedImagesResponse:
    if not body.frames:
        raise HTTPException(status_code=422, detail="frames must not be empty")

    engine = _get_engine(request)
    frame_ids = [f.frame_id for f in body.frames]
    payloads = [f.frame_data for f in body.frames]
    logger.info("embed/images count=%s", len(frame_ids))

    try:
        pairs = await asyncio.to_thread(engine.embed_images, frame_ids, payloads)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    embeddings = [EmbeddingResult(frame_id=fid, embedding=vec) for fid, vec in pairs]
    return EmbedImagesResponse(embeddings=embeddings)
