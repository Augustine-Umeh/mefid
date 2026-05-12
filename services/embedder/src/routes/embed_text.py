import asyncio

from fastapi import APIRouter, HTTPException, Request
from exports.schema.models import EmbedTextRequest, EmbedTextResponse
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
async def embed_text(request: Request, body: EmbedTextRequest) -> EmbedTextResponse:
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")

    engine = _get_engine(request)
    logger.info("embed/text chars=%s", len(text))
    embedding = await asyncio.to_thread(engine.embed_text, text)
    return EmbedTextResponse(embedding=embedding)
