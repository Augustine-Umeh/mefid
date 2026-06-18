import asyncio

from fastapi import APIRouter, HTTPException, Request
from exports.schema.models import (
    EmbedTextBatchRequest,
    EmbedTextBatchResponse,
    TextEmbeddingResult,
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


@router.post("/batch", response_model=EmbedTextBatchResponse)
async def embed_text_batch(
    request: Request, body: EmbedTextBatchRequest
) -> EmbedTextBatchResponse:
    if not body.texts:
        return EmbedTextBatchResponse(embeddings=[])

    for i, item in enumerate(body.texts):
        if not (item.text or "").strip():
            raise HTTPException(
                status_code=422,
                detail=f"texts[{i}].text must not be empty",
            )

    engine = _get_engine(request)
    logger.info("embed/text/batch count=%s", len(body.texts))
    vectors = await asyncio.to_thread(
        engine.embed_texts,
        [item.text for item in body.texts],
    )
    embeddings = [
        TextEmbeddingResult(transcript_id=item.transcript_id, embedding=vector)
        for item, vector in zip(body.texts, vectors, strict=True)
    ]
    return EmbedTextBatchResponse(embeddings=embeddings)
