from __future__ import annotations

import asyncio

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from exports.schema.constants import CLIP_DIMENSION, DEFAULT_TOP_K
from exports.schema.models import (
    IndexerVectorHit,
    SearchVectorsRequest,
    SearchVectorsResponse,
)
from exports.utils.logger import get_logger

router = APIRouter()
logger = get_logger()


@router.post("/", response_model=SearchVectorsResponse)
async def search_vectors(
    request: Request, body: SearchVectorsRequest
) -> SearchVectorsResponse:
    """Nearest-neighbour search; returns FAISS ids and inner-product scores only."""
    if len(body.embedding) != CLIP_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"embedding must have length CLIP_DIMENSION={CLIP_DIMENSION}, "
                f"got {len(body.embedding)}"
            ),
        )
    if body.top_k < 1:
        raise HTTPException(
            status_code=400,
            detail="top_k must be at least 1",
        )
    top_k = min(body.top_k, DEFAULT_TOP_K)

    faiss_store = getattr(request.app.state, "faiss", None)
    lock = getattr(request.app.state, "indexer_write_lock", None)
    if faiss_store is None or lock is None:
        raise HTTPException(
            status_code=503,
            detail="FAISS index is not available (check FAISS_INDEX_PATH and startup logs).",
        )

    q = np.asarray(body.embedding, dtype=np.float32)

    async with lock:
        try:
            raw = await asyncio.to_thread(faiss_store.search, q, top_k)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    hits = [
        IndexerVectorHit(faiss_index_id=fid, similarity_score=score)
        for fid, score in raw
    ]
    logger.info("vectors/search top_k=%s returned=%s", top_k, len(hits))
    return SearchVectorsResponse(hits=hits)
