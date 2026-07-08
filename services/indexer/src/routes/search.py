from __future__ import annotations

import asyncio

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from exports.schema.constants import CLIP_DIMENSION, DEFAULT_TOP_K, VectorType
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
    """Nearest-neighbour search; returns FAISS ids, scores, and vector type."""
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

    faiss_registry = getattr(request.app.state, "faiss", None)
    lock: asyncio.Lock | None = getattr(request.app.state, "indexer_write_lock", None)
    if faiss_registry is None or lock is None:
        raise HTTPException(
            status_code=503,
            detail="FAISS index is not available (check FAISS_INDEX_PATH and startup logs).",
        )

    q = np.asarray(body.embedding, dtype=np.float32)

    async with lock:
        try:
            if body.vector_type is None:
                raw = await asyncio.to_thread(faiss_registry.search_all, q, top_k)
                hits = [
                    IndexerVectorHit(
                        faiss_index_id=faiss_index_id,
                        similarity_score=score,
                        vector_type=vector_type,
                    )
                    for vector_type, faiss_index_id, score in raw
                ]
            else:
                raw = await asyncio.to_thread(
                    faiss_registry.search, body.vector_type, q, top_k
                )
                hits = [
                    IndexerVectorHit(
                        faiss_index_id=faiss_index_id,
                        similarity_score=score,
                        vector_type=body.vector_type,
                    )
                    for faiss_index_id, score in raw
                ]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    logger.info(
        "vectors/search top_k=%s vector_type=%s returned=%s",
        top_k,
        body.vector_type,
        len(hits),
    )
    return SearchVectorsResponse(hits=hits)
