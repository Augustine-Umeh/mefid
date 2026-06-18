from __future__ import annotations

import asyncio
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from exports.schema.constants import CLIP_DIMENSION
from exports.schema.models import (
    AddVectorsRequest,
    AddVectorsResponse,
    EmbeddingCreate,
)
from exports.utils.logger import get_logger

router = APIRouter()
logger = get_logger()


@router.post("/", response_model=AddVectorsResponse)
async def add_vectors(request: Request, body: AddVectorsRequest) -> AddVectorsResponse:
    """Add normalized vectors to FAISS and insert matching `embeddings` rows."""
    if not body.vectors:
        return AddVectorsResponse(count=0)

    for i, item in enumerate(body.vectors):
        if len(item.embedding) != CLIP_DIMENSION:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"vectors[{i}] has length {len(item.embedding)}, "
                    f"expected CLIP_DIMENSION={CLIP_DIMENSION}"
                ),
            )

    faiss_store = request.app.state.faiss
    supabase = request.app.state.supabase
    lock = request.app.state.indexer_write_lock

    rows: List[EmbeddingCreate] = []
    matrix = np.asarray(
        [item.embedding for item in body.vectors],
        dtype=np.float32,
    )

    async with lock:
        try:
            faiss_ids = await asyncio.to_thread(faiss_store.add, matrix)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        for item, faiss_index_id in zip(body.vectors, faiss_ids, strict=True):
            rows.append(
                EmbeddingCreate(
                    frame_id=item.frame_id,
                    transcript_id=item.transcript_id,
                    faiss_index_id=faiss_index_id,
                    vector_type=item.vector_type,
                )
            )

        try:
            await supabase.insert_embeddings_batch(rows)
        except Exception:
            logger.exception(
                "Supabase insert failed after FAISS add; "
                "FAISS may contain vectors without matching embeddings rows "
                "(media_id=%s, batch_size=%s)",
                body.media_id,
                len(rows),
            )
            raise HTTPException(
                status_code=502,
                detail="Failed to persist embedding metadata after indexing vectors.",
            ) from None

        try:
            await asyncio.to_thread(faiss_store.save)
        except Exception:
            logger.exception(
                "FAISS save failed after successful DB write (path=%s)",
                faiss_store.path,
            )
            raise HTTPException(
                status_code=500,
                detail="Vectors were indexed and saved to the database, "
                "but persisting the FAISS index to disk failed.",
            ) from None

    return AddVectorsResponse(count=len(rows))
