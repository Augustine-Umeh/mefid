"""Search endpoints: text search wired; other modalities still stubs."""

from typing import Dict, List, Optional
from uuid import UUID

import httpx
from fastapi import APIRouter, HTTPException, Request

from exports.db_clients.supabaseDB import IdLike, SupabaseDB
from exports.schema.constants import DEFAULT_TOP_K, QueryType, VectorType
from exports.schema.models import (
    FrameRow,
    IndexerVectorHit,
    SearchQueryCreate,
    SearchRequest,
    TextSearchRequest,
)
from exports.utils.logger import get_logger

from src.schema.responses import SearchResponse, SearchResult
from ..service_clients.embedder_client import EmbedderClient
from ..service_clients.indexer_client import IndexerClient

router = APIRouter()
logger = get_logger()


def _effective_top_k(requested: Optional[int]) -> int:
    k = DEFAULT_TOP_K if requested is None else int(requested)
    if k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1")
    return min(k, DEFAULT_TOP_K)


def _nearest_frame(frames: List[FrameRow], timestamp: float) -> Optional[FrameRow]:
    if not frames:
        return None
    return min(frames, key=lambda frame: abs(frame.timestamp - timestamp))


async def _join_faiss_hits_to_results(
    supabase: SupabaseDB, hits: List[IndexerVectorHit]
) -> List[SearchResult]:
    """Map indexer hits (faiss id + score) to ``SearchResult`` rows via Supabase."""
    if not hits:
        return []

    score_by_faiss: Dict[int, float] = {
        h.faiss_index_id: h.similarity_score for h in hits
    }
    order = [h.faiss_index_id for h in hits]
    embedding_rows = await supabase.get_embeddings_by_faiss_index_ids(order)
    by_faiss = {e.faiss_index_id: e for e in embedding_rows}

    frame_ids = list[IdLike](
        {
            e.frame_id
            for e in embedding_rows
            if e.frame_id is not None
        }
    )
    transcript_ids = list[IdLike](
        {
            e.transcript_id
            for e in embedding_rows
            if e.transcript_id is not None
        }
    )

    frames = await supabase.get_frames_by_ids(frame_ids)
    by_frame = {f.id: f for f in frames}

    transcripts = await supabase.get_transcripts_by_ids(transcript_ids)
    by_transcript = {t.id: t for t in transcripts}

    media_ids: set[UUID] = set()
    for frame in frames:
        media_ids.add(frame.media_id)
    for transcript in transcripts:
        media_ids.add(transcript.media_id)

    frames_by_media: Dict[UUID, List[FrameRow]] = {}
    for media_id in media_ids:
        frames_by_media[media_id] = await supabase.get_frames_by_media_id(media_id)

    medias = await supabase.get_media_by_ids(list(media_ids))
    by_media = {m.id: m for m in medias}

    results: List[SearchResult] = []
    for faiss_id in order:
        emb = by_faiss.get(faiss_id)
        if emb is None:
            continue

        sim = score_by_faiss[faiss_id]

        if emb.vector_type == VectorType.IMAGE and emb.frame_id is not None:
            frame = by_frame.get(emb.frame_id)
            if frame is None:
                continue
            media = by_media.get(frame.media_id)
            if media is None:
                continue
            results.append(
                SearchResult(
                    media_id=frame.media_id,
                    similarity=sim,
                    media_type=media.media_type,
                    file_name=media.file_name,
                    file_url=media.file_url,
                    vector_type=VectorType.IMAGE,
                    frame_id=frame.id,
                    timestamp=frame.timestamp,
                    frame_url=frame.frame_url,
                )
            )
            continue

        if emb.vector_type == VectorType.TEXT and emb.transcript_id is not None:
            transcript = by_transcript.get(emb.transcript_id)
            if transcript is None:
                continue
            media = by_media.get(transcript.media_id)
            if media is None:
                continue
            midpoint = (transcript.start_time + transcript.end_time) / 2.0
            preview = _nearest_frame(
                frames_by_media.get(transcript.media_id, []),
                midpoint,
            )
            results.append(
                SearchResult(
                    media_id=transcript.media_id,
                    similarity=sim,
                    media_type=media.media_type,
                    file_name=media.file_name,
                    file_url=media.file_url,
                    vector_type=VectorType.TEXT,
                    frame_id=preview.id if preview else None,
                    timestamp=midpoint,
                    frame_url=preview.frame_url if preview else None,
                    transcript_text=transcript.text,
                    start_time=transcript.start_time,
                    end_time=transcript.end_time,
                )
            )

    return results


@router.post("/text", response_model=SearchResponse)
async def search_by_text(request: Request, body: TextSearchRequest) -> SearchResponse:
    """Embed text with CLIP, search FAISS, then join hits to frames and media."""
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")

    top_k = _effective_top_k(body.top_k)
    embedder: EmbedderClient = request.app.state.embedder
    indexer: IndexerClient = request.app.state.indexer
    supabase: SupabaseDB = request.app.state.supabase

    try:
        embedding = await embedder.embed_text(text)
    except httpx.HTTPError:
        logger.exception("Embedder text embed failed")
        raise HTTPException(
            status_code=502,
            detail="Embedding service failed for text query.",
        ) from None

    try:
        hits = await indexer.search_vectors(embedding, top_k)
    except httpx.HTTPError:
        logger.exception("Indexer search failed")
        raise HTTPException(
            status_code=502,
            detail="Indexer search failed.",
        ) from None

    results = await _join_faiss_hits_to_results(supabase, hits)
    try:
        await supabase.insert_search_query(
            SearchQueryCreate(query_type=QueryType.TEXT, query_text=text)
        )
    except Exception:
        logger.exception(
            "Failed to insert search_queries row (query_type=text, text_len=%s)",
            len(text),
        )
    return SearchResponse(query_type=QueryType.TEXT, top_k=top_k, results=results)


@router.post("/image")
async def search_by_image(query: SearchRequest) -> SearchResponse:
    """Search by uploaded image (placeholder)."""
    raise NotImplementedError("Image search is not wired yet.")


@router.post("/video")
async def search_by_video(query: SearchRequest) -> SearchResponse:
    """Search by uploaded video clip (placeholder)."""
    raise NotImplementedError("Video search is not wired yet.")


@router.post("/multimodal")
async def search_by_multimodal(query: SearchRequest) -> SearchResponse:
    """Search by combined text + image/video query (placeholder)."""
    raise NotImplementedError("Multimodal search is not wired yet.")
