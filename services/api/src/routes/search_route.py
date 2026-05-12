"""Search endpoints: text search wired; other modalities still stubs."""

from typing import Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request

from exports.db_clients.supabaseDB import SupabaseDB
from exports.schema.constants import DEFAULT_TOP_K, QueryType, TEXT_SEARCH_MIN_SIMILARITY
from exports.schema.models import (
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
    rows = await supabase.get_embeddings_by_faiss_index_ids(order)
    by_faiss = {e.faiss_index_id: e for e in rows}

    frame_ids = list({by_faiss[fid].frame_id for fid in order if fid in by_faiss})
    frames = await supabase.get_frames_by_ids(frame_ids)
    by_frame = {f.id: f for f in frames}

    media_ids = list({f.media_id for f in frames})
    medias = await supabase.get_media_by_ids(media_ids)
    by_media = {m.id: m for m in medias}

    results: List[SearchResult] = []
    for faiss_id in order:
        emb = by_faiss.get(faiss_id)
        if emb is None:
            continue
        frame = by_frame.get(emb.frame_id)
        if frame is None:
            continue
        media = by_media.get(frame.media_id)
        if media is None:
            continue
        sim = score_by_faiss[faiss_id]
        if sim < TEXT_SEARCH_MIN_SIMILARITY:
            continue
        results.append(
            SearchResult(
                frame_id=frame.id,
                media_id=frame.media_id,
                timestamp=frame.timestamp,
                frame_url=frame.frame_url,
                similarity=sim,
                media_type=media.media_type,
                file_name=media.file_name,
                file_url=media.file_url,
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
