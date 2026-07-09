"""Search endpoints: text search wired; other modalities still stubs."""

import asyncio
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import httpx
from fastapi import APIRouter, HTTPException, Request

from exports.db_clients.supabaseDB import IdLike, SupabaseDB
from exports.fusion import (
    QueryClass,
    candidate_counts_for,
    classify_query,
    fuse_and_rank,
    group_by_timestamp,
    normalize_scores_by_type,
)
from exports.fusion.types import FusionHit, MomentGroup
from exports.schema.constants import DEFAULT_TOP_K, QueryType, VectorType
from exports.schema.models import (
    FrameRow,
    IndexerVectorHit,
    SearchQueryCreate,
    SearchRequest,
    TextSearchRequest,
)
from exports.utils.logger import get_logger

from src.schema.responses import SearchResponse, SearchResult, SignalMatch
from ..service_clients.embedder_client import EmbedderClient
from ..service_clients.indexer_client import IndexerClient

router = APIRouter()
logger = get_logger()

FaissHitKey = Tuple[int, VectorType]


def _effective_top_k(requested: Optional[int]) -> int:
    k = DEFAULT_TOP_K if requested is None else int(requested)
    if k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1")
    return min(k, DEFAULT_TOP_K)


def _nearest_frame(frames: List[FrameRow], timestamp: float) -> Optional[FrameRow]:
    if not frames:
        return None
    return min(frames, key=lambda frame: abs(frame.timestamp - timestamp))


def _signal_match_from_hit(hit: FusionHit, *, use_normalized: bool) -> SignalMatch:
    similarity = hit.normalized_score if use_normalized else hit.raw_score
    return SignalMatch(
        vector_type=hit.vector_type,
        similarity=similarity,
        timestamp=hit.timestamp,
        frame_id=hit.frame_id,
        frame_url=hit.frame_url,
        transcript_text=hit.transcript_text,
        caption_text=hit.caption_text,
        start_time=hit.start_time,
        end_time=hit.end_time,
    )


def _collect_group_text_fields(group: MomentGroup) -> Tuple[Optional[str], Optional[str]]:
    transcript_text: Optional[str] = None
    caption_text: Optional[str] = None
    for hit in group.hits:
        if hit.transcript_text:
            transcript_text = hit.transcript_text
        if hit.caption_text:
            caption_text = hit.caption_text
    return transcript_text, caption_text


def _collect_group_time_bounds(
    group: MomentGroup,
) -> Tuple[Optional[float], Optional[float]]:
    start_times = [hit.start_time for hit in group.hits if hit.start_time is not None]
    end_times = [hit.end_time for hit in group.hits if hit.end_time is not None]
    start_time = min(start_times) if start_times else None
    end_time = max(end_times) if end_times else None
    return start_time, end_time


def _search_result_from_group(group: MomentGroup) -> SearchResult:
    primary = group.best_hit
    signal_types = group.signal_types
    transcript_text, caption_text = _collect_group_text_fields(group)
    start_time, end_time = _collect_group_time_bounds(group)
    preview = primary
    if primary.frame_url is None:
        for hit in group.hits:
            if hit.frame_url:
                preview = hit
                break

    return SearchResult(
        media_id=primary.media_id,
        similarity=primary.fused_score,
        media_type=primary.media_type,
        file_name=primary.file_name,
        file_url=primary.file_url,
        signal_types=signal_types,
        signal_count=len(signal_types),
        signals=[_signal_match_from_hit(hit, use_normalized=True) for hit in group.hits],
        vector_type=primary.vector_type,
        frame_id=preview.frame_id,
        timestamp=preview.timestamp,
        frame_url=preview.frame_url,
        transcript_text=transcript_text,
        caption_text=caption_text,
        start_time=start_time,
        end_time=end_time,
    )


def _search_result_from_single_hit(hit: FusionHit) -> SearchResult:
    return SearchResult(
        media_id=hit.media_id,
        similarity=hit.raw_score,
        media_type=hit.media_type,
        file_name=hit.file_name,
        file_url=hit.file_url,
        signal_types=[hit.vector_type],
        signal_count=1,
        signals=[_signal_match_from_hit(hit, use_normalized=False)],
        vector_type=hit.vector_type,
        frame_id=hit.frame_id,
        timestamp=hit.timestamp,
        frame_url=hit.frame_url,
        transcript_text=hit.transcript_text,
        caption_text=hit.caption_text,
        start_time=hit.start_time,
        end_time=hit.end_time,
    )


async def _hydrate_fusion_hits(
    supabase: SupabaseDB,
    hits: List[IndexerVectorHit],
) -> List[FusionHit]:
    """Map indexer hits to hydrated fusion rows with media and timestamp metadata."""
    if not hits:
        return []

    score_by_key: Dict[FaissHitKey, float] = {
        (h.faiss_index_id, h.vector_type): h.similarity_score for h in hits
    }
    order: List[FaissHitKey] = [(h.faiss_index_id, h.vector_type) for h in hits]
    embedding_rows = await supabase.get_embeddings_by_faiss_index_ids(
        [h.faiss_index_id for h in hits]
    )
    by_key = {(e.faiss_index_id, e.vector_type): e for e in embedding_rows}

    frame_ids = list[IdLike](
        {e.frame_id for e in embedding_rows if e.frame_id is not None}
    )
    transcript_ids = list[IdLike](
        {e.transcript_id for e in embedding_rows if e.transcript_id is not None}
    )
    caption_ids = list[IdLike](
        {e.caption_id for e in embedding_rows if e.caption_id is not None}
    )

    frames = await supabase.get_frames_by_ids(frame_ids)
    by_frame = {f.id: f for f in frames}

    transcripts = await supabase.get_transcripts_by_ids(transcript_ids)
    by_transcript = {t.id: t for t in transcripts}

    captions = await supabase.get_captions_by_ids(caption_ids)
    by_caption = {c.id: c for c in captions}

    media_ids: set[UUID] = set()
    for frame in frames:
        media_ids.add(frame.media_id)
    for transcript in transcripts:
        media_ids.add(transcript.media_id)
    for caption in captions:
        media_ids.add(caption.media_id)

    frames_by_media: Dict[UUID, List[FrameRow]] = {}
    for media_id in media_ids:
        frames_by_media[media_id] = await supabase.get_frames_by_media_id(media_id)

    medias = await supabase.get_media_by_ids(list(media_ids))
    by_media = {m.id: m for m in medias}

    fusion_hits: List[FusionHit] = []
    for faiss_id, hit_vector_type in order:
        emb = by_key.get((faiss_id, hit_vector_type))
        if emb is None:
            continue

        raw_score = score_by_key[(faiss_id, hit_vector_type)]

        if emb.vector_type == VectorType.IMAGE and emb.frame_id is not None:
            frame = by_frame.get(emb.frame_id)
            if frame is None:
                continue
            media = by_media.get(frame.media_id)
            if media is None:
                continue
            fusion_hits.append(
                FusionHit(
                    faiss_index_id=faiss_id,
                    vector_type=VectorType.IMAGE,
                    raw_score=raw_score,
                    media_id=frame.media_id,
                    timestamp=frame.timestamp,
                    media_type=media.media_type,
                    file_name=media.file_name,
                    file_url=media.file_url,
                    frame_id=frame.id,
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
            fusion_hits.append(
                FusionHit(
                    faiss_index_id=faiss_id,
                    vector_type=VectorType.TEXT,
                    raw_score=raw_score,
                    media_id=transcript.media_id,
                    timestamp=midpoint,
                    media_type=media.media_type,
                    file_name=media.file_name,
                    file_url=media.file_url,
                    frame_id=preview.id if preview else None,
                    frame_url=preview.frame_url if preview else None,
                    transcript_text=transcript.text,
                    start_time=transcript.start_time,
                    end_time=transcript.end_time,
                )
            )
            continue

        if emb.vector_type == VectorType.CAPTION and emb.caption_id is not None:
            caption = by_caption.get(emb.caption_id)
            if caption is None:
                continue
            media = by_media.get(caption.media_id)
            if media is None:
                continue
            midpoint = (caption.start_time + caption.end_time) / 2.0
            preview = _nearest_frame(
                frames_by_media.get(caption.media_id, []),
                midpoint,
            )
            fusion_hits.append(
                FusionHit(
                    faiss_index_id=faiss_id,
                    vector_type=VectorType.CAPTION,
                    raw_score=raw_score,
                    media_id=caption.media_id,
                    timestamp=midpoint,
                    media_type=media.media_type,
                    file_name=media.file_name,
                    file_url=media.file_url,
                    frame_id=preview.id if preview else None,
                    frame_url=preview.frame_url if preview else None,
                    caption_text=caption.text,
                    start_time=caption.start_time,
                    end_time=caption.end_time,
                )
            )

    return fusion_hits


async def _retrieve_weighted_candidates(
    indexer: IndexerClient,
    embedding: List[float],
    query_class: QueryClass,
) -> List[IndexerVectorHit]:
    counts = candidate_counts_for(query_class)
    image_hits, caption_hits, transcript_hits = await asyncio.gather(
        indexer.search_vectors(
            embedding,
            counts[VectorType.IMAGE],
            vector_type=VectorType.IMAGE,
        ),
        indexer.search_vectors(
            embedding,
            counts[VectorType.CAPTION],
            vector_type=VectorType.CAPTION,
        ),
        indexer.search_vectors(
            embedding,
            counts[VectorType.TEXT],
            vector_type=VectorType.TEXT,
        ),
    )
    return [*image_hits, *caption_hits, *transcript_hits]


async def _fused_search_results(
    supabase: SupabaseDB,
    hits: List[IndexerVectorHit],
    top_k: int,
) -> List[SearchResult]:
    fusion_hits = await _hydrate_fusion_hits(supabase, hits)
    if not fusion_hits:
        return []

    normalize_scores_by_type(fusion_hits)
    groups = group_by_timestamp(fusion_hits)
    ranked_groups = fuse_and_rank(groups, top_k)
    return [_search_result_from_group(group) for group in ranked_groups]


async def _filtered_search_results(
    supabase: SupabaseDB,
    hits: List[IndexerVectorHit],
) -> List[SearchResult]:
    fusion_hits = await _hydrate_fusion_hits(supabase, hits)
    return [_search_result_from_single_hit(hit) for hit in fusion_hits]


@router.post("/text", response_model=SearchResponse)
async def search_by_text(request: Request, body: TextSearchRequest) -> SearchResponse:
    """Embed text with CLIP, search FAISS, then join hits to frames and media."""
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")

    top_k = _effective_top_k(body.top_k)
    vector_type_filter = body.vector_type
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
        if vector_type_filter is None:
            query_class = classify_query(text)
            hits = await _retrieve_weighted_candidates(indexer, embedding, query_class)
            results = await _fused_search_results(supabase, hits, top_k)
        else:
            hits = await indexer.search_vectors(
                embedding,
                top_k,
                vector_type=vector_type_filter,
            )
            results = await _filtered_search_results(supabase, hits)
    except httpx.HTTPError:
        logger.exception("Indexer search failed")
        raise HTTPException(
            status_code=502,
            detail="Indexer search failed.",
        ) from None

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
