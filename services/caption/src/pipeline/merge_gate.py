"""Step 6: collapse near-duplicate consecutive captions (e.g. replay angles)."""

from __future__ import annotations

import json
from math import sqrt
from typing import Callable, Optional
from urllib import error, request
from uuid import uuid4

from exports.schema.constants import (
    CAPTION_MERGE_GATE_ENABLED,
    EMBEDDER_SERVICE,
    CAPTION_MERGE_PROXIMITY_SECONDS,
    CAPTION_MERGE_SIMILARITY_THRESHOLD,
)
from exports.schema.models import EmbedTextBatchRequest, EmbedTextBatchResponse, EmbedTextItem
from exports.utils.logger import get_logger

from .types import CaptionDraft

logger = get_logger()

EmbeddingFetcher = Callable[[list[str]], Optional[list[list[float]]]]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for lval, rval in zip(left, right):
        dot += lval * rval
        left_norm += lval * lval
        right_norm += rval * rval
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (sqrt(left_norm) * sqrt(right_norm))


def _fetch_caption_embeddings(texts: list[str]) -> Optional[list[list[float]]]:
    if not texts:
        return []
    if not EMBEDDER_SERVICE:
        logger.warning("Merge gate skipped: EMBEDDER_SERVICE is not configured")
        return None

    payload = EmbedTextBatchRequest(
        texts=[EmbedTextItem(transcript_id=uuid4(), text=text) for text in texts]
    ).model_dump(mode="json")
    endpoint = f"{EMBEDDER_SERVICE.rstrip('/')}/embed/text/batch"
    http_request = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, ValueError):
        logger.exception("Merge gate skipped: failed to fetch caption embeddings")
        return None

    parsed = EmbedTextBatchResponse(**data)
    embeddings = [item.embedding for item in parsed.embeddings]
    if len(embeddings) != len(texts):
        logger.warning(
            "Merge gate skipped: embedding count mismatch (%d captions, %d vectors)",
            len(texts),
            len(embeddings),
        )
        return None
    return embeddings


def merge_consecutive(
    captions: list[CaptionDraft],
    *,
    enabled: bool = CAPTION_MERGE_GATE_ENABLED,
    similarity_threshold: float = CAPTION_MERGE_SIMILARITY_THRESHOLD,
    proximity_seconds: float = CAPTION_MERGE_PROXIMITY_SECONDS,
    embedding_fetcher: EmbeddingFetcher = _fetch_caption_embeddings,
) -> list[CaptionDraft]:
    """Merge adjacent captions when embedding similarity and timing both match."""
    if not captions or not enabled:
        return captions

    embeddings = embedding_fetcher([caption.text for caption in captions])
    if embeddings is None:
        logger.info(
            "Merge gate summary: in=%d out=%d merged_pairs=0 skipped_gap=0 "
            "skipped_similarity=0 embedder_fallback=true threshold=%.2f proximity=%.1fs",
            len(captions),
            len(captions),
            similarity_threshold,
            proximity_seconds,
        )
        return captions

    merged_pairs = 0
    skipped_gap = 0
    skipped_similarity = 0
    merged: list[CaptionDraft] = [
        CaptionDraft(
            start_time=captions[0].start_time,
            end_time=captions[0].end_time,
            text=captions[0].text,
        )
    ]
    merged_embeddings: list[list[float]] = [embeddings[0]]

    for caption, caption_embedding in zip(captions[1:], embeddings[1:]):
        previous = merged[-1]
        previous_embedding = merged_embeddings[-1]
        gap_seconds = caption.start_time - previous.end_time
        similarity = _cosine_similarity(previous_embedding, caption_embedding)

        if gap_seconds <= proximity_seconds and similarity >= similarity_threshold:
            keep_new_text = len(caption.text.strip()) > len(previous.text.strip())
            merged[-1] = CaptionDraft(
                start_time=previous.start_time,
                end_time=max(previous.end_time, caption.end_time),
                text=caption.text if keep_new_text else previous.text,
            )
            merged_embeddings[-1] = caption_embedding if keep_new_text else previous_embedding
            merged_pairs += 1
            continue

        if gap_seconds > proximity_seconds:
            skipped_gap += 1
        elif similarity < similarity_threshold:
            skipped_similarity += 1

        merged.append(
            CaptionDraft(
                start_time=caption.start_time,
                end_time=caption.end_time,
                text=caption.text,
            )
        )
        merged_embeddings.append(caption_embedding)

    logger.debug(
        "Merge gate summary: in=%d out=%d merged_pairs=%d skipped_gap=%d "
        "skipped_similarity=%d embedder_fallback=false threshold=%.2f proximity=%.1fs",
        len(captions),
        len(merged),
        merged_pairs,
        skipped_gap,
        skipped_similarity,
        similarity_threshold,
        proximity_seconds,
    )
    return merged
