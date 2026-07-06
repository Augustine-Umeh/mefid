"""Step 6: collapse near-duplicate consecutive captions (e.g. replay angles)."""

from __future__ import annotations

from exports.schema.constants import (
    CAPTION_MERGE_GATE_ENABLED,
    CAPTION_MERGE_PROXIMITY_SECONDS,
    CAPTION_MERGE_SIMILARITY_THRESHOLD,
)
from exports.utils.logger import get_logger

from .types import CaptionDraft

logger = get_logger()


def merge_consecutive(
    captions: list[CaptionDraft],
    *,
    enabled: bool = CAPTION_MERGE_GATE_ENABLED,
    similarity_threshold: float = CAPTION_MERGE_SIMILARITY_THRESHOLD,
    proximity_seconds: float = CAPTION_MERGE_PROXIMITY_SECONDS,
) -> list[CaptionDraft]:
    """Merge adjacent captions when embedding similarity and timing both match."""
    if not captions or not enabled:
        return captions

    # Similarity-based merge requires caption embeddings; deferred until caption
    # vectors are indexed. Return as-is for now so the pipeline can run end-to-end.
    logger.debug(
        "Merge gate enabled but not yet implemented "
        "(threshold=%.2f proximity=%.1fs); returning %d captions unchanged",
        similarity_threshold,
        proximity_seconds,
        len(captions),
    )
    return captions
