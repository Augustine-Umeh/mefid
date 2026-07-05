"""Step 6: collapse near-duplicate consecutive captions (e.g. replay angles)."""

from __future__ import annotations

from exports.schema.constants import (
    CAPTION_MERGE_GATE_ENABLED,
    CAPTION_MERGE_PROXIMITY_SECONDS,
    CAPTION_MERGE_SIMILARITY_THRESHOLD,
)

from .types import CaptionDraft


def merge_consecutive(
    captions: list[CaptionDraft],
    *,
    enabled: bool = CAPTION_MERGE_GATE_ENABLED,
    similarity_threshold: float = CAPTION_MERGE_SIMILARITY_THRESHOLD,
    proximity_seconds: float = CAPTION_MERGE_PROXIMITY_SECONDS,
) -> list[CaptionDraft]:
    """Merge adjacent captions when embedding similarity and timing both match."""
    raise NotImplementedError
