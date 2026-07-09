"""Score normalization, timestamp grouping, and fusion ranking."""

from __future__ import annotations

from exports.fusion.classifier import QueryClass
from exports.fusion.types import FusionHit, MomentGroup
from exports.schema.constants import (
    FUSION_FRAME_PROXIMITY,
    FUSION_INTERSECTION_MULTIPLIER_1,
    FUSION_INTERSECTION_MULTIPLIER_2,
    FUSION_INTERSECTION_MULTIPLIER_3,
    FUSION_PRIMARY_CANDIDATES,
    FUSION_SECONDARY_CANDIDATES,
    FUSION_TRANSCRIPT_PROXIMITY,
    VectorType,
)

_INTERSECTION_MULTIPLIERS: dict[int, float] = {
    1: FUSION_INTERSECTION_MULTIPLIER_1,
    2: FUSION_INTERSECTION_MULTIPLIER_2,
    3: FUSION_INTERSECTION_MULTIPLIER_3,
}

_CANDIDATE_COUNTS: dict[QueryClass, dict[VectorType, int]] = {
    QueryClass.VISUAL: {
        VectorType.IMAGE: FUSION_PRIMARY_CANDIDATES,
        VectorType.CAPTION: FUSION_PRIMARY_CANDIDATES,
        VectorType.TEXT: FUSION_SECONDARY_CANDIDATES,
    },
    QueryClass.SPEECH: {
        VectorType.IMAGE: FUSION_SECONDARY_CANDIDATES,
        VectorType.CAPTION: FUSION_PRIMARY_CANDIDATES,
        VectorType.TEXT: FUSION_PRIMARY_CANDIDATES,
    },
    QueryClass.AMBIGUOUS: {
        VectorType.IMAGE: FUSION_PRIMARY_CANDIDATES,
        VectorType.CAPTION: FUSION_PRIMARY_CANDIDATES,
        VectorType.TEXT: FUSION_PRIMARY_CANDIDATES,
    },
}


def candidate_counts_for(query_class: QueryClass) -> dict[VectorType, int]:
    return dict(_CANDIDATE_COUNTS[query_class])


def intersection_multiplier(signal_count: int) -> float:
    return _INTERSECTION_MULTIPLIERS.get(signal_count, FUSION_INTERSECTION_MULTIPLIER_1)


def normalize_scores_by_type(hits: list[FusionHit]) -> list[FusionHit]:
    """Min-max normalize scores within each vector type to [0, 1]."""
    if not hits:
        return hits

    by_type: dict[VectorType, list[FusionHit]] = {}
    for hit in hits:
        by_type.setdefault(hit.vector_type, []).append(hit)

    for type_hits in by_type.values():
        scores = [hit.raw_score for hit in type_hits]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            for hit in type_hits:
                hit.normalized_score = 1.0 if max_s > 0 else 0.0
            continue
        span = max_s - min_s
        for hit in type_hits:
            hit.normalized_score = (hit.raw_score - min_s) / span

    return hits


def proximity_radius(hit: FusionHit) -> float:
    if (
        hit.vector_type == VectorType.CAPTION
        and hit.start_time is not None
        and hit.end_time is not None
    ):
        return max(hit.end_time - hit.start_time, 0.0)
    if hit.vector_type == VectorType.TEXT:
        return FUSION_TRANSCRIPT_PROXIMITY
    return FUSION_FRAME_PROXIMITY


def group_by_timestamp(hits: list[FusionHit]) -> list[MomentGroup]:
    """Group hydrated hits by media and timestamp proximity."""
    sorted_hits = sorted(hits, key=lambda hit: (str(hit.media_id), hit.timestamp))
    groups: list[MomentGroup] = []

    for hit in sorted_hits:
        radius = proximity_radius(hit)
        matched = False
        for group in groups:
            if group.media_id != hit.media_id:
                continue
            if abs(hit.timestamp - group.anchor_time) <= radius:
                group.hits.append(hit)
                matched = True
                break
        if not matched:
            groups.append(
                MomentGroup(
                    media_id=hit.media_id,
                    anchor_time=hit.timestamp,
                    hits=[hit],
                )
            )

    return groups


def fuse_and_rank(groups: list[MomentGroup], top_k: int) -> list[MomentGroup]:
    """Score groups and return the top fused moments."""
    for group in groups:
        multiplier = intersection_multiplier(len(group.signal_types))
        group.best_hit.fused_score = group.best_normalized_score * multiplier
        for hit in group.hits:
            hit.fused_score = group.best_hit.fused_score

    ranked = sorted(
        groups,
        key=lambda group: group.best_hit.fused_score,
        reverse=True,
    )
    return ranked[:top_k]
