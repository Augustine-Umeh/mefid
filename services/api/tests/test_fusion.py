"""Unit tests for temporal signal fusion."""

from uuid import uuid4

from exports.fusion.classifier import QueryClass, classify_query
from exports.fusion.pipeline import (
    candidate_counts_for,
    fuse_and_rank,
    group_by_timestamp,
    normalize_scores_by_type,
    proximity_radius,
)
from exports.fusion.types import FusionHit, MomentGroup
from exports.schema.constants import VectorType


def _hit(
    *,
    vector_type: VectorType,
    raw_score: float,
    timestamp: float,
    media_id=None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> FusionHit:
    return FusionHit(
        faiss_index_id=1,
        vector_type=vector_type,
        raw_score=raw_score,
        media_id=media_id or uuid4(),
        timestamp=timestamp,
        start_time=start_time,
        end_time=end_time,
    )


def test_classify_visual_query() -> None:
    assert classify_query("show me the red car") == QueryClass.VISUAL


def test_classify_speech_query() -> None:
    assert classify_query('what did he say about "the project"') == QueryClass.SPEECH


def test_classify_ambiguous_query() -> None:
    assert (
        classify_query("what did the man in the red hat say")
        == QueryClass.AMBIGUOUS
    )


def test_candidate_counts_visual_prefers_image_and_caption() -> None:
    counts = candidate_counts_for(QueryClass.VISUAL)
    assert counts[VectorType.IMAGE] == 20
    assert counts[VectorType.CAPTION] == 20
    assert counts[VectorType.TEXT] == 5


def test_normalize_scores_by_type_scales_independently() -> None:
    hits = [
        _hit(vector_type=VectorType.IMAGE, raw_score=0.2, timestamp=1.0),
        _hit(vector_type=VectorType.IMAGE, raw_score=0.8, timestamp=2.0),
        _hit(vector_type=VectorType.TEXT, raw_score=0.9, timestamp=3.0),
        _hit(vector_type=VectorType.TEXT, raw_score=0.3, timestamp=4.0),
    ]
    normalize_scores_by_type(hits)
    assert hits[0].normalized_score == 0.0
    assert hits[1].normalized_score == 1.0
    assert hits[2].normalized_score == 1.0
    assert hits[3].normalized_score == 0.0


def test_proximity_radius_uses_caption_window_duration() -> None:
    hit = _hit(
        vector_type=VectorType.CAPTION,
        raw_score=0.5,
        timestamp=10.0,
        start_time=10.0,
        end_time=12.5,
    )
    assert proximity_radius(hit) == 2.5


def test_group_by_timestamp_merges_cross_signal_hits() -> None:
    media_id = uuid4()
    hits = [
        _hit(
            vector_type=VectorType.IMAGE,
            raw_score=0.7,
            timestamp=12.0,
            media_id=media_id,
        ),
        _hit(
            vector_type=VectorType.CAPTION,
            raw_score=0.6,
            timestamp=12.2,
            media_id=media_id,
            start_time=11.5,
            end_time=13.0,
        ),
    ]
    normalize_scores_by_type(hits)
    groups = group_by_timestamp(hits)
    assert len(groups) == 1
    assert len(groups[0].signal_types) == 2


def test_fuse_and_rank_boosts_intersections() -> None:
    media_id = uuid4()
    single = MomentGroup(
        media_id=media_id,
        anchor_time=1.0,
        hits=[
            _hit(
                vector_type=VectorType.IMAGE,
                raw_score=0.9,
                timestamp=1.0,
                media_id=media_id,
            )
        ],
    )
    intersection = MomentGroup(
        media_id=media_id,
        anchor_time=20.0,
        hits=[
            _hit(
                vector_type=VectorType.IMAGE,
                raw_score=0.7,
                timestamp=20.0,
                media_id=media_id,
            ),
            _hit(
                vector_type=VectorType.CAPTION,
                raw_score=0.7,
                timestamp=20.1,
                media_id=media_id,
                start_time=19.5,
                end_time=21.0,
            ),
        ],
    )
    for hit in single.hits + intersection.hits:
        hit.normalized_score = 0.7 if hit.timestamp >= 20.0 else 0.9

    ranked = fuse_and_rank([single, intersection], top_k=2)
    assert ranked[0] is intersection
    assert ranked[0].best_hit.fused_score == 0.7 * 1.3
