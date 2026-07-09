"""Unit tests for search result assembly helpers."""

from datetime import datetime, timezone
from uuid import uuid4

from exports.fusion.types import FusionHit, MomentGroup
from exports.schema.constants import MediaType, VectorType
from exports.schema.models import FrameRow

from src.routes.search_route import (
    _effective_top_k,
    _nearest_frame,
    _search_result_from_group,
    _search_result_from_single_hit,
)


def _frame(timestamp: float) -> FrameRow:
    now = datetime.now(timezone.utc)
    return FrameRow(
        id=uuid4(),
        media_id=uuid4(),
        timestamp=timestamp,
        frame_url="http://example/frame.jpg",
        sequence_number=0,
        created_at=now,
    )


def test_nearest_frame_picks_closest_timestamp() -> None:
    frames = [_frame(1.0), _frame(5.0), _frame(9.0)]
    nearest = _nearest_frame(frames, 4.8)
    assert nearest is not None
    assert nearest.timestamp == 5.0


def test_nearest_frame_returns_none_for_empty_list() -> None:
    assert _nearest_frame([], 1.0) is None


def test_effective_top_k_clamps_to_default_max() -> None:
    from exports.schema.constants import DEFAULT_TOP_K

    assert _effective_top_k(None) == DEFAULT_TOP_K
    assert _effective_top_k(10) == 10
    assert _effective_top_k(DEFAULT_TOP_K + 50) == DEFAULT_TOP_K


def test_search_result_from_single_hit_includes_signal_provenance() -> None:
    media_id = uuid4()
    hit = FusionHit(
        faiss_index_id=1,
        vector_type=VectorType.TEXT,
        raw_score=0.86,
        media_id=media_id,
        timestamp=12.3,
        media_type=MediaType.VIDEO,
        file_name="clip.mp4",
        file_url="http://example/clip.mp4",
        transcript_text="hello world",
        start_time=11.8,
        end_time=13.2,
    )
    result = _search_result_from_single_hit(hit)
    assert result.signal_count == 1
    assert result.signal_types == [VectorType.TEXT]
    assert len(result.signals) == 1
    assert result.signals[0].vector_type == VectorType.TEXT
    assert result.transcript_text == "hello world"


def test_search_result_from_group_includes_all_signals() -> None:
    media_id = uuid4()
    group = MomentGroup(
        media_id=media_id,
        anchor_time=12.0,
        hits=[
            FusionHit(
                faiss_index_id=1,
                vector_type=VectorType.IMAGE,
                raw_score=0.4,
                normalized_score=0.6,
                fused_score=0.78,
                media_id=media_id,
                timestamp=12.0,
                media_type=MediaType.VIDEO,
                file_name="clip.mp4",
                file_url="http://example/clip.mp4",
                frame_url="http://example/frame.jpg",
            ),
            FusionHit(
                faiss_index_id=2,
                vector_type=VectorType.CAPTION,
                raw_score=0.5,
                normalized_score=0.5,
                fused_score=0.78,
                media_id=media_id,
                timestamp=12.1,
                media_type=MediaType.VIDEO,
                file_name="clip.mp4",
                file_url="http://example/clip.mp4",
                caption_text="A person waves",
                start_time=11.5,
                end_time=13.0,
            ),
        ],
    )
    result = _search_result_from_group(group)
    assert result.signal_count == 2
    assert set(result.signal_types) == {VectorType.IMAGE, VectorType.CAPTION}
    assert len(result.signals) == 2
    assert result.caption_text == "A person waves"
    assert result.similarity == 0.78
