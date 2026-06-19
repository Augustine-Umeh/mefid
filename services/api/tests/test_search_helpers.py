"""Unit tests for search result assembly helpers."""

from uuid import uuid4

from exports.schema.models import FrameRow
from datetime import datetime, timezone

from src.routes.search_route import _faiss_top_k, _nearest_frame


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


def test_faiss_top_k_over_fetches_when_vector_type_filter_set() -> None:
    from exports.schema.constants import DEFAULT_TOP_K, FILTERED_SEARCH_MAX_K, VectorType

    assert _faiss_top_k(10, None) == 10
    assert _faiss_top_k(10, VectorType.IMAGE) == FILTERED_SEARCH_MAX_K
