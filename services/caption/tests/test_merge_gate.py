from src.pipeline.merge_gate import merge_consecutive
from src.pipeline.types import CaptionDraft


def _caption(start: float, end: float, text: str) -> CaptionDraft:
    return CaptionDraft(start_time=start, end_time=end, text=text)


def test_merge_consecutive_merges_neighbors_when_close_and_similar() -> None:
    captions = [
        _caption(0.0, 4.0, "A player shoots and scores."),
        _caption(4.1, 8.0, "Replay angle shows the same goal."),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],
    ]

    merged = merge_consecutive(
        captions,
        similarity_threshold=0.92,
        proximity_seconds=15.0,
        embedding_fetcher=lambda _: embeddings,
    )

    assert len(merged) == 1
    assert merged[0].start_time == 0.0
    assert merged[0].end_time == 8.0


def test_merge_consecutive_keeps_neighbors_when_similarity_is_low() -> None:
    captions = [
        _caption(0.0, 4.0, "A red team attack builds up."),
        _caption(4.1, 8.0, "A goalkeeper takes a goal kick."),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    merged = merge_consecutive(
        captions,
        similarity_threshold=0.92,
        proximity_seconds=15.0,
        embedding_fetcher=lambda _: embeddings,
    )

    assert len(merged) == 2


def test_merge_consecutive_keeps_neighbors_when_gap_is_too_large() -> None:
    captions = [
        _caption(0.0, 4.0, "A player dribbles down the wing."),
        _caption(25.0, 29.0, "A player dribbles down the wing in replay."),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.98, 0.01, 0.01],
    ]

    merged = merge_consecutive(
        captions,
        similarity_threshold=0.92,
        proximity_seconds=15.0,
        embedding_fetcher=lambda _: embeddings,
    )

    assert len(merged) == 2


def test_merge_consecutive_falls_back_when_embeddings_unavailable() -> None:
    captions = [
        _caption(0.0, 4.0, "A player prepares to shoot."),
        _caption(4.1, 8.0, "Replay of the same shot."),
    ]

    merged = merge_consecutive(
        captions,
        embedding_fetcher=lambda _: None,
    )

    assert merged == captions
