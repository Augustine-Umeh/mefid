"""Unit tests for adaptive perceptual-hash thresholding."""

from sampling.adaptive_threshold import compute_adaptive_threshold


def test_none_when_too_few_diffs() -> None:
    # 6 diffs (7 frames) — below min_samples=7, no fixed fallback.
    assert compute_adaptive_threshold([1, 2, 3, 4, 5, 6]) is None


def test_computes_when_seven_or_more_diffs() -> None:
    diffs = [2, 4, 6, 8, 10, 12, 14]
    threshold = compute_adaptive_threshold(diffs, multiplier=1.5)
    assert threshold is not None
    assert threshold > 8.0


def test_minimum_threshold_floor() -> None:
    diffs = [0, 0, 0, 0, 0, 0, 0]
    assert compute_adaptive_threshold(diffs, multiplier=0.0) == 2.0
