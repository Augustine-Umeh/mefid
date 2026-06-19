"""Unit tests for adaptive perceptual-hash thresholding."""

from sampling.adaptive_threshold import compute_adaptive_threshold


def test_fallback_when_too_few_diffs() -> None:
    assert compute_adaptive_threshold([1, 2], fallback=10) == 10.0


def test_mean_plus_std_threshold() -> None:
    diffs = [2, 4, 6, 8, 10]
    threshold = compute_adaptive_threshold(diffs, multiplier=1.5, min_samples=5)
    assert threshold > 8.0


def test_minimum_threshold_floor() -> None:
    diffs = [0, 0, 0, 0, 0]
    assert compute_adaptive_threshold(diffs, multiplier=0.0, min_samples=5) == 2.0
