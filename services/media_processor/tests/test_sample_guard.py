"""Unit tests for SampleGuard min-gap enforcement."""

from sampling.deduplication import SampleGuard


def test_allows_first_sample() -> None:
    guard = SampleGuard(min_gap_seconds=1.0)
    assert guard.try_sample(0.0) is True


def test_blocks_sample_within_gap() -> None:
    guard = SampleGuard(min_gap_seconds=1.0)
    assert guard.try_sample(0.0) is True
    assert guard.try_sample(0.5) is False


def test_allows_sample_after_gap() -> None:
    guard = SampleGuard(min_gap_seconds=1.0)
    assert guard.try_sample(0.0) is True
    assert guard.try_sample(1.0) is True
