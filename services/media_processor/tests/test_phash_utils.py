"""Unit tests for perceptual hash utilities."""

import numpy as np

from sampling.phash_utils import compute_consecutive_diffs, compute_phash


def test_identical_frames_have_zero_diff() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    h1 = compute_phash(frame)
    h2 = compute_phash(frame)
    assert h1 - h2 == 0


def test_different_frames_have_nonzero_diff() -> None:
    frame_a = np.zeros((64, 64, 3), dtype=np.uint8)
    frame_b = np.full((64, 64, 3), 255, dtype=np.uint8)
    diff = compute_phash(frame_a) - compute_phash(frame_b)
    assert diff > 0


def test_consecutive_diffs_length() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    hashes = [compute_phash(frame) for _ in range(4)]
    diffs = compute_consecutive_diffs(hashes)
    assert len(diffs) == 3
    assert all(d == 0 for d in diffs)
