"""Unit tests for time-bounded caption windowing."""

import pytest
from PIL import Image  # pyright: ignore[reportMissingImports]

from src.pipeline.types import Window
from src.pipeline.windowing import build_windows


def _frames_and_timestamps(count: int, *, start: float = 0.0, step: float = 1.0):
    frames = [Image.new("RGB", (1, 1), color=(i, i, i)) for i in range(count)]
    timestamps = [round(start + i * step, 3) for i in range(count)]
    return frames, timestamps


def test_build_windows_empty_input():
    assert build_windows([], []) == []


def test_build_windows_rejects_length_mismatch():
    frames, timestamps = _frames_and_timestamps(2)
    with pytest.raises(ValueError, match="length mismatch"):
        build_windows(frames, timestamps[:1])


def test_build_windows_rejects_non_positive_duration():
    frames, timestamps = _frames_and_timestamps(2)
    with pytest.raises(ValueError, match="window_duration must be positive"):
        build_windows(frames, timestamps, window_duration=0.0)


def test_build_windows_short_scene_is_single_window():
    frames, timestamps = _frames_and_timestamps(3, start=0.0, step=1.0)
    windows = build_windows(frames, timestamps, window_duration=4.0)

    assert len(windows) == 1
    assert windows[0].start_time == 0.0
    assert windows[0].end_time == 2.0
    assert len(windows[0].frames) == 3


def test_build_windows_long_scene_chunks_sequentially():
    frames, timestamps = _frames_and_timestamps(10, start=0.0, step=1.0)
    windows = build_windows(frames, timestamps, window_duration=4.0)

    assert len(windows) == 3
    assert windows[0].start_time == 0.0
    assert windows[0].end_time == 3.0
    assert len(windows[0].frames) == 4
    assert windows[1].start_time == 4.0
    assert windows[1].end_time == 7.0
    assert windows[2].start_time == 8.0
    assert windows[2].end_time == 9.0


def test_build_windows_merges_trailing_singleton_into_previous():
    frames, timestamps = _frames_and_timestamps(5, start=0.0, step=1.0)
    windows = build_windows(frames, timestamps, window_duration=2.0)

    assert len(windows) == 2
    assert windows[0].start_time == 0.0
    assert windows[0].end_time == 1.0
    assert len(windows[0].frames) == 2
    assert windows[1].start_time == 2.0
    assert windows[1].end_time == 4.0
    assert len(windows[1].frames) == 3


def test_build_windows_defers_leading_singleton_into_next_window():
    frames, timestamps = _frames_and_timestamps(3, start=0.0, step=2.0)
    windows = build_windows(frames, timestamps, window_duration=3.0)

    assert len(windows) == 1
    assert windows[0].start_time == 0.0
    assert windows[0].end_time == 4.0
    assert len(windows[0].frames) == 3


def test_build_windows_single_frame_scene_still_returns_window():
    frames, timestamps = _frames_and_timestamps(1)
    windows = build_windows(frames, timestamps, window_duration=4.0)

    assert len(windows) == 1
    assert isinstance(windows[0], Window)
    assert len(windows[0].frames) == 1
