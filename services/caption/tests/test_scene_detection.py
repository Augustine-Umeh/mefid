"""Unit tests for scene boundary detection."""

from unittest.mock import MagicMock, patch

import pytest  # pyright: ignore[reportMissingImports]

from src.pipeline.scene_detection import _scenes_from_timecode_pairs, detect_scenes


class _FakeTimecode:
    def __init__(self, seconds: float) -> None:
        self._seconds = seconds

    def get_seconds(self) -> float:
        return self._seconds


def test_scenes_from_timecode_pairs_converts_and_rounds():
    raw = [
        (_FakeTimecode(0.0), _FakeTimecode(5.1234)),
        (_FakeTimecode(5.1234), _FakeTimecode(10.0)),
    ]
    scenes = _scenes_from_timecode_pairs(raw)  # pyright: ignore[reportArgumentType]

    assert len(scenes) == 2
    assert scenes[0].start_time == 0.0
    assert scenes[0].end_time == 5.123
    assert scenes[1].start_time == 5.123
    assert scenes[1].end_time == 10.0


def test_scenes_from_timecode_pairs_skips_invalid_intervals():
    raw = [
        (_FakeTimecode(0.0), _FakeTimecode(5.0)),
        (_FakeTimecode(8.0), _FakeTimecode(8.0)),
    ]
    scenes = _scenes_from_timecode_pairs(raw)  # pyright: ignore[reportArgumentType]

    assert len(scenes) == 1
    assert scenes[0].end_time == 5.0


@patch("src.pipeline.scene_detection.open_video")
@patch("src.pipeline.scene_detection.SceneManager")
@patch("src.pipeline.scene_detection.AdaptiveDetector")
def test_detect_scenes_uses_adaptive_detector_and_start_in_scene(
    mock_detector_cls,
    mock_manager_cls,
    mock_open_video,
):
    mock_video = MagicMock()
    mock_open_video.return_value = mock_video

    mock_manager = MagicMock()
    mock_manager_cls.return_value = mock_manager
    mock_manager.get_scene_list.return_value = [
        (_FakeTimecode(0.0), _FakeTimecode(12.5)),
    ]

    scenes = detect_scenes("/tmp/sample.mp4")

    mock_detector_cls.assert_called_once()
    mock_manager.add_detector.assert_called_once()
    mock_manager.detect_scenes.assert_called_once_with(mock_video)
    mock_manager.get_scene_list.assert_called_once_with(start_in_scene=True)
    assert len(scenes) == 1
    assert scenes[0].duration == 12.5


@patch("src.pipeline.scene_detection.open_video")
@patch("src.pipeline.scene_detection.SceneManager")
@patch("src.pipeline.scene_detection.AdaptiveDetector")
def test_detect_scenes_raises_when_no_valid_intervals(
    mock_detector_cls,
    mock_manager_cls,
    mock_open_video,
):
    mock_manager = MagicMock()
    mock_manager_cls.return_value = mock_manager
    mock_manager.get_scene_list.return_value = []

    with pytest.raises(RuntimeError, match="no valid intervals"):
        detect_scenes("/tmp/empty.mp4")
