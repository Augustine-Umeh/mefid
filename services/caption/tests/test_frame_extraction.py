"""Unit tests for uniform-fps frame extraction within a scene."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image  # pyright: ignore[reportMissingImports]

from src.pipeline.frame_extraction import extract_scene_frames
from src.pipeline.types import Scene


def _fake_bgr_frame() -> np.ndarray:
    return np.full((240, 320, 3), 128, dtype=np.uint8)


def _mock_capture(*, video_fps: float = 30.0, pos_frames: int = 0) -> MagicMock:
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.grab.return_value = True
    mock_cap.read.return_value = (True, _fake_bgr_frame())

    def get_side_effect(prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return video_fps
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return pos_frames
        return 0.0

    mock_cap.get.side_effect = get_side_effect
    return mock_cap


@patch("src.pipeline.frame_extraction.cv2.VideoCapture")
def test_extract_scene_frames_samples_at_one_fps(mock_cap_cls):
    mock_cap = _mock_capture()
    mock_cap_cls.return_value = mock_cap

    scene = Scene(start_time=0.0, end_time=2.0)
    frames, timestamps = extract_scene_frames("fake.mp4", scene, fps=1.0)

    assert len(frames) == 2
    assert timestamps == [0.0, 1.0]
    assert all(isinstance(frame, Image.Image) for frame in frames)
    assert mock_cap.set.call_count == 1
    assert mock_cap.grab.call_count == 29


@patch("src.pipeline.frame_extraction.cv2.VideoCapture")
def test_extract_scene_frames_respects_scene_bounds(mock_cap_cls):
    mock_cap = _mock_capture()
    mock_cap_cls.return_value = mock_cap

    scene = Scene(start_time=5.0, end_time=6.5)
    frames, timestamps = extract_scene_frames("fake.mp4", scene, fps=1.0)

    assert len(frames) == 2
    assert timestamps == [5.0, 6.0]


@patch("src.pipeline.frame_extraction.cv2.VideoCapture")
def test_extract_scene_frames_higher_fps_more_samples(mock_cap_cls):
    mock_cap = _mock_capture()
    mock_cap_cls.return_value = mock_cap

    scene = Scene(start_time=0.0, end_time=1.0)
    frames, timestamps = extract_scene_frames("fake.mp4", scene, fps=3.0)

    assert len(frames) == 3
    assert timestamps == [0.0, 0.333, 0.667]


@patch("src.pipeline.frame_extraction.cv2.VideoCapture")
def test_extract_scene_frames_deduplicates_when_fps_exceeds_native(mock_cap_cls):
    mock_cap = _mock_capture(video_fps=10.0)
    mock_cap_cls.return_value = mock_cap

    scene = Scene(start_time=0.0, end_time=0.2)
    frames, timestamps = extract_scene_frames("fake.mp4", scene, fps=30.0)

    # 0.2s at 30fps => 6 candidate timestamps; at 10fps native only indices 0 and 1 remain
    assert len(frames) == 2
    assert timestamps == [0.0, 0.1]


@patch("src.pipeline.frame_extraction.cv2.VideoCapture")
def test_extract_scene_frames_uses_ceil_for_tight_scenes(mock_cap_cls):
    mock_cap = _mock_capture()
    mock_cap_cls.return_value = mock_cap

    scene = Scene(start_time=0.0, end_time=1.5)
    frames, timestamps = extract_scene_frames("fake.mp4", scene, fps=1.0)

    assert timestamps == [0.0, 1.0]


def test_extract_scene_frames_rejects_non_positive_fps():
    scene = Scene(start_time=0.0, end_time=1.0)
    with pytest.raises(ValueError, match="fps must be positive"):
        extract_scene_frames("fake.mp4", scene, fps=0.0)


@patch("src.pipeline.frame_extraction.cv2.VideoCapture")
def test_extract_scene_frames_raises_when_video_unopenable(mock_cap_cls):
    mock_cap = MagicMock()
    mock_cap_cls.return_value = mock_cap
    mock_cap.isOpened.return_value = False

    scene = Scene(start_time=0.0, end_time=1.0)
    with pytest.raises(RuntimeError, match="Cannot open video"):
        extract_scene_frames("missing.mp4", scene, fps=1.0)

    mock_cap.release.assert_called_once()


@patch("src.pipeline.frame_extraction.cv2.VideoCapture")
def test_extract_scene_frames_skips_failed_reads(mock_cap_cls):
    mock_cap = _mock_capture()
    mock_cap.read.side_effect = [(True, _fake_bgr_frame()), (False, None)]
    mock_cap_cls.return_value = mock_cap

    scene = Scene(start_time=0.0, end_time=2.0)
    frames, timestamps = extract_scene_frames("fake.mp4", scene, fps=1.0)

    assert len(frames) == 1
    assert timestamps == [0.0]


@patch("src.pipeline.frame_extraction.cv2.VideoCapture")
def test_extract_scene_frames_resets_when_seek_overshoots(mock_cap_cls):
    mock_cap = _mock_capture(pos_frames=5)
    mock_cap_cls.return_value = mock_cap

    scene = Scene(start_time=0.0, end_time=1.0)
    frames, timestamps = extract_scene_frames("fake.mp4", scene, fps=1.0)

    assert timestamps == [0.0]
    assert mock_cap.set.call_count == 2
