"""Unit tests for the caption pipeline orchestrator."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

from PIL import Image  # pyright: ignore[reportMissingImports]

from src.caption_service import caption_video
from src.pipeline.types import CaptionDraft, Scene, Window


def _window(start: float, end: float) -> Window:
    frame = Image.new("RGB", (1, 1))
    return Window(
        frames=(frame,),
        timestamps=(start,),
        start_time=start,
        end_time=end,
    )


@patch("src.caption_service.merge_consecutive")
@patch("src.caption_service.build_windows")
@patch("src.caption_service.extract_scene_frames")
@patch("src.caption_service.compute_fps_and_resolution")
@patch("src.caption_service.detect_scenes")
def test_caption_video_orchestrates_pipeline(
    mock_detect,
    mock_budget,
    mock_extract,
    mock_build_windows,
    mock_merge,
):
    media_id = uuid4()
    scene = Scene(start_time=0.0, end_time=5.0)
    mock_detect.return_value = [scene]
    mock_budget.return_value = (3.0, 1003520)
    frame = Image.new("RGB", (1, 1))
    mock_extract.return_value = ([frame], [0.0, 1.0])
    mock_build_windows.return_value = [_window(0.0, 1.0)]

    generator = MagicMock()
    generator.caption_window.return_value = "A player celebrates."

    draft = CaptionDraft(start_time=0.0, end_time=1.0, text="A player celebrates.")
    mock_merge.return_value = [draft]

    result = caption_video("video.mp4", media_id, generator)

    mock_detect.assert_called_once_with("video.mp4")
    mock_budget.assert_called_once_with(scene.duration)
    mock_extract.assert_called_once_with("video.mp4", scene, 3.0)
    mock_build_windows.assert_called_once()
    generator.caption_window.assert_called_once_with([frame], max_pixels=1003520)
    mock_merge.assert_called_once()
    assert result == [draft]
