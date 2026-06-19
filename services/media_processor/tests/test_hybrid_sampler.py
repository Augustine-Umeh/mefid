"""Unit tests for hybrid sampler orchestration (mocked video I/O)."""

from unittest.mock import MagicMock, patch

import numpy as np

from sampling.deduplication import SampleGuard
from sampling.hybrid_sampler import SampledFrame, _process_scene


class _FakeTimecode:
    def __init__(self, frame: int):
        self._frame = frame

    def get_frames(self) -> int:
        return self._frame


def _make_frame(value: int) -> np.ndarray:
    return np.full((32, 32, 3), value, dtype=np.uint8)


def test_process_scene_samples_boundary_and_floor() -> None:
    frames = [_make_frame(i * 40) for i in range(6)]
    read_results = [(True, frame) for frame in frames] + [(False, None)]

    cap = MagicMock()
    cap.read.side_effect = read_results

    guard = SampleGuard(min_gap_seconds=0.0)
    sampled = _process_scene(
        cap=cap,
        fps=1.0,
        scene_start=0,
        scene_end=5,
        boundary_idxs={0},
        guard=guard,
    )

    triggers = {sf.trigger for sf in sampled}
    assert "scene_boundary" in triggers
    assert len(sampled) >= 2


def test_sample_guard_dedupes_close_timestamps() -> None:
    frames = [_make_frame(0)] * 4
    read_results = [(True, frame) for frame in frames] + [(False, None)]

    cap = MagicMock()
    cap.read.side_effect = read_results

    guard = SampleGuard(min_gap_seconds=10.0)
    sampled = _process_scene(
        cap=cap,
        fps=1.0,
        scene_start=0,
        scene_end=3,
        boundary_idxs={0, 1, 2, 3},
        guard=guard,
    )

    assert len(sampled) == 1


@patch("sampling.hybrid_sampler.detect")
@patch("sampling.hybrid_sampler.cv2.VideoCapture")
def test_hybrid_sample_video_no_scenes_falls_back_to_whole_video(
    mock_capture_cls: MagicMock,
    mock_detect: MagicMock,
) -> None:
    from sampling.hybrid_sampler import hybrid_sample_video

    mock_detect.return_value = []

    frames = [_make_frame(i * 50) for i in range(4)]
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {5: 1.0, 7: len(frames)}.get(prop, 0)
    cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
    mock_capture_cls.return_value = cap

    with patch("sampling.hybrid_sampler._process_scene") as mock_process:
        mock_process.return_value = [
            SampledFrame(
                frame_index=0,
                timestamp=0.0,
                frame=frames[0],
                phash="abc",
                trigger="scene_boundary",
            )
        ]
        result = hybrid_sample_video("fake.mp4", scene_threshold=10)

    assert len(result) == 1
    mock_process.assert_called_once()
    args = mock_process.call_args.kwargs
    assert args["scene_start"] == 0
    assert args["scene_end"] == 3
