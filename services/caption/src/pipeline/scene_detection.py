"""Step 1: detect scene boundaries with PySceneDetect AdaptiveDetector."""

from __future__ import annotations

from scenedetect import AdaptiveDetector, SceneManager, open_video  # pyright: ignore[reportMissingImports]

from exports.schema.constants import (
    CAPTION_ADAPTIVE_SCENE_THRESHOLD,
    CAPTION_MIN_SCENE_LEN,
)
from exports.utils.logger import get_logger

from .types import Scene

logger = get_logger()


def _scenes_from_timecode_pairs(raw_scenes: list[tuple[object, object]]) -> list[Scene]:
    scenes: list[Scene] = []
    for start, end in raw_scenes:
        start_time = round(start.get_seconds(), 3)  # type: ignore[union-attr]
        end_time = round(end.get_seconds(), 3)  # type: ignore[union-attr]
        if end_time <= start_time:
            logger.warning(
                "Skipping invalid scene interval %.3f–%.3fs", start_time, end_time
            )
            continue
        scenes.append(Scene(start_time=start_time, end_time=end_time))
    return scenes


def detect_scenes(video_path: str) -> list[Scene]:
    """Return non-overlapping scene intervals for a raw video file."""
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        AdaptiveDetector(
            adaptive_threshold=CAPTION_ADAPTIVE_SCENE_THRESHOLD,
            min_scene_len=CAPTION_MIN_SCENE_LEN,
        )
    )

    logger.info("Detecting scenes in %s", video_path)
    scene_manager.detect_scenes(video)
    raw_scenes = scene_manager.get_scene_list(start_in_scene=True)
    scenes = _scenes_from_timecode_pairs(raw_scenes)

    if not scenes:
        raise RuntimeError(
            f"Scene detection produced no valid intervals for video: {video_path}"
        )

    logger.info("Detected %d scenes in %s", len(scenes), video_path)
    log_scene_size_frequency(scenes)
    return scenes

def log_scene_size_frequency(scenes: list[Scene]):
    """Log a map of scene size and their frequency."""
    scene_size_frequency: dict[float, int] = {}
    max_duration = 0.0
    for scene in scenes:
        duration = scene.end_time - scene.start_time
        if duration > max_duration:
            max_duration = duration
        scene_size_frequency[duration] = scene_size_frequency.get(duration, 0) + 1
    for duration, frequency in scene_size_frequency.items():
        logger.info("Scene size: %.3f, Frequency: %d", duration, frequency)
    logger.info("Max scene duration: %.3f", max_duration)