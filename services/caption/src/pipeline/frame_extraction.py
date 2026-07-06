"""Step 3: extract uniform-fps frames within a single scene."""

from __future__ import annotations

import math

import cv2  # pyright: ignore[reportMissingImports]
from PIL import Image  # pyright: ignore[reportMissingImports]

from exports.utils.logger import get_logger

from .types import Scene

logger = get_logger()


def extract_scene_frames(
    video_path: str,
    scene: Scene,
    fps: float,
) -> tuple[list[Image.Image], list[float]]:
    """Extract frames at ``fps`` between ``scene.start_time`` and ``scene.end_time``.

    Samples uniformly in wall-clock time: ``start_time``, ``start_time + 1/fps``, ...
    up to but not including ``end_time``. Uses math.ceil to avoid boundary truncation.
    Deduplicates target frame indices upfront to handle cases where extraction fps
    exceeds the video's native fps, preventing duplicate frames and broken
    timestamp-to-frame mappings.

    Timestamps are rounded to milliseconds (3 decimal places).
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0

        duration = scene.duration

        # Use ceil to avoid truncating the tail end of tight scenes
        num_frames = math.ceil(duration * fps)
        target_timestamps = [
            scene.start_time + i * (1.0 / fps) for i in range(num_frames)
        ]

        # Keep only timestamps that fall strictly within the scene
        target_timestamps = [t for t in target_timestamps if t < scene.end_time]
        target_indices = [int(t * video_fps) for t in target_timestamps]

        if not target_indices:
            return [], []

        # Deduplicate frame indices upfront to handle extraction fps > video native fps.
        # Preserves the first timestamp mapped to each unique index.
        seen: dict[int, float] = {}
        for t, idx in zip(target_timestamps, target_indices):
            if idx not in seen:
                seen[idx] = t

        target_indices = list(seen.keys())
        target_timestamps = list(seen.values())

        frames: list[Image.Image] = []
        timestamps: list[float] = []

        # Seek once to the starting frame of the scene
        current_idx = target_indices[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)

        # Sync with reality in case the seek landed on a nearby keyframe instead
        # of the exact requested index (common with variable framerate containers)
        actual_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if actual_idx >= 0:
            current_idx = actual_idx

        # If the seek overshot the first target, re-seek from the beginning
        # and advance sequentially to avoid reading from the wrong position
        if current_idx > target_indices[0]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_idx = 0

        for t, target_idx in zip(target_timestamps, target_indices):
            # Advance sequentially using grab() which skips decoding for speed
            while current_idx < target_idx:
                cap.grab()
                current_idx += 1

            ret, bgr = cap.read()
            current_idx += 1

            if ret:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
                timestamps.append(round(t, 3))
            else:
                logger.warning(
                    "Failed to read frame at t=%.3fs (index=%d) from %s",
                    t,
                    target_idx,
                    video_path,
                )

        return frames, timestamps

    finally:
        cap.release()
