"""Step 3: extract uniform-fps frames within a single scene."""

from __future__ import annotations

from PIL import Image  # pyright: ignore[reportMissingImports]

from .types import Scene


def extract_scene_frames(
    video_path: str,
    scene: Scene,
    fps: float,
) -> tuple[list[Image.Image], list[float]]:
    """Extract frames at ``fps`` between ``scene.start_time`` and ``scene.end_time``."""
    raise NotImplementedError
