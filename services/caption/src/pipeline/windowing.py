"""Step 4: group scene frames into time-bounded caption windows."""

from __future__ import annotations

from PIL import Image  # pyright: ignore[reportMissingImports]

from exports.schema.constants import CAPTION_WINDOW_SECONDS

from .types import Window


def build_windows(
    frames: list[Image.Image],
    timestamps: list[float],
    *,
    window_duration: float = CAPTION_WINDOW_SECONDS,
) -> list[Window]:
    """Chunk frames into non-overlapping windows that do not cross scene bounds."""
    raise NotImplementedError
