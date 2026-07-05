"""Step 5: Qwen2-VL inference for a single frame window."""

from __future__ import annotations

from PIL import Image  # pyright: ignore[reportMissingImports]

class CaptionGenerator:
    """Loads Qwen2-VL and generates text for multi-frame windows."""

    @classmethod
    def load(cls) -> CaptionGenerator:
        raise NotImplementedError

    def caption_window(self, frames: list[Image.Image], *, max_pixels: int) -> str:
        """Describe the chronological frame sequence in one concise paragraph."""
        raise NotImplementedError
