"""Shared datatypes for the captioning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image  # pyright: ignore[reportMissingImports]


@dataclass(frozen=True, slots=True)
class Scene:
    """A contiguous segment of video between PySceneDetect cuts."""

    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True, slots=True)
class Window:
    """Frames grouped for a single VLM inference call."""

    frames: tuple[Image.Image, ...]
    timestamps: tuple[float, ...]
    start_time: float
    end_time: float


@dataclass(slots=True)
class CaptionDraft:
    """One caption window before persistence."""

    start_time: float
    end_time: float
    text: str
