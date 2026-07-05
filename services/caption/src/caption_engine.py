"""Qwen2-VL wrapper for Mefid video captioning."""

from __future__ import annotations

from uuid import UUID

from exports.utils.logger import get_logger

from .caption_service import caption_video
from .pipeline.caption_generation import CaptionGenerator
from .pipeline.types import CaptionDraft

logger = get_logger()


class CaptionEngine:
    """Service-facing entry point for the captioning pipeline."""

    def __init__(self, generator: CaptionGenerator) -> None:
        self._generator = generator

    @classmethod
    def load(cls) -> CaptionEngine:
        logger.info("CaptionEngine.load: loading Qwen2-VL...")
        generator = CaptionGenerator.load()
        logger.info("CaptionEngine.load: ready.")
        return cls(generator)

    def caption_video(self, video_path: str, media_id: UUID) -> list[CaptionDraft]:
        return caption_video(video_path, media_id, self._generator)
