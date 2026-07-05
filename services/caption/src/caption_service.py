"""Orchestrates the full captioning pipeline for one video file."""

from __future__ import annotations

from uuid import UUID

from .pipeline.caption_generation import CaptionGenerator
from .pipeline.frame_extraction import extract_scene_frames
from .pipeline.merge_gate import merge_consecutive
from .pipeline.scene_detection import detect_scenes
from .pipeline.token_budget import compute_fps_and_resolution
from .pipeline.types import CaptionDraft
from .pipeline.windowing import build_windows


def caption_video(
    video_path: str,
    media_id: UUID,
    generator: CaptionGenerator,
) -> list[CaptionDraft]:
    """Run scene detect → budget → extract → window → caption → merge for one video."""
    raise NotImplementedError
