"""Orchestrates the full captioning pipeline for one video file."""

from __future__ import annotations

from uuid import UUID

from exports.utils.logger import get_logger

from .pipeline.caption_generation import CaptionGenerator
from .pipeline.frame_extraction import extract_scene_frames
from .pipeline.merge_gate import merge_consecutive
from .pipeline.scene_detection import detect_scenes
from .pipeline.token_budget import compute_fps_and_resolution
from .pipeline.types import CaptionDraft
from .pipeline.windowing import build_windows

logger = get_logger()


def caption_video(
    video_path: str,
    media_id: UUID,
    generator: CaptionGenerator,
) -> list[CaptionDraft]:
    """Run scene detect → budget → extract → window → caption → merge for one video."""
    scenes = detect_scenes(video_path)
    drafts: list[CaptionDraft] = []

    for scene in scenes:
        fps, max_pixels = compute_fps_and_resolution(scene.duration)
        frames, timestamps = extract_scene_frames(video_path, scene, fps)
        windows = build_windows(frames, timestamps)

        logger.info(
            "Caption scene %.3f–%.3fs media_id=%s fps=%.3f max_pixels=%d windows=%d",
            scene.start_time,
            scene.end_time,
            media_id,
            fps,
            max_pixels,
            len(windows),
        )

        for window in windows:
            text = generator.caption_window(list(window.frames), max_pixels=max_pixels)
            drafts.append(
                CaptionDraft(
                    start_time=window.start_time,
                    end_time=window.end_time,
                    text=text,
                )
            )

    merged = merge_consecutive(drafts)
    logger.info(
        "Caption complete media_id=%s scenes=%d captions=%d merged=%d",
        media_id,
        len(scenes),
        len(drafts),
        len(merged),
    )
    return merged
