"""Unit tests for per-scene VRAM token budget selection."""

from exports.schema.constants import (
    CAPTION_MAX_PIXELS_256,
    CAPTION_MAX_PIXELS_DEFAULT,
    CAPTION_MAX_PIXELS_FLOOR,
    CAPTION_MAX_TOTAL_TOKENS,
    CAPTION_MIN_FPS,
    CAPTION_RESOLUTION_TIERS,
    CAPTION_TARGET_FPS,
    CAPTION_TOKENS_PER_FRAME_DEFAULT,
)

from src.pipeline.token_budget import compute_fps_and_resolution


def _expected_for_duration(scene_duration: float) -> tuple[float, int]:
    """Mirror compute_fps_and_resolution logic for test expectations."""
    if scene_duration <= 0:
        return CAPTION_TARGET_FPS, CAPTION_MAX_PIXELS_DEFAULT

    num_frames = scene_duration * CAPTION_TARGET_FPS
    estimated_tokens = num_frames * CAPTION_TOKENS_PER_FRAME_DEFAULT
    if estimated_tokens <= CAPTION_MAX_TOTAL_TOKENS:
        return CAPTION_TARGET_FPS, CAPTION_MAX_PIXELS_DEFAULT

    tokens_per_one_fps = scene_duration * CAPTION_TOKENS_PER_FRAME_DEFAULT
    reduced_fps = CAPTION_MAX_TOTAL_TOKENS / tokens_per_one_fps
    if reduced_fps >= CAPTION_MIN_FPS:
        return reduced_fps, CAPTION_MAX_PIXELS_DEFAULT

    num_frames = scene_duration * CAPTION_MIN_FPS
    affordable_tokens_per_frame = CAPTION_MAX_TOTAL_TOKENS / num_frames
    for tokens_per_frame, max_pixels in CAPTION_RESOLUTION_TIERS:
        if tokens_per_frame <= affordable_tokens_per_frame:
            return CAPTION_MIN_FPS, max_pixels

    return CAPTION_MIN_FPS, CAPTION_MAX_PIXELS_FLOOR


def test_short_scene_uses_target_fps():
    fps, max_pixels = compute_fps_and_resolution(2.0)
    expected_fps, expected_pixels = _expected_for_duration(2.0)

    assert fps == expected_fps == CAPTION_TARGET_FPS
    assert max_pixels == expected_pixels == CAPTION_MAX_PIXELS_DEFAULT


def test_zero_duration_returns_target_fps():
    fps, max_pixels = compute_fps_and_resolution(0.0)
    expected_fps, expected_pixels = _expected_for_duration(0.0)

    assert fps == expected_fps == CAPTION_TARGET_FPS
    assert max_pixels == expected_pixels == CAPTION_MAX_PIXELS_DEFAULT


def test_medium_scene_scales_fps_before_resolution():
    duration = 10.0
    fps, max_pixels = compute_fps_and_resolution(duration)
    expected_fps, expected_pixels = _expected_for_duration(duration)

    assert fps == expected_fps
    assert max_pixels == expected_pixels == CAPTION_MAX_PIXELS_DEFAULT
    assert CAPTION_MIN_FPS < fps < CAPTION_TARGET_FPS


def test_long_scene_keeps_min_fps_and_lowers_resolution():
    duration = 100.0
    fps, max_pixels = compute_fps_and_resolution(duration)
    expected_fps, expected_pixels = _expected_for_duration(duration)

    assert fps == expected_fps == CAPTION_MIN_FPS
    assert max_pixels == expected_pixels == CAPTION_MAX_PIXELS_256

    projected = duration * fps * CAPTION_TOKENS_PER_FRAME_DEFAULT
    assert projected > CAPTION_MAX_TOTAL_TOKENS


def test_very_long_scene_falls_back_to_floor_resolution():
    duration = 200.0
    fps, max_pixels = compute_fps_and_resolution(duration)
    expected_fps, expected_pixels = _expected_for_duration(duration)

    assert fps == expected_fps == CAPTION_MIN_FPS
    assert max_pixels == expected_pixels == CAPTION_MAX_PIXELS_FLOOR
