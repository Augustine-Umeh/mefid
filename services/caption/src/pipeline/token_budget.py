"""Step 2: choose extraction fps and max_pixels per scene under a VRAM token budget."""

from __future__ import annotations

from exports.schema.constants import (
    CAPTION_MAX_PIXELS_DEFAULT,
    CAPTION_MAX_PIXELS_FLOOR,
    CAPTION_MAX_TOTAL_TOKENS,
    CAPTION_MIN_FPS,
    CAPTION_TARGET_FPS,
    CAPTION_TOKENS_PER_FRAME_DEFAULT,
    CAPTION_RESOLUTION_TIERS,
)


def compute_fps_and_resolution(scene_duration: float) -> tuple[float, int]:
    """Return (fps, max_pixels) for a scene of the given duration in seconds.

    Step 1: Try ``CAPTION_TARGET_FPS`` at default resolution if the projected
    token count fits within ``CAPTION_MAX_TOTAL_TOKENS``.

    Step 2: Scale fps down while holding default resolution. Compute
    ``reduced_fps = budget / (duration * tokens_per_frame_default)`` and use
    it if it is still at or above ``CAPTION_MIN_FPS``.

    Step 3: Lock fps at ``CAPTION_MIN_FPS``, compute affordable tokens per
    frame, then pick the highest tier in ``CAPTION_RESOLUTION_TIERS`` that
    fits. Fall back to floor resolution if no tier fits.
    """
    if scene_duration <= 0:
        return CAPTION_TARGET_FPS, CAPTION_MAX_PIXELS_DEFAULT

    # --- STEP 1: Try target fps at default resolution ---
    num_frames = scene_duration * CAPTION_TARGET_FPS # number of frames to extract for the scene
    estimated_tokens = num_frames * CAPTION_TOKENS_PER_FRAME_DEFAULT # estimated number of tokens to generate for the scene
    if estimated_tokens <= CAPTION_MAX_TOTAL_TOKENS:
        return CAPTION_TARGET_FPS, CAPTION_MAX_PIXELS_DEFAULT

    # --- STEP 2: Scale fps towards the floor ---
    tokens_per_one_fps = scene_duration * CAPTION_TOKENS_PER_FRAME_DEFAULT # total tokens consumed across the scene duration at 1.0 FPS
    reduced_fps = CAPTION_MAX_TOTAL_TOKENS / tokens_per_one_fps # reduced fps to keep the number of tokens below the budget
    if reduced_fps >= CAPTION_MIN_FPS:
        return reduced_fps, CAPTION_MAX_PIXELS_DEFAULT
    
    # --- STEP 3: Scale resolution towards the floor using CAPTION_MIN_FPS ---
    num_frames = scene_duration * CAPTION_MIN_FPS # number of frames to extract for the scene
    estimated_tokens_per_frame = CAPTION_MAX_TOTAL_TOKENS / num_frames # estimated number of tokens per frame

    for tokens_per_frame, max_pixels in CAPTION_RESOLUTION_TIERS:
        if tokens_per_frame <= estimated_tokens_per_frame:
            return CAPTION_MIN_FPS, max_pixels


    # --- Last resort: Return the minimum fps and floor resolution ---
    return CAPTION_MIN_FPS, CAPTION_MAX_PIXELS_FLOOR