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
    """Chunk frames into non-overlapping windows that do not cross scene bounds.

    Rules:
    - Scene span <= window_duration: one window for all frames.
    - Longer scenes: sequential fixed-duration windows with no overlap.
    - Fewer than 2 frames in a window: merge into an adjacent window rather
      than captioning a single frame in isolation.
    """
    if not frames:
        return []

    if len(frames) != len(timestamps):
        raise ValueError(
            f"frames and timestamps length mismatch: {len(frames)} vs {len(timestamps)}"
        )

    if window_duration <= 0:
        raise ValueError(f"window_duration must be positive, got {window_duration}")

    scene_start = timestamps[0]
    scene_end = timestamps[-1]

    if (scene_end - scene_start) <= window_duration:
        return [_make_window(frames, timestamps)]

    windows: list[Window] = []
    pending_frames: list[Image.Image] = []
    pending_timestamps: list[float] = []
    window_start = scene_start

    while window_start <= scene_end:
        window_end = window_start + window_duration
        window_frames = [
            frame
            for frame, ts in zip(frames, timestamps)
            if window_start <= ts < window_end
        ]
        window_ts = [ts for ts in timestamps if window_start <= ts < window_end]

        if pending_frames:
            window_frames = pending_frames + window_frames
            window_ts = pending_timestamps + window_ts
            pending_frames = []
            pending_timestamps = []

        if len(window_frames) >= 2:
            windows.append(_make_window(window_frames, window_ts))
        elif windows:
            prev = windows[-1]
            merged_frames = (*prev.frames, *window_frames)
            merged_ts = (*prev.timestamps, *window_ts)
            windows[-1] = _make_window(list(merged_frames), list(merged_ts))
        elif window_frames:
            pending_frames = window_frames
            pending_timestamps = window_ts

        window_start = window_end

    if pending_frames:
        if windows:
            prev = windows[-1]
            merged_frames = (*prev.frames, *pending_frames)
            merged_ts = (*prev.timestamps, *pending_timestamps)
            windows[-1] = _make_window(list(merged_frames), list(merged_ts))
        else:
            windows.append(_make_window(pending_frames, pending_timestamps))

    return windows


def _make_window(
    frames: list[Image.Image],
    timestamps: list[float],
) -> Window:
    return Window(
        frames=tuple(frames),
        timestamps=tuple(timestamps),
        start_time=timestamps[0],
        end_time=timestamps[-1],
    )
