import cv2
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

from scenedetect import ContentDetector, detect

from exports.schema.constants import (
    FLOOR_INTERVAL,
    MIN_SAMPLE_GAP,
    PHASH_FALLBACK_THRESHOLD,
    PHASH_MULTIPLIER,
    PHASH_SIZE,
    PHASH_WARMUP_FRAMES,
)
from exports.utils.logger import get_logger

from .adaptive_threshold import compute_adaptive_threshold
from .deduplication import SampleGuard
from .phash_utils import compute_consecutive_diffs, compute_phash

logger = get_logger()


@dataclass
class SampledFrame:
    frame_index: int
    timestamp: float
    frame: np.ndarray
    phash: str
    trigger: str  # scene_boundary | phash_change | floor


def hybrid_sample_video(video_path: str, scene_threshold: int) -> List[SampledFrame]:
    """
    Sample frames using scene boundaries, perceptual-hash change, and a floor interval.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS for video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scene_list = detect(video_path, ContentDetector(threshold=scene_threshold))

    boundary_indices: set[int] = {0}
    for start_tc, _end_tc in scene_list:
        boundary_indices.add(start_tc.get_frames())

    guard = SampleGuard(min_gap_seconds=MIN_SAMPLE_GAP)
    sampled: List[SampledFrame] = []
    scenes_with_bounds = _build_scene_bounds(scene_list, total_frames)

    for scene_start_idx, scene_end_idx in scenes_with_bounds:
        sampled.extend(
            _process_scene(
                cap=cap,
                fps=fps,
                scene_start=scene_start_idx,
                scene_end=scene_end_idx,
                boundary_idxs=boundary_indices,
                guard=guard,
            )
        )

    cap.release()

    sampled.sort(key=lambda f: f.timestamp)
    _log_trigger_counts(sampled)
    return sampled


def _build_scene_bounds(scene_list, total_frames: int) -> List[Tuple[int, int]]:
    if not scene_list:
        return [(0, max(total_frames - 1, 0))]

    bounds: List[Tuple[int, int]] = []
    for start_tc, end_tc in scene_list:
        start_frame = start_tc.get_frames()
        end_frame = max(end_tc.get_frames() - 1, start_frame)
        bounds.append((start_frame, end_frame))
    return bounds


def _process_scene(
    cap: cv2.VideoCapture,
    fps: float,
    scene_start: int,
    scene_end: int,
    boundary_idxs: set[int],
    guard: SampleGuard,
) -> List[SampledFrame]:
    sampled: List[SampledFrame] = []
    floor_interval_frames = max(int(FLOOR_INTERVAL * fps), 1)
    last_floor_idx = scene_start

    warmup_end = min(scene_start + PHASH_WARMUP_FRAMES, scene_end + 1)
    warmup_hashes: List = []
    prev_hash = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, scene_start)

    for frame_idx in range(scene_start, scene_end + 1):
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        curr_hash = compute_phash(frame, hash_size=PHASH_SIZE)

        if frame_idx < warmup_end:
            warmup_hashes.append(curr_hash)

        threshold = compute_adaptive_threshold(
            diffs=compute_consecutive_diffs(warmup_hashes),
            multiplier=PHASH_MULTIPLIER,
            fallback=PHASH_FALLBACK_THRESHOLD,
        )

        trigger: Optional[str] = None

        if frame_idx in boundary_idxs:
            trigger = "scene_boundary"

        if trigger is None and prev_hash is not None:
            diff = curr_hash - prev_hash
            if diff > threshold:
                trigger = "phash_change"

        if trigger is None and (frame_idx - last_floor_idx) >= floor_interval_frames:
            trigger = "floor"
            last_floor_idx = frame_idx

        if trigger and guard.try_sample(timestamp):
            sampled.append(
                SampledFrame(
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    frame=frame.copy(),
                    phash=str(curr_hash),
                    trigger=trigger,
                )
            )

        prev_hash = curr_hash

    return sampled


def _log_trigger_counts(sampled: List[SampledFrame]) -> None:
    if not sampled:
        logger.info("Hybrid sampling: no frames extracted")
        return

    counts = Counter(sf.trigger for sf in sampled)
    logger.info(
        "Hybrid sampling triggers: scene_boundary=%d phash_change=%d floor=%d total=%d",
        counts.get("scene_boundary", 0),
        counts.get("phash_change", 0),
        counts.get("floor", 0),
        len(sampled),
    )
