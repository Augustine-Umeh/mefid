import base64

import cv2

from exports.utils.logger import get_logger

from ..sampling.hybrid_sampler import hybrid_sample_video
from ..utils.video_utils import get_video_metadata

logger = get_logger()


def extract_frames_hybrid(video_path: str, threshold: int) -> dict:
    """
    Extract frames via hybrid sampling (scene + phash change + floor interval).

    Returns:
        {"duration": float, "frames": list of (timestamp, base64_jpeg, phash_hex)}
    """
    logger.info(f"Hybrid sampling {video_path} with scene threshold {threshold}")

    metadata = get_video_metadata(video_path)
    sampled = hybrid_sample_video(video_path, scene_threshold=threshold)

    if not sampled:
        logger.warning("Hybrid sampling produced no frames; extracting first frame")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Could not read any frames from video")
        _, buffer = cv2.imencode(".jpg", frame)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")
        return {
            "duration": metadata["duration"],
            "frames": [(0.0, frame_base64, None)],
        }

    frames = []
    for sf in sampled:
        _, buffer = cv2.imencode(".jpg", sf.frame)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")
        frames.append((sf.timestamp, frame_base64, sf.phash))

    logger.info(f"Hybrid extracted {len(frames)} frames")
    return {"duration": metadata["duration"], "frames": frames}
