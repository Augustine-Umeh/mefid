import cv2
import base64
from typing import List, Tuple
from scenedetect import detect, ContentDetector
from exports.utils.logger import get_logger
from ..utils.video_utils import get_video_metadata

logger = get_logger()


def extract_frames_scene_detect(
    video_path: str,
    threshold: int
) -> dict:
    """
    Extract frames at scene boundaries using PySceneDetect.
    
    Args:
        video_path: Path to video file
        threshold: Scene detection threshold (lower = more sensitive)
        
    Returns:
        List of (timestamp, base64_encoded_frame) tuples
    """
    logger.info(f"Detecting scenes in {video_path} with threshold {threshold}")
    
    # Get video metadata
    metadata = get_video_metadata(video_path)
    duration = metadata["duration"]
    
    # Detect scenes
    scene_list = detect(video_path, ContentDetector(threshold=threshold))
    
    if not scene_list:
        logger.warning("No scenes detected, extracting first frame only")
        # Fall back to a single frame so we still return something searchable.
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Could not read any frames from video")

        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return {
            "duration": duration,
            "frames": [(0.0, frame_base64)],
        }
    
    logger.info(f"Detected {len(scene_list)} scenes")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for scene in scene_list:
        # Get middle of scene
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        middle_time = (start_time + end_time) / 2
        
        # Seek to middle of scene
        cap.set(cv2.CAP_PROP_POS_MSEC, middle_time * 1000)
        ret, frame = cap.read()
        
        if ret:
            # Encode to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            frames.append((middle_time, frame_base64))
    
    cap.release()
    
    logger.info(f"Extracted {len(frames)} frames from scenes")
    return {
        "duration": duration,
        "frames": frames
    }
