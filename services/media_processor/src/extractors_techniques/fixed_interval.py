import cv2
import base64
from typing import List, Tuple
import numpy as np
from exports.utils.logger import get_logger

logger = get_logger()


def extract_frames_fixed_interval(
    video_path: str,
    interval_seconds: float
) -> dict:
    """
    Extract frames at fixed time intervals.
    
    Args:
        video_path: Path to video file
        interval_seconds: Seconds between frames
        
    Returns:
        List of (timestamp, base64_encoded_frame) tuples
    """
    logger.info(f"Extracting frames from {video_path} every {interval_seconds}s")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    logger.info(f"Video info - FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
    
    frames = []
    timestamp = 0.0
    
    while timestamp <= duration:
        # Seek to timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        frames.append((timestamp, frame_base64))
        timestamp += interval_seconds
    
    cap.release()
    
    logger.info(f"Extracted {len(frames)} frames")
    
    return {
        "duration": duration,
        "frames": frames
    }
