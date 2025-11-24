import cv2
from typing import Dict
from exports.utils.logger import get_logger

logger = get_logger()


def get_video_metadata(video_path: str) -> Dict:
    """
    Extract metadata from video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dict with duration, fps, frame_count, width, height
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "duration": duration,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height
    }
