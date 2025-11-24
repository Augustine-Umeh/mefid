import cv2
import tempfile
from fastapi import UploadFile, File

async def get_video_duration(
    video_file: UploadFile = File(...)
) -> float:
    """
    Returns the duration of the uploaded video in seconds.
    """

    # Save the uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(await video_file.read())
        temp_path = temp_file.name

    try:
        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise ValueError("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        cap.release()

        return frame_count / fps if fps > 0 else 0.0

    finally:
        import os
        os.unlink(temp_path)
