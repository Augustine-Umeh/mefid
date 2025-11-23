from exports.utils.logger import get_logger
from fastapi import APIRouter

router = APIRouter()
logger = get_logger()

@router.post("/fixed_interval")
async def extract_fixed_interval():
    """
    Extract frames at fixed intervals from a video.
    """
    logger.info("Extracting frames at fixed intervals...")
    # Placeholder for extraction logic
    return {
        "message": "Frames extracted at fixed intervals."
    }

@router.post("/scene_change")
async def extract_scene_change():
    """
    Extract frames based on scene changes from a video.
    """
    logger.info("Extracting frames based on scene changes...")
    # Placeholder for extraction logic
    return {"message": "Frames extracted based on scene changes."}