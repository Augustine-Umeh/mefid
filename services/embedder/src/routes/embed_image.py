from fastapi import APIRouter
from exports.utils.logger import get_logger

router = APIRouter()
logger = get_logger()

@router.post("/")
async def embed_image_root():
    """
    Root endpoint for embed image service.
    """
    logger.info("Embed Image Service Root Accessed")
    return {"message": "Embed Image Service is running."}