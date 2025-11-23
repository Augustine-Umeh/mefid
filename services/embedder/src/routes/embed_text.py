from fastapi import APIRouter
from exports.db_clients.lifespan import lifespan
from exports.utils.logger import get_logger

router = APIRouter()
logger = get_logger()

@router.post("/")
async def embed_text():
    """
    Root endpoint for embed text service.
    """
    logger.info("Embed Text Service Root Accessed")
    return {"message": "Embed Text Service is running."}