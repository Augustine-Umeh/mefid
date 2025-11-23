from exports.utils.logger import get_logger
from fastapi import APIRouter

router = APIRouter()
logger = get_logger()

@router.post("/")
async def extract_media():
    """
    Endpoint to perform search on media embeddings from the indexer.
    """
    logger.info("Searching media embeddings from the indexer...")
    # Placeholder for search logic
    return {
        "message": "Media embeddings search performed."
    }