from exports.utils.logger import get_logger
from fastapi import APIRouter

router = APIRouter()
logger = get_logger()

@router.post("/")
async def extract_media():
    """
    Endpoint to add media embeddings to the indexer (faiss).
    """
    logger.info("Adding media embeddings to the indexer...")
    # Placeholder for extraction logic
    return {
        "message": "Media embeddings extraction added to indexer."
    }