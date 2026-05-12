from fastapi import APIRouter
from exports.schema.models import AddVectorsRequest, AddVectorsResponse
from exports.utils.logger import get_logger

router = APIRouter()
logger = get_logger()


@router.post("/")
async def add_vectors(body: AddVectorsRequest) -> AddVectorsResponse:
    """Stub: accept vectors so the API pipeline parses; real FAISS + DB writes come later."""
    n = len(body.vectors)
    logger.info("Indexer stub add_vectors media_id=%s count=%s", body.media_id, n)
    return AddVectorsResponse(count=n)
