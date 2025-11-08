from fastapi import APIRouter
from services.api.src.schema.responses import SearchResponse
from services.exports.src.schema.models import SearchRequest

router = APIRouter()

@router.post("/image")
async def search_by_media(
    queyr: SearchRequest
) -> SearchResponse:
    """
    Receive image from frontend
    Return top-k results to frontend
    """
    pass

@router.post("/text")
async def search_by_text(
    query: SearchRequest
) -> SearchResponse:
    """
    Receive text from frontend
    Return top-k results to frontend
    """
    pass

@router.post("/video")
async def search_by_video(
    query: SearchRequest
) -> SearchResponse:
    """
    Receive video from frontend
    Return top-k results to frontend
    """
    pass

@router.post("/multimodal")
async def search_by_multimodal(
    query: SearchRequest
) -> SearchResponse:
    """
    Receive text + image/video or image + video from frontend
    Return top-k results to frontend
    """
    pass
