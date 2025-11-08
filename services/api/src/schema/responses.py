from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional
from services.exports.src.schema.constants import MediaType

class UploadResponse(BaseModel):
    file_url: Optional[HttpUrl]
    status: str = "uploaded"

class SearchResult(BaseModel):
    clip_id: str
    similarity: float
    media_type: MediaType
    title: Optional[str]
    description: Optional[str]
    thumbnail_url: Optional[HttpUrl]
    clip_preview_url: Optional[HttpUrl]
    timestamp: Optional[float]

class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[SearchResult]
    