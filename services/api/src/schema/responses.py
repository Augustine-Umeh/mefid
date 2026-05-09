from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, HttpUrl

from exports.schema.constants import MediaType, QueryType


class UploadResponse(BaseModel):
    """Response shape for `/upload/image` and `/upload/video`."""
    file_url: Optional[HttpUrl] = None
    status: str = "uploaded"


class SearchResult(BaseModel):
    """One row in a search result set, joining FAISS hits to media + frames."""
    frame_id: UUID
    media_id: UUID
    timestamp: float
    frame_url: str
    similarity: float
    media_type: MediaType
    file_name: str
    file_url: str


class SearchResponse(BaseModel):
    """Response shape for the various `/search/*` endpoints."""
    query_type: QueryType
    top_k: int
    results: List[SearchResult]
