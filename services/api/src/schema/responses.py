from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, HttpUrl

from exports.schema.constants import MediaType, QueryType, VectorType


class UploadResponse(BaseModel):
    """Response shape for `/upload/image` and `/upload/video`."""
    file_url: Optional[HttpUrl] = None
    status: str = "uploaded"


class SearchResult(BaseModel):
    """One row in a search result set, joining FAISS hits to media + frames/transcripts."""
    media_id: UUID
    similarity: float
    media_type: MediaType
    file_name: str
    file_url: str
    vector_type: VectorType = VectorType.IMAGE
    frame_id: Optional[UUID] = None
    timestamp: Optional[float] = None
    frame_url: Optional[str] = None
    transcript_text: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class SearchResponse(BaseModel):
    """Response shape for the various `/search/*` endpoints."""
    query_type: QueryType
    top_k: int
    results: List[SearchResult]
