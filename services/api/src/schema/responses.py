from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, HttpUrl

from exports.schema.constants import MediaType, QueryType, VectorType


class UploadResponse(BaseModel):
    """Response shape for `/upload/image` and `/upload/video`."""
    file_url: Optional[HttpUrl] = None
    status: str = "uploaded"


class SignalMatch(BaseModel):
    """One contributing signal inside a fused search moment."""
    vector_type: VectorType
    similarity: float
    timestamp: float
    frame_id: Optional[UUID] = None
    frame_url: Optional[str] = None
    transcript_text: Optional[str] = None
    caption_text: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class SearchResult(BaseModel):
    """One row in a search result set, joining FAISS hits to media + frames/transcripts/captions."""
    media_id: UUID
    similarity: float
    media_type: MediaType
    file_name: str
    file_url: str
    signal_types: List[VectorType] = []
    signal_count: int = 0
    signals: List[SignalMatch] = []
    vector_type: VectorType = VectorType.IMAGE
    frame_id: Optional[UUID] = None
    timestamp: Optional[float] = None
    frame_url: Optional[str] = None
    transcript_text: Optional[str] = None
    caption_text: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class SearchResponse(BaseModel):
    """Response shape for the various `/search/*` endpoints."""
    query_type: QueryType
    top_k: int
    results: List[SearchResult]
