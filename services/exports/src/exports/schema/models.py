"""Pydantic models for Mefid.

Layout:
  * **Row models** mirror Postgres rows in `dev_schema.sql` exactly
    (used when reading from Supabase).
  * **Create/Update models** are insert/update payloads — only the
    columns the caller is allowed to set.
  * **API request models** describe HTTP form / JSON inputs.
  * **Pipeline DTOs** describe the payloads the services exchange
    over HTTP between media_processor → embedder → indexer.

Rule of thumb: if a field is in the Postgres table, the matching
attribute name here uses the same spelling.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from .constants import (
    ContentType,
    ExtractionStrategy,
    MediaStatus,
    MediaType,
    QueryType,
    VectorType,
)


# ============================================================
# DB row models (reads)
# ============================================================
class MediaRow(BaseModel):
    """Row in `public.media`."""
    id: UUID
    file_name: str
    file_url: str
    media_type: MediaType
    duration: Optional[float] = None
    content_type: Optional[ContentType] = None
    extraction_strategy: Optional[ExtractionStrategy] = None
    status: MediaStatus = MediaStatus.PROCESSING
    created_at: datetime


class FrameRow(BaseModel):
    """Row in `public.frames`."""
    id: UUID
    media_id: UUID
    timestamp: float
    frame_url: str
    phash: Optional[str] = None
    sequence_number: int
    created_at: datetime


class EmbeddingRow(BaseModel):
    """Row in `public.embeddings`. Vectors live in FAISS, not here."""
    id: UUID
    frame_id: Optional[UUID] = None
    transcript_id: Optional[UUID] = None
    faiss_index_id: int
    vector_type: VectorType
    created_at: datetime


class TranscriptRow(BaseModel):
    """Row in `public.transcripts`."""
    id: UUID
    media_id: UUID
    start_time: float
    end_time: float
    text: str
    created_at: datetime


class SearchQueryRow(BaseModel):
    """Row in `public.search_queries`."""
    id: UUID
    query_text: Optional[str] = None
    query_type: QueryType
    created_at: datetime


# ============================================================
# Create / Update models (writes)
# ============================================================
class MediaCreate(BaseModel):
    file_name: str
    file_url: str
    media_type: MediaType
    duration: Optional[float] = None
    content_type: Optional[ContentType] = None
    extraction_strategy: Optional[ExtractionStrategy] = None
    status: MediaStatus = MediaStatus.PROCESSING


class MediaUpdate(BaseModel):
    """Partial update for a media row (e.g. status flips)."""
    duration: Optional[float] = None
    content_type: Optional[ContentType] = None
    extraction_strategy: Optional[ExtractionStrategy] = None
    status: Optional[MediaStatus] = None


class FrameCreate(BaseModel):
    media_id: UUID
    timestamp: float
    frame_url: str
    sequence_number: int
    phash: Optional[str] = None


class EmbeddingCreate(BaseModel):
    frame_id: Optional[UUID] = None
    transcript_id: Optional[UUID] = None
    faiss_index_id: int
    vector_type: VectorType

    @model_validator(mode="after")
    def exactly_one_source(self) -> "EmbeddingCreate":
        has_frame = self.frame_id is not None
        has_transcript = self.transcript_id is not None
        if has_frame == has_transcript:
            raise ValueError("Exactly one of frame_id or transcript_id must be set")
        return self


class TranscriptCreate(BaseModel):
    media_id: UUID
    start_time: float
    end_time: float
    text: str


class SearchQueryCreate(BaseModel):
    query_type: QueryType
    query_text: Optional[str] = None


# ============================================================
# API request models (HTTP form / JSON inputs)
# ============================================================
class UploadImageRequest(BaseModel):
    """Internal model that bundles metadata for an image upload."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_query: UploadFile = Field(..., description="Image file being uploaded.")
    title: Optional[str] = None
    filename: Optional[str] = None
    description: Optional[str] = None
    source_url: Optional[HttpUrl] = None


class UploadVideoRequest(BaseModel):
    """Internal model that bundles metadata for a video upload."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    video_query: UploadFile = Field(..., description="Video file being uploaded.")
    title: Optional[str] = None
    filename: Optional[str] = None
    description: Optional[str] = None
    duration_seconds: Optional[float] = None
    source_url: Optional[HttpUrl] = None


class SearchRequest(BaseModel):
    """Search input from the frontend."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    query_type: QueryType
    text_query: Optional[str] = None
    image_query: Optional[UploadFile] = None
    video_query: Optional[UploadFile] = None


class TextSearchRequest(BaseModel):
    """JSON body for ``POST /search/text`` (no multipart)."""
    text: str
    top_k: Optional[int] = None
    vector_type: Optional[VectorType] = None


# ============================================================
# Pipeline DTOs
# ------------------------------------------------------------
# Wire shapes for the media_processor -> embedder -> indexer chain.
# ============================================================

# ---- media_processor /extract response ----
class FrameData(BaseModel):
    """One extracted + persisted frame.

    Carries both the storage URL (for later display) and the in-memory
    base64 payload (so the embedder can read pixels without round-
    tripping through MinIO).
    """
    frame_id: UUID
    sequence_number: int
    timestamp: float
    frame_url: str
    frame_data: str  # base64-encoded JPEG


class ExtractFramesResponse(BaseModel):
    media_id: UUID
    frames: List[FrameData]
    frame_count: int
    strategy: ExtractionStrategy
    duration: float


# ---- embedder I/O ----
class EmbedImageItem(BaseModel):
    frame_id: UUID
    frame_data: str  # base64-encoded JPEG


class EmbedImagesRequest(BaseModel):
    frames: List[EmbedImageItem]


class EmbeddingResult(BaseModel):
    frame_id: UUID
    embedding: List[float]


class EmbedImagesResponse(BaseModel):
    embeddings: List[EmbeddingResult]


class EmbedTextRequest(BaseModel):
    text: str


class EmbedTextResponse(BaseModel):
    embedding: List[float]


class EmbedTextItem(BaseModel):
    transcript_id: UUID
    text: str


class EmbedTextBatchRequest(BaseModel):
    texts: List[EmbedTextItem]


class TextEmbeddingResult(BaseModel):
    transcript_id: UUID
    embedding: List[float]


class EmbedTextBatchResponse(BaseModel):
    embeddings: List[TextEmbeddingResult]


# ---- transcribe I/O ----
class TranscriptSegmentData(BaseModel):
    id: UUID
    start_time: float
    end_time: float
    text: str


class TranscribeRequest(BaseModel):
    video_object_key: str
    media_id: UUID
    file_name: str


class TranscribeResponse(BaseModel):
    media_id: UUID
    segments: List[TranscriptSegmentData]
    segment_count: int


# ---- indexer I/O ----
class AddVectorItem(BaseModel):
    frame_id: Optional[UUID] = None
    transcript_id: Optional[UUID] = None
    embedding: List[float]
    vector_type: VectorType = VectorType.IMAGE

    @model_validator(mode="after")
    def exactly_one_source(self) -> "AddVectorItem":
        has_frame = self.frame_id is not None
        has_transcript = self.transcript_id is not None
        if has_frame == has_transcript:
            raise ValueError("Exactly one of frame_id or transcript_id must be set")
        return self


class AddVectorsRequest(BaseModel):
    media_id: UUID
    vectors: List[AddVectorItem]


class AddVectorsResponse(BaseModel):
    count: int


class SearchVectorsRequest(BaseModel):
    embedding: List[float]
    top_k: int


class IndexerVectorHit(BaseModel):
    """One FAISS neighbour; the API joins ``faiss_index_id`` to frames via Supabase."""

    faiss_index_id: int
    similarity_score: float


class SearchVectorsResponse(BaseModel):
    hits: List[IndexerVectorHit]
