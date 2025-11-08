from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel
from .constants import MediaType
from pydantic import BaseModel, Field, HttpUrl
from fastapi import UploadFile, File, Form


# ==============================================
# Dataclasses (Internal models / domain layer)
# ==============================================
@dataclass
class BaseSchema:
    """Base class for all database models"""
    id: Optional[int] = None
    created_at: Optional[datetime] = None

    def to_dict(self, exclude_none: bool = True) -> dict:
        """Convert dataclass to dict, optionally excluding None values."""
        result = asdict(self)
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result

    def is_new(self) -> bool:
        """Check if the instance is new (has no ID)."""
        return self.id is None


@dataclass
class Media(BaseSchema):
    """Media model representing a media item."""
    title: Optional[str] = None
    description: Optional[str] = None
    source_url: Optional[str] = None
    media_type: Optional[MediaType] = None
    file_name: Optional[str] = None
    duration_seconds: Optional[float] = None
    storage_path: Optional[str] = None
    status: Optional[str] = None
    updated_at: Optional[datetime] = None


@dataclass
class Frame(BaseSchema):
    """Frame model representing a frame extracted from media."""
    media_id: Optional[int] = None
    frame_number: Optional[int] = None
    timestamp_seconds: Optional[float] = None
    storage_path: Optional[str] = None


@dataclass
class FrameMetadata(BaseSchema):
    """FrameMetadata model representing metadata for a frame."""
    frame_id: Optional[int] = None
    scene_description: Optional[str] = None
    detected_objects: Optional[List[str]] = field(default_factory=list)
    detected_text: Optional[str] = None
    color_palette: Optional[List[str]] = field(default_factory=list)


@dataclass
class SearchQuery(BaseSchema):
    """SearchQuery model representing a search query made by a user."""
    query_type: Optional[str] = None
    query_text: Optional[str] = None
    top_result_frame_id: Optional[int] = None
    clicked: Optional[bool] = False
    latency_ms: Optional[int] = None
    top_k_results: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class Embedding(BaseSchema):
    """Embedding model representing a vector embedding for a frame."""
    frame_id: Optional[int] = None
    model_name: Optional[str] = None
    vector: Optional[List[float]] = field(default_factory=list)
    faiss_index_path: Optional[str] = None


@dataclass
class Indexes(BaseSchema):
    """Indexes model representing an index for embeddings."""
    name: Optional[str] = None
    version: Optional[int] = None
    description: Optional[str] = None
    index_path: Optional[str] = None
    metadata_path: Optional[str] = None
    num_vectors: Optional[int] = None
    vector_dimension: Optional[int] = None
    model_name: Optional[str] = None
    updated_at: Optional[datetime] = None
    status: Optional[str] = None

# ==============================================
# Pydantic Models (API schemas / validation layer)
# ==============================================
class MediaCreate(BaseModel):
    """Model for creating a new media item."""
    title: str
    source_url: str
    file_name: str
    media_type: MediaType
    description: Optional[str] = None
    duration_seconds: Optional[float] = None
    storage_path: Optional[str] = None
    status: Optional[str] = "processing"


class FrameCreate(BaseModel):
    """Model for creating a new frame."""
    media_id: int
    frame_number: int
    timestamp_seconds: float
    storage_path: str


class FrameMetadataCreate(BaseModel):
    """Model for creating new frame metadata."""
    frame_id: int
    scene_description: Optional[str] = None
    detected_objects: Optional[List[str]] = []
    detected_text: Optional[str] = None
    color_palette: Optional[List[str]] = []


class SearchQueryCreate(BaseModel):
    """Model for creating a new search query."""
    query_type: str
    query_text: str
    top_result_frame_id: Optional[int] = None
    clicked: bool = False
    latency_ms: Optional[int] = None
    top_k_results: List[Dict[str, float]] = []


class EmbeddingCreate(BaseModel):
    """Model for creating a new embedding."""
    frame_id: int
    model_name: str
    vector: List[float]
    faiss_index_path: str
    
    
class IndexesCreate(BaseModel):
    """Model for creating a new index for embeddings."""
    name: str
    version: int
    description: Optional[str] = None
    index_path: str
    metadata_path: str
    num_vectors: int
    vector_dimension: int
    model_name: str
    status: Optional[str] = "building"

    
class UploadRequest(BaseModel):
    image_query: Optional[UploadFile] = File(None, description="Image file for search.")
    video_query: Optional[UploadFile] = Field(None, description="Video file for search.")
    media_type: MediaType = Field(..., description="Type of media being uploaded.")
    title: Optional[str] = Field(None, description="Title of the media.")
    filename: Optional[str] = Field(None, description="Name of the media file.")
    description: Optional[str] = Field(None, description="Description of the media.")
    duration_seconds: Optional[float] = Field(None, description="Duration of the media in seconds.")
    source_url: Optional[HttpUrl] = Field(None, description="Original source URL of the media.")


class SearchRequest(BaseModel):
    text_query: Optional[str] = Field(None, description="Text description for search.")
    image_query: Optional[UploadFile] = Field(None, description="Image file for search.")
    video_query: Optional[UploadFile] = Field(None, description="Video file for search.")
    media_type: MediaType = Field(
        None,
        description="Type of media to search against, e.g., 'image', 'video', 'all'",
    )