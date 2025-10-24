from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict


@dataclass
class BaseModel:
    """Base class for all database models"""
    id: Optional[int] = None
    created_at: Optional[str] = None
    
    def to_dict(self, exclude_none: bool = True) -> dict:
        """Convert the dataclass to a dictionary, optionally excluding None values."""
        result = asdict(self)
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result
    
    def is_new(self) -> bool:
        """Check if the model instance is new (i.e., has no ID)."""
        return self.id is None
        
@dataclass
class Media(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    source_url: Optional[str] = None
    media_type: Optional[str] = None # e.g., 'video', 'image'
    duration_seconds: Optional[float] = None
    storage_path: Optional[str] = None
    status: Optional[str] = None  # e.g., 'processing', 'completed', 'error'
    updated_at: Optional[str] = None
    

@dataclass
class Frame(BaseModel):
    media_id: Optional[int] = None
    frame_number: Optional[int] = None
    timestamp_seconds: Optional[float] = None
    storage_path: Optional[str] = None


@dataclass
class FrameMetadata(BaseModel):
    frame_id: Optional[int] = None
    scene_description: Optional[str] = None
    detected_objects: Optional[list[str]] = field(default_factory=list)
    detected_text: Optional[str] = None
    color_palette: Optional[list[str]] = field(default_factory=list)


@dataclass
class SearchQuery(BaseModel):
    query_type: Optional[str] = None # e.g., 'text', 'image'
    query_text: Optional[str] = None
    top_result_frame_id: Optional[int] = None
    clicked: Optional[bool] = False
    latency_ms: Optional[int] = None
    top_k_results: List[Dict[str, float]] = field(default_factory=list)
    

@dataclass
class Embedding(BaseModel):
    frame_id: Optional[int] = None
    model_name: Optional[str] = None
    vector: Optional[list[float]] = field(default_factory=list)
    faiss_index_path: Optional[str] = None
