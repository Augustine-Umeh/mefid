"""Data types for temporal signal fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from exports.schema.constants import MediaType, VectorType


@dataclass
class FusionHit:
    """One FAISS neighbour hydrated with media/timestamp metadata."""

    faiss_index_id: int
    vector_type: VectorType
    raw_score: float
    normalized_score: float = 0.0
    media_id: UUID = field(default_factory=lambda: UUID(int=0))
    timestamp: float = 0.0
    media_type: MediaType = MediaType.VIDEO
    file_name: str = ""
    file_url: str = ""
    frame_id: Optional[UUID] = None
    frame_url: Optional[str] = None
    transcript_text: Optional[str] = None
    caption_text: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    fused_score: float = 0.0


@dataclass
class MomentGroup:
    """Hits from different signals that describe the same moment in one video."""

    media_id: UUID
    anchor_time: float
    hits: list[FusionHit] = field(default_factory=list)

    @property
    def signal_types(self) -> list[VectorType]:
        seen: set[VectorType] = set()
        ordered: list[VectorType] = []
        for hit in self.hits:
            if hit.vector_type not in seen:
                seen.add(hit.vector_type)
                ordered.append(hit.vector_type)
        return ordered

    @property
    def best_hit(self) -> FusionHit:
        return max(self.hits, key=lambda hit: hit.normalized_score)

    @property
    def best_normalized_score(self) -> float:
        return self.best_hit.normalized_score
