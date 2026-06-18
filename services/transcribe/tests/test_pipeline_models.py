"""Validation tests for pipeline models used by Step 2."""

import pytest
from uuid import uuid4

from exports.schema.constants import VectorType
from exports.schema.models import AddVectorItem, EmbeddingCreate


def test_embedding_create_requires_exactly_one_source() -> None:
    frame_id = uuid4()
    transcript_id = uuid4()

    EmbeddingCreate(
        frame_id=frame_id,
        faiss_index_id=1,
        vector_type=VectorType.IMAGE,
    )
    EmbeddingCreate(
        transcript_id=transcript_id,
        faiss_index_id=2,
        vector_type=VectorType.TEXT,
    )

    with pytest.raises(ValueError, match="Exactly one"):
        EmbeddingCreate(
            frame_id=frame_id,
            transcript_id=transcript_id,
            faiss_index_id=3,
            vector_type=VectorType.TEXT,
        )

    with pytest.raises(ValueError, match="Exactly one"):
        EmbeddingCreate(
            faiss_index_id=4,
            vector_type=VectorType.TEXT,
        )


def test_add_vector_item_requires_exactly_one_source() -> None:
    frame_id = uuid4()
    transcript_id = uuid4()

    AddVectorItem(
        frame_id=frame_id,
        embedding=[0.1, 0.2],
        vector_type=VectorType.IMAGE,
    )
    AddVectorItem(
        transcript_id=transcript_id,
        embedding=[0.3, 0.4],
        vector_type=VectorType.TEXT,
    )

    with pytest.raises(ValueError, match="Exactly one"):
        AddVectorItem(
            frame_id=frame_id,
            transcript_id=transcript_id,
            embedding=[0.5],
            vector_type=VectorType.TEXT,
        )
