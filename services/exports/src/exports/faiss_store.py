"""In-process FAISS IndexFlatIP with disk persistence."""

from __future__ import annotations

import os
from typing import List, Tuple

import faiss
import numpy as np

from exports.schema.constants import CLIP_DIMENSION, FAISS_INDEX_PATH, VectorType


_INDEX_FILENAMES: dict[VectorType, str] = {
    VectorType.IMAGE: "image.index",
    VectorType.TEXT: "transcript.index",
    VectorType.CAPTION: "caption.index",
}


def faiss_index_dir() -> str:
    """Directory holding per-modality FAISS index files."""
    path = str(FAISS_INDEX_PATH or "").strip()
    if not path:
        return ""
    # Legacy single-file paths (embeddings.faiss, vectors.index, …) → parent dir.
    if path.endswith((".index", ".faiss")):
        return os.path.dirname(os.path.abspath(path)) or "."
    return os.path.abspath(path)


class FaissVectorStore:
    """Exact inner-product search on L2-normalized CLIP vectors."""

    def __init__(self, path: str, *, dimension: int = CLIP_DIMENSION) -> None:
        self._path = path
        self._dimension = dimension
        self._index: faiss.Index | None = None

    @property
    def path(self) -> str:
        return self._path

    @property
    def ntotal(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    def load(self) -> None:
        _ensure_parent_dir(self._path)
        if os.path.exists(self._path):
            self._index = faiss.read_index(self._path)
            if int(self._index.d) != self._dimension:
                raise RuntimeError(
                    f"FAISS index at {self._path} has dimension {self._index.d}, "
                    f"expected CLIP_DIMENSION={self._dimension}. "
                    "Remove the index file or align CLIP_DIMENSION."
                )
        else:
            self._index = faiss.IndexFlatIP(self._dimension)

    def add(self, vectors: np.ndarray) -> List[int]:
        if self._index is None:
            raise RuntimeError("FAISS index is not loaded.")
        if vectors.ndim != 2 or vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Expected shape (n, {self._dimension}), got {vectors.shape}"
            )
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)
        start = int(self._index.ntotal)
        self._index.add(vectors)
        n = int(vectors.shape[0])
        return list[int](range(start, start + n))

    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if self._index is None:
            raise RuntimeError("FAISS index is not loaded.")
        if int(self._index.ntotal) == 0:
            return []
        if top_k <= 0:
            return []
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self._dimension:
            raise ValueError(
                f"Expected query shape (1, {self._dimension}) or ({self._dimension},), "
                f"got {query.shape}"
            )
        q = query.astype(np.float32, copy=False)
        k = min(int(top_k), int(self._index.ntotal))
        scores, indices = self._index.search(q, k)
        out: List[Tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0], strict=True):
            if int(idx) < 0:
                continue
            out.append((int(idx), float(score)))
        return out

    def save(self) -> None:
        if self._index is None:
            return
        _ensure_parent_dir(self._path)
        faiss.write_index(self._index, self._path)


class FaissIndexRegistry:
    """One FAISS index per vector type (image, transcript, caption)."""

    def __init__(self, base_dir: str, *, dimension: int = CLIP_DIMENSION) -> None:
        self._base_dir = base_dir
        self._stores: dict[VectorType, FaissVectorStore] = {
            vector_type: FaissVectorStore(
                os.path.join(base_dir, filename),
                dimension=dimension,
            )
            for vector_type, filename in _INDEX_FILENAMES.items()
        }

    @property
    def base_dir(self) -> str:
        return self._base_dir

    def store_for(self, vector_type: VectorType) -> FaissVectorStore:
        return self._stores[vector_type]

    @property
    def ntotal(self) -> int:
        return sum(store.ntotal for store in self._stores.values())

    def load(self) -> None:
        os.makedirs(self._base_dir, exist_ok=True)
        for store in self._stores.values():
            store.load()

    def save(self) -> None:
        for store in self._stores.values():
            store.save()

    def add(self, vector_type: VectorType, vectors: np.ndarray) -> List[int]:
        return self._stores[vector_type].add(vectors)

    def search(
        self, vector_type: VectorType, query: np.ndarray, top_k: int
    ) -> List[Tuple[int, float]]:
        return self._stores[vector_type].search(query, top_k)

    def search_all(
        self, query: np.ndarray, top_k: int
    ) -> List[Tuple[VectorType, int, float]]:
        """Query every index with ``top_k`` neighbours each, merge by score."""
        merged: List[Tuple[VectorType, int, float]] = []
        for vector_type, store in self._stores.items():
            for faiss_index_id, score in store.search(query, top_k):
                merged.append((vector_type, faiss_index_id, score))
        merged.sort(key=lambda row: row[2], reverse=True)
        return merged[:top_k]


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
