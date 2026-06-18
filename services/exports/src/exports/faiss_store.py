"""In-process FAISS IndexFlatIP with disk persistence."""

from __future__ import annotations

import os
from typing import List, Tuple

import faiss
import numpy as np

from exports.schema.constants import CLIP_DIMENSION, FAISS_INDEX_PATH


class FaissVectorStore:
    """Exact inner-product search on L2-normalized CLIP vectors."""

    def __init__(self) -> None:
        self._path = FAISS_INDEX_PATH
        self._dimension = CLIP_DIMENSION
        self._index: faiss.Index | None = None

    @property
    def path(self) -> str:
        """
        Returns:
            str: Path to the FAISS index file
        """
        return self._path

    @property
    def ntotal(self) -> int:
        """
        Returns:
            int: Number of vectors in the index
        """
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    def load(self) -> None:
        """
        Load the FAISS index from the file system.
        """
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
        """
        Append rows to the FAISS index.
        vectors: numpy array of vectors to add to the index
        Returns:
            List[int]: List of new faiss_index_id values in row order
        """
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
        """
        query: embedding vector to search for in the index
        top_k: number of nearest neighbors to return
        Returns:
            List[Tuple[int, float]]: List of tuples containing the index of the nearest neighbor and the similarity score
        """
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
        """
        Save the FAISS index to the file system.
        """
        if self._index is None:
            return
        _ensure_parent_dir(self._path)
        faiss.write_index(self._index, self._path)


def _ensure_parent_dir(path: str) -> None:
    """
    Ensure the parent directory of the FAISS index file exists.
    path: path to the FAISS index file
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
