from __future__ import annotations

import asyncio
import os
from typing import Dict

import faiss
from fastapi import APIRouter, Request

from exports.faiss_store import FaissIndexRegistry
from exports.schema.constants import VectorType

router = APIRouter()


def _disk_ntotal(index_path: str) -> int:
    """Get the number of vectors on disk."""
    if not index_path or not os.path.exists(index_path):
        return 0
    disk_index = faiss.read_index(index_path)
    return int(disk_index.ntotal)


def _collect_stats(faiss_registry: FaissIndexRegistry) -> dict:
    ntotal: Dict[str, int] = {}
    max_id: Dict[str, int | None] = {}
    disk_ntotal: Dict[str, int] = {}

    for vector_type in VectorType:
        store = faiss_registry.store_for(vector_type)
        count = int(store.ntotal) # total number of vectors in the index
        key = vector_type.value
        ntotal[key] = count
        max_id[key] = count - 1 if count > 0 else None
        disk_ntotal[key] = _disk_ntotal(store.path)

    # check if the number of vectors on disk is different from the number of vectors in memory
    memory_disk_drift = any(
        ntotal[key] != disk_ntotal[key] for key in ntotal
    )

    return {
        "live": True,
        "index_loaded": True,
        "ready": True,
        "ntotal": ntotal,
        "ntotal_sum": sum(ntotal.values()),
        "max_id": max_id,
        "disk_ntotal": disk_ntotal,
        "memory_disk_drift": memory_disk_drift,
        "index_path": faiss_registry.base_dir,
    }


@router.get("/stats")
async def index_stats(request: Request) -> dict:
    """FAISS index counts, max IDs, and in-memory vs on-disk consistency."""
    faiss_registry: FaissIndexRegistry | None = getattr(
        request.app.state, "faiss", None
    )
    if faiss_registry is None:
        return {
            "live": True,
            "index_loaded": False,
            "ready": False,
            "ntotal": {vt.value: 0 for vt in VectorType},
            "ntotal_sum": 0,
            "max_id": {vt.value: None for vt in VectorType},
            "disk_ntotal": {vt.value: 0 for vt in VectorType},
            "memory_disk_drift": False,
            "index_path": None,
        }

    # collect stats in a separate thread to avoid blocking the main thread
    return await asyncio.to_thread(_collect_stats, faiss_registry)
