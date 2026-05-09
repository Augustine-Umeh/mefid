"""Search endpoints.

These are intentional stubs — the embedder + indexer routes they depend
on aren't implemented yet (see `MILESTONE.md`). Once those land, each
handler will: embed the query, ask the indexer for nearest neighbours,
join hits back to ``media``/``frames`` rows in Supabase, and return a
``SearchResponse``.
"""

from fastapi import APIRouter

from exports.schema.models import SearchRequest

from src.schema.responses import SearchResponse

router = APIRouter()


@router.post("/image")
async def search_by_image(query: SearchRequest) -> SearchResponse:
    """Search by uploaded image (placeholder)."""
    raise NotImplementedError("Image search is not wired yet.")


@router.post("/text")
async def search_by_text(query: SearchRequest) -> SearchResponse:
    """Search by natural-language description (placeholder)."""
    raise NotImplementedError("Text search is not wired yet.")


@router.post("/video")
async def search_by_video(query: SearchRequest) -> SearchResponse:
    """Search by uploaded video clip (placeholder)."""
    raise NotImplementedError("Video search is not wired yet.")


@router.post("/multimodal")
async def search_by_multimodal(query: SearchRequest) -> SearchResponse:
    """Search by combined text + image/video query (placeholder)."""
    raise NotImplementedError("Multimodal search is not wired yet.")
