from fastapi import APIRouter
from main_app import app

router = APIRouter()

@router.get("/media")
async def get_all_media() -> dict:
    """Route to get all media entries."""
    media_entries = await app.state.db.get_all_media()
    return {"media": media_entries}

@router.get("/media/{media_id}")
async def get_media_by_id(media_id: str) -> dict:
    """Route to get a specific media entry by its ID."""
    media_entry = await app.state.db.get_media_by_id(media_id)
    return {"media": media_entry}