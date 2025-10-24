from fastapi import APIRouter
from main_app import app

router = APIRouter()

@router.get("/frames")
async def get_all_frames() -> dict:
    """Route to get all frame entries."""
    frame_entries = await app.state.db.get_all_frames()
    return {"frames": frame_entries}

@router.get("/frames/{frame_id}")
async def get_frame_by_id(frame_id: str) -> dict:
    """Route to get a specific frame entry by its ID."""
    frame_entry = await app.state.db.get_frame_by_id(frame_id)
    return {"frame": frame_entry}