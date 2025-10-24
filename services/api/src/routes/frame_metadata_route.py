from fastapi import APIRouter
from main_app import app

router = APIRouter()

@router.get("/frame_metadata")
async def get_all_frame_metadata() -> dict:
    """Route to get all frame metadata entries."""
    frame_metadata_entries = await app.state.db.get_all_frame_metadata()
    return {"frame_metadata": frame_metadata_entries}

@router.get("/frame_metadata/{metadata_id}")
async def get_frame_metadata_by_id(metadata_id: str) -> dict:
    """Route to get a specific frame metadata entry by its ID."""
    frame_metadata_entry = await app.state.db.get_frame_metadata_by_id(metadata_id)
    return {"frame_metadata": frame_metadata_entry}