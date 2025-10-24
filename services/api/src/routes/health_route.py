from fastapi import APIRouter
from main_app import app

router = APIRouter()

@router.get("/health")
async def health_check() -> dict:
    """Simple route to verify DB is ready."""
    if not hasattr(app.state, "db") or app.state.db is None:
        return {"status": "initializing..."}
    return {"status": "ok"}