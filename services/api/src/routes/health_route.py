from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check() -> dict:
    """Simple route to verify DB is ready."""
    from main_app import app # Importing here to avoid circular imports
    if not hasattr(app.state, "db") or app.state.db is None:
        return {"status": "initializing..."}
    return {"status": "ok"}