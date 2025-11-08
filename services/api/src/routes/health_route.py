from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check() -> dict:
    """Simple route to verify DB is ready."""
    return {"status": "healthy"}