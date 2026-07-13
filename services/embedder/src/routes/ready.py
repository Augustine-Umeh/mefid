from fastapi import APIRouter, Request

from exports.schema.constants import CLIP_MODEL

router = APIRouter()


@router.get("/ready")
async def ready(request: Request) -> dict:
    """Report whether CLIP is loaded and able to serve embed requests."""
    engine = getattr(request.app.state, "clip_engine", None)
    model_loaded = engine is not None
    return {
        "live": True,
        "ready": model_loaded,
        "model_loaded": model_loaded,
        "model_name": CLIP_MODEL or None if model_loaded else None,
    }
