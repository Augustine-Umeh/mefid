from fastapi import APIRouter, Request

from exports.schema.constants import WHISPER_MODEL

router = APIRouter()


@router.get("/ready")
async def ready(request: Request) -> dict:
    """Report whether Whisper is loaded and able to serve transcribe requests."""
    engine = getattr(request.app.state, "whisper_engine", None)
    model_loaded = engine is not None
    return {
        "live": True,
        "ready": model_loaded,
        "model_loaded": model_loaded,
        "model_name": WHISPER_MODEL or None if model_loaded else None,
    }
