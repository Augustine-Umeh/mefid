from fastapi import APIRouter, Request

router = APIRouter(prefix="/health", tags=["health"])


def _live_entry(*, live: bool, error: str | None = None, **extra) -> dict:
    entry: dict = {"live": live, **extra}
    if error:
        entry["error"] = error
    return entry


def _service_status(live: bool, ready: bool | None = None) -> str:
    if not live:
        return "unhealthy"
    if ready is False:
        return "degraded"
    return "healthy"


@router.get("/")
async def health_check():
    """Check health of API service."""
    return {
        "status": "healthy",
        "service": "api",
        "live": True,
    }


@router.get("/all")
async def health_check_all(request: Request):
    """Liveness and readiness for all Mefid services."""
    media_processor = request.app.state.media_processor
    embedder = request.app.state.embedder
    indexer = request.app.state.indexer
    transcribe = request.app.state.transcribe
    caption = request.app.state.caption

    services: dict[str, dict] = {
        "api": {"live": True, "ready": True, "status": "healthy"},
        "media_processor": {"live": False, "ready": None, "status": "unknown"},
        "embedder": {"live": False, "ready": None, "status": "unknown"},
        "indexer": {"live": False, "ready": None, "status": "unknown"},
        "transcribe": {"live": False, "ready": None, "status": "unknown"},
        "caption": {"live": False, "ready": None, "status": "unknown"},
    }

    try:
        await media_processor.health_check()
        services["media_processor"] = _live_entry(live=True, ready=True)
    except Exception as exc:
        services["media_processor"] = _live_entry(live=False, error=str(exc))

    try:
        ready_payload = await embedder.get_ready()
        services["embedder"] = _live_entry(
            live=True,
            ready=bool(ready_payload.get("model_loaded")),
            model_loaded=ready_payload.get("model_loaded"),
            model_name=ready_payload.get("model_name"),
        )
    except Exception as exc:
        services["embedder"] = _live_entry(live=False, error=str(exc))

    try:
        stats = await indexer.get_index_stats()
        services["indexer"] = _live_entry(
            live=True,
            ready=bool(stats.get("index_loaded")),
            index_loaded=stats.get("index_loaded"),
            ntotal=stats.get("ntotal"),
            ntotal_sum=stats.get("ntotal_sum"),
            max_id=stats.get("max_id"),
            disk_ntotal=stats.get("disk_ntotal"),
            memory_disk_drift=stats.get("memory_disk_drift"),
        )
    except Exception as exc:
        services["indexer"] = _live_entry(live=False, error=str(exc))

    try:
        ready_payload = await transcribe.get_ready()
        services["transcribe"] = _live_entry(
            live=True,
            ready=bool(ready_payload.get("model_loaded")),
            model_loaded=ready_payload.get("model_loaded"),
            model_name=ready_payload.get("model_name"),
        )
    except Exception as exc:
        services["transcribe"] = _live_entry(live=False, error=str(exc))

    try:
        await caption.health_check()
        services["caption"] = _live_entry(live=True, ready=True)
    except Exception as exc:
        services["caption"] = _live_entry(live=False, error=str(exc))

    for name, entry in services.items():
        if name == "api":
            continue
        live = bool(entry.get("live"))
        ready = entry.get("ready")
        ready_bool = ready if isinstance(ready, bool) else None
        entry["status"] = _service_status(live, ready_bool)

    all_healthy = all(svc.get("status") == "healthy" for svc in services.values())
    any_unhealthy = any(svc.get("status") == "unhealthy" for svc in services.values())

    if all_healthy:
        overall = "healthy"
    elif any_unhealthy:
        overall = "degraded"
    else:
        overall = "degraded"

    return {
        "status": overall,
        "services": services,
    }
