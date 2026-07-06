from fastapi import APIRouter
from service_clients.embedder_client import EmbedderClient
from service_clients.indexer_client import IndexerClient
from service_clients.media_processor_client import MediaProcessorClient
from service_clients.transcribe_client import TranscribeClient
from service_clients.caption_client import CaptionClient

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check():
    """Check health of API service"""
    return {
        "status": "healthy",
        "service": "api"
    }


@router.get("/all")
async def health_check_all():
    """Check health of all services"""
    
    services = {
        "api": {"status": "healthy"},
        "media_processor": {"status": "unknown"},
        "embedder": {"status": "unknown"},
        "indexer": {"status": "unknown"},
        "transcribe": {"status": "unknown"},
        "caption": {"status": "unknown"},
    }
    
    # Check Media Processor
    try:
        async with MediaProcessorClient() as client:
            await client.health_check()
            services["media_processor"]["status"] = "healthy"
    except Exception as e:
        services["media_processor"]["status"] = "unhealthy"
        services["media_processor"]["error"] = str(e)
    
    # Check Embedder
    try:
        async with EmbedderClient() as client:
            await client.health_check()
            services["embedder"]["status"] = "healthy"
    except Exception as e:
        services["embedder"]["status"] = "unhealthy"
        services["embedder"]["error"] = str(e)
    
    # Check Indexer
    try:
        async with IndexerClient() as client:
            await client.health_check()
            services["indexer"]["status"] = "healthy"
    except Exception as e:
        services["indexer"]["status"] = "unhealthy"
        services["indexer"]["error"] = str(e)

    # Check Transcribe
    try:
        async with TranscribeClient() as client:
            await client.health_check()
            services["transcribe"]["status"] = "healthy"
    except Exception as e:
        services["transcribe"]["status"] = "unhealthy"
        services["transcribe"]["error"] = str(e)

    # Check Caption
    try:
        async with CaptionClient() as client:
            await client.health_check()
            services["caption"]["status"] = "healthy"
    except Exception as e:
        services["caption"]["status"] = "unhealthy"
        services["caption"]["error"] = str(e)

    # Overall status
    all_healthy = all(
        svc["status"] == "healthy" 
        for svc in services.values()
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": services
    }
