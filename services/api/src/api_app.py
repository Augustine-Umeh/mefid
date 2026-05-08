from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from exports.db_clients.lifespan import lifespan as shared_lifespan
from exports.utils.logger import get_logger

from .routes.upload_route import router as upload_router
from .routes.search_route import router as search_router
from .routes.health_route import router as health_router
from .service_clients.media_processor_client import MediaProcessorClient
from .service_clients.embedder_client import EmbedderClient
from .service_clients.indexer_client import IndexerClient

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """API-specific lifespan.

    Wraps the shared MinIO/Supabase lifespan and additionally owns one
    long-lived HTTP client per downstream service, exposed on `app.state`:

        app.state.media_processor : MediaProcessorClient
        app.state.embedder        : EmbedderClient
        app.state.indexer         : IndexerClient

    Routes pull these out of `request.app.state` instead of opening a
    fresh `httpx.AsyncClient` per request.
    """
    async with shared_lifespan(app):
        logger.info("Connecting downstream service clients...")
        media_processor = await MediaProcessorClient().connect()
        embedder = await EmbedderClient().connect()
        indexer = await IndexerClient().connect()

        app.state.media_processor = media_processor
        app.state.embedder = embedder
        app.state.indexer = indexer
        logger.info("✅ Downstream service clients ready.")

        try:
            yield
        finally:
            logger.info("Closing downstream service clients...")
            await media_processor.close()
            await embedder.close()
            await indexer.close()
            app.state.media_processor = None
            app.state.embedder = None
            app.state.indexer = None
            logger.info("✅ Downstream service clients closed.")


# -------------------------------
# App Initialization
# -------------------------------
app = FastAPI(
    title="Mefid API Gateway",
    description=(
        "HTTP gateway for Mefid, a personal multimodal scene-search engine. "
        "Handles media uploads and multimodal search over the user's own "
        "video and image collection."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# -------------------------------
# Middleware
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in dev, open CORS; tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Routers
# -------------------------------
app.include_router(upload_router, prefix="/upload", tags=["Upload"])
app.include_router(search_router, prefix="/search", tags=["Search"])
app.include_router(health_router, tags=["Health"])


# -------------------------------
# Root Route
# -------------------------------
@app.get("/")
async def root():
    return {"message": "Welcome to the Mefid API Gateway!"}


# -------------------------------
# Run (optional if using uvicorn CLI)
# -------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_app:app", host="0.0.0.0", port=8000, reload=True)
