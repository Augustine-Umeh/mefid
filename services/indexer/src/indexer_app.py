from fastapi import FastAPI

from exports.db_clients.lifespan import lifespan
from .routes.add import router as add_router
from .routes.search import router as search_router

app = FastAPI(
    title="Mefid Indexer Service",
    description="Maintains a FAISS vector index over the user's uploaded media so Mefid can retrieve nearest scenes for multimodal search queries.",
    version="0.1.0",
    lifespan=lifespan,
)

# -------------------------------
# Routers
# -------------------------------
app.include_router(add_router, prefix="/vectors/add", tags=["Add Vectors"])
app.include_router(search_router, prefix="/vectors/search", tags=["Search Vectors"])

# -------------------------------
# Root Route
# -------------------------------
@app.get("/")
async def root():
    return {"message": "Mefid Indexer Service is running."}