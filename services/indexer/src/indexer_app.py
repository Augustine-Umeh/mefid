from fastapi import FastAPI
from exports.db_clients.lifespan import lifespan
from .routes.add import router as add_router
from .routes.search import router as search_router

app = FastAPI(
    title="Mefid Indexer Service",
    description="Indexes media metadata for efficient retrieval in the Mefid pipeline.",
    version="0.1.0",
    lifespan=lifespan 
)

# -------------------------------
# Routers
# -------------------------------
app.include_router(add_router, prefix="/add", tags=["Add"])
app.include_router(search_router, prefix="/search", tags=["Search"])

# -------------------------------
# Root Route
# -------------------------------
@app.get("/")
async def root():
    return {"message": "Mefid Indexer Service is running."}