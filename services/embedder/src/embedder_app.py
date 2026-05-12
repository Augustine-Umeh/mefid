from fastapi import FastAPI

from exports.db_clients.lifespan import lifespan
from .routes.embed_image import router as embed_image_router
from .routes.embed_text import router as embed_text_router

app = FastAPI(
    title="Mefid Embedder Service",
    description="Generates CLIP embeddings for frames and text queries so Mefid can search the user's personal video and image collection by content, tone, and emotion.",
    version="0.1.0",
    lifespan=lifespan 
)

# -------------------------------
# Routers
# -------------------------------
app.include_router(embed_image_router, prefix="/embed/images", tags=["Embed Image"])
app.include_router(embed_text_router, prefix="/embed/text", tags=["Embed Text"])

# -------------------------------
# Root Route
# -------------------------------
@app.get("/")
async def root():
    return {"message": "Mefid Embedder Service is running."}