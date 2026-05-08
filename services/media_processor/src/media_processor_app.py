from fastapi import FastAPI
from exports.db_clients.lifespan import lifespan
from .routes.extract import router as extract_router

app = FastAPI(
    title="Mefid Media Processor",
    description="Processes user-uploaded media (videos, images) for Mefid — extracts frames so they can be embedded and searched as personal scene memories.",
    version="0.1.0",
    lifespan=lifespan 
)

# -------------------------------
# Routers
# -------------------------------
app.include_router(extract_router, prefix="/extract", tags=["Extract"])


# -------------------------------
# Root Route
# -------------------------------
@app.get("/")
async def upload_media():
    return {"message": "Mefid Media Processor is running."}
