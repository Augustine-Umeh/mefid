from fastapi import FastAPI
from exports.db_clients.lifespan import lifespan
from .routes.extract import router as extract_router

app = FastAPI(
    title="Mefid Media Processor",
    description="Processes media uploads for the Mefid indexing and retrieval pipeline.",
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
