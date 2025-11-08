from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.exports.src.db_clients.lifespan import lifespan
from services.api.src.routes.upload_route import router as upload_router
from services.api.src.routes.search_route import router as search_router
from services.api.src.routes.health_route import router as health_router

# -------------------------------
# App Initialization
# -------------------------------
app = FastAPI(
    title="Mefid API Gateway",
    description="Handles media uploads and multimodal search requests for the Mefid indexing and retrieval pipeline.",
    version="0.1.0",
    lifespan=lifespan  # <-- attach your lifespan here
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
