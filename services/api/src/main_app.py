from contextlib import asynccontextmanager
from fastapi import FastAPI
from supabase import AsyncClient
from db.supabaseDB import SupabaseDB
from db.client import create_supabase


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncClient | None: # type: ignore
    """Create and close Supabase client connection."""
    print("Connecting to Supabase...")
    supabase_client: AsyncClient = await create_supabase()
    app.state.db = SupabaseDB(supabase_client)
    print("✅ Connected.")

    # Yield control — FastAPI starts handling requests here
    yield

    # Cleanup
    print("Closing connection...")
    app.state.db = None


app = FastAPI(lifespan=lifespan)


# -----------------------------
# Import and include routers
# -----------------------------
from routes.health_route import router as health_router
from routes.media_route import router as media_router
from routes.frames_route import router as frames_router
from routes.embeddings_route import router as embeddings_router
from routes.search_queries_route import router as search_queries_router
from routes.frame_metadata_route import router as frame_metadata_router

app.include_router(health_router)
app.include_router(media_router)
app.include_router(frames_router)
app.include_router(embeddings_router)
app.include_router(search_queries_router)
app.include_router(frame_metadata_router)