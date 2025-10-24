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

app.include_router(health_router)