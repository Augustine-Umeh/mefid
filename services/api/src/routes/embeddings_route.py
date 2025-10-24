from fastapi import APIRouter
from main_app import app

router = APIRouter()

@router.get("/embeddings")
async def get_all_embeddings() -> dict:
    """Route to get all embedding entries."""
    embeddings_entries = await app.state.db.get_all_embeddings()
    return {"embeddings": embeddings_entries}

@router.get("/embeddings/{embedding_id}")
async def get_embedding_by_id(embedding_id: str) -> dict:
    """Route to get a specific embedding entry by its ID."""
    embedding_entry = await app.state.db.get_embedding_by_id(embedding_id)
    return {"embedding": embedding_entry}