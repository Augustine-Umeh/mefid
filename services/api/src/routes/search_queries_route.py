from fastapi import APIRouter
from main_app import app

router = APIRouter()

@router.get("/search_queries")
async def get_all_search_queries() -> dict:
    """Route to get all search query entries."""
    search_query_entries = await app.state.db.get_all_search_queries()
    return {"search_queries": search_query_entries}

@router.get("/search_queries/{query_id}")
async def get_search_query_by_id(query_id: str) -> dict:
    """Route to get a specific search query entry by its ID."""
    search_query_entry = await app.state.db.get_search_query_by_id(query_id)
    return {"search_query": search_query_entry}