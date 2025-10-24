from typing import List, Optional
from client import supabase_client
from models import Media, Frame, FrameMetadata, SearchQuery, Embedding


class SupabaseDB:
    def __init__(self):
        self.client = supabase_client
    
    # ========== Media CRUD operations ==========
    async def get_all_media(self) -> List[dict]:
        response = await self.client.table("media").select('*').execute()
        return response.data
    
    async def get_media_by_id(self, media_id: int) -> Optional[dict]:
        response = await self.client.table("media").select('*').eq('id', media_id).execute()
        return response.data[0] if response.data else None
    
    async def insert_media(self, media_data: Media) -> dict:
        response = await self.client.table("media").insert(media_data.to_dict()).execute()
        return response.data[0] if response.data else None
    
    async def update_media(self, media_id: int, update_data: dict) -> dict:
        response = await self.client.table("media").update(update_data).eq('id', media_id).execute()
        return response.data[0] if response.data else None
    
    async def delete_media(self, media_id: int) -> None:
        await self.client.table("media").delete().eq('id', media_id).execute()

    # ========== Frame CRUD operations ==========
    async def get_all_frames(self) -> List[dict]:
        response = await self.client.table("frames").select('*').execute()
        return response.data
    
    async def get_frames_by_media_id(self, media_id: int) -> List[dict]:
        response = await self.client.table("frames").select('*').eq('media_id', media_id).execute()
        return response.data
    
    async def get_frame_by_id(self, frame_id: int) -> Optional[dict]:
        response = await self.client.table("frames").select('*').eq('id', frame_id).execute()
        return response.data[0] if response.data else None
    
    async def insert_frame(self, frame_data: Frame) -> dict:
        response = await self.client.table("frames").insert(frame_data.to_dict()).execute()
        return response.data[0] if response.data else None
    
    async def update_frame(self, frame_id: int, update_data: dict) -> dict:
        response = await self.client.table("frames").update(update_data).eq('id', frame_id).execute()
        return response.data[0] if response.data else None
    
    async def delete_frame(self, frame_id: int) -> None:
        await self.client.table("frames").delete().eq('id', frame_id).execute()
    
    # ========== FrameMetadata CRUD operations ==========
    async def get_all_frame_metadata(self) -> List[dict]:
        response = await self.client.table("frame_metadata").select('*').execute()
        return response.data
    
    async def get_frame_metadata_by_frame_id(self, frame_id: int) -> Optional[dict]:
        response = await self.client.table("frame_metadata").select('*').eq('frame_id', frame_id).execute()
        return response.data[0] if response.data else None
    
    async def get_frame_metadata_by_id(self, metadata_id: int) -> Optional[dict]:
        response = await self.client.table("frame_metadata").select('*').eq('id', metadata_id).execute()
        return response.data[0] if response.data else None
    
    async def insert_frame_metadata(self, metadata_data: FrameMetadata) -> dict:
        response = await self.client.table("frame_metadata").insert(metadata_data.to_dict()).execute()
        return response.data[0] if response.data else None
    
    async def update_frame_metadata(self, metadata_id: int, update_data: dict) -> dict:
        response = await self.client.table("frame_metadata").update(update_data).eq('id', metadata_id).execute()
        return response.data[0] if response.data else None
    
    async def delete_frame_metadata(self, metadata_id: int) -> None:
        await self.client.table("frame_metadata").delete().eq('id', metadata_id).execute()
    
    # ========== SearchQuery CRUD operations ==========
    async def get_all_search_queries(self) -> List[dict]:
        response = await self.client.table("search_queries").select('*').execute()
        return response.data
    
    async def get_search_query_by_id(self, query_id: int) -> Optional[dict]:
        response = await self.client.table("search_queries").select('*').eq('id', query_id).execute()
        return response.data[0] if response.data else None
    
    async def insert_search_query(self, query_data: SearchQuery) -> dict:
        response = await self.client.table("search_queries").insert(query_data.to_dict()).execute()
        return response.data[0] if response.data else None
    
    async def update_search_query(self, query_id: int, update_data: dict) -> dict:
        response = await self.client.table("search_queries").update(update_data).eq('id', query_id).execute()
        return response.data[0] if response.data else None
    
    async def delete_search_query(self, query_id: int) -> None:
        await self.client.table("search_queries").delete().eq('id', query_id).execute()
    
    # ========== Embedding CRUD operations ==========
    async def get_all_embeddings(self) -> List[dict]:
        response = await self.client.table("embeddings").select('*').execute()
        return response.data
    
    async def get_embedding_by_id(self, embedding_id: int) -> Optional[dict]:
        response = await self.client.table("embeddings").select('*').eq('id', embedding_id).execute()
        return response.data[0] if response.data else None
    
    async def get_embeddings_by_frame_id(self, frame_id: int) -> List[dict]:
        response = await self.client.table("embeddings").select('*').eq('frame_id', frame_id).execute()
        return response.data
    
    async def insert_embedding(self, embedding_data: Embedding) -> dict:
        response = await self.client.table("embeddings").insert(embedding_data.to_dict()).execute()
        return response.data[0] if response.data else None
    
    async def update_embedding(self, embedding_id: int, update_data: dict) -> dict:
        response = await self.client.table("embeddings").update(update_data).eq('id', embedding_id).execute()
        return response.data[0] if response.data else None
    
    async def delete_embedding(self, embedding_id: int) -> None:
        await self.client.table("embeddings").delete().eq('id', embedding_id).execute()