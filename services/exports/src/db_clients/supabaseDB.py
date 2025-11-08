from typing import List, Optional
from supabase import AsyncClient
from services.exports.src.schema.models import (
    Media, MediaCreate,
    Frame, FrameCreate,
    FrameMetadata, FrameMetadataCreate,
    SearchQuery, SearchQueryCreate,
    Embedding, EmbeddingCreate,
    Indexes, IndexesCreate,
)


class SupabaseDB:
    def __init__(self, supabase_client: AsyncClient):
        self.client = supabase_client

    # ===================== Media =====================
    async def get_all_media(self) -> List[Media]:
        response = await self.client.table("media").select("*").execute()
        return [Media(**m) for m in response.data]

    async def get_media_by_id(self, media_id: int) -> Optional[Media]:
        response = await self.client.table("media").select("*").eq("id", media_id).execute()
        return Media(**response.data[0]) if response.data else None

    async def insert_media(self, media_data: MediaCreate) -> Media:
        response = await self.client.table("media").insert(media_data.model_dump()).execute()
        return Media(**response.data[0])

    async def update_media(self, media_id: int, update_data: Media) -> Media:
        response = await self.client.table("media").update(update_data.to_dict()).eq("id", media_id).execute()
        return Media(**response.data[0])

    async def delete_media(self, media_id: int) -> None:
        await self.client.table("media").delete().eq("id", media_id).execute()

    # ===================== Frame =====================
    async def get_all_frames(self) -> List[Frame]:
        response = await self.client.table("frames").select("*").execute()
        return [Frame(**f) for f in response.data]

    async def get_frames_by_media_id(self, media_id: int) -> List[Frame]:
        response = await self.client.table("frames").select("*").eq("media_id", media_id).execute()
        return [Frame(**f) for f in response.data]

    async def get_frame_by_id(self, frame_id: int) -> Optional[Frame]:
        response = await self.client.table("frames").select("*").eq("id", frame_id).execute()
        return Frame(**response.data[0]) if response.data else None

    async def insert_frame(self, frame_data: FrameCreate) -> Frame:
        response = await self.client.table("frames").insert(frame_data.model_dump()).execute()
        return Frame(**response.data[0])

    async def update_frame(self, frame_id: int, update_data: Frame) -> Frame:
        response = await self.client.table("frames").update(update_data.to_dict()).eq("id", frame_id).execute()
        return Frame(**response.data[0])

    async def delete_frame(self, frame_id: int) -> None:
        await self.client.table("frames").delete().eq("id", frame_id).execute()

    # ===================== FrameMetadata =====================
    async def get_all_frame_metadata(self) -> List[FrameMetadata]:
        response = await self.client.table("frame_metadata").select("*").execute()
        return [FrameMetadata(**fm) for fm in response.data]

    async def get_frame_metadata_by_frame_id(self, frame_id: int) -> Optional[FrameMetadata]:
        response = await self.client.table("frame_metadata").select("*").eq("frame_id", frame_id).execute()
        return FrameMetadata(**response.data[0]) if response.data else None

    async def get_frame_metadata_by_id(self, metadata_id: int) -> Optional[FrameMetadata]:
        response = await self.client.table("frame_metadata").select("*").eq("id", metadata_id).execute()
        return FrameMetadata(**response.data[0]) if response.data else None

    async def insert_frame_metadata(self, metadata_data: FrameMetadataCreate) -> FrameMetadata:
        response = await self.client.table("frame_metadata").insert(metadata_data.model_dump()).execute()
        return FrameMetadata(**response.data[0])

    async def update_frame_metadata(self, metadata_id: int, update_data: FrameMetadata) -> FrameMetadata:
        response = await self.client.table("frame_metadata").update(update_data.to_dict()).eq("id", metadata_id).execute()
        return FrameMetadata(**response.data[0])

    async def delete_frame_metadata(self, metadata_id: int) -> None:
        await self.client.table("frame_metadata").delete().eq("id", metadata_id).execute()

    # ===================== SearchQuery =====================
    async def get_all_search_queries(self) -> List[SearchQuery]:
        response = await self.client.table("search_queries").select("*").execute()
        return [SearchQuery(**q) for q in response.data]

    async def get_search_query_by_id(self, query_id: int) -> Optional[SearchQuery]:
        response = await self.client.table("search_queries").select("*").eq("id", query_id).execute()
        return SearchQuery(**response.data[0]) if response.data else None

    async def insert_search_query(self, query_data: SearchQueryCreate) -> SearchQuery:
        response = await self.client.table("search_queries").insert(query_data.model_dump()).execute()
        return SearchQuery(**response.data[0])

    async def update_search_query(self, query_id: int, update_data: SearchQuery) -> SearchQuery:
        response = await self.client.table("search_queries").update(update_data.to_dict()).eq("id", query_id).execute()
        return SearchQuery(**response.data[0])

    async def delete_search_query(self, query_id: int) -> None:
        await self.client.table("search_queries").delete().eq("id", query_id).execute()

    # ===================== Embedding =====================
    async def get_all_embeddings(self) -> List[Embedding]:
        response = await self.client.table("embeddings").select("*").execute()
        return [Embedding(**e) for e in response.data]

    async def get_embedding_by_id(self, embedding_id: int) -> Optional[Embedding]:
        response = await self.client.table("embeddings").select("*").eq("id", embedding_id).execute()
        return Embedding(**response.data[0]) if response.data else None

    async def get_embeddings_by_frame_id(self, frame_id: int) -> List[Embedding]:
        response = await self.client.table("embeddings").select("*").eq("frame_id", frame_id).execute()
        return [Embedding(**e) for e in response.data]

    async def insert_embedding(self, embedding_data: EmbeddingCreate) -> Embedding:
        response = await self.client.table("embeddings").insert(embedding_data.model_dump()).execute()
        return Embedding(**response.data[0])

    async def update_embedding(self, embedding_id: int, update_data: Embedding) -> Embedding:
        response = await self.client.table("embeddings").update(update_data.to_dict()).eq("id", embedding_id).execute()
        return Embedding(**response.data[0])

    async def delete_embedding(self, embedding_id: int) -> None:
        await self.client.table("embeddings").delete().eq("id", embedding_id).execute()

    # ===================== Indexes =====================
    async def get_all_indexes(self) -> List[Indexes]:
        response = await self.client.table("indexes").select("*").execute()
        return [Indexes(**idx) for idx in response.data]
    
    async def get_index_by_id(self, index_id: int) -> Optional[Indexes]:
        response = await self.client.table("indexes").select("*").eq("id", index_id).execute()
        return Indexes(**response.data[0]) if response.data else None
    
    async def insert_index(self, index_data: IndexesCreate) -> Indexes:
        response = await self.client.table("indexes").insert(index_data.model_dump()).execute()
        return Indexes(**response.data[0])
    
    async def update_index(self, index_id: int, update_data: Indexes) -> Indexes:
        response = await self.client.table("indexes").update(update_data.to_dict()).eq("id", index_id).execute()
        return Indexes(**response.data[0])
    
    async def delete_index(self, index_id: int) -> None:
        await self.client.table("indexes").delete().eq("id", index_id).execute()
        
    # ===================== Batch Inserts =====================
    async def insert_frames_batch(self, frames_data: List[FrameCreate]) -> List[Frame]:
        """
        Insert multiple frames at once.
        """
        dicts = [f.model_dump() for f in frames_data]
        response = await self.client.table("frames").insert(dicts).execute()
        return [Frame(**f) for f in response.data]

    async def insert_embeddings_batch(self, embeddings_data: List[EmbeddingCreate]) -> List[Embedding]:
        """
        Insert multiple embeddings at once.
        """
        dicts = [e.model_dump() for e in embeddings_data]
        response = await self.client.table("embeddings").insert(dicts).execute()
        return [Embedding(**e) for e in response.data]

    async def insert_frame_metadata_batch(self, metadata_data: List[FrameMetadataCreate]) -> List[FrameMetadata]:
        """
        Insert multiple frame metadata entries at once.
        """
        dicts = [m.model_dump() for m in metadata_data]
        response = await self.client.table("frame_metadata").insert(dicts).execute()
        return [FrameMetadata(**m) for m in response.data]

    # ===================== Close Client =====================
    async def close(self):
        await self.client.aclose()