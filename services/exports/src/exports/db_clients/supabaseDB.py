"""Async Supabase client for Mefid.

Tables (see `dev_schema.sql`):
    media          — one row per uploaded video/image
    frames         — one row per extracted frame, with MinIO URL
    embeddings     — one row per frame embedding; vectors live in FAISS
    transcripts    — Whisper transcript segments (defined; not yet wired)
    search_queries — query log (defined; not yet wired)

All UUIDs are passed/stored as strings (`mode="json"` on dump). Enums
serialize to their string values for the same reason.
"""

from typing import List, Optional, Union
from uuid import UUID

from supabase import AsyncClient

from exports.schema.models import (
    EmbeddingCreate,
    EmbeddingRow,
    FrameCreate,
    FrameRow,
    MediaCreate,
    MediaRow,
    MediaUpdate,
    SearchQueryCreate,
    SearchQueryRow,
    TranscriptCreate,
    TranscriptRow,
)


IdLike = Union[str, UUID]


def _id(value: IdLike) -> str:
    """Coerce a UUID/str id into the string form Supabase wants."""
    return str(value)


class SupabaseDB:
    def __init__(self, supabase_client: AsyncClient):
        self.client = supabase_client

    # ===================== Media =====================
    async def get_all_media(self) -> List[MediaRow]:
        response = await self.client.table("media").select("*").execute()
        return [MediaRow(**m) for m in response.data]

    async def get_media_by_id(self, media_id: IdLike) -> Optional[MediaRow]:
        response = (
            await self.client.table("media")
            .select("*")
            .eq("id", _id(media_id))
            .execute()
        )
        return MediaRow(**response.data[0]) if response.data else None

    async def insert_media(self, media: MediaCreate) -> MediaRow:
        payload = media.model_dump(mode="json", exclude_none=True)
        response = await self.client.table("media").insert(payload).execute()
        return MediaRow(**response.data[0])

    async def update_media(self, media_id: IdLike, update: MediaUpdate) -> MediaRow:
        payload = update.model_dump(mode="json", exclude_none=True)
        response = (
            await self.client.table("media")
            .update(payload)
            .eq("id", _id(media_id))
            .execute()
        )
        return MediaRow(**response.data[0])

    async def delete_media(self, media_id: IdLike) -> None:
        await self.client.table("media").delete().eq("id", _id(media_id)).execute()

    # ===================== Frames =====================
    async def get_frames_by_media_id(self, media_id: IdLike) -> List[FrameRow]:
        response = (
            await self.client.table("frames")
            .select("*")
            .eq("media_id", _id(media_id))
            .order("sequence_number")
            .execute()
        )
        return [FrameRow(**f) for f in response.data]

    async def get_frame_by_id(self, frame_id: IdLike) -> Optional[FrameRow]:
        response = (
            await self.client.table("frames")
            .select("*")
            .eq("id", _id(frame_id))
            .execute()
        )
        return FrameRow(**response.data[0]) if response.data else None

    async def insert_frame(self, frame: FrameCreate) -> FrameRow:
        payload = frame.model_dump(mode="json", exclude_none=True)
        response = await self.client.table("frames").insert(payload).execute()
        return FrameRow(**response.data[0])

    async def insert_frames_batch(self, frames: List[FrameCreate]) -> List[FrameRow]:
        if not frames:
            return []
        payload = [f.model_dump(mode="json", exclude_none=True) for f in frames]
        response = await self.client.table("frames").insert(payload).execute()
        return [FrameRow(**f) for f in response.data]

    async def delete_frame(self, frame_id: IdLike) -> None:
        await self.client.table("frames").delete().eq("id", _id(frame_id)).execute()

    # ===================== Embeddings =====================
    async def get_embedding_by_id(self, embedding_id: IdLike) -> Optional[EmbeddingRow]:
        response = (
            await self.client.table("embeddings")
            .select("*")
            .eq("id", _id(embedding_id))
            .execute()
        )
        return EmbeddingRow(**response.data[0]) if response.data else None

    async def get_embeddings_by_frame_id(self, frame_id: IdLike) -> List[EmbeddingRow]:
        response = (
            await self.client.table("embeddings")
            .select("*")
            .eq("frame_id", _id(frame_id))
            .execute()
        )
        return [EmbeddingRow(**e) for e in response.data]

    async def get_embedding_by_faiss_index_id(
        self, faiss_index_id: int
    ) -> Optional[EmbeddingRow]:
        response = (
            await self.client.table("embeddings")
            .select("*")
            .eq("faiss_index_id", faiss_index_id)
            .execute()
        )
        return EmbeddingRow(**response.data[0]) if response.data else None

    async def insert_embedding(self, embedding: EmbeddingCreate) -> EmbeddingRow:
        payload = embedding.model_dump(mode="json")
        response = await self.client.table("embeddings").insert(payload).execute()
        return EmbeddingRow(**response.data[0])

    async def insert_embeddings_batch(
        self, embeddings: List[EmbeddingCreate]
    ) -> List[EmbeddingRow]:
        if not embeddings:
            return []
        payload = [e.model_dump(mode="json") for e in embeddings]
        response = await self.client.table("embeddings").insert(payload).execute()
        return [EmbeddingRow(**e) for e in response.data]

    async def delete_embedding(self, embedding_id: IdLike) -> None:
        await (
            self.client.table("embeddings")
            .delete()
            .eq("id", _id(embedding_id))
            .execute()
        )

    # ===================== Transcripts (defined; not yet wired) =====================
    async def get_transcripts_by_media_id(
        self, media_id: IdLike
    ) -> List[TranscriptRow]:
        response = (
            await self.client.table("transcripts")
            .select("*")
            .eq("media_id", _id(media_id))
            .order("start_time")
            .execute()
        )
        return [TranscriptRow(**t) for t in response.data]

    async def insert_transcript(self, transcript: TranscriptCreate) -> TranscriptRow:
        payload = transcript.model_dump(mode="json")
        response = await self.client.table("transcripts").insert(payload).execute()
        return TranscriptRow(**response.data[0])

    async def insert_transcripts_batch(
        self, transcripts: List[TranscriptCreate]
    ) -> List[TranscriptRow]:
        if not transcripts:
            return []
        payload = [t.model_dump(mode="json") for t in transcripts]
        response = await self.client.table("transcripts").insert(payload).execute()
        return [TranscriptRow(**t) for t in response.data]

    # ===================== Search Queries (model defined; not yet wired) =====================
    async def insert_search_query(self, query: SearchQueryCreate) -> SearchQueryRow:
        payload = query.model_dump(mode="json", exclude_none=True)
        response = await self.client.table("search_queries").insert(payload).execute()
        return SearchQueryRow(**response.data[0])

    # ===================== Cleanup =====================
    async def close(self):
        await self.client.aclose()
