"""Async Supabase client for Mefid.

Tables (see `dev_schema.sql`):
    media          — one row per uploaded video/image
    frames         — one row per extracted frame, with MinIO URL
    embeddings     — one row per frame or transcript embedding; vectors live in FAISS
    transcripts    — Whisper transcript segments
    captions       — visual caption windows (uniform 1fps pass; not CLIP frames)
    search_queries — query log (defined; not yet wired)

All UUIDs are passed/stored as strings (`mode="json"` on dump). Enums
serialize to their string values for the same reason.
"""

from typing import List, Optional, TypeVar, Union
from uuid import UUID

from pydantic import BaseModel
from supabase import AsyncClient

from exports.schema.models import (
    CaptionCreate,
    CaptionRow,
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
ModelT = TypeVar("ModelT", bound=BaseModel)


def _parse_row(model: type[ModelT], data: object) -> ModelT:
    """Build a row model from a Supabase JSON record (typed as object for Pyright)."""
    return model.model_validate(data)


def _parse_rows(model: type[ModelT], data: object) -> List[ModelT]:
    if not data:
        return []
    return [model.model_validate(item) for item in data]  # type: ignore[union-attr]


def _id(value: IdLike) -> str:
    """Coerce a UUID/str id into the string form Supabase wants."""
    return str(value)


class SupabaseDB:
    def __init__(self, supabase_client: AsyncClient):
        self.client = supabase_client

    # ===================== Media =====================
    async def get_all_media(self) -> List[MediaRow]:
        response = await self.client.table("media").select("*").execute()
        return _parse_rows(MediaRow, response.data)

    async def get_media_by_id(self, media_id: IdLike) -> Optional[MediaRow]:
        response = (
            await self.client.table("media")
            .select("*")
            .eq("id", _id(media_id))
            .execute()
        )
        return _parse_row(MediaRow, response.data[0]) if response.data else None

    async def get_media_by_ids(self, media_ids: List[IdLike]) -> List[MediaRow]:
        if not media_ids:
            return []
        ids = [_id(m) for m in media_ids]
        response = await self.client.table("media").select("*").in_("id", ids).execute()
        return _parse_rows(MediaRow, response.data)

    async def insert_media(self, media: MediaCreate) -> MediaRow:
        payload = media.model_dump(mode="json", exclude_none=True)
        response = await self.client.table("media").insert(payload).execute()
        return _parse_row(MediaRow, response.data[0])

    async def update_media(self, media_id: IdLike, update: MediaUpdate) -> MediaRow:
        payload = update.model_dump(mode="json", exclude_none=True)
        response = (
            await self.client.table("media")
            .update(payload)
            .eq("id", _id(media_id))
            .execute()
        )
        return _parse_row(MediaRow, response.data[0])

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
        return _parse_rows(FrameRow, response.data)

    async def get_frame_by_id(self, frame_id: IdLike) -> Optional[FrameRow]:
        response = (
            await self.client.table("frames")
            .select("*")
            .eq("id", _id(frame_id))
            .execute()
        )
        return _parse_row(FrameRow, response.data[0]) if response.data else None

    async def get_frames_by_ids(self, frame_ids: List[IdLike]) -> List[FrameRow]:
        if not frame_ids:
            return []
        ids = [_id(f) for f in frame_ids]
        response = await self.client.table("frames").select("*").in_("id", ids).execute()
        return _parse_rows(FrameRow, response.data)

    async def insert_frame(self, frame: FrameCreate) -> FrameRow:
        payload = frame.model_dump(mode="json", exclude_none=True)
        response = await self.client.table("frames").insert(payload).execute()
        return _parse_row(FrameRow, response.data[0])

    async def insert_frames_batch(self, frames: List[FrameCreate]) -> List[FrameRow]:
        if not frames:
            return []
        payload = [f.model_dump(mode="json", exclude_none=True) for f in frames]
        response = await self.client.table("frames").insert(payload).execute()
        return _parse_rows(FrameRow, response.data)

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
        return _parse_row(EmbeddingRow, response.data[0]) if response.data else None

    async def get_embeddings_by_frame_id(self, frame_id: IdLike) -> List[EmbeddingRow]:
        response = (
            await self.client.table("embeddings")
            .select("*")
            .eq("frame_id", _id(frame_id))
            .execute()
        )
        return _parse_rows(EmbeddingRow, response.data)

    async def get_embedding_by_faiss_index_id(
        self, faiss_index_id: int
    ) -> Optional[EmbeddingRow]:
        response = (
            await self.client.table("embeddings")
            .select("*")
            .eq("faiss_index_id", faiss_index_id)
            .execute()
        )
        return _parse_row(EmbeddingRow, response.data[0]) if response.data else None

    async def get_embeddings_by_faiss_index_ids(
        self, faiss_index_ids: List[int]
    ) -> List[EmbeddingRow]:
        if not faiss_index_ids:
            return []
        response = (
            await self.client.table("embeddings")
            .select("*")
            .in_("faiss_index_id", faiss_index_ids)
            .execute()
        )
        return _parse_rows(EmbeddingRow, response.data)

    async def get_embeddings_by_transcript_id(
        self, transcript_id: IdLike
    ) -> List[EmbeddingRow]:
        response = (
            await self.client.table("embeddings")
            .select("*")
            .eq("transcript_id", _id(transcript_id))
            .execute()
        )
        return _parse_rows(EmbeddingRow, response.data)

    async def insert_embedding(self, embedding: EmbeddingCreate) -> EmbeddingRow:
        payload = embedding.model_dump(mode="json", exclude_none=True)
        response = await self.client.table("embeddings").insert(payload).execute()
        return _parse_row(EmbeddingRow, response.data[0])

    async def insert_embeddings_batch(
        self, embeddings: List[EmbeddingCreate]
    ) -> List[EmbeddingRow]:
        if not embeddings:
            return []
        payload = [e.model_dump(mode="json", exclude_none=True) for e in embeddings]
        response = await self.client.table("embeddings").insert(payload).execute()
        return _parse_rows(EmbeddingRow, response.data)

    async def delete_embedding(self, embedding_id: IdLike) -> None:
        await (
            self.client.table("embeddings")
            .delete()
            .eq("id", _id(embedding_id))
            .execute()
        )

    # ===================== Transcripts =====================
    async def get_transcript_by_id(self, transcript_id: IdLike) -> Optional[TranscriptRow]:
        response = (
            await self.client.table("transcripts")
            .select("*")
            .eq("id", _id(transcript_id))
            .execute()
        )
        return _parse_row(TranscriptRow, response.data[0]) if response.data else None

    async def get_transcripts_by_ids(
        self, transcript_ids: List[IdLike]
    ) -> List[TranscriptRow]:
        if not transcript_ids:
            return []
        ids = [_id(t) for t in transcript_ids]
        response = (
            await self.client.table("transcripts").select("*").in_("id", ids).execute()
        )
        return _parse_rows(TranscriptRow, response.data)

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
        return _parse_rows(TranscriptRow, response.data)

    async def insert_transcript(self, transcript: TranscriptCreate) -> TranscriptRow:
        payload = transcript.model_dump(mode="json")
        response = await self.client.table("transcripts").insert(payload).execute()
        return _parse_row(TranscriptRow, response.data[0])

    async def insert_transcripts_batch(
        self, transcripts: List[TranscriptCreate]
    ) -> List[TranscriptRow]:
        if not transcripts:
            return []
        payload = [t.model_dump(mode="json") for t in transcripts]
        response = await self.client.table("transcripts").insert(payload).execute()
        return _parse_rows(TranscriptRow, response.data)

    # ===================== Captions =====================
    async def get_caption_by_id(self, caption_id: IdLike) -> Optional[CaptionRow]:
        response = (
            await self.client.table("captions")
            .select("*")
            .eq("id", _id(caption_id))
            .execute()
        )
        return _parse_row(CaptionRow, response.data[0]) if response.data else None

    async def get_captions_by_ids(self, caption_ids: List[IdLike]) -> List[CaptionRow]:
        if not caption_ids:
            return []
        ids = [_id(c) for c in caption_ids]
        response = (
            await self.client.table("captions").select("*").in_("id", ids).execute()
        )
        return _parse_rows(CaptionRow, response.data)

    async def get_captions_by_media_id(self, media_id: IdLike) -> List[CaptionRow]:
        response = (
            await self.client.table("captions")
            .select("*")
            .eq("media_id", _id(media_id))
            .order("start_time")
            .execute()
        )
        return _parse_rows(CaptionRow, response.data)

    async def insert_caption(self, caption: CaptionCreate) -> CaptionRow:
        payload = caption.model_dump(mode="json")
        response = await self.client.table("captions").insert(payload).execute()
        return _parse_row(CaptionRow, response.data[0])

    async def insert_captions_batch(
        self, captions: List[CaptionCreate]
    ) -> List[CaptionRow]:
        if not captions:
            return []
        payload = [c.model_dump(mode="json") for c in captions]
        response = await self.client.table("captions").insert(payload).execute()
        return _parse_rows(CaptionRow, response.data)

    # ===================== Search Queries (model defined; not yet wired) =====================
    async def insert_search_query(self, query: SearchQueryCreate) -> SearchQueryRow:
        payload = query.model_dump(mode="json", exclude_none=True)
        response = await self.client.table("search_queries").insert(payload).execute()
        return _parse_row(SearchQueryRow, response.data[0])

    # ===================== Cleanup =====================
    async def close(self):
        """Best-effort shutdown for Supabase resources owned by this wrapper."""
        await self.client.remove_all_channels()

        # Supabase AsyncClient has no close/aclose API.
        # If a custom httpx client was injected via ClientOptions, close it here.
        httpx_client = getattr(self.client.options, "httpx_client", None)
        if httpx_client is not None and hasattr(httpx_client, "aclose"):
            await httpx_client.aclose()
