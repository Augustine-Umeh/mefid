"""Transcription endpoint: download video, run Whisper, persist transcript chunks."""

import asyncio
import os
import tempfile

from fastapi import APIRouter, HTTPException, Request

from exports.db_clients.minioDB import MinioDB
from exports.db_clients.supabaseDB import SupabaseDB
from exports.schema.models import (
    TranscribeRequest,
    TranscribeResponse,
    TranscriptCreate,
    TranscriptSegmentData,
)
from exports.utils.logger import get_logger

from ..whisper_service import WhisperEngine

router = APIRouter()
logger = get_logger()


def _get_engine(request: Request) -> WhisperEngine:
    engine = getattr(request.app.state, "whisper_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Whisper engine not initialized")
    return engine


@router.post("/", response_model=TranscribeResponse)
async def transcribe_video(
    request: Request, body: TranscribeRequest
) -> TranscribeResponse:
    """Download a video from MinIO, transcribe it, and store CLIP-safe chunks."""
    minio: MinioDB = request.app.state.minio
    supabase: SupabaseDB = request.app.state.supabase
    engine = _get_engine(request)

    logger.info(
        "transcribe media_id=%s object_key=%s",
        body.media_id,
        body.video_object_key,
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_path = temp_file.name
        temp_file.write(await minio.download_file(body.video_object_key))

    try:
        chunks = await asyncio.to_thread(engine.transcribe_to_chunks, temp_path)
        if not chunks:
            logger.info("No speech detected for media_id=%s", body.media_id)
            return TranscribeResponse(
                media_id=body.media_id,
                segments=[],
                segment_count=0,
            )

        creates = [
            TranscriptCreate(
                media_id=body.media_id,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                text=chunk.text,
            )
            for chunk in chunks
        ]
        rows = await supabase.insert_transcripts_batch(creates)
        segments = [
            TranscriptSegmentData(
                id=row.id,
                start_time=row.start_time,
                end_time=row.end_time,
                text=row.text,
            )
            for row in rows
        ]
        logger.info(
            "Stored %s transcript chunks for media_id=%s",
            len(segments),
            body.media_id,
        )
        return TranscribeResponse(
            media_id=body.media_id,
            segments=segments,
            segment_count=len(segments),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Transcription failed for media_id=%s: %s", body.media_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
