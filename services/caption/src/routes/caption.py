"""Caption endpoint: download video, run Qwen2-VL pipeline, persist caption windows."""

import asyncio
import os
import tempfile

from fastapi import APIRouter, HTTPException, Request

from exports.db_clients.minioDB import MinioDB
from exports.db_clients.supabaseDB import SupabaseDB
from exports.schema.models import (
    CaptionCreate,
    CaptionRequest,
    CaptionResponse,
    CaptionSegmentData,
)
from exports.utils.logger import get_logger

from ..caption_engine import CaptionEngine

router = APIRouter()
logger = get_logger()


def _get_engine(request: Request) -> CaptionEngine:
    engine = getattr(request.app.state, "caption_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Caption engine not initialized")
    return engine


@router.post("/", response_model=CaptionResponse)
async def caption_video(request: Request, body: CaptionRequest) -> CaptionResponse:
    """Download a video from MinIO, caption it, and store time-bounded windows."""
    minio: MinioDB = request.app.state.minio
    supabase: SupabaseDB = request.app.state.supabase
    engine = _get_engine(request)

    logger.info(
        "caption media_id=%s object_key=%s",
        body.media_id,
        body.video_object_key,
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_path = temp_file.name
        temp_file.write(await minio.download_file(body.video_object_key))

    try:
        drafts = await asyncio.to_thread(
            engine.caption_video, temp_path, body.media_id
        )
        if not drafts:
            logger.info("No captions generated for media_id=%s", body.media_id)
            return CaptionResponse(
                media_id=body.media_id,
                segments=[],
                segment_count=0,
            )

        creates = [
            CaptionCreate(
                media_id=body.media_id,
                start_time=draft.start_time,
                end_time=draft.end_time,
                text=draft.text,
            )
            for draft in drafts
        ]
        rows = await supabase.insert_captions_batch(creates)
        segments = [
            CaptionSegmentData(
                id=row.id,
                start_time=row.start_time,
                end_time=row.end_time,
                text=row.text,
            )
            for row in rows
        ]
        logger.info(
            "Stored %s caption windows for media_id=%s",
            len(segments),
            body.media_id,
        )
        return CaptionResponse(
            media_id=body.media_id,
            segments=segments,
            segment_count=len(segments),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Captioning failed for media_id=%s: %s", body.media_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
