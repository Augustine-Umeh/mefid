"""Frame extraction endpoint.

Pipeline (per request):
    1. Download the source video from MinIO into a temp file.
    2. Run the chosen extractor → list of (timestamp, base64) frames + duration.
    3. Insert a `media` row (status=processing) so the rest of the pipeline
       has a stable id to attach to.
    4. Upload each extracted frame as a JPEG to MinIO under
       ``frames/<media_id>/<sequence>.jpg`` and insert one ``frames`` row
       per upload (batched).
    5. Return ``ExtractFramesResponse`` with both the persisted identity
       (``frame_id``, ``frame_url``) and the in-memory base64 payload
       so the embedder can read pixels without re-downloading from MinIO.

If anything fails after the media row is inserted, we mark it as
``failed`` before re-raising so the orchestrator (API) sees the state.
"""

import base64
import io
import os
import tempfile

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from exports.db_clients.minioDB import MinioDB
from exports.db_clients.supabaseDB import SupabaseDB
from exports.schema.constants import ExtractionStrategy, MediaStatus, MediaType
from exports.schema.models import (
    ExtractFramesResponse,
    FrameCreate,
    FrameData,
    MediaCreate,
    MediaUpdate,
)
from exports.utils.logger import get_logger

from ..extractors_techniques.hybrid import extract_frames_hybrid

router = APIRouter()
logger = get_logger()


class ExtractRequest(BaseModel):
    """Request body for ``POST /extract/``."""

    video_object_key: str  # MinIO object key of the source video
    file_url: str          # public/internal MinIO URL — stored on `media.file_url`
    file_name: str


def _frame_object_key(media_id: str, sequence_number: int) -> str:
    """Where in MinIO to store an extracted frame JPEG."""
    return f"frames/{media_id}/{sequence_number:04d}.jpg"


def _unpack_frame(
    frame_item: tuple[float, str] | tuple[float, str, str | None],
) -> tuple[float, str, str | None]:
    if len(frame_item) == 3:
        return frame_item[0], frame_item[1], frame_item[2]
    return frame_item[0], frame_item[1], None


@router.post("/", response_model=ExtractFramesResponse)
async def extract_frames(
    request: Request,
    extract_req: ExtractRequest,
    strategy: ExtractionStrategy = Query(...),
    threshold: int | None = Query(None),
):
    """Extract frames from a video and persist them."""
    if strategy in (ExtractionStrategy.FIXED_INTERVAL, ExtractionStrategy.SCENE_DETECT):
        raise HTTPException(
            status_code=400,
            detail=(
                f"extraction strategy '{strategy.value}' is no longer supported; "
                "use 'hybrid'."
            ),
        )
    if threshold is None:
        raise HTTPException(400, "threshold is required for hybrid")

    minio: MinioDB = request.app.state.minio
    supabase: SupabaseDB = request.app.state.supabase

    logger.info(
        f"Extracting frames from {extract_req.video_object_key} "
        f"(strategy={strategy.value})"
    )

    # ---- 1. Download video to a temp file ------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_path = temp_file.name
        temp_file.write(await minio.download_file(extract_req.video_object_key))

    media_id: str | None = None
    try:
        # ---- 2. Extract frames in-memory -------------------------------
        extraction = extract_frames_hybrid(
            video_path=temp_path,
            threshold=threshold,
        )

        duration = float(extraction["duration"])
        raw_frames = extraction["frames"]
        logger.info(f"Extracted {len(raw_frames)} frames; duration={duration:.2f}s")

        # ---- 3. Insert media row (processing) --------------------------
        media_row = await supabase.insert_media(
            MediaCreate(
                file_name=extract_req.file_name,
                file_url=extract_req.file_url,
                media_type=MediaType.VIDEO,
                duration=duration,
                extraction_strategy=strategy,
                status=MediaStatus.PROCESSING,
            )
        )
        media_id = str(media_row.id)
        logger.info(f"Inserted media row id={media_id}")

        # ---- 4. Upload each frame to MinIO + collect FrameCreate ------
        frame_creates: list[FrameCreate] = []
        frame_urls: list[str] = []
        for seq, frame_item in enumerate(raw_frames):
            timestamp, frame_b64, phash = _unpack_frame(frame_item)
            object_key = _frame_object_key(media_id, seq)
            jpeg_bytes = base64.b64decode(frame_b64)
            frame_url = await minio.upload_file(
                object_name=object_key,
                file_data=io.BytesIO(jpeg_bytes),
                content_type="image/jpeg",
            )
            frame_urls.append(frame_url)
            frame_creates.append(
                FrameCreate(
                    media_id=media_row.id,
                    timestamp=timestamp,
                    frame_url=frame_url,
                    sequence_number=seq,
                    phash=phash,
                )
            )

        # ---- 5. Insert frames rows in one batch ------------------------
        frame_rows = await supabase.insert_frames_batch(frame_creates)
        # `insert ... returning *` should return rows in insertion order,
        # but we pair via index against frame_creates to be explicit.
        if len(frame_rows) != len(raw_frames):
            raise RuntimeError(
                f"Frame insert mismatch: inserted {len(frame_rows)} rows "
                f"for {len(raw_frames)} frames"
            )

        # ---- 6. Build response ----------------------------------------
        response_frames = [
            FrameData(
                frame_id=row.id,
                sequence_number=row.sequence_number,
                timestamp=row.timestamp,
                frame_url=row.frame_url,
                frame_data=_unpack_frame(raw_frames[idx])[1],
            )
            for idx, row in enumerate(frame_rows)
        ]

        return ExtractFramesResponse(
            media_id=media_row.id,
            frames=response_frames,
            frame_count=len(response_frames),
            strategy=strategy,
            duration=duration,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}", exc_info=True)
        if media_id is not None:
            try:
                await supabase.update_media(
                    media_id, MediaUpdate(status=MediaStatus.FAILED)
                )
            except Exception:
                logger.exception("Failed to mark media row as 'failed'")
        raise HTTPException(500, str(e))
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
