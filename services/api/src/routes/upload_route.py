from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from exports.db_clients.minioDB import MinioDB
from exports.db_clients.supabaseDB import SupabaseDB
from exports.schema.constants import (
    FRAME_INTERVAL,
    SCENE_THRESHOLD,
    MediaStatus,
    VectorType,
)
from exports.schema.models import (
    AddVectorItem,
    EmbedImageItem,
    MediaUpdate,
    UploadImageRequest,
    UploadVideoRequest,
)
from exports.utils.logger import get_logger

from src.schema.responses import UploadResponse
from ..service_clients.embedder_client import EmbedderClient
from ..service_clients.indexer_client import IndexerClient
from ..service_clients.media_processor_client import MediaProcessorClient

router = APIRouter()
logger = get_logger()


@router.post("/image", response_model=UploadResponse)
async def upload_image(
    request: Request,
    image_query: UploadFile = File(...),
    title: Optional[str] = Form(None),
    filename: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    source_url: Optional[str] = Form(None),
) -> UploadResponse:
    """Stream an image upload to MinIO.

    Image uploads currently don't insert a `media` row — they're just
    transient assets used for search-by-image queries later.
    """
    upload_data = UploadImageRequest(
        image_query=image_query,
        title=title,
        filename=filename,
        description=description,
        source_url=source_url,
    )

    object_name = upload_data.image_query.filename
    content_type = upload_data.image_query.content_type or "application/octet-stream"

    try:
        logger.info(f"Uploading {object_name} to MinIO...")
        minio_db: MinioDB = request.app.state.minio
        file_url = await minio_db.upload_file(
            object_name=object_name,
            file_data=upload_data.image_query.file,
            content_type=content_type,
        )

        logger.info(f"Uploaded successfully: {file_url}")
        return UploadResponse(file_url=file_url)

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video", response_model=UploadResponse)
async def upload_video(
    request: Request,
    video_query: UploadFile = File(...),
    title: Optional[str] = Form(None),
    filename: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    duration_seconds: Optional[float] = Form(None),
    source_url: Optional[str] = Form(None),
    extraction_strategy: Optional[str] = Form("fixed_interval"),
) -> UploadResponse:
    """Stream a video upload to MinIO, then extract → embed → index.

    On success the `media` row is flipped from ``processing`` → ``ready``.
    On any post-extraction failure we try to flip it to ``failed`` so
    pipeline state matches reality.
    """
    upload_data = UploadVideoRequest(
        video_query=video_query,
        title=title,
        filename=filename,
        description=description,
        duration_seconds=duration_seconds,
        source_url=source_url,
    )

    object_name = upload_data.video_query.filename
    content_type = upload_data.video_query.content_type or "application/octet-stream"

    minio_db: MinioDB = request.app.state.minio
    supabase: SupabaseDB = request.app.state.supabase
    media_processor: MediaProcessorClient = request.app.state.media_processor
    embedder: EmbedderClient = request.app.state.embedder
    indexer: IndexerClient = request.app.state.indexer

    media_id: str | None = None

    try:
        # ---- Stream the source video to MinIO --------------------------
        file_url = await minio_db.upload_file(
            object_name=object_name,
            file_data=upload_data.video_query.file,
            content_type=content_type,
        )

        # ---- Extract frames (also inserts media + frames rows) ---------
        if extraction_strategy == "fixed_interval":
            extracted = await media_processor.extract_frames_fixed_interval(
                video_object_key=object_name,
                file_url=file_url,
                file_name=object_name,
                interval_seconds=FRAME_INTERVAL,
            )
        elif extraction_strategy == "scene_detect":
            extracted = await media_processor.extract_frames_scene_detect(
                video_object_key=object_name,
                file_url=file_url,
                file_name=object_name,
                threshold=SCENE_THRESHOLD,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported extraction strategy. "
                    "Use 'fixed_interval' or 'scene_detect'."
                ),
            )

        media_id = str(extracted.media_id)
        logger.info(
            f"Extracted {extracted.frame_count} frames "
            f"(media_id={media_id}, duration={extracted.duration:.2f}s)"
        )

        # ---- Embed frames ----------------------------------------------
        embedder_input = [
            EmbedImageItem(frame_id=f.frame_id, frame_data=f.frame_data)
            for f in extracted.frames
        ]
        embeddings = await embedder.embed_images(embedder_input)
        logger.info(f"Generated {len(embeddings)} embeddings.")

        # ---- Index vectors (indexer also writes embeddings rows) -------
        index_input = [
            AddVectorItem(
                frame_id=e.frame_id,
                embedding=e.embedding,
                vector_type=VectorType.IMAGE,
            )
            for e in embeddings
        ]
        index_result = await indexer.add_vectors(
            media_id=extracted.media_id,
            vectors=index_input,
        )
        logger.info(
            f"Indexed {index_result.count} vectors for media_id={media_id}"
        )

        # ---- Flip status to ready --------------------------------------
        await supabase.update_media(
            media_id, MediaUpdate(status=MediaStatus.READY)
        )

        return UploadResponse(file_url=file_url)

    except HTTPException:
        await _try_mark_failed(supabase, media_id)
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        await _try_mark_failed(supabase, media_id)
        raise HTTPException(status_code=500, detail=str(e))


async def _try_mark_failed(supabase: SupabaseDB, media_id: str | None) -> None:
    if media_id is None:
        return
    try:
        await supabase.update_media(
            media_id, MediaUpdate(status=MediaStatus.FAILED)
        )
    except Exception:
        logger.exception(f"Failed to mark media {media_id} as 'failed'")
