from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from exports.db_clients.minioDB import MinioDB
from exports.schema.constants import FRAME_INTERVAL, SCENE_THRESHOLD
from exports.schema.models import UploadImageRequest, UploadVideoRequest
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
    """Stream an image upload to MinIO."""
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
        minio_object_url = await minio_db.upload_file(
            object_name=object_name,
            file_data=upload_data.image_query.file,
            content_type=content_type,
        )

        logger.info(f"Uploaded successfully: {minio_object_url}")
        return UploadResponse(file_url=minio_object_url)

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
    """Stream a video upload to MinIO, then extract → embed → index."""
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

    try:
        minio_db: MinioDB = request.app.state.minio
        minio_object_url = await minio_db.upload_file(
            object_name=object_name,
            file_data=upload_data.video_query.file,
            content_type=content_type,
        )

        media_processor: MediaProcessorClient = request.app.state.media_processor
        embedder: EmbedderClient = request.app.state.embedder
        indexer: IndexerClient = request.app.state.indexer

        source_url_str = str(upload_data.source_url) if upload_data.source_url else ""

        if extraction_strategy == "fixed_interval":
            extracted_frames = await media_processor.extract_frames_fixed_interval(
                video_path=object_name,
                interval_seconds=FRAME_INTERVAL,
                source_url=source_url_str,
                minio_path_url=minio_object_url,
                file_name=object_name,
            )
        elif extraction_strategy == "scene_detect":
            extracted_frames = await media_processor.extract_frames_scene_detect(
                video_path=object_name,
                threshold=SCENE_THRESHOLD,
                source_url=source_url_str,
                minio_path_url=minio_object_url,
                file_name=object_name,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported extraction strategy. Use 'fixed_interval' or 'scene_detect'.",
            )

        logger.info(f"Extracted {extracted_frames.frame_count} frames from video.")

        embeddings = await embedder.embed_images(
            frames=[f.model_dump() for f in extracted_frames.frames],
            media_id=extracted_frames.media_id,
        )
        logger.info(f"Generated embeddings for {len(embeddings)} frames.")

        result = await indexer.add_vectors(
            media_id=extracted_frames.media_id,
            embeddings=embeddings,
        )
        logger.info(
            f"Indexed {result.get('count', 0)} embeddings for media ID "
            f"{extracted_frames.media_id}."
        )

        return UploadResponse(file_url=minio_object_url)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
