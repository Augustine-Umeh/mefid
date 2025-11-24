from typing import Optional
from fastapi import APIRouter, File, HTTPException, Form, Request, UploadFile
from exports.schema.models import ExtractFramesResponse, Media, UploadRequest, FrameMetadata
from src.schema.responses import UploadResponse
from exports.db_clients.minioDB import MinioDB
from exports.schema.constants import MediaType, FRAME_INTERVAL, SCENE_THRESHOLD, CLIP_MODEL
from exports.utils.logger import get_logger
from ..service_clients.media_processor_client import MediaProcessorClient
from ..service_clients.indexer_client import IndexerClient
from ..service_clients.embedder_client import EmbedderClient

router = APIRouter()
logger = get_logger()

@router.post("/image", response_model=UploadResponse)
async def upload_image(
    request: Request,
    media_type: MediaType = Form(...),
    image_query: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    filename: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    source_url: Optional[str] = Form(None),
) -> UploadResponse:
    """
    Stream upload to MinIO and process file
    """
    upload_data = UploadRequest(
        media_type=media_type,
        title=title,
        filename=filename,
        image_query=image_query,
        description=description,
        source_url=source_url
    )
    
    if upload_data.media_type == MediaType.IMAGE:
        file = upload_data.image_query
    else:
        raise ValueError("Unsupported media type")
    
    object_name = file.filename
    content_type = file.content_type or "application/octet-stream"
    
    try:
        logger.info(f"Uploading {object_name} to MinIO...")
        minio_db: MinioDB = request.app.state.minio
        # file_data = await file.read()
        minio_object_url = await minio_db.upload_file(
            object_name=object_name,
            file_data=file.file,
            content_type=content_type
        )

        logger.info(f"Uploaded successfully: {minio_object_url}")
        return {
            "message": "Upload successful",
            "file_url": minio_object_url
        }

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
    """
    Stream upload video to MinIO and process file
    """
    upload_data = UploadRequest(
        media_type=MediaType.VIDEO,
        title=title,
        filename=filename,
        video_query=video_query,
        description=description,
        duration_seconds=duration_seconds,
        source_url=source_url
    )
    
    object_name = upload_data.video_query.filename
    content_type = upload_data.video_query.content_type or "application/octet-stream"
    
    try:
        minio_db: MinioDB = request.app.state.minio
        # file_data = await upload_data.video_query.read()
        minio_object_url = await minio_db.upload_file(
            object_name=object_name,
            file_data=upload_data.video_query.file,
            content_type=content_type
        )
        
        async with MediaProcessorClient() as media_processor_client:
            if extraction_strategy == "fixed_interval":
                extracted_frames = await media_processor_client.extract_frames_fixed_interval(
                    video_path=minio_object_url,
                    interval_seconds=FRAME_INTERVAL,
                    source_url=upload_data.source_url,
                    minio_path_url=minio_object_url,
                    file_name=object_name
                )
            elif extraction_strategy == "scene_detect":
                extracted_frames = await media_processor_client.extract_frames_scene_detect(
                    video_path=minio_object_url,
                    threshold=SCENE_THRESHOLD,
                    source_url=upload_data.source_url,
                    minio_path_url=minio_object_url,
                    file_name=object_name
                )
            else:
                raise ValueError("Unsupported extraction strategy. Use 'fixed_interval' or 'scene_detect'")
        
        logger.info(f"Extracted {extracted_frames.frame_count} frames from video.")
        
        async with EmbedderClient() as embedder_client:
            embeddings = await embedder_client.embed_images(
                frames=extracted_frames.frames,
                media_id=extracted_frames.media_id
            )
            
        logger.info(f"Generated embeddings for {len(embeddings)} frames.")
            
        async with IndexerClient() as indexer_client:
            reesult = await indexer_client.add_vectors(
                media_id=extracted_frames.media_id,
                embeddings=embeddings
            )
        logger.info(f"Indexed {len(reesult.amount)} embeddings for media ID {extracted_frames.media_id}.")
        
        return {
            "message": "Uploaded successfully",
            "file_url": minio_object_url
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
