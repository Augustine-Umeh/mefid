from typing import Optional
from fastapi import APIRouter, File, HTTPException, Form, Request, UploadFile
from services.exports.src.schema.models import UploadRequest
from services.api.src.schema.responses import UploadResponse
from services.exports.src.db_clients.minioDB import MinioDB
from services.exports.src.schema.constants import MediaType
from services.exports.src.utils.logger import get_logger

router = APIRouter()
logger = get_logger()

@router.post("/image", response_model=UploadResponse)
async def upload_image(
    request: Request,
    media_type: MediaType = Form(...),
    image_query: Optional[UploadFile] = File(None),
    video_query: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    filename: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    duration_seconds: Optional[float] = Form(None),
    source_url: Optional[str] = Form(None),
) -> UploadResponse:
    """
    Stream upload to MinIO and process file
    """
    upload_data = UploadRequest(
        media_type=media_type,
        title=title,
        filename=filename,
        video_query=video_query,
        image_query=image_query,
        description=description,
        duration_seconds=duration_seconds,
        source_url=source_url
    )
    
    if upload_data.media_type == MediaType.VIDEO:
        file = upload_data.video_query
    elif upload_data.media_type == MediaType.IMAGE:
        file = upload_data.image_query
    else:
        raise ValueError("Unsupported media type")
    
    object_name = file.filename
    content_type = file.content_type or "application/octet-stream"
    
    try:
        logger.info(f"Uploading {object_name} to MinIO...")
        minio_db: MinioDB = request.app.state.minio
        # file_data = await file.read()
        object_url = await minio_db.upload_file(
            object_name=object_name,
            file_data=file.file,
            content_type=content_type
        )

        logger.info(f"Uploaded successfully: {object_url}")
        return {
            "message": "Upload successful",
            "file_url": object_url
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
        logger.info(f"Uploading {object_name} to MinIO...")
        minio_db: MinioDB = request.app.state.minio
        # file_data = await upload_data.video_query.read()
        object_url = await minio_db.upload_file(
            object_name=object_name,
            file_data=upload_data.video_query.file,
            content_type=content_type
        )

        logger.info(f"Uploaded successfully: {object_url}")
        return {
            "message": "Upload successful",
            "file_url": object_url
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
