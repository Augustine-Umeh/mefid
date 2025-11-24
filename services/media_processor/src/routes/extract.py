from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
import tempfile
import os
from exports.utils.logger import get_logger
from exports.db_clients.minioDB import MinioDB
from exports.db_clients.supabaseDB import SupabaseDB
from exports.schema.models import MediaCreate, ExtractFramesResponse, FrameData
from ..extractors_techniques.fixed_interval import extract_frames_fixed_interval
from ..extractors_techniques.scene_based import extract_frames_scene_detect

router = APIRouter()
logger = get_logger()


class ExtractRequest(BaseModel):
    """Request body for frame extraction"""
    video_path: str
    source_url: str = ""
    minio_path_url: str = ""
    file_name: str = ""


@router.post("/", response_model=ExtractFramesResponse)
async def extract_frames(
    request: Request,
    extract_req: ExtractRequest,  # ← Body parameter
    strategy: str = Query(..., pattern="^(fixed_interval|scene_detect)$"),
    interval_seconds: float | None = Query(None),
    threshold: int | None = Query(None)
):
    """Extract frames from video using specified strategy"""
    
    if strategy == "fixed_interval" and interval_seconds is None:
        raise HTTPException(400, "interval_seconds is required for fixed_interval")
    if strategy == "scene_detect" and threshold is None:
        raise HTTPException(400, "threshold is required for scene_detect")
    
    try:
        logger.info(f"Extracting frames from {extract_req.video_path} using strategy={strategy}")
        
        minio: MinioDB = request.app.state.minio
        supabase: SupabaseDB = request.app.state.supabase
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_path = temp_file.name
            temp_file.write(await minio.download_file(extract_req.video_path))
        
        try:
            if strategy == "fixed_interval":
                video_frame_details = extract_frames_fixed_interval(
                    video_path=temp_path,
                    interval_seconds=interval_seconds
                )
            else:
                video_frame_details = extract_frames_scene_detect(
                    video_path=temp_path,
                    threshold=threshold
                )
            
            supabase_media = await supabase.insert_media(
                MediaCreate(
                    source_url=extract_req.source_url,
                    minio_path_url=extract_req.minio_path_url,
                    file_name=extract_req.file_name,
                    duration_seconds=video_frame_details["duration"],
                    extraction_strategy=strategy,
                    frame_count=len(video_frame_details["frames"])
                )
            )
            
            frame_data_list = [
                FrameData(timestamp=ts, frame_data=data)
                for ts, data in video_frame_details["frames"]
            ]
            
            return ExtractFramesResponse(
                video_path=extract_req.video_path,
                frames=frame_data_list,
                frame_count=len(frame_data_list),
                strategy=strategy,
                media_id=supabase_media.id
            )
        
        finally:
            os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))