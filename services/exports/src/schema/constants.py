import os
from dotenv import load_dotenv
from enum import Enum

class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    IMAGE_VIDEO = "image_video"
    TEXT_IMAGE = "text_image"
    TEXT_VIDEO = "text_video"
    ALL = "all"

load_dotenv()

DEFAULT_TOP_K = 5
API_SERVICE = os.getenv("API_SERVICE")
MEDIA_PROCESSOR_SERVICE = os.getenv("MEDIA_PROCESSOR_SERVICE")
EMBEDDER_SERVICE = os.getenv("EMBEDDER_SERVICE")
INDEXER_SERVICE = os.getenv("INDEXER_SERVICE")
DB_SERVICE = os.getenv("DB_SERVICE")

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_ADMIN_API_KEY = os.getenv("SUPABASE_ADMIN_API_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")
MINIO_USE_SSL = os.getenv("MINIO_USE_SSL", "false").lower() == "true"