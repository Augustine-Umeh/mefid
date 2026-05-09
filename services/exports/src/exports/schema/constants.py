import os
from dotenv import load_dotenv
from enum import Enum


# ============================================================
# DB-aligned enums (must match dev_schema.sql exactly)
# ============================================================
class MediaType(str, Enum):
    """Postgres `media_type` enum — what a `media` row IS."""
    VIDEO = "video"
    IMAGE = "image"


class ContentType(str, Enum):
    """Postgres `content_type` enum — pacing/vibe descriptor for a media row."""
    FAST_PACED = "fast_paced"
    STATIC = "static"
    MIXED = "mixed"


class ExtractionStrategy(str, Enum):
    """Postgres `extraction_strategy` enum — how frames were extracted."""
    FIXED_INTERVAL = "fixed_interval"
    SCENE_DETECT = "scene_detect"
    HYBRID = "hybrid"


class MediaStatus(str, Enum):
    """Postgres `media_status` enum — lifecycle state of a media row."""
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class VectorType(str, Enum):
    """Postgres `vector_type` enum — what an embedding represents."""
    IMAGE = "image"
    TEXT = "text"


class QueryType(str, Enum):
    """Postgres `query_type` enum — modality of a search query."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


load_dotenv()


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


# -----------------------------
# Service endpoints
# -----------------------------
API_SERVICE = os.getenv("API_SERVICE")
MEDIA_PROCESSOR_SERVICE = os.getenv("MEDIA_PROCESSOR_SERVICE")
EMBEDDER_SERVICE = os.getenv("EMBEDDER_SERVICE")
INDEXER_SERVICE = os.getenv("INDEXER_SERVICE")

# -----------------------------
# Supabase
# -----------------------------
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_ADMIN_API_KEY = os.getenv("SUPABASE_ADMIN_API_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# -----------------------------
# MinIO
# -----------------------------
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")
MINIO_USE_SSL = os.getenv("MINIO_USE_SSL", "false").lower() == "true"

# -----------------------------
# Pipeline tunables (typed, with safe defaults)
# -----------------------------
FRAME_INTERVAL: float = _get_float("FRAME_INTERVAL", 2.0)
SCENE_THRESHOLD: int = _get_int("SCENE_THRESHOLD", 10)
DEFAULT_TOP_K: int = _get_int("DEFAULT_TOP_K", 10)

# -----------------------------
# Embedding model
# -----------------------------
CLIP_MODEL: str = os.getenv("CLIP_MODEL")
CLIP_DIMENSION: int = _get_int("CLIP_DIMENSION", 512)
TRANSFORMERS_CACHE_DIR = os.getenv("TRANSFORMERS_CACHE_DIR")

# -----------------------------
# FAISS
# -----------------------------
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
