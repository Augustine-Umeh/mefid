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


def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        if default is not None:
            return default
        raise ValueError(f"Environment variable {name} is not set")
    return value


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in ("1", "true", "yes")


# -----------------------------
# Service endpoints (API gateway clients only)
# -----------------------------
API_SERVICE = _get_env("API_SERVICE", "")
MEDIA_PROCESSOR_SERVICE = _get_env("MEDIA_PROCESSOR_SERVICE", "")
EMBEDDER_SERVICE = _get_env("EMBEDDER_SERVICE", "")
INDEXER_SERVICE = _get_env("INDEXER_SERVICE", "")
TRANSCRIBE_SERVICE = _get_env("TRANSCRIBE_SERVICE", "")

# -----------------------------
# Supabase (lifespan services; optional at import for embedder)
# -----------------------------
SUPABASE_DB_URL = _get_env("SUPABASE_DB_URL", "")
SUPABASE_SERVICE_ROLE_KEY = _get_env("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_ADMIN_API_KEY = _get_env("SUPABASE_ADMIN_API_KEY", "")
SUPABASE_ANON_KEY = _get_env("SUPABASE_ANON_KEY", "")

# -----------------------------
# MinIO (lifespan services that store media; optional at import for embedder/indexer)
# -----------------------------
MINIO_ENDPOINT = _get_env("MINIO_ENDPOINT", "")
MINIO_ACCESS_KEY = _get_env("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = _get_env("MINIO_SECRET_KEY", "")
MINIO_BUCKET_NAME = _get_env("MINIO_BUCKET_NAME", "")
MINIO_USE_SSL = _get_env("MINIO_USE_SSL", "false").lower() == "true"

# -----------------------------
# Pipeline tunables (typed, with safe defaults)
# -----------------------------
FRAME_INTERVAL: float = _get_float("FRAME_INTERVAL", 2.0)
SCENE_THRESHOLD: int = _get_int("SCENE_THRESHOLD", 10)

# Hybrid frame sampling (media_processor service)
PHASH_SIZE: int = _get_int("PHASH_SIZE", 8)
PHASH_MULTIPLIER: float = _get_float("PHASH_MULTIPLIER", 1.5)
FLOOR_INTERVAL: float = _get_float("FLOOR_INTERVAL", 5.0)
MIN_SAMPLE_GAP: float = _get_float("MIN_SAMPLE_GAP", 1.0)
# Default number of nearest neighbours for search; also the maximum ``top_k``
# allowed per request (API and indexer both clamp to this value).
DEFAULT_TOP_K: int = _get_int("DEFAULT_TOP_K", 25)
# When ``vector_type`` filtering is active, the API over-fetches this many FAISS
# candidates before filtering so image hits are not crowded out by transcripts.
FILTERED_SEARCH_MAX_K: int = _get_int("FILTERED_SEARCH_MAX_K", 500)

# -----------------------------
# Embedding model (embedder service)
# -----------------------------
CLIP_MODEL: str = _get_env("CLIP_MODEL", "")
EMBED_IMAGE_BATCH_SIZE: int = _get_int("EMBED_IMAGE_BATCH_SIZE", 16)
CLIP_DIMENSION: int = _get_int("CLIP_DIMENSION", 768)
# Safe margin under CLIP's 77-token limit when chunking transcript text.
CLIP_TEXT_CHUNK_MAX_TOKENS: int = _get_int("CLIP_TEXT_CHUNK_MAX_TOKENS", 70)
TRANSFORMERS_CACHE = _get_env("TRANSFORMERS_CACHE", "")

# -----------------------------
# Whisper / transcribe (transcribe service)
# -----------------------------
WHISPER_MODEL: str = _get_env("WHISPER_MODEL", "")
WHISPER_DEVICE: str = _get_env("WHISPER_DEVICE", "")
WHISPER_COMPUTE_TYPE: str = _get_env("WHISPER_COMPUTE_TYPE", "")
WHISPER_CACHE = _get_env("WHISPER_CACHE", "")

# -----------------------------
# Shared FastAPI lifespan (`exports.db_clients.lifespan`)
# -----------------------------
# Embedder: skip Supabase/MinIO; load CLIP only. Set in embedder container env.
EXPORTS_LIFESPAN_EMBEDDER: bool = _get_bool("EXPORTS_LIFESPAN_EMBEDDER", False)

# -----------------------------
# FAISS (indexer service)
# -----------------------------
FAISS_INDEX_PATH = _get_env("FAISS_INDEX_PATH", "")
