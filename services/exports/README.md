# Mefid Exports Package

**Shared Python package for the Mefid project**, providing centralized access to constants, database clients, models, and utilities.  
This package acts as the **foundation layer** used by all services (API, Embedder, Indexer, Media Processor, etc.) to ensure consistent configuration and integration with MinIO, Supabase, and other shared systems.

---

## Table of Contents

* [Features](#features)
* [Tech Stack](#tech-stack)
* [Folder Structure](#folder-structure)
* [Installation](#installation)
* [Environment Variables](#environment-variables)
* [Usage Example](#usage-example)
* [Service Integration Matrix](#service-integration-matrix)

---

## Features

* Unified database clients for **MinIO** and **Supabase**
* FastAPI lifespan management for clean resource initialization and teardown
* Centralized constants and configuration schemas
* Shared Pydantic and dataclass models for cross-service consistency
* Structured logging utility for all services

---

## Tech Stack

* **Python 3.11**
* **MinIO SDK** for object storage
* **Supabase Python client** for database integration
* **Pydantic** for typed models and validation
* **FastAPI Lifespan** for startup/shutdown hooks
* **Docker** for consistent environments across services

---

## Folder Structure

```
exports/
├── pyproject.toml
├── src/
│   ├── db_clients/
│   │   ├── db_client.py      # functions to create and initialize DB clients for MinIO & Supabase
│   │   ├── lifespan.py       # FastAPI lifespan management for DB connections
│   │   ├── minioDB.py        # MinioDB class for interacting with MinIO object storage
│   │   ├── supabaseDB.py     # SupabaseDB class for interacting with Supabase database
│   ├── schemas/
│   │   ├── constants.py      # constants and configuration settings for the exports package
│   │   ├── models.py         # dataclass & Pydantic models shared across services
│   └── utils/
│       └── logger.py         # logging utility for structured, async-safe logging
└── README.md
````

---

## Installation

```bash
# Clone repo
git clone https://github.com/your-username/mefid.git
cd mefid/exports

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install as editable dependency
pip install -e .
````

When developing in other services (like `api` or `media-processor`), import directly:

```bash
pip install -e ../exports
```

---

## Environment Variables

```env
# -----------------------------
# 🪣 MinIO / Object Storage
# -----------------------------
MINIO_ENDPOINT={endpoint}
MINIO_BUCKET_NAME={bucket_name}
MINIO_ROOT_USER={root_user}
MINIO_ROOT_PASSWORD={password}

# -----------------------------
# 🧰 Supabase Database
# -----------------------------
SUPABASE_URL={supabase_url}
SUPABASE_KEY={supabase_key}

# -----------------------------
# 🧠 Other Service Endpoints
# -----------------------------
MEDIA_PROCESSOR_URL={media_processor_service}
INDEXER_SERVICE_URL={indexer_service}
```

> ⚠️ **Note:** The `exports` package reads from the environment — do not commit `.env` files or real credentials.

---

## Usage Example

```python
from exports.db_clients.minioDB import MinioDB
from exports.db_clients.supabaseDB import SupabaseDB
from exports.utils.logger import get_logger
from exports.schemas.models import FrameMetadata

logger = get_logger(__name__)

# Initialize clients
minio = MinioDB()
supabase = SupabaseDB()

# Example operation
file_path = "frames/frame_001.jpg"
frame = FrameMetadata(id="abc123", path=file_path)

await minio.put_object(
    bucket_name=MINIO_BUCKET_NAME,
    object_name=object_name,
    data=file_data,
    ...
)
logger.info(f"Uploaded {frame.id} to MinIO")

await supabase.table("frames").insert(frame.model_dump()).execute()
logger.debug("Inserted frames to Supabase")
```

---

## Service Integration Matrix

| **Service**            | **Uses**                                                                        | **Description / Integration Purpose**                                                                         |
| ---------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **API Service**        | `schemas.constants`, `schemas.models`, `utils.logger`                           | Uses shared constants and models for request validation and structured logging                                |
| **Media Processor**    | `db_clients.minioDB`, `db_clients.supabaseDB`, `schemas.models`, `utils.logger` | Uploads media files to MinIO, stores metadata in Supabase, and logs operations                                |
| **Embedder Service**   | `schemas.models`, `utils.logger`                                                | Uses shared models to represent frame and clip metadata before embedding                                      |
| **Indexer Service**    | `schemas.models`, `utils.logger`, `db_clients.supabaseDB`                       | Reads metadata from Supabase to populate FAISS index; logs search and indexing processes                      |
| **Ingest Service**     | `db_clients.minioDB`, `schemas.constants`, `utils.logger`                       | Retrieves raw files from MinIO and triggers ingestion workflows                                               |
| **Frontend (via API)** | —                                                                               | Does not import `exports` directly, but benefits from consistent schema definitions used in backend responses |

---

## Notes

* The `exports` package should remain **lightweight and dependency-agnostic**.
* Avoid adding service-specific logic — keep it generic and reusable.
* All services importing `exports` share the same base logging, model, and DB client setup.

