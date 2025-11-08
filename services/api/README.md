# Mefid API Layer

**FastAPI gateway for the Mefid project**, exposing endpoints for media uploads, search, and health checks. This layer is responsible for handling incoming HTTP requests and communicating **only with the Media Processor service**.

It does **not** handle persistence directly; all heavy processing and storage are delegated to downstream services.

---

## Table of Contents

* [Features](#features)
* [Tech Stack](#tech-stack)
* [Folder Structure](#folder-structure)
* [Installation](#installation)
* [Environment Variables](#environment-variables)
* [Running the API](#running-the-api)
* [API Endpoints](#api-endpoints)

---

## Features

* Upload images and videos to MinIO via the Media Processor service
* Stream uploads efficiently (supports large files with multipart upload)
* Media metadata capture: title, filename, description, duration, source URL
* Search endpoint (integrates with Media Processor for retrieval)
* Health check endpoint for monitoring
* Async logging with structured output

---

## Tech Stack

* **Python 3.11**
* **FastAPI** for async HTTP endpoints
* **Uvicorn** as ASGI server
* **Pydantic** for request/response validation
* **MinIO** (via Media Processor) for object storage
* **Docker** for containerized deployment

---

## Folder Structure

```
api/
├── Dockerfile
├── pyproject.toml
├── src/
│   ├── api_app.py               # FastAPI entrypoint (lifespan + routers)
│   ├── routes/
│   │   ├── upload_route.py      # /upload endpoints
│   │   ├── search_route.py      # /search endpoints
│   │   ├── health_route.py      # /health endpoint
│   │   └── __init__.py
│   ├── service_clients/         # Clients to communicate with other services
│   │   └── media_processor_client.py
│   ├── schema/                  # Pydantic models
│   │   ├── responses.py
│   │   └── __init__.py
│   └── tests/                   # Unit and integration tests
│       ├── test_upload.py
│       ├── test_search.py
│       └── __init__.py
└── README.md
```

---

## Installation

```bash
# Clone repo
git clone https://github.com/your-username/mefid.git
cd mefid/api

# Create virtual environment
python -m venv .venv
source .venv/bin/activate
```

---

## Environment Variables

Create a `.env` file in the root of `api/`:

```env
# Media Processor service endpoint
MEDIA_PROCESSOR_URL={media_processor_service}

# MinIO / object storage credentials (used by Media Processor)
MINIO_ENDPOINT={endpoint}
MINIO_BUCKET_NAME={bucketname}
MINIO_ROOT_USER={root_user}
MINIO_ROOT_PASSWORD={password}
```

---

## Running the API

```bash
# Start the API locally
uvicorn src.api_app:app --reload
```

* Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* Redoc docs: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## API Endpoints

### Upload Media

```
POST /upload/image
```

**Parameters:**

* `media_type` (IMAGE | VIDEO) – required
* `image_query` / `video_query` – file stream
* Optional metadata: `title`, `filename`, `description`, `duration_seconds`, `source_url`

**Example cURL:**

```bash
curl -X POST "http://localhost:8000/upload/image" \
  -F "media_type=IMAGE" \
  -F "image_query=@/path/to/image.png" \
  -F "title=Sample Image"
```

**Response:**

```json
{
  "message": "Upload successful",
  "object_name": ${image},
  "file_url": ${file}
}
```

---

### Search Media

```
POST /search
```

* Accepts a query (image/video) and optional filters
* Returns top-k results from Media Processor

*(Implementation depends on Media Processor integration)*

---

### Health Check

```
GET /health
```

* Returns 200 OK if the API is running