# Mefid Caption Service

Visual captioning pipeline (Qwen2-VL). Separate from CLIP hybrid sampling and Whisper transcription.

## Local dev

**Prerequisites:** Docker Desktop running, then MinIO up on port 9000:

```bash
# from repo root
docker compose up minio -d
```

Use this service's own venv (not `services/transcribe/.venv`):

```bash
cd services/caption
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ../exports
pip install -e ".[gpu,dev]"
```

Load repo `.env` (MinIO + Supabase) from the project root, then:

```bash
set -a && source ../../.env && set +a
# Docker hostnames/paths in .env need overrides for local (non-compose) runs:
export MINIO_ENDPOINT=localhost:9000
export CAPTION_CACHE=./model_cache
python -m uvicorn src.caption_app:app --host 0.0.0.0 --port 8005
```

`FAISS_INDEX_PATH` in `.env` is for the indexer container only; caption connects to MinIO, not FAISS.

Confirm you're on the **caption** venv (not transcribe):

```bash
which python   # should end with services/caption/.venv/bin/python
which uvicorn  # should end with services/caption/.venv/bin/uvicorn
```

If either points at `services/transcribe/.venv`, run `deactivate` and `source .venv/bin/activate` again from `services/caption/`.

On Mac without CUDA the model runs on CPU (slow). Production target is GPU (L4 / CUDA).

## Colab validation (T4)

Use `notebooks/caption_validation.ipynb` on a Colab T4 GPU — do not run Qwen locally on Mac.

1. Upload the repo to Drive or clone into `/content/mefid`
2. Runtime → T4 GPU
3. Upload a test `.mp4` when prompted
4. Run all cells; review captions against the checklist in section 5

