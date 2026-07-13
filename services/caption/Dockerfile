FROM python:3.11-slim

WORKDIR /app

# ffmpeg + OpenCV/scenedetect system libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch before service deps (see embedder Dockerfile for rationale).
# Default: CPU wheels for Docker Desktop on macOS.
# Linux GPU: docker compose build --build-arg TORCH_INDEX=cu121 caption
ARG TORCH_INDEX=cpu
RUN if [ "$TORCH_INDEX" = "cpu" ]; then \
      pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.1.0" "torchvision>=0.16.0"; \
    else \
      pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/${TORCH_INDEX} \
        "torch>=2.1.0" "torchvision>=0.16.0"; \
    fi

COPY services/exports /tmp/exports
RUN pip install --no-cache-dir /tmp/exports

COPY ./services/caption/pyproject.toml ./
COPY ./services/caption/src ./src

# bitsandbytes is listed in [gpu] but unused; install remaining GPU extras only.
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir \
    "transformers>=4.45.0" \
    "accelerate>=0.26.0" \
    "qwen-vl-utils>=0.0.8"

RUN mkdir -p /app/model_cache

EXPOSE 8005

CMD ["uvicorn", "src.caption_app:app", "--host", "0.0.0.0", "--port", "8005"]
