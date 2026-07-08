FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

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

# CUDA torch/torchvision are pre-installed in the base image.
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
