FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch BEFORE anything else.
#
# The default PyPI wheel for `torch` on linux pulls in the full NVIDIA
# CUDA stack (cudnn, cublas, cusparse, nccl, triton, ...) — ~2.5 GB
# of GPU libraries that are useless inside Docker Desktop on macOS
# and frequently exhaust the VM's disk during image build.
#
# By pre-installing the CPU build from PyTorch's CPU index, the
# subsequent `pip install -e .` sees `torch>=2.0.0` as already
# satisfied and doesn't reach for the GPU wheels.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.0.0"

# Install shared/exports package
COPY services/exports /tmp/exports
RUN pip install --no-cache-dir /tmp/exports

# Copy embedder service files
COPY ./services/embedder/pyproject.toml ./
COPY ./services/embedder/src ./src

# Install embedder service (torch is already satisfied above)
RUN pip install --no-cache-dir -e .

# Create cache directory for models
RUN mkdir -p /app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache

# Expose port
EXPOSE 8002

# Run FastAPI with hot reload
CMD ["uvicorn", "src.embedder_app:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
