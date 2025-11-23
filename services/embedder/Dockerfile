FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install shared/exports package
COPY services/exports /tmp/exports
RUN pip install --no-cache-dir /tmp/exports

# Copy embedder service files
COPY ./services/embedder/pyproject.toml ./
COPY ./services/embedder/src ./src

# Install embedder service
RUN pip install --no-cache-dir -e .

# Create cache directory for models
RUN mkdir -p /app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache

# Expose port
EXPOSE 8002

# Run FastAPI with hot reload
CMD ["uvicorn", "src.embedder_app:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]