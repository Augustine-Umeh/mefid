FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install shared/exports package
COPY services/exports /tmp/exports
RUN pip install --no-cache-dir /tmp/exports

# Copy indexer service files
COPY ./services/indexer/pyproject.toml ./
COPY ./services/indexer/src ./src

# Install indexer service
RUN pip install --no-cache-dir -e .

# Create data directory for FAISS index persistence
RUN mkdir -p /app/data

# Expose port
EXPOSE 8003

# Run FastAPI with hot reload
CMD ["uvicorn", "src.indexer_app:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]