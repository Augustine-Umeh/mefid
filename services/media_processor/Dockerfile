FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for video processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install shared/exports package
COPY services/exports /tmp/exports
RUN pip install --no-cache-dir /tmp/exports

# Copy media processor service files
COPY ./services/media_processor/pyproject.toml ./
COPY ./services/media_processor/src ./src

# Install media processor service
RUN pip install --no-cache-dir -e .

# Create temp directory for video processing
RUN mkdir -p /tmp/videos

# Expose port
EXPOSE 8001

# Run FastAPI with hot reload
CMD ["uvicorn", "src.media_processor_app:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]