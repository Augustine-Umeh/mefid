FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install shared/exports package first
COPY services/exports /tmp/exports
RUN pip install --no-cache-dir /tmp/exports

# Copy API service files
COPY ./services/api/pyproject.toml ./
COPY ./services/api/src ./src

# Install API service
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 8000

# Run FastAPI with hot reload for development
CMD ["uvicorn", "src.api_app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]