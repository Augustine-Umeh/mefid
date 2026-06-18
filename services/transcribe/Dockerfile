FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/exports /tmp/exports
RUN pip install --no-cache-dir /tmp/exports

COPY ./services/transcribe/pyproject.toml ./
COPY ./services/transcribe/src ./src

RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/model_cache

EXPOSE 8004

CMD ["uvicorn", "src.transcribe_app:app", "--host", "0.0.0.0", "--port", "8004", "--reload"]
