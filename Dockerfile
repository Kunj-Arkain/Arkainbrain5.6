FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy project
COPY . .

# Create output directories
RUN mkdir -p output/recon data/regulations/us_states logs

# Pre-create CrewAI config to prevent tracing prompt
RUN mkdir -p /root/.crewai /tmp/crewai_storage && \
    echo '{"tracing_enabled": false, "tracing_disabled": true}' > /root/.crewai/config.json && \
    echo '{"tracing_enabled": false, "tracing_disabled": true}' > /tmp/crewai_storage/config.json

# Railway sets PORT env var
EXPOSE ${PORT:-8080}

# Run Flask via gunicorn
# 1 worker: parallelism comes from subprocess workers (worker.py), not gunicorn.
# 8 threads: handles concurrent HTTP requests + SSE log streams.
# --max-requests: recycles worker to prevent memory leaks. Sessions survive
#   because SECRET_KEY is persisted to file (not regenerated per-process).
# Subprocess pipeline workers survive gunicorn restarts (start_new_session=True).
CMD gunicorn web_app:app \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers 1 \
    --threads 8 \
    --timeout 900 \
    --graceful-timeout 30 \
    --keep-alive 5 \
    --max-requests 500 \
    --max-requests-jitter 50
