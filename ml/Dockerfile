FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Build context = repo root (railway up uses git repo as reference)
# Copy ml/ subdirectory contents into /app/ so paths resolve correctly
COPY ml/ .

RUN pip install --no-cache-dir -r requirements.deploy.txt

EXPOSE 8000

# Railway injects $PORT at runtime
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
