FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy everything first (dockerignore keeps the image small)
COPY . .

# Install inference-only dependencies
RUN pip install --no-cache-dir -r requirements.deploy.txt

EXPOSE 8000

# Shell form so $PORT is expanded at runtime
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
