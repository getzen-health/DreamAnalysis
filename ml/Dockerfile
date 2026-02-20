FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install inference-only dependencies first (layer-cached separately from code)
COPY ml/requirements.deploy.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source.
# Large directories (ml/data/, ml/models/*.joblib > 50MB) excluded via .dockerignore.
COPY ml/ .

EXPOSE 8000

# Longer start-period: scikit-learn + onnxruntime load slowly on first request
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
