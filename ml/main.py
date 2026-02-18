import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from api.websocket import eeg_stream_endpoint

# ─── Logging ────────────────────────────────────────────────────────
log_level = os.environ.get("ML_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Neural Dream Workshop - ML Service",
    description="EEG signal processing: sleep staging, emotion classification, dream detection, flow state, creativity detection, and memory encoding prediction",
    version="1.0.0",
)

# CORS — restrict to known origins in production, allow all in development
_allowed_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5000,http://localhost:3000,http://localhost:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# WebSocket for real-time EEG streaming
app.websocket("/ws/eeg-stream")(eeg_stream_endpoint)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "neural-dream-ml"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("ML_PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
