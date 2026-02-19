import asyncio
import logging
import os
import warnings

# Datadog APM — patch before other library imports
# Only activates when DD_AGENT_HOST or DD_API_KEY is set
if os.environ.get("DD_AGENT_HOST") or os.environ.get("DD_API_KEY"):
    try:
        from ddtrace import patch_all
        patch_all()
    except ImportError:
        pass

# Suppress sklearn/LightGBM feature-name warnings (model trained with DataFrame,
# called with numpy arrays — harmless mismatch that floods the log)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn",
)

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


# ─── Twice-daily personal model retraining ──────────────────────────
_RETRAIN_INTERVAL_SEC = 12 * 3600  # every 12 hours = twice daily


async def _auto_train_loop():
    """Background task: retrain the personal model every 12 hours."""
    await asyncio.sleep(90)  # wait 90 s after startup before first run
    while True:
        try:
            from training.auto_retrainer import retrain_personal_model
            result = await asyncio.to_thread(retrain_personal_model)
            logger.info(f"[auto-retrain] {result}")
        except Exception as exc:
            logger.warning(f"[auto-retrain] error: {exc}")
        await asyncio.sleep(_RETRAIN_INTERVAL_SEC)


@app.on_event("startup")
async def _start_background_tasks():
    asyncio.create_task(_auto_train_loop())


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("ML_PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
