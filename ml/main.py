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

# NOTE: api.routes and api.websocket are intentionally NOT imported here.
# They trigger synchronous loading of all 16 ML models (~60-90 s), which
# would block uvicorn from accepting any connections until loading finishes.
# Instead they are imported inside _load_models_and_routes() below, which
# runs in a thread pool after uvicorn has already started serving.

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

# CORS — allow localhost dev ports + Vercel deployment + any ngrok tunnel
_allowed_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:4000,http://localhost:5000,http://localhost:3000,http://localhost:3030,http://localhost:5173,https://dream-analysis.vercel.app,https://dream-analysis-*.vercel.app",
).split(",")

# Also accept any ngrok tunnel origin at runtime
_allow_origin_regex = r"https://.*\.ngrok(-free)?\.app|https://.*\.ngrok\.io|https://.*\.ngrok-free\.dev"

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_origin_regex=_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "neural-dream-ml"}


# ─── Deferred model loading ─────────────────────────────────────────
async def _load_models_and_routes():
    """Import api.routes in a thread (loads 16 ML models, ~60-90 s).
    Routes are registered on the event loop after loading completes.
    This lets uvicorn serve /health immediately on startup."""
    def _do_blocking_import():
        from api.routes import router as _r
        from api.websocket import eeg_stream_endpoint as _ep
        return _r, _ep

    try:
        logger.info("[startup] Loading ML models in background thread...")
        router, eeg_stream_endpoint = await asyncio.to_thread(_do_blocking_import)
        # Back on the event loop — safe to mutate app routing tables
        app.include_router(router, prefix="/api")
        app.websocket("/ws/eeg-stream")(eeg_stream_endpoint)
        logger.info("[startup] ML models loaded, all routes registered.")
    except Exception as exc:
        logger.error(f"[startup] ML model loading failed: {exc}", exc_info=True)


# ─── Twice-daily personal model retraining ──────────────────────────
_RETRAIN_INTERVAL_SEC = 12 * 3600  # every 12 hours = twice daily


async def _auto_train_loop():
    """Background task: retrain the personal model every 12 hours."""
    await asyncio.sleep(90)  # wait 90 s after startup before first run
    while True:
        try:
            from training.auto_retrainer import retrain_personal_model
            from monitoring.datadog_reporter import report_metric, report_error
            result = await asyncio.to_thread(retrain_personal_model)
            logger.info(f"[auto-retrain] {result}")
            # Report accuracy metric to Datadog
            if isinstance(result, dict) and "accuracy" in result:
                report_metric("neural_dream.retrain.accuracy", float(result["accuracy"]))
                report_metric("neural_dream.retrain.success", 1.0, metric_type="count")
        except Exception as exc:
            logger.warning(f"[auto-retrain] error: {exc}")
            try:
                from monitoring.datadog_reporter import report_error
                report_error("auto_retrain_failed", f"Personal model retraining failed: {exc}", exc=exc)
            except Exception:
                pass
        await asyncio.sleep(_RETRAIN_INTERVAL_SEC)


@app.on_event("startup")
async def _start_background_tasks():
    asyncio.create_task(_load_models_and_routes())
    asyncio.create_task(_auto_train_loop())


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("ML_PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
