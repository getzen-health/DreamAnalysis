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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from api.auth import APIKeyMiddleware
from api.cors import get_allowed_origins, ORIGIN_REGEX
from api.rate_limit import RateLimiter

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

# ─── CORS — explicit origin allowlist (never wildcard in production) ──
_allowed_origins = get_allowed_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_origin_regex=ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Rate limiting — 100 requests/min per IP ──────────────────────────
_rate_limiter = RateLimiter(
    max_requests=int(os.environ.get("ML_RATE_LIMIT", "100")),
    window_seconds=60,
)

# Paths exempt from rate limiting (health checks, docs)
_RATE_LIMIT_EXEMPT = frozenset({"/health", "/status", "/docs", "/openapi.json", "/redoc"})


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path.rstrip("/") or "/"
    if path not in _RATE_LIMIT_EXEMPT:
        client_ip = request.client.host if request.client else "unknown"
        if not _rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Max 100 requests per minute."},
            )
    return await call_next(request)

app.add_middleware(APIKeyMiddleware)

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
    """Background task: retrain models every 12 hours.

    Two retrain paths:
      1. Per-user UserModelRetrainer: fine-tunes from accumulated corrections.
      2. Legacy retrain_personal_model: retrains from session data.
    """
    await asyncio.sleep(90)  # wait 90 s after startup before first run
    while True:
        # 1. Per-user retraining from accumulated corrections
        try:
            from training.retrain_from_user_data import UserModelRetrainer
            user_corrections_dir = os.path.join(
                os.path.dirname(__file__), "user_data", "corrections"
            )
            if os.path.exists(user_corrections_dir):
                for filename in os.listdir(user_corrections_dir):
                    if filename.endswith("_corrections.jsonl"):
                        user_id = filename.replace("_corrections.jsonl", "")
                        try:
                            retrainer = UserModelRetrainer(user_id)
                            if retrainer.should_retrain():
                                result = await asyncio.to_thread(retrainer.retrain_all)
                                logger.info(
                                    "[auto-retrain] UserModelRetrainer user=%s: %s",
                                    user_id, result,
                                )
                        except Exception as exc:
                            logger.warning(
                                "[auto-retrain] UserModelRetrainer user=%s error: %s",
                                user_id, exc,
                            )
        except Exception as exc:
            logger.warning(f"[auto-retrain] per-user retrain sweep error: {exc}")

        # 2. Legacy session-based retraining
        try:
            from training.auto_retrainer import retrain_personal_model
            result = await asyncio.to_thread(retrain_personal_model)
            logger.info(f"[auto-retrain] legacy: {result}")
            try:
                from monitoring.datadog_reporter import report_metric
                if isinstance(result, dict) and "accuracy" in result:
                    report_metric("neural_dream.retrain.accuracy", float(result["accuracy"]))
                    report_metric("neural_dream.retrain.success", 1.0, metric_type="count")
            except Exception:
                pass
        except Exception as exc:
            logger.warning(f"[auto-retrain] legacy error: {exc}")
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
    is_dev = os.environ.get("ENV", "production") == "dev"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=is_dev)
