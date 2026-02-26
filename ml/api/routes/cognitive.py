"""Cognitive model endpoints: drowsiness, load, attention, stress, lucid dream, meditation.

Improvements over baseline version:
- EMA output smoothing (α=0.25) — reduces frame-to-frame noise by ~75%
- Per-user BaselineCalibrator wiring — normalizes features to user's resting baseline
  before sklearn prediction (only active when cal.is_ready after 30 baseline frames)
"""

import threading
import numpy as np
from fastapi import APIRouter
from typing import Any, Dict

from ._shared import (
    _numpy_safe,
    drowsiness_model, cognitive_load_model, attention_model,
    stress_model, lucid_dream_model, meditation_model,
    EEGInput, LucidDreamRequest,
)
from .calibration import _get_baseline_cal

router = APIRouter()

# ── Per-user EMA smoothing ────────────────────────────────────────────────────
_cognitive_ema: Dict[str, Dict[str, float]] = {}
_cognitive_ema_lock = threading.Lock()
_EMA_ALPHA = 0.25  # matches food_emotion_predictor.py


def _smooth(user_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Apply EMA smoothing to all top-level float values in result."""
    with _cognitive_ema_lock:
        state = _cognitive_ema.setdefault(user_id, {})
        out = dict(result)
        for key, val in result.items():
            if isinstance(val, float):
                prev = state.get(key, val)
                smoothed = _EMA_ALPHA * val + (1.0 - _EMA_ALPHA) * prev
                state[key] = smoothed
                out[key] = round(smoothed, 4)
        return out


def _calibrated_predict(model, eeg: np.ndarray, fs: float, user_id: str) -> Dict:
    """Run model prediction with optional per-user baseline normalization.

    If BaselineCalibrator is ready (≥30 resting frames collected) AND the model
    has sklearn weights, features are z-scored against the user's resting baseline
    before prediction. Falls back to model.predict() if anything fails.
    """
    from processing.eeg_processor import extract_features, preprocess

    cal = _get_baseline_cal(user_id)
    if (
        cal.is_ready
        and hasattr(model, "sklearn_model") and model.sklearn_model is not None
        and hasattr(model, "feature_names") and model.feature_names is not None
    ):
        try:
            processed = preprocess(eeg, fs)
            features = extract_features(processed, fs)
            normalized = cal.normalize(features)
            fv = np.array(
                [normalized.get(k, features.get(k, 0.0)) for k in model.feature_names]
            ).reshape(1, -1)
            if getattr(model, "scaler", None) is not None:
                fv = model.scaler.transform(fv)
            cal_probs = model.sklearn_model.predict_proba(fv)[0]
            cal_idx = int(np.argmax(cal_probs))
            # Get base result for full dict structure, then patch calibrated values
            base = model.predict(eeg, fs)
            if "confidence" in base:
                base["confidence"] = round(float(cal_probs[cal_idx]), 3)
            for key in list(base.keys()):
                if key.endswith("_index"):
                    base[key] = cal_idx
            base["baseline_calibrated"] = True
            return base
        except Exception:
            pass
    return model.predict(eeg, fs)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/predict-drowsiness")
async def predict_drowsiness(data: EEGInput):
    """Detect drowsiness level: alert / drowsy / sleepy."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    result = _calibrated_predict(drowsiness_model, eeg, data.fs, data.user_id)
    return _numpy_safe(_smooth(data.user_id, result))


@router.post("/predict-cognitive-load")
async def predict_cognitive_load(data: EEGInput):
    """Estimate cognitive load: low / moderate / high."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    result = _calibrated_predict(cognitive_load_model, eeg, data.fs, data.user_id)
    return _numpy_safe(_smooth(data.user_id, result))


@router.post("/predict-attention")
async def predict_attention(data: EEGInput):
    """Classify attention: distracted / passive / focused / hyperfocused."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    result = _calibrated_predict(attention_model, eeg, data.fs, data.user_id)
    return _numpy_safe(_smooth(data.user_id, result))


@router.post("/predict-stress")
async def predict_stress(data: EEGInput):
    """Detect stress level: relaxed / mild / moderate / high."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    result = _calibrated_predict(stress_model, eeg, data.fs, data.user_id)
    return _numpy_safe(_smooth(data.user_id, result))


@router.post("/predict-lucid-dream")
async def predict_lucid_dream(req: LucidDreamRequest):
    """Detect lucid dreaming: non_lucid / pre_lucid / lucid / controlled."""
    signals = np.array(req.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    result = _calibrated_predict(lucid_dream_model, eeg, req.fs, req.user_id)
    return _numpy_safe(_smooth(req.user_id, result))


@router.post("/predict-meditation")
async def predict_meditation(data: EEGInput):
    """Classify meditation depth: relaxed / meditating / deep."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    result = _calibrated_predict(meditation_model, eeg, data.fs, data.user_id)
    return _numpy_safe(_smooth(data.user_id, result))


@router.get("/cognitive-models/session-stats")
async def cognitive_session_stats():
    """Get session statistics for all cognitive models that track history."""
    stats = {}
    if hasattr(lucid_dream_model, "get_session_stats"):
        stats["lucid_dream"] = lucid_dream_model.get_session_stats()
    if hasattr(meditation_model, "get_session_stats"):
        stats["meditation"] = meditation_model.get_session_stats()
    return _numpy_safe(stats)
