"""Cognitive model endpoints: drowsiness, load, attention, stress, lucid dream, meditation."""

import numpy as np
from fastapi import APIRouter

from ._shared import (
    _numpy_safe,
    drowsiness_model, cognitive_load_model, attention_model,
    stress_model, lucid_dream_model, meditation_model,
    EEGInput, LucidDreamRequest,
)

router = APIRouter()


@router.post("/predict-drowsiness")
async def predict_drowsiness(data: EEGInput):
    """Detect drowsiness level: alert / drowsy / sleepy."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    return _numpy_safe(drowsiness_model.predict(eeg, data.fs))


@router.post("/predict-cognitive-load")
async def predict_cognitive_load(data: EEGInput):
    """Estimate cognitive load: low / moderate / high."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    return _numpy_safe(cognitive_load_model.predict(eeg, data.fs))


@router.post("/predict-attention")
async def predict_attention(data: EEGInput):
    """Classify attention: distracted / passive / focused / hyperfocused."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    return _numpy_safe(attention_model.predict(eeg, data.fs))


@router.post("/predict-stress")
async def predict_stress(data: EEGInput):
    """Detect stress level: relaxed / mild / moderate / high."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    return _numpy_safe(stress_model.predict(eeg, data.fs))


@router.post("/predict-lucid-dream")
async def predict_lucid_dream(req: LucidDreamRequest):
    """Detect lucid dreaming: non_lucid / pre_lucid / lucid / controlled."""
    signals = np.array(req.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    return _numpy_safe(lucid_dream_model.predict(
        eeg, req.fs, is_rem=req.is_rem, sleep_stage=req.sleep_stage
    ))


@router.post("/predict-meditation")
async def predict_meditation(data: EEGInput):
    """Classify meditation depth: surface / light / moderate / deep / transcendent."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    return _numpy_safe(meditation_model.predict(eeg, data.fs))


@router.get("/cognitive-models/session-stats")
async def cognitive_session_stats():
    """Get session statistics for all cognitive models that track history."""
    stats = {}
    if hasattr(lucid_dream_model, "get_session_stats"):
        stats["lucid_dream"] = lucid_dream_model.get_session_stats()
    if hasattr(meditation_model, "get_session_stats"):
        stats["meditation"] = meditation_model.get_session_stats()
    return _numpy_safe(stats)
