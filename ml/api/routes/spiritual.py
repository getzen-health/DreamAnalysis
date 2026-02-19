"""Spiritual energy / self-awareness endpoints."""

import numpy as np
from fastapi import APIRouter

from ._shared import (
    _numpy_safe,
    EEGInput,
    preprocess,
    compute_chakra_activations, compute_chakra_balance,
    compute_meditation_depth, compute_aura_energy,
    compute_kundalini_flow, compute_prana_balance,
    compute_consciousness_level, compute_third_eye_activation,
    full_spiritual_analysis,
    CHAKRAS, CONSCIOUSNESS_LEVELS,
)

router = APIRouter()


@router.get("/spiritual/chakras/info")
async def chakra_info():
    """Get information about all 7 chakras and their EEG frequency mappings."""
    return {
        "chakras": {
            name: {
                "sanskrit": info["sanskrit"],
                "frequency_band_hz": info["frequency_band"],
                "color": info["color"],
                "element": info["element"],
                "qualities": info["qualities"],
                "location": info["location"],
                "mantra": info["mantra"],
            }
            for name, info in CHAKRAS.items()
        },
        "consciousness_levels": CONSCIOUSNESS_LEVELS,
    }


@router.post("/spiritual/chakras")
async def analyze_chakras(data: EEGInput):
    """Analyze chakra activation levels from EEG brainwaves."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    processed = preprocess(eeg, data.fs)

    activations = compute_chakra_activations(processed, data.fs)
    balance = compute_chakra_balance(activations)
    return _numpy_safe({"chakras": activations, "balance": balance})


@router.post("/spiritual/meditation-depth")
async def analyze_meditation(data: EEGInput):
    """Measure meditation depth from EEG patterns."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    return _numpy_safe(compute_meditation_depth(preprocess(eeg, data.fs), data.fs))


@router.post("/spiritual/aura")
async def analyze_aura(data: EEGInput):
    """Generate aura color and energy visualization from EEG."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    return _numpy_safe(compute_aura_energy(preprocess(eeg, data.fs), data.fs))


@router.post("/spiritual/kundalini")
async def analyze_kundalini(data: EEGInput):
    """Track kundalini energy flow through the chakra system."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    return _numpy_safe(compute_kundalini_flow(preprocess(eeg, data.fs), data.fs))


@router.post("/spiritual/prana-balance")
async def analyze_prana(data: EEGInput):
    """Analyze prana/chi energy balance from bilateral EEG."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    if signals.shape[0] < 2:
        mid = len(signals[0]) // 2
        eeg_left = preprocess(signals[0][:mid], data.fs)
        eeg_right = preprocess(signals[0][mid:], data.fs)
    else:
        eeg_left = preprocess(signals[0], data.fs)
        eeg_right = preprocess(signals[1], data.fs)

    return _numpy_safe(compute_prana_balance(eeg_left, eeg_right, data.fs))


@router.post("/spiritual/consciousness")
async def analyze_consciousness(data: EEGInput):
    """Estimate consciousness level from EEG patterns."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    return _numpy_safe(compute_consciousness_level(preprocess(eeg, data.fs), data.fs))


@router.post("/spiritual/third-eye")
async def analyze_third_eye(data: EEGInput):
    """Measure third eye (Ajna) activation through gamma analysis."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    return _numpy_safe(compute_third_eye_activation(preprocess(eeg, data.fs), data.fs))


@router.post("/spiritual/full-analysis")
async def full_spiritual_analysis_endpoint(data: EEGInput):
    """Complete spiritual energy analysis — all metrics in one call."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    processed = preprocess(eeg, data.fs)

    eeg_left = eeg_right = None
    if signals.shape[0] >= 2:
        eeg_left = preprocess(signals[0], data.fs)
        eeg_right = preprocess(signals[1], data.fs)

    return _numpy_safe(full_spiritual_analysis(processed, data.fs, eeg_left, eeg_right))
