"""Wavelet analysis and signal cleaning endpoints."""

import numpy as np
from fastapi import APIRouter, HTTPException

from ._shared import (
    EEGInput,
    preprocess,
    compute_cwt_spectrogram, compute_dwt_features,
    detect_sleep_spindles, detect_k_complexes,
    compute_signal_quality_index, ica_artifact_removal,
)

router = APIRouter()


@router.post("/analyze-wavelet")
async def analyze_wavelet(data: EEGInput):
    """Wavelet time-frequency analysis: CWT spectrogram, DWT energies, event detection."""
    try:
        signals = np.array(data.signals)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        eeg = signals[0]
        fs = data.fs

        if len(eeg) < 34:
            raise HTTPException(
                status_code=422,
                detail=f"Signal too short ({len(eeg)} samples). Need at least 34.",
            )

        processed = preprocess(eeg, fs)

        return {
            "spectrogram": compute_cwt_spectrogram(processed, fs),
            "dwt_energies": compute_dwt_features(processed, fs),
            "events": {
                "sleep_spindles": detect_sleep_spindles(processed, fs),
                "k_complexes": detect_k_complexes(processed, fs),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clean-signal")
async def clean_signal(data: EEGInput):
    """ICA-based artifact removal returning cleaned signals + report."""
    try:
        signals = np.array(data.signals)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        fs = data.fs
        result = ica_artifact_removal(signals, fs)

        before_sqi = [compute_signal_quality_index(signals[ch], fs) for ch in range(signals.shape[0])]
        cleaned = result["cleaned_signals"]
        after_sqi = [compute_signal_quality_index(cleaned[ch], fs) for ch in range(cleaned.shape[0])]

        return {
            "cleaned_signals": cleaned.tolist(),
            "removed_components": result["removed_components"],
            "n_components": result["n_components"],
            "before_sqi": before_sqi,
            "after_sqi": after_sqi,
            "improvement": float(np.mean(after_sqi) - np.mean(before_sqi)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
