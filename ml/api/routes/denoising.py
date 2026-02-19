"""Denoising and artifact classification endpoints."""

from pathlib import Path

import numpy as np
from fastapi import APIRouter

from ._shared import (
    _numpy_safe,
    detect_eye_blinks, detect_muscle_artifacts, detect_electrode_pops,
    compute_signal_quality_index,
    DenoiseRequest,
)

router = APIRouter()


def _get_denoiser_status() -> bool:
    return Path("models/saved/denoiser_model.pt").exists()


def _get_artifact_classifier_status() -> bool:
    return Path("models/saved/artifact_classifier_model.pkl").exists()


def _summarize_artifacts(classifications: list) -> dict:
    from collections import Counter
    types = [c["artifact_type"] for c in classifications]
    counts = Counter(types)
    total = len(types)
    return {
        "total_windows": total,
        "clean_windows": counts.get("clean", 0),
        "clean_ratio": counts.get("clean", 0) / max(total, 1),
        "artifact_counts": dict(counts),
    }


@router.post("/denoise")
async def denoise_signal(req: DenoiseRequest):
    """Denoise EEG signals using the trained autoencoder (falls back to classical filter)."""
    from processing.eeg_processor import preprocess_robust

    results = []
    for ch_data in req.signals:
        signal = np.array(ch_data, dtype=np.float64)
        cleaned = preprocess_robust(signal, req.fs, use_denoiser=True)
        results.append(cleaned.tolist())

    return _numpy_safe({
        "cleaned_signals": results,
        "n_channels": len(results),
        "method": "ml_denoiser" if _get_denoiser_status() else "classical_filter",
    })


@router.post("/classify-artifacts")
async def classify_artifacts(req: DenoiseRequest):
    """Classify artifact types in EEG signal segments."""
    try:
        from models.artifact_classifier import ArtifactClassifier
        classifier = ArtifactClassifier(model_path="models/saved/artifact_classifier_model.pkl")
    except Exception:
        classifier = None

    if classifier is None or classifier.model is None:
        results = []
        for ch_data in req.signals:
            signal = np.array(ch_data, dtype=np.float64)
            results.append({
                "sqi": compute_signal_quality_index(signal, req.fs),
                "eye_blinks": len(detect_eye_blinks(signal, req.fs)),
                "muscle_artifacts": len(detect_muscle_artifacts(signal, req.fs)),
                "electrode_pops": len(detect_electrode_pops(signal, req.fs)),
                "method": "heuristic",
            })
        return _numpy_safe({"channels": results})

    all_results = []
    for ch_data in req.signals:
        signal = np.array(ch_data, dtype=np.float64)
        classifications = classifier.classify_signal(signal, req.fs, window_sec=1.0)
        all_results.append({
            "sqi": compute_signal_quality_index(signal, req.fs),
            "windows": classifications,
            "artifact_summary": _summarize_artifacts(classifications),
            "method": "ml_classifier",
        })

    return _numpy_safe({"channels": all_results})


@router.get("/denoise/status")
async def denoise_status():
    """Check availability of ML denoiser and artifact classifier models."""
    return {
        "denoiser_available": _get_denoiser_status(),
        "artifact_classifier_available": _get_artifact_classifier_status(),
        "denoiser_model_path": "models/saved/denoiser_model.pt",
        "artifact_model_path": "models/saved/artifact_classifier_model.pkl",
    }
