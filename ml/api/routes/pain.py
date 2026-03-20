"""Chronic pain biomarker detection from frontal EEG asymmetry.

Endpoint:
  POST /pain/detect  -- compute pain biomarker score from EEG signals

GitHub issue: #129
"""

import numpy as np
from fastapi import APIRouter

from ._shared import EEGInput, _numpy_safe
from models.pain_detector import PainDetector

router = APIRouter(tags=["pain"])

_detector = PainDetector()


@router.post("/pain/detect")
async def detect_pain_biomarkers(data: EEGInput):
    """Estimate chronic pain biomarkers from frontal EEG asymmetry.

    Uses resting-state frontal beta/alpha asymmetry as passive pain biomarkers.
    No pain stimulus required — detects patterns during any EEG session.

    Key biomarkers (Scientific Reports 2024; eBioMedicine 2025):
    - Beta asymmetry (AF8 - AF7 beta): r = -0.375 with pain severity
    - Alpha DE asymmetry: strongest differentiator across pain levels
    - High-beta mean power: elevated during chronic pain states

    Returns pain_biomarker_score (0-1), pain_level classification, and
    component scores.

    **Disclaimer**: Wellness indicator only — not a medical device or clinical assessment.
    Requires multichannel EEG (AF7 + AF8 at minimum).
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    result = _detector.predict(signals, fs=int(data.fs))
    return _numpy_safe(result)
