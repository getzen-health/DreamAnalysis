"""Food-emotion EEG endpoints — real-time food state prediction and dietary guidance.

Endpoints
---------
POST /predict-food-emotion
    Predict current food-motivation state from raw EEG signals.

POST /food-emotion/calibrate
    Record a resting-state baseline so relative EEG changes are interpreted
    correctly per user.

GET  /food-emotion/recommendations/{food_state}
    Return static food and behaviour recommendations for a named food state.
"""

import numpy as np
from fastapi import APIRouter, HTTPException

from ._shared import (
    _numpy_safe,
    food_emotion_model,
    EEGInput,
)
from models.food_emotion_predictor import _RECOMMENDATIONS

router = APIRouter()

_VALID_FOOD_STATES = frozenset(_RECOMMENDATIONS.keys())


@router.post("/predict-food-emotion")
async def predict_food_emotion(req: EEGInput):
    """Return the current food-motivation state predicted from EEG signals.

    The model extracts four EEG biomarkers — Frontal Alpha Asymmetry (FAA),
    high-beta power, prefrontal theta, and delta — and maps them to one of
    six food states:
      craving_carbs | appetite_suppressed | comfort_seeking |
      balanced | stress_eating | mindful_eating

    Returns the predicted state, per-state probabilities, dietary
    recommendations, and the raw biomarker values used.
    """
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    result = food_emotion_model.predict(signals, req.fs)
    return _numpy_safe(result)


@router.post("/food-emotion/calibrate")
async def calibrate_food_emotion(req: EEGInput):
    """Record a resting-state EEG baseline for the food-emotion model.

    Call this endpoint once with 30+ seconds of quiet, resting EEG before
    starting a monitoring session.  The model stores per-user baseline FAA,
    high-beta, theta, and delta values and uses them to normalise subsequent
    predictions, substantially improving accuracy.

    Returns the stored baseline values and a ``calibrated`` flag.
    """
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    result = food_emotion_model.calibrate(signals, req.fs)
    return _numpy_safe(result)


@router.get("/food-emotion/recommendations/{food_state}")
async def get_food_recommendations(food_state: str):
    """Return dietary and behavioural recommendations for a food state.

    Valid food states:
      craving_carbs, appetite_suppressed, comfort_seeking,
      balanced, stress_eating, mindful_eating

    Each recommendation bundle contains:
      - ``avoid``          (list[str]): foods/behaviours to avoid
      - ``prefer``         (list[str]): foods/behaviours to favour
      - ``strategy``       (str):       short behavioural strategy
      - ``mindfulness_tip``(str):       mindfulness or breath cue
    """
    if food_state not in _VALID_FOOD_STATES:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown food state '{food_state}'. "
                f"Valid states: {sorted(_VALID_FOOD_STATES)}"
            ),
        )
    return _numpy_safe(_RECOMMENDATIONS[food_state])
