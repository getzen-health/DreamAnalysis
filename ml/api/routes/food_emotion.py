"""Food-emotion EEG endpoints — real-time food state prediction and dietary guidance.

Endpoints
---------
POST /predict-food-emotion
    Predict current food-motivation state from raw EEG signals or simulated EEG.
    Pass ``simulate: true`` (and no ``signals``) for demo/simulation mode.

POST /food-emotion/calibrate
    Record a resting-state baseline so relative EEG changes are interpreted
    correctly per user.

GET  /food-emotion/recommendations/{food_state}
    Return static food and behaviour recommendations for a named food state.
"""

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ._shared import (
    _numpy_safe,
    food_emotion_model,
)
from simulation.eeg_simulator import simulate_eeg
from models.food_emotion_predictor import _RECOMMENDATIONS

router = APIRouter()

_VALID_FOOD_STATES = frozenset(_RECOMMENDATIONS.keys())


class FoodEmotionRequest(BaseModel):
    """Request body for food-emotion prediction and calibration.

    Either provide real EEG ``signals`` (channels × samples) *or* set
    ``simulate: true`` to generate synthetic EEG for demonstration purposes.
    When ``simulate`` is true the response will include ``simulation_mode: true``.
    """

    signals: Optional[List[List[float]]] = Field(
        default=None,
        description="EEG signals — shape (n_channels, n_samples). If None, set simulate=true.",
    )
    fs: float = Field(default=256.0, description="Sampling frequency in Hz")
    simulate: bool = Field(
        default=False,
        description="Generate simulated EEG when no real signals are available (demo mode).",
    )
    state: str = Field(
        default="rest",
        description="Brain state for EEG simulation. Ignored when real signals are supplied.",
    )


def _signals_from_request(req: FoodEmotionRequest, duration: float = 4.0) -> tuple[np.ndarray, bool]:
    """Return (signals_array, is_simulated) from a FoodEmotionRequest.

    Raises HTTPException 422 if neither signals nor simulate=True is provided.
    """
    if req.signals is not None:
        signals = np.array(req.signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)
        return signals, False

    if req.simulate:
        sim = simulate_eeg(state=req.state, fs=req.fs, duration=duration, n_channels=4)
        signals = np.array(sim["signals"], dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)
        return signals, True

    raise HTTPException(
        status_code=422,
        detail="Either provide 'signals' (EEG data) or set 'simulate': true for demo mode.",
    )


@router.post("/predict-food-emotion")
async def predict_food_emotion(req: FoodEmotionRequest):
    """Return the current food-motivation state predicted from EEG signals.

    The model extracts four EEG biomarkers — Frontal Alpha Asymmetry (FAA),
    high-beta power, prefrontal theta, and delta — and maps them to one of
    six food states:
      craving_carbs | appetite_suppressed | comfort_seeking |
      balanced | stress_eating | mindful_eating

    Supply real EEG via ``signals`` or pass ``simulate: true`` for demo mode.
    Returns the predicted state, per-state probabilities, dietary
    recommendations, and the raw biomarker values used.
    """
    signals, simulated = _signals_from_request(req, duration=4.0)
    result = food_emotion_model.predict(signals, req.fs)
    if simulated:
        result["simulation_mode"] = True
    return _numpy_safe(result)


@router.post("/food-emotion/calibrate")
async def calibrate_food_emotion(req: FoodEmotionRequest):
    """Record a resting-state EEG baseline for the food-emotion model.

    Call this endpoint once with 30+ seconds of quiet, resting EEG before
    starting a monitoring session.  The model stores per-user baseline FAA,
    high-beta, theta, and delta values and uses them to normalise subsequent
    predictions, substantially improving accuracy.

    Supply real EEG via ``signals`` or pass ``simulate: true`` for demo/testing.
    Returns the stored baseline values and a ``calibrated`` flag.
    """
    signals, simulated = _signals_from_request(req, duration=30.0)
    result = food_emotion_model.calibrate(signals, req.fs)
    if simulated:
        result["simulation_mode"] = True
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
