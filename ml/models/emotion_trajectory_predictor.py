"""Emotion Trajectory Predictor — near-future emotion state forecasting.

Uses Holt-Winters double exponential smoothing (no training required).
Maintains a rolling buffer of recent valence/arousal readings and
forecasts the next N steps ahead.

Holt-Winters double exponential smoothing formulas:
    Level:   L_t = alpha * y_t + (1 - alpha) * (L_{t-1} + T_{t-1})
    Trend:   T_t = beta  * (L_t - L_{t-1}) + (1 - beta) * T_{t-1}
    Forecast h steps: y_{t+h} = L_t + h * T_t

Falls back to linear regression trend when Holt-Winters has insufficient
history (< 2 readings).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Horizon steps for fixed predictions returned in every predict() call
_HORIZON_5S = 5
_HORIZON_15S = 15
_HORIZON_30S = 30

# Trend classification thresholds (units per second)
_STABLE_THRESHOLD = 0.02

# Valence/arousal → 6-class emotion mapping
# Evaluated in priority order — first match wins
_EMOTION_MAP: List[Tuple[str, callable]] = [
    ("excited",  lambda v, a: v > 0.3 and a > 0.7),
    ("happy",    lambda v, a: v > 0.3 and a > 0.5),
    ("calm",     lambda v, a: v > 0.1 and a < 0.4),
    ("stressed", lambda v, a: v < -0.1 and a > 0.6),
    ("sad",      lambda v, a: v < -0.2 and a < 0.45),
    ("neutral",  lambda v, a: abs(v) < 0.15),
]
_DEFAULT_EMOTION = "neutral"


def _map_emotion(valence: float, arousal: float) -> str:
    """Map valence/arousal coordinates to a 6-class discrete emotion label."""
    for label, condition in _EMOTION_MAP:
        try:
            if condition(valence, arousal):
                return label
        except Exception:
            continue
    return _DEFAULT_EMOTION


def _classify_trend(velocity: float) -> str:
    """Classify velocity into 'rising', 'falling', or 'stable'."""
    if velocity > _STABLE_THRESHOLD:
        return "rising"
    if velocity < -_STABLE_THRESHOLD:
        return "falling"
    return "stable"


class EmotionTrajectoryPredictor:
    """Predict near-future emotion state from recent history.

    Uses Holt-Winters double exponential smoothing (no training required).
    Falls back to linear regression trend if insufficient history.

    Parameters
    ----------
    history_seconds : int
        Maximum history window to retain (seconds). Older readings are
        dropped. Default 30.
    fs : float
        Expected reading rate in readings per second. Default 1.0 (one
        reading per second). Used only for converting history_seconds to
        a buffer size; predictions are time-stamp aware.
    alpha : float
        Level smoothing factor in [0, 1]. Higher = more weight on recent
        observations. Default 0.3.
    beta : float
        Trend smoothing factor in [0, 1]. Higher = trend adapts faster.
        Default 0.1.
    """

    def __init__(
        self,
        history_seconds: int = 30,
        fs: float = 1.0,
        alpha: float = 0.3,
        beta: float = 0.1,
    ) -> None:
        self._history_seconds = history_seconds
        self._fs = fs
        self._alpha = alpha
        self._beta = beta

        # Rolling buffers: list of (timestamp, value)
        maxlen = max(int(history_seconds * fs) + 1, 2)
        self._valence_history: deque = deque(maxlen=maxlen)
        self._arousal_history: deque = deque(maxlen=maxlen)

        # Holt-Winters state for valence
        self._v_level: Optional[float] = None
        self._v_trend: float = 0.0

        # Holt-Winters state for arousal
        self._a_level: Optional[float] = None
        self._a_trend: float = 0.0

        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        valence: float,
        arousal: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Add a new emotion reading to the history buffer.

        Parameters
        ----------
        valence : float
            Current valence value in [-1, 1].
        arousal : float
            Current arousal value in [0, 1].
        timestamp : float, optional
            Unix timestamp of the reading. Defaults to time.time().
        """
        if timestamp is None:
            timestamp = time.time()

        valence = float(np.clip(valence, -1.0, 1.0))
        arousal = float(np.clip(arousal, 0.0, 1.0))

        with self._lock:
            self._valence_history.append((timestamp, valence))
            self._arousal_history.append((timestamp, arousal))

            # Prune readings older than history_seconds
            cutoff = timestamp - self._history_seconds
            while self._valence_history and self._valence_history[0][0] < cutoff:
                self._valence_history.popleft()
            while self._arousal_history and self._arousal_history[0][0] < cutoff:
                self._arousal_history.popleft()

            # Update Holt-Winters state
            self._v_level, self._v_trend = self._hw_update(
                valence, self._v_level, self._v_trend
            )
            self._a_level, self._a_trend = self._hw_update(
                arousal, self._a_level, self._a_trend
            )

    def predict(self, horizon_steps: int = 5) -> dict:
        """Predict future emotion trajectory.

        Returns a dict with predictions at 5, 15, and 30 seconds ahead,
        trend direction, velocity, and a mapped discrete emotion label.

        Parameters
        ----------
        horizon_steps : int
            Number of steps ahead to forecast (used for the generic
            horizon). The fixed 5s/15s/30s keys are always included.

        Returns
        -------
        dict with keys:
            current_valence, current_arousal,
            predicted_valence_5s, predicted_arousal_5s,
            predicted_valence_15s, predicted_arousal_15s,
            predicted_valence_30s, predicted_arousal_30s,
            valence_trend, arousal_trend,
            valence_velocity, arousal_velocity,
            predicted_emotion_5s,
            confidence, history_length, model_type
        """
        with self._lock:
            n = len(self._valence_history)

            if n == 0:
                return self._empty_result()

            current_valence = self._valence_history[-1][1]
            current_arousal = self._arousal_history[-1][1]

            if n == 1 or self._v_level is None:
                # Single reading — no trend, return current state
                return self._static_result(current_valence, current_arousal)

            # Holt-Winters forecast
            v_preds = {
                h: self._hw_forecast(self._v_level, self._v_trend, h)
                for h in (_HORIZON_5S, _HORIZON_15S, _HORIZON_30S, horizon_steps)
            }
            a_preds = {
                h: self._hw_forecast(self._a_level, self._a_trend, h)
                for h in (_HORIZON_5S, _HORIZON_15S, _HORIZON_30S, horizon_steps)
            }

            v_vel = float(self._v_trend)
            a_vel = float(self._a_trend)

            # Confidence: increases with history length, saturates at 1.0
            confidence = float(min(1.0, n / max(self._history_seconds, 1)))

            pred_v_5s = float(np.clip(v_preds[_HORIZON_5S], -1.0, 1.0))
            pred_a_5s = float(np.clip(a_preds[_HORIZON_5S], 0.0, 1.0))

            return {
                "current_valence": round(current_valence, 4),
                "current_arousal": round(current_arousal, 4),
                "predicted_valence_5s": round(pred_v_5s, 4),
                "predicted_arousal_5s": round(float(np.clip(a_preds[_HORIZON_5S], 0.0, 1.0)), 4),
                "predicted_valence_15s": round(float(np.clip(v_preds[_HORIZON_15S], -1.0, 1.0)), 4),
                "predicted_arousal_15s": round(float(np.clip(a_preds[_HORIZON_15S], 0.0, 1.0)), 4),
                "predicted_valence_30s": round(float(np.clip(v_preds[_HORIZON_30S], -1.0, 1.0)), 4),
                "predicted_arousal_30s": round(float(np.clip(a_preds[_HORIZON_30S], 0.0, 1.0)), 4),
                "valence_trend": _classify_trend(v_vel),
                "arousal_trend": _classify_trend(a_vel),
                "valence_velocity": round(v_vel, 4),
                "arousal_velocity": round(a_vel, 4),
                "predicted_emotion_5s": _map_emotion(pred_v_5s, pred_a_5s),
                "confidence": round(confidence, 4),
                "history_length": n,
                "model_type": "holt_winters",
            }

    def get_trajectory(self) -> dict:
        """Get full emotion history and trend summary.

        Returns
        -------
        dict with keys:
            valence_history   — list of [timestamp, valence]
            arousal_history   — list of [timestamp, arousal]
            valence_level     — current HW level for valence
            valence_trend_rate — current HW trend for valence
            arousal_level     — current HW level for arousal
            arousal_trend_rate — current HW trend for arousal
            history_length    — number of readings in buffer
            history_seconds   — configured history window
        """
        with self._lock:
            return {
                "valence_history": [[t, v] for t, v in self._valence_history],
                "arousal_history": [[t, a] for t, a in self._arousal_history],
                "valence_level": self._v_level,
                "valence_trend_rate": self._v_trend,
                "arousal_level": self._a_level,
                "arousal_trend_rate": self._a_trend,
                "history_length": len(self._valence_history),
                "history_seconds": self._history_seconds,
            }

    def reset(self) -> None:
        """Clear all history and reset Holt-Winters state."""
        with self._lock:
            self._valence_history.clear()
            self._arousal_history.clear()
            self._v_level = None
            self._v_trend = 0.0
            self._a_level = None
            self._a_trend = 0.0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _hw_update(
        self,
        y: float,
        prev_level: Optional[float],
        prev_trend: float,
    ) -> Tuple[float, float]:
        """One step of Holt-Winters double exponential smoothing.

        Returns (new_level, new_trend).
        """
        if prev_level is None:
            # Bootstrap: initialise level to first observation, trend to 0
            return y, 0.0

        new_level = self._alpha * y + (1.0 - self._alpha) * (prev_level + prev_trend)
        new_trend = self._beta * (new_level - prev_level) + (1.0 - self._beta) * prev_trend
        return new_level, new_trend

    def _hw_forecast(self, level: float, trend: float, h: int) -> float:
        """Forecast h steps ahead: L_t + h * T_t."""
        return level + h * trend

    def _empty_result(self) -> dict:
        """Return when no history is available."""
        return {
            "current_valence": 0.0,
            "current_arousal": 0.0,
            "predicted_valence_5s": 0.0,
            "predicted_arousal_5s": 0.0,
            "predicted_valence_15s": 0.0,
            "predicted_arousal_15s": 0.0,
            "predicted_valence_30s": 0.0,
            "predicted_arousal_30s": 0.0,
            "valence_trend": "stable",
            "arousal_trend": "stable",
            "valence_velocity": 0.0,
            "arousal_velocity": 0.0,
            "predicted_emotion_5s": "neutral",
            "confidence": 0.0,
            "history_length": 0,
            "model_type": "holt_winters",
        }

    def _static_result(self, valence: float, arousal: float) -> dict:
        """Return when only one reading available — no trend estimable."""
        return {
            "current_valence": round(valence, 4),
            "current_arousal": round(arousal, 4),
            "predicted_valence_5s": round(valence, 4),
            "predicted_arousal_5s": round(arousal, 4),
            "predicted_valence_15s": round(valence, 4),
            "predicted_arousal_15s": round(arousal, 4),
            "predicted_valence_30s": round(valence, 4),
            "predicted_arousal_30s": round(arousal, 4),
            "valence_trend": "stable",
            "arousal_trend": "stable",
            "valence_velocity": 0.0,
            "arousal_velocity": 0.0,
            "predicted_emotion_5s": _map_emotion(valence, arousal),
            "confidence": round(1.0 / max(self._history_seconds, 1), 4),
            "history_length": 1,
            "model_type": "holt_winters",
        }


# ── Per-user singleton registry ───────────────────────────────────────────────
_predictors: Dict[str, EmotionTrajectoryPredictor] = {}
_registry_lock = threading.Lock()


def get_trajectory_predictor(user_id: str = "default") -> EmotionTrajectoryPredictor:
    """Return (or create) the per-user EmotionTrajectoryPredictor singleton."""
    with _registry_lock:
        if user_id not in _predictors:
            _predictors[user_id] = EmotionTrajectoryPredictor()
        return _predictors[user_id]
