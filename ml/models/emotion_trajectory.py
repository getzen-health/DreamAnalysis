"""Emotion trajectory tracking and prediction.

Tracks the user's emotional state over time in valence-arousal space
and predicts short-term emotional trajectories using momentum-based
forecasting. Detects emotional transitions, stability patterns, and
provides trend analysis.

References:
    Kuppens et al. (2010) — Emotional inertia and psychological adjustment
    Houben et al. (2015) — Emotion differentiation and well-being
"""
from typing import Dict, List, Optional

import numpy as np


class EmotionTrajectoryTracker:
    """Track emotion dynamics over time in valence-arousal space.

    Monitors emotional state changes, detects transitions, computes
    inertia (resistance to change), and predicts short-term trajectories.
    """

    def __init__(self, max_history: int = 300):
        self._max_history = max_history
        self._history: Dict[str, List[Dict]] = {}

    def update(
        self,
        valence: float,
        arousal: float,
        emotion: str = "unknown",
        timestamp: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record a new emotional state and compute trajectory metrics.

        Args:
            valence: -1 to 1 (negative to positive).
            arousal: 0 to 1 (calm to energetic).
            emotion: Discrete emotion label.
            timestamp: Unix timestamp (defaults to sequential index).
            user_id: User identifier.

        Returns:
            Dict with velocity, acceleration, inertia, quadrant,
            transition_detected, and predicted_state fields.
        """
        valence = float(np.clip(valence, -1, 1))
        arousal = float(np.clip(arousal, 0, 1))

        if user_id not in self._history:
            self._history[user_id] = []

        history = self._history[user_id]
        idx = len(history)

        entry = {
            "valence": valence,
            "arousal": arousal,
            "emotion": emotion,
            "index": idx,
        }
        history.append(entry)

        if len(history) > self._max_history:
            self._history[user_id] = history[-self._max_history:]
            history = self._history[user_id]

        # Compute dynamics
        velocity = self._compute_velocity(history)
        acceleration = self._compute_acceleration(history)
        inertia = self._compute_inertia(history)
        quadrant = self._classify_quadrant(valence, arousal)
        transition = self._detect_transition(history)
        prediction = self._predict_next(history)
        stability = self._compute_stability(history)

        return {
            "valence": valence,
            "arousal": arousal,
            "emotion": emotion,
            "quadrant": quadrant,
            "velocity_valence": round(velocity[0], 4),
            "velocity_arousal": round(velocity[1], 4),
            "speed": round(float(np.sqrt(velocity[0]**2 + velocity[1]**2)), 4),
            "acceleration_valence": round(acceleration[0], 4),
            "acceleration_arousal": round(acceleration[1], 4),
            "emotional_inertia": round(inertia, 4),
            "stability_score": round(stability, 4),
            "transition_detected": transition,
            "predicted_valence": round(prediction[0], 4),
            "predicted_arousal": round(prediction[1], 4),
            "n_samples": len(history),
        }

    def get_trajectory(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Get recent trajectory points.

        Args:
            user_id: User identifier.
            last_n: Number of most recent points (None = all).

        Returns:
            List of {valence, arousal, emotion} dicts.
        """
        history = self._history.get(user_id, [])
        if last_n:
            history = history[-last_n:]
        return [
            {"valence": h["valence"], "arousal": h["arousal"], "emotion": h["emotion"]}
            for h in history
        ]

    def get_summary(self, user_id: str = "default") -> Dict:
        """Get trajectory summary statistics.

        Returns:
            Dict with mean/std valence/arousal, dominant quadrant,
            transition count, and emotional range.
        """
        history = self._history.get(user_id, [])
        if not history:
            return {
                "n_samples": 0,
                "mean_valence": 0,
                "mean_arousal": 0,
                "dominant_quadrant": "neutral",
            }

        valences = [h["valence"] for h in history]
        arousals = [h["arousal"] for h in history]

        # Count quadrant transitions
        quadrants = [self._classify_quadrant(h["valence"], h["arousal"]) for h in history]
        transitions = sum(1 for i in range(1, len(quadrants)) if quadrants[i] != quadrants[i-1])

        # Dominant quadrant
        from collections import Counter
        dominant = Counter(quadrants).most_common(1)[0][0]

        return {
            "n_samples": len(history),
            "mean_valence": round(float(np.mean(valences)), 4),
            "std_valence": round(float(np.std(valences)), 4),
            "mean_arousal": round(float(np.mean(arousals)), 4),
            "std_arousal": round(float(np.std(arousals)), 4),
            "dominant_quadrant": dominant,
            "quadrant_transitions": transitions,
            "emotional_range": round(
                float(np.ptp(valences) + np.ptp(arousals)) / 2, 4
            ),
        }

    def reset(self, user_id: str = "default"):
        """Clear trajectory history for a user."""
        self._history.pop(user_id, None)

    def _compute_velocity(self, history: List[Dict]):
        """Velocity = rate of change in valence-arousal space."""
        if len(history) < 2:
            return (0.0, 0.0)
        curr = history[-1]
        prev = history[-2]
        return (
            curr["valence"] - prev["valence"],
            curr["arousal"] - prev["arousal"],
        )

    def _compute_acceleration(self, history: List[Dict]):
        """Acceleration = change in velocity."""
        if len(history) < 3:
            return (0.0, 0.0)
        v1 = (
            history[-2]["valence"] - history[-3]["valence"],
            history[-2]["arousal"] - history[-3]["arousal"],
        )
        v2 = (
            history[-1]["valence"] - history[-2]["valence"],
            history[-1]["arousal"] - history[-2]["arousal"],
        )
        return (v2[0] - v1[0], v2[1] - v1[1])

    def _compute_inertia(self, history: List[Dict]) -> float:
        """Emotional inertia = autocorrelation of valence.

        High inertia = emotions change slowly (resistance to change).
        """
        if len(history) < 5:
            return 0.5  # default moderate inertia
        valences = [h["valence"] for h in history[-20:]]
        if np.std(valences) < 1e-6:
            return 1.0  # perfectly stable
        v = np.array(valences)
        autocorr = float(np.corrcoef(v[:-1], v[1:])[0, 1])
        return float(np.clip(autocorr, 0, 1))

    def _compute_stability(self, history: List[Dict]) -> float:
        """Stability = inverse of recent variance (0-1)."""
        if len(history) < 3:
            return 1.0
        recent = history[-10:]
        v_std = np.std([h["valence"] for h in recent])
        a_std = np.std([h["arousal"] for h in recent])
        total_std = (v_std + a_std) / 2
        return float(np.clip(1 - total_std * 2, 0, 1))

    def _classify_quadrant(self, valence: float, arousal: float) -> str:
        """Classify valence-arousal into Russell's circumplex quadrants."""
        if valence >= 0 and arousal >= 0.5:
            return "high_positive"    # excited, happy, elated
        elif valence >= 0 and arousal < 0.5:
            return "low_positive"     # calm, relaxed, serene
        elif valence < 0 and arousal >= 0.5:
            return "high_negative"    # angry, anxious, stressed
        else:
            return "low_negative"     # sad, bored, depressed

    def _detect_transition(self, history: List[Dict]) -> bool:
        """Detect if a quadrant transition just occurred."""
        if len(history) < 2:
            return False
        curr_q = self._classify_quadrant(history[-1]["valence"], history[-1]["arousal"])
        prev_q = self._classify_quadrant(history[-2]["valence"], history[-2]["arousal"])
        return curr_q != prev_q

    def _predict_next(self, history: List[Dict]):
        """Predict next valence-arousal using momentum (linear extrapolation)."""
        if len(history) < 2:
            return (
                history[-1]["valence"] if history else 0,
                history[-1]["arousal"] if history else 0.5,
            )
        velocity = self._compute_velocity(history)
        # Damped prediction (0.5 momentum factor)
        pred_v = float(np.clip(history[-1]["valence"] + velocity[0] * 0.5, -1, 1))
        pred_a = float(np.clip(history[-1]["arousal"] + velocity[1] * 0.5, 0, 1))
        return (pred_v, pred_a)
