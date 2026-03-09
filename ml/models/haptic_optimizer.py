"""EEG-driven haptic urgency optimization for BCI alert systems.

Adapts vibration/haptic feedback intensity and pattern to the user's
current cognitive state (arousal level) for optimal alerting without
startling or causing alert fatigue.

References:
    Brown et al. (2005) — Multisensory integration and attention
    2024 vibrotactile urgency recognition studies — 83% accuracy
"""
from typing import Dict

import numpy as np


# Haptic patterns ordered by urgency
HAPTIC_PATTERNS = {
    "gentle_tap": {
        "description": "Single short pulse",
        "duration_ms": 100,
        "pulses": 1,
    },
    "double_tap": {
        "description": "Two short pulses",
        "duration_ms": 200,
        "pulses": 2,
    },
    "standard_buzz": {
        "description": "Medium sustained vibration",
        "duration_ms": 400,
        "pulses": 1,
    },
    "urgent_pulse": {
        "description": "Three rapid pulses with increasing intensity",
        "duration_ms": 600,
        "pulses": 3,
    },
    "alarm_burst": {
        "description": "Long pulsing vibration for critical alerts",
        "duration_ms": 1000,
        "pulses": 5,
    },
}


class HapticUrgencyOptimizer:
    """Map EEG arousal state + alert priority to optimal haptic parameters."""

    PRIORITY_LEVELS = ("low", "medium", "high", "critical")

    def __init__(self):
        self._arousal_history: list = []

    def map_urgency(
        self,
        arousal: float,
        alert_priority: str = "medium",
        drowsiness: float = 0.0,
    ) -> Dict:
        """Compute optimal haptic feedback parameters.

        Args:
            arousal: 0-1 arousal level from emotion classifier.
            alert_priority: low/medium/high/critical alert priority.
            drowsiness: 0-1 drowsiness level (optional, amplifies urgency).

        Returns:
            Dict with intensity (0-1), pattern, duration_ms, pulses,
            rationale, and effective_urgency fields.
        """
        arousal = float(np.clip(arousal, 0, 1))
        drowsiness = float(np.clip(drowsiness, 0, 1))

        # Track arousal for trend detection
        self._arousal_history.append(arousal)
        if len(self._arousal_history) > 30:
            self._arousal_history = self._arousal_history[-30:]

        # Priority multiplier
        priority_weight = {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.8,
            "critical": 1.0,
        }.get(alert_priority, 0.5)

        # Base urgency: inverse arousal (drowsy = needs stronger alert)
        base_urgency = 1.0 - arousal

        # Drowsiness amplification
        drowsiness_boost = drowsiness * 0.3

        # Effective urgency combines state and priority
        effective_urgency = float(
            np.clip(base_urgency * 0.5 + priority_weight * 0.5 + drowsiness_boost, 0, 1)
        )

        # Map urgency to intensity
        intensity = float(np.clip(0.2 + effective_urgency * 0.8, 0.1, 1.0))

        # Select pattern based on urgency level
        pattern, pattern_info = self._select_pattern(effective_urgency, alert_priority)

        # Rationale
        if arousal < 0.3:
            rationale = "Low arousal detected — stronger haptic to ensure alerting"
        elif arousal > 0.7 and alert_priority in ("low", "medium"):
            rationale = "High arousal — gentle cue to avoid startling"
        elif alert_priority == "critical":
            rationale = "Critical alert — maximum urgency regardless of state"
        elif drowsiness > 0.5:
            rationale = "Drowsiness detected — amplified haptic feedback"
        else:
            rationale = "Standard alerting for current cognitive state"

        return {
            "intensity": intensity,
            "pattern": pattern,
            "duration_ms": pattern_info["duration_ms"],
            "pulses": pattern_info["pulses"],
            "effective_urgency": effective_urgency,
            "rationale": rationale,
            "arousal_state": self._classify_arousal(arousal),
            "alert_priority": alert_priority,
        }

    def _select_pattern(self, urgency: float, priority: str):
        """Select haptic pattern based on urgency level."""
        if priority == "critical":
            name = "alarm_burst"
        elif urgency < 0.25:
            name = "gentle_tap"
        elif urgency < 0.45:
            name = "double_tap"
        elif urgency < 0.65:
            name = "standard_buzz"
        elif urgency < 0.85:
            name = "urgent_pulse"
        else:
            name = "alarm_burst"
        return name, HAPTIC_PATTERNS[name]

    def _classify_arousal(self, arousal: float) -> str:
        """Classify arousal into descriptive state."""
        if arousal < 0.2:
            return "very_low"
        elif arousal < 0.4:
            return "low"
        elif arousal < 0.6:
            return "moderate"
        elif arousal < 0.8:
            return "high"
        return "very_high"

    def get_arousal_trend(self) -> str:
        """Get recent arousal trend."""
        if len(self._arousal_history) < 6:
            return "insufficient_data"
        recent = self._arousal_history[-6:]
        older = (
            self._arousal_history[-12:-6]
            if len(self._arousal_history) >= 12
            else self._arousal_history[:6]
        )
        diff = float(np.mean(recent)) - float(np.mean(older))
        if diff > 0.1:
            return "rising"
        elif diff < -0.1:
            return "declining"
        return "stable"
