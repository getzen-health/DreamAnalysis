"""Anomaly detector for health/brain metrics.

Computes z-scores against a 30-day rolling baseline and surfaces plain-English
descriptions of any statistically notable deviations.  Designed to be called
from the morning notification pipeline so users get actionable nudges when
something is genuinely off — not generic daily tips.

Usage
-----
    from notifications.anomaly_detector import AnomalyDetector

    detector = AnomalyDetector()
    anomalies = detector.detect_anomalies(user_data, baseline)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

log = logging.getLogger(__name__)

# ── Metric configuration ───────────────────────────────────────────────────

# Metrics we track, keyed by canonical name.
# Each entry: (user_data_key, baseline_key, friendly_name, unit, low_is_bad)
#   low_is_bad=True  → below-average is the concerning direction
#   low_is_bad=False → above-average is the concerning direction
#   low_is_bad=None  → both directions are notable
_METRIC_CONFIG: Dict[str, tuple] = {
    "sleep_quality": (
        "sleep_quality",
        "sleep_quality",
        "sleep quality",
        "%",
        True,
    ),
    "voice_valence": (
        "voice_valence",
        "voice_valence",
        "voice valence (mood)",
        "",
        True,
    ),
    "hrv": (
        "hrv_avg",
        "hrv_avg",
        "heart-rate variability (HRV)",
        "ms",
        True,
    ),
    "dream_recall": (
        "dream_recall_rate",
        "dream_recall_rate",
        "dream recall rate",
        "dreams/night",
        None,
    ),
    "stress_index": (
        "stress_index",
        "stress_index",
        "stress index",
        "",
        False,
    ),
}

# Default z-score threshold that triggers an anomaly
DEFAULT_Z_THRESHOLD: float = 1.5

# Minimum baseline std to avoid division-by-very-small-number noise
_MIN_STD: float = 1e-6


# ── Anomaly dataclass ──────────────────────────────────────────────────────

@dataclass
class Anomaly:
    """A single detected anomaly for one metric."""

    metric: str                        # canonical metric name
    value: float                       # today's observed value
    baseline_mean: float               # 30-day rolling mean
    baseline_std: float                # 30-day rolling std
    z_score: float                     # (value - mean) / std
    direction: Literal["above", "below"]  # which side of the baseline
    description: str                   # plain-English explanation


# ── AnomalyDetector ──────────────────────────────────────────────────────

class AnomalyDetector:
    """Detects metric anomalies via z-score against a rolling baseline.

    Parameters
    ----------
    z_threshold:
        Absolute z-score needed for a metric to be flagged as anomalous.
        Default is 1.5 SD (roughly the top/bottom ~7% of a normal distribution).
    """

    def __init__(self, z_threshold: float = DEFAULT_Z_THRESHOLD) -> None:
        self.z_threshold = z_threshold

    # ── Public API ────────────────────────────────────────────────────────

    def detect_anomalies(
        self,
        user_data: Dict[str, Any],
        baseline: Dict[str, Any],
    ) -> List[Anomaly]:
        """Return a list of metrics that deviate significantly from the baseline.

        Parameters
        ----------
        user_data:
            Today's (or the most-recent session's) metric values.
            Expected keys: sleep_quality, voice_valence, hrv_avg,
            dream_recall_rate, stress_index.
            Values should be floats or None.
        baseline:
            30-day rolling baseline dict with keys matching user_data plus
            ``_{metric}_std`` for standard deviations.
            E.g. ``{"sleep_quality": 0.72, "sleep_quality_std": 0.08, ...}``.

        Returns
        -------
        list of Anomaly, sorted by abs(z_score) descending.
        """
        anomalies: List[Anomaly] = []

        for metric_key, (ukey, bkey, friendly, unit, low_is_bad) in _METRIC_CONFIG.items():
            raw_value = user_data.get(ukey)
            raw_mean = baseline.get(bkey)
            raw_std = baseline.get(f"{bkey}_std")

            # Skip if any component is missing
            if raw_value is None or raw_mean is None:
                continue

            try:
                value = float(raw_value)
                mean = float(raw_mean)
                std = float(raw_std) if raw_std is not None else 0.0
            except (TypeError, ValueError):
                continue

            # Need a meaningful std to compute a real z-score
            if std < _MIN_STD:
                continue

            z = (value - mean) / std
            abs_z = abs(z)

            if abs_z < self.z_threshold:
                continue

            direction: Literal["above", "below"] = "above" if z > 0 else "below"

            # Decide whether this direction is notable given the metric's polarity
            if low_is_bad is True and direction == "above":
                # Better than usual — still surface but framed positively
                pass
            elif low_is_bad is False and direction == "below":
                # Lower stress than usual — positive anomaly
                pass

            description = self._build_description(
                friendly, value, mean, std, z, direction, unit, low_is_bad
            )

            anomalies.append(
                Anomaly(
                    metric=metric_key,
                    value=value,
                    baseline_mean=mean,
                    baseline_std=std,
                    z_score=round(z, 2),
                    direction=direction,
                    description=description,
                )
            )

        # Sort by abs(z_score) descending — most extreme first
        anomalies.sort(key=lambda a: abs(a.z_score), reverse=True)
        return anomalies

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _format_value(value: float, unit: str) -> str:
        """Format a metric value with its unit for a human-readable string."""
        if unit == "%":
            return f"{value * 100:.0f}%"
        if unit == "ms":
            return f"{value:.0f} ms"
        if unit == "dreams/night":
            return f"{value:.1f}"
        # valence and stress are -1…1 or 0…1 — show as percentage points
        return f"{value:.2f}"

    @staticmethod
    def _build_description(
        friendly: str,
        value: float,
        mean: float,
        std: float,
        z: float,
        direction: Literal["above", "below"],
        unit: str,
        low_is_bad: Optional[bool],
    ) -> str:
        """Produce a plain-English sentence describing the anomaly."""
        abs_z = abs(z)

        if abs_z >= 3.0:
            severity = "significantly"
        elif abs_z >= 2.0:
            severity = "notably"
        else:
            severity = "slightly"

        # Format value strings
        val_str = AnomalyDetector._format_value(value, unit)
        mean_str = AnomalyDetector._format_value(mean, unit)

        # Determine qualitative framing
        if low_is_bad is True:
            # Higher = better (sleep quality, HRV, valence)
            good_direction = "above"
        elif low_is_bad is False:
            # Lower = better (stress index)
            good_direction = "below"
        else:
            good_direction = None  # dream recall — neutral framing

        if good_direction == direction:
            valence_word = "higher than usual"
        elif good_direction is not None:
            valence_word = "lower than usual"
        else:
            valence_word = f"{direction} average"

        return (
            f"Your {friendly} ({val_str}) is {severity} {valence_word} "
            f"— your 30-day baseline is {mean_str} "
            f"(z-score {z:+.1f})."
        )
