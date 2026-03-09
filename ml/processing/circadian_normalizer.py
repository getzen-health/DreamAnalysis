"""Circadian-aware EEG feature normalization.

Corrects EEG band power features for time-of-day and chronotype effects.
Theta and alpha power vary systematically with hours since wake due to
homeostatic sleep pressure. Without correction, same-person EEG at 8 AM
vs 11 PM produces different emotion classifications.

Based on:
    - Fractal EEG circadian rhythms (Frontiers in Physiology, 2018)
    - Chronotype synchrony effect review (Chronobiology International, 2025)
    - 65 studies, 224,714 participants confirming circadian modulation

Integration:
    1. Estimate chronotype from Apple Health sleep data
    2. Correct raw EEG features for time-of-day drift
    3. Feed corrected features into BaselineCalibrator → Classifier
"""
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


# Circadian modulation curves (population averages)
# slope_per_hour: change in relative band power per hour awake
# Based on fractal EEG paper + chronotype meta-analysis
CIRCADIAN_CURVES = {
    "theta": {
        "slope_per_hour": 0.02,     # +2% per hour awake (sleep pressure)
        "peak_hour": 23,            # Peaks late night
    },
    "alpha": {
        "slope_per_hour": -0.015,   # Decreases with wakefulness
        "peak_hour": 8,             # Peaks shortly after wake
    },
    "beta": {
        "slope_per_hour": -0.005,   # Slight decrease, relatively stable
        "peak_hour": 10,
    },
    "delta": {
        "slope_per_hour": 0.01,     # Increases with sleep pressure
        "peak_hour": 0,
    },
}

# Chronotype offset in hours from intermediate
CHRONOTYPE_OFFSETS = {
    "morning": -1.5,
    "intermediate": 0.0,
    "evening": 1.5,
}


class CircadianNormalizer:
    """Correct EEG features for time-of-day and chronotype.

    Usage:
        normalizer = CircadianNormalizer()
        normalizer.set_wake_time(7.0)  # 7:00 AM
        corrected = normalizer.correct_features(features, datetime.now())
    """

    def __init__(self):
        self._wake_time_hour: float = 7.0  # default 7 AM
        self._chronotype: str = "intermediate"
        self._chronotype_offset: float = 0.0

    def set_wake_time(self, wake_hour: float, user_id: str = "default"):
        """Set today's wake time for hours-awake calculation.

        Args:
            wake_hour: Wake time as decimal hour (7.5 = 7:30 AM).
            user_id: User identifier (currently single-user).
        """
        self._wake_time_hour = float(np.clip(wake_hour, 0, 23.99))

    def estimate_chronotype(self, sleep_records: List[Dict]) -> Dict:
        """Estimate chronotype from sleep/wake history.

        Args:
            sleep_records: List of dicts with 'wake_time_hours' and
                'sleep_time_hours' (decimal hours, e.g., 23.5 = 11:30 PM).
                Ideally 7-14 days of data from Apple Health.

        Returns:
            Dict with chronotype, mid_sleep_point, and offset_hours.
        """
        if not sleep_records or len(sleep_records) < 3:
            self._chronotype = "intermediate"
            self._chronotype_offset = 0.0
            return {
                "chronotype": "intermediate",
                "mid_sleep_point": 3.0,
                "offset_hours": 0.0,
                "confidence": "low",
                "n_records": len(sleep_records) if sleep_records else 0,
            }

        recent = sleep_records[-14:]  # Use last 2 weeks
        wake_times = [r.get("wake_time_hours", 7.0) for r in recent]
        sleep_times = [r.get("sleep_time_hours", 23.0) for r in recent]

        avg_wake = float(np.mean(wake_times))
        avg_sleep = float(np.mean(sleep_times))

        # Mid-sleep point (handle midnight wrap)
        if avg_sleep > avg_wake:
            # Sleep time is PM, wake is AM next day
            mid_sleep = (avg_sleep + avg_wake + 24) / 2 % 24
        else:
            mid_sleep = (avg_sleep + avg_wake) / 2

        # Classify chronotype based on mid-sleep point
        if mid_sleep < 2.5:
            self._chronotype = "morning"
        elif mid_sleep > 4.5:
            self._chronotype = "evening"
        else:
            self._chronotype = "intermediate"

        self._chronotype_offset = CHRONOTYPE_OFFSETS[self._chronotype]
        self._wake_time_hour = avg_wake

        confidence = "high" if len(recent) >= 7 else "medium"

        return {
            "chronotype": self._chronotype,
            "mid_sleep_point": round(mid_sleep, 2),
            "offset_hours": self._chronotype_offset,
            "avg_wake_time": round(avg_wake, 2),
            "avg_sleep_time": round(avg_sleep, 2),
            "confidence": confidence,
            "n_records": len(recent),
        }

    def correct_features(
        self,
        features: Dict,
        session_time: Optional[datetime] = None,
    ) -> Dict:
        """Apply circadian correction to EEG band power features.

        Subtracts the expected circadian drift based on hours since wake
        and chronotype offset.

        Args:
            features: Dict with band power keys (theta, alpha, beta, delta).
            session_time: When the session is happening. Defaults to now.

        Returns:
            Copy of features with circadian drift removed and metadata added.
        """
        if session_time is None:
            session_time = datetime.now()

        hours_awake = self._estimate_hours_awake(session_time)
        corrected = dict(features)

        for band, curve in CIRCADIAN_CURVES.items():
            band_key = f"{band}_power" if f"{band}_power" in features else band
            if band_key in corrected and isinstance(corrected[band_key], (int, float)):
                expected_drift = curve["slope_per_hour"] * hours_awake
                corrected[band_key] = corrected[band_key] - expected_drift

        corrected["circadian_correction_applied"] = True
        corrected["hours_awake_estimate"] = round(hours_awake, 2)
        corrected["chronotype"] = self._chronotype
        corrected["session_hour"] = session_time.hour + session_time.minute / 60.0

        return corrected

    def get_expected_drift(self, hours_awake: float) -> Dict[str, float]:
        """Get expected circadian drift per band for given hours awake.

        Useful for debugging/visualization.

        Args:
            hours_awake: Hours since wake time.

        Returns:
            Dict mapping band names to expected drift values.
        """
        return {
            band: round(curve["slope_per_hour"] * hours_awake, 4)
            for band, curve in CIRCADIAN_CURVES.items()
        }

    def _estimate_hours_awake(self, session_time: datetime) -> float:
        """Estimate hours since wake time."""
        session_hour = session_time.hour + session_time.minute / 60.0
        adjusted_wake = self._wake_time_hour + self._chronotype_offset
        hours_awake = session_hour - adjusted_wake

        # Handle midnight wrap (e.g., wake at 7, session at 1 AM next day)
        if hours_awake < 0:
            hours_awake += 24

        return float(np.clip(hours_awake, 0, 24))
