"""Cognitive Flexibility Detector via Aperiodic Exponent + FMT.

Detects the brain's ability to shift between cognitive strategies
(persistence vs. flexibility) using frontal EEG features.

Scientific basis:
- Scientific Reports (2024): Aperiodic exponent decreases during task switching
  vs. repetition — reflects metacontrol balance (persistence vs. flexibility)
- ScienceDirect (2025): FMT power increases with task-switching difficulty;
  no modulation indicates reduced flexibility (aging/rigidity marker)
- 88% accuracy for 3-level cognitive state classification (systematic review)
- FMT computed by existing `compute_frontal_midline_theta()` in eeg_processor.py
- Aperiodic exponent shared with brain age + ADHD screening (same infrastructure)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.signal import welch


def _compute_aperiodic_exponent(signal: np.ndarray, fs: float = 256.0) -> float:
    """Estimate aperiodic 1/f exponent via log-log PSD regression."""
    nperseg = min(len(signal), int(fs * 4))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    mask = (freqs >= 2) & (freqs <= 40)
    if mask.sum() < 5:
        return 2.0

    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask] + 1e-30)
    coeffs = np.polyfit(log_f, log_p, 1)
    return float(-coeffs[0])  # positive exponent


class CognitiveFlexibilityDetector:
    """Detect cognitive flexibility from frontal EEG aperiodic features + FMT.

    Output: flexibility_index (0-1) where:
    - 0.0 - 0.33: rigid (low flexibility, perseverative thinking)
    - 0.33 - 0.67: moderate (typical adaptive thinking)
    - 0.67 - 1.0: flexible (high task-switching ability)

    For dynamic flexibility: compare aperiodic exponent between baseline and task.
    Large decrease during task = high flexibility.
    """

    # Population norms (approximate):
    # Typical frontal aperiodic exponent: 1.5-2.5
    # High flexibility: exponent decreases 0.2-0.5 during task vs rest
    NORM_EXPONENT_MEAN = 2.0
    NORM_EXPONENT_STD = 0.4

    def __init__(self):
        self._rest_exponent: Optional[float] = None
        self._rest_fmt_power: Optional[float] = None
        self._exponent_history: List[float] = []
        self._fmt_history: List[float] = []

    def predict(self, signals: np.ndarray, fs: float = 256.0) -> Dict:
        """Predict cognitive flexibility from a single EEG epoch.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) array
            fs: sampling rate

        Returns:
            dict with flexibility_index (0-1), level, fmt_power, aperiodic_exponent,
            metacontrol_bias, and component scores.
        """
        from processing.eeg_processor import (
            compute_frontal_midline_theta,
            extract_band_powers,
            preprocess,
        )

        # Use mean of AF7+AF8 for frontal signal if multichannel
        if signals.ndim == 2 and signals.shape[0] >= 3:
            frontal = (signals[1] + signals[2]) / 2.0  # AF7 + AF8
        elif signals.ndim == 2:
            frontal = signals[0]
        else:
            frontal = signals

        processed = preprocess(frontal, fs)
        bands = extract_band_powers(processed, fs)

        # Aperiodic exponent
        exponent = _compute_aperiodic_exponent(processed, fs)

        # FMT (frontal midline theta) — existing function
        try:
            fmt_result = compute_frontal_midline_theta(processed, fs)
            fmt_power = float(fmt_result.get("fmt_power", 0.0))
        except Exception:
            fmt_power = 0.0

        alpha = max(bands.get("alpha", 0.2), 1e-10)
        theta = max(bands.get("theta", 0.15), 1e-10)

        eps = 1e-10

        # 1. Aperiodic exponent score
        # Flexible brains: moderate exponent (not too high = rigid, not too low = chaotic)
        # Optimal range: 1.5-2.0 for active flexible cognition
        exp_norm = (exponent - 1.5) / 1.0  # normalize 1.5-2.5 to 0-1
        # Inverted-U: flexibility peaks at moderate exponent
        exp_score = float(max(0.0, 1.0 - (exp_norm - 0.5) ** 2 * 4.0))

        # 2. FMT power score
        # FMT increases with task-switching difficulty in flexible brains
        # Normalize relative to typical FMT power (~0.001-0.01)
        fmt_norm = float(
            np.clip(
                fmt_power / (self._rest_fmt_power + eps)
                if self._rest_fmt_power
                else fmt_power * 100,
                0,
                2,
            )
        )
        fmt_score = float(np.clip(fmt_norm / 1.5, 0, 1))  # peak at 1.5x baseline

        # 3. Alpha/theta ratio — moderate alpha with moderate theta = flexible state
        alpha_theta = alpha / (theta + eps)
        # Flexibility at moderate ratio (1.0-2.0): not too theta-dominant (rigid/drowsy)
        # nor too alpha-dominant (mind-wandering)
        at_score = float(max(0.0, 1.0 - abs(alpha_theta - 1.5) / 1.5))

        # 4. Dynamic response (if rest baseline available)
        dynamic_score = 0.5  # neutral when no baseline
        metacontrol_bias = "unknown"
        if self._rest_exponent is not None:
            exp_change = exponent - self._rest_exponent
            # Large decrease during task = high flexibility
            if exp_change < -0.3:
                dynamic_score = 0.85
                metacontrol_bias = "flexible"
            elif exp_change < -0.1:
                dynamic_score = 0.65
                metacontrol_bias = "moderately_flexible"
            elif exp_change < 0.1:
                dynamic_score = 0.45
                metacontrol_bias = "balanced"
            else:
                dynamic_score = 0.25
                metacontrol_bias = "persistent"  # exponent unchanged or increased
        else:
            # Estimate from static features
            if exponent < 1.8:
                metacontrol_bias = "flexible"
            elif exponent < 2.2:
                metacontrol_bias = "balanced"
            else:
                metacontrol_bias = "persistent"

        # Weighted flexibility index
        if self._rest_exponent is not None:
            flex = (
                0.40 * dynamic_score
                + 0.30 * fmt_score
                + 0.20 * exp_score
                + 0.10 * at_score
            )
        else:
            flex = 0.35 * exp_score + 0.35 * fmt_score + 0.30 * at_score

        flexibility_index = float(np.clip(flex, 0.0, 1.0))

        if flexibility_index >= 0.67:
            level = "flexible"
        elif flexibility_index >= 0.33:
            level = "moderate"
        else:
            level = "rigid"

        # Track history
        self._exponent_history.append(exponent)
        self._fmt_history.append(fmt_power)

        return {
            "flexibility_index": round(flexibility_index, 4),
            "level": level,
            "metacontrol_bias": metacontrol_bias,
            "fmt_power": round(fmt_power, 6),
            "aperiodic_exponent": round(exponent, 3),
            "exp_score": round(exp_score, 3),
            "fmt_score": round(fmt_score, 3),
            "alpha_theta_score": round(at_score, 3),
            "dynamic_score": round(dynamic_score, 3) if self._rest_exponent is not None else None,
            "model_type": "aperiodic_fmt_heuristic",
        }

    def record_baseline(self, signals: np.ndarray, fs: float = 256.0) -> Dict:
        """Record resting-state baseline for dynamic flexibility measurement.

        Call during a passive rest period (eyes open or closed, no task).
        The subsequent calls to predict() will compare against this baseline.
        """
        from processing.eeg_processor import compute_frontal_midline_theta, preprocess

        if signals.ndim == 2 and signals.shape[0] >= 3:
            frontal = (signals[1] + signals[2]) / 2.0
        elif signals.ndim == 2:
            frontal = signals[0]
        else:
            frontal = signals

        processed = preprocess(frontal, fs)
        self._rest_exponent = _compute_aperiodic_exponent(processed, fs)

        try:
            fmt_result = compute_frontal_midline_theta(processed, fs)
            self._rest_fmt_power = float(fmt_result.get("fmt_power", 0.0))
        except Exception:
            self._rest_fmt_power = None

        return {
            "status": "recorded",
            "rest_exponent": round(self._rest_exponent, 3),
            "rest_fmt_power": round(self._rest_fmt_power, 6) if self._rest_fmt_power else None,
        }

    def measure_dynamic_flexibility(
        self,
        rest_signals: np.ndarray,
        task_signals: np.ndarray,
        fs: float = 256.0,
    ) -> Dict:
        """Measure flexibility by comparing aperiodic exponent between rest and task.

        Args:
            rest_signals: EEG during passive rest
            task_signals: EEG during cognitive task (e.g., task-switching)
            fs: sampling rate

        Returns:
            dict with exponent_change, flexibility_response, and interpretation
        """
        self.record_baseline(rest_signals, fs)
        task_result = self.predict(task_signals, fs)

        rest_exp = self._rest_exponent or 2.0
        task_exp = task_result["aperiodic_exponent"]
        exp_change = task_exp - rest_exp

        if exp_change < -0.3:
            response = "high_flexibility"
            interpretation = "Large exponent decrease during task — hallmark of cognitive flexibility."
        elif exp_change < -0.1:
            response = "moderate_flexibility"
            interpretation = "Moderate exponent decrease — typical adaptive flexibility."
        elif exp_change < 0.1:
            response = "balanced"
            interpretation = "No significant exponent change — balanced persistence-flexibility."
        else:
            response = "low_flexibility"
            interpretation = "Exponent unchanged or increased — perseverative thinking pattern."

        return {
            "rest_exponent": round(float(rest_exp), 3),
            "task_exponent": round(float(task_exp), 3),
            "exponent_change": round(float(exp_change), 3),
            "flexibility_response": response,
            "interpretation": interpretation,
            "task_flexibility_index": task_result["flexibility_index"],
        }


_flex_instances: dict = {}


def get_flexibility_detector(user_id: str = "default") -> CognitiveFlexibilityDetector:
    if user_id not in _flex_instances:
        _flex_instances[user_id] = CognitiveFlexibilityDetector()
    return _flex_instances[user_id]
