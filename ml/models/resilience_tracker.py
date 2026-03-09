"""EEG entropy modulation as a resilience biomarker.

Based on: IBRO Neuroreports (2025) -- entropy modulation during
emotional challenge predicts psychological resilience.

Core idea: a resilient brain shows *greater* entropy change when
transitioning from resting baseline to an emotional/cognitive challenge.
Higher modulation = greater neural flexibility = higher resilience.

Usage:
    1. Record 2-min resting baseline via BaselineCalibrator or
       ResilienceTracker.set_baseline().
    2. During task/emotional challenge, call compute_modulation()
       with live EEG epochs.
    3. Call get_trend() for session-over-session resilience trajectory.
"""

import numpy as np
from typing import Dict, List, Optional

from processing.eeg_processor import spectral_entropy, preprocess


# Typical entropy modulation range from literature.
# Modulation of 0.3 (30% change from baseline) maps to resilience_score = 1.0.
_MODULATION_CEILING = 0.3


class ResilienceTracker:
    """Track entropy modulation as a resilience metric.

    Stateful: maintains baseline entropy and session history.
    Each user should get their own instance (keyed by user_id in routes).
    """

    def __init__(self) -> None:
        self._baseline_entropy: Optional[float] = None
        self._session_scores: List[float] = []

    @property
    def has_baseline(self) -> bool:
        """True when a resting baseline has been established."""
        return self._baseline_entropy is not None

    def set_baseline(self, signals: np.ndarray, fs: float = 256.0) -> None:
        """Set resting baseline entropy from multi-channel EEG.

        Call once during the resting calibration phase (eyes-closed,
        2-min minimum recommended). Computes average spectral entropy
        across all channels.

        Args:
            signals: (n_channels, n_samples) EEG array.
            fs: Sampling rate in Hz.
        """
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        entropies = []
        for ch in range(signals.shape[0]):
            processed = preprocess(signals[ch], fs)
            entropies.append(spectral_entropy(processed, fs))
        self._baseline_entropy = float(np.mean(entropies))

    def compute_modulation(self, task_signals: np.ndarray, fs: float = 256.0) -> Dict:
        """Measure entropy modulation from baseline.

        Higher modulation = brain is more responsive to stimuli = higher
        resilience. Returns has_baseline=False if no baseline is set.

        Args:
            task_signals: (n_channels, n_samples) EEG during task/challenge.
            fs: Sampling rate in Hz.

        Returns:
            Dict with resilience_score (0-1), entropy_modulation (ratio),
            baseline/task entropy values, direction, and has_baseline flag.
        """
        if task_signals.ndim == 1:
            task_signals = task_signals.reshape(1, -1)

        # Compute task-epoch spectral entropy
        task_entropies = []
        for ch in range(task_signals.shape[0]):
            processed = preprocess(task_signals[ch], fs)
            task_entropies.append(spectral_entropy(processed, fs))
        task_entropy = float(np.mean(task_entropies))

        if self._baseline_entropy is None or self._baseline_entropy < 1e-6:
            return {
                "resilience_score": 0.0,
                "entropy_modulation": 0.0,
                "baseline_entropy": 0.0,
                "task_entropy": round(task_entropy, 3),
                "direction": "unknown",
                "has_baseline": False,
            }

        # Fractional change from baseline
        modulation = abs(task_entropy - self._baseline_entropy) / (
            self._baseline_entropy + 1e-10
        )

        # Normalize to 0-1 (0.3 = 30% change maps to score 1.0)
        resilience_score = float(np.clip(modulation / _MODULATION_CEILING, 0.0, 1.0))

        direction = "increase" if task_entropy > self._baseline_entropy else "decrease"

        result = {
            "resilience_score": round(resilience_score, 3),
            "entropy_modulation": round(modulation, 4),
            "baseline_entropy": round(self._baseline_entropy, 3),
            "task_entropy": round(task_entropy, 3),
            "direction": direction,
            "has_baseline": True,
        }

        self._session_scores.append(resilience_score)
        return result

    def get_trend(self) -> Dict:
        """Get resilience trend across measurements within this session.

        Returns:
            Dict with mean_score, latest_score, measurement count, and
            directional trend (improving / declining / stable /
            insufficient_data).
        """
        if len(self._session_scores) < 1:
            return {"trend": "insufficient_data", "sessions": 0}

        scores = self._session_scores
        mean_score = float(np.mean(scores))
        latest_score = scores[-1]

        if len(scores) < 3:
            trend = "insufficient_data"
        elif latest_score > mean_score + 0.05:
            trend = "improving"
        elif latest_score < mean_score - 0.05:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "mean_score": round(mean_score, 3),
            "latest_score": round(latest_score, 3),
            "sessions": len(scores),
            "trend": trend,
        }

    def reset(self) -> None:
        """Clear baseline and session history."""
        self._baseline_entropy = None
        self._session_scores.clear()
