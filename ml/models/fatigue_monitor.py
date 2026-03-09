"""Continuous mental fatigue tracking via theta/beta ratio trend.

Based on: EEG theta/beta ratio (TBR) increases with mental fatigue.
Theta power rises (drowsiness, reduced vigilance) while beta power
drops (disengagement from active processing). The ratio is the single
most validated EEG marker for time-on-task fatigue.

Key references:
  - Wascher et al. (2014): theta increase + alpha decrease with fatigue
  - Lal & Craig (2002): theta/beta ratio predicts driving drowsiness
  - Borghini et al. (2014): review of EEG-based mental fatigue markers

Usage:
    monitor = FatigueMonitor()
    monitor.set_baseline(1.2, user_id="user1")   # session-start TBR
    result = monitor.assess(0.5, 0.3, session_minutes=15.0, user_id="user1")
    # result["fatigue_index"] -> 0.0-1.0
    # result["recommendation"] -> "continue" | "short_break_soon" | ...
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _fatigue_stage(fatigue_index: float) -> str:
    """Map fatigue index to human-readable stage label."""
    if fatigue_index < 0.3:
        return "fresh"
    elif fatigue_index < 0.5:
        return "mild"
    elif fatigue_index < 0.7:
        return "moderate"
    elif fatigue_index < 0.85:
        return "high"
    else:
        return "exhausted"


def _recommendation(fatigue_index: float) -> str:
    """Map fatigue index to actionable recommendation."""
    if fatigue_index < 0.3:
        return "continue"
    elif fatigue_index < 0.5:
        return "short_break_soon"
    elif fatigue_index < 0.7:
        return "take_break_now"
    else:
        return "end_session"


def _linear_slope(values: List[float], max_points: int = 150) -> float:
    """Compute slope of a value series via least-squares linear regression.

    Uses numpy polyfit on the most recent `max_points` values.
    Returns 0.0 if fewer than 2 data points.
    """
    if len(values) < 2:
        return 0.0
    recent = values[-max_points:]
    x = np.arange(len(recent), dtype=np.float64)
    y = np.array(recent, dtype=np.float64)
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])


class _UserState:
    """Per-user fatigue tracking state."""

    __slots__ = ("baseline_tbr", "tbr_history", "fatigue_history", "stage_history")

    def __init__(self) -> None:
        self.baseline_tbr: Optional[float] = None
        self.tbr_history: List[float] = []
        self.fatigue_history: List[float] = []
        self.stage_history: List[str] = []


class FatigueMonitor:
    """Continuous mental fatigue tracking via theta/beta ratio trend.

    Maintains a rolling TBR history per user. Computes fatigue index
    relative to a session-start baseline (if set) or using population-
    average absolute thresholds.

    Thread safety: NOT thread-safe. External synchronization required
    if multiple threads call assess() concurrently for the same user.
    """

    def __init__(self, max_history: int = 500) -> None:
        self._max_history = max_history
        self._users: Dict[str, _UserState] = defaultdict(_UserState)

    def set_baseline(
        self, theta_beta_ratio: float, user_id: str = "default"
    ) -> None:
        """Record session-start theta/beta ratio as baseline.

        Call once at the beginning of a session (ideally during the
        2-min resting calibration phase). All subsequent assess() calls
        for this user compute fatigue relative to this baseline.

        Args:
            theta_beta_ratio: Resting-state theta/beta ratio.
            user_id: User identifier for multi-user support.
        """
        self._users[user_id].baseline_tbr = theta_beta_ratio

    def assess(
        self,
        theta_power: float,
        beta_power: float,
        session_minutes: float = 0.0,
        user_id: str = "default",
    ) -> Dict:
        """Assess current mental fatigue from theta/beta ratio.

        Args:
            theta_power: Theta band power (4-8 Hz), 0-1 normalized.
            beta_power: Beta band power (12-30 Hz), 0-1 normalized.
            session_minutes: Minutes elapsed since session start.
            user_id: User identifier.

        Returns:
            Dict with fatigue_index, fatigue_stage, theta_beta_ratio,
            trend_slope, time_to_break_min, recommendation, n_samples.
        """
        state = self._users[user_id]

        # -- Compute theta/beta ratio --
        tbr = theta_power / max(beta_power, 1e-10)

        # -- Compute raw fatigue index --
        if state.baseline_tbr is not None:
            # Baseline mode: fatigue = sigmoid of normalized TBR increase
            baseline = max(state.baseline_tbr, 1e-10)
            fatigue_raw = (tbr - baseline) / baseline
            fatigue_index = _sigmoid(fatigue_raw * 3.0)
            # Shift so that at-baseline maps to ~0.5 -> rescale to 0-1
            # sigmoid(0) = 0.5, so (sigmoid - 0.5) * 2 maps baseline to 0
            fatigue_index = max(0.0, (fatigue_index - 0.5) * 2.0)
        else:
            # Absolute mode: population-average thresholds
            # TBR < 1.5 -> fresh, TBR > 3.5 -> exhausted
            fatigue_index = max(0.0, min(1.0, (tbr - 1.5) / 2.0))

        # -- Time-on-task boost --
        effective_minutes = max(session_minutes, 0.0)
        time_boost = min(effective_minutes / 30.0 * 0.05, 0.15)
        fatigue_index = min(1.0, fatigue_index + time_boost)

        # -- Store history --
        state.tbr_history.append(tbr)
        state.fatigue_history.append(fatigue_index)

        stage = _fatigue_stage(fatigue_index)
        state.stage_history.append(stage)

        # Enforce max_history cap
        if len(state.tbr_history) > self._max_history:
            excess = len(state.tbr_history) - self._max_history
            state.tbr_history = state.tbr_history[excess:]
            state.fatigue_history = state.fatigue_history[excess:]
            state.stage_history = state.stage_history[excess:]

        # -- Trend slope --
        slope = _linear_slope(state.tbr_history)

        # -- Time to break estimate --
        time_to_break: Optional[float] = None
        if fatigue_index < 0.7 and slope > 0.001 and len(state.tbr_history) >= 2:
            # Estimate how many samples until fatigue reaches 0.7
            # Simple linear extrapolation from current fatigue + slope
            fatigue_slope = _linear_slope(state.fatigue_history)
            if fatigue_slope > 0.001:
                samples_to_threshold = (0.7 - fatigue_index) / fatigue_slope
                # Assume ~1 sample per second (typical EEG analysis rate)
                time_to_break = max(0.1, samples_to_threshold / 60.0)

        rec = _recommendation(fatigue_index)

        return {
            "fatigue_index": fatigue_index,
            "fatigue_stage": stage,
            "theta_beta_ratio": tbr,
            "trend_slope": slope,
            "time_to_break_min": time_to_break,
            "recommendation": rec,
            "n_samples": len(state.tbr_history),
        }

    def get_fatigue_curve(self, user_id: str = "default") -> List[Dict]:
        """Return the fatigue index history for plotting.

        Args:
            user_id: User identifier.

        Returns:
            List of dicts with fatigue_index and theta_beta_ratio,
            in chronological order.
        """
        state = self._users[user_id]
        return [
            {
                "fatigue_index": state.fatigue_history[i],
                "theta_beta_ratio": state.tbr_history[i],
            }
            for i in range(len(state.fatigue_history))
        ]

    def get_session_summary(self, user_id: str = "default") -> Dict:
        """Summary statistics for the session.

        Args:
            user_id: User identifier.

        Returns:
            Dict with n_samples, mean_fatigue, max_fatigue, and count
            of samples in each fatigue stage.
        """
        state = self._users[user_id]
        n = len(state.fatigue_history)

        if n == 0:
            return {
                "n_samples": 0,
                "mean_fatigue": 0.0,
                "max_fatigue": 0.0,
                "time_in_fresh": 0,
                "time_in_mild": 0,
                "time_in_moderate": 0,
                "time_in_high": 0,
                "time_in_exhausted": 0,
            }

        stage_counts = {
            "fresh": 0,
            "mild": 0,
            "moderate": 0,
            "high": 0,
            "exhausted": 0,
        }
        for s in state.stage_history:
            stage_counts[s] += 1

        return {
            "n_samples": n,
            "mean_fatigue": float(np.mean(state.fatigue_history)),
            "max_fatigue": float(np.max(state.fatigue_history)),
            "time_in_fresh": stage_counts["fresh"],
            "time_in_mild": stage_counts["mild"],
            "time_in_moderate": stage_counts["moderate"],
            "time_in_high": stage_counts["high"],
            "time_in_exhausted": stage_counts["exhausted"],
        }

    def reset(self, user_id: str = "default") -> None:
        """Clear history and baseline for a user.

        Args:
            user_id: User identifier.
        """
        self._users[user_id] = _UserState()
