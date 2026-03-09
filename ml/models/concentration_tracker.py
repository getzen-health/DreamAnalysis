"""Sustained Concentration / Attention Tracker from EEG signals.

Tracks concentration over time with vigilance decrement modeling, lapse
detection, and optimal break-time recommendations.

Unlike the simpler AttentionClassifier (instantaneous 4-state classification),
this tracker models the *temporal dynamics* of sustained attention:
  - Vigilance decrement: attention naturally declines over prolonged periods
    (Warm et al., 2008; Mackworth, 1948 clock test)
  - Attention lapses: sudden drops in concentration (>30% within a window)
  - Break recommendations: Pomodoro-based (25 min) or lapse-triggered

Concentration levels (0-100 score mapped to 5 levels):
  unfocused  — score < 20: mind-wandering, default-mode network dominant
  low        — score 20-39: partially engaged, frequent distractibility
  moderate   — score 40-59: working attention, some effort required
  high       — score 60-79: sustained focus, good task engagement
  deep       — score >= 80: deep concentration, flow-adjacent

Primary EEG markers:
  - Beta/theta ratio (Clarke et al., 2001): higher = better attention
  - Alpha suppression (Klimesch, 2012): lower alpha = more engaged
  - Low-beta engagement (12-20 Hz): working memory and attentional control

Reference: Warm et al. (2008), Mackworth (1948), Clarke et al. (2001),
           Klimesch (2012), Parasuraman (1979)
"""

import numpy as np
from typing import Dict, List, Optional

from processing.eeg_processor import (
    extract_band_powers,
    preprocess,
)

CONCENTRATION_LEVELS = ["unfocused", "low", "moderate", "high", "deep"]

# Pomodoro default break interval (minutes)
_POMODORO_MINUTES = 25.0

# Lapse detection: drop threshold (fraction of recent average)
_LAPSE_DROP_FRACTION = 0.30

# Number of recent assessments to use for lapse baseline
_LAPSE_WINDOW = 5

# Max vigilance decrement time constant (minutes). At this many minutes,
# decrement reaches ~63% of max. Based on Warm et al. (2008) showing
# significant performance decline after 20-30 minutes of sustained vigilance.
_VIGILANCE_TAU_MINUTES = 45.0


class ConcentrationTracker:
    """Tracks sustained concentration over time with vigilance decrement modeling.

    Args:
        max_history: Maximum number of assessment records to keep.
    """

    def __init__(self, max_history: int = 500):
        self._max_history = max_history

        # Baseline (resting-state) band powers
        self._baseline_beta: Optional[float] = None
        self._baseline_theta: Optional[float] = None
        self._baseline_alpha: Optional[float] = None

        # History of assessment results
        self._history: List[Dict] = []

        # Lapse tracking
        self._lapse_count: int = 0

        # Raw concentration scores for lapse detection (pre-decrement)
        self._raw_scores: List[float] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_baseline(self) -> bool:
        """Whether a resting-state baseline has been recorded."""
        return self._baseline_beta is not None

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def set_baseline(self, eeg: np.ndarray, fs: float = 256.0) -> None:
        """Record resting-state EEG as the concentration baseline.

        Args:
            eeg: 1D single-channel or 2D (n_channels, n_samples) EEG array.
            fs: Sampling rate in Hz.
        """
        signal = self._extract_single_channel(eeg)
        signal = self._safe_preprocess(signal, fs)
        bands = extract_band_powers(signal, fs)

        self._baseline_beta = max(bands.get("beta", 0.15), 1e-10)
        self._baseline_theta = max(bands.get("theta", 0.15), 1e-10)
        self._baseline_alpha = max(bands.get("alpha", 0.20), 1e-10)

    # ------------------------------------------------------------------
    # Core assessment
    # ------------------------------------------------------------------

    def assess(
        self,
        eeg: np.ndarray,
        fs: float = 256.0,
        elapsed_minutes: Optional[float] = None,
    ) -> Dict:
        """Assess current concentration from an EEG epoch.

        Args:
            eeg: 1D or 2D EEG array.
            fs: Sampling rate in Hz.
            elapsed_minutes: Minutes since session/baseline start. Used for
                vigilance decrement modeling. If None, decrement is 0.

        Returns:
            Dict with keys:
                concentration_score (0-100),
                concentration_level (str),
                vigilance_decrement (0.0-1.0),
                lapse_detected (bool),
                time_since_baseline_s (float),
                break_recommendation (str or None)
        """
        signal = self._extract_single_channel(eeg)
        signal = self._safe_preprocess(signal, fs)
        bands = extract_band_powers(signal, fs)

        # --- Raw concentration score (0-1) from EEG features ---
        raw_score = self._compute_raw_score(bands)

        # --- Vigilance decrement (time-on-task penalty) ---
        minutes = max(0.0, elapsed_minutes) if elapsed_minutes is not None else 0.0
        vigilance_decrement = self._compute_vigilance_decrement(minutes)

        # --- Apply decrement to score ---
        adjusted = raw_score * (1.0 - 0.4 * vigilance_decrement)
        concentration_score = float(np.clip(adjusted * 100.0, 0.0, 100.0))

        # --- Lapse detection ---
        self._raw_scores.append(raw_score)
        if len(self._raw_scores) > self._max_history:
            self._raw_scores = self._raw_scores[-self._max_history:]
        lapse_detected = self._detect_lapse(raw_score)

        # --- Concentration level ---
        concentration_level = self._score_to_level(concentration_score)

        # --- Break recommendation ---
        break_rec = self._get_break_recommendation(minutes)

        # --- Elapsed seconds ---
        time_since_baseline_s = minutes * 60.0

        # --- Store history ---
        record = {
            "concentration_score": round(concentration_score, 1),
            "concentration_level": concentration_level,
            "vigilance_decrement": round(vigilance_decrement, 4),
            "lapse_detected": lapse_detected,
            "time_since_baseline_s": round(time_since_baseline_s, 2),
            "break_recommendation": break_rec,
        }
        self._history.append(record)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return record

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_optimal_break_time(self) -> Dict:
        """Recommend optimal break interval based on session data.

        Returns:
            Dict with 'recommended_break_after_minutes' and 'reason'.
        """
        base_minutes = _POMODORO_MINUTES

        # Shorten if lapses have occurred
        if self._lapse_count >= 3:
            recommended = max(10.0, base_minutes - self._lapse_count * 3.0)
            reason = f"Shortened from {base_minutes:.0f} min due to {self._lapse_count} attention lapses"
        elif self._lapse_count >= 1:
            recommended = base_minutes - self._lapse_count * 2.0
            reason = f"Slightly shortened due to {self._lapse_count} attention lapse(s)"
        else:
            recommended = base_minutes
            reason = "Pomodoro default (no lapses detected)"

        return {
            "recommended_break_after_minutes": round(recommended, 1),
            "reason": reason,
        }

    def get_session_stats(self) -> Dict:
        """Return summary statistics for the current session.

        Returns:
            Dict with n_assessments, lapse_count, mean/min/max concentration,
            and level_distribution (%).
        """
        n = len(self._history)
        if n == 0:
            return {
                "n_assessments": 0,
                "lapse_count": self._lapse_count,
                "mean_concentration": 0.0,
                "min_concentration": 0.0,
                "max_concentration": 0.0,
                "level_distribution": {lv: 0.0 for lv in CONCENTRATION_LEVELS},
            }

        scores = [h["concentration_score"] for h in self._history]
        levels = [h["concentration_level"] for h in self._history]
        level_counts = {lv: 0 for lv in CONCENTRATION_LEVELS}
        for lv in levels:
            level_counts[lv] = level_counts.get(lv, 0) + 1
        level_pct = {lv: round(100.0 * cnt / n, 1) for lv, cnt in level_counts.items()}

        return {
            "n_assessments": n,
            "lapse_count": self._lapse_count,
            "mean_concentration": round(float(np.mean(scores)), 1),
            "min_concentration": round(float(np.min(scores)), 1),
            "max_concentration": round(float(np.max(scores)), 1),
            "level_distribution": level_pct,
        }

    def get_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """Return assessment history, optionally limited to last N records."""
        if last_n is not None:
            return list(self._history[-last_n:])
        return list(self._history)

    def reset(self) -> None:
        """Clear all state: history, baseline, lapse counter."""
        self._baseline_beta = None
        self._baseline_theta = None
        self._baseline_alpha = None
        self._history = []
        self._lapse_count = 0
        self._raw_scores = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_single_channel(eeg: np.ndarray) -> np.ndarray:
        """Extract a single channel for analysis.

        For multichannel (Muse 2): uses AF7 (ch1, left frontal) -- the
        primary channel for attention-related beta/theta analysis.
        """
        if eeg.ndim == 2:
            if eeg.shape[0] >= 2:
                return eeg[1].copy()  # AF7
            return eeg[0].copy()
        return eeg.copy()

    @staticmethod
    def _safe_preprocess(signal: np.ndarray, fs: float) -> np.ndarray:
        """Preprocess with NaN handling."""
        # Replace NaN with zero to prevent filter failures
        if np.any(np.isnan(signal)):
            signal = np.nan_to_num(signal, nan=0.0)
        return preprocess(signal, fs)

    def _compute_raw_score(self, bands: Dict[str, float]) -> float:
        """Compute raw concentration score (0-1) from band powers.

        Primary: beta/theta ratio (higher = more focused)
        Secondary: alpha suppression (lower alpha = more engaged)
        """
        beta = bands.get("beta", 0.0)
        theta = bands.get("theta", 0.0)
        alpha = bands.get("alpha", 0.0)
        low_beta = bands.get("low_beta", 0.0)

        # --- Component 1: Beta/Theta ratio ---
        # Higher beta relative to theta = sustained attention
        bt_ratio = beta / (theta + 1e-10)
        # tanh mapping: ratio of 1.0 -> ~0.46, ratio of 2.0 -> ~0.76
        bt_score = float(np.tanh(bt_ratio * 0.5))

        # --- Component 2: Alpha suppression ---
        # Lower alpha = more engaged (alpha desynchronization)
        # When alpha is high relative to total, focus is low
        total = alpha + beta + theta + 1e-10
        alpha_fraction = alpha / total
        alpha_suppression = float(np.clip(1.0 - alpha_fraction * 2.5, 0.0, 1.0))

        # --- Component 3: Low-beta engagement (12-20 Hz) ---
        # Working memory and sustained attention marker
        lb_score = float(np.clip(np.tanh(low_beta * 5.0), 0.0, 1.0))

        # --- Baseline adjustment ---
        baseline_bonus = 0.0
        if self._baseline_beta is not None:
            beta_change = beta / (self._baseline_beta + 1e-10) - 1.0
            theta_change = theta / (self._baseline_theta + 1e-10) - 1.0
            # Positive beta change AND negative theta change = more focused
            baseline_bonus = float(np.clip(
                0.15 * np.tanh(beta_change) - 0.15 * np.tanh(theta_change),
                -0.2, 0.2
            ))

        raw = (
            0.45 * bt_score
            + 0.30 * alpha_suppression
            + 0.25 * lb_score
            + baseline_bonus
        )
        return float(np.clip(raw, 0.0, 1.0))

    @staticmethod
    def _compute_vigilance_decrement(elapsed_minutes: float) -> float:
        """Model vigilance decrement as exponential saturation.

        Based on Warm et al. (2008): performance on vigilance tasks
        declines exponentially, with most decline in first 30-45 minutes.

        Returns value in [0, 1]: 0 = no decrement, 1 = maximum decrement.
        """
        if elapsed_minutes <= 0:
            return 0.0
        # Exponential approach to 1.0
        return float(1.0 - np.exp(-elapsed_minutes / _VIGILANCE_TAU_MINUTES))

    def _detect_lapse(self, current_raw_score: float) -> bool:
        """Detect attention lapse: >30% drop from recent average.

        A lapse is defined as a sudden, large drop in concentration
        relative to the recent moving average. This distinguishes
        true lapses from gradual fatigue-related decline.
        """
        if len(self._raw_scores) < _LAPSE_WINDOW + 1:
            return False

        # Compare current to average of previous window
        recent = self._raw_scores[-(_LAPSE_WINDOW + 1):-1]
        recent_avg = float(np.mean(recent))

        if recent_avg <= 0.05:
            # Already very low -- can't drop further
            return False

        drop_fraction = (recent_avg - current_raw_score) / recent_avg
        if drop_fraction > _LAPSE_DROP_FRACTION:
            self._lapse_count += 1
            return True

        return False

    def _get_break_recommendation(self, elapsed_minutes: float) -> Optional[str]:
        """Generate break recommendation if conditions are met."""
        if elapsed_minutes >= _POMODORO_MINUTES:
            return (
                f"Consider a 5-minute break. You have been concentrating for "
                f"{elapsed_minutes:.0f} minutes (Pomodoro: {_POMODORO_MINUTES:.0f} min)."
            )
        if self._lapse_count >= 3:
            return (
                f"Consider a short break. {self._lapse_count} attention lapses "
                f"detected -- your sustained attention may be fatigued."
            )
        return None

    @staticmethod
    def _score_to_level(score: float) -> str:
        """Map 0-100 concentration score to a named level."""
        if score >= 80:
            return "deep"
        elif score >= 60:
            return "high"
        elif score >= 40:
            return "moderate"
        elif score >= 20:
            return "low"
        else:
            return "unfocused"
