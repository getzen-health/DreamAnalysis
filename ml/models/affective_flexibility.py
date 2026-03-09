"""Affective flexibility from FAA shift dynamics.

Measures the ability to shift between emotional states adaptively by tracking
frontal alpha asymmetry (FAA) variability over time.

Key biomarkers:
- FAA variability (coefficient of variation): higher = more flexible
- FAA sign-change rate: how often FAA crosses zero (valence flips)
- FAA shift speed: magnitude of epoch-to-epoch FAA changes
- FAA recovery index: how quickly FAA returns toward baseline after deviation
- FAA valence range: total range of FAA values observed

Scientific basis:
- Davidson (1998): FAA as index of approach/withdrawal motivation
- Waugh & Koster (2015): affective flexibility predicts resilience
- Hollenstein (2015): emotional inertia (low flexibility) linked to depression
- Aldao & Nolen-Hoeksema (2012): flexible emotion regulation is adaptive
- Rigid FAA patterns correlate with alexithymia and poor emotional intelligence

Muse 2 channel order (BrainFlow board_id 38):
    ch0 = TP9  (left temporal)
    ch1 = AF7  (left frontal)   -- FAA left channel
    ch2 = AF8  (right frontal)  -- FAA right channel
    ch3 = TP10 (right temporal)
"""

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.signal import welch

# Flexibility level thresholds (0-100 composite score)
_LEVEL_THRESHOLDS = {
    "rigid": (0, 20),
    "low": (20, 40),
    "moderate": (40, 60),
    "high": (60, 80),
    "very_flexible": (80, 100),
}

# Maximum epochs to keep in history per user
_MAX_HISTORY = 500

# Minimum epochs required for meaningful variability computation
_MIN_EPOCHS_FOR_VARIABILITY = 5

# Muse 2 channel indices
_CH_AF7 = 1
_CH_AF8 = 2


def _alpha_power_welch(signal: np.ndarray, fs: float) -> float:
    """Compute alpha band (8-12 Hz) power via Welch's method.

    Args:
        signal: 1D EEG signal array.
        fs: Sampling rate in Hz.

    Returns:
        Alpha band power (float, non-negative).
    """
    n = len(signal)
    if n < 16:
        return 1e-12

    nperseg = min(256, n)
    # Ensure nperseg does not exceed signal length
    if nperseg > n:
        nperseg = n

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    alpha_mask = (freqs >= 8.0) & (freqs <= 12.0)
    if not np.any(alpha_mask):
        return 1e-12

    # Integrate power in alpha band
    alpha_freqs = freqs[alpha_mask]
    alpha_psd = psd[alpha_mask]

    # Use np.trapezoid with np.trapz fallback
    try:
        power = float(np.trapezoid(alpha_psd, alpha_freqs))
    except AttributeError:
        power = float(np.trapz(alpha_psd, alpha_freqs))

    return max(power, 1e-12)


def _compute_faa(signals: np.ndarray, fs: float) -> float:
    """Compute frontal alpha asymmetry from multichannel EEG.

    FAA = log(AF8_alpha) - log(AF7_alpha)

    Positive FAA = approach motivation / positive valence.
    Negative FAA = withdrawal motivation / negative valence.

    Args:
        signals: (n_channels, n_samples) array. Needs at least 3 channels
                 (indices 1=AF7, 2=AF8).
        fs: Sampling rate in Hz.

    Returns:
        FAA value (float). Returns 0.0 if fewer than 3 channels.
    """
    if signals.ndim == 1:
        return 0.0
    if signals.shape[0] < 3:
        return 0.0

    left_alpha = _alpha_power_welch(signals[_CH_AF7], fs)
    right_alpha = _alpha_power_welch(signals[_CH_AF8], fs)

    faa = float(np.log(right_alpha) - np.log(left_alpha))
    return faa


def _score_to_level(score: float) -> str:
    """Map a 0-100 composite score to a flexibility level string."""
    for level, (low, high) in _LEVEL_THRESHOLDS.items():
        if low <= score < high:
            return level
    # score == 100
    return "very_flexible"


class AffectiveFlexibility:
    """Measures affective flexibility from FAA shift dynamics.

    Tracks FAA values over successive epochs to quantify how flexibly
    a person's emotional state shifts. Supports multiple users via
    user_id parameter.

    Usage:
        af = AffectiveFlexibility(fs=256.0)

        # Optional: set resting baseline
        af.set_baseline(resting_eeg, user_id="user1")

        # Feed successive 4-sec EEG epochs
        for epoch in epochs:
            result = af.assess(epoch, user_id="user1")
            print(result["flexibility_score"], result["flexibility_level"])

        # Session summary
        stats = af.get_session_stats(user_id="user1")
    """

    def __init__(self, fs: float = 256.0) -> None:
        self.fs = fs
        # Per-user state: dict of user_id -> user data
        self._users: Dict[str, Dict[str, Any]] = {}

    def _ensure_user(self, user_id: str) -> Dict[str, Any]:
        """Initialize user state if not present."""
        if user_id not in self._users:
            self._users[user_id] = {
                "baseline_faa": None,
                "faa_history": deque(maxlen=_MAX_HISTORY),
                "assess_history": [],
            }
        return self._users[user_id]

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Set resting-state baseline FAA for a user.

        Call with 2-3 minutes of eyes-closed resting EEG. The baseline FAA
        is used as the reference point for the recovery index.

        Args:
            signals: (4, n_samples) multichannel EEG or (n_samples,) single-channel.
            fs: Sampling rate (defaults to self.fs).
            user_id: User identifier for multi-user support.

        Returns:
            Dict with baseline_set (bool) and baseline_faa (float).
        """
        if fs is None:
            fs = self.fs

        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        user = self._ensure_user(user_id)
        faa = _compute_faa(signals, fs)
        user["baseline_faa"] = faa

        return {
            "baseline_set": True,
            "baseline_faa": round(faa, 6),
        }

    def assess(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Assess affective flexibility from a single EEG epoch.

        Feed successive 4-sec epochs for best results. Flexibility metrics
        become meaningful after at least 5 epochs (sign_change_rate,
        faa_variability, etc. return 0.5 defaults before that).

        Args:
            signals: (4, n_samples) multichannel EEG or (n_samples,) single-channel.
            fs: Sampling rate (defaults to self.fs).
            user_id: User identifier for multi-user support.

        Returns:
            Dict with flexibility_score (0-100), flexibility_level, faa_current,
            faa_variability, sign_change_rate, shift_speed, recovery_index,
            valence_range, has_baseline.
        """
        if fs is None:
            fs = self.fs

        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        user = self._ensure_user(user_id)
        faa_current = _compute_faa(signals, fs)

        # Append to FAA history
        user["faa_history"].append(faa_current)
        faa_values = list(user["faa_history"])
        n_epochs = len(faa_values)

        has_baseline = user["baseline_faa"] is not None

        # Compute sub-metrics
        if n_epochs >= _MIN_EPOCHS_FOR_VARIABILITY:
            faa_variability = self._compute_variability(faa_values)
            sign_change_rate = self._compute_sign_change_rate(faa_values)
            shift_speed = self._compute_shift_speed(faa_values)
            recovery_index = self._compute_recovery_index(
                faa_values, user["baseline_faa"]
            )
            valence_range = float(max(faa_values) - min(faa_values))
        else:
            # Not enough data -- return neutral defaults
            faa_variability = 0.5
            sign_change_rate = 0.5
            shift_speed = 0.5
            recovery_index = 0.5
            if n_epochs >= 2:
                valence_range = float(max(faa_values) - min(faa_values))
            else:
                valence_range = 0.0

        # Composite flexibility score (0-100)
        flexibility_score = self._compute_composite_score(
            faa_variability, sign_change_rate, shift_speed, recovery_index
        )
        flexibility_level = _score_to_level(flexibility_score)

        result = {
            "flexibility_score": round(flexibility_score, 2),
            "flexibility_level": flexibility_level,
            "faa_current": round(faa_current, 6),
            "faa_variability": round(faa_variability, 4),
            "sign_change_rate": round(sign_change_rate, 4),
            "shift_speed": round(shift_speed, 4),
            "recovery_index": round(recovery_index, 4),
            "valence_range": round(valence_range, 6),
            "has_baseline": has_baseline,
        }

        # Store in assess history (capped)
        if len(user["assess_history"]) >= _MAX_HISTORY:
            user["assess_history"] = user["assess_history"][-(_MAX_HISTORY - 1) :]
        user["assess_history"].append(result)

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict[str, Any]:
        """Get summary statistics for a user's session.

        Args:
            user_id: User identifier.

        Returns:
            Dict with n_epochs, mean_flexibility, dominant_level, faa_trajectory.
        """
        user = self._ensure_user(user_id)
        history = user["assess_history"]

        if not history:
            return {
                "n_epochs": 0,
                "mean_flexibility": 0.0,
                "dominant_level": "rigid",
                "faa_trajectory": [],
            }

        scores = [h["flexibility_score"] for h in history]
        mean_flex = float(np.mean(scores))
        dominant_level = _score_to_level(mean_flex)

        faa_trajectory = [h["faa_current"] for h in history]

        return {
            "n_epochs": len(history),
            "mean_flexibility": round(mean_flex, 2),
            "dominant_level": dominant_level,
            "faa_trajectory": faa_trajectory,
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get assessment history for a user.

        Args:
            user_id: User identifier.
            last_n: If provided, return only the last N entries.

        Returns:
            List of assessment result dicts.
        """
        user = self._ensure_user(user_id)
        history = user["assess_history"]

        if last_n is not None and last_n > 0:
            return list(history[-last_n:])
        return list(history)

    def reset(self, user_id: str = "default") -> None:
        """Reset all state for a user (baseline, history, FAA buffer).

        Args:
            user_id: User identifier.
        """
        if user_id in self._users:
            del self._users[user_id]

    # ── Private computation helpers ──────────────────────────────────────

    @staticmethod
    def _compute_variability(faa_values: List[float]) -> float:
        """Coefficient of variation of FAA values, normalized to 0-1.

        Higher CV = more variable FAA = more flexible.
        """
        arr = np.array(faa_values)
        std = float(np.std(arr))
        mean_abs = float(np.mean(np.abs(arr)))

        if mean_abs < 1e-10:
            # If mean is near zero, use std directly
            # Typical FAA std for flexible individuals is ~0.1-0.3
            return float(np.clip(std / 0.3, 0.0, 1.0))

        cv = std / mean_abs
        # Normalize: CV of ~1.0 maps to variability of 1.0
        return float(np.clip(cv, 0.0, 1.0))

    @staticmethod
    def _compute_sign_change_rate(faa_values: List[float]) -> float:
        """Fraction of consecutive epochs where FAA changes sign.

        High sign-change rate = frequent valence flips = more flexible.
        """
        if len(faa_values) < 2:
            return 0.0

        arr = np.array(faa_values)
        signs = np.sign(arr)
        # Count transitions (ignore zeros -- treat as no change)
        changes = 0
        total = 0
        for i in range(1, len(signs)):
            if signs[i] != 0 and signs[i - 1] != 0:
                total += 1
                if signs[i] != signs[i - 1]:
                    changes += 1
            elif signs[i] != 0 or signs[i - 1] != 0:
                total += 1

        if total == 0:
            return 0.0

        return float(np.clip(changes / total, 0.0, 1.0))

    @staticmethod
    def _compute_shift_speed(faa_values: List[float]) -> float:
        """Mean absolute epoch-to-epoch FAA change, normalized to 0-1.

        Larger shifts = faster emotional transitions = more flexible.
        """
        if len(faa_values) < 2:
            return 0.0

        arr = np.array(faa_values)
        diffs = np.abs(np.diff(arr))
        mean_diff = float(np.mean(diffs))

        # Normalize: typical FAA shift of ~0.2 per epoch maps to 1.0
        return float(np.clip(mean_diff / 0.2, 0.0, 1.0))

    @staticmethod
    def _compute_recovery_index(
        faa_values: List[float], baseline_faa: Optional[float]
    ) -> float:
        """How quickly FAA returns toward baseline after a deviation.

        Measures the tendency for FAA to move back toward baseline after
        deviating. High recovery = good emotional regulation.

        Without baseline, uses the session mean FAA as reference.
        """
        if len(faa_values) < 3:
            return 0.5

        ref = baseline_faa if baseline_faa is not None else float(np.mean(faa_values))

        # For each pair of consecutive epochs, check if deviation from ref decreased
        deviations = [abs(v - ref) for v in faa_values]
        recoveries = 0
        total = 0

        for i in range(1, len(deviations)):
            if deviations[i - 1] > 0.01:  # Only count when there was a deviation
                total += 1
                if deviations[i] < deviations[i - 1]:
                    recoveries += 1

        if total == 0:
            return 0.5

        return float(np.clip(recoveries / total, 0.0, 1.0))

    @staticmethod
    def _compute_composite_score(
        faa_variability: float,
        sign_change_rate: float,
        shift_speed: float,
        recovery_index: float,
    ) -> float:
        """Weighted composite flexibility score (0-100).

        Weights:
            - faa_variability:  25% (overall emotional range)
            - sign_change_rate: 25% (valence flip frequency)
            - shift_speed:      25% (transition magnitude)
            - recovery_index:   25% (regulation / recovery ability)
        """
        raw = (
            0.25 * faa_variability
            + 0.25 * sign_change_rate
            + 0.25 * shift_speed
            + 0.25 * recovery_index
        )
        return float(np.clip(raw * 100.0, 0.0, 100.0))
