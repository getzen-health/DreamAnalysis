"""Emotional reactivity vs regulation ratio from EEG.

Measures the balance between emotional reactivity (fast, automatic responses
to stimuli) and emotional regulation (ability to modulate and recover from
those responses).

Key biomarkers:
- Reactivity: magnitude of early beta/alpha changes from baseline (0-2 sec),
  FAA shift magnitude, high-beta spike intensity
- Regulation: alpha recovery speed (2-4+ sec), FAA return toward baseline,
  alpha rebound after stress

Scientific basis:
- Gross (2015): process model of emotion regulation
- Dennis & Hajcak (2009): ERP/EEG markers of regulation predict anxiety
- Davidson (2000): FAA as index of affective style and regulation capacity
- Beauchaine (2015): beta/alpha ratio tracks emotional lability
- High regulation/reactivity ratio correlates with emotional intelligence

Balance states:
- well_regulated: rr_ratio > 1.5 (regulation dominates)
- balanced: rr_ratio 0.8-1.5 (healthy equilibrium)
- reactive: rr_ratio 0.4-0.8 (reactivity dominant)
- dysregulated: rr_ratio < 0.4 (poor regulation, high reactivity)

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

# Maximum epochs to keep in history per user
_MAX_HISTORY = 500

# Muse 2 channel indices
_CH_AF7 = 1
_CH_AF8 = 2

# Frequency band definitions (Hz)
_ALPHA_BAND = (8.0, 12.0)
_BETA_BAND = (12.0, 30.0)
_HIGH_BETA_BAND = (20.0, 30.0)

# Balance state thresholds (rr_ratio)
_STATE_THRESHOLDS = {
    "well_regulated": 1.5,
    "balanced": 0.8,
    "reactive": 0.4,
    # below 0.4 = dysregulated
}

# Normalization ceilings for absolute-threshold mode (no baseline)
_ABS_ALPHA_BETA_RATIO_CEIL = 2.0   # typical relaxed alpha/beta ~1.5-2.0
_ABS_HIGH_BETA_FRAC_CEIL = 0.5     # high-beta as fraction of total beta


def _band_power_welch(
    signal: np.ndarray, fs: float, band: tuple[float, float]
) -> float:
    """Compute band power via Welch's method.

    Args:
        signal: 1D EEG signal.
        fs: Sampling rate in Hz.
        band: (low_freq, high_freq) tuple.

    Returns:
        Band power (float, non-negative). Floor of 1e-12.
    """
    n = len(signal)
    if n < 16:
        return 1e-12

    nperseg = min(256, n)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 1e-12

    band_freqs = freqs[mask]
    band_psd = psd[mask]

    try:
        power = float(np.trapezoid(band_psd, band_freqs))
    except AttributeError:
        power = float(np.trapz(band_psd, band_freqs))

    return max(power, 1e-12)


def _compute_faa(signals: np.ndarray, fs: float) -> float:
    """Compute frontal alpha asymmetry.

    FAA = log(AF8_alpha) - log(AF7_alpha)

    Returns 0.0 if fewer than 3 channels (need indices 1=AF7, 2=AF8).
    """
    if signals.ndim == 1:
        return 0.0
    if signals.shape[0] < 3:
        return 0.0

    left_alpha = _band_power_welch(signals[_CH_AF7], fs, _ALPHA_BAND)
    right_alpha = _band_power_welch(signals[_CH_AF8], fs, _ALPHA_BAND)

    return float(np.log(right_alpha) - np.log(left_alpha))


def _classify_balance_state(rr_ratio: float) -> str:
    """Map regulation/reactivity ratio to a balance state label."""
    if rr_ratio > _STATE_THRESHOLDS["well_regulated"]:
        return "well_regulated"
    elif rr_ratio >= _STATE_THRESHOLDS["balanced"]:
        return "balanced"
    elif rr_ratio >= _STATE_THRESHOLDS["reactive"]:
        return "reactive"
    else:
        return "dysregulated"


class ReactivityRegulationTracker:
    """Track emotional reactivity vs regulation from EEG.

    Designed for 4-channel Muse 2 (TP9, AF7, AF8, TP10, 256 Hz).
    Supports multiple users via user_id parameter.

    Usage:
        tracker = ReactivityRegulationTracker(fs=256.0)

        # Set resting baseline (recommended: 2-3 min eyes-closed)
        tracker.set_baseline(resting_eeg, user_id="user1")

        # Feed successive 4-sec EEG epochs
        for epoch in epochs:
            result = tracker.assess(epoch, user_id="user1")
            print(result["rr_ratio"], result["balance_state"])

        # Session summary
        stats = tracker.get_session_stats(user_id="user1")
    """

    def __init__(self, fs: float = 256.0) -> None:
        self.fs = fs
        self._users: Dict[str, Dict[str, Any]] = {}

    def _ensure_user(self, user_id: str) -> Dict[str, Any]:
        """Initialize user state if not present."""
        if user_id not in self._users:
            self._users[user_id] = {
                "baseline_alpha": None,
                "baseline_beta": None,
                "baseline_faa": None,
                "assess_history": [],
                "alpha_history": deque(maxlen=_MAX_HISTORY),
            }
        return self._users[user_id]

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Set resting-state baseline from multichannel EEG.

        Call with 2-3 minutes of eyes-closed resting EEG. The baseline
        alpha, beta, and FAA values serve as reference points for
        computing reactivity and regulation.

        Args:
            signals: (4, n_samples) multichannel EEG or (n_samples,) single-channel.
            fs: Sampling rate (defaults to self.fs).
            user_id: User identifier for multi-user support.

        Returns:
            Dict with baseline_set, baseline_alpha, baseline_beta, baseline_faa.
        """
        if fs is None:
            fs = self.fs

        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        user = self._ensure_user(user_id)

        # Compute baseline alpha and beta averaged across frontal channels
        alpha_powers = []
        beta_powers = []
        n_ch = signals.shape[0]
        for ch in range(n_ch):
            alpha_powers.append(_band_power_welch(signals[ch], fs, _ALPHA_BAND))
            beta_powers.append(_band_power_welch(signals[ch], fs, _BETA_BAND))

        baseline_alpha = float(np.mean(alpha_powers))
        baseline_beta = float(np.mean(beta_powers))
        baseline_faa = _compute_faa(signals, fs)

        user["baseline_alpha"] = baseline_alpha
        user["baseline_beta"] = baseline_beta
        user["baseline_faa"] = baseline_faa

        return {
            "baseline_set": True,
            "baseline_alpha": round(baseline_alpha, 6),
            "baseline_beta": round(baseline_beta, 6),
            "baseline_faa": round(baseline_faa, 6),
        }

    def assess(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Assess reactivity and regulation from a single EEG epoch.

        Reactivity is measured by the magnitude of beta/alpha change and
        FAA shift from baseline. Regulation is measured by alpha recovery
        speed and the tendency for alpha to rebound.

        When no baseline is set, uses absolute thresholds based on
        population-average alpha/beta ratios.

        Args:
            signals: (4, n_samples) multichannel EEG or (n_samples,) single-channel.
            fs: Sampling rate (defaults to self.fs).
            user_id: User identifier for multi-user support.

        Returns:
            Dict with reactivity_index, regulation_index, rr_ratio,
            balance_state, alpha_change, beta_change, faa_shift,
            recovery_speed, has_baseline.
        """
        if fs is None:
            fs = self.fs

        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        user = self._ensure_user(user_id)
        has_baseline = user["baseline_alpha"] is not None

        # Compute current alpha, beta, high-beta averaged across channels
        n_ch = signals.shape[0]
        alpha_powers = []
        beta_powers = []
        high_beta_powers = []
        for ch in range(n_ch):
            alpha_powers.append(_band_power_welch(signals[ch], fs, _ALPHA_BAND))
            beta_powers.append(_band_power_welch(signals[ch], fs, _BETA_BAND))
            high_beta_powers.append(
                _band_power_welch(signals[ch], fs, _HIGH_BETA_BAND)
            )

        current_alpha = float(np.mean(alpha_powers))
        current_beta = float(np.mean(beta_powers))
        current_high_beta = float(np.mean(high_beta_powers))
        current_faa = _compute_faa(signals, fs)

        # Track alpha history for recovery computation
        user["alpha_history"].append(current_alpha)

        if has_baseline:
            reactivity_index, regulation_index, alpha_change, beta_change, \
                faa_shift, recovery_speed = self._compute_with_baseline(
                    user, current_alpha, current_beta, current_high_beta,
                    current_faa,
                )
        else:
            reactivity_index, regulation_index, alpha_change, beta_change, \
                faa_shift, recovery_speed = self._compute_without_baseline(
                    user, current_alpha, current_beta, current_high_beta,
                    current_faa,
                )

        # Compute rr_ratio (regulation / reactivity)
        if reactivity_index > 1e-6:
            rr_ratio = regulation_index / reactivity_index
        else:
            # No reactivity detected -- default to well-regulated
            rr_ratio = 2.0

        balance_state = _classify_balance_state(rr_ratio)

        result = {
            "reactivity_index": round(float(np.clip(reactivity_index, 0.0, 1.0)), 4),
            "regulation_index": round(float(np.clip(regulation_index, 0.0, 1.0)), 4),
            "rr_ratio": round(float(rr_ratio), 4),
            "balance_state": balance_state,
            "alpha_change": round(float(alpha_change), 6),
            "beta_change": round(float(beta_change), 6),
            "faa_shift": round(float(faa_shift), 6),
            "recovery_speed": round(float(np.clip(recovery_speed, 0.0, 1.0)), 4),
            "has_baseline": has_baseline,
        }

        # Store in history (capped)
        if len(user["assess_history"]) >= _MAX_HISTORY:
            user["assess_history"] = user["assess_history"][-(_MAX_HISTORY - 1):]
        user["assess_history"].append(result)

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict[str, Any]:
        """Get summary statistics for a user's session.

        Args:
            user_id: User identifier.

        Returns:
            Dict with n_epochs, mean_rr_ratio, dominant_state,
            state_distribution.
        """
        user = self._ensure_user(user_id)
        history = user["assess_history"]

        if not history:
            return {
                "n_epochs": 0,
                "mean_rr_ratio": 0.0,
                "dominant_state": "dysregulated",
                "state_distribution": {},
            }

        rr_ratios = [h["rr_ratio"] for h in history]
        mean_rr = float(np.mean(rr_ratios))
        dominant_state = _classify_balance_state(mean_rr)

        # Count state distribution
        state_counts: Dict[str, int] = {}
        for h in history:
            state = h["balance_state"]
            state_counts[state] = state_counts.get(state, 0) + 1

        return {
            "n_epochs": len(history),
            "mean_rr_ratio": round(mean_rr, 4),
            "dominant_state": dominant_state,
            "state_distribution": state_counts,
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get assessment history for a user.

        Args:
            user_id: User identifier.
            last_n: If provided, return only the last N entries.

        Returns:
            List of assessment result dicts (copy, not reference).
        """
        user = self._ensure_user(user_id)
        history = user["assess_history"]

        if last_n is not None and last_n > 0:
            return list(history[-last_n:])
        return list(history)

    def reset(self, user_id: str = "default") -> None:
        """Reset all state for a user (baseline, history, buffers).

        Args:
            user_id: User identifier.
        """
        if user_id in self._users:
            del self._users[user_id]

    # -- Private computation helpers ------------------------------------------

    def _compute_with_baseline(
        self,
        user: Dict[str, Any],
        current_alpha: float,
        current_beta: float,
        current_high_beta: float,
        current_faa: float,
    ) -> tuple:
        """Compute reactivity and regulation using baseline reference.

        Returns:
            (reactivity_index, regulation_index, alpha_change, beta_change,
             faa_shift, recovery_speed)
        """
        baseline_alpha = user["baseline_alpha"]
        baseline_beta = user["baseline_beta"]
        baseline_faa = user["baseline_faa"]

        # Alpha change: negative = alpha suppression (stress/arousal)
        if baseline_alpha > 1e-12:
            alpha_change = (current_alpha - baseline_alpha) / baseline_alpha
        else:
            alpha_change = 0.0

        # Beta change: positive = beta increase (arousal/stress)
        if baseline_beta > 1e-12:
            beta_change = (current_beta - baseline_beta) / baseline_beta
        else:
            beta_change = 0.0

        # FAA shift: magnitude of FAA deviation from baseline
        faa_shift = current_faa - baseline_faa

        # --- Reactivity index ---
        # Components: beta increase, alpha suppression, FAA shift, high-beta spike
        beta_reactivity = float(np.clip(max(0.0, beta_change) / 1.0, 0.0, 1.0))
        alpha_suppression = float(np.clip(max(0.0, -alpha_change) / 0.5, 0.0, 1.0))
        faa_reactivity = float(np.clip(abs(faa_shift) / 0.5, 0.0, 1.0))

        # High-beta spike (stress reactivity)
        if baseline_beta > 1e-12:
            hb_frac = current_high_beta / (current_beta + 1e-12)
            baseline_hb_frac = 0.3  # population average
            hb_spike = float(np.clip(
                max(0.0, hb_frac - baseline_hb_frac) / 0.3, 0.0, 1.0
            ))
        else:
            hb_spike = 0.0

        reactivity_index = float(np.clip(
            0.30 * beta_reactivity
            + 0.25 * alpha_suppression
            + 0.25 * faa_reactivity
            + 0.20 * hb_spike,
            0.0, 1.0,
        ))

        # --- Regulation index ---
        # Components: alpha recovery, FAA recovery, alpha rebound
        recovery_speed = self._compute_recovery_speed(user, baseline_alpha)

        # Alpha rebound: current alpha exceeds or returns to baseline
        alpha_rebound = float(np.clip(
            1.0 - max(0.0, -alpha_change) / 0.5, 0.0, 1.0
        ))

        # FAA recovery: how close current FAA is to baseline
        faa_recovery = float(np.clip(1.0 - abs(faa_shift) / 0.5, 0.0, 1.0))

        regulation_index = float(np.clip(
            0.40 * recovery_speed
            + 0.35 * alpha_rebound
            + 0.25 * faa_recovery,
            0.0, 1.0,
        ))

        return (
            reactivity_index,
            regulation_index,
            alpha_change,
            beta_change,
            faa_shift,
            recovery_speed,
        )

    def _compute_without_baseline(
        self,
        user: Dict[str, Any],
        current_alpha: float,
        current_beta: float,
        current_high_beta: float,
        current_faa: float,
    ) -> tuple:
        """Compute reactivity and regulation using absolute thresholds.

        Used when no baseline has been set. Uses population-average
        thresholds for alpha/beta ratios.

        Returns:
            (reactivity_index, regulation_index, alpha_change, beta_change,
             faa_shift, recovery_speed)
        """
        # Alpha/beta ratio: high ratio = relaxed, low = reactive
        ab_ratio = current_alpha / (current_beta + 1e-12)

        # High-beta fraction of total beta
        hb_fraction = current_high_beta / (current_beta + 1e-12)

        # --- Reactivity index (absolute mode) ---
        # Low alpha/beta ratio indicates high arousal/reactivity
        ab_reactivity = float(np.clip(
            1.0 - ab_ratio / _ABS_ALPHA_BETA_RATIO_CEIL, 0.0, 1.0
        ))

        # High-beta fraction indicates stress reactivity
        hb_reactivity = float(np.clip(
            hb_fraction / _ABS_HIGH_BETA_FRAC_CEIL, 0.0, 1.0
        ))

        # FAA magnitude (without baseline, just use absolute value)
        faa_abs_reactivity = float(np.clip(abs(current_faa) / 0.5, 0.0, 1.0))

        reactivity_index = float(np.clip(
            0.40 * ab_reactivity
            + 0.35 * hb_reactivity
            + 0.25 * faa_abs_reactivity,
            0.0, 1.0,
        ))

        # --- Regulation index (absolute mode) ---
        # High alpha/beta ratio = good regulation
        ab_regulation = float(np.clip(
            ab_ratio / _ABS_ALPHA_BETA_RATIO_CEIL, 0.0, 1.0
        ))

        # Low high-beta fraction = not stress-reactive
        hb_regulation = float(np.clip(
            1.0 - hb_fraction / _ABS_HIGH_BETA_FRAC_CEIL, 0.0, 1.0
        ))

        # Recovery speed from alpha history
        recovery_speed = self._compute_recovery_speed_absolute(user)

        regulation_index = float(np.clip(
            0.40 * ab_regulation
            + 0.30 * hb_regulation
            + 0.30 * recovery_speed,
            0.0, 1.0,
        ))

        # Without baseline, changes are relative to population average
        alpha_change = ab_ratio - 1.0  # deviation from a ~1.0 ratio
        beta_change = 1.0 - ab_ratio   # inverse
        faa_shift = current_faa  # no baseline reference

        return (
            reactivity_index,
            regulation_index,
            alpha_change,
            beta_change,
            faa_shift,
            recovery_speed,
        )

    @staticmethod
    def _compute_recovery_speed(
        user: Dict[str, Any], baseline_alpha: float
    ) -> float:
        """Compute how quickly alpha recovers toward baseline.

        Looks at the last few alpha values. If alpha is trending back
        toward baseline, recovery speed is high.
        """
        alpha_vals = list(user["alpha_history"])
        if len(alpha_vals) < 3:
            return 0.5  # not enough data

        # Look at last 5 values (or fewer if not available)
        recent = alpha_vals[-min(5, len(alpha_vals)):]
        deviations = [abs(a - baseline_alpha) for a in recent]

        # Count how many consecutive deviations decreased
        recoveries = 0
        total = 0
        for i in range(1, len(deviations)):
            if deviations[i - 1] > 1e-12:
                total += 1
                if deviations[i] < deviations[i - 1]:
                    recoveries += 1

        if total == 0:
            return 0.5

        return float(np.clip(recoveries / total, 0.0, 1.0))

    @staticmethod
    def _compute_recovery_speed_absolute(user: Dict[str, Any]) -> float:
        """Compute recovery speed without baseline (absolute mode).

        Measures how stable alpha is across recent epochs. Stable alpha
        with low variance indicates good regulation.
        """
        alpha_vals = list(user["alpha_history"])
        if len(alpha_vals) < 3:
            return 0.5

        recent = alpha_vals[-min(5, len(alpha_vals)):]
        arr = np.array(recent)
        mean_val = float(np.mean(arr))
        if mean_val < 1e-12:
            return 0.5

        cv = float(np.std(arr) / mean_val)
        # Low CV = stable = good regulation. CV of 0.5 maps to recovery 0.0
        return float(np.clip(1.0 - cv / 0.5, 0.0, 1.0))
