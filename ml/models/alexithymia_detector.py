"""Alexithymia screening from 4-channel Muse 2 EEG.

Alexithymia = difficulty identifying and describing emotions (~10% prevalence).
This detector uses four EEG biomarkers validated in the literature:

  1. FAA flatness -- alexithymics show reduced FAA modulation in response to
     emotional stimuli (Luminet et al., 2004; Pollatos & Gramann, 2012).
  2. Right-hemisphere theta/alpha dominance -- atypical lateralization with
     increased right-sided slow-wave activity (Aftanas & Varlamov, 2007).
  3. Reduced interhemispheric coherence -- diminished communication between
     left and right hemispheres during emotion processing (Imperatori et al., 2016).
  4. Low emotional modulation -- EEG power spectrum changes less in response
     to emotional stimuli compared to baseline (Franz et al., 2004).

All thresholds are population-average heuristics.  Per-user baseline calibration
improves accuracy significantly.

Channel order (Muse 2): ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10.
Sampling rate: 256 Hz.

IMPORTANT: This is a screening indicator, not a diagnostic tool.
"""

import numpy as np
from typing import Dict, List, Optional

from scipy.signal import welch, coherence as scipy_coherence

# NumPy 2.0 renamed np.trapz -> np.trapezoid; 1.x only has np.trapz
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

# Muse 2 channel layout
_CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]

# EEG frequency bands (Hz)
_BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
}

# Risk level thresholds (score 0-100)
_RISK_THRESHOLDS = {
    "low": (0, 25),
    "mild": (25, 45),
    "moderate": (45, 70),
    "elevated": (70, 100),
}

# How many epochs of history to keep per user
_MAX_HISTORY = 500

# Disclaimer attached to every screening result
_DISCLAIMER = (
    "This is a wellness indicator based on EEG biomarkers, not a medical "
    "device or clinical assessment tool. Professional evaluation using "
    "validated instruments (e.g., TAS-20) is required for any clinical "
    "assessment."
)


def _band_power(signal: np.ndarray, fs: float, low: float, high: float) -> float:
    """Relative power in a frequency band via Welch PSD."""
    nperseg = min(len(signal), int(fs * 2))
    if nperseg < 4:
        return 0.0
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    total = _trapezoid(psd, freqs)
    if total <= 0:
        return 0.0
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return 0.0
    return float(_trapezoid(psd[mask], freqs[mask]) / total)


def _channel_band_powers(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """Extract theta, alpha, beta relative powers for one channel."""
    result = {}
    for band_name, (low, high) in _BANDS.items():
        result[band_name] = _band_power(signal, fs, low, high)
    return result


def _interhemispheric_coherence(
    left: np.ndarray, right: np.ndarray, fs: float, low: float, high: float
) -> float:
    """Mean coherence between left and right channel in a frequency band."""
    nperseg = min(len(left), len(right), int(fs * 2))
    if nperseg < 4:
        return 0.0
    freqs, coh = scipy_coherence(left, right, fs=fs, nperseg=nperseg)
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return 0.0
    return float(np.mean(coh[mask]))


def _compute_faa(signals: np.ndarray, fs: float) -> float:
    """Frontal alpha asymmetry: log(AF8_alpha) - log(AF7_alpha).

    Positive = left-frontal activation (approach motivation).
    """
    if signals.shape[0] < 3:
        return 0.0
    af7_alpha = max(_band_power(signals[1], fs, 8.0, 12.0), 1e-12)
    af8_alpha = max(_band_power(signals[2], fs, 8.0, 12.0), 1e-12)
    return float(np.log(af8_alpha) - np.log(af7_alpha))


class AlexithymiaDetector:
    """Screen for alexithymia markers from 4-channel Muse 2 EEG.

    Multi-user support: each user_id maintains independent baseline,
    history, and session statistics.
    """

    def __init__(self, fs: float = 256.0):
        self.fs = fs
        # Per-user state: {user_id: {"baseline": dict, "history": list}}
        self._users: Dict[str, dict] = {}

    # -- internal helpers --

    def _ensure_user(self, user_id: str) -> dict:
        if user_id not in self._users:
            self._users[user_id] = {
                "baseline": None,
                "history": [],
            }
        return self._users[user_id]

    @staticmethod
    def _ensure_multichannel(signals: np.ndarray) -> np.ndarray:
        """Reshape 1-D input to (1, n_samples)."""
        if signals.ndim == 1:
            return signals.reshape(1, -1)
        return signals

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 70:
            return "elevated"
        elif score >= 45:
            return "moderate"
        elif score >= 25:
            return "mild"
        return "low"

    # -- public API --

    def set_baseline(
        self, signals: np.ndarray, fs: Optional[float] = None, user_id: str = "default"
    ) -> Dict:
        """Record resting-state baseline for a user.

        Call during 2-3 min eyes-closed rest.  Captures baseline FAA and
        interhemispheric alpha coherence for later comparison.

        Args:
            signals: (n_channels, n_samples) EEG array.
            fs: Sampling rate override (uses self.fs if None).
            user_id: User identifier for multi-user support.

        Returns:
            Dict with baseline_set, baseline_faa, baseline_coherence.
        """
        fs = fs or self.fs
        signals = self._ensure_multichannel(signals)
        user = self._ensure_user(user_id)

        baseline_faa = _compute_faa(signals, fs)

        # Interhemispheric alpha coherence (AF7-AF8 and TP9-TP10 if available)
        n_ch = signals.shape[0]
        coh_values = []
        if n_ch >= 3:
            coh_values.append(
                _interhemispheric_coherence(signals[1], signals[2], fs, 8.0, 12.0)
            )
        if n_ch >= 4:
            coh_values.append(
                _interhemispheric_coherence(signals[0], signals[3], fs, 8.0, 12.0)
            )
        baseline_coherence = float(np.mean(coh_values)) if coh_values else 0.5

        # Per-channel band powers for modulation reference
        ch_powers = []
        for ch in range(min(n_ch, 4)):
            ch_powers.append(_channel_band_powers(signals[ch], fs))

        user["baseline"] = {
            "faa": baseline_faa,
            "coherence": baseline_coherence,
            "channel_powers": ch_powers,
        }

        return {
            "baseline_set": True,
            "baseline_faa": round(baseline_faa, 4),
            "baseline_coherence": round(baseline_coherence, 4),
        }

    def screen(
        self, signals: np.ndarray, fs: Optional[float] = None, user_id: str = "default"
    ) -> Dict:
        """Screen for alexithymia markers from EEG signals.

        Args:
            signals: (n_channels, n_samples) EEG array.
            fs: Sampling rate override.
            user_id: User identifier.

        Returns:
            Dict with alexithymia_score, risk_level, component scores,
            biomarkers, disclaimer, and has_baseline flag.
        """
        fs = fs or self.fs
        signals = self._ensure_multichannel(signals)
        user = self._ensure_user(user_id)
        has_baseline = user["baseline"] is not None
        n_ch = min(signals.shape[0], 4)

        # ---- 1. FAA flatness ----
        # Compute current FAA and compare to recent history
        current_faa = _compute_faa(signals, fs)

        # Collect recent FAA values (from history) + current
        recent_faas = [h["biomarkers"]["current_faa"] for h in user["history"][-9:]]
        recent_faas.append(current_faa)

        if len(recent_faas) >= 2:
            faa_std = float(np.std(recent_faas))
            # Normalize: typical FAA std is ~0.3-0.5 for healthy individuals
            # Lower std = flatter = more alexithymic
            faa_flatness = float(np.clip(1.0 - faa_std / 0.5, 0.0, 1.0))
        else:
            # Single epoch: compare absolute FAA to expected range
            # Very small absolute FAA suggests flat response
            faa_flatness = float(np.clip(1.0 - abs(current_faa) / 0.3, 0.0, 1.0))

        # ---- 2. Right-hemisphere theta+alpha dominance ----
        # Compare right (AF8=ch2, TP10=ch3) vs left (AF7=ch1, TP9=ch0)
        left_power = 0.0
        right_power = 0.0

        if n_ch >= 3:
            # AF7 (ch1) and AF8 (ch2)
            af7_powers = _channel_band_powers(signals[1], fs)
            af8_powers = _channel_band_powers(signals[2], fs)
            left_power += af7_powers["theta"] + af7_powers["alpha"]
            right_power += af8_powers["theta"] + af8_powers["alpha"]

        if n_ch >= 4:
            # TP9 (ch0) and TP10 (ch3)
            tp9_powers = _channel_band_powers(signals[0], fs)
            tp10_powers = _channel_band_powers(signals[3], fs)
            left_power += tp9_powers["theta"] + tp9_powers["alpha"]
            right_power += tp10_powers["theta"] + tp10_powers["alpha"]

        if n_ch < 3:
            # Single/dual channel fallback
            ch0_powers = _channel_band_powers(signals[0], fs)
            left_power = ch0_powers["theta"] + ch0_powers["alpha"]
            right_power = left_power  # no asymmetry info

        total_lr = left_power + right_power
        if total_lr > 0:
            right_dominance = float(np.clip(
                (right_power - left_power) / total_lr + 0.5,
                0.0, 1.0
            ))
        else:
            right_dominance = 0.5

        # ---- 3. Coherence deficit ----
        coh_values = []
        if n_ch >= 3:
            coh_values.append(
                _interhemispheric_coherence(signals[1], signals[2], fs, 8.0, 12.0)
            )
        if n_ch >= 4:
            coh_values.append(
                _interhemispheric_coherence(signals[0], signals[3], fs, 8.0, 12.0)
            )

        if coh_values:
            mean_coherence = float(np.mean(coh_values))
            coherence_deficit = float(np.clip(1.0 - mean_coherence, 0.0, 1.0))
        else:
            mean_coherence = 0.5
            coherence_deficit = 0.5

        # ---- 4. Emotional modulation ----
        # How much current EEG differs from baseline
        if has_baseline:
            baseline_powers = user["baseline"]["channel_powers"]
            power_diffs = []
            for ch in range(min(n_ch, len(baseline_powers))):
                current_powers = _channel_band_powers(signals[ch], fs)
                for band in ("theta", "alpha", "beta"):
                    diff = abs(current_powers[band] - baseline_powers[ch][band])
                    power_diffs.append(diff)
            if power_diffs:
                mean_diff = float(np.mean(power_diffs))
                # Normalize: typical modulation is ~0.05-0.15 relative power change
                emotional_modulation = float(np.clip(mean_diff / 0.10, 0.0, 1.0))
            else:
                emotional_modulation = 0.5
        else:
            # Without baseline, estimate from signal variability
            # Higher spectral variability = more emotional modulation
            ch_vars = []
            for ch in range(n_ch):
                powers = _channel_band_powers(signals[ch], fs)
                ch_vars.append(float(np.std(list(powers.values()))))
            emotional_modulation = float(np.clip(np.mean(ch_vars) / 0.15, 0.0, 1.0))

        # ---- Composite score (0-100) ----
        # Higher = more alexithymic markers
        # FAA flatness and low emotional modulation are the strongest markers
        raw_score = (
            0.30 * faa_flatness
            + 0.25 * (1.0 - emotional_modulation)  # low modulation = alexithymic
            + 0.25 * coherence_deficit
            + 0.20 * right_dominance
        )
        alexithymia_score = float(np.clip(raw_score * 100, 0.0, 100.0))

        risk_level = self._risk_level(alexithymia_score)

        result = {
            "alexithymia_score": round(alexithymia_score, 1),
            "risk_level": risk_level,
            "faa_flatness": round(faa_flatness, 4),
            "right_dominance": round(right_dominance, 4),
            "coherence_deficit": round(coherence_deficit, 4),
            "emotional_modulation": round(emotional_modulation, 4),
            "biomarkers": {
                "current_faa": round(current_faa, 4),
                "mean_coherence": round(mean_coherence, 4),
                "left_theta_alpha_power": round(left_power, 4),
                "right_theta_alpha_power": round(right_power, 4),
                "faa_std": round(float(np.std(recent_faas)), 4),
            },
            "disclaimer": _DISCLAIMER,
            "has_baseline": has_baseline,
        }

        # Store in history (capped at _MAX_HISTORY)
        user["history"].append(result)
        if len(user["history"]) > _MAX_HISTORY:
            user["history"] = user["history"][-_MAX_HISTORY:]

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get summary statistics for a user's screening history.

        Returns:
            Dict with n_epochs, mean_score, has_baseline.
        """
        user = self._ensure_user(user_id)
        history = user["history"]
        has_baseline = user["baseline"] is not None

        if not history:
            return {
                "n_epochs": 0,
                "mean_score": 0.0,
                "has_baseline": has_baseline,
            }

        scores = [h["alexithymia_score"] for h in history]
        return {
            "n_epochs": len(history),
            "mean_score": round(float(np.mean(scores)), 1),
            "has_baseline": has_baseline,
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Return screening history for a user.

        Args:
            user_id: User identifier.
            last_n: If provided, return only the last N entries.

        Returns:
            List of screening result dicts.
        """
        user = self._ensure_user(user_id)
        history = user["history"]
        if last_n is not None:
            return list(history[-last_n:])
        return list(history)

    def reset(self, user_id: str = "default") -> None:
        """Clear all state (baseline + history) for a user."""
        if user_id in self._users:
            self._users[user_id] = {
                "baseline": None,
                "history": [],
            }
