"""Emotional Memory Encoding Predictor via Theta-Gamma Cross-Frequency Coupling.

Detects when emotional experiences are being strongly encoded into long-term
memory by measuring theta-gamma cross-frequency coupling (CFC). The modulation
index (MI) quantifies how much gamma amplitude (30-45 Hz) is modulated by
theta phase (4-8 Hz) -- a signature of hippocampal memory formation.

Channel layout (Muse 2): ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10

Scientific basis:
    Canolty et al. (2006) -- theta-gamma CFC in human hippocampus during
        memory encoding; MI as a metric for phase-amplitude coupling
    Lisman & Jensen (2013) -- theta-gamma neural code for memory:
        gamma cycles nested within theta phases encode individual items
    Tort et al. (2010) -- Modulation Index (MI) based on KL divergence
        from uniform distribution of gamma amplitude across theta phase bins
    Lega et al. (2012) -- frontal theta power predicts successful encoding;
        theta-gamma coupling increases during emotional memory formation
    Nyhus & Curran (2010) -- frontal theta (4-8 Hz) increases during
        successful memory encoding, especially for emotional material
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch


# -- Constants ----------------------------------------------------------------

# Frequency bands
_THETA_BAND = (4.0, 8.0)
_GAMMA_BAND = (30.0, 45.0)

# Phase bins for modulation index
_N_PHASE_BINS = 18  # 20 degrees each

# Minimum samples for meaningful computation
_MIN_SAMPLES = 128

# Minimum seconds for theta-gamma coupling (need ~2 theta cycles)
_MIN_CFC_SECONDS = 2.0

# History cap per user
_MAX_HISTORY = 500

# Encoding level thresholds
_STRONG_THRESHOLD = 0.65
_MODERATE_THRESHOLD = 0.40
_WEAK_THRESHOLD = 0.20

# Muse 2 channel names
_CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]


def _bandpass_filter(
    signal: np.ndarray, fs: float, low: float, high: float, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    low_n = low / nyq
    high_n = high / nyq
    # Clamp to valid range
    low_n = max(low_n, 0.001)
    high_n = min(high_n, 0.999)
    if low_n >= high_n:
        return signal.copy()
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)


def _compute_modulation_index(
    theta_phase: np.ndarray,
    gamma_amplitude: np.ndarray,
    n_bins: int = _N_PHASE_BINS,
) -> float:
    """Compute the Modulation Index (MI) via KL divergence.

    Measures how much gamma amplitude varies with theta phase.
    MI = 0 means uniform (no coupling), MI > 0 means phase-amplitude coupling.

    Based on Tort et al. (2010).
    """
    if len(theta_phase) < 10 or len(gamma_amplitude) < 10:
        return 0.0

    # If gamma amplitude is negligible, no meaningful coupling
    if np.max(gamma_amplitude) < 1e-8:
        return 0.0

    # Bin edges from -pi to pi
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)

    # Mean gamma amplitude in each phase bin
    mean_amp = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (theta_phase >= bin_edges[i]) & (theta_phase < bin_edges[i + 1])
        if np.any(mask):
            mean_amp[i] = np.mean(gamma_amplitude[mask])
        else:
            mean_amp[i] = 0.0

    # Normalize to a probability distribution
    total = np.sum(mean_amp)
    if total <= 0:
        return 0.0
    p = mean_amp / total

    # Uniform distribution
    q = np.ones(n_bins) / n_bins

    # KL divergence: D_KL(P || Q)
    # Avoid log(0) by adding epsilon
    eps = 1e-10
    p_safe = p + eps
    p_safe = p_safe / np.sum(p_safe)  # re-normalize after epsilon

    kl_div = np.sum(p_safe * np.log(p_safe / q))

    # Normalize by log(n_bins) so MI falls in [0, 1]
    mi = kl_div / np.log(n_bins)
    return float(np.clip(mi, 0.0, 1.0))


def _compute_band_power(
    signal: np.ndarray, fs: float, band: tuple
) -> float:
    """Compute mean power in a frequency band using Welch PSD."""
    if len(signal) < 32:
        return 0.0

    nperseg = min(len(signal), int(fs * 2))
    nperseg = max(nperseg, 16)

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0

    return float(np.mean(psd[mask]))


def _safe_normalize(value: float, baseline: float, scale: float = 1.0) -> float:
    """Normalize a value to 0-1 range using a baseline and scale factor."""
    if baseline <= 0:
        return float(np.clip(value * scale, 0.0, 1.0))
    normalized = value / baseline
    return float(np.clip(normalized * scale, 0.0, 1.0))


class EmotionalMemoryPredictor:
    """Predict emotional memory encoding strength from EEG signals.

    Uses theta-gamma cross-frequency coupling (CFC) as the primary biomarker.
    Higher theta-gamma coupling during emotional experiences indicates stronger
    memory encoding in the hippocampal-cortical network.

    Designed for 4-channel Muse 2 (TP9, AF7, AF8, TP10).
    Falls back to single-channel operation for 1D input.
    """

    def __init__(self, fs: float = 256.0):
        self._fs = fs
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record resting-state baseline theta-gamma coupling.

        Call during 2-3 min eyes-closed resting state to establish
        individual baseline for theta power and coupling metrics.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate. Uses instance default if None.
            user_id: User identifier for multi-user support.

        Returns:
            Dict with baseline_set (bool), n_channels (int), and
            baseline theta/gamma/coupling values.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=np.float64)

        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        n_channels = signals.shape[0]

        # Compute baseline theta power (frontal channels preferred)
        frontal_channels = self._get_frontal_channels(signals)
        theta_powers = []
        gamma_powers = []
        mi_values = []

        for ch_signal in frontal_channels:
            if len(ch_signal) < _MIN_SAMPLES:
                continue

            theta_powers.append(
                _compute_band_power(ch_signal, fs, _THETA_BAND)
            )
            gamma_powers.append(
                _compute_band_power(ch_signal, fs, _GAMMA_BAND)
            )

            # Compute baseline MI if enough data
            if len(ch_signal) >= int(_MIN_CFC_SECONDS * fs):
                theta_filt = _bandpass_filter(ch_signal, fs, *_THETA_BAND)
                gamma_filt = _bandpass_filter(ch_signal, fs, *_GAMMA_BAND)
                theta_phase = np.angle(hilbert(theta_filt))
                gamma_amp = np.abs(hilbert(gamma_filt))
                mi = _compute_modulation_index(theta_phase, gamma_amp)
                mi_values.append(mi)

        baseline = {
            "theta_power": float(np.mean(theta_powers)) if theta_powers else 0.0,
            "gamma_power": float(np.mean(gamma_powers)) if gamma_powers else 0.0,
            "coupling_mi": float(np.mean(mi_values)) if mi_values else 0.0,
        }

        self._baselines[user_id] = baseline

        return {
            "baseline_set": True,
            "n_channels": n_channels,
            "baseline_theta_power": round(baseline["theta_power"], 6),
            "baseline_gamma_power": round(baseline["gamma_power"], 6),
            "baseline_coupling_mi": round(baseline["coupling_mi"], 6),
        }

    def predict(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Predict emotional memory encoding strength from EEG signals.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate. Uses instance default if None.
            user_id: User identifier for multi-user support.

        Returns:
            Dict with encoding_strength (0-1), theta_power, gamma_power,
            theta_gamma_coupling (MI), frontal_theta_increase, encoding_level,
            and has_baseline flag.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=np.float64)

        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        n_samples = signals.shape[1]
        has_baseline = user_id in self._baselines

        # Handle very short signals
        if n_samples < _MIN_SAMPLES:
            result = self._default_result(has_baseline)
            self._append_history(user_id, result)
            return result

        # Handle NaN/inf
        if np.any(~np.isfinite(signals)):
            signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

        frontal_channels = self._get_frontal_channels(signals)

        # Compute theta and gamma power across frontal channels
        theta_powers = []
        gamma_powers = []
        mi_values = []

        for ch_signal in frontal_channels:
            theta_powers.append(
                _compute_band_power(ch_signal, fs, _THETA_BAND)
            )
            gamma_powers.append(
                _compute_band_power(ch_signal, fs, _GAMMA_BAND)
            )

            # CFC: theta-gamma modulation index
            if n_samples >= int(_MIN_CFC_SECONDS * fs):
                theta_filt = _bandpass_filter(ch_signal, fs, *_THETA_BAND)
                gamma_filt = _bandpass_filter(ch_signal, fs, *_GAMMA_BAND)
                theta_phase = np.angle(hilbert(theta_filt))
                gamma_amp = np.abs(hilbert(gamma_filt))
                mi = _compute_modulation_index(theta_phase, gamma_amp)
                mi_values.append(mi)

        avg_theta = float(np.mean(theta_powers)) if theta_powers else 0.0
        avg_gamma = float(np.mean(gamma_powers)) if gamma_powers else 0.0
        avg_mi = float(np.mean(mi_values)) if mi_values else 0.0

        # Compute frontal theta increase from baseline
        frontal_theta_increase = 0.0
        if has_baseline:
            bl_theta = self._baselines[user_id].get("theta_power", 0.0)
            if bl_theta > 0:
                frontal_theta_increase = (avg_theta - bl_theta) / bl_theta
            frontal_theta_increase = float(
                np.clip(frontal_theta_increase, -1.0, 5.0)
            )

        # Normalize components for composite score
        # Theta power: typical resting ~5-20 uV^2/Hz, encoding can double
        theta_norm = _safe_normalize(avg_theta, baseline=10.0, scale=0.5)

        # MI: resting ~0.01-0.05, encoding ~0.05-0.20
        mi_norm = _safe_normalize(avg_mi, baseline=0.1, scale=1.0)

        # Gamma power: relative measure, lower weight due to EMG contamination
        gamma_norm = _safe_normalize(avg_gamma, baseline=5.0, scale=0.5)

        # Composite encoding strength
        encoding_strength = float(np.clip(
            0.40 * theta_norm
            + 0.35 * mi_norm
            + 0.25 * gamma_norm,
            0.0, 1.0,
        ))

        # Encoding level classification
        if encoding_strength >= _STRONG_THRESHOLD:
            encoding_level = "strong_encoding"
        elif encoding_strength >= _MODERATE_THRESHOLD:
            encoding_level = "moderate_encoding"
        elif encoding_strength >= _WEAK_THRESHOLD:
            encoding_level = "weak_encoding"
        else:
            encoding_level = "minimal_encoding"

        result = {
            "encoding_strength": round(encoding_strength, 4),
            "theta_power": round(avg_theta, 6),
            "gamma_power": round(avg_gamma, 6),
            "theta_gamma_coupling": round(avg_mi, 6),
            "frontal_theta_increase": round(frontal_theta_increase, 4),
            "encoding_level": encoding_level,
            "has_baseline": has_baseline,
        }

        self._append_history(user_id, result)
        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get summary statistics for a user's session.

        Returns:
            Dict with n_epochs, mean_encoding, peak_encoding.
        """
        history = self._history.get(user_id, [])
        if not history:
            return {
                "n_epochs": 0,
                "mean_encoding": 0.0,
                "peak_encoding": 0.0,
            }

        strengths = [h["encoding_strength"] for h in history]
        return {
            "n_epochs": len(history),
            "mean_encoding": round(float(np.mean(strengths)), 4),
            "peak_encoding": round(float(np.max(strengths)), 4),
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Get prediction history for a user.

        Args:
            user_id: User identifier.
            last_n: If provided, return only the last N entries.

        Returns:
            List of prediction result dicts.
        """
        history = self._history.get(user_id, [])
        if last_n is not None and last_n > 0:
            return history[-last_n:]
        return list(history)

    def reset(self, user_id: str = "default") -> None:
        """Clear all data for a user (baseline, history)."""
        self._baselines.pop(user_id, None)
        self._history.pop(user_id, None)

    # -- Private helpers ------------------------------------------------------

    def _get_frontal_channels(self, signals: np.ndarray) -> List[np.ndarray]:
        """Extract frontal channels (AF7, AF8) from multichannel array.

        Falls back to all available channels if fewer than 3 channels.
        """
        n_ch = signals.shape[0]
        if n_ch >= 3:
            # AF7 = ch1, AF8 = ch2
            return [signals[1], signals[2]]
        else:
            return [signals[i] for i in range(n_ch)]

    def _default_result(self, has_baseline: bool) -> Dict:
        """Return default (zero) prediction result for edge cases."""
        return {
            "encoding_strength": 0.0,
            "theta_power": 0.0,
            "gamma_power": 0.0,
            "theta_gamma_coupling": 0.0,
            "frontal_theta_increase": 0.0,
            "encoding_level": "minimal_encoding",
            "has_baseline": has_baseline,
        }

    def _append_history(self, user_id: str, result: Dict) -> None:
        """Append result to history, capping at _MAX_HISTORY entries."""
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > _MAX_HISTORY:
            self._history[user_id] = self._history[user_id][-_MAX_HISTORY:]
