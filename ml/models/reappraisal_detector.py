"""Emotion regulation vs genuine experience detector via LPP + frontal theta.

Detects whether an emotional response is genuine or involves cognitive regulation
(reappraisal/suppression) using ERP-inspired features.

Key biomarkers:
- Late Positive Potential (LPP): ERP component 400-800ms post-stimulus
  Larger amplitude → more genuine emotional response
  Smaller amplitude → suppression/regulation active
- Frontal theta (4-8 Hz): increases with cognitive control effort
- Frontal-posterior asymmetry of LPP: indicates regulation direction

Note: Without event markers (onset times), approximates LPP as the mean
amplitude in a sustained window, and tracks frontal theta power dynamics.

Scientific references:
- Hajcak & Nieuwenhuis (2006): LPP amplitude reduction during reappraisal
- Moser et al. (2006): Frontal theta increase during cognitive regulation
- Gross & John (2003): Cognitive model of emotion regulation (reappraisal vs suppression)
- Foti & Hajcak (2008): LPP as index of sustained emotional processing
"""

import threading
from typing import Dict, Optional

import numpy as np
from scipy.signal import butter, filtfilt, welch


# Regulation state labels
REGULATION_STATES = ["genuine", "mild_regulation", "active_reappraisal", "suppression"]

# Muse 2 channel indices (BrainFlow order: TP9, AF7, AF8, TP10)
_CH_TP9 = 0   # posterior left
_CH_AF7 = 1   # frontal left
_CH_AF8 = 2   # frontal right
_CH_TP10 = 3  # posterior right

# State thresholds for regulation_index
_THRESH_GENUINE = 0.25
_THRESH_MILD = 0.45
_THRESH_REAPPRAISAL = 0.65

# Per-user singleton cache
_detectors: Dict[str, "ReappraisalDetector"] = {}
_detectors_lock = threading.Lock()


def _bandpass(signal: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    # Clamp to avoid scipy errors on very short signals
    low_n = max(low_n, 1e-6)
    high_n = min(high_n, 1.0 - 1e-6)
    if low_n >= high_n:
        return signal
    b, a = butter(order, [low_n, high_n], btype="band")
    if len(signal) < max(len(a), len(b)) * 3:
        # Signal too short for filtfilt — return raw
        return signal
    return filtfilt(b, a, signal)


def _band_power(signal: np.ndarray, fs: float, low: float, high: float) -> float:
    """Compute mean power spectral density in [low, high] Hz via Welch's method."""
    n = len(signal)
    if n < 16:
        return float(np.mean(signal ** 2))
    nperseg = min(256, n // 2)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return float(np.mean(psd))
    return float(np.mean(psd[mask]))


class ReappraisalDetector:
    """EEG-based detector for emotion regulation vs genuine emotional experience.

    Uses LPP proxy (late vs early amplitude difference at posterior channels)
    combined with frontal theta power (AF7/AF8) to infer whether an emotional
    response is authentic or being actively regulated.

    Muse 2 channel map:
        ch0 = TP9  (posterior left)  — LPP approximation
        ch1 = AF7  (frontal left)    — frontal theta
        ch2 = AF8  (frontal right)   — frontal theta
        ch3 = TP10 (posterior right) — LPP approximation
    """

    def __init__(self, n_channels: int = 4, fs: float = 256.0):
        self.n_channels = n_channels
        self.fs = fs
        self.model_type = "feature-based"

        # Baseline normalization state (updated via update_baseline)
        self._baseline_lpp: Optional[float] = None
        self._baseline_theta: Optional[float] = None
        self._baseline_lock = threading.Lock()

    # ── LPP proxy ─────────────────────────────────────────────────────────────

    def _compute_lpp_proxy(self, eeg: np.ndarray, fs: float) -> float:
        """Compute LPP proxy as late vs early amplitude difference.

        Without event markers, divides the signal into early (first 1/3) and
        late (last 1/3) windows. The difference (late - early) estimates the
        Late Positive Potential:
          - Positive → late positivity (genuine emotional response)
          - Negative → late negativity or active regulatory suppression

        For 4-channel input, uses TP9 (ch0) and TP10 (ch3) as posterior
        approximations for LPP, since LPP is maximal at centro-parietal sites.
        Single-channel input uses the provided signal directly.

        Returns:
            float in [-1, 1]: positive = genuine, negative = regulated.
        """
        # Select posterior channels for LPP approximation
        if eeg.ndim == 2 and eeg.shape[0] >= 4:
            # Average TP9 and TP10 (posterior channels)
            signal = (eeg[_CH_TP9] + eeg[_CH_TP10]) / 2.0
        elif eeg.ndim == 2:
            signal = eeg[0]
        else:
            signal = eeg

        n = len(signal)
        if n < 6:
            return 0.0

        third = n // 3
        early_window = signal[:third]
        late_window = signal[2 * third:]

        early_mean = float(np.mean(early_window))
        late_mean = float(np.mean(late_window))

        # Raw LPP proxy
        lpp_raw = late_mean - early_mean

        # Normalize by signal amplitude range to get [-1, 1]
        amplitude_range = float(np.std(signal))
        if amplitude_range < 1e-8:
            return 0.0

        lpp_normalized = lpp_raw / (amplitude_range * 2.0)
        return float(np.clip(lpp_normalized, -1.0, 1.0))

    # ── Frontal theta ──────────────────────────────────────────────────────────

    def _compute_frontal_theta(self, eeg: np.ndarray, fs: float) -> float:
        """Compute frontal theta power (4-8 Hz) from AF7+AF8 channels.

        Frontal theta (fCz/Fz) increases with cognitive control effort,
        working memory load, and emotional regulation (Moser et al., 2006).

        For 4-channel input, averages AF7 (ch1) and AF8 (ch2).
        For single-channel, uses the provided channel.

        Returns:
            float >= 0: mean theta PSD at frontal sites.
        """
        if eeg.ndim == 2 and eeg.shape[0] >= 3:
            # Average left and right frontal channels
            frontal = (eeg[_CH_AF7] + eeg[_CH_AF8]) / 2.0
        elif eeg.ndim == 2:
            frontal = eeg[0]
        else:
            frontal = eeg

        return _band_power(frontal, fs, low=4.0, high=8.0)

    # ── Regulation index ───────────────────────────────────────────────────────

    def _compute_regulation_index(self, lpp_amplitude: float, frontal_theta: float) -> float:
        """Combine LPP proxy and frontal theta into a 0-1 regulation index.

        Formula:
            reg_idx = 0.5 * (1 - clip(lpp_amplitude, 0, 1))
                    + 0.5 * clip(frontal_theta / (frontal_theta + 0.1), 0, 1)

        Interpretation:
            - High frontal theta + low (or negative) LPP → high regulation (→ 1.0)
            - Low frontal theta + high positive LPP      → genuine response (→ 0.0)

        Args:
            lpp_amplitude: LPP proxy in [-1, 1]
            frontal_theta: frontal theta power (non-negative)

        Returns:
            float in [0, 1]
        """
        lpp_term = 0.5 * (1.0 - float(np.clip(lpp_amplitude, 0.0, 1.0)))
        theta_term = 0.5 * float(np.clip(frontal_theta / (frontal_theta + 0.1), 0.0, 1.0))
        return float(np.clip(lpp_term + theta_term, 0.0, 1.0))

    # ── Main predict ───────────────────────────────────────────────────────────

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Detect emotion regulation vs genuine experience.

        Args:
            eeg: EEG data as (4, n_samples) multichannel or (n_samples,) single-channel.
                 Minimum 2 seconds recommended (512 samples at 256 Hz) for stable LPP.
            fs:  Sampling frequency in Hz.

        Returns:
            dict with keys:
                regulation_state    : one of REGULATION_STATES
                regulation_index    : float 0-1 (0=genuine, 1=suppression)
                lpp_amplitude       : LPP proxy [-1, 1]
                frontal_theta_power : float >= 0
                genuine_probability : float 0-1
                regulation_probability : float 0-1
                cognitive_control_level : "low" | "medium" | "high"
                model_type          : "feature-based"
        """
        # Accept (n_samples,) single-channel or (n_ch, n_samples) multichannel
        if not isinstance(eeg, np.ndarray):
            eeg = np.array(eeg, dtype=float)

        if eeg.ndim == 1:
            eeg_arr = eeg
        elif eeg.ndim == 2:
            eeg_arr = eeg
        else:
            eeg_arr = eeg.reshape(-1)

        # Compute features
        lpp = self._compute_lpp_proxy(eeg_arr, fs)
        theta = self._compute_frontal_theta(eeg_arr, fs)

        # Apply baseline normalization if available
        with self._baseline_lock:
            base_lpp = self._baseline_lpp
            base_theta = self._baseline_theta

        if base_lpp is not None:
            # Shift LPP relative to resting baseline
            lpp = float(np.clip(lpp - base_lpp, -1.0, 1.0))
        if base_theta is not None and base_theta > 1e-8:
            # Normalize theta to multiples of resting baseline
            theta = float(np.clip(theta / base_theta, 0.0, 5.0)) * 0.1
        else:
            # Without baseline, scale raw theta to roughly [0, 1]
            # Typical frontal theta PSD is O(1e-10 to 1e-8); normalize heuristically
            theta = float(np.clip(theta * 1e8, 0.0, 5.0)) / 5.0

        reg_index = self._compute_regulation_index(lpp, theta)

        # Map to regulation state
        if reg_index < _THRESH_GENUINE:
            state = "genuine"
        elif reg_index < _THRESH_MILD:
            state = "mild_regulation"
        elif reg_index < _THRESH_REAPPRAISAL:
            state = "active_reappraisal"
        else:
            state = "suppression"

        # Genuine vs regulation probabilities (soft sigmoid-like split)
        genuine_prob = float(np.clip(1.0 - reg_index, 0.0, 1.0))
        regulation_prob = float(np.clip(reg_index, 0.0, 1.0))

        # Cognitive control level
        if reg_index < 0.33:
            control_level = "low"
        elif reg_index < 0.66:
            control_level = "medium"
        else:
            control_level = "high"

        # Recover the pre-normalization theta for reporting
        raw_theta = self._compute_frontal_theta(
            eeg_arr if eeg_arr.ndim == 2 else eeg_arr, fs
        )

        return {
            "regulation_state": state,
            "regulation_index": round(reg_index, 4),
            "lpp_amplitude": round(float(self._compute_lpp_proxy(eeg_arr, fs)), 4),
            "frontal_theta_power": round(float(raw_theta), 6),
            "genuine_probability": round(genuine_prob, 4),
            "regulation_probability": round(regulation_prob, 4),
            "cognitive_control_level": control_level,
            "model_type": self.model_type,
        }

    # ── Baseline update ────────────────────────────────────────────────────────

    def update_baseline(self, eeg: np.ndarray, fs: float) -> None:
        """Record resting-state baseline for LPP and theta normalization.

        Call with 2-3 minutes of eyes-closed resting EEG.  After this call,
        predict() normalizes features relative to the resting baseline, which
        substantially reduces inter-individual variability.

        Args:
            eeg: Resting-state EEG (same shape convention as predict()).
            fs:  Sampling frequency in Hz.
        """
        if not isinstance(eeg, np.ndarray):
            eeg = np.array(eeg, dtype=float)

        baseline_lpp = self._compute_lpp_proxy(eeg, fs)
        baseline_theta = self._compute_frontal_theta(eeg, fs)

        with self._baseline_lock:
            self._baseline_lpp = baseline_lpp
            self._baseline_theta = baseline_theta


# ── Singleton factory ──────────────────────────────────────────────────────────

def get_reappraisal_detector(user_id: str = "default") -> ReappraisalDetector:
    """Return a per-user ReappraisalDetector singleton (thread-safe)."""
    with _detectors_lock:
        if user_id not in _detectors:
            _detectors[user_id] = ReappraisalDetector()
        return _detectors[user_id]
