"""GRU-based Sleep Staging with Permutation Channel Selection.

Classifies EEG windows into 5 sleep stages:
  Wake, N1, N2, N3, REM

Architecture:
  - No PyTorch/heavy dependencies — uses numpy EMA as the recurrent state
  - Permutation channel selection: rank channels by delta-power variance, use top-2
  - Rolling buffer of last 10 band-power vectors for temporal context
  - EMA acts as a lightweight GRU-style hidden state across frames

Stage logic (band-power dominance):
  N3   → delta dominates (delta / total > 0.45)
  REM  → theta dominant   (theta / total > 0.35, low delta)
  N1   → alpha dominant   (alpha / total > 0.30)
  Wake → beta dominant    (beta / total > 0.30)
  N2   → balanced         (none of the above)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

SLEEP_STAGES = ["Wake", "N1", "N2", "N3", "REM"]

# EMA smoothing factor (α): higher = faster response, lower = more smoothing
_EMA_ALPHA = 0.35

# Buffer depth for temporal context
_BUFFER_SIZE = 10


# ---------------------------------------------------------------------------
# Band-power helper
# ---------------------------------------------------------------------------

def _bandpower(signal: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """Estimate band power via Welch PSD.

    Falls back to a simple squared-magnitude approach when the signal is too
    short for Welch.
    """
    n = len(signal)
    if n < 4:
        return float(np.mean(signal ** 2)) + 1e-12

    try:
        from scipy.signal import welch  # type: ignore
        nperseg = min(256, n)
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            return 1e-12
        # np.trapezoid added in numpy 2.0; fall back to np.trapz for older installs
        _trapz = getattr(np, "trapezoid", np.trapz)
        return float(_trapz(psd[mask], freqs[mask])) + 1e-12
    except Exception:
        # Pure-numpy fallback: DFT magnitude
        fft_mag = np.abs(np.fft.rfft(signal)) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            return 1e-12
        return float(np.mean(fft_mag[mask])) + 1e-12


def _extract_bands(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """Return delta/theta/alpha/beta band powers for a 1-D signal."""
    return {
        "delta": _bandpower(signal, fs, 0.5, 4.0),
        "theta": _bandpower(signal, fs, 4.0, 8.0),
        "alpha": _bandpower(signal, fs, 8.0, 12.0),
        "beta":  _bandpower(signal, fs, 12.0, 30.0),
    }


# ---------------------------------------------------------------------------
# Permutation channel selection
# ---------------------------------------------------------------------------

def _rank_channels_by_delta_variance(
    eeg: np.ndarray,
    fs: float,
    n_top: int = 2,
) -> List[int]:
    """Rank channels by delta-power variance across short sub-windows.

    Channels with higher delta-power variance carry more sleep-related
    information and are prioritised.

    Args:
        eeg: Shape (n_channels, n_samples)
        fs:  Sampling frequency
        n_top: Number of top channels to return

    Returns:
        List of channel indices sorted by descending delta-power variance,
        up to n_top entries.
    """
    n_channels = eeg.shape[0]
    n_samples = eeg.shape[1]

    # Sub-window length: 1 second
    win_len = max(4, int(fs))
    variances: List[float] = []

    for ch in range(n_channels):
        ch_signal = eeg[ch]
        # Compute delta power for each non-overlapping sub-window
        delta_powers = []
        for start in range(0, n_samples - win_len + 1, win_len):
            seg = ch_signal[start : start + win_len]
            delta_powers.append(_bandpower(seg, fs, 0.5, 4.0))

        if len(delta_powers) >= 2:
            variances.append(float(np.var(delta_powers)))
        else:
            # Not enough windows — use overall delta power as proxy
            variances.append(_bandpower(ch_signal, fs, 0.5, 4.0))

    # Sort descending by variance
    ranked = sorted(range(n_channels), key=lambda i: variances[i], reverse=True)
    return ranked[:n_top]


# ---------------------------------------------------------------------------
# Stage classification from aggregated band powers
# ---------------------------------------------------------------------------

def _classify_stage(bands: Dict[str, float]) -> tuple[str, Dict[str, float]]:
    """Map band powers to a sleep stage and return raw scores.

    Returns:
        (stage_name, raw_scores_dict)
    """
    d = bands["delta"]
    t = bands["theta"]
    a = bands["alpha"]
    b = bands["beta"]
    total = d + t + a + b + 1e-12

    d_frac = d / total
    t_frac = t / total
    a_frac = a / total
    b_frac = b / total

    # Raw scores per stage (higher = more likely)
    scores = {
        "N3":   d_frac * 1.8,                    # delta dominance
        "REM":  t_frac * 1.5 * (1 - d_frac),     # theta dominant, low delta
        "N1":   a_frac * 1.4,                     # alpha dominant
        "Wake": b_frac * 1.6,                     # beta dominant
        "N2":   1.0 - abs(d_frac - 0.25) - abs(t_frac - 0.20),  # balanced
    }

    stage = max(scores, key=lambda k: scores[k])
    return stage, scores


def _scores_to_probs(scores: Dict[str, float]) -> Dict[str, float]:
    """Softmax over raw scores → probabilities summing to 1."""
    keys = list(scores.keys())
    vals = np.array([scores[k] for k in keys], dtype=np.float64)
    # Shift for numerical stability
    vals -= vals.max()
    exp_vals = np.exp(vals)
    total = exp_vals.sum()
    probs = exp_vals / total
    return {k: float(p) for k, p in zip(keys, probs)}


# ---------------------------------------------------------------------------
# GRUSleepStager
# ---------------------------------------------------------------------------

class GRUSleepStager:
    """GRU-inspired sleep stage classifier using EMA temporal context.

    The "recurrent" state is a per-band EMA that is updated each time
    ``predict()`` is called with new EEG data. The buffer stores the last
    ``_BUFFER_SIZE`` band-power vectors.
    """

    def __init__(self) -> None:
        self._buffer: deque = deque(maxlen=_BUFFER_SIZE)
        # EMA hidden state: {delta, theta, alpha, beta}
        self._ema_state: Optional[Dict[str, float]] = None
        self._prediction_count: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Predict sleep stage from an EEG window.

        Args:
            eeg: Shape (n_channels, n_samples) or (n_samples,)
            fs:  Sampling frequency in Hz

        Returns:
            {
              "stage": str,
              "probabilities": {Wake/N1/N2/N3/REM: float},
              "confidence": float,
              "channel_ranking": list[int],
            }
        """
        eeg = np.asarray(eeg, dtype=np.float64)

        # Normalise shape
        if eeg.ndim == 1:
            eeg_2d = eeg[np.newaxis, :]  # (1, n_samples)
            channel_ranking: List[int] = [0]
        elif eeg.ndim == 2:
            eeg_2d = eeg
            channel_ranking = _rank_channels_by_delta_variance(eeg_2d, fs, n_top=2)
        else:
            raise ValueError(f"eeg must be 1-D or 2-D, got shape {eeg.shape}")

        # Select top channels
        selected = eeg_2d[channel_ranking, :]  # (n_top, n_samples)

        # Guard against very short signals
        if selected.shape[1] < 4:
            return self._short_signal_response(channel_ranking)

        # Average band powers across selected channels
        avg_bands: Dict[str, float] = {"delta": 0.0, "theta": 0.0, "alpha": 0.0, "beta": 0.0}
        for ch_sig in selected:
            ch_bands = _extract_bands(ch_sig, fs)
            for k in avg_bands:
                avg_bands[k] += ch_bands[k]
        n_selected = len(channel_ranking)
        for k in avg_bands:
            avg_bands[k] /= n_selected

        # Update EMA hidden state (GRU-like recurrence)
        avg_bands = self._update_ema(avg_bands)

        # Push to rolling buffer
        self._buffer.append(avg_bands.copy())
        self._prediction_count += 1

        # Classify using temporally-smoothed bands
        temporal_bands = self._temporal_aggregate()
        stage, raw_scores = _classify_stage(temporal_bands)
        probs = _scores_to_probs(raw_scores)

        # Ensure all 5 stages present
        for s in SLEEP_STAGES:
            probs.setdefault(s, 0.0)

        # Normalise probs to sum exactly to 1.0
        total_p = sum(probs.values())
        if total_p > 0:
            probs = {k: v / total_p for k, v in probs.items()}

        confidence = float(probs[stage])

        return {
            "stage": stage,
            "probabilities": probs,
            "confidence": confidence,
            "channel_ranking": channel_ranking,
        }

    def reset(self) -> None:
        """Clear temporal buffer and EMA state."""
        self._buffer.clear()
        self._ema_state = None
        self._prediction_count = 0

    def get_status(self) -> Dict:
        """Return stager status and buffer info."""
        return {
            "model": "gru_sleep_stager",
            "buffer_size": len(self._buffer),
            "buffer_capacity": _BUFFER_SIZE,
            "prediction_count": self._prediction_count,
            "ema_initialised": self._ema_state is not None,
            "stages": SLEEP_STAGES,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_ema(self, bands: Dict[str, float]) -> Dict[str, float]:
        """Apply EMA update to the hidden state and return updated bands."""
        if self._ema_state is None:
            self._ema_state = bands.copy()
            return bands.copy()

        updated: Dict[str, float] = {}
        for k in bands:
            self._ema_state[k] = (
                _EMA_ALPHA * bands[k] + (1.0 - _EMA_ALPHA) * self._ema_state[k]
            )
            updated[k] = self._ema_state[k]
        return updated

    def _temporal_aggregate(self) -> Dict[str, float]:
        """Average band powers across the rolling buffer."""
        if not self._buffer:
            return {"delta": 1.0, "theta": 0.5, "alpha": 0.5, "beta": 0.5}

        agg: Dict[str, float] = {"delta": 0.0, "theta": 0.0, "alpha": 0.0, "beta": 0.0}
        for frame in self._buffer:
            for k in agg:
                agg[k] += frame[k]
        n = len(self._buffer)
        return {k: v / n for k, v in agg.items()}

    def _short_signal_response(self, channel_ranking: List[int]) -> Dict:
        """Safe fallback when the signal is too short to process."""
        uniform = {s: 1.0 / len(SLEEP_STAGES) for s in SLEEP_STAGES}
        return {
            "stage": "Wake",
            "probabilities": uniform,
            "confidence": 1.0 / len(SLEEP_STAGES),
            "channel_ranking": channel_ranking,
        }


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_instance: Optional[GRUSleepStager] = None


def get_gru_sleep_stager() -> GRUSleepStager:
    """Return the module-level singleton GRUSleepStager instance."""
    global _instance
    if _instance is None:
        _instance = GRUSleepStager()
    return _instance
