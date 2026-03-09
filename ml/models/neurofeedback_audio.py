"""Neurofeedback audio parameter generator from real-time EEG state.

Maps brain-state metrics (band powers extracted from Muse 2 EEG) to audio
feedback parameters (frequency, volume, pan, binaural beat frequency) that
a downstream audio engine can render.

Supports multiple training protocols:
  - alpha_uptraining: increase alpha (8-12 Hz) for relaxation
  - smr_training: increase SMR / low-beta (12-15 Hz) for calm focus
  - theta_training: increase theta (4-8 Hz) for meditation / creativity

References:
  Gruzelier (2014) — EEG-neurofeedback for optimising performance
  Egner & Gruzelier (2001) — Learned self-regulation of EEG frequency components

This module generates PARAMETERS only — no audio samples.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import welch

# numpy 2.0 renamed np.trapz → np.trapezoid; support both
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol definitions
# ---------------------------------------------------------------------------

# Each protocol specifies:
#   target_band:     (low_hz, high_hz) for the band to train
#   reward_threshold: baseline_ratio above which a reward tone fires
#   binaural_center: center frequency of the binaural beat (Hz)
#   carrier_hz:      audible carrier tone base frequency
#   description:     human-readable label
PROTOCOLS: Dict[str, dict] = {
    "alpha_uptraining": {
        "target_band": (8.0, 12.0),
        "band_label": "alpha",
        "reward_threshold": 1.2,
        "binaural_center": 10.0,
        "carrier_hz": 220.0,
        "description": "Increase alpha power for relaxation and calm awareness",
    },
    "smr_training": {
        "target_band": (12.0, 15.0),
        "band_label": "smr",
        "reward_threshold": 1.2,
        "binaural_center": 13.5,
        "carrier_hz": 250.0,
        "description": "Increase sensorimotor rhythm for calm, focused attention",
    },
    "theta_training": {
        "target_band": (4.0, 8.0),
        "band_label": "theta",
        "reward_threshold": 1.2,
        "binaural_center": 6.0,
        "carrier_hz": 200.0,
        "description": "Increase theta power for meditation and creativity",
    },
}


def _band_power(signal: np.ndarray, fs: float,
                band: tuple[float, float]) -> float:
    """Compute relative power in *band* using Welch PSD.

    Works on a 1-D signal. Returns a float in [0, 1] representing the
    fraction of total power that falls within the band.
    """
    nperseg = min(len(signal), int(fs * 2))
    if nperseg < 4:
        return 0.0
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    total = _trapezoid(psd, freqs)
    if total <= 0:
        return 0.0
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(_trapezoid(psd[mask], freqs[mask]) / total)


def _laterality_index(eeg: np.ndarray, fs: float,
                      band: tuple[float, float]) -> float:
    """Compute laterality between left-frontal (AF7=ch1) and right-frontal (AF8=ch2).

    Returns value in [-1, 1].  Positive = more right-hemisphere power.
    """
    if eeg.ndim < 2 or eeg.shape[0] < 3:
        return 0.0
    left = _band_power(eeg[1], fs, band)
    right = _band_power(eeg[2], fs, band)
    denom = left + right
    if denom < 1e-12:
        return 0.0
    return float(np.clip((right - left) / denom, -1.0, 1.0))


class NeurofeedbackAudioEngine:
    """Generates audio feedback parameters from real-time EEG.

    Usage::

        engine = NeurofeedbackAudioEngine()
        engine.set_protocol("alpha_uptraining")
        engine.set_baseline(resting_eeg, fs=256)

        # In the real-time loop:
        params = engine.get_audio_params(live_eeg, fs=256)
        # params is a dict with base_frequency, volume, pan, reward_tone, etc.
    """

    def __init__(self) -> None:
        self.protocol: str = "alpha_uptraining"
        self.baseline: Optional[float] = None
        self._reward_count: int = 0
        self._total_epochs: int = 0
        self._history: List[dict] = []
        self._band_power_accum: List[float] = []

    # ------------------------------------------------------------------
    # Protocol management
    # ------------------------------------------------------------------

    def get_available_protocols(self) -> List[dict]:
        """Return list of protocol metadata dicts."""
        result = []
        for name, cfg in PROTOCOLS.items():
            result.append({
                "name": name,
                "target_band": cfg["band_label"],
                "description": cfg["description"],
                "frequency_range": cfg["target_band"],
                "binaural_center_hz": cfg["binaural_center"],
            })
        return result

    def set_protocol(self, protocol_name: str) -> None:
        """Select the active training protocol.

        Raises ``ValueError`` if *protocol_name* is not recognised.
        """
        if protocol_name not in PROTOCOLS:
            raise ValueError(
                f"Unknown protocol '{protocol_name}'. "
                f"Available: {list(PROTOCOLS.keys())}"
            )
        self.protocol = protocol_name

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def set_baseline(self, eeg: np.ndarray, fs: float) -> None:
        """Record resting-state baseline power for the active protocol's target band.

        Args:
            eeg: (n_channels, n_samples) or (n_samples,) array of resting EEG.
            fs:  Sampling frequency in Hz.
        """
        cfg = PROTOCOLS[self.protocol]
        signal = eeg[0] if eeg.ndim == 2 else eeg
        bp = _band_power(signal, fs, cfg["target_band"])
        self.baseline = max(bp, 1e-10)  # avoid division by zero
        log.info("Baseline set for %s: %.6f", self.protocol, self.baseline)

    # ------------------------------------------------------------------
    # Core: compute audio parameters
    # ------------------------------------------------------------------

    def get_audio_params(self, eeg: np.ndarray, fs: float) -> dict:
        """Compute audio feedback parameters from a live EEG epoch.

        Args:
            eeg: (n_channels, n_samples) or (n_samples,) multichannel EEG.
            fs:  Sampling frequency in Hz.

        Returns:
            Dict with keys: base_frequency, volume, pan, reward_tone,
            reward_count, binaural_beat_freq, current_band_power,
            baseline_ratio, protocol, feedback_message.
        """
        cfg = PROTOCOLS[self.protocol]
        band = cfg["target_band"]

        # --- extract target band power from first channel ---------------
        signal = eeg[0] if eeg.ndim == 2 else eeg
        current_bp = _band_power(signal, fs, band)

        # --- baseline ratio ---------------------------------------------
        if self.baseline is not None and self.baseline > 1e-12:
            ratio = current_bp / self.baseline
        else:
            ratio = 1.0

        # --- reward decision --------------------------------------------
        threshold = cfg["reward_threshold"]
        is_reward = ratio >= threshold
        if is_reward:
            self._reward_count += 1

        # --- audio parameter mapping ------------------------------------
        # Volume: proportional to how far above baseline the user is,
        # clamped to [0.05, 1.0].  Below baseline -> quiet.
        volume = float(np.clip(0.1 + 0.4 * (ratio - 0.5), 0.05, 1.0))

        # Base frequency: maps ratio to an audible pitch.
        # ratio ~1 -> carrier_hz, higher ratio -> higher pitch (reward).
        base_freq = float(np.clip(
            cfg["carrier_hz"] * (0.8 + 0.4 * min(ratio, 2.5)),
            100.0, 1000.0,
        ))

        # Pan: laterality of target band (positive = right hemisphere)
        pan = _laterality_index(eeg, fs, band)

        # Binaural beat: center frequency for the target band
        binaural_freq = cfg["binaural_center"]

        # Feedback message
        if is_reward:
            message = f"Great! {cfg['band_label'].upper()} power is {ratio:.1f}x baseline"
        elif ratio >= 1.0:
            message = f"{cfg['band_label'].upper()} at baseline level"
        else:
            message = f"{cfg['band_label'].upper()} below baseline ({ratio:.1f}x)"

        # --- bookkeeping ------------------------------------------------
        self._total_epochs += 1
        self._band_power_accum.append(current_bp)

        result = {
            "base_frequency": round(base_freq, 2),
            "volume": round(volume, 4),
            "pan": round(pan, 4),
            "reward_tone": is_reward,
            "reward_count": self._reward_count,
            "binaural_beat_freq": float(binaural_freq),
            "current_band_power": round(current_bp, 6),
            "baseline_ratio": round(ratio, 4),
            "protocol": self.protocol,
            "feedback_message": message,
        }
        self._history.append(result)
        return result

    # ------------------------------------------------------------------
    # Session stats & history
    # ------------------------------------------------------------------

    def get_session_stats(self) -> dict:
        """Return aggregate session statistics."""
        total = self._total_epochs
        reward_rate = (self._reward_count / total) if total > 0 else 0.0
        mean_bp = (
            float(np.mean(self._band_power_accum))
            if self._band_power_accum
            else 0.0
        )
        return {
            "protocol": self.protocol,
            "total_epochs": total,
            "reward_count": self._reward_count,
            "reward_rate": round(reward_rate, 4),
            "mean_band_power": round(mean_bp, 6),
        }

    def get_history(self, last_n: Optional[int] = None) -> List[dict]:
        """Return epoch-by-epoch history, optionally limited to *last_n*."""
        if last_n is not None:
            return list(self._history[-last_n:])
        return list(self._history)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear baseline, history, and counters.  Protocol is preserved."""
        self.baseline = None
        self._reward_count = 0
        self._total_epochs = 0
        self._history = []
        self._band_power_accum = []
        log.info("NeurofeedbackAudioEngine reset (protocol=%s)", self.protocol)
