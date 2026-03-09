"""Closed-loop neurostimulation guidance using EEG-driven feedback.

Computes tACS/tDCS/TMS target parameters from real-time EEG state.
Targets: alpha entrainment, theta suppression, beta up-regulation.

References:
    Zaehle et al. (2010) — tACS alpha entrainment
    Bikson et al. (2016) — tDCS montage optimization
    Thut & Miniussi (2009) — TMS-EEG closed-loop
"""
from __future__ import annotations
import numpy as np
from typing import Dict

PROTOCOLS = ["alpha_entrainment", "theta_suppression", "beta_upregulation", "delta_suppression"]

class NeurostimGuidanceModel:
    def predict(self, signals: np.ndarray, fs: float = 256.0,
                target_protocol: str = "alpha_entrainment") -> Dict:
        if signals.ndim == 1:
            signals = signals[np.newaxis, :]
        n_ch, n_samples = signals.shape
        from scipy.signal import welch
        nperseg = min(n_samples, int(fs * 2))
        ch = signals[0]
        f, psd = welch(ch, fs=fs, nperseg=nperseg)
        def bp(lo, hi):
            idx = (f >= lo) & (f <= hi)
            return float(np.mean(psd[idx])) if idx.any() else 1e-9
        delta = bp(0.5, 4); theta = bp(4, 8); alpha = bp(8, 12); beta = bp(12, 30)
        # Individual alpha frequency for entrainment target
        af = f[(f >= 8) & (f <= 12)]; ap = psd[(f >= 8) & (f <= 12)]
        iaf = float(af[np.argmax(ap)]) if len(af) > 0 else 10.0
        # Protocol-specific intensity (0–1, normalized stim amplitude suggestion)
        if target_protocol == "alpha_entrainment":
            deficit = float(np.clip(1.0 - alpha / (alpha + beta + 1e-9), 0, 1))
            stim_freq = round(iaf, 1)
            intensity = float(np.clip(deficit * 1.5, 0, 1))
        elif target_protocol == "theta_suppression":
            excess = float(np.clip(theta / (alpha + 1e-9) - 0.5, 0, 1))
            stim_freq = 40.0  # gamma tACS suppresses theta
            intensity = float(np.clip(excess, 0, 1))
        elif target_protocol == "beta_upregulation":
            deficit = float(np.clip(1.0 - beta / (alpha + beta + 1e-9) - 0.3, 0, 1))
            stim_freq = 20.0
            intensity = float(np.clip(deficit * 1.2, 0, 1))
        else:  # delta_suppression
            excess = float(np.clip(delta / (delta + alpha + 1e-9) - 0.2, 0, 1))
            stim_freq = 1.0
            intensity = float(np.clip(excess, 0, 1))
        readiness = float(np.clip(1.0 - intensity * 0.5, 0.2, 1.0))
        return {
            "protocol": target_protocol,
            "stim_frequency_hz": stim_freq,
            "suggested_intensity_normalized": round(intensity, 4),
            "iaf_hz": round(iaf, 2),
            "readiness_score": round(readiness, 4),
            "should_stimulate": intensity > 0.3,
            "model_used": "feature_based_closed_loop",
        }

_model = NeurostimGuidanceModel()
def get_model(): return _model
