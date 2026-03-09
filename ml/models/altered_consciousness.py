"""Altered consciousness state detector for psychedelic/meditation/hypnosis states.

Altered states show: increased theta/alpha power, reduced beta (default mode
network suppression), and gamma bursts (ego dissolution correlate).

References:
    Carhart-Harris et al. (2016) — psilocybin EEG signatures
    Lutz et al. (2004) — advanced meditation gamma
    Kihlstrom (1985) — hypnosis theta enhancement
"""
from __future__ import annotations
import numpy as np
from typing import Dict

STATES = ["normal", "light_altered", "moderate_altered", "deep_altered", "transcendent"]

class AlteredConsciousnessModel:
    def predict(self, signals: np.ndarray, fs: float = 256.0) -> Dict:
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
        theta = bp(4, 8); alpha = bp(8, 12); beta = bp(12, 30); gamma = bp(30, 45)
        total = theta + alpha + beta + gamma + 1e-9
        # Altered state index: high theta+alpha, low beta, occasional gamma
        altered_idx = float(np.clip(
            0.40 * (theta / total) +
            0.30 * (alpha / total) -
            0.20 * (beta / total) +
            0.10 * (gamma / total) + 0.1,
            0.0, 1.0
        ))
        # Entropy proxy: spectral flatness
        psd_norm = psd / (psd.sum() + 1e-9)
        spectral_entropy = float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))
        spectral_entropy_norm = float(np.clip(spectral_entropy / 5.0, 0, 1))
        # State mapping
        if altered_idx < 0.2:   state = "normal"
        elif altered_idx < 0.35: state = "light_altered"
        elif altered_idx < 0.50: state = "moderate_altered"
        elif altered_idx < 0.65: state = "deep_altered"
        else:                    state = "transcendent"
        return {
            "state": state,
            "altered_consciousness_index": round(altered_idx, 4),
            "theta_fraction": round(theta / total, 4),
            "alpha_fraction": round(alpha / total, 4),
            "beta_suppression": round(1.0 - beta / total, 4),
            "spectral_entropy": round(spectral_entropy_norm, 4),
            "model_used": "feature_based_spectral",
        }

_model = AlteredConsciousnessModel()
def get_model(): return _model
