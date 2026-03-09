"""Emotional memory enhancement predictor via theta-gamma coupling.

Memory encoding is enhanced during emotional arousal via:
- Theta-gamma coupling: hippocampal theta (4-8 Hz) modulating cortical gamma (30-80 Hz)
- High frontal theta during encoding predicts better recall
- Alpha suppression during encoding (attention allocated)
- Moderate arousal (inverted-U Yerkes-Dodson for memory)

References:
    Lisman & Jensen (2013) — theta-gamma coupling and memory
    Fell et al. (2011) — frontal theta and memory encoding
    LaBar & Cabeza (2006) — emotional enhancement of memory
"""
from __future__ import annotations
import numpy as np
from typing import Dict

ENCODING_QUALITY = ["poor", "moderate", "good", "excellent"]


class EmotionalMemoryEnhancer:
    def predict(self, signals: np.ndarray, fs: float = 256.0) -> Dict:
        if signals.ndim == 1:
            signals = signals[np.newaxis, :]
        n_ch, n_samples = signals.shape
        from scipy.signal import welch
        nperseg = min(n_samples, int(fs * 2))

        def bp(sig, lo, hi):
            f, p = welch(sig, fs=fs, nperseg=nperseg)
            idx = (f >= lo) & (f <= hi)
            return float(np.mean(p[idx])) if idx.any() else 1e-9

        ch = signals[0]
        theta = bp(ch, 4, 8)
        alpha = bp(ch, 8, 12)
        beta  = bp(ch, 12, 30)
        gamma = bp(ch, 30, 45)
        total = theta + alpha + beta + gamma + 1e-9

        # Theta-gamma coupling proxy: product of theta and gamma fractions
        theta_frac = theta / total
        gamma_frac = gamma / total
        tg_coupling = float(np.clip(theta_frac * gamma_frac * 20, 0, 1))

        # Frontal theta (working memory / encoding readiness)
        fmt_index = float(np.clip(theta_frac * 4, 0, 1))
        # Alpha suppression (engagement)
        alpha_suppress = float(np.clip(1.0 - alpha / total * 2, 0, 1))
        # Moderate arousal (not too high beta)
        arousal_optimal = float(np.clip(
            1.0 - abs(beta / total - 0.25) * 4, 0, 1
        ))

        encoding_score = float(np.clip(
            0.35 * tg_coupling +
            0.30 * fmt_index +
            0.20 * alpha_suppress +
            0.15 * arousal_optimal,
            0.0, 1.0
        ))

        if encoding_score < 0.25:   quality = "poor"
        elif encoding_score < 0.50: quality = "moderate"
        elif encoding_score < 0.75: quality = "good"
        else:                       quality = "excellent"

        return {
            "encoding_quality": quality,
            "encoding_score": round(encoding_score, 4),
            "theta_gamma_coupling": round(tg_coupling, 4),
            "frontal_theta_index": round(fmt_index, 4),
            "alpha_suppression": round(alpha_suppress, 4),
            "arousal_optimality": round(arousal_optimal, 4),
            "model_used": "feature_based_theta_gamma_memory",
        }


_model = EmotionalMemoryEnhancer()
def get_model(): return _model
