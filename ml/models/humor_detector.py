"""Humor and laughter response detector from frontal alpha-gamma EEG.

Humor processing involves:
- N400-like incongruity detection (frontal theta/negative wave)
- Resolution: frontal alpha synchronization + gamma bursts (insight)
- Positive FAA: approach motivation toward amusing content
- Frontal gamma (30-45 Hz): moment of 'getting the joke'

References:
    Moran et al. (2004) — frontal EEG and humor appreciation
    Samson et al. (2009) — EEG correlates of humor
    Chan et al. (2018) — mobile EEG humor detection
"""
from __future__ import annotations
import numpy as np
from typing import Dict

HUMOR_LEVELS = ["neutral", "mildly_amused", "amused", "highly_amused"]


class HumorDetector:
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

        # FAA: positive = approach (amusement)
        if n_ch >= 3:
            l_alpha = bp(signals[1], 8, 12)
            r_alpha = bp(signals[2], 8, 12)
            faa = float(np.log(l_alpha + 1e-9) - np.log(r_alpha + 1e-9))
        else:
            faa = 0.0

        # Alpha synchronization after joke resolution
        alpha_sync = float(np.clip(alpha / total * 3, 0, 1))
        # Gamma burst (insight/punchline)
        gamma_insight = float(np.clip(gamma / total * 6, 0, 1))
        # Theta incongruity (processing joke setup)
        theta_incongruity = float(np.clip(theta / total * 4, 0, 1))
        # FAA → approach (amusement)
        faa_contrib = float(np.clip((faa + 1) / 2, 0, 1))

        humor_score = float(np.clip(
            0.30 * alpha_sync +
            0.30 * gamma_insight +
            0.20 * faa_contrib +
            0.20 * theta_incongruity,
            0.0, 1.0
        ))

        if humor_score < 0.25:   level = "neutral"
        elif humor_score < 0.50: level = "mildly_amused"
        elif humor_score < 0.70: level = "amused"
        else:                    level = "highly_amused"

        return {
            "humor_level": level,
            "humor_score": round(humor_score, 4),
            "alpha_synchronization": round(alpha_sync, 4),
            "gamma_insight_index": round(gamma_insight, 4),
            "approach_motivation_faa": round(faa, 4),
            "theta_incongruity": round(theta_incongruity, 4),
            "model_used": "feature_based_humor_eeg",
        }


_model = HumorDetector()
def get_model(): return _model
