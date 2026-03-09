"""Neuroaesthetic response detector for beauty and art perception.

EEG signatures of aesthetic appreciation (Vessel et al., 2012; Pearce et al., 2016):
- Alpha synchronization (especially parieto-occipital) — perceptual fluency
- Frontal theta → default mode network engagement (meaningful stimuli)
- Gamma bursts (prefrontal) — moments of insight/beauty recognition
- Positive FAA → approach toward beautiful stimuli

References:
    Vessel et al. (2012) — neuroaesthetics and default mode network
    Pearce et al. (2016) — mobile EEG for art appreciation
    Wassiliwizky et al. (2017) — chills during art
"""
from __future__ import annotations
import numpy as np
from typing import Dict

RESPONSE_LEVELS = ["neutral", "mildly_engaged", "aesthetically_moved", "peak_experience"]


class NeuroaestheticDetector:
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
        theta  = bp(ch, 4, 8)
        alpha  = bp(ch, 8, 12)
        beta   = bp(ch, 12, 30)
        gamma  = bp(ch, 30, 45)
        total  = theta + alpha + beta + gamma + 1e-9

        # FAA: positive = approach → aesthetic engagement
        if n_ch >= 3:
            l_alpha = bp(signals[1], 8, 12)
            r_alpha = bp(signals[2], 8, 12)
            faa = float(np.log(l_alpha + 1e-9) - np.log(r_alpha + 1e-9))
        else:
            faa = 0.0

        # Alpha synchronization (perceptual fluency — higher = more engaged)
        alpha_sync = float(np.clip(alpha / total * 3, 0, 1))
        # Frontal theta (DMN engagement)
        theta_engage = float(np.clip(theta / total * 4, 0, 1))
        # Gamma (peak moment detection)
        gamma_burst = float(np.clip(gamma / total * 5, 0, 1))
        # Positive FAA contribution
        faa_engage = float(np.clip((faa + 1) / 2, 0, 1))

        # Aesthetic response score
        aes_score = float(np.clip(
            0.30 * alpha_sync +
            0.30 * theta_engage +
            0.20 * faa_engage +
            0.20 * gamma_burst,
            0.0, 1.0
        ))

        if aes_score < 0.25:   level = "neutral"
        elif aes_score < 0.50: level = "mildly_engaged"
        elif aes_score < 0.75: level = "aesthetically_moved"
        else:                  level = "peak_experience"

        return {
            "aesthetic_response_level": level,
            "aesthetic_score": round(aes_score, 4),
            "alpha_synchronization": round(alpha_sync, 4),
            "theta_engagement": round(theta_engage, 4),
            "gamma_burst_index": round(gamma_burst, 4),
            "approach_motivation_faa": round(faa, 4),
            "model_used": "feature_based_neuroaesthetics",
        }


_model = NeuroaestheticDetector()
def get_model(): return _model
