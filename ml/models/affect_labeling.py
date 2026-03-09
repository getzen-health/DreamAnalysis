"""Affect labeling efficacy tracker via LPP amplitude reduction.

Affect labeling (putting feelings into words) reduces amygdala activation
and attenuates the Late Positive Potential (LPP, 400-1000ms after stimulus)
amplitude — a validated marker of emotional regulation.

EEG proxy without time-locked stimulus:
- Reduced high-beta (20-30 Hz) = reduced emotional arousal
- Increased left PFC alpha (AF7) = greater regulation success
- Positive FAA = approach to labeling task (engagement)
- Alpha increase post-labeling = successful downregulation

References:
    Lieberman et al. (2007) — affect labeling and LPP
    Torre & Lieberman (2018) — putting feelings into words
    Burklund et al. (2014) — neural correlates of affect labeling
"""
from __future__ import annotations
import numpy as np
from typing import Dict

EFFICACY_LEVELS = ["ineffective", "partial", "effective", "highly_effective"]


class AffectLabelingTracker:
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
        alpha = bp(ch, 8, 12)
        beta  = bp(ch, 12, 30)
        hbeta = bp(ch, 20, 30)
        total = alpha + beta + 1e-9

        # FAA: left alpha higher = PFC regulation
        if n_ch >= 3:
            l_alpha = bp(signals[1], 8, 12)
            r_alpha = bp(signals[2], 8, 12)
            faa = float(np.log(l_alpha + 1e-9) - np.log(r_alpha + 1e-9))
            left_pfc_reg = float(np.clip((faa + 1) / 2, 0, 1))
        else:
            faa = 0.0
            left_pfc_reg = 0.5

        # High-beta attenuation (reduced emotional arousal after labeling)
        hbeta_attenuated = float(np.clip(1.0 - hbeta / (beta + 1e-9), 0, 1))
        # Alpha increase (regulation success)
        alpha_increase = float(np.clip(alpha / total * 2, 0, 1))

        efficacy_score = float(np.clip(
            0.35 * left_pfc_reg +
            0.35 * hbeta_attenuated +
            0.30 * alpha_increase,
            0.0, 1.0
        ))

        if efficacy_score < 0.30:   level = "ineffective"
        elif efficacy_score < 0.50: level = "partial"
        elif efficacy_score < 0.70: level = "effective"
        else:                       level = "highly_effective"

        return {
            "labeling_efficacy_level": level,
            "efficacy_score": round(efficacy_score, 4),
            "left_pfc_regulation": round(left_pfc_reg, 4),
            "high_beta_attenuation": round(hbeta_attenuated, 4),
            "alpha_regulation_index": round(alpha_increase, 4),
            "frontal_asymmetry_faa": round(faa, 4),
            "model_used": "feature_based_lpp_proxy",
        }


_model = AffectLabelingTracker()
def get_model(): return _model
