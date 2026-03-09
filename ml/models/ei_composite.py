"""Emotional Intelligence (EI) composite score from EEG multi-metric analysis.

EI components measurable via EEG:
1. Emotional awareness (FAA, alpha regulation)
2. Emotional regulation (LPP attenuation proxy, beta control)
3. Empathy (mu suppression, social cognition)
4. Emotional memory (theta-gamma coupling)
5. Self-regulation (prefrontal control index)

References:
    Mayer et al. (2008) — EI four-branch model
    Killgore (2019) — EEG and emotional intelligence
    Nummenmaa et al. (2014) — bodily maps of emotions
"""
from __future__ import annotations
import numpy as np
from typing import Dict

EI_LEVELS = ["developing", "average", "above_average", "high_ei"]


class EIComposite:
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
        mu    = bp(ch, 8, 13)
        total = theta + alpha + beta + gamma + 1e-9

        # FAA
        if n_ch >= 3:
            l_alpha = bp(signals[1], 8, 12)
            r_alpha = bp(signals[2], 8, 12)
            faa = float(np.log(l_alpha + 1e-9) - np.log(r_alpha + 1e-9))
        else:
            faa = 0.0

        # 1. Emotional awareness: balanced FAA, alpha accessibility
        awareness = float(np.clip(0.5 * (1 - abs(faa)) + 0.5 * alpha / total * 2, 0, 1))

        # 2. Emotional regulation: beta control, left PFC alpha
        regulation = float(np.clip(
            0.5 * np.clip((faa + 1) / 2, 0, 1) +
            0.5 * np.clip(1 - beta / total * 2, 0, 1),
            0, 1
        ))

        # 3. Empathy: mu suppression (social cognition)
        mu_suppress = float(np.clip(1.0 - mu / (mu + bp(ch, 12, 20) + 1e-9), 0, 1))

        # 4. Emotional memory: theta-gamma coupling
        tg_coupling = float(np.clip(theta / total * gamma / total * 20, 0, 1))

        # 5. Self-regulation: low theta/beta + moderate alpha
        self_reg = float(np.clip(
            0.5 * np.clip(1 - theta / (beta + 1e-9) * 0.3, 0, 1) +
            0.5 * np.clip(alpha / total * 2, 0, 1),
            0, 1
        ))

        ei_score = float(np.clip(
            0.25 * awareness +
            0.25 * regulation +
            0.20 * mu_suppress +
            0.15 * tg_coupling +
            0.15 * self_reg,
            0.0, 1.0
        ))

        if ei_score < 0.30:   level = "developing"
        elif ei_score < 0.55: level = "average"
        elif ei_score < 0.75: level = "above_average"
        else:                 level = "high_ei"

        return {
            "ei_level": level,
            "ei_composite_score": round(ei_score, 4),
            "emotional_awareness": round(awareness, 4),
            "emotional_regulation": round(regulation, 4),
            "empathy_index": round(mu_suppress, 4),
            "emotional_memory": round(tg_coupling, 4),
            "self_regulation": round(self_reg, 4),
            "frontal_asymmetry_faa": round(faa, 4),
            "model_used": "feature_based_ei_composite",
        }


_model = EIComposite()
def get_model(): return _model
