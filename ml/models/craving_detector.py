"""Addiction craving state detector with alpha-theta neurofeedback guidance.

Craving EEG markers (2024-2025 literature):
- Elevated beta (12-30 Hz) — hyperarousal, anxious craving
- Reduced alpha/theta — inability to relax
- Right > left frontal alpha — withdrawal motivation (Davidson asymmetry)
- High beta/alpha ratio — strongest single predictor

References:
    Wan et al. (2025, Addiction) — alpha-theta NFB meta-analysis
    Zilverstand et al. (2019) — qEEG craving biomarkers
    Sokhadze et al. (2008) — EEG NFB for addiction
"""
from __future__ import annotations
import numpy as np
from typing import Dict

CRAVING_LEVELS = ["baseline", "mild_craving", "moderate_craving", "strong_craving"]
NFB_PROTOCOLS = ["alpha_theta", "beta_suppression", "frontal_asymmetry"]


class CravingDetector:
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
        hbeta = bp(ch, 20, 30)

        # Frontal asymmetry: AF7 (ch1) vs AF8 (ch2)
        if n_ch >= 3:
            l_alpha = bp(signals[1], 8, 12)
            r_alpha = bp(signals[2], 8, 12)
            faa = float(np.log(l_alpha + 1e-9) - np.log(r_alpha + 1e-9))
            # Right > left alpha (negative FAA) = withdrawal motivation = craving
            asymmetry_craving = float(np.clip(-faa * 0.5, -1, 1))
        else:
            faa = 0.0
            asymmetry_craving = 0.0

        # Beta/alpha ratio (primary craving marker)
        beta_alpha_ratio = float(beta / (alpha + 1e-9))
        # Alpha/theta (recovery marker — should be high for relaxed state)
        alpha_theta_ratio = float(alpha / (theta + 1e-9))

        # Craving score
        craving_score = float(np.clip(
            0.40 * np.clip(beta_alpha_ratio / 3.0, 0, 1) +
            0.25 * np.clip(1.0 - alpha_theta_ratio / 2.0, 0, 1) +
            0.20 * np.clip((asymmetry_craving + 1) / 2.0, 0, 1) +
            0.15 * np.clip(hbeta / (beta + 1e-9), 0, 1),
            0.0, 1.0
        ))

        if craving_score < 0.25:   level = "baseline"
        elif craving_score < 0.50: level = "mild_craving"
        elif craving_score < 0.70: level = "moderate_craving"
        else:                      level = "strong_craving"

        # NFB recommendation
        if craving_score > 0.5:
            nfb_protocol = "alpha_theta"
            nfb_target = "increase_alpha_theta"
        elif beta_alpha_ratio > 2.0:
            nfb_protocol = "beta_suppression"
            nfb_target = "reduce_beta"
        else:
            nfb_protocol = "frontal_asymmetry"
            nfb_target = "balance_hemispheres"

        return {
            "craving_level": level,
            "craving_score": round(craving_score, 4),
            "beta_alpha_ratio": round(beta_alpha_ratio, 4),
            "alpha_theta_ratio": round(alpha_theta_ratio, 4),
            "frontal_asymmetry_faa": round(faa, 4),
            "nfb_protocol": nfb_protocol,
            "nfb_target": nfb_target,
            "model_used": "feature_based_beta_alpha",
        }


_model = CravingDetector()
def get_model(): return _model
