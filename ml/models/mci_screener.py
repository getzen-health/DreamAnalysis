"""Early MCI/Alzheimer screening using EEG slowing and coherence markers.

AD/MCI signatures: diffuse alpha/theta slowing, reduced beta, reduced
inter-hemispheric coherence, increased delta power.

References:
    Jeong (2004) — EEG dynamics in Alzheimer's disease
    Dauwels et al. (2010) — EEG markers for MCI
    Babiloni et al. (2016) — resting EEG for AD biomarkers
"""
from __future__ import annotations
import numpy as np
from typing import Dict

class MCIScreener:
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
        delta = bp(0.5, 4); theta = bp(4, 8); alpha = bp(8, 12); beta = bp(12, 30)
        total = delta + theta + alpha + beta + 1e-9
        # Slowing ratio: (delta+theta)/(alpha+beta) — elevated in MCI
        slowing_ratio = float((delta + theta) / (alpha + beta + 1e-9))
        # Peak alpha frequency (PAF) proxy — shifts left in MCI
        alpha_f = f[(f >= 8) & (f <= 12)]
        alpha_p = psd[(f >= 8) & (f <= 12)]
        paf = float(alpha_f[np.argmax(alpha_p)]) if len(alpha_f) > 0 else 10.0
        paf_norm = float(np.clip((paf - 8.0) / 4.0, 0, 1))  # 1=normal (12Hz), 0=slowed (8Hz)
        # Delta burden (elevated in moderate AD)
        delta_burden = float(np.clip(delta / total * 3, 0, 1))
        # Risk score: high slowing + low PAF + high delta = higher risk
        risk = float(np.clip(
            0.40 * np.clip(slowing_ratio / 3.0, 0, 1) +
            0.35 * (1.0 - paf_norm) +
            0.25 * delta_burden,
            0.0, 1.0
        ))
        if risk < 0.25:   category = "low_risk"
        elif risk < 0.50: category = "mild_concern"
        elif risk < 0.70: category = "moderate_concern"
        else:             category = "high_risk"
        return {
            "risk_category": category,
            "mci_risk_score": round(risk, 4),
            "slowing_ratio": round(slowing_ratio, 4),
            "peak_alpha_freq_hz": round(paf, 2),
            "delta_burden": round(delta_burden, 4),
            "note": "Screening only — not a clinical diagnosis",
            "model_used": "feature_based_eeg_slowing",
        }

_model = MCIScreener()
def get_model(): return _model
