"""EEG Parkinson's tremor and bradykinesia screening.

PD EEG signatures: elevated beta (13-30 Hz) in motor cortex due to
pathological synchronization in basal ganglia, slowing in alpha/theta,
and tremor-locked oscillations at 4-6 Hz.

References:
    Stoffers et al. (2007) — EEG slowing in early PD
    Little et al. (2013) — adaptive DBS from beta power
    Brittain & Brown (2014) — beta band in movement disorders
"""
from __future__ import annotations
import numpy as np
from typing import Dict

class ParkinsonsScreener:
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
        tremor = bp(4, 6)    # tremor-locked oscillations
        theta  = bp(6, 8)
        alpha  = bp(8, 12)
        beta   = bp(13, 30)  # pathological beta synchronization
        total  = tremor + theta + alpha + beta + 1e-9
        # Beta burden: elevated in PD motor cortex
        beta_burden = float(np.clip(beta / total - 0.2, 0, 1))
        # Tremor oscillation power
        tremor_idx = float(np.clip(tremor / (alpha + 1e-9) * 2, 0, 1))
        # Alpha slowing
        alpha_f = f[(f >= 8) & (f <= 12)]
        alpha_p = psd[(f >= 8) & (f <= 12)]
        paf = float(alpha_f[np.argmax(alpha_p)]) if len(alpha_f) > 0 else 10.0
        paf_norm = float(np.clip((paf - 8.0) / 4.0, 0, 1))
        # Risk score
        risk = float(np.clip(
            0.45 * beta_burden +
            0.30 * tremor_idx +
            0.25 * (1.0 - paf_norm),
            0.0, 1.0
        ))
        if risk < 0.20:   category = "low_risk"
        elif risk < 0.40: category = "mild_concern"
        elif risk < 0.60: category = "moderate_concern"
        else:             category = "high_risk"
        return {
            "risk_category": category,
            "pd_risk_score": round(risk, 4),
            "beta_burden": round(beta_burden, 4),
            "tremor_oscillation_index": round(tremor_idx, 4),
            "peak_alpha_freq_hz": round(paf, 2),
            "note": "Screening only — not a clinical diagnosis",
            "model_used": "feature_based_beta_tremor",
            "not_validated": True,
            "scale_context": (
                "Scores are research-grade estimates from consumer EEG hardware. "
                "They have not been validated against clinical diagnostic instruments "
                "and must not be used for clinical diagnosis."
            ),
        }

_model = ParkinsonsScreener()
def get_model(): return _model
