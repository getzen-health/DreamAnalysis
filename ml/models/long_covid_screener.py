"""Long COVID / chronic fatigue syndrome EEG screening.

EEG signatures in Long COVID/CFS (2024 literature):
- Increased delta/theta ratio (neural slowing)
- Reduced beta power (cognitive fatigue)
- Reduced alpha peak frequency (cortical sluggishness)
- Increased P2 amplitude variability (sensory gating disruption)

References:
    Schrader et al. (2024) — neurophysiological markers of Long COVID
    Ocon (2013) — CFS/ME EEG slowing
    Naviaux et al. — neuroinflammation EEG correlates
"""
from __future__ import annotations
import numpy as np
from typing import Dict

RISK_CATEGORIES = ["low_risk", "mild_concern", "moderate_concern", "high_risk"]


class LongCOVIDScreener:
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

        delta = bp(0.5, 4)
        theta = bp(4, 8)
        alpha = bp(8, 12)
        beta  = bp(12, 30)
        total = delta + theta + alpha + beta + 1e-9

        # Slowing: elevated delta+theta relative to alpha+beta
        slowing_ratio = float((delta + theta) / (alpha + beta + 1e-9))
        # Beta fatigue: reduced beta fraction
        beta_deficit = float(np.clip(1.0 - beta / total * 4, 0, 1))
        # PAF slowing
        af = f[(f >= 8) & (f <= 12)]; ap = psd[(f >= 8) & (f <= 12)]
        paf = float(af[np.argmax(ap)]) if len(af) > 0 else 10.0
        paf_deficit = float(np.clip((10.5 - paf) / 2.5, 0, 1))  # 10.5 Hz = typical normal
        # Delta burden (neuroinflammation marker)
        delta_burden = float(np.clip(delta / total * 3, 0, 1))

        risk = float(np.clip(
            0.35 * np.clip(slowing_ratio / 3.0, 0, 1) +
            0.30 * beta_deficit +
            0.20 * paf_deficit +
            0.15 * delta_burden,
            0.0, 1.0
        ))

        if risk < 0.25:   category = "low_risk"
        elif risk < 0.50: category = "mild_concern"
        elif risk < 0.70: category = "moderate_concern"
        else:             category = "high_risk"

        return {
            "risk_category": category,
            "long_covid_risk_score": round(risk, 4),
            "slowing_ratio": round(slowing_ratio, 4),
            "beta_deficit": round(beta_deficit, 4),
            "peak_alpha_freq_hz": round(paf, 2),
            "delta_burden": round(delta_burden, 4),
            "note": "Screening only — not a clinical diagnosis",
            "model_used": "feature_based_spectral_slowing",
        }


_model = LongCOVIDScreener()
def get_model(): return _model
