"""Big Five personality trait estimator from resting EEG.

EEG correlates of Big Five (De Pascalis & Varriale, 2012; Stenberg, 2008):
- Neuroticism: right > left frontal alpha (Davidson), higher beta
- Extraversion: higher left frontal alpha, higher theta
- Openness: higher alpha power overall, creativity signature
- Conscientiousness: higher beta, lower theta/beta ratio (focused)
- Agreeableness: balanced hemispheric activity, lower HF asymmetry

References:
    Stenberg (2008) — EEG and Big Five personality
    De Pascalis & Varriale (2012) — frontal alpha and extraversion
    Wake & Bhatt (2021) — resting EEG for personality (88-94% binary)
"""
from __future__ import annotations
import numpy as np
from typing import Dict


class BigFiveEstimator:
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

        # FAA: AF7 (ch1) vs AF8 (ch2)
        if n_ch >= 3:
            l_alpha = bp(signals[1], 8, 12)
            r_alpha = bp(signals[2], 8, 12)
            faa = float(np.log(l_alpha + 1e-9) - np.log(r_alpha + 1e-9))
        else:
            faa = 0.0
            l_alpha = alpha; r_alpha = alpha

        total = theta + alpha + beta + 1e-9
        alpha_frac = alpha / total
        beta_frac  = beta / total
        theta_frac = theta / total
        tb_ratio   = theta / (beta + 1e-9)

        # Trait scores (0 = low, 1 = high) — EEG-based approximations
        neuroticism      = float(np.clip(0.5 * max(0, -faa) + 0.3 * beta_frac + 0.2 * hbeta / (beta + 1e-9), 0, 1))
        extraversion     = float(np.clip(0.5 * max(0, faa)  + 0.3 * theta_frac + 0.2 * alpha_frac, 0, 1))
        openness         = float(np.clip(0.5 * alpha_frac + 0.3 * tb_ratio * 0.5 + 0.2 * theta_frac, 0, 1))
        conscientiousness = float(np.clip(0.5 * beta_frac  + 0.3 * (1.0 - tb_ratio * 0.3) + 0.2 * (1.0 - theta_frac), 0, 1))
        agreeableness    = float(np.clip(0.5 * (1.0 - abs(faa)) + 0.3 * alpha_frac + 0.2 * (1.0 - hbeta / (beta + 1e-9)), 0, 1))

        # Clamp to [0.1, 0.9] — EEG provides directional signal, not absolute value
        def clamp(v): return round(float(np.clip(v, 0.1, 0.9)), 4)

        return {
            "neuroticism": clamp(neuroticism),
            "extraversion": clamp(extraversion),
            "openness": clamp(openness),
            "conscientiousness": clamp(conscientiousness),
            "agreeableness": clamp(agreeableness),
            "frontal_alpha_asymmetry": round(faa, 4),
            "confidence": "low",  # EEG → personality is population-level, not individual
            "note": "Directional indicator only — EEG cannot reliably measure personality traits individually",
            "model_used": "feature_based_big_five",
        }


_model = BigFiveEstimator()
def get_model(): return _model
