"""Autism spectrum screening via EEG connectivity and spectral features.

ASD EEG signatures: reduced long-range coherence, atypical alpha/theta,
increased high-frequency local connectivity, reduced mu suppression.

References:
    Coben et al. (2008) — EEG coherence in autism
    Bosl et al. (2011) — EEG complexity as ASD biomarker
    Wang et al. (2013) — mu rhythm and social cognition
"""
from __future__ import annotations
import numpy as np
from typing import Dict

class AutismScreener:
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
        ch0 = signals[0]
        mu = bp(ch0, 8, 13)   # mu rhythm proxy (overlaps alpha)
        theta = bp(ch0, 4, 8)
        beta  = bp(ch0, 12, 30)
        hbeta = bp(ch0, 20, 30)
        # Mu suppression index (lower = more mu suppression = typical)
        mu_ratio = float(np.clip(mu / (mu + beta + 1e-9), 0, 1))
        # Inter-channel coherence proxy (AF7 vs AF8 if available)
        if n_ch >= 3:
            f1, p1 = welch(signals[1], fs=fs, nperseg=nperseg)
            f2, p2 = welch(signals[2], fs=fs, nperseg=nperseg)
            # Coherence proxy: 1 - |P1-P2|/(P1+P2+1e-9) in alpha band
            idx_a = (f1 >= 8) & (f1 <= 12)
            diff_ratio = float(np.mean(np.abs(p1[idx_a] - p2[idx_a]) / (p1[idx_a] + p2[idx_a] + 1e-9))) if idx_a.any() else 0.5
            coherence_proxy = float(np.clip(1.0 - diff_ratio, 0, 1))
        else:
            coherence_proxy = 0.5
        # Complexity: sample entropy proxy via autocorrelation decay
        ac = float(np.corrcoef(ch0[:-1], ch0[1:])[0, 1]) if n_samples > 2 else 0.0
        complexity = float(np.clip(1.0 - abs(ac), 0, 1))
        # ASD risk: reduced coherence + preserved mu + high hbeta
        risk = float(np.clip(
            0.35 * (1.0 - coherence_proxy) +
            0.35 * mu_ratio +
            0.20 * (hbeta / (beta + 1e-9)) * 0.3 +
            0.10 * complexity,
            0.0, 1.0
        ))
        if risk < 0.25:   category = "low_risk"
        elif risk < 0.50: category = "mild_atypical"
        elif risk < 0.70: category = "moderate_atypical"
        else:             category = "high_atypical"
        return {
            "risk_category": category,
            "asd_atypicality_score": round(risk, 4),
            "mu_suppression_index": round(1.0 - mu_ratio, 4),
            "inter_hemispheric_coherence": round(coherence_proxy, 4),
            "eeg_complexity": round(complexity, 4),
            "note": "Wellness indicator only — not a clinical assessment",
            "model_used": "feature_based_connectivity",
            "not_validated": True,
            "scale_context": (
                "Scores are research-grade wellness estimates from consumer EEG "
                "hardware. This is not a medical device. Results are for personal "
                "wellness awareness only, not validated clinical assessments."
            ),
        }

_model = AutismScreener()
def get_model(): return _model
