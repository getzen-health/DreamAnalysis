"""EEG decision confidence and risk-taking predictor.

Pre-decision EEG markers: beta desynchronization during deliberation,
alpha suppression in prefrontal cortex for risky decisions, P300-like
late positive component for confident choices.

References:
    Bechara et al. (1997) — somatic marker hypothesis + EEG
    Cohen et al. (2015) — beta and decision commitment
    Cavanagh & Frank (2014) — theta and response conflict
"""
from __future__ import annotations
import numpy as np
from typing import Dict

class DecisionConfidenceModel:
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
        theta = bp(4, 8); alpha = bp(8, 12); beta = bp(12, 30)
        # Conflict index: high theta/beta = more deliberation conflict
        conflict = float(np.clip(theta / (beta + 1e-9) * 0.5, 0, 1))
        # Confidence: high beta + low theta = more committed
        confidence = float(np.clip(
            beta / (alpha + theta + 1e-9) * 0.4 + 0.3, 0, 1
        ))
        confidence = float(np.clip(confidence - conflict * 0.3, 0, 1))
        # Risk propensity: high beta + low alpha = reward-seeking
        risk_propensity = float(np.clip(
            beta / (alpha + beta + 1e-9) - 0.4, 0, 1
        ))
        # Decision readiness: low conflict + high confidence
        readiness = float(np.clip(
            0.6 * confidence + 0.4 * (1.0 - conflict), 0, 1
        ))
        if confidence < 0.35:  conf_label = "uncertain"
        elif confidence < 0.55: conf_label = "moderate"
        elif confidence < 0.75: conf_label = "confident"
        else:                   conf_label = "highly_confident"
        return {
            "confidence_label": conf_label,
            "confidence_score": round(confidence, 4),
            "conflict_index": round(conflict, 4),
            "risk_propensity": round(risk_propensity, 4),
            "decision_readiness": round(readiness, 4),
            "model_used": "feature_based_theta_beta",
        }

_model = DecisionConfidenceModel()
def get_model(): return _model
