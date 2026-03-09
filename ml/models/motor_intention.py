"""EEG neuroprosthetic motor intention decoder.

Motor imagery produces contralateral beta ERD (8-30 Hz) and mu rhythm
suppression. Left hand imagery suppresses right-hemisphere beta; right hand
imagery suppresses left-hemisphere beta.

References:
    Pfurtscheller & Lopes da Silva (1999) — ERD/ERS review
    McFarland et al. (2000) — EEG-based motor BCI
    Wolpaw et al. (2002) — BCI for motor prosthetics
"""
from __future__ import annotations
import numpy as np
from typing import Dict

INTENTIONS = ["rest", "left_hand", "right_hand", "both_hands", "feet"]

class MotorIntentionModel:
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
        # Mu band (8-12 Hz) and beta (13-30 Hz) — key for motor imagery
        if n_ch >= 3:
            l_mu   = bp(signals[1], 8, 12)   # AF7 (left frontal)
            r_mu   = bp(signals[2], 8, 12)   # AF8 (right frontal)
            l_beta = bp(signals[1], 13, 30)
            r_beta = bp(signals[2], 13, 30)
        else:
            l_mu = r_mu = bp(signals[0], 8, 12)
            l_beta = r_beta = bp(signals[0], 13, 30)
        # ERD: lower power on contralateral side = motor intention
        # Left hand → right ERD; Right hand → left ERD
        lat_mu   = float(np.tanh((np.log(l_mu + 1e-9) - np.log(r_mu + 1e-9)) * 2))
        lat_beta = float(np.tanh((np.log(l_beta + 1e-9) - np.log(r_beta + 1e-9)) * 2))
        erd_magnitude = float(np.clip(abs(lat_beta), 0, 1))
        # Classify intention
        if erd_magnitude < 0.15:
            intention, control_signal = "rest", 0.0
        elif lat_beta > 0.15:      # right ERD → left hand imagery
            intention, control_signal = "left_hand", float(lat_beta)
        elif lat_beta < -0.15:     # left ERD → right hand imagery
            intention, control_signal = "right_hand", float(-lat_beta)
        elif erd_magnitude > 0.4:  # bilateral ERD → both hands
            intention, control_signal = "both_hands", erd_magnitude
        else:
            intention, control_signal = "feet", float(erd_magnitude * 0.5)
        probs = {i: 0.05 for i in INTENTIONS}
        probs[intention] = max(0.5, control_signal)
        total = sum(probs.values())
        probs = {k: round(v / total, 4) for k, v in probs.items()}
        return {
            "intention": intention,
            "control_signal": round(control_signal, 4),
            "lateral_mu_asymmetry": round(lat_mu, 4),
            "lateral_beta_erd": round(lat_beta, 4),
            "erd_magnitude": round(erd_magnitude, 4),
            "probabilities": probs,
            "model_used": "feature_based_erd_ers",
        }

_model = MotorIntentionModel()
def get_model(): return _model
