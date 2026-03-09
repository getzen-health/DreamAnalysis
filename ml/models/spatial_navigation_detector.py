"""Spatial navigation cognitive map detector via frontal midline theta.

During spatial navigation and mental map construction, hippocampal theta
(4-8 Hz) synchronizes with frontal midline theta (FMT). AF7 and AF8 channels
on Muse 2 pick up prefrontal theta reflecting working memory for spatial layout.

References:
    Caplan et al. (2003) — theta and spatial navigation
    Kaplan et al. (2012) — frontal midline theta during navigation
    Ekstrom et al. (2005) — hippocampal theta in humans
"""
from __future__ import annotations
import numpy as np
from typing import Dict

NAV_STATES = ["resting", "passive_viewing", "active_navigation", "map_consolidation"]


class SpatialNavigationDetector:
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
        theta  = bp(ch, 4, 8)
        alpha  = bp(ch, 8, 12)
        beta   = bp(ch, 12, 30)
        total  = theta + alpha + beta + 1e-9

        # Frontal midline theta — key marker
        fmt_power = float(theta / total)
        # Alpha suppression during active navigation (engage > rest)
        alpha_suppress = float(np.clip(1.0 - alpha / total * 2, 0, 1))
        # Beta engagement for route planning
        beta_engage = float(np.clip(beta / total * 2, 0, 1))

        # Navigation index
        nav_index = float(np.clip(
            0.50 * np.clip(fmt_power * 3, 0, 1) +
            0.30 * alpha_suppress +
            0.20 * beta_engage,
            0.0, 1.0
        ))

        # Theta/alpha ratio for map consolidation
        theta_alpha_ratio = float(theta / (alpha + 1e-9))

        if nav_index < 0.2:   state = "resting"
        elif nav_index < 0.40: state = "passive_viewing"
        elif nav_index < 0.65: state = "active_navigation"
        else:                  state = "map_consolidation"

        return {
            "navigation_state": state,
            "navigation_index": round(nav_index, 4),
            "fmt_power_fraction": round(fmt_power, 4),
            "alpha_suppression": round(alpha_suppress, 4),
            "theta_alpha_ratio": round(theta_alpha_ratio, 4),
            "model_used": "feature_based_fmt_navigation",
        }


_model = SpatialNavigationDetector()
def get_model(): return _model
