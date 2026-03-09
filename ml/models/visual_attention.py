"""EEG-based visual attention and gaze zone estimation.

Alpha suppression (8-12 Hz) at occipital-like channels indexes visual
cortex engagement. Posterior alpha ERD (event-related desynchronization)
marks where attention is directed. With Muse 2's frontal channels we use
alpha asymmetry and theta bursts as proxies.

References:
    Thut et al. (2006) — alpha-band oscillations as attention gate
    Kelly et al. (2006) — alpha lateralization and spatial attention
    Worden et al. (2000) — posterior alpha and saccadic preparation
"""
from __future__ import annotations

import numpy as np
from typing import Dict


class VisualAttentionModel:
    """Estimates visual attention zone from EEG band power."""

    # Gaze zones for 3x3 grid
    ZONES = [
        "top-left", "top-center", "top-right",
        "mid-left", "center",     "mid-right",
        "bot-left", "bot-center", "bot-right",
    ]

    def predict(self, signals: np.ndarray, fs: float = 256.0) -> Dict:
        """Estimate visual attention direction and engagement.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG
            fs: sampling rate Hz

        Returns:
            dict with attention_zone, horizontal_bias, vertical_bias,
            alpha_suppression, visual_engagement, sustained_attention_index
        """
        if signals.ndim == 1:
            signals = signals[np.newaxis, :]

        n_ch, n_samples = signals.shape

        from scipy.signal import welch
        nperseg = min(n_samples, int(fs * 2))

        def bp(sig, flo, fhi):
            f, p = welch(sig, fs=fs, nperseg=nperseg)
            idx = (f >= flo) & (f <= fhi)
            return float(np.mean(p[idx])) if idx.any() else 1e-9

        # Per-channel band powers (use up to 4 channels)
        ch_alpha, ch_theta, ch_beta = [], [], []
        for i in range(min(n_ch, 4)):
            ch_alpha.append(bp(signals[i], 8, 12))
            ch_theta.append(bp(signals[i], 4, 8))
            ch_beta.append(bp(signals[i], 12, 30))

        alpha = np.array(ch_alpha)
        theta = np.array(ch_theta)
        beta  = np.array(ch_beta)

        # Horizontal bias: AF7 (left) vs AF8 (right) alpha asymmetry
        # More left alpha suppression → attention directed right, and vice versa
        if n_ch >= 3:
            # ch1=AF7 (left), ch2=AF8 (right)
            left_alpha  = ch_alpha[1] if n_ch > 1 else ch_alpha[0]
            right_alpha = ch_alpha[2] if n_ch > 2 else ch_alpha[0]
            # Positive → rightward attention; negative → leftward
            h_bias = float(np.clip(
                np.log(left_alpha + 1e-9) - np.log(right_alpha + 1e-9), -2, 2
            ) / 2.0)  # [-1, 1]
        else:
            h_bias = 0.0

        # Vertical bias: high alpha → downward (resting gaze); low alpha + high theta → upward
        mean_alpha = float(np.mean(alpha))
        mean_theta = float(np.mean(theta))
        mean_beta  = float(np.mean(beta))
        # Upward gaze associated with increased frontal theta (working memory/visual search)
        v_bias = float(np.clip(
            (mean_theta / (mean_alpha + 1e-9) - 0.8) / 1.5, -1.0, 1.0
        ))

        # Map biases to 3x3 zone grid
        # h: -1=left, 0=center, +1=right  →  col 0,1,2
        col = int(np.clip(round((h_bias + 1.0) * 1.0), 0, 2))
        # v: -1=bottom, 0=mid, +1=top  →  row 0 (top), 1 (mid), 2 (bot)
        row = int(np.clip(round((1.0 - v_bias) * 1.0), 0, 2))
        zone = self.ZONES[row * 3 + col]

        # Alpha suppression index (0=no suppression, 1=full suppression)
        # Lower alpha relative to baseline (mean) = more visual engagement
        alpha_suppression = float(np.clip(
            1.0 - mean_alpha / (mean_alpha + mean_beta + 1e-9), 0.0, 1.0
        ))

        # Visual engagement = beta/(alpha+beta)
        visual_engagement = float(np.clip(
            mean_beta / (mean_alpha + mean_beta + 1e-9), 0.0, 1.0
        ))

        # Sustained attention: low variability in alpha + high beta
        if n_samples >= int(fs):
            # Split into 4 quarters and measure alpha variance
            q = n_samples // 4
            quarter_alphas = [
                bp(signals[0], 8, 12) if signals.shape[1] >= (i+1)*q
                else mean_alpha
                for i in range(4)
            ]
            alpha_var = float(np.var(quarter_alphas))
            sustained = float(np.clip(1.0 - alpha_var * 10, 0.0, 1.0))
        else:
            sustained = 0.5

        return {
            "attention_zone": zone,
            "horizontal_bias": round(h_bias, 4),
            "vertical_bias": round(v_bias, 4),
            "alpha_suppression": round(alpha_suppression, 4),
            "visual_engagement": round(visual_engagement, 4),
            "sustained_attention_index": round(sustained, 4),
            "grid_col": col,
            "grid_row": row,
            "model_used": "feature_based_alpha_lateralization",
        }


_model = VisualAttentionModel()


def get_model() -> VisualAttentionModel:
    return _model
