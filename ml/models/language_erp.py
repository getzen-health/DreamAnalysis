"""N400/P600 ERP-based language processing monitor.

N400: negative deflection 300-500ms post word onset — semantic surprise.
P600: positive deflection 500-800ms post word onset — syntactic violation.

References:
    Kutas & Federmeier (2011) — 30 years of N400
    Osterhout & Holcomb (1992) — P600 syntactic ERP
    Hollenstein et al. (2019) — EEG + NLP reading datasets
"""
from __future__ import annotations

import numpy as np
from typing import Dict


class LanguageERPModel:
    """Feature-based N400/P600 extractor for Muse 2 (4-channel, 256 Hz)."""

    def predict(self, signals: np.ndarray, fs: float = 256.0,
                word_onset_ms: float = 0.0) -> Dict:
        """Compute N400 and P600 language ERP features.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array
            fs: sampling rate (Hz)
            word_onset_ms: sample offset of word onset within signals (ms)

        Returns:
            dict with n400_amplitude, p600_amplitude, semantic_surprise_index,
            syntactic_load_index, comprehension_score, model_used
        """
        if signals.ndim == 1:
            signals = signals[np.newaxis, :]

        n_ch, n_samples = signals.shape

        onset_samp = int(word_onset_ms * fs / 1000)

        def window_mean(sig, t_start_ms, t_end_ms):
            s = onset_samp + int(t_start_ms * fs / 1000)
            e = onset_samp + int(t_end_ms * fs / 1000)
            s = max(0, min(s, n_samples - 1))
            e = max(s + 1, min(e, n_samples))
            return float(np.mean(sig[s:e])) if e > s else 0.0

        # Use mean across available channels (AF7/AF8 frontal N400 proxy)
        ch_mean = np.mean(signals, axis=0)

        # Baseline: pre-stimulus -200 to 0 ms (or first 50 samples if no pre)
        baseline_s = max(0, onset_samp - int(0.2 * fs))
        baseline_e = onset_samp if onset_samp > 0 else min(int(0.05 * fs), n_samples)
        baseline = float(np.mean(ch_mean[baseline_s:baseline_e])) if baseline_e > baseline_s else 0.0

        # N400 window: 300–500 ms post onset
        n400_raw = window_mean(ch_mean, 300, 500) - baseline

        # P600 window: 500–800 ms post onset
        p600_raw = window_mean(ch_mean, 500, 800) - baseline

        # Semantic surprise index: larger negative N400 → more surprise
        # Normalize to 0–1 (typical N400 effect: 2–8 µV)
        semantic_surprise = float(np.clip(-n400_raw / 8.0 + 0.5, 0.0, 1.0))

        # Syntactic load index: larger positive P600 → more syntactic effort
        syntactic_load = float(np.clip(p600_raw / 8.0 + 0.5, 0.0, 1.0))

        # Band-power features for supplementary indices
        from scipy.signal import welch
        nperseg = min(n_samples, int(fs * 2))
        f, psd = welch(ch_mean, fs=fs, nperseg=nperseg)

        def bp(flo, fhi):
            idx = (f >= flo) & (f <= fhi)
            return float(np.mean(psd[idx])) if idx.any() else 1e-9

        theta = bp(4, 8)
        alpha = bp(8, 12)
        beta = bp(12, 30)

        # Comprehension score: high alpha (relaxed reading) + low theta (low confusion)
        comprehension_score = float(np.clip(
            0.5 * (alpha / (alpha + theta + 1e-9)) +
            0.5 * (1.0 - beta / (alpha + beta + 1e-9)),
            0.0, 1.0
        ))

        return {
            "n400_amplitude_uv": round(n400_raw, 4),
            "p600_amplitude_uv": round(p600_raw, 4),
            "semantic_surprise_index": round(semantic_surprise, 4),
            "syntactic_load_index": round(syntactic_load, 4),
            "comprehension_score": round(comprehension_score, 4),
            "theta_power": round(theta, 6),
            "alpha_power": round(alpha, 6),
            "model_used": "feature_based_erp",
        }


_model = LanguageERPModel()


def get_model() -> LanguageERPModel:
    return _model
