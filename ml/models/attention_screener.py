"""Attention pattern screener using aperiodic EEG features and theta/beta ratio.

DISCLAIMER: This is a wellness indicator only. It is NOT a medical diagnosis for ADHD
or any other condition. Clinical diagnosis requires professional evaluation.

Scientific basis:
- Frontiers in Psychiatry (2025): XGBoost + aperiodic exponent for attention screening
- medRxiv (2025): Aperiodic offset AUC=0.86, exponent AUC=0.85 for ADHD screening
- FDA-cleared biomarker: theta/beta ratio (TBR) at frontal sites
- Key ADHD marker: no change in aperiodic exponent from rest to task (neurotypicals show decrease)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from typing import Dict, Optional


DISCLAIMER = (
    "This attention risk index is a wellness indicator only. "
    "It is NOT a medical diagnosis for ADHD or any attention disorder. "
    "Clinical diagnosis requires evaluation by a qualified healthcare professional."
)


def _compute_aperiodic_features(signal: np.ndarray, fs: float = 256.0) -> Dict:
    """Estimate aperiodic (1/f) spectral features without specparam dependency.

    Uses log-log linear regression on the PSD as a fast approximation.
    specparam gives more accurate results but requires an extra dependency.
    """
    nperseg = min(len(signal), int(fs * 2))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    # Fit aperiodic component: log(psd) ~ offset - exponent * log(freq)
    # Focus on 2-45 Hz range (avoid DC and high-freq artifacts)
    mask = (freqs >= 2) & (freqs <= 45)
    if mask.sum() < 5:
        return {"aperiodic_offset": 0.0, "aperiodic_exponent": 1.0, "fit_r2": 0.0}

    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask] + 1e-30)

    # Linear regression
    coeffs = np.polyfit(log_f, log_p, 1)
    exponent = -coeffs[0]  # negative slope = positive exponent
    offset = coeffs[1]

    # R² of fit
    predicted = np.polyval(coeffs, log_f)
    ss_res = np.sum((log_p - predicted) ** 2)
    ss_tot = np.sum((log_p - log_p.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))

    return {
        "aperiodic_offset": float(offset),
        "aperiodic_exponent": float(exponent),
        "fit_r2": float(np.clip(r2, 0, 1)),
    }


class AttentionScreener:
    """Attention pattern screener using aperiodic EEG features and TBR.

    Output: attention_risk_index (0-1) where higher = more atypical pattern.
    This is a wellness indicator, not a clinical diagnostic tool.
    """

    # Population norms (approximate, from literature)
    # Typical aperiodic exponent at frontal sites: ~1.5-2.5
    # ADHD: lower exponent (flatter 1/f slope)
    NORM_EXPONENT_MEAN = 2.0
    NORM_EXPONENT_STD = 0.5
    NORM_TBR_MEAN = 2.5       # typical frontal TBR
    NORM_TBR_STD = 1.0

    def __init__(self):
        self._rest_exponent: Optional[float] = None
        self._task_exponent: Optional[float] = None

    def predict(self, signals: np.ndarray, fs: float = 256.0) -> Dict:
        """Screen attention patterns from EEG.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) array
            fs: sampling rate in Hz

        Returns:
            dict with attention_risk_index, risk_level, tbr, aperiodic features,
            disclaimer, and component scores.
        """
        # Use mean of AF7+AF8 (ch1, ch2) if multichannel
        if signals.ndim == 2 and signals.shape[0] >= 3:
            frontal = (signals[1] + signals[2]) / 2.0  # AF7 + AF8
        elif signals.ndim == 2:
            frontal = signals[0]
        else:
            frontal = signals

        # Band powers for TBR
        from processing.eeg_processor import preprocess, extract_band_powers
        processed = preprocess(frontal, fs)
        bands = extract_band_powers(processed, fs)

        theta = bands.get("theta", 0.15)
        beta = bands.get("beta", 0.15)
        alpha = bands.get("alpha", 0.20)

        eps = 1e-10
        tbr = float(theta / (beta + eps))

        # Aperiodic features
        ap_feats = _compute_aperiodic_features(processed, fs)
        exponent = ap_feats["aperiodic_exponent"]

        # Attention risk from TBR
        # High TBR (>3.5) = more atypical, consistent with frontal theta elevation
        tbr_z = (tbr - self.NORM_TBR_MEAN) / self.NORM_TBR_STD
        tbr_risk = float(np.clip(tbr_z / 3.0, -1, 1) * 0.5 + 0.5)

        # Attention risk from aperiodic exponent
        # Lower exponent = flatter slope = more atypical
        exp_z = (self.NORM_EXPONENT_MEAN - exponent) / self.NORM_EXPONENT_STD
        exp_risk = float(np.clip(exp_z / 3.0, -1, 1) * 0.5 + 0.5)

        # Alpha peak frequency proxy (lower IAF associated with attention issues)
        # Look for spectral peak in alpha band
        nperseg = min(len(processed), int(fs * 2))
        freqs, psd = welch(processed, fs=fs, nperseg=nperseg)
        alpha_mask = (freqs >= 7) & (freqs <= 13)
        if alpha_mask.any():
            iaf = float(freqs[alpha_mask][np.argmax(psd[alpha_mask])])
        else:
            iaf = 10.0
        # Lower IAF (<9 Hz) = slightly elevated risk
        iaf_risk = float(np.clip((9.5 - iaf) / 2.0, 0, 1))

        # Combined risk index (weighted)
        risk_index = float(
            0.40 * tbr_risk
            + 0.40 * exp_risk
            + 0.20 * iaf_risk
        )
        risk_index = float(np.clip(risk_index, 0.0, 1.0))

        # Dynamic response check (requires prior rest recording)
        dynamic_pattern = None
        if self._rest_exponent is not None:
            exp_change = exponent - self._rest_exponent
            # Neurotypical: exponent decreases during task (more focused)
            # ADHD-like: no change or increase
            if exp_change < -0.15:
                dynamic_pattern = "typical"  # clear decrease
            elif exp_change < 0.05:
                dynamic_pattern = "borderline"
            else:
                dynamic_pattern = "atypical"  # no decrease or increase
            self._task_exponent = exponent

        if risk_index >= 0.65:
            risk_level = "elevated"
        elif risk_index >= 0.45:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return {
            "attention_risk_index": round(risk_index, 4),
            "risk_level": risk_level,
            "tbr": round(tbr, 3),
            "tbr_risk": round(tbr_risk, 3),
            "aperiodic_exponent": round(exponent, 3),
            "aperiodic_offset": round(ap_feats["aperiodic_offset"], 3),
            "aperiodic_risk": round(exp_risk, 3),
            "iaf_hz": round(iaf, 2),
            "iaf_risk": round(iaf_risk, 3),
            "dynamic_pattern": dynamic_pattern,
            "alpha_power": round(float(alpha), 4),
            "theta_power": round(float(theta), 4),
            "beta_power": round(float(beta), 4),
            "disclaimer": DISCLAIMER,
            "model_type": "aperiodic_tbr_heuristic",
            "not_validated": True,
            "scale_context": (
                "Scores are research-grade estimates from consumer EEG hardware. "
                "They have not been validated against clinical diagnostic instruments "
                "and must not be used for clinical diagnosis."
            ),
        }

    def record_rest_baseline(self, signals: np.ndarray, fs: float = 256.0) -> Dict:
        """Record resting-state aperiodic baseline for dynamic response test.

        Call this during an eyes-closed resting period before a cognitive task.
        """
        if signals.ndim == 2 and signals.shape[0] >= 3:
            frontal = (signals[1] + signals[2]) / 2.0
        elif signals.ndim == 2:
            frontal = signals[0]
        else:
            frontal = signals

        from processing.eeg_processor import preprocess
        processed = preprocess(frontal, fs)
        ap_feats = _compute_aperiodic_features(processed, fs)
        self._rest_exponent = ap_feats["aperiodic_exponent"]
        return {"status": "recorded", "rest_exponent": round(self._rest_exponent, 3)}


_screener_instance: Optional[AttentionScreener] = None


def get_attention_screener() -> AttentionScreener:
    global _screener_instance
    if _screener_instance is None:
        _screener_instance = AttentionScreener()
    return _screener_instance
