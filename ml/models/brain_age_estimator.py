"""Brain Age Estimation from 4-channel Muse 2 EEG.

Uses aperiodic spectral features (1/f slope) + band powers to estimate biological
brain age. Outputs Brain Age Gap = predicted_age - chronological_age.

Scientific basis:
- Banville et al. (2024, MIT Press): R²=0.3-0.5 using Muse S headband (n=5,200+)
- PMC 12321796 (2025): Aperiodic exponent top predictor via Shapley analysis
- Aperiodic exponent (1/f slope) decreases with age — measurable with 4 channels
- Brain Age Gap: positive = accelerated neural aging, negative = younger-than-expected

DISCLAIMER: This is a wellness indicator. Not a medical diagnosis.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from typing import Dict, Optional


DISCLAIMER = (
    "Brain age estimation is a research-grade wellness indicator based on spectral EEG features. "
    "It is NOT a medical diagnosis. Individual variation is high. "
    "Consult a healthcare professional for medical concerns."
)

# Population norms derived from Banville 2024 + literature
# Aperiodic exponent decreases linearly with age
# At age 20: ~2.3, at age 70: ~1.6 (approximate)
_AGE_EXPONENT_SLOPE = -0.014   # exponent change per year
_AGE_EXPONENT_INTERCEPT = 2.58  # expected exponent at age 0


def _compute_aperiodic(signal: np.ndarray, fs: float = 256.0) -> Dict:
    """Fast aperiodic exponent estimation via log-log PSD regression."""
    nperseg = min(len(signal), int(fs * 4))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    mask = (freqs >= 2) & (freqs <= 40)
    if mask.sum() < 5:
        return {"exponent": 2.0, "offset": 1.0, "r2": 0.0}

    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask] + 1e-30)
    coeffs = np.polyfit(log_f, log_p, 1)
    exponent = float(-coeffs[0])
    offset = float(coeffs[1])

    predicted = np.polyval(coeffs, log_f)
    ss_res = np.sum((log_p - predicted) ** 2)
    ss_tot = np.sum((log_p - log_p.mean()) ** 2)
    r2 = float(np.clip(1 - ss_res / (ss_tot + 1e-10), 0, 1))

    return {"exponent": exponent, "offset": offset, "r2": r2}


class BrainAgeEstimator:
    """Estimate biological brain age from 4-channel Muse 2 EEG.

    Uses aperiodic 1/f features + band powers as age predictors.
    The model uses population-level norms from Banville 2024.
    A trained LightGBM model can be loaded from models/saved/ if available.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._lgbm = None
        self._scaler = None
        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str):
        try:
            import joblib
            data = joblib.load(path)
            self._lgbm = data.get("model")
            self._scaler = data.get("scaler")
        except Exception:
            pass

    def _extract_features(self, signals: np.ndarray, fs: float) -> np.ndarray:
        """Extract age-predictive features per channel, then average."""
        from processing.eeg_processor import preprocess, extract_band_powers

        if signals.ndim == 1:
            channels = [signals]
        else:
            channels = [signals[i] for i in range(signals.shape[0])]

        all_features = []
        for ch in channels:
            processed = preprocess(ch, fs)
            bands = extract_band_powers(processed, fs)
            ap = _compute_aperiodic(processed, fs)

            # Per-channel feature vector
            feats = [
                ap["exponent"],
                ap["offset"],
                ap["r2"],
                bands.get("delta", 0.3),
                bands.get("theta", 0.15),
                bands.get("alpha", 0.20),
                bands.get("beta", 0.15),
                bands.get("high_beta", 0.05),
                bands.get("gamma", 0.05),
                bands.get("alpha", 0.20) / (bands.get("beta", 0.15) + 1e-10),
                bands.get("theta", 0.15) / (bands.get("alpha", 0.20) + 1e-10),
            ]
            all_features.append(feats)

        return np.mean(all_features, axis=0)

    def predict(
        self,
        signals: np.ndarray,
        fs: float = 256.0,
        chronological_age: Optional[float] = None,
    ) -> Dict:
        """Estimate biological brain age from EEG.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array
            fs: sampling rate
            chronological_age: user's actual age in years (for gap calculation)

        Returns:
            dict with predicted_age, brain_age_gap, percentile, confidence,
            aperiodic features, and disclaimer.
        """
        feats = self._extract_features(signals, fs)
        exponent = feats[0]

        # Heuristic age prediction from aperiodic exponent (Banville 2024 norms)
        # exponent = intercept + slope * age → age = (exponent - intercept) / slope
        predicted_age = (_AGE_EXPONENT_INTERCEPT - exponent) / (-_AGE_EXPONENT_SLOPE)
        predicted_age = float(np.clip(predicted_age, 15.0, 90.0))

        # If trained model available, use it instead
        if self._lgbm is not None:
            try:
                x = feats.reshape(1, -1)
                if self._scaler is not None:
                    x = self._scaler.transform(x)
                predicted_age = float(self._lgbm.predict(x)[0])
                predicted_age = float(np.clip(predicted_age, 15.0, 90.0))
            except Exception:
                pass

        # Brain Age Gap
        gap = None
        gap_interpretation = None
        if chronological_age is not None:
            gap = round(predicted_age - chronological_age, 1)
            if gap <= -5:
                gap_interpretation = "Your brain appears younger than average for your age."
            elif gap >= 5:
                gap_interpretation = "Your brain appears older than average for your age."
            else:
                gap_interpretation = "Your brain age is within typical range for your age."

        # Confidence based on fit quality
        r2 = float(feats[2])
        confidence = float(np.clip(0.3 + 0.5 * r2, 0.3, 0.8))

        # Rough percentile (normal distribution, population SD ~8 years)
        percentile = None
        if chronological_age is not None and gap is not None:
            from scipy.stats import norm
            percentile = int(norm.cdf(gap, 0, 8) * 100)

        return {
            "predicted_age": round(predicted_age, 1),
            "brain_age_gap": gap,
            "gap_interpretation": gap_interpretation,
            "percentile": percentile,
            "confidence": round(confidence, 3),
            "aperiodic_exponent": round(exponent, 3),
            "aperiodic_offset": round(float(feats[1]), 3),
            "aperiodic_r2": round(r2, 3),
            "alpha_power": round(float(feats[5]), 4),
            "beta_power": round(float(feats[6]), 4),
            "delta_power": round(float(feats[3]), 4),
            "disclaimer": DISCLAIMER,
            "model_type": "aperiodic_heuristic" if self._lgbm is None else "lgbm",
        }


_instance: Optional[BrainAgeEstimator] = None


def get_brain_age_estimator() -> BrainAgeEstimator:
    global _instance
    if _instance is None:
        import os
        model_path = os.path.join(
            os.path.dirname(__file__), "saved", "brain_age_lgbm.pkl"
        )
        _instance = BrainAgeEstimator(model_path if os.path.exists(model_path) else None)
    return _instance
