"""Chronic pain biomarker detection from frontal EEG asymmetry.

Uses resting-state frontal beta/alpha asymmetry as passive pain biomarkers.
No pain stimulus needed — detects patterns during any EEG session.

Reference: Scientific Reports (2024) — frontal Fp1/Fp2 biomarkers:
  beta asymmetry (r=-0.375), gamma asymmetry (r=-0.433),
  low-beta to low-gamma coupling (r=-0.397)

Caveat: This is a screening indicator, NOT a medical diagnosis.
"""
import numpy as np
from typing import Dict


class PainDetector:
    """Detect chronic pain biomarkers from frontal EEG asymmetry."""

    def predict(self, signals, fs: int = 256) -> Dict:
        """Estimate pain biomarker score from multichannel EEG.

        Args:
            signals: (n_channels, n_samples) EEG array
                     Muse 2 order: [TP9, AF7, AF8, TP10]
            fs: sampling rate

        Returns:
            Dict with pain_biomarker_score, components, pain_level, disclaimer
        """
        from processing.eeg_processor import extract_band_powers, differential_entropy

        if signals.ndim == 1:
            return self._no_multichannel_result()

        n_ch = signals.shape[0]
        if n_ch < 2:
            return self._no_multichannel_result()

        # Extract band powers for AF7 (ch1) and AF8 (ch2)
        af7_idx = 1 if n_ch >= 3 else 0
        af8_idx = 2 if n_ch >= 3 else min(1, n_ch - 1)

        af7_bands = extract_band_powers(signals[af7_idx], fs)
        af8_bands = extract_band_powers(signals[af8_idx], fs)

        # Pain biomarker 1: Beta asymmetry (r=-0.375 with pain severity)
        af7_beta = af7_bands.get('beta', 0)
        af8_beta = af8_bands.get('beta', 0)
        beta_asymmetry = af8_beta - af7_beta

        # Pain biomarker 2: Alpha DE asymmetry
        af7_de = differential_entropy(signals[af7_idx], fs)
        af8_de = differential_entropy(signals[af8_idx], fs)
        af7_alpha_de = af7_de.get('alpha', 0.0)
        af8_alpha_de = af8_de.get('alpha', 0.0)
        alpha_de_asymmetry = af8_alpha_de - af7_alpha_de

        # Pain biomarker 3: High-beta power (pain increases high-beta)
        af7_high_beta = af7_bands.get('high_beta', af7_bands.get('beta', 0) * 0.5)
        af8_high_beta = af8_bands.get('high_beta', af8_bands.get('beta', 0) * 0.5)
        high_beta_mean = (af7_high_beta + af8_high_beta) / 2.0

        # Composite pain score (higher = more pain-related patterns)
        # Normalized components
        beta_asym_score = float(np.clip(abs(beta_asymmetry) * 5.0, 0, 1))
        alpha_de_score = float(np.clip(abs(alpha_de_asymmetry) * 3.0, 0, 1))
        high_beta_score = float(np.clip(high_beta_mean * 4.0, 0, 1))

        pain_score = (0.40 * beta_asym_score +
                      0.35 * alpha_de_score +
                      0.25 * high_beta_score)

        # Pain level classification
        if pain_score < 0.3:
            pain_level = 'none_detected'
        elif pain_score < 0.5:
            pain_level = 'mild_indicators'
        elif pain_score < 0.7:
            pain_level = 'moderate_indicators'
        else:
            pain_level = 'elevated_indicators'

        return {
            'pain_biomarker_score': float(pain_score),
            'pain_level': pain_level,
            'components': {
                'beta_asymmetry': float(beta_asymmetry),
                'beta_asymmetry_score': beta_asym_score,
                'alpha_de_asymmetry': float(alpha_de_asymmetry),
                'alpha_de_score': alpha_de_score,
                'high_beta_mean': float(high_beta_mean),
                'high_beta_score': high_beta_score,
            },
            'model_type': 'pain_biomarker',
            'disclaimer': 'Wellness indicator only — not a medical device or clinical assessment'
        }

    def _no_multichannel_result(self) -> Dict:
        return {
            'pain_biomarker_score': 0.0,
            'pain_level': 'insufficient_data',
            'components': {},
            'model_type': 'pain_biomarker',
            'disclaimer': 'Requires multichannel EEG (AF7+AF8) for pain biomarkers'
        }
