"""Empathy and social engagement detector via mu rhythm suppression.

Measures mu rhythm (8-13 Hz alpha) suppression at temporal channels TP9/TP10
as a proxy for empathic engagement. During social observation or emotional
content viewing, mu suppression at sensorimotor-adjacent sites indicates
mirror neuron activation and empathic resonance.

Reference: Experimental Brain Research (2025) — hyperscanning studies show
mu suppression indexes empathic engagement during social interaction.

Caveat: TP9/TP10 are not ideal for mu rhythm (C3/C4 would be better),
but temporal alpha suppression during social observation has been validated
as an empathy proxy in consumer EEG studies.
"""
import numpy as np
from typing import Dict, Optional


class EmpathyDetector:
    """Detect empathic engagement via temporal alpha (mu) suppression."""

    def __init__(self):
        self._baseline_mu: Dict[str, float] = {}  # per-user baseline mu power

    def set_baseline(self, mu_power: float, user_id: str = 'default'):
        """Set resting-state mu power baseline for a user.

        Args:
            mu_power: mean alpha power at TP9+TP10 during resting state
            user_id: user identifier for per-user baselines
        """
        if mu_power > 0:
            self._baseline_mu[user_id] = mu_power

    def predict(self, signals, fs: int = 256, user_id: str = 'default') -> Dict:
        """Estimate empathic engagement from mu rhythm suppression.

        Args:
            signals: (n_channels, n_samples) EEG array.
                     Channels: [TP9, AF7, AF8, TP10] (Muse 2 order)
            fs: sampling rate in Hz
            user_id: user identifier for baseline lookup

        Returns:
            Dict with empathy_index, mu_suppression, social_engagement, etc.
        """
        from processing.eeg_processor import extract_band_powers

        if signals.ndim == 1:
            # Single channel — can't compute temporal mu
            return {
                'empathy_index': 0.0,
                'mu_suppression': 0.0,
                'social_engagement': 'unknown',
                'model_type': 'mu_suppression',
                'note': 'Requires multichannel (TP9+TP10) for mu suppression',
            }

        # Extract alpha power at temporal channels
        # Muse 2: ch0=TP9, ch3=TP10
        tp9_bands = extract_band_powers(signals[0], fs)
        tp10_bands = extract_band_powers(signals[min(3, signals.shape[0] - 1)], fs)

        tp9_alpha = tp9_bands.get('alpha', 0)
        tp10_alpha = tp10_bands.get('alpha', 0)
        current_mu = (tp9_alpha + tp10_alpha) / 2.0

        # Compute suppression relative to baseline
        baseline = self._baseline_mu.get(user_id)
        if baseline and baseline > 0:
            suppression = (baseline - current_mu) / baseline
        else:
            # No baseline — use absolute level as rough proxy
            # Lower mu power generally = more suppression
            suppression = 0.0

        # Empathy index: suppression scaled to 0-1
        empathy_index = float(np.clip(suppression * 2.0, 0.0, 1.0))

        # Engagement level
        if empathy_index > 0.6:
            engagement = 'high'
        elif empathy_index > 0.3:
            engagement = 'moderate'
        else:
            engagement = 'low'

        return {
            'empathy_index': empathy_index,
            'mu_suppression': float(suppression),
            'mu_power_current': float(current_mu),
            'mu_power_baseline': float(baseline) if baseline else None,
            'social_engagement': engagement,
            'model_type': 'mu_suppression',
        }

    def predict_from_features(
        self, features: Dict, user_id: str = 'default'
    ) -> Dict:
        """Estimate empathy from pre-extracted band power features.

        Args:
            features: dict with 'alpha' key (average temporal alpha power)
            user_id: user identifier

        Returns:
            Same dict structure as predict()
        """
        current_mu = features.get('alpha', 0)
        baseline = self._baseline_mu.get(user_id)

        if baseline and baseline > 0:
            suppression = (baseline - current_mu) / baseline
        else:
            suppression = 0.0

        empathy_index = float(np.clip(suppression * 2.0, 0.0, 1.0))

        if empathy_index > 0.6:
            engagement = 'high'
        elif empathy_index > 0.3:
            engagement = 'moderate'
        else:
            engagement = 'low'

        return {
            'empathy_index': empathy_index,
            'mu_suppression': float(suppression),
            'mu_power_current': float(current_mu),
            'mu_power_baseline': float(baseline) if baseline else None,
            'social_engagement': engagement,
            'model_type': 'mu_suppression',
        }
