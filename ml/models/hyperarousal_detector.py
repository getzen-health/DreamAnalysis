"""PTSD/trauma hyperarousal detection from frontal EEG.

Based on: Scientific Reports (2024) -- Shannon entropy at AF3/AF4
negatively correlates with PTSD severity (r=-0.43 to -0.47).
AF3/AF4 maps to Muse 2's AF7/AF8.

Lower Shannon entropy + lower frontal alpha = higher hyperarousal risk.
The hyperarousal index combines three components:
  1. Frontal Shannon entropy (40%) -- time-domain complexity measure
  2. AF8 alpha power (35%) -- suppressed alpha = reduced inhibitory tone
  3. Spectral entropy (25%) -- frequency-domain regularity

All thresholds are population-average heuristics. Per-user baseline
calibration (via BaselineCalibrator) would improve accuracy significantly.
"""

import numpy as np
from typing import Dict

from processing.eeg_processor import extract_band_powers, preprocess, spectral_entropy


# Muse 2 channel order: TP9=0, AF7=1, AF8=2, TP10=3
_CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]

# Typical resting-state alpha relative power (population average).
# Used to normalize the alpha suppression component.
_RESTING_ALPHA = 0.3


def compute_shannon_entropy(signal: np.ndarray, n_bins: int = 64) -> float:
    """Time-domain Shannon entropy via histogram binning.

    Normalizes by log2(n_bins) so the result falls in [0, 1].
    Lower values indicate more regular/constrained signal dynamics,
    which correlates with PTSD hyperarousal severity.

    Args:
        signal: 1-D EEG time series (any units).
        n_bins: Number of histogram bins for amplitude distribution.

    Returns:
        Normalized Shannon entropy in [0, 1].
    """
    if len(signal) < 2:
        return 0.5  # insufficient data -- return neutral

    hist, _ = np.histogram(signal, bins=n_bins, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.5

    max_entropy = np.log2(n_bins)
    if max_entropy <= 0:
        return 0.5

    return float(-np.sum(hist * np.log2(hist + 1e-10)) / max_entropy)


class HyperarousalDetector:
    """Detect chronic hyperarousal from frontal EEG entropy + alpha.

    Designed for 4-channel Muse 2 (TP9, AF7, AF8, TP10).
    Falls back gracefully to single-channel operation.
    """

    def predict(self, signals: np.ndarray, fs: float = 256.0) -> Dict:
        """Compute hyperarousal index from EEG signals.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
                     For Muse 2: (4, n_samples) with TP9, AF7, AF8, TP10.
            fs: Sampling rate in Hz.

        Returns:
            Dict with hyperarousal_index (0-1), risk_level, components,
            and per-channel Shannon entropies.
        """
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        n_ch = min(signals.shape[0], len(_CHANNEL_NAMES))

        # --- Shannon entropy per channel ---
        channel_entropies: Dict[str, float] = {}
        for i in range(n_ch):
            channel_entropies[_CHANNEL_NAMES[i]] = compute_shannon_entropy(signals[i])

        # Frontal entropy: average of AF7 (ch1) and AF8 (ch2) when available
        if n_ch >= 3:
            frontal_entropy = float(np.mean([
                channel_entropies.get("AF7", 0.5),
                channel_entropies.get("AF8", 0.5),
            ]))
        else:
            # Single channel -- use whatever is available
            frontal_entropy = float(np.mean(list(channel_entropies.values())))

        # --- Spectral entropy (average across all channels) ---
        spec_entropies = []
        for ch in range(n_ch):
            processed = preprocess(signals[ch], fs)
            se = spectral_entropy(processed, fs)
            spec_entropies.append(se)
        avg_spectral_entropy = float(np.mean(spec_entropies))

        # --- Alpha power at AF8 (right frontal) ---
        # AF8 alpha negatively correlates with PTSD severity (r=-0.429)
        if n_ch >= 3:
            af8_processed = preprocess(signals[2], fs)
        else:
            af8_processed = preprocess(signals[0], fs)
        af8_bands = extract_band_powers(af8_processed, fs)
        af8_alpha = af8_bands.get("alpha", 0.2)

        # --- Compute component scores ---
        # Each component: higher value = more hyperarousal
        entropy_component = float(np.clip(1.0 - frontal_entropy, 0.0, 1.0))
        alpha_component = float(np.clip(1.0 - af8_alpha / _RESTING_ALPHA, 0.0, 1.0))
        spectral_component = float(np.clip(1.0 - avg_spectral_entropy, 0.0, 1.0))

        # --- Weighted combination ---
        hyperarousal_index = float(np.clip(
            0.40 * entropy_component
            + 0.35 * alpha_component
            + 0.25 * spectral_component,
            0.0, 1.0,
        ))

        # --- Risk level ---
        if hyperarousal_index >= 0.7:
            risk_level = "high"
        elif hyperarousal_index >= 0.4:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return {
            "hyperarousal_index": round(hyperarousal_index, 3),
            "risk_level": risk_level,
            "components": {
                "frontal_entropy": round(frontal_entropy, 3),
                "af8_alpha_power": round(float(af8_alpha), 4),
                "spectral_entropy": round(avg_spectral_entropy, 3),
                "entropy_component": round(entropy_component, 3),
                "alpha_component": round(alpha_component, 3),
                "spectral_component": round(spectral_component, 3),
            },
            "channel_entropies": {k: round(v, 3) for k, v in channel_entropies.items()},
        }
