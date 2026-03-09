"""Tinnitus Severity Assessment via EEG Biomarkers.

Estimates tinnitus severity from temporal-channel EEG using two established
biomarkers:

1. **Alpha reduction at temporal sites** (TP9/TP10):
   Tinnitus patients consistently show reduced alpha (8-12 Hz) power over
   auditory cortex relative to healthy controls.
   Reference: Weisz et al. (2005), Lorenz et al. (2009).

2. **Gamma elevation at temporal sites** (TP9/TP10):
   Increased gamma (30-45 Hz) activity over auditory cortex correlates with
   tinnitus loudness and distress.
   Reference: Weisz et al. (2007), van der Loo et al. (2009).

   **EMG caveat**: On Muse 2, TP9/TP10 electrodes sit near the temporalis
   muscle. Jaw clenching injects broadband EMG artifact into 20-100 Hz,
   contaminating gamma measurements. Gamma-based tinnitus indicators from
   dry-electrode consumer EEG should be interpreted with caution. Clinical
   assessment requires medical-grade equipment.

Severity levels are heuristic thresholds — this is NOT a clinical diagnostic
tool.

MEDICAL DISCLAIMER: This is not a medical diagnosis.
Consult an audiologist for clinical assessment.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Dict, Optional

# NumPy 2.0 renamed np.trapz -> np.trapezoid; 1.x only has np.trapz
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

MEDICAL_DISCLAIMER = (
    "This is not a medical diagnosis. "
    "Consult an audiologist for clinical assessment."
)

# Band definitions for tinnitus biomarkers
_ALPHA_BAND = (8.0, 12.0)
_GAMMA_BAND = (30.0, 45.0)

# Severity thresholds on severity_index (0-1)
_SEVERITY_LEVELS = {
    "none_detected": (0.0, 0.2),
    "mild_indicators": (0.2, 0.4),
    "moderate_indicators": (0.4, 0.6),
    "elevated_indicators": (0.6, 1.0),
}

# Population-average estimates (relative band power, unitless) used when
# no per-user baseline has been recorded.  Derived from published normative
# temporal-alpha/gamma values for healthy adults with eyes-open resting EEG.
_DEFAULT_BASELINE_ALPHA = 0.25
_DEFAULT_BASELINE_GAMMA = 0.04


def _compute_psd(signal_1d: np.ndarray, fs: float):
    """Welch PSD matching the pattern in eeg_processor.py."""
    nperseg = min(len(signal_1d), int(fs * 2))
    return scipy_signal.welch(signal_1d, fs=fs, nperseg=nperseg)


def _band_relative_power(
    freqs: np.ndarray, psd: np.ndarray, band: tuple
) -> float:
    """Relative band power as fraction of total power."""
    total = _trapezoid(psd, freqs)
    if total <= 0:
        return 0.0
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0
    return float(_trapezoid(psd[mask], freqs[mask]) / total)


class TinnitusAssessor:
    """EEG-based tinnitus severity estimator.

    Uses alpha reduction and gamma elevation at temporal sites (TP9/TP10,
    Muse 2 channels 0 and 3) relative to a resting-state baseline.

    Workflow:
        1. (Optional) ``set_baseline(signals, fs)`` during 2-3 min
           eyes-closed resting state.
        2. ``assess(signals, fs)`` during live recording or on stored epochs.

    If no baseline is set, population-average estimates are used, which
    will be less accurate.
    """

    def __init__(self) -> None:
        # Per-user baselines: user_id -> {"alpha": float, "gamma": float}
        self._baselines: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: float = 256.0,
        user_id: str = "default",
    ) -> Dict:
        """Record resting-state baseline alpha/gamma at temporal sites.

        Args:
            signals: EEG array. Accepted shapes:
                - (n_channels, n_samples) with n_channels >= 4: uses TP9
                  (ch0) and TP10 (ch3).
                - (n_channels, n_samples) with n_channels < 4: averages
                  all channels.
                - (n_samples,): single channel treated as temporal proxy.
            fs: Sampling rate in Hz.
            user_id: Identifier for per-user baselines.

        Returns:
            Dict with recorded baseline values and confirmation.
        """
        alpha_values, gamma_values = self._extract_temporal_powers(signals, fs)

        baseline = {
            "alpha": float(np.mean(alpha_values)),
            "gamma": float(np.mean(gamma_values)),
        }
        self._baselines[user_id] = baseline

        return {
            "status": "baseline_set",
            "user_id": user_id,
            "baseline_alpha": round(baseline["alpha"], 6),
            "baseline_gamma": round(baseline["gamma"], 6),
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    def reset_baseline(self, user_id: str = "default") -> Dict:
        """Clear stored baseline for a user.

        Args:
            user_id: User whose baseline should be cleared.

        Returns:
            Status dict.
        """
        removed = self._baselines.pop(user_id, None)
        return {
            "status": "baseline_reset",
            "user_id": user_id,
            "had_baseline": removed is not None,
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    def has_baseline(self, user_id: str = "default") -> bool:
        """Check whether a baseline exists for the given user."""
        return user_id in self._baselines

    # ------------------------------------------------------------------
    # Assessment
    # ------------------------------------------------------------------

    def assess(
        self,
        signals: np.ndarray,
        fs: float = 256.0,
        user_id: str = "default",
    ) -> Dict:
        """Compute tinnitus severity relative to baseline.

        Args:
            signals: EEG array (same shapes accepted as ``set_baseline``).
            fs: Sampling rate in Hz.
            user_id: User whose baseline to compare against.

        Returns:
            Dict containing severity classification, biomarker scores,
            component details, and medical disclaimer.
        """
        alpha_values, gamma_values = self._extract_temporal_powers(signals, fs)

        current_alpha = float(np.mean(alpha_values))
        current_gamma = float(np.mean(gamma_values))

        # Retrieve or fall back to population-average baseline
        baseline = self._baselines.get(user_id)
        using_baseline = baseline is not None
        if baseline is None:
            baseline = {
                "alpha": _DEFAULT_BASELINE_ALPHA,
                "gamma": _DEFAULT_BASELINE_GAMMA,
            }

        base_alpha = baseline["alpha"]
        base_gamma = baseline["gamma"]

        # --- Alpha reduction ---
        # Higher value = more reduction relative to baseline = worse.
        # alpha_reduction in [0, 1].
        if base_alpha > 0:
            alpha_ratio = current_alpha / base_alpha
            alpha_reduction = float(np.clip(1.0 - alpha_ratio, 0.0, 1.0))
        else:
            alpha_reduction = 0.0

        # --- Gamma elevation ---
        # Higher value = more elevation relative to baseline = worse.
        # gamma_elevation in [0, 1].
        if base_gamma > 0:
            gamma_ratio = current_gamma / base_gamma
            # tanh saturates gently; factor of 0.5 keeps scale reasonable
            gamma_elevation = float(
                np.clip(np.tanh((gamma_ratio - 1.0) * 0.5), 0.0, 1.0)
            )
        else:
            gamma_elevation = 0.0

        # --- Severity index ---
        severity_index = float(
            np.clip(0.5 * alpha_reduction + 0.5 * gamma_elevation, 0.0, 1.0)
        )

        # --- Classify severity ---
        severity = "none_detected"
        for level, (lo, hi) in _SEVERITY_LEVELS.items():
            if lo <= severity_index < hi:
                severity = level
                break
        # Handle exact 1.0
        if severity_index >= 0.6:
            severity = "elevated_indicators"

        return {
            "severity": severity,
            "severity_index": round(severity_index, 4),
            "alpha_reduction": round(alpha_reduction, 4),
            "gamma_elevation": round(gamma_elevation, 4),
            "current_alpha": round(current_alpha, 6),
            "current_gamma": round(current_gamma, 6),
            "baseline_alpha": round(base_alpha, 6),
            "baseline_gamma": round(base_gamma, 6),
            "using_personal_baseline": using_baseline,
            "gamma_emg_caveat": (
                "Gamma power at TP9/TP10 may be contaminated by temporalis "
                "muscle (EMG) artifact on consumer-grade EEG devices. "
                "Interpret gamma-based indicators with caution."
            ),
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_temporal_powers(self, signals: np.ndarray, fs: float):
        """Extract alpha and gamma relative powers from temporal channels.

        Returns:
            (alpha_values, gamma_values) — each a list of per-channel
            relative powers.
        """
        if signals.ndim == 1:
            # Single channel — treat as temporal proxy
            channels = [signals]
        elif signals.ndim == 2:
            if signals.shape[0] >= 4:
                # Standard Muse 2 layout: TP9=ch0, TP10=ch3
                channels = [signals[0], signals[3]]
            else:
                # Fewer than 4 channels — use all available
                channels = [signals[i] for i in range(signals.shape[0])]
        else:
            raise ValueError(
                f"Expected 1D or 2D signal array, got {signals.ndim}D"
            )

        alpha_values = []
        gamma_values = []
        for ch in channels:
            freqs, psd = _compute_psd(ch, fs)
            alpha_values.append(_band_relative_power(freqs, psd, _ALPHA_BAND))
            gamma_values.append(_band_relative_power(freqs, psd, _GAMMA_BAND))

        return alpha_values, gamma_values
