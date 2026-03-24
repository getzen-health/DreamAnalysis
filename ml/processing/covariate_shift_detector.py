"""Runtime covariate shift detection for EEG feature distributions.

Detects when live EEG features have drifted from the training/baseline
distribution. When shift is detected, produces a confidence penalty factor
and a recalibration recommendation.

Method: Per-feature two-sample Kolmogorov-Smirnov test comparing a sliding
window of recent live features against a stored reference distribution.

Evidence:
- Rabanser et al., NeurIPS 2019 ("Failing Loudly: An Empirical Study of
  Methods for Detecting Dataset Shift"): two-sample tests on features with
  dimensionality reduction are the most effective approach. KS test is the
  simplest and most interpretable option for per-feature monitoring.
- EEG non-stationarity is well-documented: alpha peak frequency drops with
  time-on-task (NeuroImage 2019), cross-day variability degrades accuracy,
  electrode impedance drifts within sessions.

Usage:
    detector = CovariateShiftDetector(n_features=85)

    # During baseline/calibration period:
    for epoch_features in baseline_epochs:
        detector.add_reference(epoch_features)

    # During live inference:
    detector.add_live(live_features)
    result = detector.detect()
    if result["shift_detected"]:
        confidence *= result["confidence_penalty"]
"""

import numpy as np
from collections import deque
from scipy.stats import ks_2samp
from typing import Dict, List, Optional


class CovariateShiftDetector:
    """Detects covariate shift between reference and live feature distributions.

    Maintains two sliding windows:
    - Reference window: features from baseline/training (stable distribution)
    - Live window: recent features from inference (potentially drifted)

    Uses per-feature KS test to detect shift. Reports:
    - Whether shift is detected (Bonferroni-corrected p-value < alpha)
    - Which features have shifted
    - A confidence penalty factor in [0.5, 1.0]
    - Recalibration recommendation when appropriate

    Args:
        n_features: Number of features in each sample (85 for LGBM, 68 for enhanced).
        reference_window: Max samples to keep in reference buffer.
        live_window: Max samples to keep in live buffer.
        alpha: Significance level for KS test (before Bonferroni correction).
        min_reference: Minimum reference samples before detection activates.
        min_live: Minimum live samples before detection activates.
    """

    def __init__(
        self,
        n_features: int = 85,
        reference_window: int = 50,
        live_window: int = 20,
        alpha: float = 0.05,
        min_reference: int = 30,
        min_live: int = 10,
    ):
        self.n_features = n_features
        self.reference_window = reference_window
        self.live_window = live_window
        self.alpha = alpha
        self.min_reference = min_reference
        self.min_live = min_live

        # Buffers — deques for automatic oldest-out behavior
        self._reference: deque = deque(maxlen=reference_window)
        self._live: deque = deque(maxlen=live_window)

    @property
    def reference_count(self) -> int:
        """Number of reference samples currently stored."""
        return len(self._reference)

    @property
    def live_count(self) -> int:
        """Number of live samples currently stored."""
        return len(self._live)

    @property
    def is_ready(self) -> bool:
        """Whether enough reference data has been collected for detection."""
        return self.reference_count >= self.min_reference

    def add_reference(self, features: np.ndarray) -> None:
        """Add a feature vector to the reference distribution.

        Call during baseline calibration or load from training statistics.

        Args:
            features: 1D array of shape (n_features,).
        """
        features = np.asarray(features, dtype=np.float64).ravel()
        self._reference.append(features)

    def add_live(self, features: np.ndarray) -> None:
        """Add a feature vector to the live sliding window.

        Call after each epoch's features are extracted during inference.

        Args:
            features: 1D array of shape (n_features,).
        """
        features = np.asarray(features, dtype=np.float64).ravel()
        self._live.append(features)

    def detect(self) -> Dict:
        """Run covariate shift detection.

        Returns:
            Dict with keys:
            - shift_detected: bool
            - status: "collecting" | "monitoring" | "shift_detected"
            - mean_ks_statistic: float (average KS stat across features)
            - fraction_shifted: float (fraction of features with p < alpha)
            - shifted_feature_indices: List[int]
            - confidence_penalty: float in [0.5, 1.0]
            - recommendation: Optional[str]
        """
        # Not enough data yet
        if self.reference_count < self.min_reference or self.live_count < self.min_live:
            return {
                "shift_detected": False,
                "status": "collecting",
                "mean_ks_statistic": 0.0,
                "fraction_shifted": 0.0,
                "shifted_feature_indices": [],
                "confidence_penalty": 1.0,
                "recommendation": None,
            }

        ref_array = np.array(list(self._reference))  # (n_ref, n_features)
        live_array = np.array(list(self._live))       # (n_live, n_features)

        n_feat = min(ref_array.shape[1], live_array.shape[1])

        # Per-feature KS test with Bonferroni correction
        bonferroni_alpha = self.alpha / n_feat
        ks_stats = []
        shifted_indices: List[int] = []

        for i in range(n_feat):
            ref_col = ref_array[:, i]
            live_col = live_array[:, i]

            # Skip constant features (no variance = no meaningful test)
            if np.std(ref_col) < 1e-10 and np.std(live_col) < 1e-10:
                ks_stats.append(0.0)
                continue

            stat, p_value = ks_2samp(ref_col, live_col)
            ks_stats.append(stat)

            if p_value < bonferroni_alpha:
                shifted_indices.append(i)

        mean_ks = float(np.mean(ks_stats))
        fraction_shifted = len(shifted_indices) / n_feat if n_feat > 0 else 0.0

        # Shift is detected if more than 10% of features have shifted
        # (Bonferroni-corrected). This threshold prevents false alarms from
        # random fluctuations in a few features.
        shift_detected = fraction_shifted > 0.10

        # Confidence penalty: maps fraction_shifted to [0.5, 1.0]
        # No shift -> 1.0, 100% shifted -> 0.5
        # Linear mapping: penalty = 1.0 - 0.5 * fraction_shifted
        confidence_penalty = float(np.clip(1.0 - 0.5 * fraction_shifted, 0.5, 1.0))

        # Recommendation
        recommendation = None
        if shift_detected:
            recommendation = (
                "Recalibration recommended: input feature distribution has shifted "
                f"({fraction_shifted:.0%} of features affected, "
                f"mean KS statistic = {mean_ks:.3f}). "
                "This may be caused by electrode repositioning, time-of-day changes, "
                "or environmental differences."
            )

        status = "shift_detected" if shift_detected else "monitoring"

        return {
            "shift_detected": shift_detected,
            "status": status,
            "mean_ks_statistic": mean_ks,
            "fraction_shifted": fraction_shifted,
            "shifted_feature_indices": shifted_indices,
            "confidence_penalty": confidence_penalty,
            "recommendation": recommendation,
        }

    def reset_live(self) -> None:
        """Clear the live window (e.g., after recalibration)."""
        self._live.clear()

    def reset_all(self) -> None:
        """Clear both reference and live windows."""
        self._reference.clear()
        self._live.clear()

    def get_status(self) -> Dict:
        """Get current detector status without running detection."""
        return {
            "reference_count": self.reference_count,
            "live_count": self.live_count,
            "is_ready": self.is_ready,
            "reference_window": self.reference_window,
            "live_window": self.live_window,
        }
