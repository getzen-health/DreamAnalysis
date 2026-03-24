"""Tests for runtime covariate shift detection.

Covariate shift = input feature distribution changes between training and
inference. When detected, the system should reduce confidence and surface
a recalibration recommendation.

Method: Two-sample Kolmogorov-Smirnov test on per-feature distributions,
comparing a sliding window of recent live features against a stored
reference distribution (from training or baseline calibration).

Evidence:
- Rabanser et al., NeurIPS 2019 ("Failing Loudly"): two-sample tests on
  features are the most effective way to detect dataset shift.
- EEG non-stationarity is extensively documented: alpha peak frequency drops
  with time-on-task, cross-day accuracy degrades significantly, electrode
  impedance drifts within sessions.
- The CLAUDE.md notes "non-stationarity / session drift degrades over time"
  as a known issue with no existing fix.

Verifies:
1. Detector stores reference features from baseline period
2. Identical distribution -> no shift detected
3. Shifted distribution -> shift detected with high KS statistic
4. Confidence penalty scales with shift severity
5. Recalibration recommendation surfaced when shift detected
6. Sliding window updates correctly (old data drops off)
7. Works with the 85-feature vector used by the LGBM classifier
8. Integrates with ConfidenceCalibrator to reduce confidence
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.covariate_shift_detector import CovariateShiftDetector
from processing.confidence_calibration import ConfidenceCalibrator


@pytest.fixture
def detector():
    """Create a shift detector with default settings."""
    return CovariateShiftDetector(
        n_features=85,
        reference_window=50,
        live_window=20,
        alpha=0.05,
    )


def _make_reference_features(n_samples: int = 60, n_features: int = 85,
                              seed: int = 42) -> np.ndarray:
    """Generate stable reference features (simulating baseline/training)."""
    rng = np.random.RandomState(seed)
    # Realistic EEG feature range: mean ~0, std ~1 (after z-scoring)
    return rng.randn(n_samples, n_features)


def _make_shifted_features(n_samples: int = 30, n_features: int = 85,
                            seed: int = 99, shift_magnitude: float = 3.0) -> np.ndarray:
    """Generate features with a clear distribution shift."""
    rng = np.random.RandomState(seed)
    # Shift mean by shift_magnitude standard deviations
    return rng.randn(n_samples, n_features) + shift_magnitude


class TestReferenceCollection:
    """Test that the detector correctly stores reference features."""

    def test_add_reference_features(self, detector):
        """Reference features should accumulate up to reference_window."""
        features = _make_reference_features(n_samples=10)
        for row in features:
            detector.add_reference(row)

        assert detector.reference_count == 10

    def test_reference_window_caps_at_limit(self, detector):
        """Reference should not exceed reference_window size."""
        features = _make_reference_features(n_samples=100)
        for row in features:
            detector.add_reference(row)

        assert detector.reference_count == detector.reference_window

    def test_is_ready_after_enough_reference(self, detector):
        """Detector should report ready once minimum reference collected."""
        features = _make_reference_features(n_samples=50)
        for row in features:
            detector.add_reference(row)

        assert detector.is_ready


class TestShiftDetection:
    """Test the core shift detection logic."""

    def test_no_shift_with_identical_distribution(self, detector):
        """Same distribution -> no shift detected."""
        ref = _make_reference_features(n_samples=50, seed=42)
        for row in ref:
            detector.add_reference(row)

        # Live features from same distribution
        live = _make_reference_features(n_samples=20, seed=43)
        for row in live:
            detector.add_live(row)

        result = detector.detect()
        assert not result["shift_detected"]

    def test_shift_with_shifted_distribution(self, detector):
        """Clearly shifted distribution -> shift detected."""
        ref = _make_reference_features(n_samples=50, seed=42)
        for row in ref:
            detector.add_reference(row)

        # Live features from shifted distribution
        shifted = _make_shifted_features(n_samples=20, shift_magnitude=3.0)
        for row in shifted:
            detector.add_live(row)

        result = detector.detect()
        assert result["shift_detected"]
        assert result["mean_ks_statistic"] > 0.3

    def test_not_ready_without_enough_data(self, detector):
        """Should return not-ready when insufficient data collected."""
        ref = _make_reference_features(n_samples=5)
        for row in ref:
            detector.add_reference(row)

        result = detector.detect()
        assert not result["shift_detected"]
        assert result["status"] == "collecting"


class TestConfidencePenalty:
    """Test that shift severity maps to an appropriate confidence penalty."""

    def test_no_penalty_when_no_shift(self, detector):
        """No shift -> penalty factor is 1.0 (no reduction)."""
        ref = _make_reference_features(n_samples=50, seed=42)
        for row in ref:
            detector.add_reference(row)

        live = _make_reference_features(n_samples=20, seed=43)
        for row in live:
            detector.add_live(row)

        result = detector.detect()
        assert result["confidence_penalty"] >= 0.9  # near 1.0

    def test_large_penalty_for_severe_shift(self, detector):
        """Severe shift -> large penalty (low multiplier)."""
        ref = _make_reference_features(n_samples=50, seed=42)
        for row in ref:
            detector.add_reference(row)

        shifted = _make_shifted_features(n_samples=20, shift_magnitude=5.0)
        for row in shifted:
            detector.add_live(row)

        result = detector.detect()
        assert result["confidence_penalty"] < 0.7  # significant reduction
        assert result["shift_detected"]

    def test_penalty_between_0_and_1(self, detector):
        """Penalty should always be in [0.5, 1.0] range."""
        ref = _make_reference_features(n_samples=50, seed=42)
        for row in ref:
            detector.add_reference(row)

        shifted = _make_shifted_features(n_samples=20, shift_magnitude=10.0)
        for row in shifted:
            detector.add_live(row)

        result = detector.detect()
        assert 0.5 <= result["confidence_penalty"] <= 1.0


class TestRecalibrationRecommendation:
    """Test that shift detection surfaces actionable recommendations."""

    def test_recommends_recalibration_on_shift(self, detector):
        """Should recommend recalibration when shift detected."""
        ref = _make_reference_features(n_samples=50, seed=42)
        for row in ref:
            detector.add_reference(row)

        shifted = _make_shifted_features(n_samples=20, shift_magnitude=3.0)
        for row in shifted:
            detector.add_live(row)

        result = detector.detect()
        assert result["shift_detected"]
        assert "recalibration" in result.get("recommendation", "").lower()

    def test_no_recommendation_when_stable(self, detector):
        """Should not recommend recalibration when distribution is stable."""
        ref = _make_reference_features(n_samples=50, seed=42)
        for row in ref:
            detector.add_reference(row)

        live = _make_reference_features(n_samples=20, seed=43)
        for row in live:
            detector.add_live(row)

        result = detector.detect()
        assert result.get("recommendation") is None


class TestSlidingWindow:
    """Test that the live feature window slides correctly."""

    def test_live_window_caps_at_limit(self, detector):
        """Live buffer should not exceed live_window size."""
        ref = _make_reference_features(n_samples=50)
        for row in ref:
            detector.add_reference(row)

        # Add more than live_window samples
        live = _make_reference_features(n_samples=40, seed=99)
        for row in live:
            detector.add_live(row)

        assert detector.live_count == detector.live_window

    def test_recovery_after_shift_corrected(self, detector):
        """If distribution returns to normal, shift should no longer be detected."""
        ref = _make_reference_features(n_samples=50, seed=42)
        for row in ref:
            detector.add_reference(row)

        # First: shifted data
        shifted = _make_shifted_features(n_samples=20, shift_magnitude=5.0)
        for row in shifted:
            detector.add_live(row)
        result1 = detector.detect()
        assert result1["shift_detected"]

        # Then: overwrite with normal data (window slides)
        normal = _make_reference_features(n_samples=20, seed=44)
        for row in normal:
            detector.add_live(row)
        result2 = detector.detect()
        assert not result2["shift_detected"]


class TestFeatureReport:
    """Test per-feature shift reporting."""

    def test_reports_shifted_feature_indices(self, detector):
        """Should identify which specific features have shifted."""
        ref = _make_reference_features(n_samples=50, seed=42)
        for row in ref:
            detector.add_reference(row)

        # Shift only the first 10 features
        live = _make_reference_features(n_samples=20, seed=43)
        live[:, :10] += 5.0  # shift first 10 features severely
        for row in live:
            detector.add_live(row)

        result = detector.detect()
        shifted_indices = result.get("shifted_feature_indices", [])
        # At least some of the first 10 should be flagged
        assert any(i < 10 for i in shifted_indices)

    def test_fraction_shifted_is_reported(self, detector):
        """Should report what fraction of features have shifted."""
        ref = _make_reference_features(n_samples=50, seed=42)
        for row in ref:
            detector.add_reference(row)

        shifted = _make_shifted_features(n_samples=20, shift_magnitude=5.0)
        for row in shifted:
            detector.add_live(row)

        result = detector.detect()
        assert "fraction_shifted" in result
        assert 0.0 <= result["fraction_shifted"] <= 1.0


class TestConfidenceCalibratorIntegration:
    """Test integration between CovariateShiftDetector and ConfidenceCalibrator."""

    def test_calibrator_applies_shift_penalty(self, detector):
        """Calibrator should reduce confidence when shift detector reports shift."""
        # Build reference
        ref = _make_reference_features(n_samples=50, seed=42)
        for row in ref:
            detector.add_reference(row)

        # Add shifted live data
        shifted = _make_shifted_features(n_samples=20, shift_magnitude=5.0)
        for row in shifted:
            detector.add_live(row)

        calibrator = ConfidenceCalibrator()
        calibrator.set_shift_detector(detector)

        # Calibrate with shift detector active
        result_with_shift = calibrator.calibrate("emotion", 0.80)
        # Calibrate without shift detector
        calibrator_plain = ConfidenceCalibrator()
        result_without_shift = calibrator_plain.calibrate("emotion", 0.80)

        # Shift-penalized confidence should be lower
        assert result_with_shift["calibrated_confidence"] < result_without_shift["calibrated_confidence"]

    def test_calibrator_no_penalty_when_stable(self, detector):
        """Calibrator should not reduce confidence when no shift detected."""
        ref = _make_reference_features(n_samples=50, seed=42)
        for row in ref:
            detector.add_reference(row)

        live = _make_reference_features(n_samples=20, seed=43)
        for row in live:
            detector.add_live(row)

        calibrator = ConfidenceCalibrator()
        calibrator.set_shift_detector(detector)

        result_with_detector = calibrator.calibrate("emotion", 0.80)
        calibrator_plain = ConfidenceCalibrator()
        result_without_detector = calibrator_plain.calibrate("emotion", 0.80)

        # No shift -> same confidence (within floating point tolerance)
        assert abs(
            result_with_detector["calibrated_confidence"]
            - result_without_detector["calibrated_confidence"]
        ) < 0.01

    def test_calibrator_includes_shift_info(self, detector):
        """Calibrator should include covariate_shift info when shift detected."""
        ref = _make_reference_features(n_samples=50, seed=42)
        for row in ref:
            detector.add_reference(row)

        shifted = _make_shifted_features(n_samples=20, shift_magnitude=5.0)
        for row in shifted:
            detector.add_live(row)

        calibrator = ConfidenceCalibrator()
        calibrator.set_shift_detector(detector)

        result = calibrator.calibrate("emotion", 0.80)
        assert "covariate_shift" in result
        assert result["covariate_shift"]["shift_detected"]

    def test_explicit_shift_penalty_overrides_detector(self, detector):
        """Explicit shift_penalty arg should override the detector."""
        ref = _make_reference_features(n_samples=50, seed=42)
        for row in ref:
            detector.add_reference(row)

        live = _make_reference_features(n_samples=20, seed=43)
        for row in live:
            detector.add_live(row)

        calibrator = ConfidenceCalibrator()
        calibrator.set_shift_detector(detector)

        # Pass explicit severe penalty even though no shift detected
        result = calibrator.calibrate("emotion", 0.80, shift_penalty=0.5)
        calibrator_plain = ConfidenceCalibrator()
        result_plain = calibrator_plain.calibrate("emotion", 0.80)

        assert result["calibrated_confidence"] < result_plain["calibrated_confidence"]
