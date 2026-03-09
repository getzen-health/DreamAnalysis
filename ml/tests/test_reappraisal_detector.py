"""Tests for ReappraisalDetector — emotion regulation vs genuine experience.

Covers:
  - Instantiation and defaults
  - predict() with 4-channel (4, 512) input
  - predict() with single-channel (512,) input
  - Output key presence and type contracts
  - Value range constraints
  - regulation_state membership in REGULATION_STATES
  - cognitive_control_level membership in expected set
  - genuine_probability + regulation_probability sum ≈ 1.0
  - Frontal theta power is positive
  - update_baseline() runs without error
  - High frontal theta input produces higher regulation_index than low theta
  - Edge cases: very short signal (64 samples), all-zeros, 2-channel input
  - Singleton factory returns same object for same user_id
  - Different user_ids get different singleton instances
"""

import math
import numpy as np
import pytest

from models.reappraisal_detector import ReappraisalDetector, get_reappraisal_detector, REGULATION_STATES

_VALID_STATES = set(REGULATION_STATES)
_VALID_CONTROL_LEVELS = {"low", "medium", "high"}

FS = 256.0


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def detector():
    return ReappraisalDetector()


@pytest.fixture
def eeg_4ch():
    """4-channel × 2 seconds (512 samples) of synthetic EEG."""
    np.random.seed(0)
    return np.random.randn(4, 512) * 20.0


@pytest.fixture
def eeg_1ch():
    """Single-channel 2 seconds (512 samples) of synthetic EEG."""
    np.random.seed(1)
    return np.random.randn(512) * 20.0


# ── 1. Instantiation ──────────────────────────────────────────────────────────

class TestInstantiation:
    def test_default_construction(self):
        d = ReappraisalDetector()
        assert d.n_channels == 4
        assert abs(d.fs - 256.0) < 1e-9
        assert d.model_type == "feature-based"

    def test_custom_fs(self):
        d = ReappraisalDetector(fs=128.0)
        assert abs(d.fs - 128.0) < 1e-9

    def test_baseline_initially_none(self):
        d = ReappraisalDetector()
        assert d._baseline_lpp is None
        assert d._baseline_theta is None


# ── 2. predict() with 4-channel input ─────────────────────────────────────────

class TestPredictMultichannel:
    def test_returns_dict(self, detector, eeg_4ch):
        result = detector.predict(eeg_4ch, FS)
        assert isinstance(result, dict)

    def test_required_keys_present(self, detector, eeg_4ch):
        result = detector.predict(eeg_4ch, FS)
        expected_keys = {
            "regulation_state",
            "regulation_index",
            "lpp_amplitude",
            "frontal_theta_power",
            "genuine_probability",
            "regulation_probability",
            "cognitive_control_level",
            "model_type",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

    def test_regulation_state_valid(self, detector, eeg_4ch):
        result = detector.predict(eeg_4ch, FS)
        assert result["regulation_state"] in _VALID_STATES

    def test_regulation_index_in_range(self, detector, eeg_4ch):
        result = detector.predict(eeg_4ch, FS)
        assert 0.0 <= result["regulation_index"] <= 1.0

    def test_lpp_amplitude_is_float(self, detector, eeg_4ch):
        result = detector.predict(eeg_4ch, FS)
        assert isinstance(result["lpp_amplitude"], float)

    def test_frontal_theta_power_positive(self, detector, eeg_4ch):
        result = detector.predict(eeg_4ch, FS)
        assert result["frontal_theta_power"] > 0.0

    def test_probabilities_sum_to_one(self, detector, eeg_4ch):
        result = detector.predict(eeg_4ch, FS)
        total = result["genuine_probability"] + result["regulation_probability"]
        assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}, expected ~1.0"

    def test_genuine_probability_in_range(self, detector, eeg_4ch):
        result = detector.predict(eeg_4ch, FS)
        assert 0.0 <= result["genuine_probability"] <= 1.0

    def test_regulation_probability_in_range(self, detector, eeg_4ch):
        result = detector.predict(eeg_4ch, FS)
        assert 0.0 <= result["regulation_probability"] <= 1.0

    def test_cognitive_control_level_valid(self, detector, eeg_4ch):
        result = detector.predict(eeg_4ch, FS)
        assert result["cognitive_control_level"] in _VALID_CONTROL_LEVELS

    def test_model_type_is_feature_based(self, detector, eeg_4ch):
        result = detector.predict(eeg_4ch, FS)
        assert result["model_type"] == "feature-based"


# ── 3. predict() with single-channel input ────────────────────────────────────

class TestPredictSingleChannel:
    def test_returns_dict(self, detector, eeg_1ch):
        result = detector.predict(eeg_1ch, FS)
        assert isinstance(result, dict)

    def test_regulation_state_valid(self, detector, eeg_1ch):
        result = detector.predict(eeg_1ch, FS)
        assert result["regulation_state"] in _VALID_STATES

    def test_regulation_index_in_range(self, detector, eeg_1ch):
        result = detector.predict(eeg_1ch, FS)
        assert 0.0 <= result["regulation_index"] <= 1.0

    def test_frontal_theta_positive(self, detector, eeg_1ch):
        result = detector.predict(eeg_1ch, FS)
        assert result["frontal_theta_power"] > 0.0

    def test_cognitive_control_level_valid(self, detector, eeg_1ch):
        result = detector.predict(eeg_1ch, FS)
        assert result["cognitive_control_level"] in _VALID_CONTROL_LEVELS


# ── 4. update_baseline() ─────────────────────────────────────────────────────

class TestUpdateBaseline:
    def test_update_baseline_no_error_multichannel(self, detector, eeg_4ch):
        detector.update_baseline(eeg_4ch, FS)
        assert detector._baseline_lpp is not None
        assert detector._baseline_theta is not None

    def test_update_baseline_no_error_single_channel(self, detector, eeg_1ch):
        detector.update_baseline(eeg_1ch, FS)
        assert detector._baseline_lpp is not None

    def test_predict_after_baseline_stays_valid(self, detector, eeg_4ch):
        detector.update_baseline(eeg_4ch, FS)
        result = detector.predict(eeg_4ch, FS)
        assert result["regulation_state"] in _VALID_STATES
        assert 0.0 <= result["regulation_index"] <= 1.0


# ── 5. High frontal theta → higher regulation_index ──────────────────────────

class TestHighThetaEffect:
    def test_high_theta_input_higher_regulation_index(self):
        """Signal dominated by 4-8 Hz (theta) should score higher regulation
        than a signal with very little theta."""
        d = ReappraisalDetector()
        t = np.arange(1024) / FS

        # High theta: pure 6 Hz sine × 4 channels
        theta_dominant = np.outer(np.ones(4), 50.0 * np.sin(2 * math.pi * 6.0 * t))

        # Low theta: pure 15 Hz beta (minimal theta power)
        beta_dominant = np.outer(np.ones(4), 50.0 * np.sin(2 * math.pi * 15.0 * t))

        res_theta = d.predict(theta_dominant, FS)
        res_beta = d.predict(beta_dominant, FS)

        assert res_theta["regulation_index"] >= res_beta["regulation_index"], (
            f"Expected theta-dominant to have higher regulation_index "
            f"({res_theta['regulation_index']}) than beta-dominant "
            f"({res_beta['regulation_index']})"
        )


# ── 6. Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_very_short_signal_64_samples(self, detector):
        """Very short signal should not raise — returns valid dict."""
        short = np.random.randn(64) * 10.0
        result = detector.predict(short, FS)
        assert isinstance(result, dict)
        assert result["regulation_state"] in _VALID_STATES
        assert 0.0 <= result["regulation_index"] <= 1.0

    def test_all_zeros_multichannel(self, detector):
        """All-zero signal should not raise and should return valid output."""
        zeros = np.zeros((4, 512))
        result = detector.predict(zeros, FS)
        assert isinstance(result, dict)
        assert result["regulation_state"] in _VALID_STATES
        assert 0.0 <= result["regulation_index"] <= 1.0

    def test_all_zeros_single_channel(self, detector):
        zeros = np.zeros(512)
        result = detector.predict(zeros, FS)
        assert isinstance(result, dict)
        assert 0.0 <= result["regulation_index"] <= 1.0

    def test_two_channel_input(self, detector):
        """2-channel input (non-standard) should gracefully fall back."""
        two_ch = np.random.randn(2, 512) * 20.0
        result = detector.predict(two_ch, FS)
        assert isinstance(result, dict)
        assert result["regulation_state"] in _VALID_STATES


# ── 7. Singleton factory ──────────────────────────────────────────────────────

class TestSingletonFactory:
    def test_same_user_id_returns_same_instance(self):
        d1 = get_reappraisal_detector("user_singleton_test")
        d2 = get_reappraisal_detector("user_singleton_test")
        assert d1 is d2

    def test_different_user_ids_return_different_instances(self):
        da = get_reappraisal_detector("user_alpha_99")
        db = get_reappraisal_detector("user_beta_99")
        assert da is not db

    def test_default_user_is_reappraisal_detector(self):
        d = get_reappraisal_detector()
        assert isinstance(d, ReappraisalDetector)
