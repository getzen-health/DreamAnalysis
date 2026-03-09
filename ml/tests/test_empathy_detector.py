"""Tests for EmpathyDetector — mu rhythm suppression at TP9/TP10.

Covers:
  - set_baseline stores value per user
  - With baseline, lower current mu -> higher empathy
  - With baseline, equal current mu -> zero empathy
  - With baseline, higher current mu -> zero (clipped) empathy
  - No baseline -> empathy_index 0.0
  - Social engagement thresholds: high (>0.6), moderate (>0.3), low
  - Single channel input -> returns unknown
  - Multichannel input works with 4 channels
  - predict_from_features() matches predict() logic
  - Per-user baseline isolation
  - Zero/negative baseline rejected (not stored)
  - Return dict has all expected keys
"""

import numpy as np
import pytest

from models.empathy_detector import EmpathyDetector

FS = 256

EXPECTED_KEYS = {
    'empathy_index',
    'mu_suppression',
    'mu_power_current',
    'mu_power_baseline',
    'social_engagement',
    'model_type',
}

EXPECTED_KEYS_SINGLE = {
    'empathy_index',
    'mu_suppression',
    'social_engagement',
    'model_type',
    'note',
}


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture
def detector():
    return EmpathyDetector()


@pytest.fixture
def eeg_4ch():
    """4-channel x 2 seconds (512 samples) of synthetic EEG."""
    np.random.seed(42)
    return np.random.randn(4, 512) * 20.0


@pytest.fixture
def eeg_1ch():
    """Single-channel 2 seconds (512 samples) of synthetic EEG."""
    np.random.seed(1)
    return np.random.randn(512) * 20.0


# -- 1. set_baseline stores value per user -----------------------------------

class TestSetBaseline:
    def test_stores_value(self, detector):
        detector.set_baseline(0.5, user_id='alice')
        assert detector._baseline_mu['alice'] == 0.5

    def test_per_user_isolation(self, detector):
        detector.set_baseline(0.3, user_id='alice')
        detector.set_baseline(0.7, user_id='bob')
        assert detector._baseline_mu['alice'] == 0.3
        assert detector._baseline_mu['bob'] == 0.7

    def test_zero_baseline_rejected(self, detector):
        detector.set_baseline(0.0, user_id='alice')
        assert 'alice' not in detector._baseline_mu

    def test_negative_baseline_rejected(self, detector):
        detector.set_baseline(-1.0, user_id='alice')
        assert 'alice' not in detector._baseline_mu

    def test_default_user_id(self, detector):
        detector.set_baseline(0.4)
        assert detector._baseline_mu['default'] == 0.4


# -- 2. predict() with baseline — suppression logic --------------------------

class TestPredictWithBaseline:
    def test_lower_mu_higher_empathy(self, detector, eeg_4ch):
        """When current mu < baseline, empathy_index should be positive."""
        detector.set_baseline(0.8)
        result = detector.predict(eeg_4ch, fs=FS)
        # Current mu from random EEG is some value — we just check the math
        # is consistent: if current_mu < baseline, suppression > 0
        current = result['mu_power_current']
        if current < 0.8:
            assert result['empathy_index'] > 0.0
            assert result['mu_suppression'] > 0.0

    def test_equal_mu_zero_empathy(self, detector):
        """When current mu equals baseline exactly, empathy should be ~0."""
        detector.set_baseline(0.5)
        result = detector.predict_from_features({'alpha': 0.5})
        assert abs(result['empathy_index']) < 1e-9
        assert abs(result['mu_suppression']) < 1e-9

    def test_higher_mu_clipped_zero(self, detector):
        """When current mu > baseline, suppression is negative -> clipped to 0."""
        detector.set_baseline(0.3)
        result = detector.predict_from_features({'alpha': 0.6})
        assert result['empathy_index'] == 0.0
        assert result['mu_suppression'] < 0.0  # raw suppression is negative


# -- 3. No baseline -> empathy_index 0.0 -------------------------------------

class TestNoBaseline:
    def test_no_baseline_empathy_zero(self, detector, eeg_4ch):
        result = detector.predict(eeg_4ch, fs=FS)
        assert result['empathy_index'] == 0.0
        assert result['mu_suppression'] == 0.0
        assert result['mu_power_baseline'] is None

    def test_no_baseline_features(self, detector):
        result = detector.predict_from_features({'alpha': 0.4})
        assert result['empathy_index'] == 0.0
        assert result['mu_power_baseline'] is None


# -- 4. Social engagement thresholds -----------------------------------------

class TestEngagementThresholds:
    def test_high_engagement(self, detector):
        """empathy_index > 0.6 -> 'high'"""
        detector.set_baseline(1.0)
        # current_mu = 0.3 -> suppression = 0.7 -> empathy = clip(1.4, 0, 1) = 1.0
        result = detector.predict_from_features({'alpha': 0.3})
        assert result['social_engagement'] == 'high'
        assert result['empathy_index'] > 0.6

    def test_moderate_engagement(self, detector):
        """empathy_index between 0.3 and 0.6 -> 'moderate'"""
        detector.set_baseline(1.0)
        # current_mu = 0.8 -> suppression = 0.2 -> empathy = 0.4
        result = detector.predict_from_features({'alpha': 0.8})
        assert result['social_engagement'] == 'moderate'
        assert 0.3 < result['empathy_index'] <= 0.6

    def test_low_engagement(self, detector):
        """empathy_index <= 0.3 -> 'low'"""
        detector.set_baseline(1.0)
        # current_mu = 0.9 -> suppression = 0.1 -> empathy = 0.2
        result = detector.predict_from_features({'alpha': 0.9})
        assert result['social_engagement'] == 'low'
        assert result['empathy_index'] <= 0.3


# -- 5. Single channel input -> returns unknown ------------------------------

class TestSingleChannel:
    def test_single_channel_returns_unknown(self, detector, eeg_1ch):
        result = detector.predict(eeg_1ch, fs=FS)
        assert result['social_engagement'] == 'unknown'
        assert result['empathy_index'] == 0.0
        assert 'note' in result
        assert result['model_type'] == 'mu_suppression'

    def test_single_channel_all_keys(self, detector, eeg_1ch):
        result = detector.predict(eeg_1ch, fs=FS)
        assert EXPECTED_KEYS_SINGLE == set(result.keys())


# -- 6. Multichannel input works with 4 channels -----------------------------

class TestMultichannel:
    def test_4ch_returns_all_keys(self, detector, eeg_4ch):
        result = detector.predict(eeg_4ch, fs=FS)
        assert EXPECTED_KEYS == set(result.keys())

    def test_4ch_model_type(self, detector, eeg_4ch):
        result = detector.predict(eeg_4ch, fs=FS)
        assert result['model_type'] == 'mu_suppression'

    def test_4ch_empathy_in_range(self, detector, eeg_4ch):
        detector.set_baseline(0.5)
        result = detector.predict(eeg_4ch, fs=FS)
        assert 0.0 <= result['empathy_index'] <= 1.0

    def test_2ch_uses_ch0_for_both(self, detector):
        """With only 2 channels, min(3, shape[0]-1) = 1; still works."""
        np.random.seed(7)
        eeg_2ch = np.random.randn(2, 512) * 20.0
        result = detector.predict(eeg_2ch, fs=FS)
        assert EXPECTED_KEYS == set(result.keys())
        assert result['mu_power_current'] >= 0.0


# -- 7. predict_from_features matches predict logic --------------------------

class TestPredictFromFeatures:
    def test_returns_all_keys(self, detector):
        result = detector.predict_from_features({'alpha': 0.4})
        assert EXPECTED_KEYS == set(result.keys())

    def test_model_type(self, detector):
        result = detector.predict_from_features({'alpha': 0.4})
        assert result['model_type'] == 'mu_suppression'

    def test_consistent_with_baseline(self, detector):
        detector.set_baseline(0.6)
        result = detector.predict_from_features({'alpha': 0.3})
        expected_suppression = (0.6 - 0.3) / 0.6
        assert abs(result['mu_suppression'] - expected_suppression) < 1e-9
        expected_empathy = float(np.clip(expected_suppression * 2.0, 0.0, 1.0))
        assert abs(result['empathy_index'] - expected_empathy) < 1e-9

    def test_missing_alpha_key(self, detector):
        """If features dict has no 'alpha', defaults to 0."""
        detector.set_baseline(0.5)
        result = detector.predict_from_features({})
        # current_mu = 0, suppression = (0.5 - 0) / 0.5 = 1.0
        assert result['mu_suppression'] == 1.0
        assert result['empathy_index'] == 1.0
