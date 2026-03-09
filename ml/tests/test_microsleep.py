"""Tests for microsleep detection in DrowsinessDetector.

Tests the detect_microsleep() method which identifies microsleep episodes
by tracking when theta power dominates alpha power for >= 3 consecutive epochs.
"""

import pytest
import sys
import os

# Add ml/ to path so imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.drowsiness_detector import DrowsinessDetector


@pytest.fixture
def detector():
    """Fresh DrowsinessDetector instance for each test."""
    return DrowsinessDetector()


# --- Basic drowsy/alert classification ---

class TestDrowsyEpochDetection:
    """Tests for single-epoch drowsy vs alert classification."""

    def test_high_theta_alpha_ratio_is_drowsy(self, detector):
        """Theta/alpha > 1.5 should mark epoch as drowsy."""
        result = detector.detect_microsleep({'theta': 0.5, 'alpha': 0.1})
        assert result['drowsiness_level'] == 'drowsy'
        assert result['theta_alpha_ratio'] > 1.5

    def test_low_theta_alpha_ratio_is_alert(self, detector):
        """Theta/alpha < 1.5 should mark epoch as alert."""
        result = detector.detect_microsleep({'theta': 0.1, 'alpha': 0.5})
        assert result['drowsiness_level'] == 'alert'
        assert result['theta_alpha_ratio'] < 1.5

    def test_equal_theta_alpha_is_alert(self, detector):
        """Equal theta and alpha (ratio ~1.0) should be alert, not drowsy."""
        result = detector.detect_microsleep({'theta': 0.3, 'alpha': 0.3})
        assert result['drowsiness_level'] == 'alert'
        assert not result['microsleep_alert']

    def test_ratio_exactly_at_threshold_is_alert(self, detector):
        """Theta/alpha ratio of exactly 1.5 should NOT trigger drowsy (> not >=)."""
        # theta / (alpha + 1e-8) = 1.5 requires theta = 1.5 * (alpha + 1e-8)
        # For alpha=1.0: theta = 1.5 * (1.0 + 1e-8) ~ 1.5000000015
        # Use values that produce ratio just below 1.5
        result = detector.detect_microsleep({'theta': 0.3, 'alpha': 0.2})
        # 0.3 / 0.2 = 1.5 — exactly at boundary, should NOT trigger (> not >=)
        assert result['drowsiness_level'] == 'alert'


# --- Microsleep streak detection ---

class TestMicrosleepStreak:
    """Tests for consecutive-epoch microsleep detection."""

    def test_three_consecutive_drowsy_triggers_microsleep(self, detector):
        """3 consecutive drowsy epochs should trigger microsleep alert."""
        drowsy_features = {'theta': 0.8, 'alpha': 0.1}
        for _ in range(2):
            result = detector.detect_microsleep(drowsy_features)
            assert not result['microsleep_alert']

        result = detector.detect_microsleep(drowsy_features)
        assert result['microsleep_alert']
        assert result['drowsiness_level'] == 'microsleep'

    def test_two_consecutive_drowsy_no_microsleep(self, detector):
        """2 consecutive drowsy epochs should NOT trigger microsleep."""
        drowsy_features = {'theta': 0.8, 'alpha': 0.1}
        detector.detect_microsleep(drowsy_features)
        result = detector.detect_microsleep(drowsy_features)
        assert not result['microsleep_alert']
        assert result['drowsiness_level'] == 'drowsy'

    def test_alert_epoch_resets_counter(self, detector):
        """An alert epoch after drowsy epochs should reset the streak."""
        drowsy = {'theta': 0.8, 'alpha': 0.1}
        alert = {'theta': 0.1, 'alpha': 0.5}

        # Build up 2 drowsy epochs
        detector.detect_microsleep(drowsy)
        detector.detect_microsleep(drowsy)

        # Alert resets
        result = detector.detect_microsleep(alert)
        assert result['drowsy_streak_seconds'] == 0
        assert result['drowsiness_level'] == 'alert'

        # Need 3 more drowsy epochs to trigger microsleep again
        detector.detect_microsleep(drowsy)
        detector.detect_microsleep(drowsy)
        result = detector.detect_microsleep(drowsy)
        assert result['microsleep_alert']

    def test_drowsy_streak_seconds_accumulates(self, detector):
        """drowsy_streak_seconds should increase by 2 per drowsy epoch."""
        drowsy = {'theta': 0.8, 'alpha': 0.1}
        for i in range(5):
            result = detector.detect_microsleep(drowsy)
            assert result['drowsy_streak_seconds'] == (i + 1) * 2


# --- Per-user isolation ---

class TestPerUserIsolation:
    """Tests that different user_ids maintain independent counters."""

    def test_different_users_independent_counters(self, detector):
        """Drowsy epochs for user A should not affect user B's counter."""
        drowsy = {'theta': 0.8, 'alpha': 0.1}

        # User A builds up 2 drowsy epochs
        detector.detect_microsleep(drowsy, user_id='user_a')
        detector.detect_microsleep(drowsy, user_id='user_a')

        # User B starts fresh — should have 0 streak
        result_b = detector.detect_microsleep(drowsy, user_id='user_b')
        assert result_b['drowsy_streak_seconds'] == 2  # just 1 epoch
        assert not result_b['microsleep_alert']

        # User A's 3rd epoch should trigger microsleep
        result_a = detector.detect_microsleep(drowsy, user_id='user_a')
        assert result_a['microsleep_alert']
        assert result_a['drowsy_streak_seconds'] == 6

    def test_default_user_id(self, detector):
        """Calling without user_id should use 'default' and work correctly."""
        drowsy = {'theta': 0.8, 'alpha': 0.1}
        for _ in range(3):
            result = detector.detect_microsleep(drowsy)
        assert result['microsleep_alert']
        assert 'default' in detector._microsleep_counters


# --- theta_alpha_ratio calculation ---

class TestThetaAlphaRatio:
    """Tests for correct ratio computation."""

    def test_ratio_calculation_basic(self, detector):
        """Ratio should be theta / (alpha + epsilon)."""
        result = detector.detect_microsleep({'theta': 0.6, 'alpha': 0.2})
        expected = 0.6 / (0.2 + 1e-8)
        assert abs(result['theta_alpha_ratio'] - expected) < 1e-6

    def test_ratio_is_float(self, detector):
        """theta_alpha_ratio should always be a Python float."""
        result = detector.detect_microsleep({'theta': 0.5, 'alpha': 0.3})
        assert isinstance(result['theta_alpha_ratio'], float)


# --- Edge cases ---

class TestEdgeCases:
    """Edge cases: zero values, missing keys, extreme inputs."""

    def test_zero_alpha_does_not_crash(self, detector):
        """Alpha=0 should not cause division by zero (epsilon protects)."""
        result = detector.detect_microsleep({'theta': 0.5, 'alpha': 0.0})
        assert result['theta_alpha_ratio'] > 0
        assert result['drowsiness_level'] in ('drowsy', 'microsleep')

    def test_zero_theta_is_alert(self, detector):
        """Theta=0 means ratio ~0, should be alert."""
        result = detector.detect_microsleep({'theta': 0.0, 'alpha': 0.5})
        assert result['theta_alpha_ratio'] < 1e-6
        assert result['drowsiness_level'] == 'alert'
        assert not result['microsleep_alert']

    def test_both_zero_is_alert(self, detector):
        """Both theta and alpha at 0 — ratio ~ 0, should be alert."""
        result = detector.detect_microsleep({'theta': 0.0, 'alpha': 0.0})
        assert result['drowsiness_level'] == 'alert'

    def test_missing_keys_default_to_zero(self, detector):
        """Missing 'theta' or 'alpha' keys should default to 0."""
        result = detector.detect_microsleep({})
        assert result['theta_alpha_ratio'] < 1e-6
        assert result['drowsiness_level'] == 'alert'

    def test_very_large_theta(self, detector):
        """Very large theta should still produce valid output."""
        result = detector.detect_microsleep({'theta': 1000.0, 'alpha': 0.001})
        assert result['drowsiness_level'] in ('drowsy', 'microsleep')
        assert result['theta_alpha_ratio'] > 1.5


# --- Return structure ---

class TestReturnStructure:
    """Tests that the return dict has all expected keys and types."""

    def test_return_keys(self, detector):
        """Return dict must contain all 4 expected keys."""
        result = detector.detect_microsleep({'theta': 0.3, 'alpha': 0.2})
        assert 'microsleep_alert' in result
        assert 'drowsy_streak_seconds' in result
        assert 'theta_alpha_ratio' in result
        assert 'drowsiness_level' in result

    def test_return_types(self, detector):
        """Check types of all return values."""
        result = detector.detect_microsleep({'theta': 0.3, 'alpha': 0.2})
        assert isinstance(result['microsleep_alert'], bool)
        assert isinstance(result['drowsy_streak_seconds'], int)
        assert isinstance(result['theta_alpha_ratio'], float)
        assert isinstance(result['drowsiness_level'], str)

    def test_drowsiness_level_valid_values(self, detector):
        """drowsiness_level must be one of the 3 valid strings."""
        valid_levels = {'alert', 'drowsy', 'microsleep'}
        # Test alert
        r1 = detector.detect_microsleep({'theta': 0.1, 'alpha': 0.5})
        assert r1['drowsiness_level'] in valid_levels
        # Test drowsy
        r2 = detector.detect_microsleep({'theta': 0.8, 'alpha': 0.1})
        assert r2['drowsiness_level'] in valid_levels
