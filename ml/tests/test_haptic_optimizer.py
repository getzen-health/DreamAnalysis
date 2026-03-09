"""Tests for haptic urgency optimizer."""
import pytest

from models.haptic_optimizer import HapticUrgencyOptimizer, HAPTIC_PATTERNS


@pytest.fixture
def optimizer():
    return HapticUrgencyOptimizer()


def test_low_arousal_high_intensity(optimizer):
    """Low arousal should produce high intensity to ensure alerting."""
    result = optimizer.map_urgency(0.1, alert_priority="medium")
    assert result["intensity"] > 0.6


def test_high_arousal_lower_intensity(optimizer):
    """High arousal should produce lower intensity to avoid startling."""
    result = optimizer.map_urgency(0.9, alert_priority="low")
    assert result["intensity"] < 0.6


def test_critical_always_alarm_burst(optimizer):
    """Critical priority should always use alarm_burst pattern."""
    for arousal in [0.1, 0.5, 0.9]:
        result = optimizer.map_urgency(arousal, alert_priority="critical")
        assert result["pattern"] == "alarm_burst"


def test_intensity_range(optimizer):
    """Intensity should always be between 0.1 and 1.0."""
    for arousal in [0, 0.25, 0.5, 0.75, 1.0]:
        for priority in ["low", "medium", "high", "critical"]:
            result = optimizer.map_urgency(arousal, alert_priority=priority)
            assert 0.1 <= result["intensity"] <= 1.0


def test_effective_urgency_range(optimizer):
    """Effective urgency should be between 0 and 1."""
    for arousal in [0, 0.5, 1.0]:
        result = optimizer.map_urgency(arousal)
        assert 0 <= result["effective_urgency"] <= 1.0


def test_drowsiness_amplifies(optimizer):
    """Drowsiness should increase effective urgency."""
    no_drowsy = optimizer.map_urgency(0.5, drowsiness=0.0)
    drowsy = optimizer.map_urgency(0.5, drowsiness=0.8)
    assert drowsy["effective_urgency"] > no_drowsy["effective_urgency"]


def test_priority_ordering(optimizer):
    """Higher priority should produce higher urgency."""
    low = optimizer.map_urgency(0.5, alert_priority="low")
    high = optimizer.map_urgency(0.5, alert_priority="high")
    assert high["effective_urgency"] > low["effective_urgency"]


def test_pattern_has_required_fields(optimizer):
    """Output should include all required fields."""
    result = optimizer.map_urgency(0.5)
    required = {"intensity", "pattern", "duration_ms", "pulses",
                "effective_urgency", "rationale", "arousal_state", "alert_priority"}
    assert required.issubset(set(result.keys()))


def test_pattern_from_known_set(optimizer):
    """Pattern should be one of the defined haptic patterns."""
    for arousal in [0.1, 0.3, 0.5, 0.7, 0.9]:
        result = optimizer.map_urgency(arousal)
        assert result["pattern"] in HAPTIC_PATTERNS


def test_duration_positive(optimizer):
    """Duration should always be positive."""
    result = optimizer.map_urgency(0.5)
    assert result["duration_ms"] > 0


def test_pulses_positive(optimizer):
    """Pulses should always be positive."""
    result = optimizer.map_urgency(0.5)
    assert result["pulses"] >= 1


def test_arousal_state_classification(optimizer):
    """Arousal state should map to descriptive levels."""
    states = set()
    for arousal in [0.05, 0.3, 0.5, 0.7, 0.9]:
        result = optimizer.map_urgency(arousal)
        states.add(result["arousal_state"])
    assert len(states) >= 3  # Should have multiple distinct states


def test_rationale_not_empty(optimizer):
    """Rationale should always be a non-empty string."""
    for arousal in [0.1, 0.5, 0.9]:
        result = optimizer.map_urgency(arousal)
        assert isinstance(result["rationale"], str)
        assert len(result["rationale"]) > 0


def test_trend_insufficient_data(optimizer):
    """Trend needs at least 6 data points."""
    optimizer.map_urgency(0.5)
    assert optimizer.get_arousal_trend() == "insufficient_data"


def test_trend_stable(optimizer):
    """Constant arousal should show stable trend."""
    for _ in range(12):
        optimizer.map_urgency(0.5)
    assert optimizer.get_arousal_trend() == "stable"


def test_trend_declining(optimizer):
    """Decreasing arousal should show declining trend."""
    for i in range(6):
        optimizer.map_urgency(0.8)
    for i in range(7):
        optimizer.map_urgency(0.3)
    assert optimizer.get_arousal_trend() == "declining"


def test_extreme_arousal_clamped(optimizer):
    """Extreme arousal values should be clamped."""
    result = optimizer.map_urgency(-5.0)
    assert result["arousal_state"] == "very_low"
    result = optimizer.map_urgency(50.0)
    assert result["arousal_state"] == "very_high"


def test_unknown_priority_defaults(optimizer):
    """Unknown priority should default to medium-like behavior."""
    result = optimizer.map_urgency(0.5, alert_priority="unknown")
    assert "intensity" in result
