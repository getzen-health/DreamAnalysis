"""Tests for adaptive learning speed detector."""
import pytest

from models.adaptive_learning import AdaptiveLearningDetector


@pytest.fixture
def detector():
    return AdaptiveLearningDetector()


def test_no_baseline_high_load(detector):
    """Without baseline, high cognitive load → slow down."""
    result = detector.assess_pace(0.5, cognitive_load=0.9, time_on_task_min=10)
    assert result["recommendation"] == "slow_down"
    assert result["theta_ratio"] is None


def test_no_baseline_low_load(detector):
    """Without baseline, low load + time → increase challenge."""
    result = detector.assess_pace(0.3, cognitive_load=0.1, time_on_task_min=10)
    assert result["recommendation"] == "increase_challenge"


def test_no_baseline_extended_session(detector):
    """Without baseline, 30+ min → take break."""
    result = detector.assess_pace(0.4, cognitive_load=0.5, time_on_task_min=35)
    assert result["recommendation"] == "take_break"


def test_no_baseline_normal(detector):
    """Without baseline and moderate state → maintain pace."""
    result = detector.assess_pace(0.4, cognitive_load=0.5, time_on_task_min=5)
    assert result["recommendation"] == "maintain_pace"


def test_with_baseline_overload(detector):
    """Very high theta ratio → cognitive overload."""
    detector.set_baseline(0.3, user_id="u1")
    result = detector.assess_pace(0.6, cognitive_load=0.5, user_id="u1")
    assert result["recommendation"] == "slow_down"
    assert result["reason"] == "cognitive_overload"
    assert result["theta_ratio"] == pytest.approx(2.0)


def test_with_baseline_high_theta_high_load(detector):
    """Elevated theta + high cognitive load → slow down."""
    detector.set_baseline(0.3, user_id="u1")
    result = detector.assess_pace(0.48, cognitive_load=0.8, user_id="u1")
    assert result["recommendation"] == "slow_down"
    assert result["reason"] == "high_load_high_theta"


def test_with_baseline_disengaged(detector):
    """Low theta ratio after some time → increase challenge."""
    detector.set_baseline(0.5, user_id="u1")
    result = detector.assess_pace(0.3, cognitive_load=0.3, time_on_task_min=10, user_id="u1")
    assert result["recommendation"] == "increase_challenge"
    assert result["reason"] == "disengaged"


def test_with_baseline_under_stimulated(detector):
    """Low theta + very low load → increase challenge."""
    detector.set_baseline(0.5, user_id="u1")
    result = detector.assess_pace(0.35, cognitive_load=0.1, time_on_task_min=2, user_id="u1")
    assert result["recommendation"] == "increase_challenge"
    assert result["reason"] == "under_stimulated"


def test_with_baseline_optimal(detector):
    """Moderate theta elevation + moderate load → optimal pace."""
    detector.set_baseline(0.3, user_id="u1")
    result = detector.assess_pace(0.4, cognitive_load=0.4, user_id="u1")
    assert result["recommendation"] == "optimal_pace"
    assert result["reason"] == "active_encoding"


def test_encoding_quality_optimal(detector):
    """Encoding quality peaks at moderate theta elevation."""
    detector.set_baseline(0.3, user_id="u1")
    result = detector.assess_pace(0.4, user_id="u1")
    assert result["encoding_quality"] > 0.5


def test_encoding_quality_ranges(detector):
    """Encoding quality should be between 0 and 1."""
    detector.set_baseline(0.3, user_id="u1")
    for theta in [0.01, 0.1, 0.3, 0.5, 0.8, 1.5]:
        result = detector.assess_pace(theta, user_id="u1")
        assert 0 <= result["encoding_quality"] <= 1


def test_history_tracking(detector):
    """History stores FMT values."""
    for i in range(5):
        detector.assess_pace(0.3 + i * 0.01, user_id="u1")
    history = detector.get_history("u1")
    assert len(history) == 5


def test_history_cap(detector):
    """History caps at 60 entries."""
    for i in range(70):
        detector.assess_pace(0.3, user_id="u1")
    history = detector.get_history("u1")
    assert len(history) == 60


def test_trend_insufficient_data(detector):
    """Trend needs at least 6 data points."""
    result = detector.assess_pace(0.3)
    assert result["theta_trend"] == "insufficient_data"


def test_trend_rising(detector):
    """Rising FMT should show rising trend."""
    detector.set_baseline(0.3)
    for i in range(12):
        detector.assess_pace(0.2 + i * 0.05)
    result = detector.assess_pace(0.9)
    assert result["theta_trend"] == "rising"


def test_trend_declining(detector):
    """Declining FMT should show declining trend."""
    detector.set_baseline(0.5)
    for i in range(6):
        detector.assess_pace(0.8 - i * 0.01)
    for i in range(7):
        detector.assess_pace(0.5 - i * 0.05)
    result = detector.assess_pace(0.1)
    assert result["theta_trend"] == "declining"


def test_multiple_users(detector):
    """Different users have independent histories."""
    detector.set_baseline(0.3, user_id="alice")
    detector.set_baseline(0.5, user_id="bob")
    r1 = detector.assess_pace(0.6, user_id="alice")  # 2x baseline → overload
    r2 = detector.assess_pace(0.6, user_id="bob")    # 1.2x baseline → optimal
    assert r1["recommendation"] == "slow_down"
    assert r2["recommendation"] == "optimal_pace"


def test_reset(detector):
    """Reset clears history and baseline."""
    detector.set_baseline(0.3, user_id="u1")
    detector.assess_pace(0.4, user_id="u1")
    detector.reset("u1")
    assert detector.get_history("u1") == []
    result = detector.assess_pace(0.4, user_id="u1")
    assert result["theta_ratio"] is None  # no baseline


def test_zero_baseline_ignored(detector):
    """Zero baseline should be rejected."""
    detector.set_baseline(0.0, user_id="u1")
    result = detector.assess_pace(0.4, user_id="u1")
    assert result["theta_ratio"] is None


def test_output_fields(detector):
    """All expected fields present in output."""
    result = detector.assess_pace(0.4, cognitive_load=0.5, time_on_task_min=10)
    expected_keys = {"recommendation", "reason", "theta_ratio", "encoding_quality",
                     "cognitive_load", "time_on_task_min", "theta_trend", "fmt_power"}
    assert expected_keys.issubset(set(result.keys()))
