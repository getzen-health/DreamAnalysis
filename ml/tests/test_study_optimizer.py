"""Tests for the EEG-guided study session optimizer."""

import pytest

from ml.models.study_optimizer import StudyOptimizer


@pytest.fixture
def optimizer():
    """Return a fresh StudyOptimizer instance."""
    return StudyOptimizer()


# --- Rule-based recommendation tests ---


class TestTakeBreakHighUrgency:
    """Very low attention (< 0.2) + time > 10 min => take_break, high urgency."""

    def test_very_low_attention_triggers_urgent_break(self, optimizer):
        result = optimizer.recommend(
            attention_score=0.1,
            cognitive_load=0.5,
            time_since_break_min=15,
        )
        assert result["action"] == "take_break"
        assert result["urgency"] == "high"
        assert result["break_duration_min"] == 10

    def test_low_attention_but_recent_break_does_not_trigger(self, optimizer):
        """If break was < 10 min ago, the urgent-break rule should NOT fire."""
        result = optimizer.recommend(
            attention_score=0.1,
            cognitive_load=0.5,
            time_since_break_min=5,
        )
        # Should fall through to a different action (continue, etc.)
        assert result["action"] != "take_break" or result["urgency"] != "high"


class TestTakeBreakMediumUrgency:
    """Low attention (< 0.4) + time > 20 min => take_break, medium urgency."""

    def test_sustained_low_attention_suggests_break(self, optimizer):
        result = optimizer.recommend(
            attention_score=0.3,
            cognitive_load=0.5,
            time_since_break_min=25,
        )
        assert result["action"] == "take_break"
        assert result["urgency"] == "medium"
        assert result["break_duration_min"] == 5


class TestReduceDifficulty:
    """High cognitive load (> 0.8) + session > 15 min => reduce_difficulty."""

    def test_cognitive_overload_reduces_difficulty(self, optimizer):
        result = optimizer.recommend(
            attention_score=0.6,
            cognitive_load=0.9,
            session_duration_min=20,
        )
        assert result["action"] == "reduce_difficulty"
        assert result["urgency"] == "medium"


class TestReviewNow:
    """FMT elevated + good attention => review_now."""

    def test_fmt_elevated_triggers_review(self, optimizer):
        optimizer.set_baseline(fmt_power=1.0)
        result = optimizer.recommend(
            attention_score=0.7,
            cognitive_load=0.4,
            fmt_power=1.5,  # > 1.0 * 1.3 = 1.3
        )
        assert result["action"] == "review_now"
        assert result["encoding_state"] is True


class TestIncreaseDifficulty:
    """High attention (> 0.7) + low load (< 0.3) + session > 10 min."""

    def test_boredom_increases_difficulty(self, optimizer):
        result = optimizer.recommend(
            attention_score=0.8,
            cognitive_load=0.2,
            session_duration_min=15,
        )
        assert result["action"] == "increase_difficulty"
        assert result["urgency"] == "low"


class TestLongSessionBreak:
    """Time since break > 45 min => take_break."""

    def test_long_session_forces_break(self, optimizer):
        result = optimizer.recommend(
            attention_score=0.6,
            cognitive_load=0.5,
            time_since_break_min=50,
        )
        assert result["action"] == "take_break"
        assert result["break_duration_min"] == 10


class TestContinue:
    """Normal state => continue."""

    def test_normal_state_continues(self, optimizer):
        result = optimizer.recommend(
            attention_score=0.6,
            cognitive_load=0.5,
            session_duration_min=5,
            time_since_break_min=5,
        )
        assert result["action"] == "continue"
        assert result["urgency"] == "none"


# --- Trend detection ---


class TestTrendDetection:
    """Engagement trend computation from attention history."""

    def test_rising_trend(self, optimizer):
        # Feed 12 data points: first 6 low, then 6 high
        for val in [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]:
            optimizer.recommend(attention_score=val, cognitive_load=0.5)
        for val in [0.7, 0.7, 0.7, 0.7, 0.7]:
            optimizer.recommend(attention_score=val, cognitive_load=0.5)
        result = optimizer.recommend(attention_score=0.7, cognitive_load=0.5)
        assert result["engagement_trend"] == "rising"

    def test_declining_trend(self, optimizer):
        # Feed 12 data points: first 6 high, then 6 low
        for val in [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]:
            optimizer.recommend(attention_score=val, cognitive_load=0.5)
        for val in [0.4, 0.4, 0.4, 0.4, 0.4]:
            optimizer.recommend(attention_score=val, cognitive_load=0.5)
        result = optimizer.recommend(attention_score=0.4, cognitive_load=0.5)
        assert result["engagement_trend"] == "declining"

    def test_stable_trend(self, optimizer):
        # Feed 12 identical data points
        for _ in range(12):
            optimizer.recommend(attention_score=0.5, cognitive_load=0.5)
        result = optimizer.recommend(attention_score=0.5, cognitive_load=0.5)
        assert result["engagement_trend"] == "stable"

    def test_insufficient_data_trend(self, optimizer):
        result = optimizer.recommend(attention_score=0.5, cognitive_load=0.5)
        assert result["engagement_trend"] == "insufficient_data"


# --- Session stats ---


class TestSessionStats:
    """get_session_stats returns correct aggregates."""

    def test_stats_after_recommendations(self, optimizer):
        optimizer.recommend(attention_score=0.3, cognitive_load=0.5)
        optimizer.recommend(attention_score=0.7, cognitive_load=0.5)
        optimizer.recommend(attention_score=0.5, cognitive_load=0.5)

        stats = optimizer.get_session_stats()
        assert stats["n_samples"] == 3
        assert stats["min_attention"] == pytest.approx(0.3)
        assert stats["max_attention"] == pytest.approx(0.7)
        assert stats["mean_attention"] == pytest.approx(0.5)

    def test_stats_empty(self, optimizer):
        stats = optimizer.get_session_stats()
        assert stats["n_samples"] == 0
        assert stats["mean_attention"] == 0


# --- Baseline and history capping ---


class TestSetBaseline:
    """set_baseline stores the FMT baseline value."""

    def test_baseline_stored(self, optimizer):
        optimizer.set_baseline(fmt_power=2.5)
        assert optimizer._fmt_baseline == 2.5


class TestAttentionHistoryCapping:
    """Attention history is capped at 30 entries."""

    def test_history_caps_at_30(self, optimizer):
        for i in range(40):
            optimizer.recommend(attention_score=i / 40.0, cognitive_load=0.5)
        assert len(optimizer._attention_history) == 30


class TestStartSession:
    """start_session resets internal state."""

    def test_start_clears_history(self, optimizer):
        optimizer.recommend(attention_score=0.5, cognitive_load=0.5)
        assert len(optimizer._attention_history) == 1

        optimizer.start_session()
        assert len(optimizer._attention_history) == 0
        assert optimizer._session_start is not None
        assert optimizer._last_break is not None
