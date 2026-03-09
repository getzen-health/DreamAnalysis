"""Tests for NeuralEfficiencyTracker.

Covers:
  - First session (no baseline) behavior
  - Multiple sessions showing improvement
  - Plateau / stable trend detection
  - Declining trend detection
  - Different task types (isolation)
  - Different users (isolation)
  - Reset functionality
  - Edge cases: zero alpha, near-zero alpha, negative values
  - Mastery stage boundaries
  - Session count gate (<5 sessions = novice)
  - assess() auto-records first session
  - get_history returns chronological order
"""

import numpy as np
import pytest

from models.neural_efficiency import (
    NeuralEfficiencyTracker,
    MASTERY_STAGES,
    _MIN_SESSIONS_FOR_PROGRESSION,
)


@pytest.fixture
def tracker():
    return NeuralEfficiencyTracker()


# ── First session / baseline ────────────────────────────────────────────────

class TestFirstSession:
    def test_assess_no_prior_history_creates_baseline(self, tracker):
        """First assess() call auto-records the session and returns score 1.0."""
        result = tracker.assess(0.25, user_id="u1")
        assert result["efficiency_score"] == 1.0
        assert result["mastery_stage"] == "novice"
        assert result["sessions_completed"] == 1
        assert result["improvement_pct"] == 0.0
        assert result["alpha_trend"] == "stable"

    def test_record_session_returns_session_number(self, tracker):
        """record_session should return the session index."""
        info = tracker.record_session("u1", 0.20)
        assert info["session_number"] == 1
        assert info["baseline_alpha"] == 0.20

        info2 = tracker.record_session("u1", 0.25)
        assert info2["session_number"] == 2
        assert info2["baseline_alpha"] == 0.20  # baseline stays as first

    def test_baseline_is_first_session(self, tracker):
        """Baseline alpha should always be the first recorded session."""
        tracker.record_session("u1", 0.18)
        tracker.record_session("u1", 0.22)
        tracker.record_session("u1", 0.30)
        result = tracker.assess(0.30, user_id="u1")
        assert result["baseline_alpha"] == pytest.approx(0.18, abs=1e-6)


# ── Improvement tracking ────────────────────────────────────────────────────

class TestImprovement:
    def test_efficiency_score_above_one_means_improvement(self, tracker):
        """Score > 1.0 when current alpha exceeds baseline."""
        tracker.record_session("u1", 0.20)
        result = tracker.assess(0.26, user_id="u1")
        assert result["efficiency_score"] == pytest.approx(1.3, abs=1e-3)

    def test_improvement_pct_positive(self, tracker):
        """improvement_pct should be positive when alpha increases."""
        tracker.record_session("u1", 0.20)
        result = tracker.assess(0.24, user_id="u1")
        assert result["improvement_pct"] == pytest.approx(20.0, abs=0.1)

    def test_improvement_pct_negative_when_declining(self, tracker):
        """improvement_pct should be negative when alpha drops."""
        tracker.record_session("u1", 0.30)
        result = tracker.assess(0.24, user_id="u1")
        assert result["improvement_pct"] == pytest.approx(-20.0, abs=0.1)

    def test_multiple_sessions_rising_trend(self, tracker):
        """Steadily increasing alpha should produce a 'rising' trend."""
        for val in [0.20, 0.22, 0.24, 0.26, 0.28]:
            tracker.record_session("u1", val)
        result = tracker.assess(0.30, user_id="u1")
        assert result["alpha_trend"] == "rising"


# ── Plateau / stable detection ──────────────────────────────────────────────

class TestPlateau:
    def test_stable_alpha_gives_stable_trend(self, tracker):
        """Nearly constant alpha across sessions -> stable trend."""
        for val in [0.25, 0.251, 0.249, 0.250, 0.252]:
            tracker.record_session("u1", val)
        result = tracker.assess(0.250, user_id="u1")
        assert result["alpha_trend"] == "stable"

    def test_declining_alpha_gives_declining_trend(self, tracker):
        """Steadily decreasing alpha -> declining trend."""
        for val in [0.30, 0.28, 0.26, 0.24, 0.22]:
            tracker.record_session("u1", val)
        result = tracker.assess(0.20, user_id="u1")
        assert result["alpha_trend"] == "declining"


# ── Different task types ────────────────────────────────────────────────────

class TestTaskTypes:
    def test_task_types_are_isolated(self, tracker):
        """Different task types maintain separate histories."""
        tracker.record_session("u1", 0.20, task_type="meditation")
        tracker.record_session("u1", 0.30, task_type="focus")

        med_history = tracker.get_history("u1", "meditation")
        focus_history = tracker.get_history("u1", "focus")

        assert len(med_history) == 1
        assert len(focus_history) == 1
        assert med_history[0] == pytest.approx(0.20)
        assert focus_history[0] == pytest.approx(0.30)

    def test_assess_uses_correct_task_baseline(self, tracker):
        """assess() should use the baseline for the specified task type."""
        tracker.record_session("u1", 0.20, task_type="meditation")
        tracker.record_session("u1", 0.40, task_type="focus")

        # Assess meditation — baseline is 0.20
        result_med = tracker.assess(0.24, user_id="u1", task_type="meditation")
        assert result_med["efficiency_score"] == pytest.approx(1.2, abs=1e-3)

        # Assess focus — baseline is 0.40
        result_focus = tracker.assess(0.44, user_id="u1", task_type="focus")
        assert result_focus["efficiency_score"] == pytest.approx(1.1, abs=1e-3)


# ── Different users ─────────────────────────────────────────────────────────

class TestMultipleUsers:
    def test_users_are_isolated(self, tracker):
        """Different users have independent histories."""
        tracker.record_session("alice", 0.20)
        tracker.record_session("bob", 0.35)

        assert tracker.get_history("alice", "default") == [pytest.approx(0.20)]
        assert tracker.get_history("bob", "default") == [pytest.approx(0.35)]

    def test_reset_one_user_preserves_others(self, tracker):
        """Resetting one user should not affect another."""
        tracker.record_session("alice", 0.20)
        tracker.record_session("bob", 0.30)

        tracker.reset("alice")

        assert tracker.get_history("alice", "default") == []
        assert tracker.get_history("bob", "default") == [pytest.approx(0.30)]


# ── Reset ───────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_all_task_types(self, tracker):
        """Reset should remove history for all task types of that user."""
        tracker.record_session("u1", 0.20, task_type="meditation")
        tracker.record_session("u1", 0.25, task_type="focus")

        tracker.reset("u1")

        assert tracker.get_history("u1", "meditation") == []
        assert tracker.get_history("u1", "focus") == []

    def test_reset_nonexistent_user_is_safe(self, tracker):
        """Reset on unknown user should not raise."""
        tracker.reset("nonexistent")  # should not raise


# ── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_alpha_baseline(self, tracker):
        """Zero baseline alpha should not cause division by zero."""
        tracker.record_session("u1", 0.0)
        result = tracker.assess(0.25, user_id="u1")
        # With zero baseline, efficiency_score defaults to 1.0
        assert result["efficiency_score"] == 1.0
        assert result["improvement_pct"] == 0.0

    def test_zero_current_alpha(self, tracker):
        """Zero current alpha should return score 0."""
        tracker.record_session("u1", 0.25)
        result = tracker.assess(0.0, user_id="u1")
        assert result["efficiency_score"] == pytest.approx(0.0, abs=1e-4)
        assert result["improvement_pct"] == pytest.approx(-100.0, abs=0.1)

    def test_very_small_alpha_no_crash(self, tracker):
        """Near-zero values should not cause numerical instability."""
        tracker.record_session("u1", 1e-12)
        result = tracker.assess(2e-12, user_id="u1")
        # Should not be NaN or Inf
        assert np.isfinite(result["efficiency_score"])
        assert np.isfinite(result["improvement_pct"])

    def test_get_history_unknown_user(self, tracker):
        """Querying unknown user should return empty list, not error."""
        assert tracker.get_history("nobody", "default") == []

    def test_get_history_unknown_task(self, tracker):
        """Querying unknown task type should return empty list."""
        tracker.record_session("u1", 0.20)
        assert tracker.get_history("u1", "nonexistent") == []


# ── Mastery stage classification ────────────────────────────────────────────

class TestMasteryStages:
    def test_novice_with_few_sessions(self, tracker):
        """Even with high score, < 5 sessions = novice."""
        for i in range(4):
            tracker.record_session("u1", 0.20 + i * 0.05)
        result = tracker.assess(0.50, user_id="u1")
        assert result["sessions_completed"] == 4
        assert result["mastery_stage"] == "novice"

    def test_novice_with_low_score(self, tracker):
        """Score < 1.0 is novice even with many sessions."""
        for _ in range(10):
            tracker.record_session("u1", 0.30)
        result = tracker.assess(0.20, user_id="u1")
        assert result["mastery_stage"] == "novice"

    def test_developing_stage(self, tracker):
        """Score 1.0-1.15 with >= 5 sessions = developing."""
        for _ in range(6):
            tracker.record_session("u1", 0.20)
        result = tracker.assess(0.22, user_id="u1")  # 0.22/0.20 = 1.10
        assert result["efficiency_score"] == pytest.approx(1.1, abs=1e-3)
        assert result["mastery_stage"] == "developing"

    def test_proficient_stage(self, tracker):
        """Score 1.15-1.3 with >= 5 sessions = proficient."""
        for _ in range(6):
            tracker.record_session("u1", 0.20)
        result = tracker.assess(0.24, user_id="u1")  # 0.24/0.20 = 1.20
        assert result["efficiency_score"] == pytest.approx(1.2, abs=1e-3)
        assert result["mastery_stage"] == "proficient"

    def test_expert_stage(self, tracker):
        """Score > 1.3 with >= 5 sessions = expert."""
        for _ in range(6):
            tracker.record_session("u1", 0.20)
        result = tracker.assess(0.30, user_id="u1")  # 0.30/0.20 = 1.50
        assert result["efficiency_score"] == pytest.approx(1.5, abs=1e-3)
        assert result["mastery_stage"] == "expert"

    def test_all_mastery_stages_exist(self):
        """Sanity: the expected stages are defined."""
        assert MASTERY_STAGES == ["novice", "developing", "proficient", "expert"]


# ── Output structure ────────────────────────────────────────────────────────

class TestOutputStructure:
    def test_assess_returns_all_required_keys(self, tracker):
        """assess() output should contain all documented keys."""
        tracker.record_session("u1", 0.20)
        result = tracker.assess(0.25, user_id="u1")
        required_keys = {
            "efficiency_score",
            "mastery_stage",
            "sessions_completed",
            "alpha_trend",
            "improvement_pct",
            "baseline_alpha",
            "current_alpha",
        }
        assert required_keys.issubset(result.keys())

    def test_alpha_trend_values_are_valid(self, tracker):
        """alpha_trend must be one of rising/stable/declining."""
        tracker.record_session("u1", 0.20)
        result = tracker.assess(0.25, user_id="u1")
        assert result["alpha_trend"] in ("rising", "stable", "declining")
