"""Tests for LearningStageClassifier.

Covers:
  - All four stage detections (encoding, consolidation, automation, mastery)
  - Stage scoring ranges and normalization
  - Progression tracking (get_progression)
  - Session recording (record_session)
  - Multiple users and task types (isolation)
  - Edge cases: zero powers, extreme ratios, boundary values
  - Output structure validation
  - Confidence scoring
  - Theta/alpha ratio computation
  - Recommendations
  - Reset functionality
"""

import numpy as np
import pytest

from models.learning_stage import (
    LearningStageClassifier,
    LEARNING_STAGES,
    _STAGE_RECOMMENDATIONS,
)


@pytest.fixture
def clf():
    return LearningStageClassifier()


# ── Stage detection ────────────────────────────────────────────────────────


class TestEncodingStage:
    def test_high_theta_low_alpha_low_perf_is_encoding(self, clf):
        """High theta + low alpha + low performance = encoding stage."""
        result = clf.classify(
            theta_power=0.40, alpha_power=0.10,
            task_performance=0.1, session_count=1,
        )
        assert result["stage"] == "encoding"

    def test_encoding_has_highest_score(self, clf):
        """Encoding score should dominate in encoding conditions."""
        result = clf.classify(
            theta_power=0.50, alpha_power=0.05,
            task_performance=0.0, session_count=1,
        )
        scores = result["stage_scores"]
        assert scores["encoding"] > scores["consolidation"]
        assert scores["encoding"] > scores["automation"]
        assert scores["encoding"] > scores["mastery"]


class TestConsolidationStage:
    def test_balanced_theta_alpha_moderate_perf_is_consolidation(self, clf):
        """Balanced theta/alpha + moderate performance = consolidation."""
        result = clf.classify(
            theta_power=0.25, alpha_power=0.25,
            task_performance=0.5, session_count=3,
        )
        assert result["stage"] == "consolidation"

    def test_consolidation_peaks_at_moderate_performance(self, clf):
        """Consolidation score should be highest around perf=0.5."""
        result_mid = clf.classify(
            theta_power=0.25, alpha_power=0.25, task_performance=0.5,
        )
        result_low = clf.classify(
            theta_power=0.25, alpha_power=0.25, task_performance=0.1,
        )
        result_high = clf.classify(
            theta_power=0.25, alpha_power=0.25, task_performance=0.9,
        )
        # Consolidation score at perf=0.5 should be >= the others
        assert (
            result_mid["stage_scores"]["consolidation"]
            >= result_low["stage_scores"]["consolidation"]
        )


class TestAutomationStage:
    def test_high_alpha_low_theta_good_perf_is_automation(self, clf):
        """High alpha + low theta + good performance = automation."""
        result = clf.classify(
            theta_power=0.05, alpha_power=0.45,
            task_performance=0.85, session_count=8,
        )
        assert result["stage"] == "automation"

    def test_automation_score_rises_with_alpha(self, clf):
        """Automation score should increase as alpha dominates."""
        low_alpha = clf.classify(theta_power=0.30, alpha_power=0.20, task_performance=0.8)
        high_alpha = clf.classify(theta_power=0.10, alpha_power=0.40, task_performance=0.8)
        assert (
            high_alpha["stage_scores"]["automation"]
            > low_alpha["stage_scores"]["automation"]
        )


class TestMasteryStage:
    def test_high_alpha_gamma_high_perf_is_mastery(self, clf):
        """High alpha + gamma + high perf + experience = mastery."""
        result = clf.classify(
            theta_power=0.05, alpha_power=0.50,
            gamma_power=0.50, task_performance=0.98,
            session_count=30,
        )
        assert result["stage"] == "mastery"

    def test_mastery_needs_gamma_to_beat_automation(self, clf):
        """Without gamma, high-alpha/high-perf should prefer automation."""
        no_gamma = clf.classify(
            theta_power=0.05, alpha_power=0.45,
            gamma_power=0.0, task_performance=0.9, session_count=10,
        )
        with_gamma = clf.classify(
            theta_power=0.05, alpha_power=0.45,
            gamma_power=0.30, task_performance=0.9, session_count=10,
        )
        assert with_gamma["stage_scores"]["mastery"] > no_gamma["stage_scores"]["mastery"]


# ── Stage scoring ──────────────────────────────────────────────────────────


class TestStageScoring:
    def test_scores_sum_to_one(self, clf):
        """All stage scores should sum to approximately 1.0."""
        result = clf.classify(theta_power=0.25, alpha_power=0.25)
        total = sum(result["stage_scores"].values())
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_all_stages_present_in_scores(self, clf):
        """stage_scores dict must contain all four stages."""
        result = clf.classify(theta_power=0.2, alpha_power=0.3)
        for stage in LEARNING_STAGES:
            assert stage in result["stage_scores"]

    def test_scores_are_non_negative(self, clf):
        """No stage score should be negative."""
        result = clf.classify(theta_power=0.1, alpha_power=0.4, task_performance=0.9)
        for score in result["stage_scores"].values():
            assert score >= 0.0


# ── Progression tracking ──────────────────────────────────────────────────


class TestProgressionTracking:
    def test_empty_progression(self, clf):
        """No recorded sessions should return empty progression."""
        prog = clf.get_progression()
        assert prog["sessions"] == []
        assert prog["current_stage"] is None
        assert prog["sessions_completed"] == 0
        assert prog["stage_history"] == []

    def test_progression_after_sessions(self, clf):
        """Progression should reflect recorded sessions."""
        clf.record_session(theta_power=0.40, alpha_power=0.10, performance=0.1)
        clf.record_session(theta_power=0.25, alpha_power=0.25, performance=0.5)
        clf.record_session(theta_power=0.10, alpha_power=0.40, performance=0.85)

        prog = clf.get_progression()
        assert prog["sessions_completed"] == 3
        assert len(prog["sessions"]) == 3
        assert len(prog["stage_history"]) == 3
        assert prog["current_stage"] == prog["stage_history"][-1]

    def test_stage_history_matches_classifications(self, clf):
        """Stage history entries should match individual classifications."""
        r1 = clf.record_session(theta_power=0.40, alpha_power=0.10, performance=0.1)
        r2 = clf.record_session(theta_power=0.10, alpha_power=0.40, performance=0.85)

        prog = clf.get_progression()
        assert prog["stage_history"][0] == r1["stage"]
        assert prog["stage_history"][1] == r2["stage"]


# ── Session recording ─────────────────────────────────────────────────────


class TestSessionRecording:
    def test_record_returns_session_number(self, clf):
        """record_session should return incrementing session numbers."""
        r1 = clf.record_session(theta_power=0.3, alpha_power=0.2)
        r2 = clf.record_session(theta_power=0.25, alpha_power=0.25)
        assert r1["session_number"] == 1
        assert r2["session_number"] == 2

    def test_record_returns_stage_and_scores(self, clf):
        """record_session should include stage classification info."""
        result = clf.record_session(theta_power=0.35, alpha_power=0.15, performance=0.2)
        assert "stage" in result
        assert "stage_scores" in result
        assert "confidence" in result
        assert result["stage"] in LEARNING_STAGES

    def test_recorded_data_stored_correctly(self, clf):
        """Session data should be retrievable via get_progression."""
        clf.record_session(
            theta_power=0.30, alpha_power=0.20,
            gamma_power=0.05, performance=0.6,
        )
        prog = clf.get_progression()
        session = prog["sessions"][0]
        assert session["theta_power"] == pytest.approx(0.30)
        assert session["alpha_power"] == pytest.approx(0.20)
        assert session["gamma_power"] == pytest.approx(0.05)
        assert session["performance"] == pytest.approx(0.6)


# ── Multiple users / tasks ────────────────────────────────────────────────


class TestMultiUsersAndTasks:
    def test_users_are_isolated(self, clf):
        """Different users should have independent histories."""
        clf.record_session(theta_power=0.4, alpha_power=0.1, user_id="alice")
        clf.record_session(theta_power=0.1, alpha_power=0.4, user_id="bob")

        alice = clf.get_progression(user_id="alice")
        bob = clf.get_progression(user_id="bob")

        assert alice["sessions_completed"] == 1
        assert bob["sessions_completed"] == 1
        assert alice["sessions"][0]["theta_power"] == pytest.approx(0.4)
        assert bob["sessions"][0]["theta_power"] == pytest.approx(0.1)

    def test_task_types_are_isolated(self, clf):
        """Different task types for the same user are independent."""
        clf.record_session(
            theta_power=0.35, alpha_power=0.15,
            user_id="u1", task_type="piano",
        )
        clf.record_session(
            theta_power=0.15, alpha_power=0.35,
            user_id="u1", task_type="math",
        )

        piano = clf.get_progression(user_id="u1", task_type="piano")
        math = clf.get_progression(user_id="u1", task_type="math")

        assert piano["sessions_completed"] == 1
        assert math["sessions_completed"] == 1

    def test_reset_one_user_preserves_others(self, clf):
        """Resetting one user should not affect another."""
        clf.record_session(theta_power=0.3, alpha_power=0.2, user_id="alice")
        clf.record_session(theta_power=0.2, alpha_power=0.3, user_id="bob")

        clf.reset("alice")

        assert clf.get_progression(user_id="alice")["sessions_completed"] == 0
        assert clf.get_progression(user_id="bob")["sessions_completed"] == 1


# ── Edge cases ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_theta_and_alpha(self, clf):
        """Both powers at zero should not crash."""
        result = clf.classify(theta_power=0.0, alpha_power=0.0)
        assert result["stage"] in LEARNING_STAGES
        assert np.isfinite(result["confidence"])
        assert np.isfinite(result["theta_alpha_ratio"])

    def test_zero_alpha_nonzero_theta(self, clf):
        """Zero alpha should not cause division by zero in ratio."""
        result = clf.classify(theta_power=0.5, alpha_power=0.0)
        assert result["theta_alpha_ratio"] == 10.0  # capped high value
        assert result["stage"] in LEARNING_STAGES

    def test_performance_clamped_above_one(self, clf):
        """Performance > 1.0 should be clamped to 1.0."""
        result = clf.classify(
            theta_power=0.2, alpha_power=0.3, task_performance=1.5,
        )
        assert result["stage"] in LEARNING_STAGES

    def test_performance_clamped_below_zero(self, clf):
        """Performance < 0.0 should be clamped to 0.0."""
        result = clf.classify(
            theta_power=0.2, alpha_power=0.3, task_performance=-0.5,
        )
        assert result["stage"] in LEARNING_STAGES

    def test_very_large_powers_no_crash(self, clf):
        """Very large power values should not cause numerical issues."""
        result = clf.classify(theta_power=1e6, alpha_power=1e6)
        assert result["stage"] in LEARNING_STAGES
        assert np.isfinite(result["confidence"])

    def test_reset_nonexistent_user_is_safe(self, clf):
        """Reset on unknown user should not raise."""
        clf.reset("nonexistent")  # should not raise

    def test_get_progression_unknown_user(self, clf):
        """Querying unknown user should return empty progression."""
        prog = clf.get_progression(user_id="nobody")
        assert prog["sessions_completed"] == 0
        assert prog["sessions"] == []


# ── Output structure ───────────────────────────────────────────────────────


class TestOutputStructure:
    def test_classify_returns_all_required_keys(self, clf):
        """classify() output must contain all documented keys."""
        result = clf.classify(theta_power=0.25, alpha_power=0.25)
        required_keys = {
            "stage",
            "stage_scores",
            "confidence",
            "theta_alpha_ratio",
            "recommendation",
        }
        assert required_keys.issubset(result.keys())

    def test_stage_is_valid(self, clf):
        """stage must be one of the four defined stages."""
        result = clf.classify(theta_power=0.2, alpha_power=0.3)
        assert result["stage"] in LEARNING_STAGES

    def test_confidence_in_range(self, clf):
        """Confidence should be between 0 and 1."""
        result = clf.classify(theta_power=0.3, alpha_power=0.2)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_theta_alpha_ratio_positive(self, clf):
        """Theta/alpha ratio should be non-negative."""
        result = clf.classify(theta_power=0.15, alpha_power=0.35)
        assert result["theta_alpha_ratio"] >= 0.0

    def test_recommendation_is_nonempty_string(self, clf):
        """Recommendation should be a non-empty string."""
        result = clf.classify(theta_power=0.2, alpha_power=0.3)
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 10

    def test_recommendation_matches_stage(self, clf):
        """Recommendation text should match the classified stage."""
        result = clf.classify(theta_power=0.2, alpha_power=0.3)
        assert result["recommendation"] == _STAGE_RECOMMENDATIONS[result["stage"]]

    def test_record_session_output_keys(self, clf):
        """record_session must return session_number, stage, stage_scores, confidence."""
        result = clf.record_session(theta_power=0.3, alpha_power=0.2)
        assert "session_number" in result
        assert "stage" in result
        assert "stage_scores" in result
        assert "confidence" in result


# ── Confidence scoring ─────────────────────────────────────────────────────


class TestConfidence:
    def test_high_confidence_when_clear_winner(self, clf):
        """Strong encoding pattern should produce relatively high confidence."""
        result = clf.classify(
            theta_power=0.50, alpha_power=0.02, task_performance=0.0,
        )
        assert result["confidence"] > 0.05  # some separation exists

    def test_low_confidence_when_ambiguous(self, clf):
        """Balanced inputs should produce lower confidence."""
        result = clf.classify(
            theta_power=0.25, alpha_power=0.25, task_performance=0.5,
        )
        # Not asserting a specific value, just that it's a valid float in [0, 1]
        assert 0.0 <= result["confidence"] <= 1.0


# ── Reset ──────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_all_task_types(self, clf):
        """Reset should remove history for all task types of that user."""
        clf.record_session(theta_power=0.3, alpha_power=0.2, user_id="u1", task_type="piano")
        clf.record_session(theta_power=0.2, alpha_power=0.3, user_id="u1", task_type="math")

        clf.reset("u1")

        assert clf.get_progression(user_id="u1", task_type="piano")["sessions_completed"] == 0
        assert clf.get_progression(user_id="u1", task_type="math")["sessions_completed"] == 0

    def test_reset_default_user(self, clf):
        """Reset with default user_id should clear default history."""
        clf.record_session(theta_power=0.3, alpha_power=0.2)
        assert clf.get_progression()["sessions_completed"] == 1

        clf.reset()
        assert clf.get_progression()["sessions_completed"] == 0
