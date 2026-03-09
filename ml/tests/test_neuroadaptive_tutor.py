"""Tests for neuroadaptive tutor."""
import pytest

from models.neuroadaptive_tutor import (
    NeuroadaptiveTutor, LEARNING_ZONES, INTERVENTIONS,
)


@pytest.fixture
def tutor():
    return NeuroadaptiveTutor()


class TestBasicAssessment:
    def test_output_keys(self, tutor):
        result = tutor.assess(0.3, 0.3, 0.5)
        expected = {
            "learning_zone", "zone_scores", "zone_confidence",
            "intervention", "difficulty_adjustment", "difficulty_level",
            "difficulty_label", "engagement_score", "break_recommended",
            "consecutive_zone", "session_minutes", "n_samples",
        }
        assert expected.issubset(set(result.keys()))

    def test_zone_is_valid(self, tutor):
        result = tutor.assess(0.3, 0.3, 0.5)
        assert result["learning_zone"] in LEARNING_ZONES

    def test_intervention_matches_zone(self, tutor):
        result = tutor.assess(0.3, 0.3, 0.5)
        assert result["intervention"] == INTERVENTIONS[result["learning_zone"]]

    def test_engagement_score_range(self, tutor):
        result = tutor.assess(0.3, 0.3, 0.5)
        assert 0 <= result["engagement_score"] <= 1

    def test_difficulty_range(self, tutor):
        result = tutor.assess(0.3, 0.3, 0.5)
        assert 0 <= result["difficulty_level"] <= 1

    def test_n_samples_increments(self, tutor):
        tutor.assess(0.3, 0.3, 0.5)
        result = tutor.assess(0.3, 0.3, 0.5)
        assert result["n_samples"] == 2


class TestLearningZones:
    def test_boredom_detection(self, tutor):
        # High alpha + low beta + low theta → boredom
        result = tutor.assess(theta_power=0.1, alpha_power=0.9, beta_power=0.1)
        assert result["learning_zone"] == "boredom"

    def test_confusion_detection(self, tutor):
        # High theta + moderate beta + low alpha → confusion
        result = tutor.assess(theta_power=0.8, alpha_power=0.1, beta_power=0.3)
        assert result["learning_zone"] == "confusion"

    def test_flow_detection(self, tutor):
        # High beta + moderate theta + low alpha → flow
        result = tutor.assess(theta_power=0.3, alpha_power=0.1, beta_power=0.9)
        assert result["learning_zone"] == "flow"

    def test_frustration_with_fatigue(self, tutor):
        # Very high beta + moderate theta + high fatigue → frustration
        result = tutor.assess(
            theta_power=0.5, alpha_power=0.1, beta_power=0.9, fatigue_index=0.9
        )
        assert result["learning_zone"] == "frustration"


class TestDifficultyAdaptation:
    def test_boredom_increases_difficulty(self, tutor):
        initial = 0.5
        # Boredom should push difficulty up
        for _ in range(20):
            result = tutor.assess(0.1, 0.9, 0.1)
        assert result["difficulty_level"] > initial

    def test_confusion_decreases_difficulty(self, tutor):
        initial = 0.5
        for _ in range(20):
            result = tutor.assess(0.8, 0.1, 0.3)
        assert result["difficulty_level"] < initial

    def test_difficulty_bounded(self, tutor):
        for _ in range(200):
            result = tutor.assess(0.1, 0.9, 0.1)
        assert result["difficulty_level"] <= 1.0
        tutor.reset()
        for _ in range(200):
            result = tutor.assess(0.8, 0.1, 0.3)
        assert result["difficulty_level"] >= 0.0

    def test_difficulty_labels(self, tutor):
        result = tutor.assess(0.3, 0.3, 0.5)
        assert result["difficulty_label"] in ("easy", "moderate", "hard")


class TestBreakRecommendation:
    def test_no_break_initially(self, tutor):
        result = tutor.assess(0.3, 0.3, 0.5)
        assert result["break_recommended"] is False

    def test_break_on_high_fatigue(self, tutor):
        # Fill cooldown
        for _ in range(55):
            tutor.assess(0.3, 0.3, 0.5, fatigue_index=0.0)
        result = tutor.assess(0.3, 0.3, 0.5, fatigue_index=0.8)
        assert result["break_recommended"] is True

    def test_break_on_prolonged_frustration(self, tutor):
        # 10+ consecutive frustration epochs after cooldown
        for _ in range(50):
            tutor.assess(0.3, 0.3, 0.5)
        for _ in range(10):
            result = tutor.assess(0.7, 0.1, 0.7, fatigue_index=0.8)
        assert result["break_recommended"] is True

    def test_acknowledge_break_resets_cooldown(self, tutor):
        for _ in range(60):
            tutor.assess(0.3, 0.3, 0.5, fatigue_index=0.0)
        tutor.acknowledge_break()
        result = tutor.assess(0.3, 0.3, 0.5, fatigue_index=0.8)
        # Cooldown just reset, so no break yet
        assert result["break_recommended"] is False


class TestConsecutiveZone:
    def test_consecutive_increments(self, tutor):
        tutor.assess(0.1, 0.9, 0.1)  # boredom
        result = tutor.assess(0.1, 0.9, 0.1)  # boredom again
        assert result["consecutive_zone"] == 2

    def test_consecutive_resets_on_change(self, tutor):
        tutor.assess(0.1, 0.9, 0.1)  # boredom
        result = tutor.assess(0.3, 0.1, 0.9)  # flow
        assert result["consecutive_zone"] == 1


class TestSessionSummary:
    def test_empty_summary(self, tutor):
        assert tutor.get_session_summary()["n_samples"] == 0

    def test_summary_with_data(self, tutor):
        for _ in range(5):
            tutor.assess(0.3, 0.3, 0.5)
        s = tutor.get_session_summary()
        assert s["n_samples"] == 5
        assert "mean_engagement" in s
        assert "dominant_zone" in s

    def test_zone_percentages_sum(self, tutor):
        for _ in range(10):
            tutor.assess(0.3, 0.3, 0.5)
        s = tutor.get_session_summary()
        total = (s["boredom_pct"] + s["confusion_pct"] + s["frustration_pct"]
                 + s["optimal_pct"] + s["flow_pct"])
        assert abs(total - 100.0) < 1.0


class TestHistory:
    def test_empty_history(self, tutor):
        assert tutor.get_history() == []

    def test_history_grows(self, tutor):
        tutor.assess(0.3, 0.3, 0.5)
        tutor.assess(0.3, 0.3, 0.5)
        assert len(tutor.get_history()) == 2

    def test_history_last_n(self, tutor):
        for _ in range(10):
            tutor.assess(0.3, 0.3, 0.5)
        assert len(tutor.get_history(last_n=3)) == 3

    def test_history_cap(self):
        t = NeuroadaptiveTutor(max_history=50)
        for _ in range(60):
            t.assess(0.3, 0.3, 0.5)
        assert len(t.get_history()) == 50


class TestMultiUser:
    def test_independent_users(self, tutor):
        tutor.assess(0.1, 0.9, 0.1, user_id="alice")
        tutor.assess(0.3, 0.1, 0.9, user_id="bob")
        alice = tutor.get_session_summary("alice")
        bob = tutor.get_session_summary("bob")
        assert alice["n_samples"] == 1
        assert bob["n_samples"] == 1


class TestReset:
    def test_reset_clears(self, tutor):
        tutor.assess(0.3, 0.3, 0.5)
        tutor.reset()
        assert tutor.get_session_summary()["n_samples"] == 0
        assert tutor.get_history() == []
