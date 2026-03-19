"""Tests for emotion-aware adaptive education (issue #450)."""

import pytest

from models.adaptive_education import (
    LEARNING_STATES,
    DIFFICULTY_ACTIONS,
    PACING_ACTIONS,
    LearningEEGFeatures,
    LearningState,
    EducationProfile,
    detect_learning_state,
    recommend_difficulty_adjustment,
    recommend_pacing,
    track_attention_span,
    compute_learning_windows,
    compute_education_profile,
    profile_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _features(
    theta=0.3, alpha=0.3, beta=0.3, valence=0.0, fatigue=0.0, timestamp=0.0,
):
    return LearningEEGFeatures(
        theta=theta, alpha=alpha, beta=beta,
        valence=valence, fatigue=fatigue, timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# detect_learning_state
# ---------------------------------------------------------------------------


class TestDetectLearningState:
    def test_returns_learning_state(self):
        result = detect_learning_state(_features())
        assert isinstance(result, LearningState)

    def test_state_in_valid_set(self):
        result = detect_learning_state(_features())
        assert result.state in LEARNING_STATES

    def test_confidence_range(self):
        result = detect_learning_state(_features())
        assert 0.0 <= result.confidence <= 1.0

    def test_scores_cover_all_states(self):
        result = detect_learning_state(_features())
        assert set(result.scores.keys()) == set(LEARNING_STATES)

    def test_engagement_range(self):
        result = detect_learning_state(_features())
        assert 0.0 <= result.engagement_level <= 1.0

    def test_bored_detection(self):
        """High alpha, low beta, low theta -> bored."""
        result = detect_learning_state(_features(theta=0.1, alpha=0.9, beta=0.1))
        assert result.state == "bored"

    def test_confused_detection(self):
        """High theta, high beta (frontal activation) -> confused."""
        result = detect_learning_state(_features(theta=0.9, alpha=0.1, beta=0.7))
        assert result.state == "confused"

    def test_frustrated_detection(self):
        """High beta, negative valence, fatigue -> frustrated."""
        result = detect_learning_state(
            _features(theta=0.5, alpha=0.1, beta=0.8, valence=-0.8, fatigue=0.9)
        )
        assert result.state == "frustrated"

    def test_engaged_detection(self):
        """Moderate beta, low theta, low alpha -> engaged."""
        result = detect_learning_state(
            _features(theta=0.15, alpha=0.15, beta=0.45, valence=0.3, fatigue=0.0)
        )
        assert result.state == "engaged"

    def test_flow_detection(self):
        """Balanced alpha/theta (alpha-theta border), moderate beta, strong positive valence, low fatigue."""
        result = detect_learning_state(
            _features(theta=0.4, alpha=0.4, beta=0.4, valence=0.9, fatigue=0.0)
        )
        assert result.state == "flow"

    def test_difficulty_recommendation_matches_state(self):
        for theta, alpha, beta in [(0.1, 0.9, 0.1), (0.9, 0.1, 0.7), (0.3, 0.3, 0.5)]:
            result = detect_learning_state(_features(theta=theta, alpha=alpha, beta=beta))
            assert result.difficulty_recommendation == DIFFICULTY_ACTIONS[result.state]

    def test_pacing_recommendation_matches_state(self):
        for theta, alpha, beta in [(0.1, 0.9, 0.1), (0.9, 0.1, 0.7), (0.3, 0.3, 0.5)]:
            result = detect_learning_state(_features(theta=theta, alpha=alpha, beta=beta))
            assert result.pacing_recommendation == PACING_ACTIONS[result.state]

    def test_clamps_out_of_range_inputs(self):
        """Features outside 0-1 are clamped, not errors."""
        result = detect_learning_state(_features(theta=2.0, alpha=-0.5, beta=1.5))
        assert result.state in LEARNING_STATES


# ---------------------------------------------------------------------------
# recommend_difficulty_adjustment
# ---------------------------------------------------------------------------


class TestDifficultyAdjustment:
    def test_bored_increases_difficulty(self):
        state = detect_learning_state(_features(theta=0.1, alpha=0.9, beta=0.1))
        adj = recommend_difficulty_adjustment(state, 0.5)
        assert adj["new_difficulty"] > adj["previous_difficulty"]

    def test_confused_decreases_difficulty(self):
        state = detect_learning_state(_features(theta=0.9, alpha=0.1, beta=0.7))
        adj = recommend_difficulty_adjustment(state, 0.5)
        assert adj["new_difficulty"] < adj["previous_difficulty"]

    def test_frustrated_decreases_difficulty(self):
        state = detect_learning_state(
            _features(theta=0.5, alpha=0.1, beta=0.8, valence=-0.8, fatigue=0.9)
        )
        adj = recommend_difficulty_adjustment(state, 0.5)
        assert adj["new_difficulty"] < adj["previous_difficulty"]

    def test_difficulty_bounded_0_1(self):
        # Push low
        state = detect_learning_state(_features(theta=0.9, alpha=0.1, beta=0.7))
        adj = recommend_difficulty_adjustment(state, 0.01)
        assert adj["new_difficulty"] >= 0.0

        # Push high
        state = detect_learning_state(_features(theta=0.1, alpha=0.9, beta=0.1))
        adj = recommend_difficulty_adjustment(state, 0.99)
        assert adj["new_difficulty"] <= 1.0

    def test_output_keys(self):
        state = detect_learning_state(_features())
        adj = recommend_difficulty_adjustment(state, 0.5)
        assert "action" in adj
        assert "adjustment" in adj
        assert "previous_difficulty" in adj
        assert "new_difficulty" in adj
        assert "reason" in adj


# ---------------------------------------------------------------------------
# recommend_pacing
# ---------------------------------------------------------------------------


class TestPacing:
    def test_bored_speeds_up(self):
        state = detect_learning_state(_features(theta=0.1, alpha=0.9, beta=0.1))
        pacing = recommend_pacing(state)
        assert pacing["pacing"] == "speed_up"
        assert pacing["speed_factor"] > 1.0

    def test_confused_slows_down(self):
        state = detect_learning_state(_features(theta=0.9, alpha=0.1, beta=0.7))
        pacing = recommend_pacing(state)
        assert pacing["pacing"] == "slow_down"
        assert pacing["speed_factor"] < 1.0

    def test_frustrated_suggests_break(self):
        state = detect_learning_state(
            _features(theta=0.5, alpha=0.1, beta=0.8, valence=-0.8, fatigue=0.9)
        )
        pacing = recommend_pacing(state)
        assert pacing["pacing"] == "take_break"
        assert pacing["speed_factor"] == 0.0

    def test_prolonged_frustration_forces_break(self):
        state = detect_learning_state(
            _features(theta=0.5, alpha=0.1, beta=0.8, valence=-0.8, fatigue=0.9)
        )
        pacing = recommend_pacing(state, consecutive_same_state=5)
        assert pacing["pacing"] == "take_break"

    def test_session_time_break_suggestion(self):
        state = detect_learning_state(_features(theta=0.3, alpha=0.3, beta=0.5))
        pacing = recommend_pacing(state, session_minutes=30.0)
        # Should mention break in suggestions for >25 min
        assert any("break" in s.lower() for s in pacing["suggestions"])


# ---------------------------------------------------------------------------
# track_attention_span
# ---------------------------------------------------------------------------


class TestAttentionSpan:
    def test_insufficient_data(self):
        result = track_attention_span([0.5])
        assert result["trend"] == "insufficient_data"
        assert result["attention_span_minutes"] == 0.0

    def test_stable_engagement(self):
        # Constant engagement -> no decline
        history = [0.7] * 30
        result = track_attention_span(history)
        assert result["is_declining"] is False

    def test_declining_engagement_detected(self):
        # Start high, drop to low
        history = [0.8] * 20 + [0.4] * 20
        result = track_attention_span(history, window_size=10, decline_threshold=0.15)
        assert result["is_declining"] is True
        assert result["attention_span_minutes"] > 0

    def test_peak_and_current(self):
        history = [0.3, 0.5, 0.9, 0.7, 0.6]
        result = track_attention_span(history)
        assert result["peak_engagement"] == 0.9
        assert result["current_engagement"] == 0.6

    def test_empty_history(self):
        result = track_attention_span([])
        assert result["peak_engagement"] == 0.0


# ---------------------------------------------------------------------------
# compute_learning_windows
# ---------------------------------------------------------------------------


class TestLearningWindows:
    def test_empty_data(self):
        result = compute_learning_windows({})
        assert result["optimal_hours"] == []

    def test_identifies_best_hours(self):
        hourly = {
            9: [0.8, 0.85, 0.9],
            14: [0.5, 0.4, 0.45],
            21: [0.3, 0.35, 0.25],
        }
        result = compute_learning_windows(hourly)
        assert 9 in result["optimal_hours"]
        assert 21 in result["worst_hours"]

    def test_hourly_means_computed(self):
        hourly = {10: [0.6, 0.8], 15: [0.4, 0.5]}
        result = compute_learning_windows(hourly)
        assert 10 in result["hourly_means"]
        assert 15 in result["hourly_means"]
        assert abs(result["hourly_means"][10] - 0.7) < 0.01


# ---------------------------------------------------------------------------
# compute_education_profile
# ---------------------------------------------------------------------------


class TestEducationProfile:
    def test_empty_history(self):
        profile = compute_education_profile([])
        assert profile.n_samples == 0

    def test_basic_profile(self):
        history = [_features(theta=0.3, alpha=0.3, beta=0.5) for _ in range(10)]
        profile = compute_education_profile(history)
        assert profile.n_samples == 10
        assert 0.0 <= profile.session_engagement_mean <= 1.0
        assert 0.0 <= profile.difficulty_level <= 1.0
        assert profile.learning_state.state in LEARNING_STATES

    def test_state_distribution_sums_to_100(self):
        history = [_features(theta=0.3, alpha=0.3, beta=0.5) for _ in range(20)]
        profile = compute_education_profile(history)
        total = sum(profile.state_distribution.values())
        assert abs(total - 100.0) < 1.0

    def test_recommendations_generated(self):
        history = [_features(theta=0.3, alpha=0.3, beta=0.5) for _ in range(10)]
        profile = compute_education_profile(history)
        assert len(profile.recommendations) >= 1

    def test_with_hourly_engagement(self):
        history = [_features(theta=0.3, alpha=0.3, beta=0.5) for _ in range(5)]
        hourly = {9: [0.8, 0.9], 14: [0.4, 0.5]}
        profile = compute_education_profile(history, hourly)
        assert len(profile.optimal_window_hours) > 0


# ---------------------------------------------------------------------------
# profile_to_dict
# ---------------------------------------------------------------------------


class TestProfileToDict:
    def test_serializable(self):
        history = [_features(theta=0.3, alpha=0.3, beta=0.5) for _ in range(5)]
        profile = compute_education_profile(history)
        d = profile_to_dict(profile)
        assert isinstance(d, dict)
        assert "learning_state" in d
        assert "state" in d["learning_state"]
        assert "n_samples" in d
        assert "recommendations" in d

    def test_empty_profile_serializable(self):
        profile = EducationProfile()
        d = profile_to_dict(profile)
        assert d["n_samples"] == 0
        assert d["learning_state"]["state"] == "unknown"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_all_states_have_difficulty_action(self):
        for s in LEARNING_STATES:
            assert s in DIFFICULTY_ACTIONS

    def test_all_states_have_pacing_action(self):
        for s in LEARNING_STATES:
            assert s in PACING_ACTIONS
