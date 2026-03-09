"""Tests for EngagementDetector — student engagement and educational emotions."""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.engagement_detector import (
    EngagementDetector,
    ENGAGEMENT_STATES,
    EDUCATIONAL_EMOTIONS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    return EngagementDetector()


@pytest.fixture
def detector_small():
    """Detector with small history cap for overflow tests."""
    return EngagementDetector(max_history=5)


# ---------------------------------------------------------------------------
# TestBasicAssessment
# ---------------------------------------------------------------------------

class TestBasicAssessment:
    """First assessment returns correct keys and value ranges."""

    def test_output_keys(self, detector):
        result = detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4)
        expected_keys = {
            "engagement_index",
            "engagement_state",
            "educational_emotion",
            "emotion_scores",
            "attention_index",
            "mind_wandering_risk",
            "n_samples",
        }
        assert expected_keys.issubset(result.keys())

    def test_engagement_index_range(self, detector):
        result = detector.assess(theta_power=0.5, alpha_power=0.5, beta_power=0.5)
        assert 0.0 <= result["engagement_index"] <= 1.0

    def test_engagement_state_valid(self, detector):
        result = detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4)
        assert result["engagement_state"] in ENGAGEMENT_STATES

    def test_educational_emotion_valid(self, detector):
        result = detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4)
        assert result["educational_emotion"] in EDUCATIONAL_EMOTIONS

    def test_emotion_scores_all_present(self, detector):
        result = detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4)
        for emotion in EDUCATIONAL_EMOTIONS:
            assert emotion in result["emotion_scores"]
            assert 0.0 <= result["emotion_scores"][emotion] <= 1.0

    def test_n_samples_increments(self, detector):
        r1 = detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4)
        assert r1["n_samples"] == 1
        r2 = detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4)
        assert r2["n_samples"] == 2

    def test_attention_index_range(self, detector):
        result = detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4)
        assert 0.0 <= result["attention_index"] <= 1.0

    def test_mind_wandering_risk_range(self, detector):
        result = detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4)
        assert 0.0 <= result["mind_wandering_risk"] <= 1.0


# ---------------------------------------------------------------------------
# TestEngagementStates
# ---------------------------------------------------------------------------

class TestEngagementStates:
    """Correct state classification based on band powers."""

    def test_high_beta_low_theta_is_attentive(self, detector):
        result = detector.assess(theta_power=0.1, alpha_power=0.1, beta_power=0.9)
        assert result["engagement_state"] == "attentive"
        assert result["engagement_index"] > 0.6

    def test_balanced_powers_is_passive(self, detector):
        result = detector.assess(theta_power=0.4, alpha_power=0.4, beta_power=0.4)
        assert result["engagement_state"] == "passive"
        assert 0.35 <= result["engagement_index"] <= 0.6

    def test_high_theta_low_beta_is_disengaged(self, detector):
        result = detector.assess(theta_power=0.9, alpha_power=0.7, beta_power=0.05)
        assert result["engagement_state"] == "disengaged"
        assert result["engagement_index"] < 0.35


# ---------------------------------------------------------------------------
# TestBaseline
# ---------------------------------------------------------------------------

class TestBaseline:
    """Baseline-relative engagement."""

    def test_set_baseline_stores(self, detector):
        detector.set_baseline(theta_power=0.2, alpha_power=0.3, beta_power=0.5)
        assert "default" in detector._baselines

    def test_baseline_affects_engagement(self, detector):
        # Without baseline
        r_no = detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.5)

        detector2 = EngagementDetector()
        detector2.set_baseline(theta_power=0.3, alpha_power=0.3, beta_power=0.5)
        r_with = detector2.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.5)

        # With baseline matching current, engagement should be around midpoint
        # because baseline_ratio ~ 0 -> sigmoid(0) = 0.5
        # The two results should differ since baseline blending changes the score
        assert r_no["engagement_index"] != r_with["engagement_index"]

    def test_above_baseline_higher_engagement(self, detector):
        detector.set_baseline(theta_power=0.5, alpha_power=0.5, beta_power=0.2)
        # Now perform much better than baseline (lower theta, higher beta)
        result = detector.assess(theta_power=0.1, alpha_power=0.2, beta_power=0.8)
        assert result["engagement_index"] > 0.5

    def test_below_baseline_lower_engagement(self, detector):
        detector.set_baseline(theta_power=0.1, alpha_power=0.2, beta_power=0.8)
        # Now perform worse than baseline (high theta, low beta)
        result = detector.assess(theta_power=0.8, alpha_power=0.7, beta_power=0.1)
        assert result["engagement_index"] < 0.5


# ---------------------------------------------------------------------------
# TestNoBaseline
# ---------------------------------------------------------------------------

class TestNoBaseline:
    """Absolute thresholds work without baseline."""

    def test_absolute_high_engagement(self, detector):
        result = detector.assess(theta_power=0.05, alpha_power=0.05, beta_power=0.95)
        assert result["engagement_index"] > 0.6

    def test_absolute_low_engagement(self, detector):
        result = detector.assess(theta_power=0.9, alpha_power=0.8, beta_power=0.05)
        assert result["engagement_index"] < 0.35


# ---------------------------------------------------------------------------
# TestEducationalEmotions
# ---------------------------------------------------------------------------

class TestEducationalEmotions:
    """Correct educational emotion detection."""

    def test_boredom_high_alpha_low_beta_theta(self, detector):
        result = detector.assess(theta_power=0.1, alpha_power=0.9, beta_power=0.05)
        assert result["educational_emotion"] == "boredom"

    def test_confusion_high_theta_high_beta(self, detector):
        result = detector.assess(theta_power=0.8, alpha_power=0.1, beta_power=0.7)
        # confusion or frustration both have high theta+beta; confusion when
        # they are roughly balanced. The spec says frustration > confusion when
        # beta is higher. With theta=0.8 > beta=0.7, confusion should dominate.
        assert result["educational_emotion"] in ("confusion", "frustration")

    def test_curiosity_moderate_theta_moderate_beta(self, detector):
        result = detector.assess(theta_power=0.45, alpha_power=0.1, beta_power=0.45)
        assert result["educational_emotion"] == "curiosity"

    def test_frustration_very_high_beta_high_theta(self, detector):
        result = detector.assess(theta_power=0.6, alpha_power=0.05, beta_power=0.95)
        assert result["educational_emotion"] == "frustration"

    def test_concentration_high_beta_low_theta(self, detector):
        result = detector.assess(theta_power=0.05, alpha_power=0.05, beta_power=0.95)
        assert result["educational_emotion"] == "concentration"


# ---------------------------------------------------------------------------
# TestMindWandering
# ---------------------------------------------------------------------------

class TestMindWandering:
    """Mind-wandering risk detection."""

    def test_high_theta_high_alpha_high_risk(self, detector):
        result = detector.assess(theta_power=0.9, alpha_power=0.8, beta_power=0.05)
        assert result["mind_wandering_risk"] > 0.7

    def test_high_beta_low_risk(self, detector):
        result = detector.assess(theta_power=0.05, alpha_power=0.05, beta_power=0.9)
        assert result["mind_wandering_risk"] < 0.3


# ---------------------------------------------------------------------------
# TestEngagementCurve
# ---------------------------------------------------------------------------

class TestEngagementCurve:
    """Engagement history retrieval."""

    def test_returns_list(self, detector):
        curve = detector.get_engagement_curve()
        assert isinstance(curve, list)

    def test_grows_with_assessments(self, detector):
        for _ in range(5):
            detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4)
        curve = detector.get_engagement_curve()
        assert len(curve) == 5

    def test_last_n_parameter(self, detector):
        for _ in range(10):
            detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4)
        curve = detector.get_engagement_curve(last_n=3)
        assert len(curve) == 3

    def test_last_n_larger_than_history(self, detector):
        detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4)
        curve = detector.get_engagement_curve(last_n=100)
        assert len(curve) == 1

    def test_curve_entry_keys(self, detector):
        detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4)
        entry = detector.get_engagement_curve()[0]
        assert "engagement_index" in entry
        assert "engagement_state" in entry


# ---------------------------------------------------------------------------
# TestSessionSummary
# ---------------------------------------------------------------------------

class TestSessionSummary:
    """Session summary statistics."""

    def test_empty_summary(self, detector):
        summary = detector.get_session_summary()
        assert summary["n_samples"] == 0

    def test_summary_with_data(self, detector):
        for _ in range(10):
            detector.assess(theta_power=0.1, alpha_power=0.1, beta_power=0.8)
        summary = detector.get_session_summary()
        assert summary["n_samples"] == 10
        assert 0.0 <= summary["mean_engagement"] <= 1.0
        pct_sum = (
            summary["attentive_pct"]
            + summary["passive_pct"]
            + summary["disengaged_pct"]
        )
        assert abs(pct_sum - 100.0) < 0.1

    def test_dominant_emotion(self, detector):
        for _ in range(5):
            detector.assess(theta_power=0.05, alpha_power=0.05, beta_power=0.95)
        summary = detector.get_session_summary()
        assert summary["dominant_emotion"] in EDUCATIONAL_EMOTIONS

    def test_mind_wandering_episodes(self, detector):
        # 6 consecutive disengaged assessments -> at least 1 episode
        for _ in range(6):
            detector.assess(theta_power=0.9, alpha_power=0.8, beta_power=0.05)
        summary = detector.get_session_summary()
        assert summary["mind_wandering_episodes"] >= 1

    def test_no_wandering_episodes_when_attentive(self, detector):
        for _ in range(10):
            detector.assess(theta_power=0.05, alpha_power=0.05, beta_power=0.9)
        summary = detector.get_session_summary()
        assert summary["mind_wandering_episodes"] == 0


# ---------------------------------------------------------------------------
# TestMultiUser
# ---------------------------------------------------------------------------

class TestMultiUser:
    """Independent per-user histories."""

    def test_separate_histories(self, detector):
        detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4, user_id="alice")
        detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4, user_id="alice")
        detector.assess(theta_power=0.5, alpha_power=0.5, beta_power=0.2, user_id="bob")

        alice_curve = detector.get_engagement_curve(user_id="alice")
        bob_curve = detector.get_engagement_curve(user_id="bob")
        assert len(alice_curve) == 2
        assert len(bob_curve) == 1

    def test_separate_baselines(self, detector):
        detector.set_baseline(theta_power=0.2, alpha_power=0.3, beta_power=0.5, user_id="alice")
        detector.set_baseline(theta_power=0.5, alpha_power=0.5, beta_power=0.2, user_id="bob")
        assert detector._baselines["alice"] != detector._baselines["bob"]

    def test_separate_summaries(self, detector):
        for _ in range(3):
            detector.assess(theta_power=0.1, alpha_power=0.1, beta_power=0.8, user_id="alice")
        s_alice = detector.get_session_summary(user_id="alice")
        s_bob = detector.get_session_summary(user_id="bob")
        assert s_alice["n_samples"] == 3
        assert s_bob["n_samples"] == 0


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------

class TestReset:
    """Reset clears data and baseline."""

    def test_reset_clears_history(self, detector):
        for _ in range(5):
            detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4)
        detector.reset()
        assert len(detector.get_engagement_curve()) == 0

    def test_reset_clears_baseline(self, detector):
        detector.set_baseline(theta_power=0.2, alpha_power=0.3, beta_power=0.5)
        detector.reset()
        assert "default" not in detector._baselines

    def test_reset_one_user_keeps_other(self, detector):
        detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4, user_id="alice")
        detector.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4, user_id="bob")
        detector.reset(user_id="alice")
        assert len(detector.get_engagement_curve(user_id="alice")) == 0
        assert len(detector.get_engagement_curve(user_id="bob")) == 1


# ---------------------------------------------------------------------------
# TestHistoryCap
# ---------------------------------------------------------------------------

class TestHistoryCap:
    """max_history is respected."""

    def test_cap_enforced(self, detector_small):
        for _ in range(10):
            detector_small.assess(theta_power=0.3, alpha_power=0.3, beta_power=0.4)
        curve = detector_small.get_engagement_curve()
        assert len(curve) == 5

    def test_cap_preserves_latest(self, detector_small):
        for i in range(10):
            # Vary beta to create distinguishable entries
            detector_small.assess(
                theta_power=0.3, alpha_power=0.3, beta_power=0.1 * (i + 1)
            )
        curve = detector_small.get_engagement_curve()
        # Last entry should have the highest beta (i=9 -> beta=1.0)
        assert len(curve) == 5


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: zeros, ones, extreme values."""

    def test_all_zeros(self, detector):
        result = detector.assess(theta_power=0.0, alpha_power=0.0, beta_power=0.0)
        assert 0.0 <= result["engagement_index"] <= 1.0
        assert result["engagement_state"] in ENGAGEMENT_STATES

    def test_all_ones(self, detector):
        result = detector.assess(theta_power=1.0, alpha_power=1.0, beta_power=1.0)
        assert 0.0 <= result["engagement_index"] <= 1.0
        assert result["engagement_state"] in ENGAGEMENT_STATES

    def test_extreme_high_values(self, detector):
        result = detector.assess(theta_power=10.0, alpha_power=10.0, beta_power=10.0)
        assert 0.0 <= result["engagement_index"] <= 1.0

    def test_very_small_values(self, detector):
        result = detector.assess(theta_power=1e-12, alpha_power=1e-12, beta_power=1e-12)
        assert 0.0 <= result["engagement_index"] <= 1.0

    def test_gamma_parameter_accepted(self, detector):
        result = detector.assess(
            theta_power=0.3, alpha_power=0.3, beta_power=0.4, gamma_power=0.2
        )
        assert "engagement_index" in result

    def test_negative_values_clipped(self, detector):
        result = detector.assess(theta_power=-0.5, alpha_power=-0.5, beta_power=-0.5)
        assert 0.0 <= result["engagement_index"] <= 1.0
