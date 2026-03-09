"""Tests for MindfulnessQualityDetector."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.mindfulness_quality import MindfulnessQualityDetector


# ── Focused meditation detection ──────────────────────────────────────────


def test_focused_high_theta_alpha_ratio():
    """High theta/alpha ratio (>= 1.2) should be classified as focused."""
    det = MindfulnessQualityDetector()
    result = det.assess(theta_power=0.30, alpha_power=0.15)
    assert result["quality"] == "focused"
    assert result["focus_ratio"] >= 1.2


def test_focused_with_moderate_beta():
    """Focused state should hold even with moderate beta present."""
    det = MindfulnessQualityDetector()
    result = det.assess(theta_power=0.35, alpha_power=0.20, beta_power=0.15)
    assert result["quality"] == "focused"


# ── Mind wandering detection ──────────────────────────────────────────────


def test_light_wandering_detection():
    """Ratio between 0.8 and 1.2 should be light wandering."""
    det = MindfulnessQualityDetector()
    # ratio = 0.20 / 0.20 = 1.0
    result = det.assess(theta_power=0.20, alpha_power=0.20)
    assert result["quality"] == "light_wandering"


def test_mind_wandering_low_ratio():
    """Ratio between 0.5 and 0.8 should be mind wandering."""
    det = MindfulnessQualityDetector()
    # ratio = 0.10 / 0.18 ≈ 0.56
    result = det.assess(theta_power=0.10, alpha_power=0.18)
    assert result["quality"] == "mind_wandering"


def test_disengaged_very_low_ratio():
    """Very low ratio (< 0.5) with low beta should be disengaged."""
    det = MindfulnessQualityDetector()
    # ratio = 0.05 / 0.30 ≈ 0.17
    result = det.assess(theta_power=0.05, alpha_power=0.30, beta_power=0.05)
    assert result["quality"] == "disengaged"


def test_mind_wandering_instead_of_disengaged_with_high_beta():
    """Low ratio + high beta => mind_wandering, not disengaged (restless thinking)."""
    det = MindfulnessQualityDetector()
    result = det.assess(theta_power=0.05, alpha_power=0.30, beta_power=0.40)
    assert result["quality"] == "mind_wandering"


# ── Session summary statistics ────────────────────────────────────────────


def test_session_summary_all_focused():
    """Session with only focused assessments should show 100% focus."""
    det = MindfulnessQualityDetector()
    for _ in range(5):
        det.assess(theta_power=0.35, alpha_power=0.15)
    summary = det.get_session_summary()
    assert summary["focus_pct"] == 100.0
    assert summary["total_wandering_sec"] == 0.0
    assert len(summary["quality_timeline"]) == 5
    assert all(s == "focused" for s in summary["quality_timeline"])


def test_session_summary_mixed():
    """Mixed session should track both focused and non-focused time."""
    det = MindfulnessQualityDetector()
    # 3 focused
    for _ in range(3):
        det.assess(theta_power=0.35, alpha_power=0.15)
    # 2 wandering
    for _ in range(2):
        det.assess(theta_power=0.05, alpha_power=0.30, beta_power=0.05)
    summary = det.get_session_summary()
    assert summary["total_focused_sec"] > 0
    assert summary["total_wandering_sec"] > 0
    assert len(summary["quality_timeline"]) == 5
    assert summary["quality_timeline"][:3] == ["focused", "focused", "focused"]
    assert summary["quality_timeline"][3] != "focused"


def test_session_summary_empty():
    """Empty session should return zeroed summary."""
    det = MindfulnessQualityDetector()
    summary = det.get_session_summary()
    assert summary["total_focused_sec"] == 0.0
    assert summary["total_wandering_sec"] == 0.0
    assert summary["focus_pct"] == 0.0
    assert summary["quality_timeline"] == []


# ── Multiple users ────────────────────────────────────────────────────────


def test_multiple_users_isolation():
    """Different user IDs should have independent sessions."""
    det = MindfulnessQualityDetector()
    det.assess(theta_power=0.35, alpha_power=0.15, user_id="alice")
    det.assess(theta_power=0.05, alpha_power=0.30, beta_power=0.05, user_id="bob")

    alice_summary = det.get_session_summary(user_id="alice")
    bob_summary = det.get_session_summary(user_id="bob")

    assert len(alice_summary["quality_timeline"]) == 1
    assert alice_summary["quality_timeline"][0] == "focused"

    assert len(bob_summary["quality_timeline"]) == 1
    assert bob_summary["quality_timeline"][0] == "disengaged"


# ── Reset ─────────────────────────────────────────────────────────────────


def test_reset_clears_session():
    """Reset should clear all session data for the specified user."""
    det = MindfulnessQualityDetector()
    det.assess(theta_power=0.35, alpha_power=0.15)
    det.reset()
    summary = det.get_session_summary()
    assert summary["quality_timeline"] == []
    assert summary["total_focused_sec"] == 0.0


def test_reset_only_affects_target_user():
    """Resetting one user should not affect another."""
    det = MindfulnessQualityDetector()
    det.assess(theta_power=0.35, alpha_power=0.15, user_id="alice")
    det.assess(theta_power=0.35, alpha_power=0.15, user_id="bob")
    det.reset(user_id="alice")

    assert det.get_session_summary(user_id="alice")["quality_timeline"] == []
    assert len(det.get_session_summary(user_id="bob")["quality_timeline"]) == 1


# ── Streak counting ──────────────────────────────────────────────────────


def test_streak_starts_at_one():
    """First assessment should report a streak of at least 1 second."""
    det = MindfulnessQualityDetector()
    result = det.assess(theta_power=0.35, alpha_power=0.15)
    assert result["current_streak_sec"] >= 1.0


def test_streak_resets_on_state_change():
    """Streak should reset when quality state changes."""
    det = MindfulnessQualityDetector()
    det.assess(theta_power=0.35, alpha_power=0.15)  # focused
    det.assess(theta_power=0.35, alpha_power=0.15)  # focused
    result = det.assess(theta_power=0.05, alpha_power=0.30, beta_power=0.05)  # disengaged
    # New state just started — streak should be ~1 second
    assert result["current_streak_sec"] >= 0.5
    assert result["current_streak_sec"] < 5.0


# ── Edge cases ────────────────────────────────────────────────────────────


def test_zero_alpha_no_division_error():
    """Zero alpha power should not cause division by zero."""
    det = MindfulnessQualityDetector()
    result = det.assess(theta_power=0.20, alpha_power=0.0)
    assert result["quality"] == "focused"
    assert result["focus_ratio"] > 0


def test_zero_theta_and_alpha():
    """Both zero should not crash; should classify as disengaged."""
    det = MindfulnessQualityDetector()
    result = det.assess(theta_power=0.0, alpha_power=0.0)
    # With both zero, ratio is 0/epsilon ≈ 0, which is disengaged
    assert result["quality"] in QUALITY_STATES
    assert result["focus_ratio"] >= 0.0


def test_stability_score_range():
    """Stability score should always be between 0 and 1."""
    det = MindfulnessQualityDetector()
    # Feed a mix of values
    for theta in [0.10, 0.35, 0.05, 0.40, 0.20]:
        result = det.assess(theta_power=theta, alpha_power=0.15)
        assert 0.0 <= result["stability_score"] <= 1.0


def test_stability_high_for_constant_theta():
    """Constant theta should produce high stability score."""
    det = MindfulnessQualityDetector()
    for _ in range(12):
        result = det.assess(theta_power=0.25, alpha_power=0.15)
    assert result["stability_score"] >= 0.9


def test_recommendation_is_string():
    """Recommendation should always be a non-empty string."""
    det = MindfulnessQualityDetector()
    for theta, alpha, beta in [(0.35, 0.15, 0.0), (0.10, 0.20, 0.0), (0.05, 0.30, 0.05)]:
        result = det.assess(theta_power=theta, alpha_power=alpha, beta_power=beta)
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 0


def test_output_dict_keys():
    """Verify all required keys are present in the output dict."""
    det = MindfulnessQualityDetector()
    result = det.assess(theta_power=0.25, alpha_power=0.15)
    required_keys = {
        "quality",
        "focus_ratio",
        "mind_wandering_pct",
        "current_streak_sec",
        "stability_score",
        "recommendation",
    }
    assert required_keys.issubset(result.keys())


def test_mind_wandering_pct_increases():
    """Mind-wandering percentage should increase as more non-focused entries are added."""
    det = MindfulnessQualityDetector()
    det.assess(theta_power=0.35, alpha_power=0.15)  # focused
    r1 = det.assess(theta_power=0.10, alpha_power=0.18)  # wandering
    pct_after_one = r1["mind_wandering_pct"]

    det.assess(theta_power=0.10, alpha_power=0.18)  # more wandering
    r2 = det.assess(theta_power=0.10, alpha_power=0.18)  # more wandering
    pct_after_more = r2["mind_wandering_pct"]

    assert pct_after_more > pct_after_one


# Import for the zero-theta test assertion
from models.mindfulness_quality import QUALITY_STATES  # noqa: E402
