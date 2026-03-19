"""Unit tests for flow state deep work neurofeedback (issue #441)."""
from __future__ import annotations

import sys
import os
import time

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.flow_neurofeedback import (
    compute_flow_score,
    detect_distraction,
    should_protect_flow,
    track_flow_session,
    compute_optimal_conditions,
    compute_flow_profile,
    profile_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flow_features(
    alpha: float = 0.2,
    beta: float = 0.15,
    theta: float = 0.15,
    delta: float = 0.1,
    high_beta: float = 0.05,
    low_beta: float = 0.10,
) -> dict:
    return {
        "alpha": alpha,
        "beta": beta,
        "theta": theta,
        "delta": delta,
        "high_beta": high_beta,
        "low_beta": low_beta,
    }


def _make_session(
    user_id: str = "user1",
    session_id: str = "s1",
    mean_score: float = 60.0,
    duration_minutes: float = 45.0,
    peak_score: float = 80.0,
    distraction_count: int = 2,
    time_of_day: str = "morning",
    sound_env: str = "silence",
    caffeine: bool = True,
    day_of_week: str = "monday",
) -> dict:
    entry = time.time() - duration_minutes * 60
    return {
        "user_id": user_id,
        "session_id": session_id,
        "entry_time": entry,
        "exit_time": entry + duration_minutes * 60,
        "mean_score": mean_score,
        "peak_score": peak_score,
        "duration_minutes": duration_minutes,
        "distraction_count": distraction_count,
        "conditions": {
            "time_of_day": time_of_day,
            "sound_env": sound_env,
            "caffeine": caffeine,
            "day_of_week": day_of_week,
        },
    }


# ---------------------------------------------------------------------------
# compute_flow_score
# ---------------------------------------------------------------------------

class TestComputeFlowScore:

    def test_returns_required_keys(self):
        result = compute_flow_score(_flow_features())
        for key in ("flow_score", "focus_depth", "creative_engagement",
                     "time_distortion", "effortlessness", "flow_level",
                     "band_contributions"):
            assert key in result, f"Missing key: {key}"

    def test_score_range(self):
        result = compute_flow_score(_flow_features())
        assert 0 <= result["flow_score"] <= 100

    def test_subdimensions_range(self):
        result = compute_flow_score(_flow_features())
        for dim in ("focus_depth", "creative_engagement", "time_distortion", "effortlessness"):
            assert 0 <= result[dim] <= 100, f"{dim} out of range: {result[dim]}"

    def test_high_focus_features_produce_high_focus_depth(self):
        """High beta/theta ratio should produce high focus_depth."""
        high_focus = compute_flow_score(_flow_features(beta=0.5, theta=0.05))
        low_focus = compute_flow_score(_flow_features(beta=0.05, theta=0.5))
        assert high_focus["focus_depth"] > low_focus["focus_depth"]

    def test_deep_flow_level(self):
        """Very favorable features should reach deep flow."""
        result = compute_flow_score(_flow_features(
            alpha=0.25, beta=0.20, theta=0.20,
            delta=0.05, high_beta=0.02, low_beta=0.18,
        ))
        assert result["flow_level"] in ("deep", "moderate")

    def test_no_flow_with_high_delta(self):
        """High delta (drowsiness) should depress flow score."""
        drowsy = compute_flow_score(_flow_features(delta=0.6, alpha=0.05, beta=0.05, theta=0.05))
        alert = compute_flow_score(_flow_features(delta=0.05, alpha=0.2, beta=0.2, theta=0.15))
        assert drowsy["flow_score"] < alert["flow_score"]

    def test_zero_inputs_no_crash(self):
        result = compute_flow_score({"alpha": 0, "beta": 0, "theta": 0, "delta": 0})
        assert result["flow_score"] >= 0

    def test_flow_level_classification(self):
        """Verify level thresholds are respected."""
        # Force known scores by checking level assignment
        result = compute_flow_score(_flow_features())
        score = result["flow_score"]
        level = result["flow_level"]
        if score >= 75:
            assert level == "deep"
        elif score >= 50:
            assert level == "moderate"
        elif score >= 30:
            assert level == "light"
        else:
            assert level == "none"


# ---------------------------------------------------------------------------
# detect_distraction
# ---------------------------------------------------------------------------

class TestDetectDistraction:

    def test_no_distraction_normal_features(self):
        features = _flow_features()
        baseline = {
            "alpha": (0.2, 0.03),
            "beta": (0.15, 0.03),
            "theta": (0.15, 0.03),
        }
        result = detect_distraction(features, baseline)
        assert result is None

    def test_beta_spike_detected(self):
        features = _flow_features(beta=0.5)  # way above baseline
        baseline = {
            "alpha": (0.2, 0.03),
            "beta": (0.15, 0.03),
            "theta": (0.15, 0.03),
        }
        result = detect_distraction(features, baseline)
        assert result is not None
        assert result["distraction_type"] == "beta_spike"
        assert result["severity"] > 0

    def test_alpha_dropout_detected(self):
        features = _flow_features(alpha=0.01)  # far below baseline
        baseline = {
            "alpha": (0.2, 0.03),
            "beta": (0.15, 0.03),
            "theta": (0.15, 0.03),
        }
        result = detect_distraction(features, baseline)
        assert result is not None
        assert result["distraction_type"] == "alpha_dropout"

    def test_theta_surge_detected(self):
        features = _flow_features(theta=0.5)  # far above baseline
        baseline = {
            "alpha": (0.2, 0.03),
            "beta": (0.15, 0.03),
            "theta": (0.15, 0.03),
        }
        result = detect_distraction(features, baseline)
        assert result is not None
        assert result["distraction_type"] == "theta_surge"

    def test_empty_baseline_no_crash(self):
        result = detect_distraction(_flow_features(), {})
        assert result is None

    def test_recovery_suggestion_present(self):
        features = _flow_features(beta=0.5)
        baseline = {"beta": (0.15, 0.03), "alpha": (0.2, 0.03), "theta": (0.15, 0.03)}
        result = detect_distraction(features, baseline)
        assert result is not None
        assert "recovery_suggestion" in result
        assert len(result["recovery_suggestion"]) > 0


# ---------------------------------------------------------------------------
# should_protect_flow
# ---------------------------------------------------------------------------

class TestShouldProtectFlow:

    def test_no_protection_below_threshold(self):
        result = should_protect_flow(flow_score=30, flow_duration_minutes=10)
        assert result["should_suppress"] is False
        assert result["flow_shield_active"] is False

    def test_suppress_normal_during_flow(self):
        result = should_protect_flow(
            flow_score=70, flow_duration_minutes=20, notification_priority="normal",
        )
        assert result["should_suppress"] is True
        assert result["flow_shield_active"] is True

    def test_urgent_bypasses_flow(self):
        result = should_protect_flow(
            flow_score=90, flow_duration_minutes=60, notification_priority="urgent",
        )
        assert result["should_suppress"] is False

    def test_high_priority_suppressed_during_deep_flow(self):
        result = should_protect_flow(
            flow_score=85, flow_duration_minutes=30, notification_priority="high",
        )
        assert result["should_suppress"] is True

    def test_high_priority_not_suppressed_moderate_flow(self):
        result = should_protect_flow(
            flow_score=60, flow_duration_minutes=10, notification_priority="high",
        )
        assert result["should_suppress"] is False


# ---------------------------------------------------------------------------
# track_flow_session
# ---------------------------------------------------------------------------

class TestTrackFlowSession:

    def test_basic_session_tracking(self):
        now = time.time()
        data = {
            "user_id": "u1",
            "session_id": "s1",
            "entry_time": now - 3600,  # 1 hour ago
            "exit_time": now,
            "readings": [50, 60, 70, 80, 75, 65],
            "exit_reason": "natural",
            "distraction_count": 2,
        }
        result = track_flow_session(data)

        assert result["user_id"] == "u1"
        assert result["duration_minutes"] == pytest.approx(60.0, abs=0.2)
        assert result["peak_score"] == 80.0
        assert result["mean_score"] == pytest.approx(66.7, abs=0.1)
        assert result["depth_category"] == "moderate"
        assert result["distraction_count"] == 2
        assert result["productivity_index"] > 0

    def test_empty_readings(self):
        now = time.time()
        result = track_flow_session({
            "user_id": "u1", "session_id": "s2",
            "entry_time": now - 600, "exit_time": now,
            "readings": [],
        })
        assert result["peak_score"] == 0.0
        assert result["mean_score"] == 0.0
        assert result["depth_category"] == "none"

    def test_depth_categories(self):
        now = time.time()
        base = {"user_id": "u1", "session_id": "s", "entry_time": now - 60, "exit_time": now}

        deep = track_flow_session({**base, "readings": [80, 85, 90]})
        assert deep["depth_category"] == "deep"

        light = track_flow_session({**base, "readings": [35, 40, 30]})
        assert light["depth_category"] == "light"


# ---------------------------------------------------------------------------
# compute_optimal_conditions
# ---------------------------------------------------------------------------

class TestComputeOptimalConditions:

    def test_insufficient_sessions(self):
        result = compute_optimal_conditions([_make_session()])
        assert result["confidence"] == 0.0
        assert result["best_time_of_day"] is None

    def test_learns_best_time_of_day(self):
        sessions = []
        # Morning sessions score higher
        for i in range(5):
            sessions.append(_make_session(
                session_id=f"m{i}", mean_score=80, time_of_day="morning",
            ))
        for i in range(5):
            sessions.append(_make_session(
                session_id=f"e{i}", mean_score=40, time_of_day="evening",
            ))
        result = compute_optimal_conditions(sessions)
        assert result["best_time_of_day"] == "morning"

    def test_learns_caffeine_effect(self):
        sessions = []
        for i in range(5):
            sessions.append(_make_session(
                session_id=f"c{i}", mean_score=75, caffeine=True,
            ))
        for i in range(5):
            sessions.append(_make_session(
                session_id=f"n{i}", mean_score=50, caffeine=False,
            ))
        result = compute_optimal_conditions(sessions)
        assert result["caffeine_effect"] == "positive"

    def test_confidence_scales(self):
        sessions_5 = [_make_session(session_id=f"s{i}") for i in range(5)]
        sessions_20 = [_make_session(session_id=f"s{i}") for i in range(20)]
        r5 = compute_optimal_conditions(sessions_5)
        r20 = compute_optimal_conditions(sessions_20)
        assert r20["confidence"] > r5["confidence"]


# ---------------------------------------------------------------------------
# compute_flow_profile
# ---------------------------------------------------------------------------

class TestComputeFlowProfile:

    def test_empty_history(self):
        result = compute_flow_profile("u1", [])
        assert result["total_sessions"] == 0
        assert result["recent_trend"] == "stable"

    def test_basic_profile(self):
        sessions = [_make_session(session_id=f"s{i}") for i in range(6)]
        result = compute_flow_profile("u1", sessions)
        assert result["total_sessions"] == 6
        assert result["total_flow_minutes"] > 0
        assert result["avg_flow_score"] > 0

    def test_improving_trend(self):
        sessions = []
        for i in range(6):
            sessions.append(_make_session(
                session_id=f"s{i}",
                mean_score=40 + i * 10,  # 40, 50, 60, 70, 80, 90
            ))
        result = compute_flow_profile("u1", sessions)
        assert result["recent_trend"] == "improving"

    def test_declining_trend(self):
        sessions = []
        for i in range(6):
            sessions.append(_make_session(
                session_id=f"s{i}",
                mean_score=90 - i * 10,  # 90, 80, 70, 60, 50, 40
            ))
        result = compute_flow_profile("u1", sessions)
        assert result["recent_trend"] == "declining"


# ---------------------------------------------------------------------------
# profile_to_dict
# ---------------------------------------------------------------------------

class TestProfileToDict:

    def test_serialization(self):
        sessions = [_make_session(session_id=f"s{i}") for i in range(6)]
        profile = compute_flow_profile("u1", sessions)
        d = profile_to_dict(profile)

        assert "user_id" in d
        assert "optimal_conditions" in d
        assert isinstance(d["optimal_conditions"], dict)
        assert "total_sessions" in d
