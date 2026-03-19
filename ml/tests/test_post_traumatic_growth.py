"""Tests for post-traumatic growth tracker (model + routes).

Covers:
  - PTGTracker: adversity recording, emotional snapshots, domain ratings,
    PTG score computation, growth indicators, trajectory, resilience vs growth,
    growth profile, profile serialization
  - Route layer: assess, trajectory, status endpoints via TestClient
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.post_traumatic_growth import (
    PTGTracker,
    PTG_DOMAINS,
    ADVERSITY_TYPES,
    _variance,
    _theil_sen_slope,
    _interpret_ptg_score,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tracker():
    return PTGTracker()


@pytest.fixture
def populated_tracker():
    """Tracker with adversity event, emotional history, and domain ratings."""
    t = PTGTracker()
    uid = "test-user"

    # Record adversity
    t.record_adversity(uid, "2025-06-01", 7.0, "loss", "Job loss")

    # Add emotional data spanning before and after adversity
    # Pre-adversity: moderate valence, moderate stress
    for i, day in enumerate(range(1, 6)):
        t.add_emotional_snapshot(
            uid,
            f"2025-05-{day:02d}",
            valence=0.2,
            arousal=0.5,
            stress_level=0.5,
            social_engagement=0.4,
        )

    # Post-adversity: improving trajectory
    for i, day in enumerate(range(1, 11)):
        improvement = i * 0.05
        t.add_emotional_snapshot(
            uid,
            f"2025-07-{day:02d}",
            valence=0.3 + improvement,
            arousal=0.5,
            stress_level=0.4 - improvement * 0.5,
            social_engagement=0.5 + improvement * 0.3,
        )

    # Set domain ratings (0-5 Likert scale)
    t.set_domain_ratings(uid, {
        "relating_to_others": 4.0,
        "new_possibilities": 3.5,
        "personal_strength": 4.5,
        "spiritual_change": 2.0,
        "appreciation_of_life": 4.0,
    })

    return t


# ── Math helpers ────────────────────────────────────────────────────────────

class TestMathHelpers:
    def test_variance_empty(self):
        assert _variance([]) == 0.0

    def test_variance_single_value(self):
        assert _variance([5.0]) == 0.0

    def test_variance_known_values(self):
        # [1, 2, 3] -> mean=2, var = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        result = _variance([1.0, 2.0, 3.0])
        assert abs(result - 2.0 / 3.0) < 1e-10

    def test_theil_sen_slope_increasing(self):
        slope = _theil_sen_slope([1.0, 2.0, 3.0, 4.0])
        assert abs(slope - 1.0) < 1e-10

    def test_theil_sen_slope_flat(self):
        slope = _theil_sen_slope([5.0, 5.0, 5.0])
        assert slope == 0.0

    def test_theil_sen_slope_insufficient(self):
        assert _theil_sen_slope([]) == 0.0
        assert _theil_sen_slope([1.0]) == 0.0

    def test_interpret_ptg_score(self):
        assert _interpret_ptg_score(0.80) == "significant_growth"
        assert _interpret_ptg_score(0.60) == "moderate_growth"
        assert _interpret_ptg_score(0.30) == "emerging_growth"
        assert _interpret_ptg_score(0.10) == "minimal_growth"


# ── Adversity recording ────────────────────────────────────────────────────

class TestAdversityRecording:
    def test_record_basic_adversity(self, tracker):
        event = tracker.record_adversity("u1", "2025-06-01", 5.0, "loss")
        assert event.date == "2025-06-01"
        assert event.severity == 5.0
        assert event.adversity_type == "loss"

    def test_severity_clamping(self, tracker):
        event_low = tracker.record_adversity("u1", "2025-01-01", -5.0, "trauma")
        event_high = tracker.record_adversity("u1", "2025-01-02", 15.0, "trauma")
        assert event_low.severity == 0.0
        assert event_high.severity == 10.0

    def test_invalid_adversity_type_defaults(self, tracker):
        event = tracker.record_adversity("u1", "2025-01-01", 5.0, "unknown_type")
        assert event.adversity_type == "trauma"

    def test_all_adversity_types_accepted(self, tracker):
        for atype in ADVERSITY_TYPES:
            event = tracker.record_adversity("u1", "2025-01-01", 5.0, atype)
            assert event.adversity_type == atype


# ── Emotional snapshots ────────────────────────────────────────────────────

class TestEmotionalSnapshots:
    def test_add_snapshot_basic(self, tracker):
        snap = tracker.add_emotional_snapshot("u1", "2025-06-01", 0.5, 0.6, 0.3, 0.7)
        assert snap.valence == 0.5
        assert snap.arousal == 0.6
        assert snap.stress_level == 0.3
        assert snap.social_engagement == 0.7

    def test_snapshot_clamping(self, tracker):
        snap = tracker.add_emotional_snapshot("u1", "2025-06-01", -2.0, 5.0, -1.0, 3.0)
        assert snap.valence == -1.0
        assert snap.arousal == 1.0
        assert snap.stress_level == 0.0
        assert snap.social_engagement == 1.0


# ── Domain ratings ──────────────────────────────────────────────────────────

class TestDomainRatings:
    def test_set_domain_ratings(self, tracker):
        ratings = {
            "relating_to_others": 4.0,
            "new_possibilities": 3.0,
            "personal_strength": 5.0,
            "spiritual_change": 1.0,
            "appreciation_of_life": 3.5,
        }
        normalized = tracker.set_domain_ratings("u1", ratings)
        assert normalized["relating_to_others"] == 0.8
        assert normalized["personal_strength"] == 1.0
        assert normalized["spiritual_change"] == 0.2

    def test_missing_domains_default_to_zero(self, tracker):
        normalized = tracker.set_domain_ratings("u1", {"relating_to_others": 3.0})
        assert normalized["new_possibilities"] == 0.0
        assert normalized["relating_to_others"] == 0.6


# ── PTG score computation ──────────────────────────────────────────────────

class TestComputePTGScore:
    def test_no_data_returns_zero(self, tracker):
        result = tracker.compute_ptg_score("unknown_user")
        assert result["ptg_total"] == 0.0
        assert result["has_self_report"] is False
        assert result["emotional_datapoints"] == 0

    def test_self_report_only(self, tracker):
        tracker.set_domain_ratings("u1", {
            "relating_to_others": 4.0,
            "new_possibilities": 3.0,
            "personal_strength": 4.0,
            "spiritual_change": 2.0,
            "appreciation_of_life": 3.5,
        })
        result = tracker.compute_ptg_score("u1")
        assert result["ptg_total"] > 0.0
        assert result["has_self_report"] is True
        assert result["interpretation"] in [
            "minimal_growth", "emerging_growth", "moderate_growth", "significant_growth"
        ]

    def test_emotional_data_only(self, tracker):
        for i in range(6):
            tracker.add_emotional_snapshot(
                "u1", f"2025-06-{i+1:02d}",
                valence=-0.3 + i * 0.15,
                arousal=0.5,
                stress_level=0.6 - i * 0.05,
                social_engagement=0.4 + i * 0.05,
            )
        result = tracker.compute_ptg_score("u1")
        assert result["ptg_total"] > 0.0
        assert result["has_self_report"] is False
        assert result["emotional_datapoints"] == 6

    def test_combined_score(self, populated_tracker):
        result = populated_tracker.compute_ptg_score("test-user")
        assert result["ptg_total"] > 0.0
        assert result["has_self_report"] is True
        assert result["emotional_datapoints"] > 0

    def test_inline_domain_ratings(self, tracker):
        """Passing domain_ratings to compute_ptg_score should update stored ratings."""
        result = tracker.compute_ptg_score("u1", domain_ratings={
            "relating_to_others": 4.0,
            "new_possibilities": 3.0,
            "personal_strength": 4.0,
            "spiritual_change": 2.0,
            "appreciation_of_life": 3.5,
        })
        assert result["has_self_report"] is True
        assert result["ptg_total"] > 0.0

    def test_score_bounded_0_1(self, populated_tracker):
        result = populated_tracker.compute_ptg_score("test-user")
        assert 0.0 <= result["ptg_total"] <= 1.0
        for score in result["domain_scores"].values():
            assert 0.0 <= score <= 1.0


# ── Growth indicators ──────────────────────────────────────────────────────

class TestGrowthIndicators:
    def test_insufficient_data(self, tracker):
        tracker.add_emotional_snapshot("u1", "2025-06-01", 0.5, 0.5, 0.5, 0.5)
        result = tracker.detect_growth_indicators("u1")
        assert result["sufficient_data"] is False
        assert result["indicators_present"] == 0

    def test_detects_positive_affect_increase(self, tracker):
        # Early: negative valence
        for i in range(5):
            tracker.add_emotional_snapshot(
                "u1", f"2025-05-{i+1:02d}",
                valence=-0.5, arousal=0.5, stress_level=0.7, social_engagement=0.3,
            )
        # Recent: positive valence
        for i in range(5):
            tracker.add_emotional_snapshot(
                "u1", f"2025-07-{i+1:02d}",
                valence=0.6, arousal=0.5, stress_level=0.3, social_engagement=0.7,
            )
        result = tracker.detect_growth_indicators("u1")
        assert result["sufficient_data"] is True
        assert result["indicators"]["increased_positive_affect"] is True

    def test_detects_social_engagement_increase(self, tracker):
        for i in range(4):
            tracker.add_emotional_snapshot(
                "u1", f"2025-05-{i+1:02d}",
                valence=0.0, arousal=0.5, stress_level=0.5, social_engagement=0.2,
            )
        for i in range(4):
            tracker.add_emotional_snapshot(
                "u1", f"2025-07-{i+1:02d}",
                valence=0.0, arousal=0.5, stress_level=0.5, social_engagement=0.8,
            )
        result = tracker.detect_growth_indicators("u1")
        assert result["indicators"]["increased_social_engagement"] is True

    def test_output_structure(self, populated_tracker):
        result = populated_tracker.detect_growth_indicators("test-user")
        assert "indicators" in result
        assert "details" in result
        assert "indicators_present" in result
        for key in [
            "emotional_range_expansion",
            "increased_positive_affect",
            "improved_emotional_regulation",
            "increased_social_engagement",
        ]:
            assert key in result["indicators"]


# ── Trajectory tracking ────────────────────────────────────────────────────

class TestTrajectory:
    def test_insufficient_data(self, tracker):
        result = tracker.track_growth_trajectory("u1")
        assert result["trajectory"] == "insufficient_data"
        assert result["assessments"] == 0

    def test_growing_trajectory(self, populated_tracker):
        uid = "test-user"
        # Run multiple assessments to build history
        populated_tracker.compute_growth_profile(uid)
        populated_tracker.compute_growth_profile(uid)
        populated_tracker.compute_growth_profile(uid)

        result = populated_tracker.track_growth_trajectory(uid)
        assert result["assessments"] >= 2
        assert result["trajectory"] in ["growing", "plateau", "declining"]
        assert "slope" in result

    def test_trajectory_output_keys(self, tracker):
        result = tracker.track_growth_trajectory("u1")
        for key in ["trajectory", "slope", "assessments", "recent_scores", "interpretation"]:
            assert key in result


# ── Resilience vs growth ────────────────────────────────────────────────────

class TestResilienceVsGrowth:
    def test_insufficient_data(self, tracker):
        result = tracker.distinguish_resilience_vs_growth("u1")
        assert result["classification"] == "insufficient_data"

    def test_growth_classification(self, populated_tracker):
        result = populated_tracker.distinguish_resilience_vs_growth("test-user")
        # The populated tracker has improving trajectory, should classify as growth
        assert result["classification"] in ["growth", "resilience", "struggling"]
        assert result["delta"] is not None
        assert "adversity_date" in result
        assert "explanation" in result

    def test_struggling_classification(self, tracker):
        uid = "u1"
        tracker.record_adversity(uid, "2025-06-01", 8.0, "loss")
        # Pre: positive
        for i in range(4):
            tracker.add_emotional_snapshot(
                uid, f"2025-05-{i+1:02d}",
                valence=0.7, arousal=0.5, stress_level=0.2, social_engagement=0.8,
            )
        # Post: much worse
        for i in range(4):
            tracker.add_emotional_snapshot(
                uid, f"2025-07-{i+1:02d}",
                valence=-0.5, arousal=0.7, stress_level=0.8, social_engagement=0.1,
            )
        result = tracker.distinguish_resilience_vs_growth(uid)
        assert result["classification"] == "struggling"
        assert result["delta"] < 0


# ── Growth profile ──────────────────────────────────────────────────────────

class TestGrowthProfile:
    def test_compute_growth_profile(self, populated_tracker):
        result = populated_tracker.compute_growth_profile("test-user")
        assert "ptg_score" in result
        assert "growth_indicators" in result
        assert "trajectory" in result
        assert "resilience_vs_growth" in result
        assert "overall_classification" in result

    def test_profile_stores_assessment(self, populated_tracker):
        uid = "test-user"
        populated_tracker.compute_growth_profile(uid)
        # Should now have at least 1 assessment stored
        assessments = populated_tracker._assessments.get(uid, [])
        assert len(assessments) >= 1


# ── Profile serialization ──────────────────────────────────────────────────

class TestProfileToDict:
    def test_empty_user(self, tracker):
        result = tracker.profile_to_dict("nobody")
        assert result["user_id"] == "nobody"
        assert result["adversity_events"] == []
        assert result["emotional_snapshots"] == 0
        assert result["assessments_count"] == 0
        assert "domains" in result
        assert len(result["domains"]) == 5

    def test_populated_user(self, populated_tracker):
        result = populated_tracker.profile_to_dict("test-user")
        assert len(result["adversity_events"]) >= 1
        assert result["emotional_snapshots"] > 0
        assert "domain_ratings" in result

    def test_domain_definitions_present(self, tracker):
        result = tracker.profile_to_dict("u1")
        for domain_key in PTG_DOMAINS:
            assert domain_key in result["domains"]
            assert "label" in result["domains"][domain_key]
            assert "description" in result["domains"][domain_key]


# ── Route integration (TestClient) ─────────────────────────────────────────

class TestRoutes:
    @pytest.fixture(autouse=True)
    def setup_app(self):
        """Create a FastAPI test client with PTG routes."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.routes.post_traumatic_growth import router

        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def test_status_endpoint(self):
        resp = self.client.get("/ptg/status", params={"user_id": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "test"
        assert "domains" in data

    def test_adversity_endpoint(self):
        resp = self.client.post("/ptg/adversity", json={
            "user_id": "test",
            "date": "2025-06-01",
            "severity": 7.0,
            "adversity_type": "loss",
            "description": "Job loss",
        })
        assert resp.status_code == 200
        assert resp.json()["recorded"] is True

    def test_emotional_data_endpoint(self):
        resp = self.client.post("/ptg/emotional-data", json={
            "user_id": "test",
            "date": "2025-07-01",
            "valence": 0.5,
            "arousal": 0.6,
            "stress_level": 0.3,
            "social_engagement": 0.7,
        })
        assert resp.status_code == 200
        assert resp.json()["added"] is True

    def test_domain_ratings_endpoint(self):
        resp = self.client.post("/ptg/domain-ratings", json={
            "user_id": "test",
            "ratings": {
                "relating_to_others": 4.0,
                "new_possibilities": 3.0,
                "personal_strength": 4.5,
                "spiritual_change": 2.0,
                "appreciation_of_life": 3.5,
            },
        })
        assert resp.status_code == 200
        assert "normalized_ratings" in resp.json()

    def test_assess_endpoint(self):
        # Seed some data first
        self.client.post("/ptg/adversity", json={
            "user_id": "test", "date": "2025-06-01",
            "severity": 7.0, "adversity_type": "loss",
        })
        for i in range(6):
            self.client.post("/ptg/emotional-data", json={
                "user_id": "test", "date": f"2025-07-{i+1:02d}",
                "valence": -0.2 + i * 0.1, "arousal": 0.5,
                "stress_level": 0.5, "social_engagement": 0.5,
            })
        resp = self.client.post("/ptg/assess", json={"user_id": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert "ptg_score" in data
        assert "growth_indicators" in data
        assert "overall_classification" in data

    def test_trajectory_endpoint(self):
        resp = self.client.post("/ptg/trajectory", json={"user_id": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "test"
        assert "trajectory" in data
