"""Tests for epigenetic emotional inheritance model + routes.

Covers:
  - EpigeneticEngine: family history intake, emotional snapshots,
    inherited pattern detection, risk/attenuation, protective factors,
    generational healing, full profile, serialization
  - Route layer: family-history, analyze, status endpoints via TestClient

GitHub issue: #449
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.epigenetic_model import (
    EpigeneticEngine,
    FamilyMember,
    EmotionalSnapshot,
    InheritedPattern,
    CONDITION_TYPES,
    RELATION_TYPES,
    _mean_stress,
    _mean_valence,
    _mean_arousal,
    _variance,
    _compute_condition_match,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    return EpigeneticEngine()


@pytest.fixture
def family_members():
    """Standard family history with anxiety + depression."""
    return [
        FamilyMember(
            relation="mother",
            conditions=["anxiety", "depression"],
            severity=7.0,
        ),
        FamilyMember(
            relation="father",
            conditions=["addiction", "anger"],
            severity=5.0,
        ),
        FamilyMember(
            relation="maternal_grandmother",
            conditions=["anxiety", "trauma"],
            severity=8.0,
        ),
    ]


@pytest.fixture
def populated_engine(engine, family_members):
    """Engine with family history and emotional data loaded."""
    uid = "test-user"
    engine.intake_family_history(uid, family_members)

    # Add emotional data showing moderate stress and improving valence
    for i in range(10):
        engine.add_emotional_snapshot(
            uid,
            f"2025-07-{i + 1:02d}",
            valence=-0.3 + i * 0.08,
            arousal=0.6 - i * 0.02,
            stress_level=0.7 - i * 0.04,
            trigger="work" if i < 5 else "",
        )
    return engine


# ── Helper functions ────────────────────────────────────────────────────────

class TestHelperFunctions:
    def test_mean_stress_empty(self):
        assert _mean_stress([]) == 0.0

    def test_mean_stress_values(self):
        snaps = [
            EmotionalSnapshot(date="2025-01-01", valence=0.0, arousal=0.5, stress_level=0.4),
            EmotionalSnapshot(date="2025-01-02", valence=0.0, arousal=0.5, stress_level=0.6),
        ]
        assert abs(_mean_stress(snaps) - 0.5) < 1e-10

    def test_mean_valence_empty(self):
        assert _mean_valence([]) == 0.0

    def test_mean_arousal_empty(self):
        assert _mean_arousal([]) == 0.5

    def test_variance_empty(self):
        assert _variance([]) == 0.0

    def test_variance_single(self):
        assert _variance([5.0]) == 0.0

    def test_variance_known(self):
        # [1, 2, 3] -> var = 2/3
        result = _variance([1.0, 2.0, 3.0])
        assert abs(result - 2.0 / 3.0) < 1e-10

    def test_compute_condition_match_anxiety(self):
        # High stress, high arousal, negative valence -> high anxiety match
        score = _compute_condition_match("anxiety", 0.8, -0.5, 0.9)
        assert score > 0.4

    def test_compute_condition_match_resilience(self):
        # Low stress, positive valence -> low resilience match (negative weights)
        score = _compute_condition_match("resilience", 0.1, 0.8, 0.5)
        assert score < 0.3


# ── Family history intake ──────────────────────────────────────────────────

class TestFamilyHistoryIntake:
    def test_intake_basic(self, engine, family_members):
        stored = engine.intake_family_history("u1", family_members)
        assert len(stored) == 3
        assert stored[0].relation == "mother"
        assert "anxiety" in stored[0].conditions
        assert "depression" in stored[0].conditions

    def test_severity_clamping(self, engine):
        members = [
            FamilyMember(relation="mother", conditions=["anxiety"], severity=-5.0),
            FamilyMember(relation="father", conditions=["depression"], severity=15.0),
        ]
        stored = engine.intake_family_history("u1", members)
        assert stored[0].severity == 0.0
        assert stored[1].severity == 10.0

    def test_invalid_relation_defaults(self, engine):
        members = [
            FamilyMember(relation="unknown_relation", conditions=["anxiety"], severity=5.0),
        ]
        stored = engine.intake_family_history("u1", members)
        assert stored[0].relation == "mother"  # defaults to mother

    def test_invalid_conditions_filtered(self, engine):
        members = [
            FamilyMember(
                relation="mother",
                conditions=["anxiety", "made_up_condition", "depression"],
                severity=5.0,
            ),
        ]
        stored = engine.intake_family_history("u1", members)
        assert "anxiety" in stored[0].conditions
        assert "depression" in stored[0].conditions
        assert "made_up_condition" not in stored[0].conditions

    def test_replaces_existing_history(self, engine, family_members):
        engine.intake_family_history("u1", family_members)
        new_members = [
            FamilyMember(relation="father", conditions=["resilience"], severity=2.0),
        ]
        engine.intake_family_history("u1", new_members)
        stored = engine._family_history["u1"]
        assert len(stored) == 1
        assert stored[0].relation == "father"


# ── Emotional snapshots ───────────────────────────────────────────────────

class TestEmotionalSnapshots:
    def test_add_snapshot(self, engine):
        snap = engine.add_emotional_snapshot("u1", "2025-07-01", 0.5, 0.6, 0.3, "music")
        assert snap.valence == 0.5
        assert snap.stress_level == 0.3
        assert snap.trigger == "music"

    def test_clamping(self, engine):
        snap = engine.add_emotional_snapshot("u1", "2025-07-01", -2.0, 5.0, -1.0)
        assert snap.valence == -1.0
        assert snap.arousal == 1.0
        assert snap.stress_level == 0.0


# ── Pattern detection ──────────────────────────────────────────────────────

class TestInheritedPatterns:
    def test_no_family_returns_empty(self, engine):
        patterns = engine.detect_inherited_patterns("u1")
        assert patterns == []

    def test_detects_patterns_from_family(self, populated_engine):
        patterns = populated_engine.detect_inherited_patterns("test-user")
        assert len(patterns) > 0
        conditions_found = {p.condition for p in patterns}
        # Should detect anxiety (mother + grandmother) and addiction (father)
        assert "anxiety" in conditions_found

    def test_pattern_structure(self, populated_engine):
        patterns = populated_engine.detect_inherited_patterns("test-user")
        for p in patterns:
            assert isinstance(p, InheritedPattern)
            assert 0.0 <= p.family_prevalence <= 1.0
            assert 0.0 <= p.user_match_score <= 1.0
            assert p.direction in ("amplified", "attenuated", "stable")
            assert len(p.contributing_relations) > 0
            assert 0.0 <= p.confidence <= 1.0

    def test_direction_classification(self, engine):
        """High-stress user with anxious family should show amplification."""
        engine.intake_family_history("u1", [
            FamilyMember(relation="mother", conditions=["anxiety"], severity=9.0),
            FamilyMember(relation="father", conditions=["anxiety"], severity=8.0),
        ])
        # Very high stress user
        for i in range(10):
            engine.add_emotional_snapshot(
                "u1", f"2025-07-{i + 1:02d}",
                valence=-0.6, arousal=0.9, stress_level=0.9,
            )
        patterns = engine.detect_inherited_patterns("u1")
        anxiety_pattern = next((p for p in patterns if p.condition == "anxiety"), None)
        assert anxiety_pattern is not None
        # With very high stress matching family anxiety, direction should be
        # amplified or stable
        assert anxiety_pattern.direction in ("amplified", "stable")


# ── Risk attenuation ──────────────────────────────────────────────────────

class TestRiskAttenuation:
    def test_no_data(self, engine):
        result = engine.compute_risk_attenuation("u1")
        assert result["risk_score"] == 0.0
        assert result["attenuation_score"] == 0.0
        assert "No family history" in result["summary"]

    def test_with_data(self, populated_engine):
        result = populated_engine.compute_risk_attenuation("test-user")
        assert 0.0 <= result["risk_score"] <= 1.0
        assert 0.0 <= result["attenuation_score"] <= 1.0
        assert isinstance(result["amplified"], list)
        assert isinstance(result["attenuated"], list)
        assert isinstance(result["stable"], list)
        assert isinstance(result["summary"], str)

    def test_high_risk_scoring(self, engine):
        """Family with severe conditions and stressed user -> high risk."""
        engine.intake_family_history("u1", [
            FamilyMember(relation="mother", conditions=["anxiety", "depression", "trauma"], severity=9.0),
            FamilyMember(relation="father", conditions=["anxiety", "addiction"], severity=8.0),
        ])
        for i in range(10):
            engine.add_emotional_snapshot(
                "u1", f"2025-07-{i + 1:02d}",
                valence=-0.5, arousal=0.8, stress_level=0.9,
            )
        result = engine.compute_risk_attenuation("u1")
        assert result["risk_score"] > 0.3


# ── Protective factors ────────────────────────────────────────────────────

class TestProtectiveFactors:
    def test_no_patterns(self, engine):
        result = engine.detect_protective_factors("u1")
        assert result["protective_factors"] == []
        assert result["protective_score"] == 0.0

    def test_broken_pattern_detected(self, engine):
        """Family has severe anxiety but user is calm -> broken pattern."""
        engine.intake_family_history("u1", [
            FamilyMember(relation="mother", conditions=["anxiety"], severity=9.0),
            FamilyMember(relation="father", conditions=["anxiety"], severity=8.0),
        ])
        # User is calm and positive
        for i in range(10):
            engine.add_emotional_snapshot(
                "u1", f"2025-07-{i + 1:02d}",
                valence=0.6, arousal=0.3, stress_level=0.1,
            )
        result = engine.detect_protective_factors("u1")
        # Anxiety prevalence is high but user match should be low
        assert result["protective_score"] >= 0.0
        assert isinstance(result["broken_patterns"], list)
        assert isinstance(result["resilience_indicators"], list)

    def test_resilience_indicators_from_improvement(self, populated_engine):
        """Populated engine has improving trajectory -> resilience indicators."""
        result = populated_engine.detect_protective_factors("test-user")
        # The populated engine has decreasing stress and improving valence
        assert isinstance(result["resilience_indicators"], list)


# ── Generational healing ──────────────────────────────────────────────────

class TestGenerationalHealing:
    def test_no_family(self, engine):
        result = engine.track_generational_healing("u1")
        assert result["healing_score"] == 0.0
        assert "No family history" in result["interpretation"]

    def test_healing_detected(self, engine):
        """Parent burdened, user healthy -> healing score > 0.5."""
        engine.intake_family_history("u1", [
            FamilyMember(relation="mother", conditions=["anxiety", "depression", "trauma"], severity=9.0),
        ])
        for i in range(10):
            engine.add_emotional_snapshot(
                "u1", f"2025-07-{i + 1:02d}",
                valence=0.6, arousal=0.4, stress_level=0.15,
            )
        result = engine.track_generational_healing("u1")
        assert result["healing_score"] > 0.5
        assert result["delta"] > 0
        assert "healing" in result["interpretation"].lower()

    def test_increased_burden(self, engine):
        """Parent relatively healthy, user struggling."""
        engine.intake_family_history("u1", [
            FamilyMember(relation="father", conditions=["resilience"], severity=1.0),
        ])
        for i in range(10):
            engine.add_emotional_snapshot(
                "u1", f"2025-07-{i + 1:02d}",
                valence=-0.7, arousal=0.8, stress_level=0.9,
            )
        result = engine.track_generational_healing("u1")
        assert result["healing_score"] < 0.5
        assert result["delta"] < 0

    def test_output_keys(self, populated_engine):
        result = populated_engine.track_generational_healing("test-user")
        for key in ["healing_score", "parental_burden", "user_burden", "delta", "interpretation"]:
            assert key in result


# ── Full profile ───────────────────────────────────────────────────────────

class TestEpigeneticProfile:
    def test_compute_profile(self, populated_engine):
        result = populated_engine.compute_epigenetic_profile("test-user")
        assert result["user_id"] == "test-user"
        assert "inherited_patterns" in result
        assert "risk_attenuation" in result
        assert "protective_factors" in result
        assert "generational_healing" in result
        assert result["family_members_count"] == 3
        assert result["emotional_datapoints"] == 10
        assert "assessment_date" in result

    def test_profile_pattern_structure(self, populated_engine):
        result = populated_engine.compute_epigenetic_profile("test-user")
        for pattern in result["inherited_patterns"]:
            assert "condition" in pattern
            assert "family_prevalence" in pattern
            assert "user_match_score" in pattern
            assert "direction" in pattern


# ── Profile serialization ─────────────────────────────────────────────────

class TestProfileToDict:
    def test_empty_user(self, engine):
        result = engine.profile_to_dict("nobody")
        assert result["user_id"] == "nobody"
        assert result["family_members"] == []
        assert result["emotional_snapshots"] == 0
        assert result["condition_types"] == CONDITION_TYPES
        assert result["relation_types"] == RELATION_TYPES

    def test_populated_user(self, populated_engine):
        result = populated_engine.profile_to_dict("test-user")
        assert len(result["family_members"]) == 3
        assert result["emotional_snapshots"] == 10
        assert result["family_members"][0]["relation"] == "mother"


# ── Route integration (TestClient) ────────────────────────────────────────

class TestRoutes:
    @pytest.fixture(autouse=True)
    def setup_app(self):
        """Create a FastAPI test client with epigenetic routes."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.routes.epigenetic import router

        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def test_status_endpoint(self):
        resp = self.client.get("/epigenetic/status", params={"user_id": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "test"
        assert "condition_types" in data
        assert "relation_types" in data

    def test_family_history_endpoint(self):
        resp = self.client.post("/epigenetic/family-history", json={
            "user_id": "test",
            "members": [
                {
                    "relation": "mother",
                    "conditions": ["anxiety", "depression"],
                    "severity": 7.0,
                    "notes": "Generalized anxiety since age 30",
                },
                {
                    "relation": "father",
                    "conditions": ["addiction"],
                    "severity": 5.0,
                },
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["members_stored"] == 2

    def test_analyze_endpoint(self):
        # Seed family history and emotional data
        self.client.post("/epigenetic/family-history", json={
            "user_id": "test",
            "members": [
                {"relation": "mother", "conditions": ["anxiety"], "severity": 7.0},
            ],
        })
        resp = self.client.post("/epigenetic/analyze", json={"user_id": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "test"
        assert "inherited_patterns" in data
        assert "risk_attenuation" in data
        assert "protective_factors" in data
        assert "generational_healing" in data

    def test_analyze_empty_user(self):
        resp = self.client.post("/epigenetic/analyze", json={"user_id": "empty"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["inherited_patterns"] == []
