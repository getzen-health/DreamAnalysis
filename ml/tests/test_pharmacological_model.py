"""Comprehensive tests for the PharmacologicalTracker model and API routes."""
import os
import sys
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.pharmacological_model import (
    PharmacologicalTracker,
    Medication,
    MedicationEffect,
    PharmacologicalProfile,
    EmotionReading,
    DRUG_EFFECT_DB,
    VALID_CATEGORIES,
    VALID_EFFECT_TYPES,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def tracker():
    """Fresh PharmacologicalTracker for each test."""
    return PharmacologicalTracker()


@pytest.fixture
def base_time():
    """A fixed base timestamp for deterministic tests."""
    return 1_700_000_000.0


# ── Drug effect database ────────────────────────────────────────────


def test_drug_db_has_required_categories():
    """DB must contain at least 10 drug categories."""
    assert len(DRUG_EFFECT_DB) >= 10


def test_drug_db_entries_have_required_fields():
    """Every DB entry must have display_name, effect_type, onset/peak/duration, effects."""
    required_keys = {
        "display_name", "effect_type", "onset_hours", "peak_hours",
        "duration_hours", "emotional_effects",
    }
    effect_keys = {"valence_shift", "arousal_shift", "range_compression"}
    for cat, entry in DRUG_EFFECT_DB.items():
        assert required_keys.issubset(entry.keys()), f"{cat} missing keys"
        assert effect_keys.issubset(entry["emotional_effects"].keys()), (
            f"{cat} emotional_effects missing keys"
        )


def test_valid_categories_match_db_keys():
    assert VALID_CATEGORIES == frozenset(DRUG_EFFECT_DB.keys())


def test_valid_effect_types_cover_all_db():
    """Every effect_type in the DB must be in VALID_EFFECT_TYPES."""
    for entry in DRUG_EFFECT_DB.values():
        assert entry["effect_type"] in VALID_EFFECT_TYPES


# ── Medication logging ──────────────────────────────────────────────


def test_log_medication_returns_medication(tracker, base_time):
    med = tracker.log_medication(
        "user1", "sertraline", "ssri", 50.0, start_date=base_time,
    )
    assert isinstance(med, Medication)
    assert med.name == "sertraline"
    assert med.category == "ssri"
    assert med.dosage_mg == 50.0
    assert med.start_date == base_time
    assert med.is_active is True


def test_log_medication_default_timestamp(tracker):
    before = time.time()
    med = tracker.log_medication("user1", "propranolol", "beta_blocker", 20.0)
    after = time.time()
    assert before <= med.start_date <= after


def test_log_medication_with_end_date(tracker, base_time):
    med = tracker.log_medication(
        "user1", "lorazepam", "benzodiazepine", 1.0,
        start_date=base_time, end_date=base_time + 86400,
    )
    assert med.is_active is False
    assert med.end_date == base_time + 86400


def test_log_medication_strips_and_lowercases_name(tracker, base_time):
    med = tracker.log_medication(
        "user1", "  Sertraline  ", "ssri", 50.0, start_date=base_time,
    )
    assert med.name == "sertraline"


def test_get_medication_log(tracker, base_time):
    tracker.log_medication("user1", "sertraline", "ssri", 50.0, start_date=base_time)
    tracker.log_medication(
        "user1", "propranolol", "beta_blocker", 20.0,
        start_date=base_time + 100,
    )
    log = tracker.get_medication_log("user1")
    assert len(log) == 2
    assert log[0]["name"] == "sertraline"
    assert log[1]["name"] == "propranolol"


def test_get_active_medications_excludes_stopped(tracker, base_time):
    tracker.log_medication("user1", "sertraline", "ssri", 50.0, start_date=base_time)
    tracker.log_medication(
        "user1", "lorazepam", "benzodiazepine", 1.0,
        start_date=base_time, end_date=base_time + 86400,
    )
    active = tracker.get_active_medications("user1")
    assert len(active) == 1
    assert active[0]["name"] == "sertraline"


# ── Medication effect computation ───────────────────────────────────


def test_compute_effect_no_medications(tracker, base_time):
    result = tracker.compute_medication_effect(
        "user1", {"valence": 0.5, "arousal": 0.6}, base_time,
    )
    assert result["adjusted_valence"] == 0.5
    assert result["adjusted_arousal"] == 0.6
    assert result["medication_count"] == 0


def test_compute_effect_ssri_at_peak(tracker, base_time):
    """SSRI at full peak should show blunting and arousal reduction."""
    tracker.log_medication(
        "user1", "sertraline", "ssri", 50.0, start_date=base_time,
    )
    # 6+ weeks later = peak
    peak_time = base_time + 1008 * 3600
    result = tracker.compute_medication_effect(
        "user1", {"valence": 0.5, "arousal": 0.6}, peak_time,
    )
    # Should have range compression and arousal reduction
    assert result["total_range_compression"] > 0
    assert result["total_arousal_modifier"] < 0
    assert result["adjusted_arousal"] < 0.6
    assert result["medication_count"] == 1
    assert len(result["effects"]) == 1
    assert result["effects"][0]["onset_fraction"] == 1.0


def test_compute_effect_stimulant_fast_onset(tracker, base_time):
    """Stimulant at 2h should be at peak."""
    tracker.log_medication(
        "user1", "methylphenidate", "stimulant", 20.0, start_date=base_time,
    )
    two_hours = base_time + 2 * 3600
    result = tracker.compute_medication_effect(
        "user1", {"valence": 0.0, "arousal": 0.5}, two_hours,
    )
    assert result["total_arousal_modifier"] > 0
    assert result["adjusted_arousal"] > 0.5


def test_compute_effect_before_onset(tracker, base_time):
    """Medication just started should have minimal effect."""
    tracker.log_medication(
        "user1", "sertraline", "ssri", 50.0, start_date=base_time,
    )
    # 1 hour after start — SSRI onset is 336 hours
    result = tracker.compute_medication_effect(
        "user1", {"valence": 0.5, "arousal": 0.6}, base_time + 3600,
    )
    # Effect should be very small
    assert abs(result["total_valence_modifier"]) < 0.01
    assert result["effects"][0]["onset_fraction"] < 0.05


def test_compute_effect_multiple_medications(tracker, base_time):
    """Multiple active meds should stack effects."""
    tracker.log_medication(
        "user1", "sertraline", "ssri", 50.0, start_date=base_time,
    )
    tracker.log_medication(
        "user1", "propranolol", "beta_blocker", 20.0, start_date=base_time,
    )
    # Both at peak
    peak_time = base_time + 2000 * 3600
    result = tracker.compute_medication_effect(
        "user1", {"valence": 0.0, "arousal": 0.5}, peak_time,
    )
    assert result["medication_count"] == 2
    assert len(result["effects"]) == 2


def test_compute_effect_range_compression_flattens_valence(tracker, base_time):
    """Range compression should push valence toward 0."""
    tracker.log_medication(
        "user1", "lithium", "mood_stabilizer", 900.0, start_date=base_time,
    )
    peak_time = base_time + 700 * 3600
    result = tracker.compute_medication_effect(
        "user1", {"valence": 0.8, "arousal": 0.5}, peak_time,
    )
    # Valence should be compressed toward 0
    assert result["adjusted_valence"] < 0.8
    assert result["total_range_compression"] > 0


# ── Onset fraction ──────────────────────────────────────────────────


def test_onset_fraction_at_zero():
    frac = PharmacologicalTracker._compute_onset_fraction(0, 100, 200)
    assert frac == 0.0


def test_onset_fraction_at_peak():
    frac = PharmacologicalTracker._compute_onset_fraction(200, 100, 200)
    assert frac == 1.0


def test_onset_fraction_past_peak():
    frac = PharmacologicalTracker._compute_onset_fraction(500, 100, 200)
    assert frac == 1.0


def test_onset_fraction_midway():
    frac = PharmacologicalTracker._compute_onset_fraction(150, 100, 200)
    assert 0.0 < frac < 1.0


# ── Emotional blunting detection ────────────────────────────────────


def _populate_blunting_data(tracker, base_time):
    """Helper: create pre/post medication data showing blunting."""
    # Log an SSRI
    tracker.log_medication(
        "user1", "sertraline", "ssri", 50.0, start_date=base_time,
    )

    # Pre-medication readings: wide emotional range
    for i in range(10):
        tracker.log_emotion_reading("user1", base_time - 86400 + i * 3600, {
            "valence": -0.5 + i * 0.1,  # range: -0.5 to 0.4
            "arousal": 0.2 + i * 0.06,
            "stress_index": 0.3,
        })

    # Post-medication readings: narrowed range (blunted)
    for i in range(10):
        tracker.log_emotion_reading("user1", base_time + i * 3600, {
            "valence": -0.1 + i * 0.02,  # range: -0.1 to 0.08
            "arousal": 0.4 + i * 0.01,
            "stress_index": 0.2,
        })


def test_detect_blunting_positive(tracker, base_time):
    _populate_blunting_data(tracker, base_time)
    result = tracker.detect_emotional_blunting("user1", window_days=2)
    assert result["blunting_detected"] is True
    assert result["blunting_score"] > 0.2
    assert result["pre_range"] > result["post_range"]


def test_detect_blunting_no_medications(tracker, base_time):
    # No meds logged
    for i in range(10):
        tracker.log_emotion_reading("user1", base_time + i * 3600, {
            "valence": 0.3, "arousal": 0.5, "stress_index": 0.2,
        })
    result = tracker.detect_emotional_blunting("user1")
    assert result["blunting_detected"] is False
    assert result["reason"] == "insufficient_data"


def test_detect_blunting_no_blunting_type_med(tracker, base_time):
    """Stimulant is not a blunting-type medication."""
    tracker.log_medication(
        "user1", "methylphenidate", "stimulant", 20.0, start_date=base_time,
    )
    for i in range(10):
        tracker.log_emotion_reading("user1", base_time + i * 3600, {
            "valence": 0.3, "arousal": 0.5, "stress_index": 0.2,
        })
    result = tracker.detect_emotional_blunting("user1")
    assert result["blunting_detected"] is False
    assert result["reason"] == "no_blunting_medications"


def test_detect_blunting_insufficient_readings(tracker, base_time):
    tracker.log_medication(
        "user1", "sertraline", "ssri", 50.0, start_date=base_time,
    )
    # Only 2 readings total
    tracker.log_emotion_reading("user1", base_time - 100, {
        "valence": 0.5, "arousal": 0.6, "stress_index": 0.2,
    })
    tracker.log_emotion_reading("user1", base_time + 100, {
        "valence": 0.1, "arousal": 0.4, "stress_index": 0.2,
    })
    result = tracker.detect_emotional_blunting("user1")
    assert result["blunting_detected"] is False
    assert result["reason"] == "insufficient_data"


# ── Onset curve ─────────────────────────────────────────────────────


def test_onset_curve_basic(tracker, base_time):
    tracker.log_medication(
        "user1", "sertraline", "ssri", 50.0, start_date=base_time,
    )
    # Log readings at various times after start
    for i in range(10):
        tracker.log_emotion_reading(
            "user1", base_time + i * 86400, {
                "valence": 0.1 * i,
                "arousal": 0.5,
                "stress_index": 0.3 - 0.02 * i,
            },
        )
    result = tracker.compute_onset_curve("user1", "sertraline", bucket_hours=24.0)
    assert len(result["onset_curve"]) > 0
    assert result["onset_curve"][0]["bucket_index"] == 0
    assert result["total_readings"] == 10


def test_onset_curve_no_medication(tracker):
    result = tracker.compute_onset_curve("user1", "nonexistent")
    assert result["onset_curve"] == []
    assert result["reason"] == "no_medication_entries"


def test_onset_curve_insufficient_readings(tracker, base_time):
    tracker.log_medication(
        "user1", "sertraline", "ssri", 50.0, start_date=base_time,
    )
    # Only 2 readings
    tracker.log_emotion_reading("user1", base_time + 100, {
        "valence": 0.1, "arousal": 0.5, "stress_index": 0.3,
    })
    tracker.log_emotion_reading("user1", base_time + 200, {
        "valence": 0.2, "arousal": 0.5, "stress_index": 0.3,
    })
    result = tracker.compute_onset_curve("user1", "sertraline")
    assert result["onset_curve"] == []
    assert result["reason"] == "insufficient_emotion_data"


# ── Withdrawal detection ────────────────────────────────────────────


def test_detect_withdrawal_positive(tracker, base_time):
    """Detect withdrawal after stopping a benzodiazepine."""
    end_time = base_time + 30 * 86400
    tracker.log_medication(
        "user1", "lorazepam", "benzodiazepine", 1.0,
        start_date=base_time, end_date=end_time,
    )
    # While on medication: low arousal, low stress
    for i in range(8):
        tracker.log_emotion_reading("user1", base_time + i * 86400, {
            "valence": 0.1, "arousal": 0.25, "stress_index": 0.15,
        })
    # After stopping: arousal and stress spike (withdrawal)
    for i in range(8):
        tracker.log_emotion_reading("user1", end_time + i * 86400, {
            "valence": -0.2, "arousal": 0.65, "stress_index": 0.55,
        })
    result = tracker.detect_withdrawal("user1")
    assert result["withdrawal_detected"] is True
    assert len(result["affected_medications"]) == 1
    assert result["affected_medications"][0]["medication_name"] == "lorazepam"
    assert result["affected_medications"][0]["arousal_rebound"] > 0


def test_detect_withdrawal_no_stopped_meds(tracker, base_time):
    tracker.log_medication(
        "user1", "sertraline", "ssri", 50.0, start_date=base_time,
    )
    result = tracker.detect_withdrawal("user1")
    assert result["withdrawal_detected"] is False
    assert result["reason"] == "no_stopped_medications"


def test_detect_withdrawal_insufficient_data(tracker, base_time):
    end_time = base_time + 86400
    tracker.log_medication(
        "user1", "lorazepam", "benzodiazepine", 1.0,
        start_date=base_time, end_date=end_time,
    )
    # Only 1 reading in each window (below threshold)
    tracker.log_emotion_reading("user1", base_time + 100, {
        "valence": 0.1, "arousal": 0.3, "stress_index": 0.2,
    })
    tracker.log_emotion_reading("user1", end_time + 100, {
        "valence": -0.3, "arousal": 0.7, "stress_index": 0.6,
    })
    result = tracker.detect_withdrawal("user1")
    # Not enough readings to trigger
    assert result["withdrawal_detected"] is False


# ── Full profile ────────────────────────────────────────────────────


def test_profile_basic(tracker, base_time):
    tracker.log_medication(
        "user1", "sertraline", "ssri", 50.0, start_date=base_time,
    )
    profile = tracker.compute_pharmacological_profile("user1", base_time + 2000 * 3600)
    assert isinstance(profile, PharmacologicalProfile)
    assert profile.user_id == "user1"
    assert len(profile.active_medications) == 1
    assert profile.dominant_effect_type == "blunting"


def test_profile_to_dict(tracker, base_time):
    tracker.log_medication(
        "user1", "sertraline", "ssri", 50.0, start_date=base_time,
    )
    d = tracker.profile_to_dict("user1", base_time + 2000 * 3600)
    assert isinstance(d, dict)
    assert d["user_id"] == "user1"
    assert "active_medications" in d
    assert "onset_status" in d


def test_profile_empty_user(tracker):
    profile = tracker.compute_pharmacological_profile("nobody")
    assert profile.dominant_effect_type == "none"
    assert len(profile.active_medications) == 0


# ── User isolation ──────────────────────────────────────────────────


def test_user_isolation(tracker, base_time):
    tracker.log_medication("user1", "sertraline", "ssri", 50.0, start_date=base_time)
    tracker.log_medication("user2", "propranolol", "beta_blocker", 20.0, start_date=base_time)

    active1 = tracker.get_active_medications("user1")
    active2 = tracker.get_active_medications("user2")
    assert len(active1) == 1
    assert active1[0]["name"] == "sertraline"
    assert len(active2) == 1
    assert active2[0]["name"] == "propranolol"


# ── Reset ───────────────────────────────────────────────────────────


def test_reset_clears_all_data(tracker, base_time):
    tracker.log_medication("user1", "sertraline", "ssri", 50.0, start_date=base_time)
    tracker.log_emotion_reading("user1", base_time, {
        "valence": 0.3, "arousal": 0.5, "stress_index": 0.2,
    })
    tracker.reset("user1")
    assert tracker.get_medication_log("user1") == []
    assert tracker.get_active_medications("user1") == []


def test_reset_nonexistent_user(tracker):
    tracker.reset("nobody")  # Should not raise


# ── Storage caps ────────────────────────────────────────────────────


def test_medication_cap(base_time):
    t = PharmacologicalTracker(max_medications=5, max_emotion_readings=100)
    for i in range(10):
        t.log_medication(
            "user1", f"med{i}", "ssri", 50.0, start_date=base_time + i,
        )
    log = t.get_medication_log("user1", last_n=100)
    assert len(log) == 5
    assert log[0]["name"] == "med5"


def test_emotion_reading_cap(base_time):
    t = PharmacologicalTracker(max_medications=100, max_emotion_readings=5)
    for i in range(10):
        t.log_emotion_reading("user1", base_time + i, {
            "valence": 0.1 * i, "arousal": 0.5, "stress_index": 0.3,
        })
    assert len(t._emotion_readings["user1"]) == 5


# ── Dataclass tests ─────────────────────────────────────────────────


def test_medication_to_dict():
    med = Medication(
        name="sertraline", category="ssri", dosage_mg=50.0,
        start_date=1000.0, end_date=None, notes="test",
    )
    d = med.to_dict()
    assert d["name"] == "sertraline"
    assert d["end_date"] is None
    assert med.is_active is True


def test_medication_effect_to_dict():
    effect = MedicationEffect(
        medication_name="sertraline", category="ssri",
        effect_type="blunting", hours_since_start=100.0,
        onset_fraction=0.5, valence_modifier=0.03,
        arousal_modifier=-0.05, range_compression=0.12,
    )
    d = effect.to_dict()
    assert d["medication_name"] == "sertraline"
    assert d["onset_fraction"] == 0.5


def test_emotion_reading_to_dict():
    reading = EmotionReading(
        timestamp=1000.0, valence=0.5, arousal=0.6,
        stress_index=0.2, source="eeg",
    )
    d = reading.to_dict()
    assert d["valence"] == 0.5
    assert d["source"] == "eeg"


# ── Route integration ──────────────────────────────────────────────


def test_route_module_imports():
    from api.routes.pharmacological import router, get_tracker
    assert router is not None
    assert get_tracker() is not None


def test_log_route(base_time):
    from api.routes.pharmacological import router, get_tracker

    tracker = get_tracker()
    tracker.reset("route_user")

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.post("/pharmacological/log", json={
        "user_id": "route_user",
        "name": "sertraline",
        "category": "ssri",
        "dosage_mg": 50.0,
        "start_date": base_time,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["medication"]["name"] == "sertraline"
    assert "category_info" in body

    # Cleanup
    tracker.reset("route_user")


def test_log_route_invalid_category():
    from api.routes.pharmacological import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.post("/pharmacological/log", json={
        "user_id": "route_user",
        "name": "unknown_drug",
        "category": "invalid_category",
        "dosage_mg": 50.0,
    })
    assert response.status_code == 422


def test_analyze_route(base_time):
    from api.routes.pharmacological import router, get_tracker

    tracker = get_tracker()
    tracker.reset("analyze_user")
    tracker.log_medication(
        "analyze_user", "propranolol", "beta_blocker", 20.0,
        start_date=base_time,
    )

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.post("/pharmacological/analyze", json={
        "user_id": "analyze_user",
        "base_valence": 0.5,
        "base_arousal": 0.6,
        "current_time": base_time + 4 * 3600,
    })
    assert response.status_code == 200
    body = response.json()
    assert "adjusted_valence" in body
    assert "adjusted_arousal" in body
    assert body["medication_count"] == 1

    tracker.reset("analyze_user")


def test_status_route(base_time):
    from api.routes.pharmacological import router, get_tracker

    tracker = get_tracker()
    tracker.reset("status_user")
    tracker.log_medication(
        "status_user", "sertraline", "ssri", 50.0,
        start_date=base_time,
    )

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get(
        "/pharmacological/status",
        params={"user_id": "status_user", "current_time": base_time + 100},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "status_user"
    assert len(body["active_medications"]) == 1

    tracker.reset("status_user")
