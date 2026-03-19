"""Tests for micro-intervention engine (issue #435).

Covers: trigger detection, suppression (cooldown, flow, daily cap),
intervention selection, outcome recording, effectiveness computation,
circadian-aware learning, and API route integration.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.micro_intervention import (
    EmotionState,
    Intervention,
    InterventionCategory,
    InterventionEngine,
    InterventionOutcome,
    TriggerEvent,
    TriggerType,
    INTERVENTION_LIBRARY,
    COOLDOWN_SECONDS,
    MAX_INTERVENTIONS_PER_DAY,
    FLOW_STATE_THRESHOLD,
    _get_circadian_phase,
    _LIBRARY_BY_ID,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine() -> InterventionEngine:
    return InterventionEngine()


def _normal_state(**overrides) -> EmotionState:
    defaults = dict(valence=0.3, arousal=0.4, stress_index=0.3, focus_index=0.6, flow_index=0.2)
    defaults.update(overrides)
    return EmotionState(**defaults)


def _stressed_state(**overrides) -> EmotionState:
    defaults = dict(valence=-0.3, arousal=0.8, stress_index=0.8, focus_index=0.4, flow_index=0.1)
    defaults.update(overrides)
    return EmotionState(**defaults)


# ---------------------------------------------------------------------------
# 1. Intervention library structure
# ---------------------------------------------------------------------------

def test_library_has_all_categories():
    """Library must contain at least one intervention per category."""
    categories_found = {i.category for i in INTERVENTION_LIBRARY}
    for cat in InterventionCategory:
        assert cat in categories_found, f"Missing category: {cat.value}"


def test_library_ids_are_unique():
    """All intervention IDs must be unique."""
    ids = [i.id for i in INTERVENTION_LIBRARY]
    assert len(ids) == len(set(ids))


def test_intervention_to_dict():
    """Intervention.to_dict() should include all required fields."""
    iv = INTERVENTION_LIBRARY[0]
    d = iv.to_dict()
    assert "id" in d
    assert "name" in d
    assert "category" in d
    assert "duration_seconds" in d
    assert isinstance(d["intensity_range"], list)
    assert len(d["intensity_range"]) == 2


# ---------------------------------------------------------------------------
# 2. Trigger detection
# ---------------------------------------------------------------------------

def test_no_trigger_normal_state():
    """Normal emotion state should not fire any trigger."""
    engine = _make_engine()
    state = _normal_state()
    result = engine.check_trigger("user1", state)
    assert result["triggered"] is False
    assert result["trigger_type"] is None


def test_stress_spike_trigger():
    """Valence drop + high arousal should fire stress_spike."""
    engine = _make_engine()
    # Set previous state with high valence
    prev = _normal_state(valence=0.5, arousal=0.3)
    engine.check_trigger("user2", prev)
    # Now drop valence, spike arousal
    stressed = _stressed_state(valence=0.1, arousal=0.75)
    result = engine.check_trigger("user2", stressed)
    assert result["triggered"] is True
    assert result["trigger_type"] == "stress_spike"


def test_high_absolute_stress_triggers():
    """Stress index >= 0.75 should trigger even without valence drop."""
    engine = _make_engine()
    state = _normal_state(stress_index=0.80)
    result = engine.check_trigger("user3", state)
    assert result["triggered"] is True
    assert result["trigger_type"] == "stress_spike"


def test_focus_decline_trigger():
    """Low focus index should fire focus_decline."""
    engine = _make_engine()
    state = _normal_state(focus_index=0.20, stress_index=0.2)
    result = engine.check_trigger("user4", state)
    assert result["triggered"] is True
    assert result["trigger_type"] == "focus_decline"


def test_energy_crash_trigger():
    """Low arousal + low valence should fire energy_crash."""
    engine = _make_engine()
    state = _normal_state(arousal=0.15, valence=0.10, stress_index=0.2, focus_index=0.5)
    result = engine.check_trigger("user5", state)
    assert result["triggered"] is True
    assert result["trigger_type"] == "energy_crash"


def test_pre_meeting_anxiety_trigger():
    """Moderate stress + upcoming meeting should fire pre_meeting_anxiety."""
    engine = _make_engine()
    state = _normal_state(stress_index=0.55, focus_index=0.5)
    result = engine.check_trigger("user6", state, upcoming_meeting_minutes=5.0)
    assert result["triggered"] is True
    assert result["trigger_type"] == "pre_meeting_anxiety"


# ---------------------------------------------------------------------------
# 3. Suppression mechanisms
# ---------------------------------------------------------------------------

def test_flow_state_suppresses():
    """High flow state should suppress triggers."""
    engine = _make_engine()
    state = _stressed_state(flow_index=0.85, stress_index=0.80)
    result = engine.check_trigger("user7", state)
    assert result["suppressed"] is True
    assert result["suppression_reason"] == "flow_state"


def test_cooldown_suppresses():
    """Recent intervention should suppress triggers during cooldown."""
    engine = _make_engine()
    engine._last_intervention_ts["user8"] = time.time()  # just now
    state = _stressed_state(stress_index=0.80)
    result = engine.check_trigger("user8", state)
    assert result["suppressed"] is True
    assert result["suppression_reason"] == "cooldown"


def test_daily_cap_suppresses():
    """Exceeding daily max should suppress triggers."""
    engine = _make_engine()
    day_key = time.strftime("%Y-%m-%d")
    engine._daily_counts["user9"] = {day_key: MAX_INTERVENTIONS_PER_DAY}
    state = _stressed_state(stress_index=0.80)
    result = engine.check_trigger("user9", state)
    assert result["suppressed"] is True
    assert result["suppression_reason"] == "daily_cap"


# ---------------------------------------------------------------------------
# 4. Intervention selection
# ---------------------------------------------------------------------------

def test_select_for_stress_spike():
    """Stress spike should select from breathing/grounding categories."""
    engine = _make_engine()
    state = _stressed_state()
    result = engine.select_intervention("user10", state, "stress_spike", hour_of_day=10.0)
    assert "intervention" in result
    assert result["intervention"]["category"] in ("breathing", "grounding")
    assert result["trigger_type"] == "stress_spike"


def test_select_for_focus_decline():
    """Focus decline should select from movement/cognitive categories."""
    engine = _make_engine()
    state = _normal_state(focus_index=0.2, stress_index=0.3)
    result = engine.select_intervention("user11", state, "focus_decline", hour_of_day=14.0)
    assert result["intervention"]["category"] in ("movement", "cognitive")


def test_select_includes_circadian_phase():
    """Selection result should include the circadian phase."""
    engine = _make_engine()
    state = _stressed_state()
    result = engine.select_intervention("user12", state, "stress_spike", hour_of_day=7.0)
    assert result["circadian_phase"] == "morning"


def test_select_updates_daily_count():
    """Selecting an intervention should increment today's count."""
    engine = _make_engine()
    state = _stressed_state()
    engine.select_intervention("user13", state, "stress_spike", hour_of_day=12.0)
    day_key = time.strftime("%Y-%m-%d")
    assert engine._daily_counts["user13"][day_key] == 1


# ---------------------------------------------------------------------------
# 5. Outcome recording and effectiveness
# ---------------------------------------------------------------------------

def test_record_outcome_effective():
    """Recording a positive outcome should mark it as effective."""
    engine = _make_engine()
    result = engine.record_outcome(
        user_id="user14",
        intervention_id="box_breathing",
        trigger_type="stress_spike",
        valence_before=-0.3,
        arousal_before=0.8,
        valence_after=0.1,
        arousal_after=0.5,
        stress_before=0.7,
        stress_after=0.3,
        hour_of_day=10.0,
    )
    assert result["recorded"] is True
    assert result["effective"] is True
    assert result["valence_delta"] > 0  # improved
    assert result["stress_delta"] > 0   # improved


def test_record_outcome_felt_helpful():
    """Self-report of helpfulness should override numeric deltas."""
    engine = _make_engine()
    result = engine.record_outcome(
        user_id="user15",
        intervention_id="gratitude_pause",
        trigger_type="focus_decline",
        valence_before=0.1,
        arousal_before=0.4,
        valence_after=0.1,  # no change in valence
        arousal_after=0.4,  # no change
        felt_helpful=True,
        hour_of_day=15.0,
    )
    assert result["effective"] is True


def test_compute_effectiveness_no_history():
    """Effectiveness with no history should return zeros."""
    engine = _make_engine()
    eff = engine.compute_effectiveness("nobody")
    assert eff["total_outcomes"] == 0
    assert eff["success_rate"] == 0.0


def test_compute_effectiveness_with_outcomes():
    """Effectiveness should aggregate outcomes correctly."""
    engine = _make_engine()
    # Record 3 outcomes: 2 effective, 1 not
    engine.record_outcome("user16", "box_breathing", "stress_spike",
                          -0.3, 0.8, 0.1, 0.5, 0.7, 0.3, hour_of_day=10.0)
    engine.record_outcome("user16", "box_breathing", "stress_spike",
                          -0.2, 0.7, 0.2, 0.4, 0.6, 0.2, hour_of_day=11.0)
    engine.record_outcome("user16", "box_breathing", "stress_spike",
                          -0.1, 0.6, -0.1, 0.6, 0.5, 0.5, hour_of_day=12.0)  # no improvement

    eff = engine.compute_effectiveness("user16")
    assert eff["total_outcomes"] == 3
    assert eff["success_rate"] > 0  # at least 2/3
    assert "box_breathing" in eff["by_intervention"]
    assert eff["by_intervention"]["box_breathing"]["count"] == 3


# ---------------------------------------------------------------------------
# 6. Stats and serialization
# ---------------------------------------------------------------------------

def test_get_intervention_stats():
    """Stats should include counts, daily usage, cooldown info."""
    engine = _make_engine()
    engine.record_outcome("user17", "desk_stretch", "focus_decline",
                          0.1, 0.3, 0.3, 0.5, 0.3, 0.2, hour_of_day=14.0)
    stats = engine.get_intervention_stats("user17")
    assert stats["user_id"] == "user17"
    assert stats["total_outcomes"] == 1
    assert stats["max_per_day"] == MAX_INTERVENTIONS_PER_DAY
    assert "cooldown_remaining_seconds" in stats


def test_engine_to_dict():
    """engine_to_dict should serialize without errors."""
    engine = _make_engine()
    engine.record_outcome("user18", "physiological_sigh", "stress_spike",
                          -0.5, 0.9, 0.0, 0.5, 0.8, 0.3, hour_of_day=9.0)
    d = engine.engine_to_dict()
    assert "users" in d
    assert "user18" in d["users"]
    assert d["total_outcomes"] >= 1
    assert len(d["library"]) == len(INTERVENTION_LIBRARY)


# ---------------------------------------------------------------------------
# 7. Circadian phase helper
# ---------------------------------------------------------------------------

def test_circadian_phase_mapping():
    """Hour of day should map to correct phase labels."""
    assert _get_circadian_phase(3.0) == "night"
    assert _get_circadian_phase(8.0) == "morning"
    assert _get_circadian_phase(12.0) == "midday"
    assert _get_circadian_phase(15.0) == "afternoon"
    assert _get_circadian_phase(19.0) == "evening"
    assert _get_circadian_phase(23.0) == "night"


# ---------------------------------------------------------------------------
# 8. Learned selection improves with history
# ---------------------------------------------------------------------------

def test_selection_prefers_effective_intervention():
    """After recording outcomes, engine should prefer the more effective intervention."""
    engine = _make_engine()
    user = "user19"

    # Record: physiological_sigh works great, box_breathing does not
    for _ in range(5):
        engine.record_outcome(user, "physiological_sigh", "stress_spike",
                              -0.4, 0.8, 0.2, 0.4, 0.8, 0.3, hour_of_day=10.0)
        engine.record_outcome(user, "box_breathing", "stress_spike",
                              -0.4, 0.8, -0.4, 0.8, 0.8, 0.8, hour_of_day=10.0)

    state = _stressed_state()
    result = engine.select_intervention(user, state, "stress_spike", hour_of_day=10.0)
    # Physiological sigh should score higher
    scores = result["selection_scores"]
    assert scores.get("physiological_sigh", 0) > scores.get("box_breathing", 0)


# ---------------------------------------------------------------------------
# 9. EmotionState dataclass
# ---------------------------------------------------------------------------

def test_emotion_state_defaults():
    """EmotionState should auto-set timestamp."""
    before = time.time()
    state = EmotionState(valence=0.0, arousal=0.5, stress_index=0.3, focus_index=0.5)
    after = time.time()
    assert before <= state.timestamp <= after


# ---------------------------------------------------------------------------
# 10. InterventionOutcome properties
# ---------------------------------------------------------------------------

def test_outcome_deltas():
    """Outcome should compute correct deltas."""
    o = InterventionOutcome(
        user_id="u",
        intervention_id="box_breathing",
        trigger_type="stress_spike",
        valence_before=-0.5,
        arousal_before=0.9,
        valence_after=0.1,
        arousal_after=0.4,
        stress_before=0.8,
        stress_after=0.3,
    )
    assert abs(o.valence_delta - 0.6) < 0.001
    assert abs(o.arousal_delta - (-0.5)) < 0.001
    assert abs(o.stress_delta - 0.5) < 0.001
    assert o.effective is True


def test_outcome_not_effective():
    """Outcome with no improvement should not be effective."""
    o = InterventionOutcome(
        user_id="u",
        intervention_id="desk_stretch",
        trigger_type="focus_decline",
        valence_before=0.0,
        arousal_before=0.5,
        valence_after=0.0,
        arousal_after=0.5,
        stress_before=0.5,
        stress_after=0.5,
    )
    assert o.effective is False


# ---------------------------------------------------------------------------
# 11. API route integration (sync logic via httpx TestClient)
# ---------------------------------------------------------------------------

def test_api_status_endpoint():
    """The /status endpoint should return ok."""
    from fastapi.testclient import TestClient
    from api.routes.micro_intervention import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.get("/interventions/jit/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["library_size"] > 0


def test_api_check_trigger_no_trigger():
    """POST /check-trigger with normal state should return no trigger."""
    from fastapi.testclient import TestClient
    from api.routes.micro_intervention import router, _engine
    from fastapi import FastAPI

    # Reset engine state for test isolation
    _engine._outcomes.clear()
    _engine._triggers.clear()
    _engine._prev_state.clear()
    _engine._last_intervention_ts.clear()
    _engine._daily_counts.clear()

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.post("/interventions/jit/check-trigger", json={
        "user_id": "api_test_1",
        "valence": 0.3,
        "arousal": 0.4,
        "stress_index": 0.3,
        "focus_index": 0.6,
    })
    assert resp.status_code == 200
    assert resp.json()["triggered"] is False


def test_api_full_flow():
    """Full API flow: check trigger -> select -> record outcome -> stats."""
    from fastapi.testclient import TestClient
    from api.routes.micro_intervention import router, _engine
    from fastapi import FastAPI

    _engine._outcomes.clear()
    _engine._triggers.clear()
    _engine._prev_state.clear()
    _engine._last_intervention_ts.clear()
    _engine._daily_counts.clear()

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    user = "api_flow_user"

    # 1. Check trigger (high stress)
    resp = client.post("/interventions/jit/check-trigger", json={
        "user_id": user,
        "valence": -0.4,
        "arousal": 0.8,
        "stress_index": 0.80,
        "focus_index": 0.5,
    })
    assert resp.status_code == 200
    assert resp.json()["triggered"] is True

    # 2. Select intervention
    resp = client.post("/interventions/jit/select", json={
        "user_id": user,
        "valence": -0.4,
        "arousal": 0.8,
        "stress_index": 0.80,
        "focus_index": 0.5,
        "trigger_type": "stress_spike",
        "hour_of_day": 10.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "intervention" in data
    iv_id = data["intervention"]["id"]

    # 3. Record outcome
    resp = client.post("/interventions/jit/record-outcome", json={
        "user_id": user,
        "intervention_id": iv_id,
        "trigger_type": "stress_spike",
        "valence_before": -0.4,
        "arousal_before": 0.8,
        "valence_after": 0.1,
        "arousal_after": 0.5,
        "stress_before": 0.8,
        "stress_after": 0.3,
        "hour_of_day": 10.0,
    })
    assert resp.status_code == 200
    assert resp.json()["recorded"] is True

    # 4. Get stats
    resp = client.get(f"/interventions/jit/stats/{user}")
    assert resp.status_code == 200
    stats = resp.json()
    assert stats["total_outcomes"] == 1
    assert "effectiveness" in stats
