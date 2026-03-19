"""Tests for emotional first-aid protocols (issue #438).

Covers: crisis detection (panic, acute stress, dissociation, rage, none),
protocol selection, step retrieval, step evaluation, advance/repeat logic,
protocol library structure, safety rails, and API route integration.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.emotional_first_aid import (
    CrisisState,
    CrisisType,
    Protocol,
    ProtocolCategory,
    ProtocolStep,
    StepEvaluation,
    PROTOCOL_LIBRARY,
    _PROTOCOL_BY_ID,
    _CLINICAL_DISCLAIMER,
    _IMPROVEMENT_THRESHOLD,
    _SEVERE_EPISODE_THRESHOLD,
    advance_or_repeat,
    detect_crisis_type,
    evaluate_step_effectiveness,
    get_current_step,
    protocol_to_dict,
    select_protocol,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _panic_state(**overrides) -> CrisisState:
    defaults = dict(valence=-0.7, arousal=0.90, stress_index=0.5, anger_index=0.1)
    defaults.update(overrides)
    return CrisisState(**defaults)


def _stress_state(**overrides) -> CrisisState:
    defaults = dict(valence=-0.2, arousal=0.6, stress_index=0.85, anger_index=0.1)
    defaults.update(overrides)
    return CrisisState(**defaults)


def _dissociation_state(**overrides) -> CrisisState:
    defaults = dict(valence=0.05, arousal=0.10, stress_index=0.2, anger_index=0.0)
    defaults.update(overrides)
    return CrisisState(**defaults)


def _rage_state(**overrides) -> CrisisState:
    defaults = dict(valence=-0.3, arousal=0.80, stress_index=0.6, anger_index=0.75)
    defaults.update(overrides)
    return CrisisState(**defaults)


def _normal_state(**overrides) -> CrisisState:
    defaults = dict(valence=0.3, arousal=0.4, stress_index=0.3, anger_index=0.1)
    defaults.update(overrides)
    return CrisisState(**defaults)


# ---------------------------------------------------------------------------
# 1. Crisis detection
# ---------------------------------------------------------------------------


def test_detect_panic():
    """High arousal + very negative valence should detect panic."""
    state = _panic_state()
    result = detect_crisis_type(state)
    assert result["crisis_type"] == "panic"
    assert result["detected"] is True
    assert result["severity"] > 0.5
    assert "very_high_arousal" in result["indicators"]
    assert "very_negative_valence" in result["indicators"]


def test_detect_acute_stress():
    """High stress index should detect acute stress."""
    state = _stress_state()
    result = detect_crisis_type(state)
    assert result["crisis_type"] == "acute_stress"
    assert result["detected"] is True
    assert result["severity"] >= 0.8


def test_detect_dissociation():
    """Very low arousal + flat affect should detect dissociation."""
    state = _dissociation_state()
    result = detect_crisis_type(state)
    assert result["crisis_type"] == "dissociation"
    assert result["detected"] is True
    assert "very_low_arousal" in result["indicators"]
    assert "flat_affect" in result["indicators"]


def test_detect_rage():
    """High arousal + high anger should detect rage."""
    state = _rage_state()
    result = detect_crisis_type(state)
    assert result["crisis_type"] == "rage"
    assert result["detected"] is True
    assert "high_arousal" in result["indicators"]
    assert "high_anger" in result["indicators"]


def test_detect_no_crisis():
    """Normal state should detect no crisis."""
    state = _normal_state()
    result = detect_crisis_type(state)
    assert result["crisis_type"] == "none"
    assert result["detected"] is False
    assert result["severity"] == 0.0


def test_severe_episode_flag():
    """Severity >= threshold should flag as severe with safety note."""
    state = _panic_state(arousal=0.95, valence=-0.9)
    result = detect_crisis_type(state)
    assert result["is_severe"] is True
    assert "professional" in result["safety_note"].lower()


def test_clinical_disclaimer_always_present():
    """Every detection result must include the clinical disclaimer."""
    for state in [_panic_state(), _normal_state(), _dissociation_state()]:
        result = detect_crisis_type(state)
        assert result["clinical_disclaimer"] == _CLINICAL_DISCLAIMER


# ---------------------------------------------------------------------------
# 2. Protocol selection
# ---------------------------------------------------------------------------


def test_select_protocol_for_panic():
    """Panic crisis should select a panic or grounding protocol."""
    result = select_protocol(CrisisType.PANIC, severity=0.7)
    assert result["selected"] is True
    assert result["protocol"]["category"] in ("panic", "grounding")


def test_select_protocol_for_dissociation():
    """Dissociation crisis should select a dissociation or grounding protocol."""
    result = select_protocol(CrisisType.DISSOCIATION, severity=0.5)
    assert result["selected"] is True
    # Grounding protocols (e.g. 5-4-3-2-1) also target dissociation
    assert result["protocol"]["category"] in ("dissociation", "grounding")


def test_select_protocol_for_rage():
    """Rage crisis should select a rage or grounding protocol."""
    result = select_protocol(CrisisType.RAGE, severity=0.6)
    assert result["selected"] is True
    assert result["protocol"]["category"] in ("rage", "grounding")


def test_select_protocol_none_crisis():
    """No crisis should return selected=False."""
    result = select_protocol(CrisisType.NONE, severity=0.0)
    assert result["selected"] is False
    assert result["reason"] == "no_crisis_detected"


def test_select_protocol_includes_alternatives():
    """Selection result should list alternative protocol IDs."""
    result = select_protocol(CrisisType.PANIC, severity=0.7)
    assert "alternatives" in result
    assert isinstance(result["alternatives"], list)


# ---------------------------------------------------------------------------
# 3. Step retrieval
# ---------------------------------------------------------------------------


def test_get_current_step_valid():
    """Should return step details for a valid protocol and step."""
    protocol_id = PROTOCOL_LIBRARY[0].id
    result = get_current_step(protocol_id, 1)
    assert "step" in result
    assert result["step"]["step_number"] == 1
    assert "instruction" in result["step"]
    assert result["progress"]["current"] == 1


def test_get_current_step_last():
    """Last step should have is_last_step=True."""
    protocol = PROTOCOL_LIBRARY[0]
    total = len(protocol.steps)
    result = get_current_step(protocol.id, total)
    assert result["is_last_step"] is True


def test_get_current_step_invalid_protocol():
    """Unknown protocol ID should return error."""
    result = get_current_step("nonexistent_protocol", 1)
    assert "error" in result


def test_get_current_step_out_of_range():
    """Step number out of range should return error."""
    protocol_id = PROTOCOL_LIBRARY[0].id
    result = get_current_step(protocol_id, 999)
    assert "error" in result


# ---------------------------------------------------------------------------
# 4. Step evaluation
# ---------------------------------------------------------------------------


def test_evaluate_step_effective():
    """Arousal decrease above threshold should be effective."""
    evaluation = evaluate_step_effectiveness(
        arousal_before=0.8, arousal_after=0.6,
        stress_before=0.7, stress_after=0.5,
        step_number=1,
    )
    assert evaluation.effective is True
    assert evaluation.arousal_improved is True
    assert evaluation.stress_improved is True
    assert evaluation.recommendation == "advance"


def test_evaluate_step_not_effective():
    """No improvement should be not effective."""
    evaluation = evaluate_step_effectiveness(
        arousal_before=0.8, arousal_after=0.8,
        stress_before=0.7, stress_after=0.7,
        step_number=2,
    )
    assert evaluation.effective is False
    assert evaluation.recommendation == "repeat"


def test_evaluate_step_partial_improvement():
    """Improvement in only arousal should still be effective."""
    evaluation = evaluate_step_effectiveness(
        arousal_before=0.8, arousal_after=0.6,
        stress_before=0.7, stress_after=0.68,  # stress barely changed
        step_number=1,
    )
    assert evaluation.effective is True
    assert evaluation.arousal_improved is True
    assert evaluation.stress_improved is False


# ---------------------------------------------------------------------------
# 5. Advance/repeat logic
# ---------------------------------------------------------------------------


def test_advance_on_effective_step():
    """Effective step should advance to next step."""
    protocol = PROTOCOL_LIBRARY[0]
    result = advance_or_repeat(
        protocol_id=protocol.id,
        current_step=1,
        arousal_before=0.8, arousal_after=0.6,
        stress_before=0.7, stress_after=0.5,
    )
    assert result["action"] == "advance"
    assert result["next_step"] == 2
    assert result["protocol_complete"] is False


def test_repeat_on_ineffective_step():
    """Ineffective step should repeat."""
    protocol = PROTOCOL_LIBRARY[0]
    result = advance_or_repeat(
        protocol_id=protocol.id,
        current_step=1,
        arousal_before=0.8, arousal_after=0.8,
        stress_before=0.7, stress_after=0.7,
    )
    assert result["action"] == "repeat"
    assert result["next_step"] == 1
    assert result["repeat_count"] == 1


def test_force_advance_after_max_repeats():
    """Should advance after max repeats even without improvement."""
    protocol = PROTOCOL_LIBRARY[0]
    result = advance_or_repeat(
        protocol_id=protocol.id,
        current_step=1,
        arousal_before=0.8, arousal_after=0.8,
        stress_before=0.7, stress_after=0.7,
        repeat_count=2,
        max_repeats=2,
    )
    assert result["action"] == "advance"
    assert result["next_step"] == 2


def test_protocol_complete_on_last_step():
    """Advancing past last step should complete the protocol."""
    protocol = PROTOCOL_LIBRARY[0]
    total = len(protocol.steps)
    result = advance_or_repeat(
        protocol_id=protocol.id,
        current_step=total,
        arousal_before=0.8, arousal_after=0.6,
        stress_before=0.7, stress_after=0.5,
    )
    assert result["action"] == "complete"
    assert result["protocol_complete"] is True


def test_advance_unknown_protocol():
    """Unknown protocol should return error."""
    result = advance_or_repeat(
        protocol_id="nonexistent",
        current_step=1,
        arousal_before=0.8, arousal_after=0.6,
        stress_before=0.7, stress_after=0.5,
    )
    assert "error" in result


# ---------------------------------------------------------------------------
# 6. Protocol library structure
# ---------------------------------------------------------------------------


def test_library_has_all_categories():
    """Library must have protocols for every category."""
    categories_found = {p.category for p in PROTOCOL_LIBRARY}
    for cat in ProtocolCategory:
        assert cat in categories_found, f"Missing category: {cat.value}"


def test_library_has_all_crisis_types_covered():
    """Every non-none crisis type should have at least one protocol."""
    for crisis in CrisisType:
        if crisis == CrisisType.NONE:
            continue
        matching = [p for p in PROTOCOL_LIBRARY if crisis in p.crisis_types]
        assert len(matching) > 0, f"No protocol covers crisis type: {crisis.value}"


def test_protocol_ids_unique():
    """All protocol IDs must be unique."""
    ids = [p.id for p in PROTOCOL_LIBRARY]
    assert len(ids) == len(set(ids))


def test_all_protocols_have_steps():
    """Every protocol must have at least 4 steps."""
    for p in PROTOCOL_LIBRARY:
        assert len(p.steps) >= 4, f"Protocol {p.id} has only {len(p.steps)} steps"


def test_protocol_to_dict():
    """protocol_to_dict should serialize a known protocol."""
    protocol = PROTOCOL_LIBRARY[0]
    result = protocol_to_dict(protocol.id)
    assert result is not None
    assert result["id"] == protocol.id
    assert "steps" in result
    assert result["total_steps"] == len(protocol.steps)


def test_protocol_to_dict_unknown():
    """protocol_to_dict for unknown ID should return None."""
    assert protocol_to_dict("nonexistent") is None


# ---------------------------------------------------------------------------
# 7. Safety rails
# ---------------------------------------------------------------------------


def test_safety_note_for_severe():
    """Severe episodes should include professional help recommendation."""
    state = _panic_state(arousal=0.95, valence=-0.95)
    result = detect_crisis_type(state)
    assert result["is_severe"] is True
    assert "988" in result["safety_note"]


def test_safety_note_for_moderate():
    """Non-severe detected crises should still include safety note."""
    state = _stress_state(stress_index=0.82)
    result = detect_crisis_type(state)
    assert result["detected"] is True
    assert len(result["safety_note"]) > 0


# ---------------------------------------------------------------------------
# 8. Dataclass serialization
# ---------------------------------------------------------------------------


def test_crisis_state_to_dict():
    """CrisisState.to_dict should include all fields."""
    state = _panic_state()
    d = state.to_dict()
    assert "valence" in d
    assert "arousal" in d
    assert "stress_index" in d
    assert "anger_index" in d
    assert "timestamp" in d


def test_step_evaluation_to_dict():
    """StepEvaluation.to_dict should include all fields."""
    ev = evaluate_step_effectiveness(0.8, 0.6, 0.7, 0.5, step_number=1)
    d = ev.to_dict()
    assert "step_number" in d
    assert "arousal_before" in d
    assert "effective" in d
    assert "recommendation" in d


# ---------------------------------------------------------------------------
# 9. Priority ordering in detection
# ---------------------------------------------------------------------------


def test_panic_takes_priority_over_rage():
    """When both panic and rage conditions are met, panic wins."""
    # Panic: arousal >= 0.85 + valence <= -0.5
    # Rage: arousal >= 0.75 + anger >= 0.6
    state = CrisisState(
        valence=-0.7, arousal=0.90, stress_index=0.5, anger_index=0.80
    )
    result = detect_crisis_type(state)
    assert result["crisis_type"] == "panic"


def test_dissociation_takes_priority_over_stress():
    """Dissociation check comes before acute stress."""
    # Dissociation: arousal <= 0.20 + flat valence
    # Also has high stress
    state = CrisisState(
        valence=0.05, arousal=0.15, stress_index=0.85, anger_index=0.0
    )
    result = detect_crisis_type(state)
    assert result["crisis_type"] == "dissociation"


# ---------------------------------------------------------------------------
# 10. API route integration
# ---------------------------------------------------------------------------


def test_api_status_endpoint():
    """The /status endpoint should return ok."""
    from fastapi.testclient import TestClient
    from api.routes.emotional_first_aid import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.get("/first-aid/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["total_protocols"] == len(PROTOCOL_LIBRARY)


def test_api_detect_no_crisis():
    """POST /detect with normal state should return no crisis."""
    from fastapi.testclient import TestClient
    from api.routes.emotional_first_aid import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.post("/first-aid/detect", json={
        "valence": 0.3,
        "arousal": 0.4,
        "stress_index": 0.3,
        "anger_index": 0.1,
    })
    assert resp.status_code == 200
    assert resp.json()["crisis_type"] == "none"
    assert resp.json()["detected"] is False


def test_api_detect_panic():
    """POST /detect with panic state should detect panic."""
    from fastapi.testclient import TestClient
    from api.routes.emotional_first_aid import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.post("/first-aid/detect", json={
        "valence": -0.7,
        "arousal": 0.90,
        "stress_index": 0.5,
        "anger_index": 0.1,
    })
    assert resp.status_code == 200
    assert resp.json()["crisis_type"] == "panic"


def test_api_full_flow():
    """Full API flow: detect -> start -> evaluate-step -> advance."""
    from fastapi.testclient import TestClient
    from api.routes.emotional_first_aid import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    # 1. Detect crisis
    resp = client.post("/first-aid/detect", json={
        "valence": -0.7,
        "arousal": 0.90,
        "stress_index": 0.7,
        "anger_index": 0.1,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["crisis_type"] == "panic"
    severity = data["severity"]

    # 2. Start protocol
    resp = client.post("/first-aid/start", json={
        "crisis_type": "panic",
        "severity": severity,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["selected"] is True
    protocol_id = data["protocol"]["id"]
    assert "first_step" in data

    # 3. Evaluate step (effective)
    resp = client.post("/first-aid/evaluate-step", json={
        "protocol_id": protocol_id,
        "current_step": 1,
        "arousal_before": 0.9,
        "arousal_after": 0.7,
        "stress_before": 0.7,
        "stress_after": 0.5,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "advance"
    assert data["next_step"] == 2

    # 4. List protocols
    resp = client.get("/first-aid/protocols")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_protocols"] == len(PROTOCOL_LIBRARY)
    assert len(data["categories"]) > 0


def test_api_start_invalid_crisis_type():
    """POST /start with invalid crisis type should return error."""
    from fastapi.testclient import TestClient
    from api.routes.emotional_first_aid import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.post("/first-aid/start", json={
        "crisis_type": "invalid_type",
        "severity": 0.5,
    })
    assert resp.status_code == 200
    assert "error" in resp.json()
