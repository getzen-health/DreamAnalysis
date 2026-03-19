"""Tests for embodied AI emotional companion (issue #457).

Covers: emotional tone detection, conversation state machine, therapeutic
stance selection, EEG-aware response adaptation, response template generation,
session memory tracking, companion profile computation, serialization,
and API route integration.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.embodied_companion import (
    CompanionProfile,
    ConversationState,
    EEGState,
    EmotionalTone,
    ResponseTemplate,
    SessionMemory,
    TherapeuticStance,
    _CLINICAL_DISCLAIMER,
    _RESPONSE_TEMPLATES,
    adapt_response_to_eeg,
    compute_companion_profile,
    detect_conversation_state,
    generate_response_template,
    profile_to_dict,
    select_therapeutic_stance,
    track_session,
)


# ---------------------------------------------------------------------------
# Helpers — EEG state factories
# ---------------------------------------------------------------------------


def _stressed_eeg(**overrides) -> EEGState:
    defaults = dict(
        valence=-0.3, arousal=0.75, stress_index=0.80,
        focus_index=0.3, anger_index=0.2, relaxation_index=0.1,
    )
    defaults.update(overrides)
    return EEGState(**defaults)


def _calm_eeg(**overrides) -> EEGState:
    defaults = dict(
        valence=0.1, arousal=0.40, stress_index=0.15,
        focus_index=0.6, anger_index=0.05, relaxation_index=0.7,
    )
    defaults.update(overrides)
    return EEGState(**defaults)


def _positive_eeg(**overrides) -> EEGState:
    defaults = dict(
        valence=0.5, arousal=0.55, stress_index=0.10,
        focus_index=0.7, anger_index=0.0, relaxation_index=0.6,
    )
    defaults.update(overrides)
    return EEGState(**defaults)


def _low_energy_eeg(**overrides) -> EEGState:
    defaults = dict(
        valence=-0.1, arousal=0.15, stress_index=0.2,
        focus_index=0.3, anger_index=0.0, relaxation_index=0.4,
    )
    defaults.update(overrides)
    return EEGState(**defaults)


def _anxious_eeg(**overrides) -> EEGState:
    defaults = dict(
        valence=-0.4, arousal=0.80, stress_index=0.50,
        focus_index=0.4, anger_index=0.15, relaxation_index=0.2,
    )
    defaults.update(overrides)
    return EEGState(**defaults)


def _neutral_eeg(**overrides) -> EEGState:
    defaults = dict(
        valence=0.0, arousal=0.65, stress_index=0.40,
        focus_index=0.5, anger_index=0.05, relaxation_index=0.5,
    )
    defaults.update(overrides)
    return EEGState(**defaults)


# ---------------------------------------------------------------------------
# 1. Emotional tone detection (via adapt_response_to_eeg)
# ---------------------------------------------------------------------------


class TestEmotionalToneDetection:
    def test_stressed_tone(self):
        """High stress should detect stressed tone."""
        result = adapt_response_to_eeg(_stressed_eeg())
        assert result["tone"] == "stressed"

    def test_anxious_tone(self):
        """High arousal + negative valence (non-high stress) -> anxious."""
        result = adapt_response_to_eeg(_anxious_eeg())
        assert result["tone"] == "anxious"

    def test_low_energy_tone(self):
        """Very low arousal -> low_energy."""
        result = adapt_response_to_eeg(_low_energy_eeg())
        assert result["tone"] == "low_energy"

    def test_positive_tone(self):
        """Positive valence + moderate arousal -> positive."""
        result = adapt_response_to_eeg(_positive_eeg())
        assert result["tone"] == "positive"

    def test_calm_tone(self):
        """Low stress + moderate arousal -> calm."""
        result = adapt_response_to_eeg(_calm_eeg())
        assert result["tone"] == "calm"

    def test_neutral_fallback(self):
        """Ambiguous state -> neutral."""
        result = adapt_response_to_eeg(_neutral_eeg())
        assert result["tone"] == "neutral"


# ---------------------------------------------------------------------------
# 2. EEG-aware response adaptation
# ---------------------------------------------------------------------------


class TestEEGAdaptation:
    def test_stressed_uses_simple_slow(self):
        """Stressed user -> simple complexity, slow pace."""
        result = adapt_response_to_eeg(_stressed_eeg())
        assert result["complexity"] == "simple"
        assert result["pace"] == "slow"
        assert result["primary_action"] == "validate"

    def test_anxious_uses_grounding(self):
        """Anxious user -> grounding as primary action."""
        result = adapt_response_to_eeg(_anxious_eeg())
        assert result["primary_action"] == "ground"

    def test_calm_uses_moderate(self):
        """Calm user -> moderate complexity, normal pace."""
        result = adapt_response_to_eeg(_calm_eeg())
        assert result["complexity"] == "moderate"
        assert result["pace"] == "normal"
        assert result["primary_action"] == "explore"

    def test_positive_uses_detailed(self):
        """Positive user -> detailed complexity, engaged pace."""
        result = adapt_response_to_eeg(_positive_eeg())
        assert result["complexity"] == "detailed"
        assert result["pace"] == "engaged"
        assert result["primary_action"] == "build"

    def test_low_energy_gentle_activate(self):
        """Low energy -> gently_activate."""
        result = adapt_response_to_eeg(_low_energy_eeg())
        assert result["primary_action"] == "gently_activate"

    def test_includes_eeg_state(self):
        """Adaptation result should include the EEG state."""
        result = adapt_response_to_eeg(_calm_eeg())
        assert "eeg_state" in result
        assert "valence" in result["eeg_state"]


# ---------------------------------------------------------------------------
# 3. Conversation state machine
# ---------------------------------------------------------------------------


class TestConversationState:
    def test_starts_at_greeting(self):
        """Invalid state defaults to greeting."""
        result = detect_conversation_state("invalid", 0, 0.0)
        assert result["current_state"] == "greeting"

    def test_greeting_advances_after_one_turn(self):
        """Greeting should advance after 1 turn."""
        result = detect_conversation_state("greeting", 1, 5.0)
        assert result["can_advance"] is True
        assert result["next_state"] == "check_in"

    def test_check_in_needs_time_and_turns(self):
        """Check-in should not advance too early."""
        result = detect_conversation_state("check_in", 1, 5.0)
        assert result["can_advance"] is False

    def test_check_in_advances_with_engagement(self):
        """Check-in advances with 2+ turns and sufficient time."""
        result = detect_conversation_state("check_in", 3, 25.0)
        assert result["can_advance"] is True
        assert result["next_state"] == "active_listening"

    def test_user_wants_to_close(self):
        """User requesting close -> jumps to closing."""
        result = detect_conversation_state(
            "active_listening", 5, 60.0, user_wants_to_close=True
        )
        assert result["next_state"] == "closing"

    def test_closing_stays_at_closing(self):
        """Closing state should not advance further."""
        result = detect_conversation_state("closing", 10, 300.0)
        assert result["next_state"] == "closing"
        assert result["can_advance"] is False

    def test_time_based_force_advance(self):
        """Very long time in state should force advance."""
        # check_in min duration is 20s; 3x = 60s
        result = detect_conversation_state("check_in", 1, 65.0)
        assert result["can_advance"] is True
        assert result["reason"] == "time_based_advance"


# ---------------------------------------------------------------------------
# 4. Therapeutic stance selection
# ---------------------------------------------------------------------------


class TestTherapeuticStance:
    def test_stressed_always_supportive(self):
        """Stressed user -> supportive stance regardless of state."""
        result = select_therapeutic_stance(
            _stressed_eeg(), "guidance", turn_count=10, themes=["anxiety"]
        )
        assert result["stance"] == "supportive"

    def test_early_state_supportive(self):
        """Greeting/check-in -> supportive stance."""
        result = select_therapeutic_stance(
            _calm_eeg(), "greeting", turn_count=0
        )
        assert result["stance"] == "supportive"

    def test_calm_in_guidance_psychoeducational(self):
        """Calm user in guidance -> psychoeducational."""
        result = select_therapeutic_stance(
            _calm_eeg(), "guidance", turn_count=5
        )
        assert result["stance"] == "psychoeducational"

    def test_positive_with_themes_reflective(self):
        """Positive user with themes -> reflective."""
        result = select_therapeutic_stance(
            _positive_eeg(), "active_listening", turn_count=5, themes=["growth"]
        )
        assert result["stance"] == "reflective"

    def test_many_turns_neutral_challenging(self):
        """Many turns + neutral state -> challenging."""
        result = select_therapeutic_stance(
            _neutral_eeg(), "active_listening", turn_count=10
        )
        assert result["stance"] == "challenging"

    def test_includes_alternatives(self):
        """Result should include alternative stances."""
        result = select_therapeutic_stance(
            _calm_eeg(), "guidance", turn_count=5
        )
        assert "alternatives" in result
        assert isinstance(result["alternatives"], list)

    def test_includes_emotional_tone(self):
        """Result should include the detected emotional tone."""
        result = select_therapeutic_stance(
            _stressed_eeg(), "check_in", turn_count=2
        )
        assert result["emotional_tone"] == "stressed"


# ---------------------------------------------------------------------------
# 5. Response template generation
# ---------------------------------------------------------------------------


class TestResponseTemplate:
    def test_generates_template_for_stressed_greeting(self):
        """Should generate a template for stressed user at greeting."""
        result = generate_response_template(
            _stressed_eeg(), "greeting", "supportive"
        )
        assert "template" in result
        assert result["template"]["tone"] == "stressed"
        assert result["template"]["conversation_state"] == "greeting"
        assert len(result["template"]["text"]) > 0

    def test_generates_template_for_calm_guidance(self):
        """Should generate a template for calm user at guidance."""
        result = generate_response_template(
            _calm_eeg(), "guidance", "psychoeducational"
        )
        assert result["template"]["conversation_state"] == "guidance"
        assert result["template"]["complexity"] == "moderate"

    def test_includes_follow_up_prompt(self):
        """Template should include a follow-up prompt."""
        result = generate_response_template(
            _calm_eeg(), "check_in", "reflective"
        )
        assert result["template"]["follow_up_prompt"] is not None

    def test_includes_clinical_disclaimer(self):
        """Template result should include clinical disclaimer."""
        result = generate_response_template(
            _neutral_eeg(), "check_in", "supportive"
        )
        assert result["clinical_disclaimer"] == _CLINICAL_DISCLAIMER

    def test_invalid_state_defaults_to_check_in(self):
        """Invalid conversation state should default to check_in."""
        result = generate_response_template(
            _neutral_eeg(), "invalid_state", "supportive"
        )
        assert result["template"]["conversation_state"] == "check_in"

    def test_all_states_have_templates(self):
        """Every conversation state should have templates for every tone."""
        for state in ConversationState:
            assert state.value in _RESPONSE_TEMPLATES, (
                f"Missing templates for state {state.value}"
            )
            for tone in EmotionalTone:
                templates = _RESPONSE_TEMPLATES[state.value].get(tone.value, [])
                assert len(templates) >= 1, (
                    f"Missing template for state={state.value}, tone={tone.value}"
                )


# ---------------------------------------------------------------------------
# 6. Session tracking
# ---------------------------------------------------------------------------


class TestSessionTracking:
    def test_initializes_empty_session(self):
        """First call with empty memory should initialize structure."""
        result = track_session({}, _neutral_eeg())
        assert "themes" in result
        assert "emotional_shifts" in result
        assert "readings" in result
        assert result["turn_count"] == 1

    def test_records_theme(self):
        """Should record new themes."""
        result = track_session({}, _neutral_eeg(), theme="anxiety")
        assert "anxiety" in result["themes"]

    def test_no_duplicate_themes(self):
        """Should not duplicate existing themes."""
        mem = track_session({}, _neutral_eeg(), theme="anxiety")
        mem = track_session(mem, _neutral_eeg(), theme="anxiety")
        assert mem["themes"].count("anxiety") == 1

    def test_records_intervention(self):
        """Should record interventions tried."""
        result = track_session(
            {}, _stressed_eeg(), intervention="breathing_exercise"
        )
        assert "breathing_exercise" in result["interventions_tried"]

    def test_detects_emotional_shift(self):
        """Significant valence change should record an emotional shift."""
        mem = track_session({}, EEGState(valence=-0.3, arousal=0.5))
        mem = track_session(mem, EEGState(valence=0.2, arousal=0.5))
        assert len(mem["emotional_shifts"]) >= 1
        shift = mem["emotional_shifts"][0]
        assert shift["direction"] == "positive"

    def test_no_shift_for_small_change(self):
        """Small valence changes should not register as shifts."""
        mem = track_session({}, EEGState(valence=0.1, arousal=0.5))
        mem = track_session(mem, EEGState(valence=0.15, arousal=0.5))
        assert len(mem["emotional_shifts"]) == 0

    def test_tracks_duration(self):
        """Session memory should track duration."""
        mem = track_session({}, _neutral_eeg())
        assert "duration_seconds" in mem

    def test_intervention_effectiveness(self):
        """Should mark intervention as effective if stress drops."""
        mem = track_session(
            {}, EEGState(valence=0.0, arousal=0.5, stress_index=0.7)
        )
        mem = track_session(
            mem,
            EEGState(valence=0.0, arousal=0.5, stress_index=0.5),
            intervention="grounding",
        )
        assert "grounding" in mem["interventions_effective"]


# ---------------------------------------------------------------------------
# 7. Companion profile computation
# ---------------------------------------------------------------------------


class TestCompanionProfile:
    def test_compute_basic_profile(self):
        """Should compute a complete profile from basic inputs."""
        profile = compute_companion_profile(_neutral_eeg())
        assert isinstance(profile, CompanionProfile)
        assert profile.conversation_state in ConversationState
        assert profile.therapeutic_stance in TherapeuticStance
        assert profile.emotional_tone in EmotionalTone

    def test_profile_for_stressed_user(self):
        """Stressed user profile should be supportive with simple language."""
        profile = compute_companion_profile(
            _stressed_eeg(), conversation_state="check_in", turn_count=3
        )
        assert profile.therapeutic_stance == TherapeuticStance.SUPPORTIVE
        assert profile.response_template.complexity == "simple"

    def test_profile_advances_state(self):
        """Profile should advance conversation state when ready."""
        profile = compute_companion_profile(
            _calm_eeg(),
            conversation_state="greeting",
            turn_count=2,
            session_duration=15.0,
        )
        assert profile.conversation_state == ConversationState.CHECK_IN

    def test_profile_includes_session_summary(self):
        """Profile should include session summary."""
        profile = compute_companion_profile(
            _calm_eeg(), turn_count=5, session_duration=120.0
        )
        assert "turn_count" in profile.session_summary
        assert "duration_seconds" in profile.session_summary

    def test_profile_includes_eeg_adaptation(self):
        """Profile should include EEG adaptation parameters."""
        profile = compute_companion_profile(_stressed_eeg())
        assert "complexity" in profile.eeg_adaptation
        assert "pace" in profile.eeg_adaptation

    def test_profile_to_dict_serialization(self):
        """profile_to_dict should produce a complete dict."""
        profile = compute_companion_profile(_calm_eeg())
        d = profile_to_dict(profile)
        assert "conversation_state" in d
        assert "therapeutic_stance" in d
        assert "emotional_tone" in d
        assert "response_template" in d
        assert "session_summary" in d
        assert "eeg_adaptation" in d
        assert "clinical_disclaimer" in d


# ---------------------------------------------------------------------------
# 8. Dataclass serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_eeg_state_to_dict(self):
        """EEGState.to_dict should include all fields."""
        eeg = _stressed_eeg()
        d = eeg.to_dict()
        assert "valence" in d
        assert "arousal" in d
        assert "stress_index" in d
        assert "focus_index" in d
        assert "anger_index" in d
        assert "relaxation_index" in d
        assert "timestamp" in d

    def test_eeg_state_values_rounded(self):
        """EEGState.to_dict should round values to 4 decimal places."""
        eeg = EEGState(valence=0.123456789)
        d = eeg.to_dict()
        assert d["valence"] == 0.1235

    def test_response_template_to_dict(self):
        """ResponseTemplate.to_dict should include all fields."""
        template = ResponseTemplate(
            text="Hello",
            stance=TherapeuticStance.SUPPORTIVE,
            tone=EmotionalTone.CALM,
            conversation_state=ConversationState.GREETING,
            complexity="simple",
            pace="slow",
            follow_up_prompt="How are you?",
        )
        d = template.to_dict()
        assert d["text"] == "Hello"
        assert d["stance"] == "supportive"
        assert d["tone"] == "calm"
        assert d["conversation_state"] == "greeting"
        assert d["follow_up_prompt"] == "How are you?"

    def test_session_memory_to_dict(self):
        """SessionMemory.to_dict should include all tracking fields."""
        mem = SessionMemory(
            session_id="test-123",
            themes=["anxiety", "work"],
            turn_count=5,
        )
        d = mem.to_dict()
        assert d["session_id"] == "test-123"
        assert "anxiety" in d["themes"]
        assert d["turn_count"] == 5
        assert "duration_seconds" in d


# ---------------------------------------------------------------------------
# 9. API route integration
# ---------------------------------------------------------------------------


class TestAPIRoutes:
    def test_status_endpoint(self):
        """GET /companion/status should return ok."""
        from fastapi.testclient import TestClient
        from api.routes.embodied_companion import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.get("/companion/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "embodied_companion"
        assert "conversation_states" in data
        assert len(data["conversation_states"]) == 6
        assert "therapeutic_stances" in data
        assert "clinical_disclaimer" in data

    def test_respond_endpoint_stressed(self):
        """POST /companion/respond with stressed state should return supportive response."""
        from fastapi.testclient import TestClient
        from api.routes.embodied_companion import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.post("/companion/respond", json={
            "valence": -0.3,
            "arousal": 0.75,
            "stress_index": 0.80,
            "conversation_state": "check_in",
            "turn_count": 3,
            "session_duration": 30.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["therapeutic_stance"] == "supportive"
        assert data["emotional_tone"] == "stressed"
        assert "response_template" in data
        assert "clinical_disclaimer" in data

    def test_respond_endpoint_calm_guidance(self):
        """POST /companion/respond with calm state in guidance should use psychoeducational."""
        from fastapi.testclient import TestClient
        from api.routes.embodied_companion import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        # Use turn_count=1 and short duration so the state machine stays in guidance
        # (not enough turns/time to advance to reflection)
        resp = client.post("/companion/respond", json={
            "valence": 0.1,
            "arousal": 0.40,
            "stress_index": 0.15,
            "conversation_state": "guidance",
            "turn_count": 1,
            "session_duration": 10.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["therapeutic_stance"] == "psychoeducational"

    def test_session_endpoint(self):
        """POST /companion/session should track session data."""
        from fastapi.testclient import TestClient
        from api.routes.embodied_companion import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.post("/companion/session", json={
            "valence": -0.3,
            "arousal": 0.7,
            "stress_index": 0.6,
            "theme": "work_stress",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "work_stress" in data["themes"]
        assert data["turn_count"] == 1

    def test_session_tracks_multiple_calls(self):
        """Multiple session track calls should accumulate data."""
        from fastapi.testclient import TestClient
        from api.routes.embodied_companion import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        # First call
        resp1 = client.post("/companion/session", json={
            "valence": -0.3,
            "arousal": 0.7,
            "stress_index": 0.6,
            "theme": "work_stress",
        })
        mem = resp1.json()

        # Second call with updated memory
        resp2 = client.post("/companion/session", json={
            "valence": 0.2,
            "arousal": 0.5,
            "stress_index": 0.3,
            "session_memory": mem,
            "theme": "coping",
            "intervention": "breathing",
        })
        data = resp2.json()
        assert data["turn_count"] == 2
        assert "work_stress" in data["themes"]
        assert "coping" in data["themes"]
        assert "breathing" in data["interventions_tried"]


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_zeros_eeg(self):
        """All-zero EEG state should not crash."""
        eeg = EEGState()
        result = adapt_response_to_eeg(eeg)
        assert "tone" in result

    def test_extreme_values(self):
        """Extreme but valid values should not crash."""
        eeg = EEGState(
            valence=-1.0, arousal=1.0, stress_index=1.0,
            focus_index=1.0, anger_index=1.0, relaxation_index=0.0,
        )
        profile = compute_companion_profile(eeg)
        assert isinstance(profile, CompanionProfile)

    def test_empty_session_memory_in_profile(self):
        """Empty session memory should produce valid profile."""
        profile = compute_companion_profile(
            _calm_eeg(), session_memory={}
        )
        assert "turn_count" in profile.session_summary

    def test_invalid_stance_defaults(self):
        """Invalid stance value should default to supportive."""
        result = generate_response_template(
            _calm_eeg(), "check_in", "invalid_stance"
        )
        assert result["template"]["stance"] == "supportive"
