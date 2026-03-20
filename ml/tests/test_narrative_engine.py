"""Tests for the generative neural narratives engine (#415)."""

import pytest

from models.narrative_engine import (
    BrainStateCategory,
    EEGFeedback,
    EEGTrend,
    EmotionalArcPhase,
    FrameworkType,
    ImageryPreference,
    NarrativeEngine,
    NarrativeProfile,
    NarrativeSegment,
    StoryFramework,
    _FRAMEWORK_LIBRARY,
    _MINIMIZING_PHRASES,
)


@pytest.fixture
def engine():
    return NarrativeEngine()


# ---------------------------------------------------------------------------
# 1. EEGFeedback.classify_state
# ---------------------------------------------------------------------------

class TestEEGFeedbackClassification:
    def test_high_beta_yields_anxious(self):
        fb = EEGFeedback(beta_level=0.8)
        assert fb.classify_state() == BrainStateCategory.ANXIOUS

    def test_alpha_increasing_yields_relaxing(self):
        fb = EEGFeedback(alpha_trend=EEGTrend.INCREASING, beta_level=0.2)
        assert fb.classify_state() == BrainStateCategory.RELAXING

    def test_theta_increasing_yields_disengaging(self):
        fb = EEGFeedback(
            alpha_trend=EEGTrend.STABLE,
            theta_trend=EEGTrend.INCREASING,
            beta_level=0.3,
        )
        assert fb.classify_state() == BrainStateCategory.DISENGAGING

    def test_stable_low_beta_yields_engaged(self):
        fb = EEGFeedback(
            alpha_trend=EEGTrend.STABLE,
            beta_level=0.2,
            theta_trend=EEGTrend.STABLE,
        )
        assert fb.classify_state() == BrainStateCategory.ENGAGED

    def test_high_beta_overrides_alpha_increasing(self):
        """Anxiety check takes priority over relaxation."""
        fb = EEGFeedback(
            alpha_trend=EEGTrend.INCREASING,
            beta_level=0.9,
        )
        assert fb.classify_state() == BrainStateCategory.ANXIOUS


# ---------------------------------------------------------------------------
# 2. Safety filter
# ---------------------------------------------------------------------------

class TestSafetyFilter:
    def test_clean_text_passes(self, engine):
        result = engine.check_safety("The leaves float gently on the stream.")
        assert result["passed"] is True
        assert result["flagged_phrases"] == []

    def test_minimizing_language_blocked(self, engine):
        result = engine.check_safety("Just relax, it's not that bad.")
        assert result["passed"] is False
        assert "just relax" in result["flagged_phrases"]

    def test_default_trigger_word_blocked(self, engine):
        result = engine.check_safety("The story mentions violence in passing.")
        assert result["passed"] is False
        assert "violence" in result["flagged_phrases"]

    def test_user_specific_trigger_words(self, engine):
        engine.build_narrative_profile(
            user_id="user1",
            trigger_words=["storm", "thunder"],
        )
        result = engine.check_safety("Thunder rolled across the sky.", user_id="user1")
        assert result["passed"] is False
        assert "thunder" in result["flagged_phrases"]

    def test_user_trigger_words_do_not_affect_other_users(self, engine):
        engine.build_narrative_profile(
            user_id="userA",
            trigger_words=["fire"],
        )
        result = engine.check_safety("A fire crackled in the hearth.", user_id="userB")
        assert result["passed"] is True


# ---------------------------------------------------------------------------
# 3. Profile management
# ---------------------------------------------------------------------------

class TestProfileManagement:
    def test_build_profile_creates_new(self, engine):
        profile = engine.build_narrative_profile(
            user_id="new_user",
            preferred_imagery=["ocean"],
        )
        assert isinstance(profile, NarrativeProfile)
        assert profile.user_id == "new_user"
        assert ImageryPreference.OCEAN in profile.preferred_imagery

    def test_build_profile_updates_existing(self, engine):
        engine.build_narrative_profile(user_id="u1", preferred_imagery=["nature"])
        profile = engine.build_narrative_profile(
            user_id="u1",
            preferred_imagery=["space"],
            trigger_words=["darkness"],
        )
        assert ImageryPreference.SPACE in profile.preferred_imagery
        assert "darkness" in profile.trigger_words

    def test_invalid_imagery_ignored(self, engine):
        profile = engine.build_narrative_profile(
            user_id="u2",
            preferred_imagery=["nonexistent", "ocean"],
        )
        assert ImageryPreference.OCEAN in profile.preferred_imagery
        assert len(profile.preferred_imagery) == 1

    def test_get_profile_returns_none_for_unknown(self, engine):
        assert engine.get_profile("nobody") is None


# ---------------------------------------------------------------------------
# 4. Framework selection
# ---------------------------------------------------------------------------

class TestFrameworkSelection:
    def test_anxious_state_selects_act_or_imagery(self, engine):
        fw = engine.select_story_framework(BrainStateCategory.ANXIOUS)
        assert fw.framework_type in (
            FrameworkType.ACT_METAPHOR,
            FrameworkType.GUIDED_IMAGERY,
        )

    def test_disengaging_selects_externalization_or_bilateral(self, engine):
        fw = engine.select_story_framework(BrainStateCategory.DISENGAGING)
        assert fw.framework_type in (
            FrameworkType.NARRATIVE_EXTERNALIZATION,
            FrameworkType.EMDR_BILATERAL,
        )

    def test_exclude_ids_respected(self, engine):
        all_ids = list(_FRAMEWORK_LIBRARY.keys())
        # Exclude all but one
        exclude = all_ids[:-1]
        fw = engine.select_story_framework(
            BrainStateCategory.NEUTRAL,
            exclude_ids=exclude,
        )
        assert fw.id == all_ids[-1]

    def test_user_effective_frameworks_preferred(self, engine):
        engine.build_narrative_profile(user_id="pref_user")
        profile = engine.get_profile("pref_user")
        profile.effective_frameworks = ["guided_ocean"]

        fw = engine.select_story_framework(
            BrainStateCategory.RELAXING,
            user_id="pref_user",
        )
        assert fw.id == "guided_ocean"


# ---------------------------------------------------------------------------
# 5. EEG feedback adaptation
# ---------------------------------------------------------------------------

class TestAdaptToEEGFeedback:
    def test_relaxing_advances_phase(self, engine):
        fb = EEGFeedback(alpha_trend=EEGTrend.INCREASING, beta_level=0.2)
        result = engine.adapt_to_eeg_feedback(
            fb, EmotionalArcPhase.OPENING
        )
        assert result["action"] == "advance"
        assert result["next_phase"] == EmotionalArcPhase.BUILDING.value

    def test_anxious_inserts_grounding(self, engine):
        engine.build_narrative_profile(
            user_id="anx_user", preferred_imagery=["ocean"]
        )
        fb = EEGFeedback(beta_level=0.8)
        result = engine.adapt_to_eeg_feedback(
            fb, EmotionalArcPhase.BUILDING, user_id="anx_user"
        )
        assert result["action"] == "ground"
        assert result["imagery_insert"] is not None
        assert len(result["imagery_insert"]) > 0

    def test_disengaging_inserts_re_engage(self, engine):
        fb = EEGFeedback(
            alpha_trend=EEGTrend.STABLE,
            theta_trend=EEGTrend.INCREASING,
            beta_level=0.3,
        )
        result = engine.adapt_to_eeg_feedback(
            fb, EmotionalArcPhase.PEAK
        )
        assert result["action"] == "re_engage"
        assert result["imagery_insert"] is not None

    def test_relaxing_at_integration_completes(self, engine):
        fb = EEGFeedback(alpha_trend=EEGTrend.INCREASING, beta_level=0.2)
        result = engine.adapt_to_eeg_feedback(
            fb, EmotionalArcPhase.INTEGRATION
        )
        assert result["action"] == "complete"


# ---------------------------------------------------------------------------
# 6. Segment generation
# ---------------------------------------------------------------------------

class TestGenerateNarrativeSegment:
    def test_basic_generation(self, engine):
        segment = engine.generate_narrative_segment()
        assert isinstance(segment, NarrativeSegment)
        assert len(segment.text) > 0
        assert segment.safety_passed is True

    def test_explicit_framework_and_phase(self, engine):
        segment = engine.generate_narrative_segment(
            framework_id="act_leaves",
            arc_phase="peak",
        )
        assert segment.framework_id == "act_leaves"
        assert segment.arc_phase == EmotionalArcPhase.PEAK

    def test_anxious_eeg_appends_grounding(self, engine):
        engine.build_narrative_profile(
            user_id="u_anx", preferred_imagery=["nature"]
        )
        fb = EEGFeedback(beta_level=0.9)
        segment = engine.generate_narrative_segment(
            user_id="u_anx",
            arc_phase="opening",
            eeg_feedback=fb,
        )
        # Grounding imagery should be appended
        assert "supported" in segment.text.lower() or "root" in segment.text.lower() or "earth" in segment.text.lower() or "ground" in segment.text.lower() or "feet" in segment.text.lower()

    def test_narrative_to_dict_keys(self, engine):
        segment = engine.generate_narrative_segment()
        d = NarrativeEngine.narrative_to_dict(segment)
        expected_keys = {
            "id", "text", "framework_id", "arc_phase",
            "imagery_type", "brain_state", "safety_passed",
            "timestamp", "disclaimer",
        }
        assert set(d.keys()) == expected_keys

    def test_disclaimer_present(self, engine):
        segment = engine.generate_narrative_segment()
        d = NarrativeEngine.narrative_to_dict(segment)
        assert "research and educational" in d["disclaimer"]


# ---------------------------------------------------------------------------
# 7. Framework library integrity
# ---------------------------------------------------------------------------

class TestFrameworkLibrary:
    def test_all_frameworks_have_all_phases(self):
        for fw_id, fw in _FRAMEWORK_LIBRARY.items():
            for phase in EmotionalArcPhase:
                assert phase in fw.segments, (
                    f"Framework {fw_id} missing phase {phase.value}"
                )

    def test_list_frameworks_returns_all(self):
        frameworks = NarrativeEngine.list_frameworks()
        assert len(frameworks) == len(_FRAMEWORK_LIBRARY)
        for fw in frameworks:
            assert "id" in fw
            assert "name" in fw
            assert "framework_type" in fw
            assert "description" in fw


# ---------------------------------------------------------------------------
# 8. Profile scoring updates
# ---------------------------------------------------------------------------

class TestProfileScoring:
    def test_relaxing_state_increments_imagery_score(self, engine):
        engine.build_narrative_profile(
            user_id="score_user", preferred_imagery=["space"]
        )
        fb = EEGFeedback(alpha_trend=EEGTrend.INCREASING, beta_level=0.2)
        engine.adapt_to_eeg_feedback(
            fb, EmotionalArcPhase.OPENING, user_id="score_user"
        )
        profile = engine.get_profile("score_user")
        assert profile.imagery_scores.get("space", 0) > 0

    def test_anxious_state_does_not_increment_score(self, engine):
        engine.build_narrative_profile(
            user_id="no_score", preferred_imagery=["ocean"]
        )
        fb = EEGFeedback(beta_level=0.9)
        engine.adapt_to_eeg_feedback(
            fb, EmotionalArcPhase.OPENING, user_id="no_score"
        )
        profile = engine.get_profile("no_score")
        assert profile.imagery_scores.get("ocean", 0) == 0
