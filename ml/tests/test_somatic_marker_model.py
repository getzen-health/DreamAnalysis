"""Tests for somatic marker mapping model.

Covers:
  - Chest tightness -> anxiety mapping
  - Warmth in hands -> positive valence
  - Stomach tension -> fear/anxiety
  - Full body activation -> anger/excitement (high arousal)
  - Somatic signature computation
  - Dissociation detection
  - Awareness scoring
  - Empty and minimal input handling
  - Personal learning overrides population priors
  - Multi-user isolation
  - Edge cases (invalid regions, out-of-range intensity)
"""

import pytest

from models.somatic_marker_model import (
    BODY_REGIONS,
    SENSATION_TYPES,
    SomaticMarkerModel,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _bm(region: str, sensation: str, intensity: float = 3) -> dict:
    """Shorthand for a single body-map entry."""
    return {"region": region, "sensation_type": sensation, "intensity": intensity}


@pytest.fixture
def model():
    return SomaticMarkerModel()


# ── Population Prior Mappings ────────────────────────────────────────


class TestChestTightnessAnxiety:
    """Chest tension should predict negative valence + high arousal (anxiety)."""

    def test_chest_tension_negative_valence(self, model):
        result = model.predict_emotion_from_soma([_bm("chest", "tension", 4)])
        assert result["valence"] < 0, "Chest tension should produce negative valence"

    def test_chest_tension_positive_arousal(self, model):
        result = model.predict_emotion_from_soma([_bm("chest", "tension", 4)])
        assert result["arousal"] > 0, "Chest tension should produce positive arousal"

    def test_chest_tension_predicts_anxiety(self, model):
        result = model.predict_emotion_from_soma([
            _bm("chest", "tension", 5),
            _bm("chest", "pressure", 4),
        ])
        assert result["predicted_emotion"] in ("anxiety", "alertness"), (
            f"Expected anxiety-related emotion, got {result['predicted_emotion']}"
        )


class TestWarmthPositiveValence:
    """Warmth in hands should predict positive valence."""

    def test_warm_hands_positive_valence(self, model):
        result = model.predict_emotion_from_soma([
            _bm("left_hand", "warmth", 4),
            _bm("right_hand", "warmth", 4),
        ])
        assert result["valence"] > 0, "Warm hands should produce positive valence"

    def test_warm_chest_positive_valence(self, model):
        result = model.predict_emotion_from_soma([_bm("chest", "warmth", 5)])
        assert result["valence"] > 0, "Warm chest should produce positive valence"


class TestStomachTensionFear:
    """Stomach tension should predict negative valence (fear/anxiety)."""

    def test_stomach_tension_negative_valence(self, model):
        result = model.predict_emotion_from_soma([_bm("stomach", "tension", 4)])
        assert result["valence"] < 0, "Stomach tension should produce negative valence"

    def test_stomach_nausea_negative_valence(self, model):
        result = model.predict_emotion_from_soma([_bm("stomach", "nausea", 4)])
        assert result["valence"] < 0, "Stomach nausea should produce negative valence"

    def test_stomach_tension_elevated_arousal(self, model):
        result = model.predict_emotion_from_soma([_bm("stomach", "tension", 5)])
        assert result["arousal"] > 0, "Stomach tension should raise arousal"


class TestFullBodyActivation:
    """Full body activation (tension in arms, legs, chest) -> anger/excitement."""

    def test_full_body_tension_high_arousal(self, model):
        body_map = [
            _bm("chest", "tension", 5),
            _bm("left_arm", "tension", 5),
            _bm("right_arm", "tension", 5),
            _bm("left_leg", "tension", 5),
            _bm("right_leg", "tension", 5),
            _bm("face", "tension", 5),
            _bm("neck", "tension", 5),
        ]
        result = model.predict_emotion_from_soma(body_map)
        assert result["arousal"] > 0.3, (
            f"Full body tension should produce high arousal, got {result['arousal']}"
        )

    def test_full_body_tension_negative_valence(self, model):
        body_map = [
            _bm("chest", "tension", 5),
            _bm("left_arm", "tension", 5),
            _bm("right_arm", "tension", 5),
            _bm("left_hand", "tension", 5),
            _bm("right_hand", "tension", 5),
        ]
        result = model.predict_emotion_from_soma(body_map)
        assert result["valence"] < 0, "Full body tension should produce negative valence"


# ── Somatic Signature ────────────────────────────────────────────────


class TestSomaticSignature:
    def test_no_data_no_signature(self, model):
        sig = model.compute_somatic_signature("nobody")
        assert sig["has_signature"] is False
        assert sig["n_pairings"] == 0

    def test_signature_after_learning(self, model):
        for _ in range(5):
            model.learn_somatic_pairing(
                [_bm("chest", "warmth", 4)],
                reported_valence=0.8,
                reported_arousal=0.2,
                user_id="alice",
            )
        sig = model.compute_somatic_signature("alice")
        assert sig["has_signature"] is True
        assert sig["n_pairings"] >= 5
        assert len(sig["top_positive"]) > 0

    def test_signature_strength_increases_with_data(self, model):
        for i in range(30):
            model.learn_somatic_pairing(
                [_bm("chest", "tension", 3), _bm("stomach", "nausea", 4)],
                reported_valence=-0.7,
                reported_arousal=0.5,
                user_id="bob",
            )
        sig = model.compute_somatic_signature("bob")
        assert sig["signature_strength"] > 0.5, "Many pairings should increase strength"


# ── Dissociation Detection ───────────────────────────────────────────


class TestDissociationDetection:
    def test_matching_body_and_report_no_dissociation(self, model):
        """Body shows tension (anxiety) and user reports anxiety -> no dissociation."""
        result = model.detect_somatic_dissociation(
            [_bm("chest", "tension", 5), _bm("stomach", "tension", 4)],
            reported_valence=-0.5,
            reported_arousal=0.5,
        )
        assert result["dissociation_detected"] is False

    def test_body_tense_report_calm_dissociation(self, model):
        """Body shows high tension but user reports calm -> dissociation."""
        result = model.detect_somatic_dissociation(
            [_bm("chest", "tension", 5), _bm("neck", "tension", 5),
             _bm("stomach", "tension", 5)],
            reported_valence=0.8,   # reports positive
            reported_arousal=-0.8,  # reports calm
        )
        assert result["dissociation_score"] > 0.2, (
            f"Expected dissociation, score={result['dissociation_score']}"
        )

    def test_dissociation_result_structure(self, model):
        result = model.detect_somatic_dissociation(
            [_bm("chest", "warmth", 3)],
            reported_valence=0.5,
            reported_arousal=0.3,
        )
        assert "dissociation_detected" in result
        assert "dissociation_score" in result
        assert "valence_mismatch" in result
        assert "arousal_mismatch" in result
        assert "somatic_valence" in result
        assert "somatic_arousal" in result

    def test_dissociation_does_not_store_history(self, model):
        """Dissociation check should not pollute prediction history."""
        model.detect_somatic_dissociation(
            [_bm("chest", "warmth", 3)],
            reported_valence=0.5,
            reported_arousal=0.3,
        )
        assert len(model.get_history()) == 0


# ── Awareness Scoring ────────────────────────────────────────────────


class TestAwarenessScoring:
    def test_no_data_minimal_awareness(self, model):
        score = model.compute_somatic_awareness_score("nobody")
        assert score["awareness_score"] == 0.0
        assert score["level"] == "minimal"

    def test_diverse_reports_increase_awareness(self, model):
        """Using many different regions and sensations -> higher awareness."""
        for region in BODY_REGIONS[:8]:
            for sensation in SENSATION_TYPES[:4]:
                model.learn_somatic_pairing(
                    [_bm(region, sensation, 3)],
                    reported_valence=0.0,
                    reported_arousal=0.0,
                    user_id="aware_user",
                )
                model.predict_emotion_from_soma(
                    [_bm(region, sensation, 3)],
                    user_id="aware_user",
                )
        score = model.compute_somatic_awareness_score("aware_user")
        assert score["awareness_score"] > 0.3, (
            f"Diverse usage should raise awareness, got {score['awareness_score']}"
        )
        assert score["distinct_regions"] >= 8
        assert score["distinct_sensations"] >= 4


# ── Empty and Minimal Input ──────────────────────────────────────────


class TestEmptyMinimalInput:
    def test_empty_body_map(self, model):
        result = model.predict_emotion_from_soma([])
        assert result["valence"] == 0.0
        assert result["arousal"] == 0.0
        assert result["predicted_emotion"] == "neutral"
        assert result["confidence"] == 0.0
        assert result["n_activations"] == 0

    def test_single_activation(self, model):
        result = model.predict_emotion_from_soma([_bm("chest", "warmth", 3)])
        assert result["n_activations"] == 1
        assert result["dominant_region"] == "chest"
        assert result["dominant_sensation"] == "warmth"

    def test_invalid_region_ignored(self, model):
        result = model.predict_emotion_from_soma([
            {"region": "nosuch", "sensation_type": "warmth", "intensity": 3},
        ])
        assert result["n_activations"] == 0
        assert result["predicted_emotion"] == "neutral"

    def test_invalid_sensation_ignored(self, model):
        result = model.predict_emotion_from_soma([
            {"region": "chest", "sensation_type": "nosuch", "intensity": 3},
        ])
        assert result["n_activations"] == 0

    def test_intensity_clamped_low(self, model):
        """Intensity below 1 should be clamped to 1."""
        result = model.predict_emotion_from_soma([_bm("chest", "warmth", -10)])
        assert result["n_activations"] == 1

    def test_intensity_clamped_high(self, model):
        """Intensity above 5 should be clamped to 5."""
        result = model.predict_emotion_from_soma([_bm("chest", "warmth", 100)])
        assert result["n_activations"] == 1


# ── Personal Learning ────────────────────────────────────────────────


class TestPersonalLearning:
    def test_learn_returns_status(self, model):
        result = model.learn_somatic_pairing(
            [_bm("chest", "warmth", 3)],
            reported_valence=0.5,
            reported_arousal=0.2,
        )
        assert result["status"] == "learned"
        assert result["n_pairings_learned"] == 1
        assert result["total_pairings"] >= 1

    def test_learned_priors_affect_predictions(self, model):
        """After learning that chest warmth = negative for this user,
        the prediction should shift toward negative valence."""
        # Population prior: chest warmth = positive
        baseline = model.predict_emotion_from_soma([_bm("chest", "warmth", 4)])
        baseline_valence = baseline["valence"]

        # Teach the model that for this user, chest warmth = very negative
        for _ in range(15):
            model.learn_somatic_pairing(
                [_bm("chest", "warmth", 4)],
                reported_valence=-0.9,
                reported_arousal=-0.5,
                user_id="learner",
            )

        learned = model.predict_emotion_from_soma(
            [_bm("chest", "warmth", 4)],
            user_id="learner",
        )
        assert learned["valence"] < baseline_valence, (
            "Personal learning should shift predictions"
        )

    def test_learn_empty_body_map(self, model):
        result = model.learn_somatic_pairing(
            [],
            reported_valence=0.5,
            reported_arousal=0.2,
        )
        assert result["status"] == "no_valid_activations"


# ── Multi-User Isolation ─────────────────────────────────────────────


class TestMultiUser:
    def test_independent_histories(self, model):
        model.predict_emotion_from_soma([_bm("chest", "warmth", 3)], user_id="u1")
        model.predict_emotion_from_soma([_bm("chest", "warmth", 3)], user_id="u2")
        model.predict_emotion_from_soma([_bm("chest", "warmth", 3)], user_id="u2")
        assert len(model.get_history("u1")) == 1
        assert len(model.get_history("u2")) == 2

    def test_independent_signatures(self, model):
        for _ in range(5):
            model.learn_somatic_pairing(
                [_bm("chest", "warmth", 4)],
                reported_valence=0.8,
                reported_arousal=0.3,
                user_id="alice",
            )
        sig_alice = model.compute_somatic_signature("alice")
        sig_bob = model.compute_somatic_signature("bob")
        assert sig_alice["has_signature"] is True
        assert sig_bob["has_signature"] is False

    def test_reset_one_user_keeps_other(self, model):
        model.predict_emotion_from_soma([_bm("chest", "warmth", 3)], user_id="u1")
        model.predict_emotion_from_soma([_bm("chest", "warmth", 3)], user_id="u2")
        model.reset("u1")
        assert len(model.get_history("u1")) == 0
        assert len(model.get_history("u2")) == 1


# ── Output Structure and Ranges ──────────────────────────────────────


class TestOutputStructure:
    def test_predict_keys(self, model):
        result = model.predict_emotion_from_soma([_bm("chest", "warmth", 3)])
        expected = {
            "valence", "arousal", "predicted_emotion", "confidence",
            "n_activations", "dominant_region", "dominant_sensation",
        }
        assert expected == set(result.keys())

    def test_valence_range(self, model):
        result = model.predict_emotion_from_soma([_bm("chest", "warmth", 5)])
        assert -1.0 <= result["valence"] <= 1.0

    def test_arousal_range(self, model):
        result = model.predict_emotion_from_soma([_bm("chest", "tension", 5)])
        assert -1.0 <= result["arousal"] <= 1.0

    def test_confidence_range(self, model):
        result = model.predict_emotion_from_soma([
            _bm("chest", "warmth", 5),
            _bm("face", "warmth", 5),
        ])
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predicted_emotion_is_string(self, model):
        result = model.predict_emotion_from_soma([_bm("chest", "warmth", 3)])
        assert isinstance(result["predicted_emotion"], str)


# ── Constants Validation ─────────────────────────────────────────────


class TestConstants:
    def test_body_regions_count(self):
        assert len(BODY_REGIONS) == 14

    def test_sensation_types_count(self):
        assert len(SENSATION_TYPES) == 12

    def test_no_duplicate_regions(self):
        assert len(BODY_REGIONS) == len(set(BODY_REGIONS))

    def test_no_duplicate_sensations(self):
        assert len(SENSATION_TYPES) == len(set(SENSATION_TYPES))
