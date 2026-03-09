"""Tests for emotional granularity: dominance dimension and VAD-to-label mapping.

Covers:
- EMOTION_VAD_MAP structure and completeness (27 emotions)
- map_vad_to_granular_emotions correctness, sorting, top_k, similarity bounds
- Dominance calculation bounds and formula behavior
- VAD quadrant mapping (positive/low arousal -> content/serene, etc.)
- Edge cases (extreme values, zero, degenerate inputs)
- Integration with EmotionClassifier output (dominance + granular_emotions present)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.emotion_granularity import (
    EMOTION_VAD_MAP,
    map_vad_to_granular_emotions,
    _euclidean_distance,
    _distance_to_similarity,
)
from models.emotion_classifier import _compute_dominance


# -- EMOTION_VAD_MAP structure -------------------------------------------


class TestEmotionVADMap:
    def test_map_has_at_least_15_emotions(self):
        assert len(EMOTION_VAD_MAP) >= 15

    def test_map_has_27_emotions(self):
        assert len(EMOTION_VAD_MAP) == 27

    def test_all_values_are_3_tuples(self):
        for label, coord in EMOTION_VAD_MAP.items():
            assert len(coord) == 3, f"{label} has {len(coord)} values, expected 3"

    def test_valence_bounds(self):
        for label, (v, a, d) in EMOTION_VAD_MAP.items():
            assert -1.0 <= v <= 1.0, f"{label} valence {v} out of [-1, 1]"

    def test_arousal_bounds(self):
        for label, (v, a, d) in EMOTION_VAD_MAP.items():
            assert 0.0 <= a <= 1.0, f"{label} arousal {a} out of [0, 1]"

    def test_dominance_bounds(self):
        for label, (v, a, d) in EMOTION_VAD_MAP.items():
            assert 0.0 <= d <= 1.0, f"{label} dominance {d} out of [0, 1]"


# -- map_vad_to_granular_emotions ----------------------------------------


class TestMapVADToGranularEmotions:
    def test_returns_correct_top_k(self):
        result = map_vad_to_granular_emotions(0.5, 0.5, 0.5, top_k=3)
        assert len(result) == 3

    def test_returns_top_k_5(self):
        result = map_vad_to_granular_emotions(0.0, 0.3, 0.5, top_k=5)
        assert len(result) == 5

    def test_returns_top_k_1(self):
        result = map_vad_to_granular_emotions(0.0, 0.3, 0.5, top_k=1)
        assert len(result) == 1

    def test_top_k_zero_returns_one(self):
        """top_k < 1 is clamped to 1."""
        result = map_vad_to_granular_emotions(0.0, 0.3, 0.5, top_k=0)
        assert len(result) == 1

    def test_result_structure(self):
        result = map_vad_to_granular_emotions(0.5, 0.5, 0.5, top_k=1)
        entry = result[0]
        assert "emotion" in entry
        assert "similarity" in entry
        assert "distance" in entry

    def test_similarity_scores_between_0_and_1(self):
        result = map_vad_to_granular_emotions(0.3, 0.4, 0.5, top_k=5)
        for entry in result:
            assert 0.0 <= entry["similarity"] <= 1.0, (
                f"{entry['emotion']} similarity {entry['similarity']} out of [0, 1]"
            )

    def test_results_sorted_by_descending_similarity(self):
        result = map_vad_to_granular_emotions(0.0, 0.5, 0.5, top_k=5)
        similarities = [e["similarity"] for e in result]
        assert similarities == sorted(similarities, reverse=True)

    def test_exact_match_has_high_similarity(self):
        """Querying with exact coordinates of a known emotion should return similarity 1.0."""
        v, a, d = EMOTION_VAD_MAP["happy"]
        result = map_vad_to_granular_emotions(v, a, d, top_k=1)
        assert result[0]["emotion"] == "happy"
        assert result[0]["similarity"] == 1.0
        assert result[0]["distance"] == 0.0

    # -- Quadrant mapping tests ------------------------------------------

    def test_positive_valence_low_arousal_maps_to_content_serene_calm(self):
        """Positive valence + low arousal -> content, serene, calm."""
        result = map_vad_to_granular_emotions(0.60, 0.12, 0.52, top_k=3)
        labels = {e["emotion"] for e in result}
        expected = {"content", "serene", "calm", "grateful"}
        assert labels & expected, f"Expected overlap with {expected}, got {labels}"

    def test_negative_valence_high_arousal_maps_to_anxious_stressed_fearful(self):
        """Negative valence + high arousal -> anxious, stressed, fearful."""
        result = map_vad_to_granular_emotions(-0.55, 0.80, 0.20, top_k=3)
        labels = {e["emotion"] for e in result}
        expected = {"anxious", "stressed", "frustrated", "fearful"}
        assert labels & expected, f"Expected overlap with {expected}, got {labels}"

    def test_high_dominance_positive_valence_maps_to_proud(self):
        """High dominance + positive valence -> proud."""
        result = map_vad_to_granular_emotions(0.70, 0.50, 0.90, top_k=3)
        labels = {e["emotion"] for e in result}
        expected = {"proud", "happy", "elated"}
        assert labels & expected, f"Expected overlap with {expected}, got {labels}"

    def test_low_dominance_negative_valence_maps_to_fearful_overwhelmed(self):
        """Low dominance + negative valence + high arousal -> fearful."""
        result = map_vad_to_granular_emotions(-0.70, 0.85, 0.12, top_k=3)
        labels = {e["emotion"] for e in result}
        expected = {"fearful", "anxious"}
        assert labels & expected, f"Expected overlap with {expected}, got {labels}"

    def test_neutral_state_maps_to_neutral_pensive_bored(self):
        """Near-zero valence + low arousal -> neutral, pensive, bored."""
        result = map_vad_to_granular_emotions(0.05, 0.18, 0.45, top_k=3)
        labels = {e["emotion"] for e in result}
        expected = {"neutral", "pensive", "bored", "nostalgic"}
        assert labels & expected, f"Expected overlap with {expected}, got {labels}"

    def test_extreme_positive_valence(self):
        """Very high valence + high arousal -> elated, excited."""
        result = map_vad_to_granular_emotions(0.90, 0.85, 0.75, top_k=2)
        labels = {e["emotion"] for e in result}
        expected = {"elated", "excited"}
        assert labels & expected, f"Expected overlap with {expected}, got {labels}"

    def test_extreme_negative_valence_low_arousal(self):
        """Very negative valence + low arousal -> hopeless, sad."""
        result = map_vad_to_granular_emotions(-0.80, 0.10, 0.05, top_k=2)
        labels = {e["emotion"] for e in result}
        expected = {"hopeless", "sad"}
        assert labels & expected, f"Expected overlap with {expected}, got {labels}"


# -- Dominance computation -----------------------------------------------


class TestDominanceComputation:
    def test_dominance_bounds_normal(self):
        """Dominance should be in [0, 1] for typical band ratio values."""
        for ba in [0.1, 0.5, 1.0, 2.0, 5.0]:
            for tbr in [0.1, 0.5, 1.0, 2.0, 5.0]:
                d = _compute_dominance(ba, tbr)
                assert 0.0 <= d <= 1.0, f"Dominance {d} out of [0, 1] for ba={ba}, tbr={tbr}"

    def test_high_beta_alpha_low_theta_beta_gives_high_dominance(self):
        """High beta/alpha + low theta/beta ratio -> in control -> high dominance."""
        d = _compute_dominance(beta_alpha=3.0, theta_beta_ratio=0.1)
        assert d > 0.7, f"Expected high dominance, got {d}"

    def test_low_beta_alpha_high_theta_beta_gives_low_dominance(self):
        """Low beta/alpha + high theta/beta -> overwhelmed -> low dominance."""
        d = _compute_dominance(beta_alpha=0.1, theta_beta_ratio=3.0)
        assert d < 0.3, f"Expected low dominance, got {d}"

    def test_dominance_at_zero_inputs(self):
        """Dominance with zero ratios should be within bounds."""
        d = _compute_dominance(0.0, 0.0)
        assert 0.0 <= d <= 1.0

    def test_dominance_extreme_inputs(self):
        """Dominance with extreme ratios should still be clipped to [0, 1]."""
        d_high = _compute_dominance(100.0, 0.001)
        d_low = _compute_dominance(0.001, 100.0)
        assert 0.0 <= d_high <= 1.0
        assert 0.0 <= d_low <= 1.0

    def test_dominance_monotonic_with_beta_alpha(self):
        """Increasing beta/alpha should increase dominance (theta/beta fixed)."""
        d1 = _compute_dominance(0.5, 0.5)
        d2 = _compute_dominance(2.0, 0.5)
        assert d2 > d1, f"Expected d2 ({d2}) > d1 ({d1}) with higher beta/alpha"


# -- Euclidean distance / similarity helpers -----------------------------


class TestDistanceHelpers:
    def test_euclidean_distance_zero(self):
        assert _euclidean_distance((0, 0, 0), (0, 0, 0)) == 0.0

    def test_euclidean_distance_unit(self):
        d = _euclidean_distance((0, 0, 0), (1, 0, 0))
        assert abs(d - 1.0) < 1e-9

    def test_euclidean_distance_3d(self):
        d = _euclidean_distance((1, 1, 1), (0, 0, 0))
        assert abs(d - (3 ** 0.5)) < 1e-9

    def test_distance_to_similarity_zero_distance(self):
        assert _distance_to_similarity(0.0) == 1.0

    def test_distance_to_similarity_max_distance(self):
        assert _distance_to_similarity(2.5) == 0.0

    def test_distance_to_similarity_over_max(self):
        assert _distance_to_similarity(3.0) == 0.0

    def test_distance_to_similarity_midpoint(self):
        s = _distance_to_similarity(1.25)
        assert abs(s - 0.5) < 1e-9


# -- Integration: EmotionClassifier output -------------------------------


class TestEmotionClassifierIntegration:
    def test_predict_features_returns_dominance_and_granular(self):
        """The feature-based path should include dominance and granular_emotions."""
        from models.emotion_classifier import EmotionClassifier

        clf = EmotionClassifier()
        eeg = np.random.randn(4, 1024) * 20
        result = clf.predict(eeg, fs=256.0)
        assert "dominance" in result, "Missing 'dominance' in prediction output"
        assert "granular_emotions" in result, "Missing 'granular_emotions' in prediction output"
        assert 0.0 <= result["dominance"] <= 1.0
        assert isinstance(result["granular_emotions"], list)
        assert len(result["granular_emotions"]) == 3  # default top_k

    def test_granular_emotions_entries_have_required_keys(self):
        from models.emotion_classifier import EmotionClassifier

        clf = EmotionClassifier()
        eeg = np.random.randn(4, 1024) * 20
        result = clf.predict(eeg, fs=256.0)
        for entry in result["granular_emotions"]:
            assert "emotion" in entry
            assert "similarity" in entry
            assert "distance" in entry

    def test_valence_arousal_still_present(self):
        """Existing valence/arousal fields must not be removed."""
        from models.emotion_classifier import EmotionClassifier

        clf = EmotionClassifier()
        eeg = np.random.randn(4, 1024) * 20
        result = clf.predict(eeg, fs=256.0)
        assert "valence" in result
        assert "arousal" in result
        assert -1.0 <= result["valence"] <= 1.0
        assert 0.0 <= result["arousal"] <= 1.0
