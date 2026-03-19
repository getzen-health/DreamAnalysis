"""Tests for EmotionalGranularityEngine (#423).

Covers: scoring, vocabulary, exercises, taxonomy, trend, edge cases,
classification levels, ICC computation, and exercise difficulty modes.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.emotional_granularity_engine import (
    EMOTION_TAXONOMY,
    EmotionalGranularityEngine,
    _ALL_BASIC,
    _ALL_SECONDARY,
    _ALL_TERTIARY,
    _N_BASIC,
    _N_SECONDARY,
    _N_TERTIARY,
    _resolve_emotion_level,
    _resolve_to_basic,
    _get_sibling_emotions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_history(labels):
    """Create a simple emotion history from a list of label strings."""
    return [{"label": lbl} for lbl in labels]


def _make_rich_history(labels, intensities=None):
    """Create emotion history with optional intensities."""
    history = []
    for i, lbl in enumerate(labels):
        entry = {"label": lbl}
        if intensities and i < len(intensities):
            entry["intensity"] = intensities[i]
        history.append(entry)
    return history


# ===========================================================================
# TestTaxonomy
# ===========================================================================


class TestTaxonomy:
    """Tests for the emotion taxonomy constants and structure."""

    def test_taxonomy_has_six_basic_emotions(self):
        assert _N_BASIC == 6
        assert set(_ALL_BASIC) == {
            "joy", "sadness", "anger", "fear", "surprise", "disgust"
        }

    def test_taxonomy_has_at_least_25_secondary(self):
        assert _N_SECONDARY >= 25

    def test_taxonomy_has_at_least_50_tertiary(self):
        assert _N_TERTIARY >= 50

    def test_every_basic_has_secondaries(self):
        for basic in _ALL_BASIC:
            assert len(EMOTION_TAXONOMY[basic]) >= 1

    def test_every_secondary_has_tertiaries(self):
        for basic, secondaries in EMOTION_TAXONOMY.items():
            for sec, tertiaries in secondaries.items():
                assert len(tertiaries) >= 2, (
                    f"{basic} -> {sec} has fewer than 2 tertiaries"
                )

    def test_resolve_basic_level(self):
        assert _resolve_emotion_level("joy") == "basic"
        assert _resolve_emotion_level("anger") == "basic"

    def test_resolve_secondary_level(self):
        assert _resolve_emotion_level("frustration") == "secondary"
        assert _resolve_emotion_level("grief") == "secondary"

    def test_resolve_tertiary_level(self):
        assert _resolve_emotion_level("cheerfulness") == "tertiary"
        assert _resolve_emotion_level("irritation") == "tertiary"

    def test_resolve_unknown_level(self):
        assert _resolve_emotion_level("xyz_not_real") == "unknown"

    def test_resolve_to_basic_from_tertiary(self):
        assert _resolve_to_basic("irritation") == "anger"
        assert _resolve_to_basic("cheerfulness") == "joy"

    def test_resolve_to_basic_from_secondary(self):
        assert _resolve_to_basic("frustration") == "anger"
        assert _resolve_to_basic("grief") == "sadness"

    def test_resolve_to_basic_from_basic(self):
        assert _resolve_to_basic("joy") == "joy"

    def test_resolve_to_basic_unknown(self):
        assert _resolve_to_basic("made_up_emotion") is None

    def test_get_taxonomy_method(self):
        engine = EmotionalGranularityEngine()
        result = engine.get_taxonomy()
        assert "taxonomy" in result
        assert "counts" in result
        assert result["counts"]["basic"] == 6
        assert result["counts"]["total"] == _N_BASIC + _N_SECONDARY + _N_TERTIARY


# ===========================================================================
# TestGranularityScoring
# ===========================================================================


class TestGranularityScoring:
    """Tests for compute_granularity_score."""

    def test_empty_history_returns_zero(self):
        engine = EmotionalGranularityEngine()
        result = engine.compute_granularity_score([])
        assert result["granularity_score"] == 0.0
        assert result["n_reports"] == 0
        assert result["level"] == "insufficient_data"

    def test_single_basic_label_repeated(self):
        engine = EmotionalGranularityEngine()
        history = _make_history(["joy"] * 20)
        result = engine.compute_granularity_score(history)
        # One label used repeatedly = low granularity
        assert result["granularity_score"] < 0.3
        assert result["n_reports"] == 20

    def test_diverse_tertiary_labels_higher_than_basic(self):
        engine = EmotionalGranularityEngine()
        basic_history = _make_history(["joy", "sadness", "anger"] * 5)
        tertiary_history = _make_history([
            "cheerfulness", "contentment", "delight",
            "sorrow", "mourning", "heartbreak",
            "irritation", "exasperation", "annoyance",
            "nervousness", "worry", "apprehension",
            "amazement", "wonder", "awe",
        ])
        basic_score = engine.compute_granularity_score(basic_history)
        tertiary_score = engine.compute_granularity_score(tertiary_history)
        assert tertiary_score["granularity_score"] > basic_score["granularity_score"]

    def test_score_between_zero_and_one(self):
        engine = EmotionalGranularityEngine()
        history = _make_history([
            "cheerfulness", "sorrow", "irritation", "nervousness",
            "amazement", "repulsion", "contentment", "worry",
        ])
        result = engine.compute_granularity_score(history)
        assert 0.0 <= result["granularity_score"] <= 1.0

    def test_all_subscores_present(self):
        engine = EmotionalGranularityEngine()
        history = _make_history(["joy", "sadness", "anger", "fear"] * 3)
        result = engine.compute_granularity_score(history)
        assert "label_diversity" in result
        assert "taxonomy_depth" in result
        assert "distribution_evenness" in result
        assert "icc_score" in result

    def test_distribution_evenness_higher_for_balanced(self):
        engine = EmotionalGranularityEngine()
        # Balanced: each label appears equally
        balanced = _make_history(["joy", "sadness", "anger", "fear"] * 5)
        # Imbalanced: one label dominates
        imbalanced = _make_history(["joy"] * 17 + ["sadness"] * 1 + ["anger"] * 1 + ["fear"] * 1)
        bal_score = engine.compute_granularity_score(balanced)
        imbal_score = engine.compute_granularity_score(imbalanced)
        assert bal_score["distribution_evenness"] > imbal_score["distribution_evenness"]


# ===========================================================================
# TestClassifyLevel
# ===========================================================================


class TestClassifyLevel:
    """Tests for classify_granularity_level."""

    def test_very_high(self):
        assert EmotionalGranularityEngine.classify_granularity_level(0.85) == "very_high"

    def test_high(self):
        assert EmotionalGranularityEngine.classify_granularity_level(0.65) == "high"

    def test_moderate(self):
        assert EmotionalGranularityEngine.classify_granularity_level(0.45) == "moderate"

    def test_low(self):
        assert EmotionalGranularityEngine.classify_granularity_level(0.25) == "low"

    def test_very_low(self):
        assert EmotionalGranularityEngine.classify_granularity_level(0.1) == "very_low"

    def test_boundary_zero(self):
        assert EmotionalGranularityEngine.classify_granularity_level(0.0) == "very_low"

    def test_boundary_one(self):
        assert EmotionalGranularityEngine.classify_granularity_level(1.0) == "very_high"


# ===========================================================================
# TestVocabulary
# ===========================================================================


class TestVocabulary:
    """Tests for compute_emotion_vocabulary."""

    def test_empty_history_vocabulary(self):
        engine = EmotionalGranularityEngine()
        result = engine.compute_emotion_vocabulary([])
        assert result["total_reports"] == 0
        assert result["unique_labels"] == 0
        assert len(result["categories_missing"]) == 6

    def test_vocabulary_counts_levels(self):
        engine = EmotionalGranularityEngine()
        history = _make_history([
            "joy",           # basic
            "frustration",   # secondary
            "cheerfulness",  # tertiary
        ])
        result = engine.compute_emotion_vocabulary(history)
        assert result["basic_count"] == 1
        assert result["secondary_count"] == 1
        assert result["tertiary_count"] == 1

    def test_missing_categories_detected(self):
        engine = EmotionalGranularityEngine()
        history = _make_history(["joy", "cheerfulness", "contentment"])
        result = engine.compute_emotion_vocabulary(history)
        assert "joy" not in result["categories_missing"]
        assert "sadness" in result["categories_missing"]
        assert "anger" in result["categories_missing"]

    def test_most_used_sorted(self):
        engine = EmotionalGranularityEngine()
        history = _make_history(["joy"] * 10 + ["sadness"] * 5 + ["anger"] * 2)
        result = engine.compute_emotion_vocabulary(history)
        assert result["most_used"][0]["label"] == "joy"
        assert result["most_used"][0]["count"] == 10

    def test_depth_score_increases_with_tertiary(self):
        engine = EmotionalGranularityEngine()
        basic_only = _make_history(["joy", "sadness", "anger"] * 5)
        tertiary_rich = _make_history([
            "cheerfulness", "contentment", "delight",
            "sorrow", "mourning", "heartbreak",
            "irritation", "exasperation", "annoyance",
        ] * 2)
        basic_result = engine.compute_emotion_vocabulary(basic_only)
        tertiary_result = engine.compute_emotion_vocabulary(tertiary_rich)
        assert tertiary_result["depth_score"] > basic_result["depth_score"]

    def test_suggestions_present(self):
        engine = EmotionalGranularityEngine()
        history = _make_history(["joy"] * 15)
        result = engine.compute_emotion_vocabulary(history)
        assert len(result["suggestions"]) >= 1


# ===========================================================================
# TestExercises
# ===========================================================================


class TestExercises:
    """Tests for generate_differentiation_exercise."""

    def test_easy_exercise_structure(self):
        engine = EmotionalGranularityEngine()
        ex = engine.generate_differentiation_exercise([], difficulty="easy")
        assert ex["difficulty"] == "easy"
        assert "scenario" in ex
        assert "question" in ex
        assert "options" in ex
        assert "correct_answer" in ex
        assert "explanation" in ex
        assert len(ex["options"]) >= 2

    def test_medium_exercise_structure(self):
        engine = EmotionalGranularityEngine()
        history = _make_history(["anger", "frustration", "resentment"] * 3)
        ex = engine.generate_differentiation_exercise(history, difficulty="medium")
        assert ex["difficulty"] == "medium"
        assert "options" in ex
        assert len(ex["options"]) >= 2

    def test_hard_exercise_structure(self):
        engine = EmotionalGranularityEngine()
        history = _make_history(["irritation", "exasperation"] * 3)
        ex = engine.generate_differentiation_exercise(history, difficulty="hard")
        assert ex["difficulty"] == "hard"
        assert len(ex["options"]) >= 2

    def test_auto_difficulty_low_granularity(self):
        engine = EmotionalGranularityEngine()
        # Low granularity history = should get easy exercise
        history = _make_history(["joy"] * 5)
        ex = engine.generate_differentiation_exercise(history, difficulty="auto")
        assert ex["difficulty"] == "easy"

    def test_correct_answer_in_options(self):
        engine = EmotionalGranularityEngine()
        ex = engine.generate_differentiation_exercise([], difficulty="easy")
        assert ex["correct_answer"] in ex["options"]

    def test_exercise_correct_answer_in_options_medium(self):
        engine = EmotionalGranularityEngine()
        history = _make_history(["sadness", "grief", "melancholy"] * 3)
        ex = engine.generate_differentiation_exercise(history, difficulty="medium")
        assert ex["correct_answer"] in ex["options"]


# ===========================================================================
# TestTrend
# ===========================================================================


class TestTrend:
    """Tests for track_granularity_trend."""

    def test_initial_trend(self):
        engine = EmotionalGranularityEngine()
        history = _make_history(["joy", "sadness", "anger"] * 3)
        result = engine.track_granularity_trend("user1", history)
        assert result["trend_length"] == 1
        assert "current" in result
        assert "improvement" in result

    def test_trend_grows(self):
        engine = EmotionalGranularityEngine()
        history = _make_history(["joy", "sadness", "anger"] * 3)
        engine.track_granularity_trend("user1", history)
        engine.track_granularity_trend("user1", history)
        result = engine.track_granularity_trend("user1", history)
        assert result["trend_length"] == 3

    def test_improvement_detected(self):
        engine = EmotionalGranularityEngine()
        # First: low granularity
        low_history = _make_history(["joy"] * 10)
        engine.track_granularity_trend("user1", low_history)

        # Then: higher granularity
        high_history = _make_history([
            "cheerfulness", "contentment", "delight", "amusement",
            "sorrow", "mourning", "heartbreak", "anguish",
            "irritation", "exasperation", "annoyance", "agitation",
        ])
        result = engine.track_granularity_trend("user1", high_history)
        assert result["improvement"] > 0

    def test_trend_per_user(self):
        engine = EmotionalGranularityEngine()
        history = _make_history(["joy"] * 5)
        engine.track_granularity_trend("alice", history)
        engine.track_granularity_trend("bob", history)
        result_a = engine.track_granularity_trend("alice", history)
        result_b = engine.track_granularity_trend("bob", history)
        assert result_a["trend_length"] == 2
        assert result_b["trend_length"] == 2


# ===========================================================================
# TestProfileToDict
# ===========================================================================


class TestProfileToDict:
    """Tests for profile_to_dict."""

    def test_profile_structure(self):
        engine = EmotionalGranularityEngine()
        engine.add_emotion_report("user1", "joy", 0.8)
        engine.add_emotion_report("user1", "frustration", 0.6)
        profile = engine.profile_to_dict("user1")
        assert profile["user_id"] == "user1"
        assert profile["n_reports"] == 2
        assert "granularity" in profile
        assert "vocabulary" in profile

    def test_profile_empty_user(self):
        engine = EmotionalGranularityEngine()
        profile = engine.profile_to_dict("nobody")
        assert profile["n_reports"] == 0


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_unknown_labels_handled(self):
        engine = EmotionalGranularityEngine()
        history = _make_history(["totally_made_up", "not_real_emotion"])
        result = engine.compute_granularity_score(history)
        assert result["n_reports"] == 2
        # Should not crash; score can be computed

    def test_mixed_case_labels(self):
        engine = EmotionalGranularityEngine()
        history = _make_history(["Joy", "JOY", "joy"])
        result = engine.compute_granularity_score(history)
        # All should resolve to same label
        assert result["n_reports"] == 3

    def test_whitespace_in_labels(self):
        engine = EmotionalGranularityEngine()
        history = _make_history(["  joy  ", "joy", " joy"])
        vocab = engine.compute_emotion_vocabulary(history)
        assert vocab["unique_labels"] == 1

    def test_empty_label_entries(self):
        engine = EmotionalGranularityEngine()
        history = [{"label": ""}, {"label": None}, {}]
        result = engine.compute_granularity_score(history)
        assert result["n_reports"] == 0

    def test_large_history_does_not_crash(self):
        engine = EmotionalGranularityEngine()
        labels = _ALL_TERTIARY * 20  # ~1000+ entries
        history = _make_history(labels)
        result = engine.compute_granularity_score(history)
        assert result["granularity_score"] > 0

    def test_add_emotion_report(self):
        engine = EmotionalGranularityEngine()
        result = engine.add_emotion_report("user1", "irritation", 0.7, "work meeting")
        assert result["status"] == "recorded"
        assert result["level"] == "tertiary"
        assert result["basic_category"] == "anger"

    def test_sibling_emotions(self):
        siblings = _get_sibling_emotions("irritation")
        assert "exasperation" in siblings
        assert "annoyance" in siblings
        assert "irritation" not in siblings

    def test_sibling_emotions_secondary(self):
        siblings = _get_sibling_emotions("frustration")
        # Frustration is secondary under anger; siblings are other secondaries
        assert "resentment" in siblings or "rage" in siblings
        assert "frustration" not in siblings

    def test_sibling_emotions_unknown(self):
        siblings = _get_sibling_emotions("nonexistent")
        assert siblings == []

    def test_icc_proxy_basic_only(self):
        """ICC should be 0 if user only uses basic labels."""
        engine = EmotionalGranularityEngine()
        history = _make_history(["joy", "sadness", "anger"] * 5)
        result = engine.compute_granularity_score(history)
        assert result["icc_score"] == 0.0

    def test_icc_proxy_with_sublabels(self):
        """ICC should be > 0 if user uses sublabels within a category."""
        engine = EmotionalGranularityEngine()
        history = _make_history([
            "frustration", "resentment", "rage", "contempt",
            "frustration", "resentment",
        ])
        result = engine.compute_granularity_score(history)
        assert result["icc_score"] > 0.0
