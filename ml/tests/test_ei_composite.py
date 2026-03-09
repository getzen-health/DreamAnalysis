"""Tests for EmotionalIntelligenceIndex (ei_composite.py).

Comprehensive test suite covering:
- Baseline setting
- EIQ computation from raw EEG
- EIQ computation from component scores
- Component update and mapping
- Dimension score ranges and computation
- Grading thresholds (A/B/C/D/F)
- Strengths and growth areas
- Session stats (empty, after data, trend)
- History tracking
- Reset behavior
- Multi-user independence
- Edge cases (constant, 1D, short, zero, NaN signals)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ei_composite import EmotionalIntelligenceIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eeg(
    n_channels: int = 4,
    duration_sec: float = 4.0,
    fs: float = 256.0,
    alpha_amp: float = 15.0,
    beta_amp: float = 5.0,
    theta_amp: float = 8.0,
    noise_amp: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic EEG with controllable band amplitudes."""
    rng = np.random.RandomState(seed)
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs
    eeg = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
        beta = beta_amp * np.sin(2 * np.pi * 20 * t + rng.uniform(0, 2 * np.pi))
        theta = theta_amp * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2 * np.pi))
        noise = noise_amp * rng.randn(n_samples)
        eeg[ch] = alpha + beta + theta + noise
    return eeg


def _make_asymmetric_eeg(
    duration_sec: float = 4.0,
    fs: float = 256.0,
    left_alpha: float = 15.0,
    right_alpha: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate 4-ch EEG with different alpha amplitudes on AF7 vs AF8."""
    rng = np.random.RandomState(seed)
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs
    eeg = np.zeros((4, n_samples))
    for ch in range(4):
        if ch == 1:
            alpha_amp = left_alpha
        elif ch == 2:
            alpha_amp = right_alpha
        else:
            alpha_amp = 10.0
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
        beta = 5.0 * np.sin(2 * np.pi * 20 * t + rng.uniform(0, 2 * np.pi))
        theta = 8.0 * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2 * np.pi))
        noise = 2.0 * rng.randn(n_samples)
        eeg[ch] = alpha + beta + theta + noise
    return eeg


# ===========================================================================
# TestBaseline
# ===========================================================================


class TestBaseline:
    """Tests for set_baseline method."""

    def test_set_baseline_returns_dict(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        result = idx.set_baseline(eeg)
        assert isinstance(result, dict)

    def test_set_baseline_keys(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        result = idx.set_baseline(eeg)
        assert result["baseline_set"] is True
        assert result["n_channels"] == 4
        assert result["n_samples"] == 1024

    def test_set_baseline_marks_has_baseline(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        idx.set_baseline(eeg)
        result = idx.compute_eiq(eeg)
        assert result["has_baseline"] is True

    def test_no_baseline_has_baseline_false(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        result = idx.compute_eiq(eeg)
        assert result["has_baseline"] is False

    def test_set_baseline_1d_input(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = np.random.randn(1024) * 20
        result = idx.set_baseline(eeg)
        assert result["baseline_set"] is True
        assert result["n_channels"] == 1

    def test_set_baseline_with_custom_fs(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42, fs=128.0)
        result = idx.set_baseline(eeg, fs=128.0)
        assert result["baseline_set"] is True


# ===========================================================================
# TestComputeEIQ
# ===========================================================================


class TestComputeEIQ:
    """Tests for compute_eiq from raw EEG."""

    def test_returns_dict(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        result = idx.compute_eiq(eeg)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        result = idx.compute_eiq(eeg)
        required = ["eiq_score", "eiq_grade", "dimensions", "strengths", "growth_areas", "has_baseline"]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_eiq_score_0_to_100(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        result = idx.compute_eiq(eeg)
        assert 0.0 <= result["eiq_score"] <= 100.0

    def test_eiq_score_is_float(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        result = idx.compute_eiq(eeg)
        assert isinstance(result["eiq_score"], float)

    def test_eiq_grade_is_string(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        result = idx.compute_eiq(eeg)
        assert isinstance(result["eiq_grade"], str)
        assert result["eiq_grade"] in {"A", "B", "C", "D", "F"}

    def test_dimensions_has_5_keys(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        result = idx.compute_eiq(eeg)
        dims = result["dimensions"]
        expected_dims = ["self_perception", "self_expression", "interpersonal",
                         "decision_making", "stress_management"]
        for d in expected_dims:
            assert d in dims, f"Missing dimension: {d}"

    def test_dimensions_all_0_to_100(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        result = idx.compute_eiq(eeg)
        for d, v in result["dimensions"].items():
            assert 0.0 <= v <= 100.0, f"Dimension {d} out of range: {v}"

    def test_strengths_is_list(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        result = idx.compute_eiq(eeg)
        assert isinstance(result["strengths"], list)

    def test_growth_areas_is_list(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        result = idx.compute_eiq(eeg)
        assert isinstance(result["growth_areas"], list)

    def test_no_signals_no_components_returns_none(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        result = idx.compute_eiq()
        assert result is None

    def test_compute_with_custom_fs(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42, fs=128.0, duration_sec=4.0)
        result = idx.compute_eiq(eeg, fs=128.0)
        assert result is not None
        assert 0.0 <= result["eiq_score"] <= 100.0

    def test_eiq_is_mean_of_dimensions(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        result = idx.compute_eiq(eeg)
        dims = result["dimensions"]
        expected_mean = np.mean(list(dims.values()))
        assert abs(result["eiq_score"] - round(expected_mean, 2)) < 0.02


# ===========================================================================
# TestFromComponentScores
# ===========================================================================


class TestFromComponentScores:
    """Tests for compute_eiq from pre-computed component scores."""

    def test_component_scores_only(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        scores = {
            "granularity": 0.8,
            "flexibility": 0.7,
            "synchrony": 0.6,
            "emotional_memory": 0.5,
            "reactivity_regulation": 0.9,
        }
        result = idx.compute_eiq(component_scores=scores)
        assert result is not None
        assert 0.0 <= result["eiq_score"] <= 100.0

    def test_granularity_maps_to_self_perception(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        scores = {"granularity": 0.9}
        result = idx.compute_eiq(component_scores=scores)
        # self_perception should be high (granularity -> 90)
        assert result["dimensions"]["self_perception"] > 70.0

    def test_alexithymia_inverts_for_self_perception(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        # High alexithymia = low self-perception
        high_alex = idx.compute_eiq(component_scores={"alexithymia": 0.9})
        low_alex = idx.compute_eiq(component_scores={"alexithymia": 0.1})
        assert low_alex["dimensions"]["self_perception"] > high_alex["dimensions"]["self_perception"]

    def test_flexibility_maps_to_self_expression(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        scores = {"flexibility": 0.9}
        result = idx.compute_eiq(component_scores=scores)
        assert result["dimensions"]["self_expression"] > 70.0

    def test_synchrony_maps_to_interpersonal(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        scores = {"synchrony": 0.9}
        result = idx.compute_eiq(component_scores=scores)
        assert result["dimensions"]["interpersonal"] > 70.0

    def test_emotional_memory_maps_to_decision_making(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        scores = {"emotional_memory": 0.9}
        result = idx.compute_eiq(component_scores=scores)
        assert result["dimensions"]["decision_making"] > 70.0

    def test_reactivity_regulation_maps_to_stress_management(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        scores = {"reactivity_regulation": 0.9}
        result = idx.compute_eiq(component_scores=scores)
        assert result["dimensions"]["stress_management"] > 70.0

    def test_mood_stability_maps_to_stress_management(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        scores = {"mood_stability": 0.9}
        result = idx.compute_eiq(component_scores=scores)
        assert result["dimensions"]["stress_management"] > 70.0

    def test_all_high_components_high_eiq(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        scores = {
            "granularity": 0.95,
            "flexibility": 0.95,
            "synchrony": 0.95,
            "emotional_memory": 0.95,
            "reactivity_regulation": 0.95,
            "mood_stability": 0.95,
            "alexithymia": 0.05,
            "affect_labeling": 0.95,
            "interoception": 0.95,
        }
        result = idx.compute_eiq(component_scores=scores)
        assert result["eiq_score"] > 70.0

    def test_all_low_components_low_eiq(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        scores = {
            "granularity": 0.05,
            "flexibility": 0.05,
            "synchrony": 0.05,
            "emotional_memory": 0.05,
            "reactivity_regulation": 0.05,
            "mood_stability": 0.05,
            "alexithymia": 0.95,
            "affect_labeling": 0.05,
            "interoception": 0.05,
        }
        result = idx.compute_eiq(component_scores=scores)
        assert result["eiq_score"] < 40.0

    def test_both_signals_and_components(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        scores = {"granularity": 0.8, "synchrony": 0.7}
        result = idx.compute_eiq(eeg, component_scores=scores)
        assert result is not None
        assert 0.0 <= result["eiq_score"] <= 100.0

    def test_unmapped_dimensions_get_default(self):
        """Dimensions with no component mapping should get 50.0 (neutral)."""
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        # Only provide granularity -> only self_perception mapped
        scores = {"granularity": 0.5}
        result = idx.compute_eiq(component_scores=scores)
        # interpersonal, decision_making, stress_management should be 50.0
        assert result["dimensions"]["interpersonal"] == 50.0
        assert result["dimensions"]["decision_making"] == 50.0
        assert result["dimensions"]["stress_management"] == 50.0


# ===========================================================================
# TestUpdateComponent
# ===========================================================================


class TestUpdateComponent:
    """Tests for update_component method."""

    def test_update_and_use(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        idx.update_component("granularity", 0.85)
        # Now compute_eiq with no signals should use stored component
        result = idx.compute_eiq()
        assert result is not None
        assert result["dimensions"]["self_perception"] > 60.0

    def test_update_multiple_components(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        idx.update_component("granularity", 0.9)
        idx.update_component("synchrony", 0.8)
        idx.update_component("emotional_memory", 0.7)
        result = idx.compute_eiq()
        assert result is not None

    def test_update_overrides_previous(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        idx.update_component("granularity", 0.2)
        r1 = idx.compute_eiq()
        idx.update_component("granularity", 0.9)
        r2 = idx.compute_eiq()
        assert r2["dimensions"]["self_perception"] > r1["dimensions"]["self_perception"]

    def test_update_clamps_to_0_1(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        idx.update_component("granularity", 1.5)
        result = idx.compute_eiq()
        # Should not exceed 100
        assert result["dimensions"]["self_perception"] <= 100.0

    def test_update_unknown_component_ignored(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        # Should not raise
        idx.update_component("nonexistent", 0.5)
        result = idx.compute_eiq()
        assert result is None  # No valid stored components

    def test_update_per_user(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        idx.update_component("granularity", 0.9, user_id="alice")
        idx.update_component("granularity", 0.1, user_id="bob")
        r_alice = idx.compute_eiq(user_id="alice")
        r_bob = idx.compute_eiq(user_id="bob")
        assert r_alice["dimensions"]["self_perception"] > r_bob["dimensions"]["self_perception"]

    def test_update_blends_with_eeg(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        idx.update_component("granularity", 0.95)
        result = idx.compute_eiq(eeg)
        # Should use blended EEG + component scores
        assert result is not None
        assert 0.0 <= result["eiq_score"] <= 100.0


# ===========================================================================
# TestDimensions
# ===========================================================================


class TestDimensions:
    """Tests for individual dimension computation and ranges."""

    def test_all_dimensions_present(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        result = idx.compute_eiq(eeg)
        dims = result["dimensions"]
        expected = {"self_perception", "self_expression", "interpersonal",
                    "decision_making", "stress_management"}
        assert set(dims.keys()) == expected

    def test_dimension_values_are_floats(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        result = idx.compute_eiq(eeg)
        for d, v in result["dimensions"].items():
            assert isinstance(v, float), f"Dimension {d} is not float: {type(v)}"

    def test_each_dimension_0_to_100(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        # Run several different signals to test range robustness
        for seed in [42, 100, 200, 300]:
            eeg = _make_eeg(seed=seed)
            result = idx.compute_eiq(eeg)
            for d, v in result["dimensions"].items():
                assert 0.0 <= v <= 100.0, f"Dimension {d} = {v} out of [0,100]"

    def test_different_signals_different_dimensions(self):
        """Distinct EEG patterns should produce at least slightly different scores."""
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg1 = _make_eeg(seed=42, alpha_amp=25.0, beta_amp=2.0)
        eeg2 = _make_eeg(seed=42, alpha_amp=2.0, beta_amp=25.0)
        r1 = idx.compute_eiq(eeg1)
        r2 = idx.compute_eiq(eeg2)
        # At least one dimension should differ
        diffs = [abs(r1["dimensions"][d] - r2["dimensions"][d]) for d in r1["dimensions"]]
        assert max(diffs) > 1.0


# ===========================================================================
# TestGrading
# ===========================================================================


class TestGrading:
    """Tests for A/B/C/D/F grade thresholds."""

    def test_grade_a(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        # Use all high component scores to get A
        scores = {
            "granularity": 0.95, "flexibility": 0.95, "synchrony": 0.95,
            "emotional_memory": 0.95, "reactivity_regulation": 0.95,
            "mood_stability": 0.95, "alexithymia": 0.05,
            "affect_labeling": 0.95, "interoception": 0.95,
        }
        result = idx.compute_eiq(component_scores=scores)
        assert result["eiq_grade"] == "A"
        assert result["eiq_score"] >= 80.0

    def test_grade_b(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        scores = {
            "granularity": 0.72, "flexibility": 0.72, "synchrony": 0.72,
            "emotional_memory": 0.72, "reactivity_regulation": 0.72,
        }
        result = idx.compute_eiq(component_scores=scores)
        # With 72/100 on mapped dims and 50 on unmapped, expect ~B range
        assert result["eiq_grade"] in {"B", "C"}

    def test_grade_f(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        scores = {
            "granularity": 0.05, "flexibility": 0.05, "synchrony": 0.05,
            "emotional_memory": 0.05, "reactivity_regulation": 0.05,
            "mood_stability": 0.05, "alexithymia": 0.95,
            "affect_labeling": 0.05, "interoception": 0.05,
        }
        result = idx.compute_eiq(component_scores=scores)
        assert result["eiq_grade"] in {"F", "D"}
        assert result["eiq_score"] < 50.0

    def test_grade_thresholds_boundary(self):
        """Verify the grade method directly at boundary values."""
        idx = EmotionalIntelligenceIndex(fs=256.0)
        assert idx._grade(80.0) == "A"
        assert idx._grade(79.99) == "B"
        assert idx._grade(65.0) == "B"
        assert idx._grade(64.99) == "C"
        assert idx._grade(50.0) == "C"
        assert idx._grade(49.99) == "D"
        assert idx._grade(35.0) == "D"
        assert idx._grade(34.99) == "F"
        assert idx._grade(0.0) == "F"
        assert idx._grade(100.0) == "A"


# ===========================================================================
# TestStrengthsAndGrowthAreas
# ===========================================================================


class TestStrengthsAndGrowthAreas:
    """Tests for strengths (>70) and growth_areas (<40) identification."""

    def test_high_dimension_in_strengths(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        # All components very high -> all dimensions should be strengths
        scores = {
            "granularity": 0.95, "flexibility": 0.95, "synchrony": 0.95,
            "emotional_memory": 0.95, "reactivity_regulation": 0.95,
            "mood_stability": 0.95, "alexithymia": 0.05,
            "affect_labeling": 0.95, "interoception": 0.95,
        }
        result = idx.compute_eiq(component_scores=scores)
        assert len(result["strengths"]) > 0

    def test_low_dimension_in_growth_areas(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        scores = {
            "granularity": 0.05, "flexibility": 0.05, "synchrony": 0.05,
            "emotional_memory": 0.05, "reactivity_regulation": 0.05,
            "mood_stability": 0.05, "alexithymia": 0.95,
            "affect_labeling": 0.05, "interoception": 0.05,
        }
        result = idx.compute_eiq(component_scores=scores)
        assert len(result["growth_areas"]) > 0

    def test_strengths_are_valid_dimension_names(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        result = idx.compute_eiq(eeg)
        valid = {"self_perception", "self_expression", "interpersonal",
                 "decision_making", "stress_management"}
        for s in result["strengths"]:
            assert s in valid

    def test_growth_areas_are_valid_dimension_names(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        result = idx.compute_eiq(eeg)
        valid = {"self_perception", "self_expression", "interpersonal",
                 "decision_making", "stress_management"}
        for ga in result["growth_areas"]:
            assert ga in valid

    def test_strength_dimension_exceeds_70(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        scores = {"granularity": 0.95, "alexithymia": 0.05, "interoception": 0.95}
        result = idx.compute_eiq(component_scores=scores)
        for s in result["strengths"]:
            assert result["dimensions"][s] > 70.0

    def test_growth_area_below_40(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        scores = {"granularity": 0.05, "alexithymia": 0.95, "interoception": 0.05}
        result = idx.compute_eiq(component_scores=scores)
        for ga in result["growth_areas"]:
            assert result["dimensions"][ga] < 40.0


# ===========================================================================
# TestSessionStats
# ===========================================================================


class TestSessionStats:
    """Tests for get_session_stats method."""

    def test_empty_stats(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        stats = idx.get_session_stats()
        assert stats["n_assessments"] == 0
        assert stats["mean_eiq"] is None
        assert stats["trend"] is None

    def test_stats_after_one_assessment(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        idx.compute_eiq(eeg)
        stats = idx.get_session_stats()
        assert stats["n_assessments"] == 1
        assert stats["mean_eiq"] is not None
        assert isinstance(stats["mean_eiq"], float)

    def test_stats_after_multiple_assessments(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        for seed in range(5):
            eeg = _make_eeg(seed=seed)
            idx.compute_eiq(eeg)
        stats = idx.get_session_stats()
        assert stats["n_assessments"] == 5

    def test_trend_with_enough_data(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        for seed in range(9):
            eeg = _make_eeg(seed=seed)
            idx.compute_eiq(eeg)
        stats = idx.get_session_stats()
        assert stats["trend"] in {"improving", "declining", "stable"}

    def test_trend_none_with_few_data(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        idx.compute_eiq(eeg)
        stats = idx.get_session_stats()
        assert stats["trend"] is None

    def test_mean_eiq_is_average(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        scores_list = []
        for seed in range(3):
            eeg = _make_eeg(seed=seed)
            result = idx.compute_eiq(eeg)
            scores_list.append(result["eiq_score"])
        stats = idx.get_session_stats()
        expected_mean = round(float(np.mean(scores_list)), 2)
        assert abs(stats["mean_eiq"] - expected_mean) < 0.02


# ===========================================================================
# TestHistory
# ===========================================================================


class TestHistory:
    """Tests for get_history method."""

    def test_empty_history(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        history = idx.get_history()
        assert history == []

    def test_history_grows(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        for seed in range(3):
            eeg = _make_eeg(seed=seed)
            idx.compute_eiq(eeg)
        history = idx.get_history()
        assert len(history) == 3

    def test_history_last_n(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        for seed in range(5):
            eeg = _make_eeg(seed=seed)
            idx.compute_eiq(eeg)
        history = idx.get_history(last_n=2)
        assert len(history) == 2

    def test_history_last_n_larger_than_history(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        idx.compute_eiq(eeg)
        history = idx.get_history(last_n=100)
        assert len(history) == 1

    def test_history_entries_are_dicts(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        idx.compute_eiq(eeg)
        history = idx.get_history()
        assert all(isinstance(h, dict) for h in history)

    def test_history_entries_have_eiq_score(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        idx.compute_eiq(eeg)
        history = idx.get_history()
        assert "eiq_score" in history[0]

    def test_history_cap_enforced(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        scores = {"granularity": 0.5}
        for _ in range(510):
            idx.compute_eiq(component_scores=scores)
        history = idx.get_history()
        assert len(history) == 500


# ===========================================================================
# TestReset
# ===========================================================================


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_history(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        idx.compute_eiq(eeg)
        idx.reset()
        assert idx.get_history() == []

    def test_reset_clears_baseline(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        idx.set_baseline(eeg)
        idx.reset()
        result = idx.compute_eiq(eeg)
        assert result["has_baseline"] is False

    def test_reset_clears_components(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        idx.update_component("granularity", 0.9)
        idx.reset()
        result = idx.compute_eiq()
        assert result is None

    def test_reset_clears_session_stats(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        idx.compute_eiq(eeg)
        idx.reset()
        stats = idx.get_session_stats()
        assert stats["n_assessments"] == 0

    def test_reset_allows_reuse(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        idx.compute_eiq(eeg)
        idx.reset()
        result = idx.compute_eiq(eeg)
        assert result is not None


# ===========================================================================
# TestMultiUser
# ===========================================================================


class TestMultiUser:
    """Tests for multi-user independence."""

    def test_users_are_independent(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        idx.compute_eiq(eeg, user_id="alice")
        stats_alice = idx.get_session_stats("alice")
        stats_bob = idx.get_session_stats("bob")
        assert stats_alice["n_assessments"] == 1
        assert stats_bob["n_assessments"] == 0

    def test_history_is_per_user(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        idx.compute_eiq(eeg, user_id="alice")
        idx.compute_eiq(eeg, user_id="alice")
        idx.compute_eiq(eeg, user_id="bob")
        assert len(idx.get_history("alice")) == 2
        assert len(idx.get_history("bob")) == 1

    def test_baseline_is_per_user(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        idx.set_baseline(eeg, user_id="alice")
        result_alice = idx.compute_eiq(eeg, user_id="alice")
        result_bob = idx.compute_eiq(eeg, user_id="bob")
        assert result_alice["has_baseline"] is True
        assert result_bob["has_baseline"] is False

    def test_components_are_per_user(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        idx.update_component("granularity", 0.9, user_id="alice")
        r_alice = idx.compute_eiq(user_id="alice")
        r_bob = idx.compute_eiq(user_id="bob")
        assert r_alice is not None
        assert r_bob is None

    def test_reset_one_user_preserves_other(self):
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        idx.compute_eiq(eeg, user_id="alice")
        idx.compute_eiq(eeg, user_id="bob")
        idx.reset("alice")
        assert idx.get_session_stats("alice")["n_assessments"] == 0
        assert idx.get_session_stats("bob")["n_assessments"] == 1


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_constant_signal(self):
        """Constant (DC) signal should not crash."""
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = np.ones((4, 1024)) * 100.0
        result = idx.compute_eiq(eeg)
        assert result is not None
        assert 0.0 <= result["eiq_score"] <= 100.0

    def test_1d_input(self):
        """1D input should be reshaped to (1, n_samples)."""
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = np.random.randn(1024) * 20
        result = idx.compute_eiq(eeg)
        assert result is not None
        assert 0.0 <= result["eiq_score"] <= 100.0

    def test_short_signal(self):
        """Short signal (< 64 samples) should not crash."""
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = np.random.randn(4, 32) * 20
        result = idx.compute_eiq(eeg)
        assert result is not None

    def test_very_short_signal(self):
        """Very short signal (8 samples) should not crash."""
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = np.random.randn(4, 8) * 20
        result = idx.compute_eiq(eeg)
        assert result is not None

    def test_single_sample_signal(self):
        """Single sample signal should not crash."""
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = np.random.randn(4, 1) * 20
        result = idx.compute_eiq(eeg)
        assert result is not None

    def test_zero_signal(self):
        """All-zero signal should produce valid output."""
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = np.zeros((4, 1024))
        result = idx.compute_eiq(eeg)
        assert result is not None
        assert 0.0 <= result["eiq_score"] <= 100.0

    def test_nan_in_signal(self):
        """NaN values in EEG should be handled gracefully."""
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = np.random.randn(4, 512) * 20
        eeg[0, 100] = np.nan
        eeg[2, 200] = np.nan
        result = idx.compute_eiq(eeg)
        assert result is not None
        assert np.isfinite(result["eiq_score"])

    def test_inf_in_signal(self):
        """Inf values in EEG should be handled gracefully."""
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = np.random.randn(4, 512) * 20
        eeg[1, 50] = np.inf
        eeg[3, 100] = -np.inf
        result = idx.compute_eiq(eeg)
        assert result is not None
        assert np.isfinite(result["eiq_score"])

    def test_two_channel_signal(self):
        """2-channel signal should work (limited FAA)."""
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = np.random.randn(2, 1024) * 20
        result = idx.compute_eiq(eeg)
        assert result is not None
        assert 0.0 <= result["eiq_score"] <= 100.0

    def test_single_channel_signal(self):
        """1-channel 2D signal should work."""
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = np.random.randn(1, 1024) * 20
        result = idx.compute_eiq(eeg)
        assert result is not None

    def test_large_amplitude_signal(self):
        """Very large amplitude should not cause overflow."""
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = np.random.randn(4, 512) * 100000
        result = idx.compute_eiq(eeg)
        assert result is not None
        assert np.isfinite(result["eiq_score"])

    def test_component_score_out_of_range(self):
        """Component scores outside [0,1] should be clamped."""
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        scores = {"granularity": 2.0, "synchrony": -0.5}
        result = idx.compute_eiq(component_scores=scores)
        assert result is not None
        for d, v in result["dimensions"].items():
            assert 0.0 <= v <= 100.0

    def test_empty_component_scores_dict(self):
        """Empty component_scores dict should still return a valid result with defaults."""
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        result = idx.compute_eiq(component_scores={})
        assert result is not None
        assert 0 <= result["eiq_score"] <= 100

    def test_baseline_with_nan(self):
        """Baseline with NaN should not crash."""
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = np.random.randn(4, 1024) * 20
        eeg[0, 50] = np.nan
        result = idx.set_baseline(eeg)
        assert result["baseline_set"] is True

    def test_repeated_compute_consistent(self):
        """Same input should produce same output."""
        np.random.seed(42)
        idx = EmotionalIntelligenceIndex(fs=256.0)
        eeg = _make_eeg(seed=42)
        r1 = idx.compute_eiq(eeg.copy())
        r2 = idx.compute_eiq(eeg.copy())
        assert r1["eiq_score"] == r2["eiq_score"]


# ===========================================================================
# TestGetModel
# ===========================================================================


class TestGetModel:
    """Tests for module-level get_model function."""

    def test_get_model_returns_instance(self):
        from models.ei_composite import get_model
        model = get_model()
        assert isinstance(model, EmotionalIntelligenceIndex)

    def test_get_model_is_singleton(self):
        from models.ei_composite import get_model
        m1 = get_model()
        m2 = get_model()
        assert m1 is m2
