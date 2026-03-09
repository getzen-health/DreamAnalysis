"""Tests for BrainHealthScore -- composite brain health from EEG.

Covers:
  - Output keys and value ranges
  - Grade thresholds (A>=80, B>=65, C>=50, D>=35, F<35)
  - Domain score ranges (each 0-100)
  - High-quality vs low-quality signals
  - Baseline effects
  - Session stats
  - History tracking
  - Multi-user isolation
  - Reset
  - Edge cases (single channel, zeros, short signal, constant, large amplitude)
  - Recommendations
  - Reproducibility
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.brain_health_score import (
    BrainHealthScore,
    DOMAINS,
    GRADE_THRESHOLDS,
    _grade_from_score,
)


# -- Helpers --------------------------------------------------------------- #


def _make_eeg_4ch(rng, n_samples=1024, scale=20.0):
    """4-channel synthetic EEG at ~20 uV RMS."""
    return rng.normal(0, scale, (4, n_samples)).astype(np.float64)


def _make_clean_alpha_eeg(rng, n_samples=2048, fs=256.0):
    """4-channel EEG with strong 10 Hz alpha + mild noise.

    Should score well on spectral and asymmetry domains.
    """
    t = np.arange(n_samples) / fs
    alpha_10hz = 30.0 * np.sin(2 * np.pi * 10.0 * t)
    channels = []
    for _ in range(4):
        noise = rng.normal(0, 3, n_samples)
        channels.append(alpha_10hz + noise)
    return np.array(channels)


def _make_noisy_eeg(rng, n_samples=2048, fs=256.0):
    """4-channel EEG dominated by high-frequency noise (low quality)."""
    t = np.arange(n_samples) / fs
    channels = []
    for _ in range(4):
        noise = rng.normal(0, 50, n_samples)
        hf = 40.0 * np.sin(2 * np.pi * 45.0 * t)
        channels.append(noise + hf)
    return np.array(channels)


# -- 1. Output structure --------------------------------------------------- #


class TestOutputStructure:
    """Verify assess() returns all required keys with correct types."""

    def test_assess_output_keys(self):
        rng = np.random.default_rng(42)
        scorer = BrainHealthScore(fs=256.0)
        result = scorer.assess(_make_eeg_4ch(rng), fs=256.0)

        assert "overall_score" in result
        assert "grade" in result
        assert "domain_scores" in result
        assert "recommendations" in result
        assert "has_baseline" in result

    def test_assess_domain_keys(self):
        rng = np.random.default_rng(1)
        scorer = BrainHealthScore()
        result = scorer.assess(_make_eeg_4ch(rng))

        for domain in DOMAINS:
            assert domain in result["domain_scores"], (
                f"Missing domain: {domain}"
            )

    def test_set_baseline_output_keys(self):
        rng = np.random.default_rng(2)
        scorer = BrainHealthScore()
        result = scorer.set_baseline(_make_eeg_4ch(rng))

        assert result["baseline_set"] is True
        assert "domain_scores" in result
        for domain in DOMAINS:
            assert domain in result["domain_scores"]


# -- 2. Score ranges ------------------------------------------------------- #


class TestScoreRanges:
    """All scores must be in [0, 100]."""

    def test_overall_score_range(self):
        rng = np.random.default_rng(10)
        scorer = BrainHealthScore()
        result = scorer.assess(_make_eeg_4ch(rng, n_samples=2048))

        assert 0 <= result["overall_score"] <= 100

    def test_domain_scores_range(self):
        rng = np.random.default_rng(11)
        scorer = BrainHealthScore()
        result = scorer.assess(_make_eeg_4ch(rng, n_samples=2048))

        for domain, score in result["domain_scores"].items():
            assert 0 <= score <= 100, f"{domain} score {score} out of range"

    def test_all_domains_present(self):
        rng = np.random.default_rng(12)
        scorer = BrainHealthScore()
        result = scorer.assess(_make_eeg_4ch(rng))

        assert len(result["domain_scores"]) == 5


# -- 3. Grade thresholds -------------------------------------------------- #


class TestGradeThresholds:
    """Verify A >= 80, B >= 65, C >= 50, D >= 35, F < 35."""

    def test_grade_A(self):
        assert _grade_from_score(80.0) == "A"
        assert _grade_from_score(95.0) == "A"
        assert _grade_from_score(100.0) == "A"

    def test_grade_B(self):
        assert _grade_from_score(65.0) == "B"
        assert _grade_from_score(79.9) == "B"

    def test_grade_C(self):
        assert _grade_from_score(50.0) == "C"
        assert _grade_from_score(64.9) == "C"

    def test_grade_D(self):
        assert _grade_from_score(35.0) == "D"
        assert _grade_from_score(49.9) == "D"

    def test_grade_F(self):
        assert _grade_from_score(0.0) == "F"
        assert _grade_from_score(34.9) == "F"

    def test_assess_returns_valid_grade(self):
        rng = np.random.default_rng(20)
        scorer = BrainHealthScore()
        result = scorer.assess(_make_eeg_4ch(rng))
        assert result["grade"] in ("A", "B", "C", "D", "F")


# -- 4. Signal quality impact --------------------------------------------- #


class TestSignalQuality:
    """Clean alpha EEG should score higher than noisy broadband."""

    def test_clean_alpha_scores_higher_spectral(self):
        rng = np.random.default_rng(30)
        scorer_clean = BrainHealthScore()
        scorer_noisy = BrainHealthScore()

        clean = _make_clean_alpha_eeg(rng)
        noisy = _make_noisy_eeg(rng)

        res_clean = scorer_clean.assess(clean)
        res_noisy = scorer_noisy.assess(noisy)

        assert (
            res_clean["domain_scores"]["spectral"]
            > res_noisy["domain_scores"]["spectral"]
        ), "Clean alpha should have higher spectral score"

    def test_clean_alpha_scores_higher_overall(self):
        rng = np.random.default_rng(31)
        scorer = BrainHealthScore()

        clean = _make_clean_alpha_eeg(rng)
        noisy = _make_noisy_eeg(rng)

        res_clean = scorer.assess(clean)
        res_noisy = scorer.assess(noisy)

        assert res_clean["overall_score"] >= res_noisy["overall_score"]


# -- 5. Baseline effects -------------------------------------------------- #


class TestBaseline:
    """Baseline should affect has_baseline and stability scoring."""

    def test_no_baseline_by_default(self):
        rng = np.random.default_rng(40)
        scorer = BrainHealthScore()
        result = scorer.assess(_make_eeg_4ch(rng))
        assert result["has_baseline"] is False

    def test_baseline_sets_flag(self):
        rng = np.random.default_rng(41)
        scorer = BrainHealthScore()
        signals = _make_eeg_4ch(rng)
        scorer.set_baseline(signals)
        result = scorer.assess(signals)
        assert result["has_baseline"] is True

    def test_baseline_returns_domain_scores(self):
        rng = np.random.default_rng(42)
        scorer = BrainHealthScore()
        result = scorer.set_baseline(_make_eeg_4ch(rng))
        assert result["baseline_set"] is True
        assert len(result["domain_scores"]) == 5

    def test_baseline_improves_stability_when_signal_matches(self):
        """Same signal as baseline should give high stability."""
        rng = np.random.default_rng(43)
        signals = _make_eeg_4ch(rng, n_samples=2048)
        scorer = BrainHealthScore()
        scorer.set_baseline(signals)
        result = scorer.assess(signals)
        assert result["domain_scores"]["stability"] >= 40.0


# -- 6. Session stats ----------------------------------------------------- #


class TestSessionStats:
    """get_session_stats should track epochs and compute aggregates."""

    def test_empty_session_stats(self):
        scorer = BrainHealthScore()
        stats = scorer.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["has_baseline"] is False
        assert "mean_score" not in stats

    def test_session_stats_after_assess(self):
        rng = np.random.default_rng(50)
        scorer = BrainHealthScore()
        scorer.assess(_make_eeg_4ch(rng))
        scorer.assess(_make_eeg_4ch(rng))

        stats = scorer.get_session_stats()
        assert stats["n_epochs"] == 2
        assert "mean_score" in stats
        assert "best_domain" in stats
        assert "worst_domain" in stats
        assert stats["best_domain"] in DOMAINS
        assert stats["worst_domain"] in DOMAINS

    def test_session_stats_with_baseline(self):
        rng = np.random.default_rng(51)
        scorer = BrainHealthScore()
        scorer.set_baseline(_make_eeg_4ch(rng))
        stats = scorer.get_session_stats()
        assert stats["has_baseline"] is True
        assert stats["n_epochs"] == 0  # set_baseline is not an assess epoch

    def test_session_stats_mean_score_in_range(self):
        rng = np.random.default_rng(52)
        scorer = BrainHealthScore()
        scorer.assess(_make_eeg_4ch(rng))
        stats = scorer.get_session_stats()
        assert 0 <= stats["mean_score"] <= 100


# -- 7. History ------------------------------------------------------------ #


class TestHistory:
    """get_history should return past assess results."""

    def test_empty_history(self):
        scorer = BrainHealthScore()
        assert scorer.get_history() == []

    def test_history_grows_with_assess(self):
        rng = np.random.default_rng(60)
        scorer = BrainHealthScore()
        for _ in range(3):
            scorer.assess(_make_eeg_4ch(rng))

        history = scorer.get_history()
        assert len(history) == 3

    def test_history_last_n(self):
        rng = np.random.default_rng(61)
        scorer = BrainHealthScore()
        for _ in range(5):
            scorer.assess(_make_eeg_4ch(rng))

        assert len(scorer.get_history(last_n=2)) == 2
        assert len(scorer.get_history(last_n=10)) == 5

    def test_history_entries_have_correct_keys(self):
        rng = np.random.default_rng(62)
        scorer = BrainHealthScore()
        scorer.assess(_make_eeg_4ch(rng))

        entry = scorer.get_history()[0]
        assert "overall_score" in entry
        assert "grade" in entry
        assert "domain_scores" in entry

    def test_set_baseline_does_not_add_to_history(self):
        rng = np.random.default_rng(63)
        scorer = BrainHealthScore()
        scorer.set_baseline(_make_eeg_4ch(rng))
        assert scorer.get_history() == []


# -- 8. Multi-user support ------------------------------------------------ #


class TestMultiUser:
    """Baselines and histories are isolated per user_id."""

    def test_separate_baselines_per_user(self):
        rng = np.random.default_rng(70)
        scorer = BrainHealthScore()

        scorer.set_baseline(_make_eeg_4ch(rng), user_id="alice")
        scorer.set_baseline(_make_eeg_4ch(rng), user_id="bob")

        res_alice = scorer.assess(_make_eeg_4ch(rng), user_id="alice")
        res_carol = scorer.assess(_make_eeg_4ch(rng), user_id="carol")

        assert res_alice["has_baseline"] is True
        assert res_carol["has_baseline"] is False

    def test_separate_histories_per_user(self):
        rng = np.random.default_rng(71)
        scorer = BrainHealthScore()

        scorer.assess(_make_eeg_4ch(rng), user_id="alice")
        scorer.assess(_make_eeg_4ch(rng), user_id="alice")
        scorer.assess(_make_eeg_4ch(rng), user_id="bob")

        assert len(scorer.get_history(user_id="alice")) == 2
        assert len(scorer.get_history(user_id="bob")) == 1

    def test_reset_one_user_preserves_other(self):
        rng = np.random.default_rng(72)
        scorer = BrainHealthScore()

        scorer.set_baseline(_make_eeg_4ch(rng), user_id="alice")
        scorer.assess(_make_eeg_4ch(rng), user_id="alice")
        scorer.set_baseline(_make_eeg_4ch(rng), user_id="bob")
        scorer.assess(_make_eeg_4ch(rng), user_id="bob")

        scorer.reset(user_id="alice")

        assert scorer.get_history(user_id="alice") == []
        assert len(scorer.get_history(user_id="bob")) == 1
        assert scorer.get_session_stats(user_id="bob")["has_baseline"] is True


# -- 9. Reset -------------------------------------------------------------- #


class TestReset:
    """reset() clears baseline and history."""

    def test_reset_clears_history(self):
        rng = np.random.default_rng(80)
        scorer = BrainHealthScore()
        scorer.assess(_make_eeg_4ch(rng))
        scorer.assess(_make_eeg_4ch(rng))
        scorer.reset()
        assert scorer.get_history() == []
        assert scorer.get_session_stats()["n_epochs"] == 0

    def test_reset_clears_baseline(self):
        rng = np.random.default_rng(81)
        scorer = BrainHealthScore()
        scorer.set_baseline(_make_eeg_4ch(rng))
        scorer.reset()
        result = scorer.assess(_make_eeg_4ch(rng))
        assert result["has_baseline"] is False

    def test_usable_after_reset(self):
        rng = np.random.default_rng(82)
        scorer = BrainHealthScore()
        scorer.assess(_make_eeg_4ch(rng))
        scorer.reset()
        result = scorer.assess(_make_eeg_4ch(rng))
        assert 0 <= result["overall_score"] <= 100


# -- 10. Edge cases -------------------------------------------------------- #


class TestEdgeCases:
    """Handle unusual inputs gracefully."""

    def test_single_channel_input(self):
        rng = np.random.default_rng(90)
        eeg = rng.normal(0, 20, 1024)
        scorer = BrainHealthScore()
        result = scorer.assess(eeg)
        assert 0 <= result["overall_score"] <= 100
        # Connectivity and asymmetry should fall back to 50
        assert result["domain_scores"]["connectivity"] == 50.0
        assert result["domain_scores"]["asymmetry"] == 50.0

    def test_zeros_signal(self):
        scorer = BrainHealthScore()
        zeros = np.zeros((4, 1024))
        result = scorer.assess(zeros)
        assert 0 <= result["overall_score"] <= 100
        assert result["grade"] in ("A", "B", "C", "D", "F")

    def test_short_signal(self):
        rng = np.random.default_rng(92)
        short = rng.normal(0, 20, (4, 32))
        scorer = BrainHealthScore()
        result = scorer.assess(short)
        assert 0 <= result["overall_score"] <= 100

    def test_two_channel_input(self):
        rng = np.random.default_rng(93)
        eeg = rng.normal(0, 20, (2, 1024))
        scorer = BrainHealthScore()
        result = scorer.assess(eeg)
        assert 0 <= result["overall_score"] <= 100
        assert 0 <= result["domain_scores"]["connectivity"] <= 100
        assert 0 <= result["domain_scores"]["asymmetry"] <= 100

    def test_large_amplitude_signal(self):
        rng = np.random.default_rng(94)
        large = rng.normal(0, 500, (4, 1024))
        scorer = BrainHealthScore()
        result = scorer.assess(large)
        assert 0 <= result["overall_score"] <= 100

    def test_constant_signal_does_not_crash(self):
        scorer = BrainHealthScore()
        const = np.ones((4, 1024)) * 10.0
        result = scorer.assess(const)
        assert 0 <= result["overall_score"] <= 100

    def test_all_values_finite(self):
        rng = np.random.default_rng(96)
        scorer = BrainHealthScore()
        result = scorer.assess(_make_eeg_4ch(rng))
        assert np.isfinite(result["overall_score"])
        for score in result["domain_scores"].values():
            assert np.isfinite(score), f"Non-finite domain score: {score}"


# -- 11. Recommendations -------------------------------------------------- #


class TestRecommendations:
    """Recommendations list should be non-empty strings."""

    def test_recommendations_non_empty(self):
        rng = np.random.default_rng(100)
        scorer = BrainHealthScore()
        result = scorer.assess(_make_eeg_4ch(rng))
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) > 0

    def test_good_signal_gets_positive_recommendation(self):
        rng = np.random.default_rng(101)
        scorer = BrainHealthScore()
        clean = _make_clean_alpha_eeg(rng)
        result = scorer.assess(clean)
        assert len(result["recommendations"]) >= 1


# -- 12. Reproducibility -------------------------------------------------- #


class TestReproducibility:
    """Same input should produce same output."""

    def test_deterministic_output(self):
        rng1 = np.random.default_rng(200)
        rng2 = np.random.default_rng(200)
        scorer1 = BrainHealthScore()
        scorer2 = BrainHealthScore()

        eeg1 = _make_eeg_4ch(rng1, n_samples=2048)
        eeg2 = _make_eeg_4ch(rng2, n_samples=2048)

        r1 = scorer1.assess(eeg1)
        r2 = scorer2.assess(eeg2)

        assert r1["overall_score"] == r2["overall_score"]
        assert r1["grade"] == r2["grade"]
        for d in DOMAINS:
            assert r1["domain_scores"][d] == r2["domain_scores"][d]
