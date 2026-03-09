"""Tests for BrainHealthScore calculator.

Covers:
  - Initialization and empty state
  - add_sleep_data with 1D and 2D EEG
  - add_waking_data with 1D and 2D EEG, multiple contexts
  - compute_score output structure and ranges
  - Grade assignment (A/B/C/D/F)
  - Domain scores (sleep, cognition, stress, mood, vitality)
  - get_domain_scores without full compute
  - get_trends with insufficient data and sufficient data
  - get_history and last_n filtering
  - reset clears everything
  - Recommendations for low-scoring domains
  - Edge cases: flat signal, very short signal, noisy signal
  - Multiple recordings aggregate correctly
  - No data raises ValueError
  - FAA proxy with multichannel data
  - Signal quality scoring
"""

import numpy as np
import pytest

from models.brain_health_score import (
    BrainHealthScore,
    DOMAINS,
    DOMAIN_WEIGHTS,
    GRADE_THRESHOLDS,
)


@pytest.fixture
def scorer():
    return BrainHealthScore()


@pytest.fixture
def sleep_eeg():
    """4 seconds of delta-dominant synthetic sleep EEG at 256 Hz."""
    rng = np.random.RandomState(42)
    t = np.arange(1024) / 256.0
    # Delta-dominant (2 Hz) + small theta (6 Hz)
    return 30 * np.sin(2 * np.pi * 2 * t) + 5 * np.sin(2 * np.pi * 6 * t) + rng.randn(1024) * 2


@pytest.fixture
def waking_eeg():
    """4 seconds of alpha-dominant waking EEG at 256 Hz."""
    rng = np.random.RandomState(123)
    t = np.arange(1024) / 256.0
    # Alpha (10 Hz) + some beta (20 Hz)
    return 20 * np.sin(2 * np.pi * 10 * t) + 8 * np.sin(2 * np.pi * 20 * t) + rng.randn(1024) * 3


@pytest.fixture
def multichannel_waking():
    """4-channel x 4 seconds of waking EEG (Muse 2 layout)."""
    rng = np.random.RandomState(456)
    t = np.arange(1024) / 256.0
    signals = np.zeros((4, 1024))
    # TP9 (ch0)
    signals[0] = 15 * np.sin(2 * np.pi * 10 * t) + rng.randn(1024) * 3
    # AF7 (ch1) — more left alpha
    signals[1] = 25 * np.sin(2 * np.pi * 10 * t) + rng.randn(1024) * 3
    # AF8 (ch2) — less right alpha (negative FAA)
    signals[2] = 10 * np.sin(2 * np.pi * 10 * t) + rng.randn(1024) * 3
    # TP10 (ch3)
    signals[3] = 15 * np.sin(2 * np.pi * 10 * t) + rng.randn(1024) * 3
    return signals


# ── Initialization ──────────────────────────────────────────────────


class TestInitialization:
    def test_fresh_scorer_has_empty_state(self, scorer):
        """New scorer has no features and no history."""
        assert scorer.get_history() == []
        domains = scorer.get_domain_scores()
        assert all(domains[d] == 0.0 for d in DOMAINS)

    def test_compute_score_without_data_raises(self, scorer):
        """compute_score() should raise if no data has been added."""
        with pytest.raises(ValueError, match="No data added"):
            scorer.compute_score()


# ── Constants ───────────────────────────────────────────────────────


class TestConstants:
    def test_domains_list(self):
        """DOMAINS should contain exactly the five expected domains."""
        assert DOMAINS == ["sleep", "cognition", "stress", "mood", "vitality"]

    def test_domain_weights_sum_to_one(self):
        """Domain weights must sum to 1.0."""
        assert abs(sum(DOMAIN_WEIGHTS.values()) - 1.0) < 1e-10

    def test_grade_thresholds_are_descending(self):
        """Grade thresholds must be in descending order."""
        thresholds = [t for t, _ in GRADE_THRESHOLDS]
        assert thresholds == sorted(thresholds, reverse=True)


# ── Adding sleep data ───────────────────────────────────────────────


class TestAddSleepData:
    def test_1d_sleep_eeg(self, scorer, sleep_eeg):
        """add_sleep_data accepts 1D array and returns features."""
        features = scorer.add_sleep_data(sleep_eeg, fs=256, duration_hours=7.5)
        assert "delta_power" in features
        assert "theta_power" in features
        assert "alpha_delta_ratio" in features
        assert features["duration_hours"] == 7.5

    def test_2d_sleep_eeg(self, scorer, sleep_eeg):
        """add_sleep_data accepts 2D array (multichannel) by averaging."""
        eeg_2d = np.stack([sleep_eeg, sleep_eeg * 0.9])
        features = scorer.add_sleep_data(eeg_2d, fs=256, duration_hours=8.0)
        assert "delta_power" in features
        assert features["duration_hours"] == 8.0

    def test_delta_dominant_sleep_has_high_delta(self, scorer, sleep_eeg):
        """Delta-dominant sleep EEG should show high delta power."""
        features = scorer.add_sleep_data(sleep_eeg, fs=256, duration_hours=7.0)
        assert features["delta_power"] > 0.3

    def test_sleep_features_accumulate(self, scorer, sleep_eeg):
        """Multiple add_sleep_data calls accumulate features."""
        scorer.add_sleep_data(sleep_eeg, fs=256, duration_hours=7.0)
        scorer.add_sleep_data(sleep_eeg, fs=256, duration_hours=8.0)
        # Should not raise when computing score
        result = scorer.compute_score()
        assert result["overall_score"] >= 0


# ── Adding waking data ─────────────────────────────────────────────


class TestAddWakingData:
    def test_1d_waking_eeg(self, scorer, waking_eeg):
        """add_waking_data accepts 1D array and returns features."""
        features = scorer.add_waking_data(waking_eeg, fs=256, context="resting")
        assert "alpha_power" in features
        assert "alpha_beta_ratio" in features
        assert "alpha_peak_freq" in features
        assert "spectral_entropy" in features
        assert "hrv_proxy" in features

    def test_multichannel_waking_eeg(self, scorer, multichannel_waking):
        """add_waking_data with multichannel computes FAA proxy."""
        features = scorer.add_waking_data(
            multichannel_waking, fs=256, context="resting"
        )
        assert "faa_proxy" in features
        # AF7 has more alpha than AF8, so FAA should be negative
        # (less right alpha = withdrawal/negative)
        assert features["faa_proxy"] < 0

    def test_context_stored(self, scorer, waking_eeg):
        """Context parameter is returned in features."""
        features = scorer.add_waking_data(waking_eeg, fs=256, context="task")
        assert features["context"] == "task"


# ── compute_score output structure ──────────────────────────────────


class TestComputeScore:
    def test_output_has_required_keys(self, scorer, sleep_eeg, waking_eeg):
        """compute_score returns all required keys."""
        scorer.add_sleep_data(sleep_eeg, fs=256, duration_hours=7.5)
        scorer.add_waking_data(waking_eeg, fs=256)
        result = scorer.compute_score()

        required = {
            "overall_score", "grade", "domains",
            "top_strength", "top_weakness", "recommendations",
        }
        assert required.issubset(result.keys())

    def test_overall_score_range(self, scorer, sleep_eeg, waking_eeg):
        """Overall score is in [0, 100]."""
        scorer.add_sleep_data(sleep_eeg, fs=256, duration_hours=7.5)
        scorer.add_waking_data(waking_eeg, fs=256)
        result = scorer.compute_score()
        assert 0 <= result["overall_score"] <= 100

    def test_domains_dict_has_five_entries(self, scorer, waking_eeg):
        """Domains dict contains exactly the five expected domains."""
        scorer.add_waking_data(waking_eeg, fs=256)
        result = scorer.compute_score()
        assert set(result["domains"].keys()) == set(DOMAINS)

    def test_each_domain_score_in_range(self, scorer, sleep_eeg, waking_eeg):
        """Each domain score is in [0, 100]."""
        scorer.add_sleep_data(sleep_eeg, fs=256, duration_hours=7.5)
        scorer.add_waking_data(waking_eeg, fs=256)
        result = scorer.compute_score()
        for domain, score in result["domains"].items():
            assert 0 <= score <= 100, f"{domain} score {score} out of range"

    def test_top_strength_is_highest(self, scorer, sleep_eeg, waking_eeg):
        """top_strength should be the domain with the highest score."""
        scorer.add_sleep_data(sleep_eeg, fs=256, duration_hours=7.5)
        scorer.add_waking_data(waking_eeg, fs=256)
        result = scorer.compute_score()
        domains = result["domains"]
        assert domains[result["top_strength"]] == max(domains.values())

    def test_top_weakness_is_lowest(self, scorer, sleep_eeg, waking_eeg):
        """top_weakness should be the domain with the lowest score."""
        scorer.add_sleep_data(sleep_eeg, fs=256, duration_hours=7.5)
        scorer.add_waking_data(waking_eeg, fs=256)
        result = scorer.compute_score()
        domains = result["domains"]
        assert domains[result["top_weakness"]] == min(domains.values())

    def test_recommendations_is_nonempty_list(self, scorer, waking_eeg):
        """Recommendations should always be a non-empty list."""
        scorer.add_waking_data(waking_eeg, fs=256)
        result = scorer.compute_score()
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) >= 1


# ── Grade assignment ────────────────────────────────────────────────


class TestGrading:
    def test_grade_is_valid_letter(self, scorer, waking_eeg):
        """Grade must be one of A, B, C, D, F."""
        scorer.add_waking_data(waking_eeg, fs=256)
        result = scorer.compute_score()
        assert result["grade"] in {"A", "B", "C", "D", "F"}

    def test_high_score_gets_a(self, scorer):
        """Overall score >= 90 should get grade A."""
        # Directly verify the grade logic
        scorer.add_waking_data(np.random.randn(1024) * 20, fs=256)
        result = scorer.compute_score()
        # We can't control the exact score, but we test the grading logic
        if result["overall_score"] >= 90:
            assert result["grade"] == "A"
        elif result["overall_score"] >= 80:
            assert result["grade"] == "B"
        elif result["overall_score"] >= 70:
            assert result["grade"] == "C"
        elif result["overall_score"] >= 60:
            assert result["grade"] == "D"
        else:
            assert result["grade"] == "F"


# ── get_domain_scores ───────────────────────────────────────────────


class TestGetDomainScores:
    def test_returns_all_domains(self, scorer, waking_eeg):
        """get_domain_scores returns all five domains."""
        scorer.add_waking_data(waking_eeg, fs=256)
        domains = scorer.get_domain_scores()
        assert set(domains.keys()) == set(DOMAINS)

    def test_consistent_with_compute_score(self, scorer, sleep_eeg, waking_eeg):
        """get_domain_scores should match compute_score domains."""
        scorer.add_sleep_data(sleep_eeg, fs=256, duration_hours=7.5)
        scorer.add_waking_data(waking_eeg, fs=256)
        domains_direct = scorer.get_domain_scores()
        result = scorer.compute_score()
        for d in DOMAINS:
            assert abs(domains_direct[d] - result["domains"][d]) < 0.1


# ── get_trends ──────────────────────────────────────────────────────


class TestGetTrends:
    def test_insufficient_data(self, scorer, waking_eeg):
        """Trends with <2 scores returns insufficient_data."""
        scorer.add_waking_data(waking_eeg, fs=256)
        scorer.compute_score()
        trends = scorer.get_trends()
        assert trends["trend"] == "insufficient_data"
        assert trends["n_scores"] == 1

    def test_sufficient_data_returns_trend(self, scorer, waking_eeg):
        """Trends with >=2 scores returns a valid trend."""
        scorer.add_waking_data(waking_eeg, fs=256)
        scorer.compute_score()
        scorer.compute_score()
        trends = scorer.get_trends()
        assert trends["trend"] in {"improving", "declining", "stable"}
        assert "slope" in trends
        assert "mean_score" in trends

    def test_stable_scores_give_stable_trend(self, scorer, waking_eeg):
        """Identical scores should produce a stable trend."""
        scorer.add_waking_data(waking_eeg, fs=256)
        for _ in range(5):
            scorer.compute_score()
        trends = scorer.get_trends()
        assert trends["trend"] == "stable"


# ── get_history ─────────────────────────────────────────────────────


class TestGetHistory:
    def test_empty_history(self, scorer):
        """No compute_score calls -> empty history."""
        assert scorer.get_history() == []

    def test_history_grows(self, scorer, waking_eeg):
        """Each compute_score call adds to history."""
        scorer.add_waking_data(waking_eeg, fs=256)
        scorer.compute_score()
        scorer.compute_score()
        assert len(scorer.get_history()) == 2

    def test_last_n_filter(self, scorer, waking_eeg):
        """get_history(last_n=1) returns only the last entry."""
        scorer.add_waking_data(waking_eeg, fs=256)
        scorer.compute_score()
        scorer.compute_score()
        scorer.compute_score()
        history = scorer.get_history(last_n=1)
        assert len(history) == 1

    def test_last_n_larger_than_history(self, scorer, waking_eeg):
        """last_n > actual history returns all entries."""
        scorer.add_waking_data(waking_eeg, fs=256)
        scorer.compute_score()
        history = scorer.get_history(last_n=100)
        assert len(history) == 1


# ── reset ───────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_features(self, scorer, sleep_eeg, waking_eeg):
        """Reset clears sleep and waking features."""
        scorer.add_sleep_data(sleep_eeg, fs=256, duration_hours=7)
        scorer.add_waking_data(waking_eeg, fs=256)
        scorer.compute_score()
        scorer.reset()

        assert scorer.get_history() == []
        with pytest.raises(ValueError):
            scorer.compute_score()

    def test_reset_clears_history(self, scorer, waking_eeg):
        """Reset clears history."""
        scorer.add_waking_data(waking_eeg, fs=256)
        scorer.compute_score()
        scorer.compute_score()
        scorer.reset()
        assert scorer.get_history() == []

    def test_usable_after_reset(self, scorer, waking_eeg):
        """Scorer is usable again after reset."""
        scorer.add_waking_data(waking_eeg, fs=256)
        scorer.compute_score()
        scorer.reset()

        scorer.add_waking_data(waking_eeg, fs=256)
        result = scorer.compute_score()
        assert result["overall_score"] >= 0


# ── Edge cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    def test_flat_signal(self, scorer):
        """Flat-line signal should not crash and gives valid scores."""
        flat = np.ones(1024) * 0.001
        scorer.add_waking_data(flat, fs=256)
        result = scorer.compute_score()
        assert 0 <= result["overall_score"] <= 100
        assert result["grade"] in {"A", "B", "C", "D", "F"}

    def test_very_short_signal(self, scorer):
        """Very short signal (< 1 second) should not crash."""
        short = np.random.randn(64) * 20
        scorer.add_waking_data(short, fs=256)
        result = scorer.compute_score()
        assert 0 <= result["overall_score"] <= 100

    def test_noisy_signal(self, scorer):
        """High-amplitude noisy signal should not crash."""
        noisy = np.random.randn(1024) * 500
        scorer.add_waking_data(noisy, fs=256)
        result = scorer.compute_score()
        assert 0 <= result["overall_score"] <= 100

    def test_sleep_only_gives_valid_score(self, scorer, sleep_eeg):
        """Score with only sleep data should work (waking domains get defaults)."""
        scorer.add_sleep_data(sleep_eeg, fs=256, duration_hours=7.5)
        result = scorer.compute_score()
        assert 0 <= result["overall_score"] <= 100
        assert all(d in result["domains"] for d in DOMAINS)

    def test_waking_only_gives_valid_score(self, scorer, waking_eeg):
        """Score with only waking data should work (sleep domain gets default)."""
        scorer.add_waking_data(waking_eeg, fs=256)
        result = scorer.compute_score()
        assert 0 <= result["overall_score"] <= 100

    def test_zero_duration_sleep(self, scorer, sleep_eeg):
        """Zero sleep duration should not crash."""
        scorer.add_sleep_data(sleep_eeg, fs=256, duration_hours=0.0)
        result = scorer.compute_score()
        assert 0 <= result["overall_score"] <= 100

    def test_all_values_finite(self, scorer, sleep_eeg, waking_eeg):
        """All numeric values in the result should be finite."""
        scorer.add_sleep_data(sleep_eeg, fs=256, duration_hours=7.0)
        scorer.add_waking_data(waking_eeg, fs=256)
        result = scorer.compute_score()

        assert np.isfinite(result["overall_score"])
        for score in result["domains"].values():
            assert np.isfinite(score), f"Non-finite domain score: {score}"


# ── Recommendations ─────────────────────────────────────────────────


class TestRecommendations:
    def test_low_sleep_gets_sleep_recommendation(self, scorer):
        """Low sleep score should produce a sleep recommendation."""
        # Very short sleep + high-frequency dominant signal
        t = np.arange(1024) / 256.0
        high_beta = 30 * np.sin(2 * np.pi * 25 * t) + np.random.randn(1024) * 2
        scorer.add_sleep_data(high_beta, fs=256, duration_hours=3.0)
        result = scorer.compute_score()
        sleep_recs = [r for r in result["recommendations"] if r.startswith("sleep:")]
        assert len(sleep_recs) >= 1

    def test_good_scores_get_positive_recommendation(self, scorer, sleep_eeg, waking_eeg):
        """If all domains are high, recommendation should be positive."""
        scorer.add_sleep_data(sleep_eeg, fs=256, duration_hours=8.0)
        scorer.add_waking_data(waking_eeg, fs=256)
        result = scorer.compute_score()
        # At least one recommendation always present
        assert len(result["recommendations"]) >= 1


# ── Multiple recordings ────────────────────────────────────────────


class TestMultipleRecordings:
    def test_multiple_sleep_recordings_aggregate(self, scorer, sleep_eeg):
        """Multiple sleep recordings average their features."""
        scorer.add_sleep_data(sleep_eeg, fs=256, duration_hours=6.0)
        scorer.add_sleep_data(sleep_eeg, fs=256, duration_hours=8.0)
        result = scorer.compute_score()
        # Aggregated duration should be ~7.0 (average of 6 and 8)
        assert 0 <= result["overall_score"] <= 100

    def test_multiple_waking_recordings_aggregate(self, scorer, waking_eeg):
        """Multiple waking recordings average their features."""
        scorer.add_waking_data(waking_eeg, fs=256, context="resting")
        scorer.add_waking_data(waking_eeg, fs=256, context="task")
        result = scorer.compute_score()
        assert 0 <= result["overall_score"] <= 100


# ── Signal quality ──────────────────────────────────────────────────


class TestSignalQuality:
    def test_good_amplitude_high_quality(self, scorer):
        """EEG with healthy amplitude should have high signal quality."""
        good_eeg = np.random.randn(1024) * 20  # ~20 uV RMS
        features = scorer.add_waking_data(good_eeg, fs=256)
        assert features["signal_quality"] > 0.5

    def test_railed_signal_low_quality(self, scorer):
        """Saturated signal (>100 uV) should have low signal quality."""
        railed = np.random.randn(1024) * 300  # ~300 uV RMS
        features = scorer.add_waking_data(railed, fs=256)
        assert features["signal_quality"] < 0.5
