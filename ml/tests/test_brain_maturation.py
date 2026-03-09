"""Tests for brain maturation tracker."""
import numpy as np
import pytest

from models.brain_maturation import BrainMaturationTracker


@pytest.fixture
def tracker():
    return BrainMaturationTracker(fs=256.0)


def _make_child_eeg(fs=256, duration=10):
    """Child-like EEG: low alpha peak (~7 Hz), high theta, steep 1/f."""
    t = np.arange(int(fs * duration)) / fs
    theta = 20.0 * np.sin(2 * np.pi * 6 * t)  # dominant theta
    alpha = 5.0 * np.sin(2 * np.pi * 7 * t)   # low alpha peak
    noise = 3.0 * np.random.randn(len(t))
    return (theta + alpha + noise).reshape(1, -1)


def _make_adult_eeg(fs=256, duration=10):
    """Adult EEG: alpha peak ~10 Hz, moderate beta, less theta."""
    t = np.arange(int(fs * duration)) / fs
    alpha = 20.0 * np.sin(2 * np.pi * 10 * t)  # dominant alpha
    beta = 8.0 * np.sin(2 * np.pi * 20 * t)
    theta = 3.0 * np.sin(2 * np.pi * 6 * t)    # low theta
    noise = 3.0 * np.random.randn(len(t))
    return (alpha + beta + theta + noise).reshape(1, -1)


def _make_multichannel(fs=256, duration=10, n_channels=4):
    """4-channel adult EEG."""
    signals = []
    for ch in range(n_channels):
        t = np.arange(int(fs * duration)) / fs
        alpha = 15.0 * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        beta = 6.0 * np.sin(2 * np.pi * 22 * t + ch * 0.5)
        noise = 3.0 * np.random.randn(len(t))
        signals.append(alpha + beta + noise)
    return np.array(signals)


class TestBasicAssessment:
    def test_output_keys(self, tracker):
        np.random.seed(42)
        result = tracker.assess(_make_adult_eeg())
        expected = {"estimated_brain_age", "brain_age_gap", "developmental_stage",
                    "maturation_features", "normative_comparison", "n_channels"}
        assert expected.issubset(set(result.keys()))

    def test_age_positive(self, tracker):
        np.random.seed(42)
        result = tracker.assess(_make_adult_eeg())
        assert result["estimated_brain_age"] > 0

    def test_no_bag_without_age(self, tracker):
        np.random.seed(42)
        result = tracker.assess(_make_adult_eeg())
        assert result["brain_age_gap"] is None

    def test_bag_computed_with_age(self, tracker):
        np.random.seed(42)
        result = tracker.assess(_make_adult_eeg(), chronological_age=25)
        assert result["brain_age_gap"] is not None


class TestDevelopmentalStages:
    def test_child_eeg_younger(self, tracker):
        np.random.seed(42)
        child = tracker.assess(_make_child_eeg())
        adult = tracker.assess(_make_adult_eeg())
        assert child["estimated_brain_age"] < adult["estimated_brain_age"]

    def test_child_stage(self, tracker):
        np.random.seed(42)
        result = tracker.assess(_make_child_eeg())
        assert result["developmental_stage"] in ("early_development", "developing")

    def test_adult_stage(self, tracker):
        np.random.seed(42)
        result = tracker.assess(_make_adult_eeg())
        assert result["developmental_stage"] in ("adolescent", "mature")


class TestFeatures:
    def test_maturation_features_present(self, tracker):
        np.random.seed(42)
        result = tracker.assess(_make_adult_eeg())
        feats = result["maturation_features"]
        expected_feats = {"alpha_peak_hz", "aperiodic_exponent", "theta_relative",
                          "alpha_relative", "beta_relative", "spectral_entropy"}
        assert expected_feats.issubset(set(feats.keys()))

    def test_relative_powers_sum(self, tracker):
        np.random.seed(42)
        feats = tracker.assess(_make_adult_eeg())["maturation_features"]
        total = feats["delta_relative"] + feats["theta_relative"] + feats["alpha_relative"] + feats["beta_relative"]
        assert abs(total - 1.0) < 0.15  # approximately 1 (gamma excluded)

    def test_alpha_peak_range(self, tracker):
        np.random.seed(42)
        feats = tracker.assess(_make_adult_eeg())["maturation_features"]
        assert 6 <= feats["alpha_peak_hz"] <= 13


class TestNormativeComparison:
    def test_no_age_no_comparison(self, tracker):
        np.random.seed(42)
        result = tracker.assess(_make_adult_eeg())
        assert result["normative_comparison"]["status"] == "no_age_provided"

    def test_with_age_has_comparisons(self, tracker):
        np.random.seed(42)
        result = tracker.assess(_make_adult_eeg(), chronological_age=30)
        assert "age_group" in result["normative_comparison"]
        assert result["normative_comparison"]["age_group"] == "adult"


class TestMultichannel:
    def test_multichannel_works(self, tracker):
        np.random.seed(42)
        result = tracker.assess(_make_multichannel())
        assert result["n_channels"] == 4

    def test_single_channel_input(self, tracker):
        np.random.seed(42)
        signal = np.random.randn(256 * 10)
        result = tracker.assess(signal)
        assert result["n_channels"] == 1


class TestTrajectory:
    def test_empty_trajectory(self, tracker):
        assert tracker.get_trajectory() == []

    def test_trajectory_grows(self, tracker):
        np.random.seed(42)
        tracker.assess(_make_adult_eeg())
        tracker.assess(_make_adult_eeg())
        assert len(tracker.get_trajectory()) == 2


class TestSummary:
    def test_empty_summary(self, tracker):
        assert tracker.get_summary()["n_assessments"] == 0

    def test_summary_with_data(self, tracker):
        np.random.seed(42)
        tracker.assess(_make_adult_eeg(), chronological_age=25)
        tracker.assess(_make_adult_eeg(), chronological_age=25)
        s = tracker.get_summary()
        assert s["n_assessments"] == 2
        assert s["mean_bag"] is not None


class TestMultiUser:
    def test_independent_users(self, tracker):
        np.random.seed(42)
        tracker.assess(_make_child_eeg(), user_id="kid")
        tracker.assess(_make_adult_eeg(), user_id="adult")
        assert len(tracker.get_trajectory("kid")) == 1
        assert len(tracker.get_trajectory("adult")) == 1


class TestReset:
    def test_reset_clears(self, tracker):
        np.random.seed(42)
        tracker.assess(_make_adult_eeg())
        tracker.reset()
        assert tracker.get_summary()["n_assessments"] == 0
