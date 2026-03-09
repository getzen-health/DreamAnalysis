"""Tests for CognitiveFlexibilityDetector (cognitive_flexibility.py).

Covers baseline, assess, switch trials, flexibility levels, session stats,
history, multi-user independence, and reset.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.cognitive_flexibility import CognitiveFlexibilityDetector


# ---- Helpers --------------------------------------------------------

def _make_eeg(
    fs: int = 256,
    duration: float = 4.0,
    n_channels: int = 4,
    theta_amp: float = 10.0,
    alpha_amp: float = 10.0,
    noise_amp: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic EEG with controllable theta and alpha amplitude."""
    rng = np.random.default_rng(seed)
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    signals = []
    for ch in range(n_channels):
        theta = theta_amp * np.sin(2 * np.pi * 6 * t + ch * 0.5)
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        noise = noise_amp * rng.standard_normal(n_samples)
        signals.append(theta + alpha + noise)
    return np.array(signals)


def _make_eeg_1d(
    fs: int = 256,
    duration: float = 4.0,
    theta_amp: float = 10.0,
    alpha_amp: float = 10.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate single-channel synthetic EEG."""
    rng = np.random.default_rng(seed)
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    theta = theta_amp * np.sin(2 * np.pi * 6 * t)
    alpha = alpha_amp * np.sin(2 * np.pi * 10 * t)
    noise = 2.0 * rng.standard_normal(n_samples)
    return theta + alpha + noise


# ---- Fixtures -------------------------------------------------------

@pytest.fixture
def detector():
    return CognitiveFlexibilityDetector()


# ---- TestBaseline ---------------------------------------------------

class TestBaseline:
    def test_set_baseline_returns_dict(self, detector):
        eeg = _make_eeg(seed=1)
        result = detector.set_baseline(eeg, fs=256)
        assert isinstance(result, dict)
        assert result["baseline_set"] is True

    def test_baseline_has_theta_and_alpha(self, detector):
        eeg = _make_eeg(seed=2)
        result = detector.set_baseline(eeg, fs=256)
        assert "baseline_theta" in result
        assert "baseline_alpha" in result
        assert result["baseline_theta"] > 0
        assert result["baseline_alpha"] > 0

    def test_baseline_from_1d_signal(self, detector):
        eeg = _make_eeg_1d(seed=3)
        result = detector.set_baseline(eeg, fs=256)
        assert result["baseline_set"] is True
        assert result["baseline_theta"] > 0


# ---- TestAssess -----------------------------------------------------

class TestAssess:
    def test_output_keys(self, detector):
        eeg = _make_eeg(seed=10)
        result = detector.assess(eeg, fs=256)
        expected_keys = {
            "flexibility_score",
            "switch_cost",
            "frontal_theta_power",
            "alpha_suppression",
            "cognitive_state",
            "recommendations",
            "trial_type",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_score_range(self, detector):
        eeg = _make_eeg(seed=11)
        result = detector.assess(eeg, fs=256)
        assert 0.0 <= result["flexibility_score"] <= 100.0

    def test_switch_cost_range(self, detector):
        eeg = _make_eeg(seed=12)
        result = detector.assess(eeg, fs=256)
        assert 0.0 <= result["switch_cost"] <= 1.0

    def test_state_labels_valid(self, detector):
        valid_states = {"rigid", "moderate", "flexible", "highly_flexible"}
        for seed in range(20, 30):
            eeg = _make_eeg(seed=seed)
            result = detector.assess(eeg, fs=256)
            assert result["cognitive_state"] in valid_states

    def test_recommendations_is_list(self, detector):
        eeg = _make_eeg(seed=13)
        result = detector.assess(eeg, fs=256)
        assert isinstance(result["recommendations"], list)

    def test_single_channel_input(self, detector):
        eeg = _make_eeg_1d(seed=14)
        result = detector.assess(eeg, fs=256)
        assert 0.0 <= result["flexibility_score"] <= 100.0

    def test_alpha_suppression_range(self, detector):
        eeg = _make_eeg(seed=15)
        result = detector.assess(eeg, fs=256)
        assert 0.0 <= result["alpha_suppression"] <= 1.0

    def test_frontal_theta_positive(self, detector):
        eeg = _make_eeg(theta_amp=15.0, seed=16)
        result = detector.assess(eeg, fs=256)
        assert result["frontal_theta_power"] > 0


# ---- TestSwitchTrials -----------------------------------------------

class TestSwitchTrials:
    def test_switch_trial_flagged(self, detector):
        eeg = _make_eeg(seed=30)
        result = detector.assess(eeg, fs=256, is_switch_trial=True)
        assert result["trial_type"] == "switch"

    def test_sustain_trial_flagged(self, detector):
        eeg = _make_eeg(seed=31)
        result = detector.assess(eeg, fs=256, is_switch_trial=False)
        assert result["trial_type"] == "sustain"

    def test_switch_cost_updates_with_trials(self, detector):
        # Run several sustain trials, then a switch trial
        for i in range(3):
            detector.assess(
                _make_eeg(theta_amp=8.0, seed=40 + i), fs=256, is_switch_trial=False
            )
        # Switch trial with higher theta
        result = detector.assess(
            _make_eeg(theta_amp=20.0, seed=50), fs=256, is_switch_trial=True
        )
        # Switch cost should reflect theta difference
        assert result["switch_cost"] >= 0.0

    def test_switch_cost_with_both_types(self, detector):
        # Sustain trials (lower theta)
        for i in range(3):
            detector.assess(
                _make_eeg(theta_amp=5.0, seed=60 + i), fs=256, is_switch_trial=False
            )
        # Switch trials (higher theta)
        for i in range(3):
            detector.assess(
                _make_eeg(theta_amp=25.0, seed=70 + i), fs=256, is_switch_trial=True
            )
        stats = detector.get_session_stats()
        assert stats["n_switch"] == 3
        assert stats["n_sustain"] == 3


# ---- TestFlexibilityLevels -----------------------------------------

class TestFlexibilityLevels:
    def test_high_theta_switch_higher_flexibility(self, detector):
        """High frontal theta during switch trial should yield higher flexibility."""
        detector.set_baseline(
            _make_eeg(theta_amp=5.0, alpha_amp=15.0, seed=80), fs=256
        )
        # Assess with high theta (strong cognitive control engagement)
        high_theta = detector.assess(
            _make_eeg(theta_amp=30.0, alpha_amp=5.0, seed=81), fs=256
        )
        # Assess with low theta (weak engagement)
        detector_low = CognitiveFlexibilityDetector()
        detector_low.set_baseline(
            _make_eeg(theta_amp=5.0, alpha_amp=15.0, seed=80), fs=256
        )
        low_theta = detector_low.assess(
            _make_eeg(theta_amp=2.0, alpha_amp=15.0, seed=82), fs=256
        )
        assert high_theta["flexibility_score"] > low_theta["flexibility_score"]

    def test_rigid_state_with_low_engagement(self, detector):
        """Very low theta and no alpha suppression should tend toward rigid."""
        detector.set_baseline(
            _make_eeg(theta_amp=10.0, alpha_amp=10.0, seed=83), fs=256
        )
        result = detector.assess(
            _make_eeg(theta_amp=1.0, alpha_amp=20.0, seed=84), fs=256
        )
        # With very low theta relative to baseline and high alpha (no suppression),
        # score should be low
        assert result["flexibility_score"] < 50.0

    def test_score_varies_with_theta(self):
        """Flexibility score should generally increase with frontal theta."""
        scores = []
        for theta_amp in [2.0, 10.0, 30.0]:
            det = CognitiveFlexibilityDetector()
            result = det.assess(
                _make_eeg(theta_amp=theta_amp, alpha_amp=5.0, seed=85),
                fs=256,
            )
            scores.append(result["flexibility_score"])
        # Monotonically increasing (or at least last > first)
        assert scores[-1] > scores[0]


# ---- TestSessionStats -----------------------------------------------

class TestSessionStats:
    def test_empty_stats(self, detector):
        stats = detector.get_session_stats()
        assert stats["n_trials"] == 0
        assert stats["n_switch"] == 0
        assert stats["n_sustain"] == 0

    def test_stats_after_data(self, detector):
        for i in range(5):
            detector.assess(_make_eeg(seed=90 + i), fs=256)
        stats = detector.get_session_stats()
        assert stats["n_trials"] == 5
        assert "mean_flexibility" in stats
        assert "mean_switch_cost" in stats

    def test_switch_sustain_counts(self, detector):
        for i in range(3):
            detector.assess(
                _make_eeg(seed=100 + i), fs=256, is_switch_trial=True
            )
        for i in range(2):
            detector.assess(
                _make_eeg(seed=110 + i), fs=256, is_switch_trial=False
            )
        stats = detector.get_session_stats()
        assert stats["n_switch"] == 3
        assert stats["n_sustain"] == 2
        assert stats["n_trials"] == 5

    def test_has_baseline_flag(self, detector):
        stats = detector.get_session_stats()
        assert stats["has_baseline"] is False
        detector.set_baseline(_make_eeg(seed=120), fs=256)
        stats = detector.get_session_stats()
        assert stats["has_baseline"] is True


# ---- TestHistory ----------------------------------------------------

class TestHistory:
    def test_empty_history(self, detector):
        history = detector.get_history()
        assert history == []

    def test_history_grows(self, detector):
        for i in range(4):
            detector.assess(_make_eeg(seed=130 + i), fs=256)
        history = detector.get_history()
        assert len(history) == 4

    def test_last_n(self, detector):
        for i in range(10):
            detector.assess(_make_eeg(seed=140 + i), fs=256)
        last_3 = detector.get_history(last_n=3)
        assert len(last_3) == 3
        # Should be the last 3 entries
        full = detector.get_history()
        assert last_3 == full[-3:]

    def test_history_returns_copy(self, detector):
        detector.assess(_make_eeg(seed=150), fs=256)
        h1 = detector.get_history()
        h2 = detector.get_history()
        assert h1 is not h2  # different list objects


# ---- TestMultiUser --------------------------------------------------

class TestMultiUser:
    def test_independent_detectors(self):
        """Two detector instances should be fully independent."""
        det_a = CognitiveFlexibilityDetector()
        det_b = CognitiveFlexibilityDetector()

        det_a.set_baseline(_make_eeg(seed=200), fs=256)
        det_a.assess(_make_eeg(seed=201), fs=256)
        det_a.assess(_make_eeg(seed=202), fs=256)

        det_b.assess(_make_eeg(seed=210), fs=256)

        assert det_a.get_session_stats()["n_trials"] == 2
        assert det_b.get_session_stats()["n_trials"] == 1
        assert det_a.get_session_stats()["has_baseline"] is True
        assert det_b.get_session_stats()["has_baseline"] is False


# ---- TestReset ------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self, detector):
        detector.set_baseline(_make_eeg(seed=300), fs=256)
        detector.assess(_make_eeg(seed=301), fs=256)
        detector.assess(_make_eeg(seed=302), fs=256, is_switch_trial=True)

        detector.reset()

        assert detector.get_history() == []
        stats = detector.get_session_stats()
        assert stats["n_trials"] == 0
        assert stats["n_switch"] == 0
        assert stats["n_sustain"] == 0
        assert stats["has_baseline"] is False

    def test_assess_works_after_reset(self, detector):
        detector.set_baseline(_make_eeg(seed=310), fs=256)
        detector.assess(_make_eeg(seed=311), fs=256)
        detector.reset()

        result = detector.assess(_make_eeg(seed=312), fs=256)
        assert 0.0 <= result["flexibility_score"] <= 100.0
        assert detector.get_session_stats()["n_trials"] == 1
