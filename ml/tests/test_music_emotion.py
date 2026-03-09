"""Tests for MusicEmotionDetector -- music-induced emotion detection from EEG.

Covers:
  - Output schema and value ranges
  - Baseline calibration
  - Temporal asymmetry (TP9 vs TP10) as primary marker
  - Frontal theta for frisson/chills detection
  - 4 music emotion quadrants
  - Engagement levels
  - Frisson detection
  - Session statistics
  - History tracking
  - Reset behaviour
  - Edge cases (zeros, flat signal, single-channel)
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.music_emotion import (
    MusicEmotionDetector,
    MUSIC_QUADRANTS,
    ENGAGEMENT_LEVELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eeg(duration_s: float = 4.0, fs: float = 256.0, n_channels: int = 4,
              alpha_amp: float = 10.0, beta_amp: float = 5.0,
              theta_amp: float = 3.0, seed: int = 42) -> np.ndarray:
    """Synthesise multichannel EEG with controllable band amplitudes.

    Channels follow Muse 2 order: TP9, AF7, AF8, TP10.
    """
    rng = np.random.RandomState(seed)
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs
    eeg = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        eeg[ch] = (
            alpha_amp * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
            + beta_amp * np.sin(2 * np.pi * 20 * t + rng.uniform(0, 2 * np.pi))
            + theta_amp * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2 * np.pi))
            + rng.randn(n_samples) * 2.0
        )
    return eeg


def _make_asymmetric_eeg(left_alpha: float = 10.0, right_alpha: float = 10.0,
                          duration_s: float = 4.0, fs: float = 256.0,
                          seed: int = 99) -> np.ndarray:
    """Create EEG where TP9 (ch0) and TP10 (ch3) have different alpha power.

    This controls temporal asymmetry, the primary music emotion marker.
    """
    rng = np.random.RandomState(seed)
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs
    eeg = np.zeros((4, n_samples))
    alphas = [left_alpha, 10.0, 10.0, right_alpha]
    for ch in range(4):
        eeg[ch] = (
            alphas[ch] * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
            + 5.0 * np.sin(2 * np.pi * 20 * t + rng.uniform(0, 2 * np.pi))
            + 3.0 * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2 * np.pi))
            + rng.randn(n_samples) * 2.0
        )
    return eeg


def _make_frisson_eeg(fs: float = 256.0, seed: int = 77) -> np.ndarray:
    """Create EEG that should trigger frisson: high frontal theta burst + alpha drop."""
    rng = np.random.RandomState(seed)
    n_samples = int(4.0 * fs)
    t = np.arange(n_samples) / fs
    eeg = np.zeros((4, n_samples))
    for ch in range(4):
        # High theta (6 Hz) on frontal channels (AF7=ch1, AF8=ch2)
        theta_amp = 25.0 if ch in (1, 2) else 5.0
        # Low alpha on frontal channels (suppressed during frisson)
        alpha_amp = 2.0 if ch in (1, 2) else 10.0
        eeg[ch] = (
            alpha_amp * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
            + theta_amp * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2 * np.pi))
            + 3.0 * np.sin(2 * np.pi * 20 * t + rng.uniform(0, 2 * np.pi))
            + rng.randn(n_samples) * 1.5
        )
    return eeg


@pytest.fixture
def detector():
    return MusicEmotionDetector()


@pytest.fixture
def calibrated_detector():
    det = MusicEmotionDetector()
    baseline_eeg = _make_eeg(duration_s=4.0, seed=10)
    det.set_baseline(baseline_eeg, fs=256.0)
    return det


# ---------------------------------------------------------------------------
# TestOutputSchema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """assess() returns all required keys with correct types and ranges."""

    def test_required_keys(self, detector):
        eeg = _make_eeg()
        result = detector.assess(eeg, fs=256.0)
        required = {
            "music_valence", "music_arousal", "music_emotion",
            "engagement_level", "temporal_asymmetry", "frisson_detected",
        }
        assert required.issubset(result.keys())

    def test_valence_range(self, detector):
        result = detector.assess(_make_eeg(), fs=256.0)
        assert -1.0 <= result["music_valence"] <= 1.0

    def test_arousal_range(self, detector):
        result = detector.assess(_make_eeg(), fs=256.0)
        assert 0.0 <= result["music_arousal"] <= 1.0

    def test_emotion_is_valid_quadrant(self, detector):
        result = detector.assess(_make_eeg(), fs=256.0)
        assert result["music_emotion"] in MUSIC_QUADRANTS

    def test_engagement_is_valid_level(self, detector):
        result = detector.assess(_make_eeg(), fs=256.0)
        assert result["engagement_level"] in ENGAGEMENT_LEVELS

    def test_frisson_is_bool(self, detector):
        result = detector.assess(_make_eeg(), fs=256.0)
        assert isinstance(result["frisson_detected"], bool)

    def test_temporal_asymmetry_is_float(self, detector):
        result = detector.assess(_make_eeg(), fs=256.0)
        assert isinstance(result["temporal_asymmetry"], float)


# ---------------------------------------------------------------------------
# TestBaseline
# ---------------------------------------------------------------------------

class TestBaseline:
    """set_baseline stores calibration data and affects subsequent assess()."""

    def test_set_baseline_marks_calibrated(self, detector):
        eeg = _make_eeg(seed=5)
        detector.set_baseline(eeg, fs=256.0)
        assert detector._baseline is not None

    def test_baseline_changes_output(self):
        det_no = MusicEmotionDetector()
        det_yes = MusicEmotionDetector()

        eeg = _make_eeg(seed=20)
        r_no = det_no.assess(eeg, fs=256.0)

        det_yes.set_baseline(_make_eeg(seed=30), fs=256.0)
        r_yes = det_yes.assess(eeg, fs=256.0)

        # At least one metric should differ because baseline normalisation
        # changes the computation
        differs = (
            r_no["music_valence"] != r_yes["music_valence"]
            or r_no["music_arousal"] != r_yes["music_arousal"]
        )
        assert differs

    def test_baseline_stores_band_powers(self, detector):
        detector.set_baseline(_make_eeg(seed=7), fs=256.0)
        assert "alpha" in detector._baseline
        assert "theta" in detector._baseline
        assert "beta" in detector._baseline


# ---------------------------------------------------------------------------
# TestTemporalAsymmetry
# ---------------------------------------------------------------------------

class TestTemporalAsymmetry:
    """Temporal asymmetry (TP9 vs TP10) is the primary music emotion marker."""

    def test_symmetric_near_zero(self, detector):
        eeg = _make_asymmetric_eeg(left_alpha=10.0, right_alpha=10.0)
        result = detector.assess(eeg, fs=256.0)
        assert abs(result["temporal_asymmetry"]) < 0.5

    def test_right_dominant_positive(self, detector):
        eeg = _make_asymmetric_eeg(left_alpha=3.0, right_alpha=20.0)
        result = detector.assess(eeg, fs=256.0)
        assert result["temporal_asymmetry"] > 0.0

    def test_left_dominant_negative(self, detector):
        eeg = _make_asymmetric_eeg(left_alpha=20.0, right_alpha=3.0)
        result = detector.assess(eeg, fs=256.0)
        assert result["temporal_asymmetry"] < 0.0


# ---------------------------------------------------------------------------
# TestMusicQuadrants
# ---------------------------------------------------------------------------

class TestMusicQuadrants:
    """4 music emotion quadrants based on valence and arousal."""

    def test_all_quadrants_exist(self):
        assert len(MUSIC_QUADRANTS) == 4
        assert "energetic_positive" in MUSIC_QUADRANTS
        assert "calm_positive" in MUSIC_QUADRANTS
        assert "energetic_negative" in MUSIC_QUADRANTS
        assert "calm_negative" in MUSIC_QUADRANTS

    def test_high_arousal_positive_valence(self, detector):
        """High beta (arousal) + right-dominant temporal alpha (positive valence)."""
        rng = np.random.RandomState(50)
        n = int(4.0 * 256)
        t = np.arange(n) / 256.0
        eeg = np.zeros((4, n))
        # TP9 (ch0): low alpha -> higher right temporal dominance
        eeg[0] = 3.0 * np.sin(2 * np.pi * 10 * t) + 15.0 * np.sin(2 * np.pi * 22 * t) + rng.randn(n) * 1.5
        eeg[1] = 5.0 * np.sin(2 * np.pi * 10 * t) + 15.0 * np.sin(2 * np.pi * 22 * t) + rng.randn(n) * 1.5
        eeg[2] = 5.0 * np.sin(2 * np.pi * 10 * t) + 15.0 * np.sin(2 * np.pi * 22 * t) + rng.randn(n) * 1.5
        # TP10 (ch3): high alpha
        eeg[3] = 20.0 * np.sin(2 * np.pi * 10 * t) + 15.0 * np.sin(2 * np.pi * 22 * t) + rng.randn(n) * 1.5
        result = detector.assess(eeg, fs=256.0)
        assert result["music_emotion"] == "energetic_positive"

    def test_low_arousal_positive_valence(self, detector):
        """High alpha (calm) + right-dominant temporal alpha (positive valence)."""
        rng = np.random.RandomState(51)
        n = int(4.0 * 256)
        t = np.arange(n) / 256.0
        eeg = np.zeros((4, n))
        # TP9: low alpha
        eeg[0] = 3.0 * np.sin(2 * np.pi * 10 * t) + 2.0 * np.sin(2 * np.pi * 20 * t) + rng.randn(n) * 1.5
        eeg[1] = 15.0 * np.sin(2 * np.pi * 10 * t) + 2.0 * np.sin(2 * np.pi * 20 * t) + rng.randn(n) * 1.5
        eeg[2] = 15.0 * np.sin(2 * np.pi * 10 * t) + 2.0 * np.sin(2 * np.pi * 20 * t) + rng.randn(n) * 1.5
        # TP10: high alpha
        eeg[3] = 20.0 * np.sin(2 * np.pi * 10 * t) + 2.0 * np.sin(2 * np.pi * 20 * t) + rng.randn(n) * 1.5
        result = detector.assess(eeg, fs=256.0)
        assert result["music_emotion"] == "calm_positive"


# ---------------------------------------------------------------------------
# TestFrisson
# ---------------------------------------------------------------------------

class TestFrisson:
    """Frisson detection: frontal theta burst + alpha drop = musical chills."""

    def test_frisson_detected_with_theta_burst(self, detector):
        eeg = _make_frisson_eeg()
        result = detector.detect_frisson(eeg, fs=256.0)
        assert result["frisson_detected"] is True

    def test_frisson_score_range(self, detector):
        eeg = _make_frisson_eeg()
        result = detector.detect_frisson(eeg, fs=256.0)
        assert 0.0 <= result["frisson_score"] <= 1.0

    def test_no_frisson_with_normal_eeg(self, detector):
        eeg = _make_eeg(alpha_amp=10.0, theta_amp=3.0, seed=88)
        result = detector.detect_frisson(eeg, fs=256.0)
        assert result["frisson_detected"] is False

    def test_frisson_in_assess(self, detector):
        eeg = _make_frisson_eeg()
        result = detector.assess(eeg, fs=256.0)
        assert "frisson_detected" in result

    def test_detect_frisson_output_keys(self, detector):
        eeg = _make_eeg()
        result = detector.detect_frisson(eeg, fs=256.0)
        assert "frisson_detected" in result
        assert "frisson_score" in result
        assert "frontal_theta_ratio" in result


# ---------------------------------------------------------------------------
# TestEngagement
# ---------------------------------------------------------------------------

class TestEngagement:
    """Engagement level from alpha/beta ratio: passive, moderate, deep."""

    def test_all_levels_exist(self):
        assert len(ENGAGEMENT_LEVELS) == 3
        assert "passive" in ENGAGEMENT_LEVELS
        assert "moderate" in ENGAGEMENT_LEVELS
        assert "deep" in ENGAGEMENT_LEVELS

    def test_high_beta_deep_engagement(self, detector):
        eeg = _make_eeg(alpha_amp=2.0, beta_amp=20.0)
        result = detector.assess(eeg, fs=256.0)
        assert result["engagement_level"] in ("moderate", "deep")

    def test_high_alpha_passive(self, detector):
        eeg = _make_eeg(alpha_amp=25.0, beta_amp=2.0)
        result = detector.assess(eeg, fs=256.0)
        assert result["engagement_level"] == "passive"


# ---------------------------------------------------------------------------
# TestSessionStats
# ---------------------------------------------------------------------------

class TestSessionStats:
    """get_session_stats() returns aggregated statistics."""

    def test_empty_session(self, detector):
        stats = detector.get_session_stats()
        assert stats["n_assessments"] == 0

    def test_stats_after_assessments(self, detector):
        for i in range(5):
            detector.assess(_make_eeg(seed=i), fs=256.0)
        stats = detector.get_session_stats()
        assert stats["n_assessments"] == 5
        assert "mean_valence" in stats
        assert "mean_arousal" in stats
        assert "frisson_count" in stats
        assert "dominant_quadrant" in stats

    def test_mean_valence_range(self, detector):
        for i in range(3):
            detector.assess(_make_eeg(seed=i + 100), fs=256.0)
        stats = detector.get_session_stats()
        assert -1.0 <= stats["mean_valence"] <= 1.0

    def test_mean_arousal_range(self, detector):
        for i in range(3):
            detector.assess(_make_eeg(seed=i + 200), fs=256.0)
        stats = detector.get_session_stats()
        assert 0.0 <= stats["mean_arousal"] <= 1.0

    def test_dominant_quadrant_valid(self, detector):
        for i in range(5):
            detector.assess(_make_eeg(seed=i + 300), fs=256.0)
        stats = detector.get_session_stats()
        assert stats["dominant_quadrant"] in MUSIC_QUADRANTS


# ---------------------------------------------------------------------------
# TestHistory
# ---------------------------------------------------------------------------

class TestHistory:
    """get_history() returns per-assessment records."""

    def test_empty_history(self, detector):
        assert detector.get_history() == []

    def test_history_grows(self, detector):
        for i in range(4):
            detector.assess(_make_eeg(seed=i + 400), fs=256.0)
        history = detector.get_history()
        assert len(history) == 4

    def test_history_entry_keys(self, detector):
        detector.assess(_make_eeg(), fs=256.0)
        entry = detector.get_history()[0]
        assert "music_valence" in entry
        assert "music_arousal" in entry
        assert "music_emotion" in entry


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------

class TestReset:
    """reset() clears all state."""

    def test_reset_clears_history(self, detector):
        for i in range(3):
            detector.assess(_make_eeg(seed=i), fs=256.0)
        detector.reset()
        assert len(detector.get_history()) == 0

    def test_reset_clears_baseline(self, calibrated_detector):
        calibrated_detector.reset()
        assert calibrated_detector._baseline is None

    def test_reset_clears_stats(self, detector):
        detector.assess(_make_eeg(), fs=256.0)
        detector.reset()
        stats = detector.get_session_stats()
        assert stats["n_assessments"] == 0


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: single-channel, zeros, short signals."""

    def test_single_channel_input(self, detector):
        eeg_1d = np.random.randn(1024) * 10
        result = detector.assess(eeg_1d, fs=256.0)
        assert result["music_emotion"] in MUSIC_QUADRANTS
        # temporal_asymmetry should be 0 without multiple channels
        assert result["temporal_asymmetry"] == 0.0

    def test_two_channel_input(self, detector):
        eeg_2ch = np.random.randn(2, 1024) * 10
        result = detector.assess(eeg_2ch, fs=256.0)
        assert result["music_emotion"] in MUSIC_QUADRANTS

    def test_zeros_input(self, detector):
        eeg = np.zeros((4, 1024))
        result = detector.assess(eeg, fs=256.0)
        assert result["music_emotion"] in MUSIC_QUADRANTS
        assert 0.0 <= result["music_arousal"] <= 1.0

    def test_short_signal(self, detector):
        """Signal shorter than optimal 4-sec window should not crash."""
        eeg = np.random.randn(4, 128) * 10  # 0.5 seconds
        result = detector.assess(eeg, fs=256.0)
        assert result["music_emotion"] in MUSIC_QUADRANTS

    def test_set_baseline_with_short_signal(self, detector):
        eeg = np.random.randn(4, 128) * 10
        detector.set_baseline(eeg, fs=256.0)
        assert detector._baseline is not None
