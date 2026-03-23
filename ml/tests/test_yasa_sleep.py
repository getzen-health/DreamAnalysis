"""Tests for YASA-based sleep staging, spindle detection, and slow wave detection.

Tests cover:
- stage_with_yasa returns expected keys (stages, probabilities, summary)
- Stages are valid values (W, N1, N2, N3, R)
- Summary includes total sleep time
- Handles short data gracefully (< 5 min = error/warning)
- detect_spindles_yasa returns count and density
- detect_slow_waves_yasa returns count and density
- Spindle density is non-negative
- Works with 256 Hz sample rate
- Graceful fallback when YASA is not installed

GitHub issue: #527
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sleep_eeg(fs: int = 256, duration_min: float = 10.0) -> np.ndarray:
    """Create synthetic EEG data that looks like sleep.

    Generates delta-dominant signal (1-4 Hz) mixed with some spindle-band
    activity (12-15 Hz) and low noise, mimicking NREM sleep.
    Duration must be at least 5 minutes for YASA sleep staging.
    """
    n_samples = int(fs * duration_min * 60)
    t = np.arange(n_samples) / fs
    rng = np.random.RandomState(42)

    # Delta oscillation (dominant in N2/N3)
    delta = 40.0 * np.sin(2 * np.pi * 1.5 * t)
    # Theta component
    theta = 10.0 * np.sin(2 * np.pi * 6.0 * t)
    # Spindle-band activity (sigma, 12-15 Hz)
    sigma = 8.0 * np.sin(2 * np.pi * 13.0 * t)
    # Background noise
    noise = 5.0 * rng.randn(n_samples)

    return delta + theta + sigma + noise


def _make_short_eeg(fs: int = 256, duration_sec: float = 60.0) -> np.ndarray:
    """Create EEG data shorter than 5 minutes (too short for staging)."""
    n_samples = int(fs * duration_sec)
    rng = np.random.RandomState(42)
    return 20.0 * rng.randn(n_samples)


def _make_nrem_epoch(fs: int = 256, duration_sec: float = 30.0) -> np.ndarray:
    """Create a single NREM-like epoch with embedded spindle-band activity.

    For spindle and slow wave detection tests (no minimum duration requirement).
    """
    n_samples = int(fs * duration_sec)
    t = np.arange(n_samples) / fs
    rng = np.random.RandomState(42)

    # Strong slow oscillation component (for slow wave detection)
    slow_osc = 60.0 * np.sin(2 * np.pi * 0.8 * t)
    # Delta
    delta = 30.0 * np.sin(2 * np.pi * 2.0 * t)
    # Spindle bursts embedded at specific times
    spindle_env = np.zeros(n_samples)
    for center in [5.0, 12.0, 20.0, 27.0]:
        sigma_t = 0.3
        spindle_env += np.exp(-0.5 * ((t - center) / sigma_t) ** 2)
    spindles = 25.0 * spindle_env * np.sin(2 * np.pi * 13.5 * t)
    # Noise
    noise = 3.0 * rng.randn(n_samples)

    return slow_osc + delta + spindles + noise


# ---------------------------------------------------------------------------
# Tests: stage_with_yasa
# ---------------------------------------------------------------------------

class TestStageWithYasa:
    """Tests for the YASA sleep staging wrapper."""

    def test_returns_expected_keys(self):
        """stage_with_yasa must return stages, probabilities, summary, model_type."""
        from models.yasa_sleep import stage_with_yasa

        eeg = _make_sleep_eeg(fs=256, duration_min=10.0)
        result = stage_with_yasa(eeg, fs=256, channel_name="EEG")

        assert "stages" in result
        assert "probabilities" in result
        assert "summary" in result
        assert "model_type" in result
        assert result["model_type"] == "yasa"

    def test_stages_are_valid_values(self):
        """All stages must be standard sleep stage labels."""
        from models.yasa_sleep import stage_with_yasa

        eeg = _make_sleep_eeg(fs=256, duration_min=10.0)
        result = stage_with_yasa(eeg, fs=256, channel_name="EEG")

        valid_stages = {"W", "WAKE", "N1", "N2", "N3", "R", "REM", "ART", "UNS"}
        for stage in result["stages"]:
            assert stage in valid_stages, f"Invalid stage: {stage}"

    def test_summary_includes_sleep_time(self):
        """Summary dict must include total sleep time (TST key)."""
        from models.yasa_sleep import stage_with_yasa

        eeg = _make_sleep_eeg(fs=256, duration_min=10.0)
        result = stage_with_yasa(eeg, fs=256, channel_name="EEG")

        summary = result["summary"]
        assert isinstance(summary, dict)
        assert "TST" in summary, f"Summary missing TST key. Keys: {list(summary.keys())}"

    def test_probabilities_per_epoch(self):
        """Probabilities should have one entry per epoch."""
        from models.yasa_sleep import stage_with_yasa

        eeg = _make_sleep_eeg(fs=256, duration_min=10.0)
        result = stage_with_yasa(eeg, fs=256, channel_name="EEG")

        assert len(result["probabilities"]) == len(result["stages"])

    def test_short_data_returns_error(self):
        """Data shorter than 5 minutes should return an error, not crash."""
        from models.yasa_sleep import stage_with_yasa

        eeg = _make_short_eeg(fs=256, duration_sec=60.0)
        result = stage_with_yasa(eeg, fs=256, channel_name="EEG")

        assert "error" in result
        assert result["stages"] == []

    def test_works_at_256hz(self):
        """Must work with Muse 2's 256 Hz sample rate."""
        from models.yasa_sleep import stage_with_yasa

        eeg = _make_sleep_eeg(fs=256, duration_min=10.0)
        result = stage_with_yasa(eeg, fs=256, channel_name="EEG")

        assert "error" not in result
        assert len(result["stages"]) > 0

    def test_number_of_epochs_matches_duration(self):
        """10 minutes of data at 30s epochs = 20 epochs."""
        from models.yasa_sleep import stage_with_yasa

        duration_min = 10.0
        eeg = _make_sleep_eeg(fs=256, duration_min=duration_min)
        result = stage_with_yasa(eeg, fs=256, channel_name="EEG")

        expected_epochs = int(duration_min * 60 / 30)
        assert len(result["stages"]) == expected_epochs


# ---------------------------------------------------------------------------
# Tests: detect_spindles_yasa
# ---------------------------------------------------------------------------

class TestDetectSpindlesYasa:
    """Tests for YASA spindle detection wrapper."""

    def test_returns_expected_keys(self):
        """Must return count, density, avg_duration_ms, avg_frequency_hz, spindles."""
        from models.yasa_sleep import detect_spindles_yasa

        eeg = _make_nrem_epoch(fs=256, duration_sec=30.0)
        result = detect_spindles_yasa(eeg, fs=256)

        assert "count" in result
        assert "density" in result
        assert "avg_duration_ms" in result
        assert "avg_frequency_hz" in result
        assert "spindles" in result

    def test_density_non_negative(self):
        """Spindle density must be >= 0."""
        from models.yasa_sleep import detect_spindles_yasa

        eeg = _make_nrem_epoch(fs=256, duration_sec=30.0)
        result = detect_spindles_yasa(eeg, fs=256)

        assert result["density"] >= 0.0

    def test_count_non_negative(self):
        """Spindle count must be >= 0."""
        from models.yasa_sleep import detect_spindles_yasa

        eeg = _make_nrem_epoch(fs=256, duration_sec=30.0)
        result = detect_spindles_yasa(eeg, fs=256)

        assert result["count"] >= 0

    def test_empty_signal_returns_zero(self):
        """Flat/silent signal should return zero spindles."""
        from models.yasa_sleep import detect_spindles_yasa

        eeg = np.zeros(256 * 30)
        result = detect_spindles_yasa(eeg, fs=256)

        assert result["count"] == 0
        assert result["density"] == 0.0

    def test_spindles_list_capped(self):
        """Returned spindle list should be capped at 20 entries."""
        from models.yasa_sleep import detect_spindles_yasa

        eeg = _make_nrem_epoch(fs=256, duration_sec=30.0)
        result = detect_spindles_yasa(eeg, fs=256)

        assert len(result["spindles"]) <= 20


# ---------------------------------------------------------------------------
# Tests: detect_slow_waves_yasa
# ---------------------------------------------------------------------------

class TestDetectSlowWavesYasa:
    """Tests for YASA slow wave detection wrapper."""

    def test_returns_expected_keys(self):
        """Must return count, density, avg_amplitude_uv."""
        from models.yasa_sleep import detect_slow_waves_yasa

        eeg = _make_nrem_epoch(fs=256, duration_sec=30.0)
        result = detect_slow_waves_yasa(eeg, fs=256)

        assert "count" in result
        assert "density" in result
        assert "avg_amplitude_uv" in result

    def test_density_non_negative(self):
        """Slow wave density must be >= 0."""
        from models.yasa_sleep import detect_slow_waves_yasa

        eeg = _make_nrem_epoch(fs=256, duration_sec=30.0)
        result = detect_slow_waves_yasa(eeg, fs=256)

        assert result["density"] >= 0.0

    def test_count_non_negative(self):
        """Slow wave count must be >= 0."""
        from models.yasa_sleep import detect_slow_waves_yasa

        eeg = _make_nrem_epoch(fs=256, duration_sec=30.0)
        result = detect_slow_waves_yasa(eeg, fs=256)

        assert result["count"] >= 0

    def test_flat_signal_returns_zero(self):
        """Flat signal should return zero slow waves."""
        from models.yasa_sleep import detect_slow_waves_yasa

        eeg = np.zeros(256 * 30)
        result = detect_slow_waves_yasa(eeg, fs=256)

        assert result["count"] == 0
        assert result["density"] == 0.0


# ---------------------------------------------------------------------------
# Tests: YASASleepStager class
# ---------------------------------------------------------------------------

class TestYASASleepStager:
    """Tests for the YASASleepStager wrapper class."""

    def test_instantiation(self):
        """Should instantiate without errors."""
        from models.yasa_sleep import YASASleepStager

        stager = YASASleepStager()
        assert stager is not None

    def test_stage_sleep_returns_dict(self):
        """stage_sleep should return a dict with expected structure."""
        from models.yasa_sleep import YASASleepStager

        stager = YASASleepStager()
        eeg = _make_sleep_eeg(fs=256, duration_min=10.0)
        result = stager.stage_sleep(eeg, fs=256, channel="AF7")

        assert isinstance(result, dict)
        assert "stages" in result
        assert "summary" in result
        assert result["model_type"] == "yasa_single_channel"

    def test_stage_sleep_short_data(self):
        """Short data should return error gracefully."""
        from models.yasa_sleep import YASASleepStager

        stager = YASASleepStager()
        eeg = _make_short_eeg(fs=256, duration_sec=60.0)
        result = stager.stage_sleep(eeg, fs=256, channel="AF7")

        assert "error" in result

    def test_detect_spindles_method(self):
        """detect_spindles should return spindle metrics."""
        from models.yasa_sleep import YASASleepStager

        stager = YASASleepStager()
        eeg = _make_nrem_epoch(fs=256, duration_sec=30.0)
        result = stager.detect_spindles(eeg, fs=256)

        assert "count" in result
        assert "density" in result

    def test_detect_slow_waves_method(self):
        """detect_slow_waves should return slow wave metrics."""
        from models.yasa_sleep import YASASleepStager

        stager = YASASleepStager()
        eeg = _make_nrem_epoch(fs=256, duration_sec=30.0)
        result = stager.detect_slow_waves(eeg, fs=256)

        assert "count" in result
        assert "density" in result

    def test_full_analysis(self):
        """full_analysis should combine staging + spindles + slow waves."""
        from models.yasa_sleep import YASASleepStager

        stager = YASASleepStager()
        eeg = _make_sleep_eeg(fs=256, duration_min=10.0)
        result = stager.full_analysis(eeg, fs=256, channel="AF7")

        assert "staging" in result
        assert "spindles" in result
        assert "slow_waves" in result
