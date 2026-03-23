"""Tests for advanced sleep biomarkers (#499) and YASA wrapper (#527)."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSpindleDensity:
    """Tests for compute_spindle_density in sleep_staging.py."""

    def test_returns_zero_for_non_n2_stages(self, sample_eeg, fs):
        from models.sleep_staging import compute_spindle_density

        for stage in ["Wake", "N1", "N3", "REM"]:
            result = compute_spindle_density(sample_eeg, fs, stage)
            assert result == 0.0, f"Expected 0.0 for stage {stage}, got {result}"

    def test_returns_float_for_n2(self, sample_eeg, fs):
        from models.sleep_staging import compute_spindle_density

        result = compute_spindle_density(sample_eeg, fs, "N2")
        assert isinstance(result, float)
        assert result >= 0.0

    def test_handles_short_signal(self, fs):
        from models.sleep_staging import compute_spindle_density

        short = np.random.randn(64) * 20
        result = compute_spindle_density(short, fs, "N2")
        assert isinstance(result, float)
        assert result >= 0.0

    def test_handles_flat_signal(self, flat_signal, fs):
        from models.sleep_staging import compute_spindle_density

        result = compute_spindle_density(flat_signal, fs, "N2")
        assert isinstance(result, float)
        # Flat signal = no spindles
        assert result == 0.0


class TestSOSpindleCoupling:
    """Tests for compute_so_spindle_coupling in sleep_staging.py."""

    def test_returns_zero_for_non_n2_n3_stages(self, sample_eeg, fs):
        from models.sleep_staging import compute_so_spindle_coupling

        for stage in ["Wake", "N1", "REM"]:
            result = compute_so_spindle_coupling(sample_eeg, fs, stage)
            assert result == 0.0, f"Expected 0.0 for stage {stage}, got {result}"

    def test_returns_float_for_n2(self, sample_eeg, fs):
        from models.sleep_staging import compute_so_spindle_coupling

        result = compute_so_spindle_coupling(sample_eeg, fs, "N2")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_returns_float_for_n3(self, sample_eeg, fs):
        from models.sleep_staging import compute_so_spindle_coupling

        result = compute_so_spindle_coupling(sample_eeg, fs, "N3")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_handles_short_signal(self, fs):
        from models.sleep_staging import compute_so_spindle_coupling

        short = np.random.randn(64) * 20
        result = compute_so_spindle_coupling(short, fs, "N2")
        assert result == 0.0  # too short

    def test_handles_very_long_signal(self, fs):
        """Test with a longer signal (30s epoch) typical of sleep staging."""
        from models.sleep_staging import compute_so_spindle_coupling

        epoch_30s = np.random.randn(int(fs * 30)) * 20
        result = compute_so_spindle_coupling(epoch_30s, fs, "N3")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestPredictSequenceBiomarkers:
    """Test that predict_sequence returns biomarker fields."""

    def test_sequence_includes_spindle_density(self, fs):
        from models.sleep_staging import SleepStagingModel

        model = SleepStagingModel()
        # Create 3 synthetic 30s epochs
        epochs = [np.random.randn(int(fs * 30)) * 20 for _ in range(3)]
        results = model.predict_sequence(epochs, fs)

        assert len(results) == 3
        for result in results:
            assert "spindle_density" in result
            assert isinstance(result["spindle_density"], float)
            assert result["spindle_density"] >= 0.0

    def test_sequence_includes_so_coupling(self, fs):
        from models.sleep_staging import SleepStagingModel

        model = SleepStagingModel()
        epochs = [np.random.randn(int(fs * 30)) * 20 for _ in range(3)]
        results = model.predict_sequence(epochs, fs)

        for result in results:
            assert "so_spindle_coupling" in result
            assert isinstance(result["so_spindle_coupling"], float)
            assert 0.0 <= result["so_spindle_coupling"] <= 1.0


class TestYasaSleepWrapper:
    """Tests for the YASA sleep staging wrapper (#527)."""

    def test_detect_spindles_returns_valid_dict(self, sample_eeg, fs):
        from processing.yasa_sleep import detect_spindles

        result = detect_spindles(sample_eeg, fs)
        assert isinstance(result, dict)
        assert "spindles_detected" in result
        assert "count" in result
        assert "density" in result
        assert "method" in result
        assert isinstance(result["count"], int)
        assert result["count"] >= 0
        assert result["method"] in ("yasa", "fallback", "none")

    def test_detect_spindles_too_short_signal(self, fs):
        from processing.yasa_sleep import detect_spindles

        short = np.random.randn(10)
        result = detect_spindles(short, fs)
        assert result["method"] == "none"
        assert result["count"] == 0

    def test_detect_spindles_empty_array(self, fs):
        from processing.yasa_sleep import detect_spindles

        result = detect_spindles(np.array([]), fs)
        assert result["method"] == "none"

    def test_detect_slow_oscillations_returns_valid_dict(self, sample_eeg, fs):
        from processing.yasa_sleep import detect_slow_oscillations

        result = detect_slow_oscillations(sample_eeg, fs)
        assert isinstance(result, dict)
        assert "so_detected" in result
        assert "count" in result
        assert "density" in result
        assert "method" in result

    def test_detect_slow_oscillations_too_short(self, fs):
        from processing.yasa_sleep import detect_slow_oscillations

        short = np.random.randn(10)
        result = detect_slow_oscillations(short, fs)
        assert result["method"] == "none"
        assert result["count"] == 0

    def test_advanced_sleep_staging_returns_structure(self, fs):
        from processing.yasa_sleep import advanced_sleep_staging

        # Create a long enough signal (at least 2 epochs of 30s)
        eeg = np.random.randn(int(fs * 120)) * 20  # 2 minutes
        result = advanced_sleep_staging(eeg, fs)

        assert isinstance(result, dict)
        assert "stages" in result
        assert "hypnogram" in result
        assert "n_epochs" in result
        assert "method" in result
        assert "spindle_summary" in result
        assert "so_summary" in result

    def test_advanced_sleep_staging_short_signal(self, fs):
        from processing.yasa_sleep import advanced_sleep_staging

        # Too short for staging
        short = np.random.randn(int(fs * 10)) * 20
        result = advanced_sleep_staging(short, fs)
        assert result["method"] == "none"
        assert result["n_epochs"] == 0

    def test_advanced_sleep_staging_non_array(self, fs):
        from processing.yasa_sleep import advanced_sleep_staging

        result = advanced_sleep_staging(np.array([[1, 2], [3, 4]]), fs)
        assert result["method"] == "none"  # 2D not supported
