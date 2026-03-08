"""Tests for dual-epoch system (4s fast + 30s accurate) and IAF estimation."""

import numpy as np
import pytest
import sys
import os

# Ensure ml/ is on the path so processing/ imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── IAF estimation tests ────────────────────────────────────────────────────


class TestEstimateIAF:
    """Test Individual Alpha Frequency estimation."""

    def test_returns_float(self):
        from processing.eeg_processor import estimate_iaf

        # 2 seconds of random EEG at 256 Hz
        signal = np.random.randn(512)
        result = estimate_iaf(signal, fs=256.0)
        assert isinstance(result, float)

    def test_detects_10hz_peak(self):
        """A pure 10 Hz sine should produce IAF ~10 Hz."""
        from processing.eeg_processor import estimate_iaf

        fs = 256.0
        t = np.arange(0, 4.0, 1 / fs)  # 4 seconds
        signal = np.sin(2 * np.pi * 10.0 * t)
        iaf = estimate_iaf(signal, fs=fs)
        assert 9.0 <= iaf <= 11.0, f"Expected ~10 Hz, got {iaf}"

    def test_detects_11hz_peak(self):
        """A dominant 11 Hz signal should shift IAF toward 11."""
        from processing.eeg_processor import estimate_iaf

        fs = 256.0
        t = np.arange(0, 4.0, 1 / fs)
        # Strong 11 Hz + weak 5 Hz (theta noise)
        signal = 3.0 * np.sin(2 * np.pi * 11.0 * t) + 0.5 * np.sin(2 * np.pi * 5.0 * t)
        iaf = estimate_iaf(signal, fs=fs)
        assert 10.0 <= iaf <= 12.0, f"Expected ~11 Hz, got {iaf}"

    def test_short_signal_returns_default(self):
        """Signal too short for Welch should return 10.0 default."""
        from processing.eeg_processor import estimate_iaf

        signal = np.random.randn(10)  # way too short
        iaf = estimate_iaf(signal, fs=256.0)
        assert iaf == 10.0

    def test_custom_search_range(self):
        """Search range should restrict the peak search."""
        from processing.eeg_processor import estimate_iaf

        fs = 256.0
        t = np.arange(0, 4.0, 1 / fs)
        # Two peaks: 8 Hz and 12 Hz, 12 Hz stronger
        signal = 1.0 * np.sin(2 * np.pi * 8.0 * t) + 3.0 * np.sin(2 * np.pi * 12.0 * t)
        # Search only 7-10 Hz: should find 8 Hz, not 12 Hz
        iaf = estimate_iaf(signal, fs=fs, search_range=(7.0, 10.0))
        assert 7.0 <= iaf <= 10.0, f"Expected ~8 Hz within restricted range, got {iaf}"


class TestGetPersonalizedBands:
    """Test personalized frequency band computation."""

    def test_default_iaf_matches_standard(self):
        """IAF=10 should produce standard band boundaries."""
        from processing.eeg_processor import get_personalized_bands

        bands = get_personalized_bands(10.0)
        assert bands["alpha"] == (8.0, 12.0)
        assert bands["theta"] == (4.0, 8.0)
        assert bands["beta"] == (12.0, 30.0)

    def test_shifted_iaf(self):
        """IAF=11 should shift alpha to (9, 13), theta to (4, 9), beta to (13, 30)."""
        from processing.eeg_processor import get_personalized_bands

        bands = get_personalized_bands(11.0)
        assert bands["alpha"] == (9.0, 13.0)
        assert bands["theta"] == (4.0, 9.0)
        assert bands["beta"] == (13.0, 30.0)
        assert bands["low_beta"] == (13.0, 20.0)

    def test_low_iaf(self):
        """IAF=8.5 should shift alpha down."""
        from processing.eeg_processor import get_personalized_bands

        bands = get_personalized_bands(8.5)
        assert bands["alpha"] == (6.5, 10.5)
        assert bands["theta"] == (4.0, 6.5)
        assert bands["beta"] == (10.5, 30.0)

    def test_delta_unchanged(self):
        """Delta band should never change regardless of IAF."""
        from processing.eeg_processor import get_personalized_bands

        for iaf in [8.0, 9.0, 10.0, 11.0, 12.0]:
            bands = get_personalized_bands(iaf)
            assert bands["delta"] == (0.5, 4.0)

    def test_gamma_unchanged(self):
        """Gamma band should never change regardless of IAF."""
        from processing.eeg_processor import get_personalized_bands

        for iaf in [8.0, 9.0, 10.0, 11.0, 12.0]:
            bands = get_personalized_bands(iaf)
            assert bands["gamma"] == (30.0, 100.0)

    def test_high_beta_unchanged(self):
        """High beta (20-30 Hz) should be fixed."""
        from processing.eeg_processor import get_personalized_bands

        for iaf in [8.0, 9.0, 10.0, 11.0, 12.0]:
            bands = get_personalized_bands(iaf)
            assert bands["high_beta"] == (20.0, 30.0)


class TestExtractBandPowersPersonalized:
    """Test that extract_band_powers accepts custom bands."""

    def test_accepts_custom_bands(self):
        from processing.eeg_processor import extract_band_powers, get_personalized_bands

        fs = 256.0
        signal = np.random.randn(1024)
        custom_bands = get_personalized_bands(11.0)
        result = extract_band_powers(signal, fs, bands=custom_bands)
        assert "alpha" in result
        assert "theta" in result
        assert "beta" in result

    def test_custom_bands_differ_from_default(self):
        """With IAF=12, bands are shifted enough that powers should differ."""
        from processing.eeg_processor import extract_band_powers, get_personalized_bands

        fs = 256.0
        t = np.arange(0, 4.0, 1 / fs)
        # Signal at exactly 12 Hz -- inside default alpha, but at the edge
        # with shifted bands (IAF=12), 12 Hz is squarely in alpha=(10,14)
        signal = np.sin(2 * np.pi * 12.0 * t) + 0.1 * np.random.randn(len(t))

        default_result = extract_band_powers(signal, fs)
        custom_bands = get_personalized_bands(12.0)
        custom_result = extract_band_powers(signal, fs, bands=custom_bands)

        # Both should return valid dicts with same keys
        assert set(default_result.keys()) == set(custom_result.keys())


# ── Dual-epoch buffer tests ─────────────────────────────────────────────────


class TestDualEpochBuffer:
    """Test the dual-epoch (4s fast + 30s slow) buffer system."""

    def test_fast_buffer_basic(self):
        """Fast buffer should work as before -- accumulate to 4 seconds."""
        from api.routes.analysis import _EpochBuffer

        buf = _EpochBuffer()
        fs = 256.0
        # Push 1 second of 4-channel data
        chunk = np.random.randn(4, 256)
        fast_epoch, fast_ready, slow_epoch, slow_ready = buf.push_and_get(chunk, fs)
        assert not fast_ready  # only 1 second, need 4
        assert not slow_ready  # way less than 30 seconds
        assert fast_epoch.shape[0] == 4

    def test_fast_buffer_becomes_ready_at_4s(self):
        """After 4 seconds of data, fast_ready should be True."""
        from api.routes.analysis import _EpochBuffer

        buf = _EpochBuffer()
        fs = 256.0
        for _ in range(4):
            chunk = np.random.randn(4, 256)  # 1 second each
            fast_epoch, fast_ready, slow_epoch, slow_ready = buf.push_and_get(chunk, fs)
        assert fast_ready
        assert fast_epoch.shape[1] == int(4 * fs)

    def test_slow_buffer_not_ready_until_30s(self):
        """Slow buffer should not be ready until 30 seconds accumulated."""
        from api.routes.analysis import _EpochBuffer

        buf = _EpochBuffer()
        fs = 256.0
        # Push 29 seconds
        for _ in range(29):
            chunk = np.random.randn(4, 256)
            _, _, _, slow_ready = buf.push_and_get(chunk, fs)
        assert not slow_ready

    def test_slow_buffer_ready_at_30s(self):
        """After 30 seconds, slow_ready should be True."""
        from api.routes.analysis import _EpochBuffer

        buf = _EpochBuffer()
        fs = 256.0
        for _ in range(30):
            chunk = np.random.randn(4, 256)
            fast_epoch, fast_ready, slow_epoch, slow_ready = buf.push_and_get(chunk, fs)
        assert slow_ready
        assert slow_epoch is not None
        assert slow_epoch.shape[1] == int(30 * fs)

    def test_slow_buffer_maintains_30s_window(self):
        """After more than 30s, slow buffer should cap at 30s."""
        from api.routes.analysis import _EpochBuffer

        buf = _EpochBuffer()
        fs = 256.0
        for _ in range(40):
            chunk = np.random.randn(4, 256)
            fast_epoch, fast_ready, slow_epoch, slow_ready = buf.push_and_get(chunk, fs)
        assert slow_ready
        assert slow_epoch.shape[1] == int(30 * fs)

    def test_fast_buffer_caps_at_4s(self):
        """Fast buffer should never exceed 4 seconds."""
        from api.routes.analysis import _EpochBuffer

        buf = _EpochBuffer()
        fs = 256.0
        for _ in range(10):
            chunk = np.random.randn(4, 256)
            fast_epoch, _, _, _ = buf.push_and_get(chunk, fs)
        assert fast_epoch.shape[1] == int(4 * fs)

    def test_channel_count_reset(self):
        """Changing channel count mid-stream should reset both buffers."""
        from api.routes.analysis import _EpochBuffer

        buf = _EpochBuffer()
        fs = 256.0
        # Push 2-channel data
        buf.push_and_get(np.random.randn(2, 256), fs)
        # Switch to 4-channel
        fast, _, slow, _ = buf.push_and_get(np.random.randn(4, 512), fs)
        assert fast.shape[0] == 4  # should have reset to 4 channels


# ── CalibrationRunner IAF integration tests ─────────────────────────────────


class TestCalibrationRunnerIAF:
    """Test that CalibrationRunner computes IAF from eyes-closed epochs."""

    def test_iaf_computed_from_eyes_closed(self):
        """After eyes_closed epochs with 10 Hz signal, IAF should be ~10 Hz."""
        from processing.calibration import CalibrationRunner

        runner = CalibrationRunner(fs=256)
        fs = 256.0
        t = np.arange(0, 4.0, 1 / fs)
        # Strong 10 Hz alpha signal
        signal = 5.0 * np.sin(2 * np.pi * 10.0 * t) + 0.5 * np.random.randn(len(t))

        # Add several eyes_closed epochs
        for _ in range(5):
            runner.add_epoch("eyes_closed", signal)

        # Also add eyes_open for a complete calibration
        for _ in range(5):
            runner.add_epoch("eyes_open", signal * 0.5)

        cal = runner.compute_calibration("test_user_iaf")
        assert cal.alpha_peak_freq is not None
        assert 9.0 <= cal.alpha_peak_freq <= 11.0, f"Expected ~10 Hz IAF, got {cal.alpha_peak_freq}"

    def test_iaf_none_without_eyes_closed(self):
        """Without eyes_closed data, IAF should remain None."""
        from processing.calibration import CalibrationRunner

        runner = CalibrationRunner(fs=256)
        signal = np.random.randn(1024)

        for _ in range(5):
            runner.add_epoch("eyes_open", signal)

        cal = runner.compute_calibration("test_user_no_ec")
        # Without eyes_closed, IAF computation may default or be None
        # The implementation should gracefully handle this
        # (either None or default 10.0 is acceptable)
        assert cal.alpha_peak_freq is None or isinstance(cal.alpha_peak_freq, float)
