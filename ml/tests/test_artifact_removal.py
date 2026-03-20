"""Tests for wavelet denoising and adaptive blink filter artifact removal."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestWaveletDenoiseChannel:
    """Tests for wavelet_denoise_channel()."""

    def test_reduces_high_frequency_noise(self):
        """White noise added to a 10 Hz sine wave — SNR should improve."""
        from processing.eeg_processor import wavelet_denoise_channel

        fs = 256
        t = np.arange(0, 4, 1 / fs)  # 4 seconds
        clean = 20.0 * np.sin(2 * np.pi * 10 * t)  # 10 Hz, 20 uV
        noise = np.random.RandomState(42).randn(len(t)) * 10.0  # 10 uV RMS noise
        noisy = clean + noise

        denoised = wavelet_denoise_channel(noisy, fs=fs)

        # SNR should improve: residual error after denoising < original noise
        noise_power_before = np.mean((noisy - clean) ** 2)
        noise_power_after = np.mean((denoised - clean) ** 2)
        assert noise_power_after < noise_power_before, (
            f"Denoising did not reduce noise: before={noise_power_before:.2f}, "
            f"after={noise_power_after:.2f}"
        )

    def test_preserves_alpha_band_signal(self):
        """10 Hz alpha sine + noise — denoised should retain the alpha component."""
        from processing.eeg_processor import wavelet_denoise_channel

        fs = 256
        t = np.arange(0, 4, 1 / fs)
        alpha_signal = 15.0 * np.sin(2 * np.pi * 10 * t)
        noise = np.random.RandomState(99).randn(len(t)) * 8.0
        noisy = alpha_signal + noise

        denoised = wavelet_denoise_channel(noisy, fs=fs)

        # Compute correlation between denoised and original alpha — should be high
        corr = np.corrcoef(denoised, alpha_signal)[0, 1]
        assert corr > 0.7, f"Alpha preservation too low: correlation={corr:.3f}"

    def test_output_same_length_as_input(self):
        """Output must have the same number of samples as input."""
        from processing.eeg_processor import wavelet_denoise_channel

        signal = np.random.RandomState(1).randn(1024)
        denoised = wavelet_denoise_channel(signal, fs=256)
        assert len(denoised) == len(signal)

    def test_handles_all_zero_signal(self):
        """All-zero input should return all zeros (no crash)."""
        from processing.eeg_processor import wavelet_denoise_channel

        signal = np.zeros(1024)
        denoised = wavelet_denoise_channel(signal, fs=256)
        assert len(denoised) == 1024
        assert np.allclose(denoised, 0.0)

    def test_handles_very_short_signal(self):
        """Signal shorter than wavelet decomposition needs — should not crash."""
        from processing.eeg_processor import wavelet_denoise_channel

        signal = np.random.RandomState(7).randn(10)
        denoised = wavelet_denoise_channel(signal, fs=256)
        assert len(denoised) == len(signal)
        assert np.all(np.isfinite(denoised))

    def test_handles_nan_values(self):
        """Signal with NaN values should not produce NaN output."""
        from processing.eeg_processor import wavelet_denoise_channel

        signal = np.random.RandomState(3).randn(1024) * 20.0
        signal[100:105] = np.nan
        signal[500] = np.inf

        denoised = wavelet_denoise_channel(signal, fs=256)
        assert len(denoised) == len(signal)
        assert np.all(np.isfinite(denoised))

    def test_output_is_finite(self):
        """No NaN or inf in output for normal input."""
        from processing.eeg_processor import wavelet_denoise_channel

        signal = np.random.RandomState(5).randn(2048) * 30.0
        denoised = wavelet_denoise_channel(signal, fs=256)
        assert np.all(np.isfinite(denoised))


class TestAdaptiveBlinkFilter:
    """Tests for adaptive_blink_filter()."""

    def test_removes_spike_artifact(self):
        """Inject a 200 uV spike into clean signal — spike should be reduced."""
        from processing.eeg_processor import adaptive_blink_filter

        fs = 256
        t = np.arange(0, 4, 1 / fs)
        clean = 10.0 * np.sin(2 * np.pi * 10 * t)  # background alpha

        # Inject a blink-like spike (~50 ms wide, 200 uV)
        spiked = clean.copy()
        spike_center = 512  # at 2 seconds
        spike_width = int(0.05 * fs)  # 50 ms
        half = spike_width // 2
        spike_region = slice(spike_center - half, spike_center + half)
        spiked[spike_region] += 200.0

        filtered = adaptive_blink_filter(spiked, fs=fs)

        # The spike peak should be significantly reduced
        spike_amplitude_before = np.max(np.abs(spiked[spike_region] - clean[spike_region]))
        spike_amplitude_after = np.max(np.abs(filtered[spike_region] - clean[spike_region]))
        assert spike_amplitude_after < spike_amplitude_before * 0.5, (
            f"Spike not sufficiently removed: before={spike_amplitude_before:.1f}, "
            f"after={spike_amplitude_after:.1f}"
        )

    def test_preserves_clean_signal(self):
        """Clean signal with no artifacts should pass through nearly unchanged."""
        from processing.eeg_processor import adaptive_blink_filter

        fs = 256
        t = np.arange(0, 4, 1 / fs)
        clean = 10.0 * np.sin(2 * np.pi * 10 * t)

        filtered = adaptive_blink_filter(clean, fs=fs)

        # Should be very similar to input (high correlation, low residual)
        corr = np.corrcoef(filtered, clean)[0, 1]
        assert corr > 0.95, f"Clean signal distorted: correlation={corr:.3f}"

    def test_output_same_length(self):
        """Output length must match input."""
        from processing.eeg_processor import adaptive_blink_filter

        signal = np.random.RandomState(2).randn(1024) * 15.0
        filtered = adaptive_blink_filter(signal, fs=256)
        assert len(filtered) == len(signal)

    def test_handles_all_zeros(self):
        """All-zero input should not crash."""
        from processing.eeg_processor import adaptive_blink_filter

        signal = np.zeros(1024)
        filtered = adaptive_blink_filter(signal, fs=256)
        assert len(filtered) == 1024
        assert np.all(np.isfinite(filtered))

    def test_handles_short_signal(self):
        """Very short signal — should return without crash."""
        from processing.eeg_processor import adaptive_blink_filter

        signal = np.random.RandomState(8).randn(20) * 10.0
        filtered = adaptive_blink_filter(signal, fs=256)
        assert len(filtered) == len(signal)
        assert np.all(np.isfinite(filtered))

    def test_handles_nan_values(self):
        """NaN values in input should not produce NaN output."""
        from processing.eeg_processor import adaptive_blink_filter

        signal = np.random.RandomState(4).randn(1024) * 15.0
        signal[200:203] = np.nan

        filtered = adaptive_blink_filter(signal, fs=256)
        assert len(filtered) == len(signal)
        assert np.all(np.isfinite(filtered))


class TestPreprocessWithArtifactRemoval:
    """Verify the full preprocess pipeline still works after adding new steps."""

    def test_preprocess_preserves_length(self):
        """preprocess() output length must equal input length."""
        from processing.eeg_processor import preprocess

        signal = np.random.RandomState(10).randn(1024) * 20.0
        processed = preprocess(signal, fs=256.0)
        assert len(processed) == len(signal)

    def test_preprocess_output_is_finite(self):
        """No NaN or inf in preprocess output."""
        from processing.eeg_processor import preprocess

        signal = np.random.RandomState(11).randn(2048) * 25.0
        processed = preprocess(signal, fs=256.0)
        assert np.all(np.isfinite(processed))

    def test_preprocess_with_nan_input(self):
        """preprocess should handle NaN input (via _sanitize_nan)."""
        from processing.eeg_processor import preprocess

        signal = np.random.RandomState(12).randn(1024) * 20.0
        signal[50:55] = np.nan

        processed = preprocess(signal, fs=256.0)
        assert len(processed) == len(signal)
        assert np.all(np.isfinite(processed))

    def test_preprocess_reduces_noise(self):
        """preprocess pipeline should reduce high-freq noise overall."""
        from processing.eeg_processor import preprocess

        fs = 256
        t = np.arange(0, 4, 1 / fs)
        clean_alpha = 20.0 * np.sin(2 * np.pi * 10 * t)
        hf_noise = 30.0 * np.sin(2 * np.pi * 80 * t)  # 80 Hz — above bandpass cutoff
        noisy = clean_alpha + hf_noise

        processed = preprocess(noisy, fs=float(fs))

        # 80 Hz should be gone after 1-50 Hz bandpass + wavelet denoising
        residual_hf = np.mean((processed - clean_alpha) ** 2)
        original_hf = np.mean(hf_noise ** 2)
        assert residual_hf < original_hf * 0.1, (
            f"High-freq noise not sufficiently removed: "
            f"original={original_hf:.1f}, residual={residual_hf:.1f}"
        )
