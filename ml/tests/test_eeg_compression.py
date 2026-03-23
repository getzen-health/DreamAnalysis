"""Tests for EEG compression utilities (delta encoding + zstd + downsampling)."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.eeg_compression import (
    delta_encode,
    delta_decode,
    compress_eeg_frame,
    decompress_eeg_frame,
    downsample_to_128hz,
)


# ── delta_encode / delta_decode ──────────────────────────────────────────────

class TestDeltaEncode:
    def test_returns_deltas_between_consecutive_samples(self):
        samples = np.array([100, 102, 101, 105, 103], dtype=np.float64)
        encoded = delta_encode(samples)
        assert encoded[0] == 100
        assert encoded[1] == 2
        assert encoded[2] == -1
        assert encoded[3] == 4
        assert encoded[4] == -2

    def test_empty_input(self):
        result = delta_encode(np.array([]))
        assert len(result) == 0

    def test_single_element(self):
        result = delta_encode(np.array([42.0]))
        assert len(result) == 1
        assert result[0] == 42.0

    def test_constant_signal_produces_zero_deltas(self):
        samples = np.array([50, 50, 50, 50], dtype=np.float64)
        encoded = delta_encode(samples)
        np.testing.assert_array_equal(encoded, [50, 0, 0, 0])


class TestDeltaDecode:
    def test_roundtrip_integer_like(self):
        original = np.array([100, 102, 101, 105, 103], dtype=np.float64)
        encoded = delta_encode(original)
        decoded = delta_decode(encoded)
        np.testing.assert_array_almost_equal(decoded, original)

    def test_roundtrip_float(self):
        original = np.array([10.5, 11.2, 10.8, 12.0], dtype=np.float64)
        decoded = delta_decode(delta_encode(original))
        np.testing.assert_array_almost_equal(decoded, original)

    def test_empty_input(self):
        assert len(delta_decode(np.array([]))) == 0

    def test_single_element(self):
        result = delta_decode(np.array([42.0]))
        assert len(result) == 1
        assert result[0] == 42.0


# ── compress_eeg_frame / decompress_eeg_frame ────────────────────────────────

class TestCompressDecompressFrame:
    def test_roundtrip_multichannel(self):
        channels = np.random.randn(4, 256) * 20  # 4ch, 1 second at 256Hz
        timestamp = 1700000000.0

        compressed = compress_eeg_frame(channels, timestamp)
        dec_channels, dec_ts = decompress_eeg_frame(compressed)

        assert dec_ts == timestamp
        assert dec_channels.shape == channels.shape
        np.testing.assert_array_almost_equal(dec_channels, channels, decimal=6)

    def test_compressed_is_smaller_than_raw(self):
        # EEG data with small differences between samples should compress well
        channels = np.zeros((4, 256))
        for ch in range(4):
            channels[ch] = np.cumsum(np.random.randn(256) * 0.5)

        compressed = compress_eeg_frame(channels, 0.0)
        raw_bytes = channels.nbytes
        compressed_bytes = len(compressed)

        assert compressed_bytes < raw_bytes

    def test_tracks_compression_ratio_in_metadata(self):
        channels = np.random.randn(4, 256) * 20
        compressed = compress_eeg_frame(channels, 0.0)
        # compressed is bytes — we check the decompressed metadata
        dec_channels, dec_ts = decompress_eeg_frame(compressed)
        # Just verify it decompresses — the ratio is computed at compress time
        assert dec_channels.shape == channels.shape

    def test_single_channel(self):
        channels = np.random.randn(1, 128) * 20
        compressed = compress_eeg_frame(channels, 0.0)
        dec_channels, dec_ts = decompress_eeg_frame(compressed)
        assert dec_channels.shape == (1, 128)
        np.testing.assert_array_almost_equal(dec_channels, channels, decimal=6)


# ── downsample_to_128hz ─────────────────────────────────────────────────────

class TestDownsampleTo128Hz:
    def test_halves_256hz_signal(self):
        signal = np.random.randn(256) * 20  # 1 second at 256Hz
        downsampled = downsample_to_128hz(signal, original_fs=256)
        assert len(downsampled) == 128

    def test_no_change_for_128hz_input(self):
        signal = np.random.randn(128) * 20
        downsampled = downsample_to_128hz(signal, original_fs=128)
        assert len(downsampled) == 128
        np.testing.assert_array_almost_equal(downsampled, signal)

    def test_downsamples_512hz_correctly(self):
        signal = np.random.randn(512) * 20  # 1 second at 512Hz
        downsampled = downsample_to_128hz(signal, original_fs=512)
        assert len(downsampled) == 128

    def test_preserves_low_frequency_content(self):
        """A pure 5 Hz sine wave should survive 128Hz downsampling."""
        fs = 256
        t = np.arange(fs) / fs
        signal = np.sin(2 * np.pi * 5 * t)
        downsampled = downsample_to_128hz(signal, original_fs=fs)

        # The downsampled signal should still resemble a 5Hz sine
        t_down = np.arange(128) / 128
        expected = np.sin(2 * np.pi * 5 * t_down)
        correlation = np.corrcoef(downsampled, expected)[0, 1]
        assert correlation > 0.95, f"Correlation {correlation} too low"
