"""Tests for CNN-KAN pseudo-RGB feature engineering pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cnn_kan_feature_engineering import (
    BAND_NAMES,
    CHANNEL_NAMES,
    FREQUENCY_BANDS,
    N_BANDS,
    N_CHANNELS,
    N_ROWS,
    compute_de_features,
    compute_evi_features,
    compute_psd_features,
    create_pseudo_rgb,
    feature_stats_to_dict,
    prepare_cnn_kan_input,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def eeg_4ch():
    """4 channels x 4 seconds synthetic EEG at 256 Hz."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((4, 1024)) * 20.0


@pytest.fixture
def eeg_1ch():
    """Single-channel 1-D synthetic EEG."""
    rng = np.random.default_rng(7)
    return rng.standard_normal(1024) * 20.0


@pytest.fixture
def eeg_short():
    """Short signal: 4 channels x 128 samples (0.5 sec)."""
    rng = np.random.default_rng(13)
    return rng.standard_normal((4, 128)) * 20.0


# ── Constants ────────────────────────────────────────────────────────────────


class TestConstants:
    def test_frequency_bands_count(self):
        assert N_BANDS == 5

    def test_frequency_bands_names(self):
        expected = {"delta", "theta", "alpha", "beta", "gamma"}
        assert set(BAND_NAMES) == expected

    def test_channel_names_count(self):
        assert N_CHANNELS == 4

    def test_channel_names_muse2(self):
        assert CHANNEL_NAMES == ["TP9", "AF7", "AF8", "TP10"]

    def test_n_rows_is_channels_times_bands(self):
        assert N_ROWS == N_CHANNELS * N_BANDS


# ── compute_de_features ─────────────────────────────────────────────────────


class TestComputeDEFeatures:
    def test_output_shape_4ch(self, eeg_4ch):
        de = compute_de_features(eeg_4ch, fs=256.0)
        assert de.shape == (4, N_BANDS)

    def test_output_shape_1ch(self, eeg_1ch):
        de = compute_de_features(eeg_1ch, fs=256.0)
        assert de.shape == (1, N_BANDS)

    def test_values_are_finite(self, eeg_4ch):
        de = compute_de_features(eeg_4ch, fs=256.0)
        assert np.all(np.isfinite(de))

    def test_de_not_all_zero(self, eeg_4ch):
        """DE should have nonzero values for realistic EEG."""
        de = compute_de_features(eeg_4ch, fs=256.0)
        assert np.any(de != 0.0)


# ── compute_psd_features ────────────────────────────────────────────────────


class TestComputePSDFeatures:
    def test_output_shape_4ch(self, eeg_4ch):
        psd = compute_psd_features(eeg_4ch, fs=256.0)
        assert psd.shape == (4, N_BANDS)

    def test_output_shape_1ch(self, eeg_1ch):
        psd = compute_psd_features(eeg_1ch, fs=256.0)
        assert psd.shape == (1, N_BANDS)

    def test_values_are_finite(self, eeg_4ch):
        psd = compute_psd_features(eeg_4ch, fs=256.0)
        assert np.all(np.isfinite(psd))


# ── compute_evi_features ────────────────────────────────────────────────────


class TestComputeEVIFeatures:
    def test_output_shape_4ch(self, eeg_4ch):
        evi = compute_evi_features(eeg_4ch, fs=256.0)
        assert evi.shape == (4, N_BANDS)

    def test_values_are_finite(self, eeg_4ch):
        evi = compute_evi_features(eeg_4ch, fs=256.0)
        assert np.all(np.isfinite(evi))


# ── create_pseudo_rgb ───────────────────────────────────────────────────────


class TestCreatePseudoRGB:
    def test_single_window_shape(self, eeg_4ch):
        de = compute_de_features(eeg_4ch)
        psd = compute_psd_features(eeg_4ch)
        evi = compute_evi_features(eeg_4ch)
        image = create_pseudo_rgb(de, psd, evi)
        # (n_ch * n_bands, 1, 3) for single window
        assert image.shape == (N_ROWS, 1, 3)

    def test_multi_window_shape(self, eeg_4ch):
        """Multiple time windows should produce (n_rows, n_windows, 3)."""
        de1 = compute_de_features(eeg_4ch)
        de2 = compute_de_features(eeg_4ch)
        psd1 = compute_psd_features(eeg_4ch)
        psd2 = compute_psd_features(eeg_4ch)
        evi1 = compute_evi_features(eeg_4ch)
        evi2 = compute_evi_features(eeg_4ch)

        de_stack = np.stack([de1, de2], axis=0)  # (2, 4, 5)
        psd_stack = np.stack([psd1, psd2], axis=0)
        evi_stack = np.stack([evi1, evi2], axis=0)

        image = create_pseudo_rgb(de_stack, psd_stack, evi_stack)
        assert image.shape == (N_ROWS, 2, 3)

    def test_pixel_values_normalized(self, eeg_4ch):
        """All pixel values should be in [0, 1]."""
        de = compute_de_features(eeg_4ch)
        psd = compute_psd_features(eeg_4ch)
        evi = compute_evi_features(eeg_4ch)
        image = create_pseudo_rgb(de, psd, evi)
        assert np.all(image >= 0.0)
        assert np.all(image <= 1.0)

    def test_three_channels(self, eeg_4ch):
        """Last dimension must be 3 (R, G, B)."""
        de = compute_de_features(eeg_4ch)
        psd = compute_psd_features(eeg_4ch)
        evi = compute_evi_features(eeg_4ch)
        image = create_pseudo_rgb(de, psd, evi)
        assert image.shape[-1] == 3


# ── prepare_cnn_kan_input ───────────────────────────────────────────────────


class TestPrepareCNNKANInput:
    def test_returns_dict(self, eeg_4ch):
        result = prepare_cnn_kan_input(eeg_4ch, fs=256.0)
        assert isinstance(result, dict)

    def test_required_keys(self, eeg_4ch):
        result = prepare_cnn_kan_input(eeg_4ch, fs=256.0)
        required = {"image", "n_rows", "n_windows", "n_channels", "n_bands", "shape"}
        assert required.issubset(result.keys())

    def test_image_shape_valid(self, eeg_4ch):
        result = prepare_cnn_kan_input(eeg_4ch, fs=256.0)
        image = result["image"]
        assert image.ndim == 3
        assert image.shape[0] == N_ROWS  # channels * bands
        assert image.shape[2] == 3  # RGB

    def test_single_window_for_exact_4sec(self, eeg_4ch):
        """4 sec signal with 4 sec window should produce 1 window."""
        result = prepare_cnn_kan_input(eeg_4ch, fs=256.0, window_seconds=4.0, overlap=0.0)
        assert result["n_windows"] == 1

    def test_short_signal_handled(self, eeg_short):
        """Signal shorter than window should still produce output."""
        result = prepare_cnn_kan_input(eeg_short, fs=256.0, window_seconds=4.0)
        assert result["n_windows"] >= 1
        assert result["image"].ndim == 3

    def test_single_channel_input(self, eeg_1ch):
        """1-D input should be handled gracefully."""
        result = prepare_cnn_kan_input(eeg_1ch, fs=256.0)
        assert result["n_channels"] == 1
        assert result["image"].ndim == 3


# ── feature_stats_to_dict ───────────────────────────────────────────────────


class TestFeatureStatsToDict:
    def test_returns_dict(self, eeg_4ch):
        de = compute_de_features(eeg_4ch)
        psd = compute_psd_features(eeg_4ch)
        evi = compute_evi_features(eeg_4ch)
        stats = feature_stats_to_dict(de, psd, evi)
        assert isinstance(stats, dict)

    def test_has_stat_keys(self, eeg_4ch):
        de = compute_de_features(eeg_4ch)
        psd = compute_psd_features(eeg_4ch)
        evi = compute_evi_features(eeg_4ch)
        stats = feature_stats_to_dict(de, psd, evi)
        for prefix in ("de", "psd", "evi"):
            assert f"{prefix}_mean" in stats
            assert f"{prefix}_std" in stats
            assert f"{prefix}_min" in stats
            assert f"{prefix}_max" in stats

    def test_has_bands_breakdown(self, eeg_4ch):
        de = compute_de_features(eeg_4ch)
        psd = compute_psd_features(eeg_4ch)
        evi = compute_evi_features(eeg_4ch)
        stats = feature_stats_to_dict(de, psd, evi)
        assert "bands" in stats
        for band in BAND_NAMES:
            assert band in stats["bands"]

    def test_all_values_finite(self, eeg_4ch):
        de = compute_de_features(eeg_4ch)
        psd = compute_psd_features(eeg_4ch)
        evi = compute_evi_features(eeg_4ch)
        stats = feature_stats_to_dict(de, psd, evi)
        for key in ("de_mean", "psd_mean", "evi_mean"):
            assert np.isfinite(stats[key])
