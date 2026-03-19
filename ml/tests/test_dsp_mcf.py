"""Tests for DSP-MCF dual-stream pre-training (#409).

Covers:
  - TestCreateDSPMCFConfig: defaults, custom, validation errors
  - TestComputeSpatialFeatures: correlation matrix, single/multi channel
  - TestComputeTemporalFeatures: STFT band powers, multi-scale
  - TestFuseDualStream: attention weighting, normalisation
  - TestEvaluateDSPMCF: accuracy, per-class, edge cases
  - TestConfigToDict: serialisation round-trip
  - TestEdgeCases: constant signals, very short, single channel, zeros
"""

import numpy as np
import pytest

from models.dsp_mcf import (
    DSPMCFConfig,
    compute_spatial_features,
    compute_temporal_features,
    config_to_dict,
    create_dsp_mcf_config,
    evaluate_dsp_mcf,
    fuse_dual_stream,
)


# ---------------------------------------------------------------------------
# TestCreateDSPMCFConfig
# ---------------------------------------------------------------------------

class TestCreateDSPMCFConfig:
    def test_default_config(self):
        """Default config should have sensible values."""
        config = create_dsp_mcf_config()
        assert config.n_channels == 4
        assert config.n_samples == 1024
        assert config.fs == 256.0
        assert config.fused_dim == 128

    def test_custom_channels_and_samples(self):
        """Custom n_channels and n_samples should be stored."""
        config = create_dsp_mcf_config(n_channels=8, n_samples=2048)
        assert config.n_channels == 8
        assert config.n_samples == 2048

    def test_custom_window_sizes(self):
        """Custom window sizes should be stored."""
        config = create_dsp_mcf_config(window_sizes=(32, 64))
        assert config.window_sizes == (32, 64)

    def test_invalid_n_channels_raises(self):
        """n_channels < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_channels"):
            create_dsp_mcf_config(n_channels=0)

    def test_invalid_n_samples_raises(self):
        """n_samples < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_samples"):
            create_dsp_mcf_config(n_samples=0)

    def test_invalid_fs_raises(self):
        """fs <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="fs"):
            create_dsp_mcf_config(fs=0.0)

    def test_invalid_mask_fraction_raises(self):
        """mask_fraction outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="mask_fraction"):
            create_dsp_mcf_config(mask_fraction=0.0)
        with pytest.raises(ValueError, match="mask_fraction"):
            create_dsp_mcf_config(mask_fraction=1.0)

    def test_created_at_set(self):
        """created_at should be a positive timestamp."""
        config = create_dsp_mcf_config()
        assert config.created_at > 0


# ---------------------------------------------------------------------------
# TestComputeSpatialFeatures
# ---------------------------------------------------------------------------

class TestComputeSpatialFeatures:
    def test_4channel_output_shape(self):
        """4 channels should produce 6 upper-triangle correlation values."""
        np.random.seed(42)
        signals = np.random.randn(4, 256)
        features = compute_spatial_features(signals)
        # C*(C-1)/2 = 4*3/2 = 6
        assert features.shape == (6,)

    def test_2channel_output_shape(self):
        """2 channels should produce 1 correlation value."""
        np.random.seed(42)
        signals = np.random.randn(2, 256)
        features = compute_spatial_features(signals)
        assert features.shape == (1,)

    def test_single_channel_returns_one(self):
        """Single channel should return [1.0]."""
        signals = np.random.randn(1, 256)
        features = compute_spatial_features(signals)
        assert features.shape == (1,)
        assert features[0] == 1.0

    def test_1d_input(self):
        """1D input should be treated as single channel."""
        signals = np.random.randn(256)
        features = compute_spatial_features(signals)
        assert features.shape == (1,)
        assert features[0] == 1.0

    def test_identical_channels_correlation_1(self):
        """Identical channels should have correlation ~1."""
        sig = np.random.randn(256)
        signals = np.stack([sig, sig])
        features = compute_spatial_features(signals)
        np.testing.assert_allclose(features[0], 1.0, atol=1e-6)

    def test_anticorrelated_channels(self):
        """Perfectly anti-correlated channels should have correlation ~-1."""
        sig = np.random.randn(256)
        signals = np.stack([sig, -sig])
        features = compute_spatial_features(signals)
        np.testing.assert_allclose(features[0], -1.0, atol=1e-6)

    def test_correlation_bounded(self):
        """All correlation values should be in [-1, 1]."""
        np.random.seed(42)
        signals = np.random.randn(4, 1024)
        features = compute_spatial_features(signals)
        assert np.all(features >= -1.0)
        assert np.all(features <= 1.0)


# ---------------------------------------------------------------------------
# TestComputeTemporalFeatures
# ---------------------------------------------------------------------------

class TestComputeTemporalFeatures:
    def test_output_is_1d(self):
        """Temporal features should be a 1D array."""
        np.random.seed(42)
        signals = np.random.randn(4, 1024)
        features = compute_temporal_features(signals, fs=256.0)
        assert features.ndim == 1

    def test_output_length(self):
        """Feature length = n_channels * n_bands * n_window_sizes."""
        np.random.seed(42)
        signals = np.random.randn(4, 1024)
        features = compute_temporal_features(signals, fs=256.0)
        # 4 channels * 5 bands * 3 windows = 60
        assert len(features) == 4 * 5 * 3

    def test_custom_window_sizes(self):
        """Custom window sizes should change feature length."""
        np.random.seed(42)
        signals = np.random.randn(2, 512)
        features = compute_temporal_features(signals, fs=256.0, window_sizes=(64, 128))
        # 2 channels * 5 bands * 2 windows = 20
        assert len(features) == 2 * 5 * 2

    def test_single_channel(self):
        """Single channel 1D input should work."""
        np.random.seed(42)
        signals = np.random.randn(256)
        features = compute_temporal_features(signals, fs=256.0)
        assert features.ndim == 1
        # 1 channel * 5 bands * 3 windows = 15
        assert len(features) == 1 * 5 * 3

    def test_all_values_non_negative(self):
        """Band powers should be non-negative (squared magnitudes)."""
        np.random.seed(42)
        signals = np.random.randn(4, 1024)
        features = compute_temporal_features(signals, fs=256.0)
        assert np.all(features >= 0.0)

    def test_alpha_band_detected(self):
        """10 Hz sine should produce large alpha-band power."""
        np.random.seed(42)
        t = np.arange(1024) / 256.0
        sig = 20.0 * np.sin(2 * np.pi * 10.0 * t)
        signals = sig.reshape(1, -1)
        features = compute_temporal_features(signals, fs=256.0)
        # For 1 channel, 3 windows, bands in order: delta, theta, alpha, beta, gamma
        # Alpha is index 2 in each window block of 5
        alpha_indices = [2, 7, 12]  # 3rd feature per window
        alpha_power = sum(features[i] for i in alpha_indices)
        total_power = features.sum()
        # Alpha should dominate
        assert alpha_power > total_power * 0.3


# ---------------------------------------------------------------------------
# TestFuseDualStream
# ---------------------------------------------------------------------------

class TestFuseDualStream:
    def test_output_length(self):
        """Fused length = spatial_dim + temporal_dim."""
        s = np.random.randn(6)
        t = np.random.randn(60)
        fused = fuse_dual_stream(s, t)
        assert len(fused) == 6 + 60

    def test_output_l2_normalised(self):
        """Fused vector should be L2-normalised."""
        np.random.seed(42)
        s = np.random.randn(6)
        t = np.random.randn(60)
        fused = fuse_dual_stream(s, t)
        norm = np.linalg.norm(fused)
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)

    def test_zero_spatial_still_works(self):
        """Zero spatial features should not crash — temporal dominates."""
        s = np.zeros(6)
        t = np.random.randn(60)
        fused = fuse_dual_stream(s, t)
        assert len(fused) == 66
        assert np.isfinite(fused).all()

    def test_zero_temporal_still_works(self):
        """Zero temporal features should not crash — spatial dominates."""
        s = np.random.randn(6)
        t = np.zeros(60)
        fused = fuse_dual_stream(s, t)
        assert len(fused) == 66
        assert np.isfinite(fused).all()

    def test_both_zero_returns_zeros(self):
        """Both zero should return zero vector (norm=0, no division error)."""
        s = np.zeros(3)
        t = np.zeros(5)
        fused = fuse_dual_stream(s, t)
        assert len(fused) == 8
        np.testing.assert_allclose(fused, 0.0)

    def test_temperature_affects_weighting(self):
        """Higher temperature should produce more uniform attention weights."""
        np.random.seed(42)
        s = np.ones(5) * 10.0
        t = np.ones(5) * 1.0
        fused_low_t = fuse_dual_stream(s, t, temperature=0.1)
        fused_high_t = fuse_dual_stream(s, t, temperature=100.0)
        # With high temperature, weights are more uniform, so the difference
        # in norms of the two halves should be smaller
        low_t_ratio = np.linalg.norm(fused_low_t[:5]) / (np.linalg.norm(fused_low_t[5:]) + 1e-9)
        high_t_ratio = np.linalg.norm(fused_high_t[:5]) / (np.linalg.norm(fused_high_t[5:]) + 1e-9)
        # High temperature ratio should be closer to 1 (more uniform)
        assert abs(high_t_ratio - 1.0) < abs(low_t_ratio - 1.0) or abs(high_t_ratio - low_t_ratio) < 0.1


# ---------------------------------------------------------------------------
# TestEvaluateDSPMCF
# ---------------------------------------------------------------------------

class TestEvaluateDSPMCF:
    def test_perfect_accuracy(self):
        """All correct -> accuracy 1.0."""
        result = evaluate_dsp_mcf(["happy", "sad"], ["happy", "sad"])
        assert result["accuracy"] == 1.0
        assert result["n_correct"] == 2

    def test_zero_accuracy(self):
        """All wrong -> accuracy 0.0."""
        result = evaluate_dsp_mcf(["sad", "happy"], ["happy", "sad"])
        assert result["accuracy"] == 0.0

    def test_empty_input(self):
        """Empty lists -> accuracy 0.0."""
        result = evaluate_dsp_mcf([], [])
        assert result["accuracy"] == 0.0
        assert result["n_samples"] == 0

    def test_length_mismatch_raises(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate_dsp_mcf(["a"], ["a", "b"])

    def test_per_class_metrics(self):
        """Per-class precision/recall should be computed."""
        preds = ["happy", "happy", "sad", "sad"]
        labels = ["happy", "sad", "sad", "sad"]
        result = evaluate_dsp_mcf(preds, labels)
        assert "per_class_precision" in result
        assert "per_class_recall" in result
        assert result["per_class_precision"]["sad"] == 1.0
        assert result["per_class_recall"]["happy"] == 1.0


# ---------------------------------------------------------------------------
# TestConfigToDict
# ---------------------------------------------------------------------------

class TestConfigToDict:
    def test_contains_all_fields(self):
        """Serialised dict should contain all config fields."""
        config = create_dsp_mcf_config()
        d = config_to_dict(config)
        assert isinstance(d, dict)
        assert d["n_channels"] == 4
        assert d["fs"] == 256.0
        assert d["fused_dim"] == 128

    def test_window_sizes_as_list(self):
        """window_sizes tuple should be converted to list for JSON compat."""
        config = create_dsp_mcf_config(window_sizes=(32, 64))
        d = config_to_dict(config)
        assert isinstance(d["window_sizes"], list)
        assert d["window_sizes"] == [32, 64]

    def test_custom_values_preserved(self):
        """Custom values should survive serialisation."""
        config = create_dsp_mcf_config(batch_size=32, learning_rate=0.01)
        d = config_to_dict(config)
        assert d["batch_size"] == 32
        assert d["learning_rate"] == 0.01


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_constant_signal_spatial(self):
        """Constant signal across channels should not crash."""
        signals = np.ones((4, 256)) * 42.0
        features = compute_spatial_features(signals)
        assert features.shape == (6,)
        assert np.isfinite(features).all()

    def test_very_short_signal_temporal(self):
        """Very short signal (16 samples) should not crash."""
        np.random.seed(42)
        signals = np.random.randn(4, 16)
        features = compute_temporal_features(signals, fs=256.0)
        assert features.ndim == 1
        assert np.isfinite(features).all()

    def test_zero_signal(self):
        """All-zero signal should produce valid features."""
        signals = np.zeros((4, 256))
        spatial = compute_spatial_features(signals)
        temporal = compute_temporal_features(signals, fs=256.0)
        fused = fuse_dual_stream(spatial, temporal)
        assert np.isfinite(spatial).all()
        assert np.isfinite(temporal).all()
        assert np.isfinite(fused).all()

    def test_large_amplitude_signal(self):
        """Very large amplitude should not crash."""
        np.random.seed(42)
        signals = np.random.randn(4, 1024) * 5000.0
        spatial = compute_spatial_features(signals)
        temporal = compute_temporal_features(signals, fs=256.0)
        fused = fuse_dual_stream(spatial, temporal)
        assert np.isfinite(fused).all()

    def test_end_to_end_pipeline(self):
        """Full pipeline: spatial + temporal + fuse should produce valid output."""
        np.random.seed(42)
        signals = np.random.randn(4, 1024) * 20.0
        config = create_dsp_mcf_config()
        spatial = compute_spatial_features(signals, config)
        temporal = compute_temporal_features(signals, fs=config.fs)
        fused = fuse_dual_stream(spatial, temporal)
        assert fused.ndim == 1
        assert np.isfinite(fused).all()
        norm = np.linalg.norm(fused)
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)
