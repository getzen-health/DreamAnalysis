"""Tests for AMEEGNet attention-enhanced EEGNet architecture specification."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.amee_gnet import (
    DEFAULT_ATTENTION_REDUCTION_RATIO,
    DEFAULT_DROPOUT,
    DEFAULT_N_CHANNELS,
    DEFAULT_N_CLASSES,
    DEFAULT_TEMPORAL_KERNELS,
    EMOTION_CLASSES,
    compare_architectures,
    compute_attention_weights,
    config_to_dict,
    create_ameegnet_config,
    multi_scale_feature_fusion,
)

# ── create_ameegnet_config ───────────────────────────────────────────────────


class TestCreateAMEEGNetConfig:
    def test_returns_dict(self):
        config = create_ameegnet_config()
        assert isinstance(config, dict)

    def test_default_architecture_name(self):
        config = create_ameegnet_config()
        assert config["architecture"] == "AMEEGNet"

    def test_default_n_channels(self):
        config = create_ameegnet_config()
        assert config["n_channels"] == DEFAULT_N_CHANNELS

    def test_default_n_classes(self):
        config = create_ameegnet_config()
        assert config["n_classes"] == DEFAULT_N_CLASSES

    def test_default_temporal_kernels(self):
        config = create_ameegnet_config()
        assert config["temporal_kernels"] == DEFAULT_TEMPORAL_KERNELS

    def test_n_branches_matches_kernels(self):
        config = create_ameegnet_config()
        assert config["n_branches"] == len(DEFAULT_TEMPORAL_KERNELS)

    def test_custom_temporal_kernels(self):
        kernels = [16, 32]
        config = create_ameegnet_config(temporal_kernels=kernels)
        assert config["temporal_kernels"] == kernels
        assert config["n_branches"] == 2

    def test_has_attention_config(self):
        config = create_ameegnet_config()
        assert "attention" in config
        assert config["attention"]["type"] == "squeeze-excitation"
        assert config["attention"]["reduction_ratio"] == DEFAULT_ATTENTION_REDUCTION_RATIO

    def test_has_estimated_params(self):
        config = create_ameegnet_config()
        assert "estimated_params" in config
        params = config["estimated_params"]
        assert params["total"] > 0
        assert params["branches"] > 0
        assert params["attention"] > 0
        assert params["classifier"] > 0

    def test_params_total_is_sum(self):
        config = create_ameegnet_config()
        params = config["estimated_params"]
        expected = params["branches"] + params["attention"] + params["classifier"]
        assert params["total"] == expected

    def test_has_emotion_classes(self):
        config = create_ameegnet_config()
        assert config["emotion_classes"] == EMOTION_CLASSES

    def test_custom_n_classes(self):
        config = create_ameegnet_config(n_classes=2)
        assert config["n_classes"] == 2
        assert len(config["emotion_classes"]) == 2

    def test_has_description(self):
        config = create_ameegnet_config()
        assert isinstance(config["description"], str)
        assert "AMEEGNet" in config["description"]


# ── compute_attention_weights ────────────────────────────────────────────────


class TestComputeAttentionWeights:
    def test_1d_input_shape(self):
        features = np.random.randn(64)
        weights = compute_attention_weights(features)
        assert weights.shape == (64,)

    def test_2d_input_shape(self):
        features = np.random.randn(8, 64)
        weights = compute_attention_weights(features)
        assert weights.shape == (8, 64)

    def test_values_in_0_1_range(self):
        """Sigmoid output must be in [0, 1]."""
        features = np.random.randn(64) * 10
        weights = compute_attention_weights(features)
        assert np.all(weights >= 0.0)
        assert np.all(weights <= 1.0)

    def test_all_finite(self):
        features = np.random.randn(128)
        weights = compute_attention_weights(features)
        assert np.all(np.isfinite(weights))

    def test_different_inputs_different_weights(self):
        a = np.ones(64) * 5.0
        b = np.ones(64) * -5.0
        w_a = compute_attention_weights(a)
        w_b = compute_attention_weights(b)
        assert not np.allclose(w_a, w_b)


# ── multi_scale_feature_fusion ───────────────────────────────────────────────


class TestMultiScaleFeatureFusion:
    def test_1d_fusion(self):
        branches = [np.ones(32), np.ones(32), np.ones(32)]
        fused = multi_scale_feature_fusion(branches)
        assert fused.shape == (96,)

    def test_2d_fusion(self):
        branches = [np.ones((4, 32)), np.ones((4, 32))]
        fused = multi_scale_feature_fusion(branches)
        assert fused.shape == (4, 64)

    def test_different_sizes(self):
        branches = [np.ones(16), np.ones(32), np.ones(64)]
        fused = multi_scale_feature_fusion(branches)
        assert fused.shape == (112,)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            multi_scale_feature_fusion([])

    def test_values_preserved(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        fused = multi_scale_feature_fusion([a, b])
        np.testing.assert_array_equal(fused, [1.0, 2.0, 3.0, 4.0])


# ── compare_architectures ───────────────────────────────────────────────────


class TestCompareArchitectures:
    def test_returns_dict(self):
        result = compare_architectures()
        assert isinstance(result, dict)

    def test_has_eegnet_key(self):
        result = compare_architectures()
        assert "eegnet" in result
        assert result["eegnet"]["architecture"] == "EEGNet"

    def test_has_ameegnet_key(self):
        result = compare_architectures()
        assert "ameegnet" in result
        assert result["ameegnet"]["architecture"] == "AMEEGNet"

    def test_has_comparison(self):
        result = compare_architectures()
        assert "comparison" in result
        assert "param_ratio" in result["comparison"]

    def test_has_recommendation(self):
        result = compare_architectures()
        assert "recommendation" in result
        assert isinstance(result["recommendation"], str)

    def test_ameegnet_has_more_params(self):
        result = compare_architectures()
        assert result["comparison"]["param_ratio"] > 1.0

    def test_custom_eegnet_params(self):
        result = compare_architectures(eegnet_params=5000)
        assert result["eegnet"]["params"] == 5000


# ── config_to_dict ───────────────────────────────────────────────────────────


class TestConfigToDict:
    def test_returns_dict(self):
        config = create_ameegnet_config()
        result = config_to_dict(config)
        assert isinstance(result, dict)

    def test_preserves_keys(self):
        config = create_ameegnet_config()
        result = config_to_dict(config)
        assert "architecture" in result
        assert "n_channels" in result

    def test_numpy_int_converted(self):
        config = {"value": np.int64(42)}
        result = config_to_dict(config)
        assert isinstance(result["value"], int)

    def test_numpy_float_converted(self):
        config = {"value": np.float64(3.14)}
        result = config_to_dict(config)
        assert isinstance(result["value"], float)

    def test_nested_dict_handled(self):
        config = create_ameegnet_config()
        result = config_to_dict(config)
        assert isinstance(result["attention"], dict)
        assert isinstance(result["estimated_params"], dict)
