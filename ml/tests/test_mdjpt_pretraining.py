"""Tests for mdJPT multi-dataset joint pre-training (#408).

Covers:
  - TestHarmonizeDatasets: z-score normalisation, multi-dataset, errors
  - TestAlignChannels: virtual channel mapping, single/multi sample, errors
  - TestAlignLabels: all label schemes (VA continuous, 3-class, 4-class)
  - TestCreatePretrainingConfig: defaults, custom, validation errors
  - TestComputeDatasetStatistics: shapes, label distribution
  - TestEvaluateTransfer: accuracy, per-class, edge cases
  - TestConfigToDict: round-trip serialisation
  - TestEdgeCases: empty, single sample, constant features
"""

import numpy as np
import pytest

from models.mdjpt_pretraining import (
    SAMPLING_STRATEGIES,
    SUPPORTED_DATASETS,
    UNIFIED_LABELS,
    PretrainingConfig,
    align_channels,
    align_labels,
    compute_dataset_statistics,
    config_to_dict,
    create_pretraining_config,
    evaluate_transfer,
    harmonize_datasets,
)


# ---------------------------------------------------------------------------
# TestHarmonizeDatasets
# ---------------------------------------------------------------------------

class TestHarmonizeDatasets:
    def test_single_dataset_zscore(self):
        """Single dataset should be z-scored (mean~0, std~1)."""
        np.random.seed(42)
        features = np.random.randn(100, 10) * 5 + 3
        result = harmonize_datasets({"DEAP": features})
        h = result["DEAP"]
        assert h.shape == (100, 10)
        np.testing.assert_allclose(h.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(h.std(axis=0), 1.0, atol=1e-10)

    def test_multiple_datasets(self):
        """Multiple datasets should each be independently z-scored."""
        np.random.seed(42)
        feats = {
            "DEAP": np.random.randn(50, 5) * 10 + 100,
            "SEED": np.random.randn(80, 5) * 2 - 5,
        }
        result = harmonize_datasets(feats)
        assert set(result.keys()) == {"DEAP", "SEED"}
        assert result["DEAP"].shape == (50, 5)
        assert result["SEED"].shape == (80, 5)

    def test_harmonized_means_near_zero(self):
        """After harmonisation, per-feature means should be near zero."""
        np.random.seed(42)
        feats = {"GAMEEMO": np.random.randn(60, 8) * 20 + 50}
        result = harmonize_datasets(feats)
        np.testing.assert_allclose(result["GAMEEMO"].mean(axis=0), 0.0, atol=1e-10)

    def test_unsupported_dataset_raises(self):
        """Unknown dataset name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            harmonize_datasets({"UNKNOWN": np.ones((10, 5))})

    def test_1d_input_reshaped(self):
        """Single-sample 1D input should be reshaped to (1, n_features)."""
        result = harmonize_datasets({"DEAP": np.array([1.0, 2.0, 3.0])})
        assert result["DEAP"].shape == (1, 3)

    def test_with_precomputed_stats(self):
        """Pre-computed stats should be used instead of data stats."""
        from models.mdjpt_pretraining import DatasetStatistics
        features = np.ones((10, 3)) * 5.0
        stats = {
            "DEAP": DatasetStatistics(
                name="DEAP",
                feature_means=[5.0, 5.0, 5.0],
                feature_stds=[1.0, 1.0, 1.0],
            )
        }
        result = harmonize_datasets({"DEAP": features}, dataset_stats=stats)
        np.testing.assert_allclose(result["DEAP"], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# TestAlignChannels
# ---------------------------------------------------------------------------

class TestAlignChannels:
    def test_deap_4_virtual_channels(self):
        """DEAP signals should map to 4 virtual channels."""
        np.random.seed(42)
        montage = [
            "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
            "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
            "Fp2", "AF4", "F4", "F8", "FC6", "FC2", "C4", "T8",
            "CP6", "CP2", "P4", "P8", "PO4", "O2", "Fz", "Cz",
        ]
        signals = np.random.randn(5, len(montage))
        result = align_channels(signals, montage, "DEAP")
        assert result.shape == (5, 4)

    def test_single_sample(self):
        """Single-sample 1D input should produce (1, 4)."""
        montage = ["F3", "AF3", "F7", "F4", "AF4", "F8", "T7", "CP5", "P7", "T8", "CP6", "P8"]
        signals = np.random.randn(len(montage))
        result = align_channels(signals, montage, "GAMEEMO")
        assert result.shape == (1, 4)

    def test_unsupported_dataset_raises(self):
        """Unknown dataset should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            align_channels(np.ones((1, 4)), ["a", "b", "c", "d"], "FAKE")

    def test_output_values_are_weighted_averages(self):
        """Virtual channels should be weighted averages of source channels."""
        montage = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
                    "P8", "T8", "FC6", "F4", "F8", "AF4"]
        signals = np.zeros((1, 14))
        # Set F3=10, AF3=10, F7=10 -> VF_left should be ~10
        signals[0, 2] = 10.0   # F3
        signals[0, 0] = 10.0   # AF3
        signals[0, 1] = 10.0   # F7
        result = align_channels(signals, montage, "GAMEEMO")
        # VF_left = (F3*0.5 + AF3*0.3 + F7*0.2) / (0.5+0.3+0.2) = 10
        assert abs(result[0, 0] - 10.0) < 1e-6

    def test_missing_channels_zero_fill(self):
        """If source channels are not in montage, virtual channel is 0."""
        montage = ["Cz", "Pz"]  # none of the mapped channels
        signals = np.ones((1, 2)) * 5.0
        result = align_channels(signals, montage, "DEAP")
        # No mapped channels found, so output is zeros
        np.testing.assert_allclose(result, 0.0)


# ---------------------------------------------------------------------------
# TestAlignLabels
# ---------------------------------------------------------------------------

class TestAlignLabels:
    def test_deap_high_valence_high_arousal(self):
        """DEAP (v=8, a=8) -> happy."""
        result = align_labels([(8.0, 8.0)], "DEAP")
        assert result == ["happy"]

    def test_deap_low_valence_low_arousal(self):
        """DEAP (v=2, a=2) -> sad."""
        result = align_labels([(2.0, 2.0)], "DEAP")
        assert result == ["sad"]

    def test_deap_high_valence_low_arousal(self):
        """DEAP (v=7, a=2) -> neutral."""
        result = align_labels([(7.0, 2.0)], "DEAP")
        assert result == ["neutral"]

    def test_deap_low_valence_high_arousal_angry(self):
        """DEAP (v=4.5, a=7) -> angry (above mid-1)."""
        result = align_labels([(4.5, 7.0)], "DEAP")
        assert result == ["angry"]

    def test_deap_very_low_valence_high_arousal_fear(self):
        """DEAP (v=2, a=7) -> fear (below mid-1)."""
        result = align_labels([(2.0, 7.0)], "DEAP")
        assert result == ["fear"]

    def test_seed_positive(self):
        """SEED label 1 -> happy."""
        assert align_labels([1], "SEED") == ["happy"]

    def test_seed_neutral(self):
        """SEED label 0 -> neutral."""
        assert align_labels([0], "SEED") == ["neutral"]

    def test_seed_negative(self):
        """SEED label -1 -> sad."""
        assert align_labels([-1], "SEED") == ["sad"]

    def test_gameemo_relaxed_maps_to_neutral(self):
        """GAMEEMO relaxed -> neutral."""
        assert align_labels(["relaxed"], "GAMEEMO") == ["neutral"]

    def test_gameemo_fear(self):
        """GAMEEMO fear -> fear."""
        assert align_labels(["fear"], "GAMEEMO") == ["fear"]

    def test_multiple_labels(self):
        """Multiple labels should be processed in order."""
        result = align_labels([(8, 8), (2, 2), (7, 2)], "DEAP")
        assert len(result) == 3
        assert all(lbl in UNIFIED_LABELS for lbl in result)

    def test_unsupported_dataset_raises(self):
        """Unknown dataset should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            align_labels(["happy"], "FAKE")

    def test_dreamer_maps_correctly(self):
        """DREAMER uses 1-5 scale, mid=3. (v=4, a=4) -> happy."""
        result = align_labels([(4.0, 4.0)], "DREAMER")
        assert result == ["happy"]


# ---------------------------------------------------------------------------
# TestCreatePretrainingConfig
# ---------------------------------------------------------------------------

class TestCreatePretrainingConfig:
    def test_default_config(self):
        """Default config should include all datasets."""
        config = create_pretraining_config()
        assert set(config.datasets) == set(SUPPORTED_DATASETS)
        assert config.sampling_strategy == "proportional"
        assert config.shared_encoder_dim == 128

    def test_custom_datasets(self):
        """Custom dataset list should be respected."""
        config = create_pretraining_config(datasets=["DEAP", "SEED"])
        assert config.datasets == ["DEAP", "SEED"]

    def test_invalid_dataset_raises(self):
        """Invalid dataset should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            create_pretraining_config(datasets=["FAKE"])

    def test_invalid_strategy_raises(self):
        """Invalid sampling strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_pretraining_config(sampling_strategy="invalid")

    def test_invalid_encoder_dim_raises(self):
        """Encoder dim < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="shared_encoder_dim"):
            create_pretraining_config(shared_encoder_dim=0)

    def test_invalid_batch_size_raises(self):
        """Batch size < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="batch_size"):
            create_pretraining_config(batch_size=0)

    def test_curriculum_strategy(self):
        """Curriculum strategy should be accepted."""
        config = create_pretraining_config(sampling_strategy="curriculum")
        assert config.sampling_strategy == "curriculum"

    def test_freeze_encoder_flag(self):
        """Freeze encoder flag should be stored."""
        config = create_pretraining_config(freeze_encoder_for_transfer=True)
        assert config.freeze_encoder_for_transfer is True


# ---------------------------------------------------------------------------
# TestComputeDatasetStatistics
# ---------------------------------------------------------------------------

class TestComputeDatasetStatistics:
    def test_basic_statistics(self):
        """Should compute means and stds correctly."""
        np.random.seed(42)
        features = np.random.randn(100, 5)
        stats = compute_dataset_statistics(features, dataset_name="DEAP")
        assert stats.name == "DEAP"
        assert stats.n_samples == 100
        assert stats.n_features == 5
        assert len(stats.feature_means) == 5
        assert len(stats.feature_stds) == 5

    def test_with_labels(self):
        """Label distribution should be computed when labels provided."""
        features = np.ones((10, 3))
        labels = ["happy", "happy", "sad", "sad", "sad",
                   "neutral", "neutral", "neutral", "neutral", "neutral"]
        stats = compute_dataset_statistics(features, labels=labels)
        assert stats.label_distribution == {"happy": 2, "sad": 3, "neutral": 5}

    def test_1d_input(self):
        """1D input should be reshaped to (1, n_features)."""
        stats = compute_dataset_statistics(np.array([1.0, 2.0, 3.0]))
        assert stats.n_samples == 1
        assert stats.n_features == 3

    def test_timestamp_set(self):
        """computed_at should be a positive float."""
        stats = compute_dataset_statistics(np.ones((5, 2)))
        assert stats.computed_at > 0


# ---------------------------------------------------------------------------
# TestEvaluateTransfer
# ---------------------------------------------------------------------------

class TestEvaluateTransfer:
    def test_perfect_accuracy(self):
        """All correct predictions should give accuracy=1.0."""
        preds = ["happy", "sad", "neutral"]
        labels = ["happy", "sad", "neutral"]
        result = evaluate_transfer(preds, labels)
        assert result["accuracy"] == 1.0
        assert result["n_correct"] == 3

    def test_zero_accuracy(self):
        """All wrong predictions should give accuracy=0.0."""
        preds = ["sad", "happy", "angry"]
        labels = ["happy", "sad", "neutral"]
        result = evaluate_transfer(preds, labels)
        assert result["accuracy"] == 0.0
        assert result["n_correct"] == 0

    def test_partial_accuracy(self):
        """Partial correctness should give fractional accuracy."""
        preds = ["happy", "sad", "angry", "neutral"]
        labels = ["happy", "sad", "happy", "neutral"]
        result = evaluate_transfer(preds, labels)
        assert result["accuracy"] == 0.75
        assert result["n_correct"] == 3

    def test_per_class_precision(self):
        """Precision should be computed per class."""
        preds = ["happy", "happy", "sad"]
        labels = ["happy", "sad", "sad"]
        result = evaluate_transfer(preds, labels)
        assert result["per_class_precision"]["happy"] == 0.5
        assert result["per_class_precision"]["sad"] == 1.0

    def test_per_class_recall(self):
        """Recall should be computed per class."""
        preds = ["happy", "happy", "sad"]
        labels = ["happy", "sad", "sad"]
        result = evaluate_transfer(preds, labels)
        assert result["per_class_recall"]["happy"] == 1.0
        assert result["per_class_recall"]["sad"] == 0.5

    def test_empty_input(self):
        """Empty lists should return zero accuracy."""
        result = evaluate_transfer([], [])
        assert result["accuracy"] == 0.0
        assert result["n_samples"] == 0

    def test_length_mismatch_raises(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate_transfer(["happy"], ["happy", "sad"])


# ---------------------------------------------------------------------------
# TestConfigToDict
# ---------------------------------------------------------------------------

class TestConfigToDict:
    def test_round_trip(self):
        """Config -> dict should contain all fields."""
        config = create_pretraining_config()
        d = config_to_dict(config)
        assert isinstance(d, dict)
        assert d["sampling_strategy"] == "proportional"
        assert d["shared_encoder_dim"] == 128
        assert set(d["datasets"]) == set(SUPPORTED_DATASETS)

    def test_custom_config_preserved(self):
        """Custom values should survive serialisation."""
        config = create_pretraining_config(
            datasets=["DEAP"],
            batch_size=32,
            learning_rate=0.01,
        )
        d = config_to_dict(config)
        assert d["datasets"] == ["DEAP"]
        assert d["batch_size"] == 32
        assert d["learning_rate"] == 0.01


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_constant_features_harmonize(self):
        """Constant features (zero variance) should not crash."""
        features = np.ones((10, 5)) * 42.0
        result = harmonize_datasets({"DEAP": features})
        # std=0 -> replaced with 1.0 -> (42-42)/1 = 0
        np.testing.assert_allclose(result["DEAP"], 0.0, atol=1e-10)

    def test_single_sample_harmonize(self):
        """Single sample should be harmonised without error."""
        features = np.array([[1.0, 2.0, 3.0]])
        result = harmonize_datasets({"SEED": features})
        assert result["SEED"].shape == (1, 3)

    def test_all_datasets_harmonize(self):
        """All four supported datasets should harmonise together."""
        np.random.seed(42)
        feats = {d: np.random.randn(20, 4) for d in SUPPORTED_DATASETS}
        result = harmonize_datasets(feats)
        assert len(result) == 4

    def test_align_labels_scalar_va(self):
        """Scalar (non-tuple) VA label should use valence only."""
        result = align_labels([8.0], "DEAP")
        assert result[0] in UNIFIED_LABELS
