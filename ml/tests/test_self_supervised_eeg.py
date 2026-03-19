"""Tests for self-supervised EEG pretraining for few-shot calibration (issue #387).

15+ tests covering generate_masked_sample, generate_contrastive_pair,
generate_temporal_order_task, create_pretext_task, compute_few_shot_config,
and pretraining_config_to_dict.
"""

import numpy as np
import pytest

from models.self_supervised_eeg import (
    EMOTION_CLASSES,
    SUPPORTED_TASKS,
    PretrainingConfig,
    compute_few_shot_config,
    create_pretext_task,
    generate_contrastive_pair,
    generate_masked_sample,
    generate_temporal_order_task,
    pretraining_config_to_dict,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)
SIGNAL_4CH = RNG.standard_normal((4, 1024)).astype(np.float32) * 20.0
SIGNAL_1D = RNG.standard_normal(1024).astype(np.float32) * 20.0
SIGNAL_4CH_ALT = RNG.standard_normal((4, 1024)).astype(np.float32) * 20.0


# ══════════════════════════════════════════════════════════════════════════════
#  generate_masked_sample
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateMaskedSample:
    def test_masked_signal_shape_2d(self):
        result = generate_masked_sample(SIGNAL_4CH, seed=0)
        assert result["masked_signal"].shape == SIGNAL_4CH.shape

    def test_masked_signal_shape_1d(self):
        result = generate_masked_sample(SIGNAL_1D, seed=0)
        assert result["masked_signal"].shape == SIGNAL_1D.shape

    def test_mask_has_correct_ratio(self):
        result = generate_masked_sample(SIGNAL_4CH, mask_ratio=0.15, seed=0)
        actual_ratio = result["mask"].sum() / len(result["mask"])
        assert 0.10 < actual_ratio < 0.20

    def test_masked_positions_are_zero_2d(self):
        result = generate_masked_sample(SIGNAL_4CH, mask_ratio=0.15, seed=0)
        mask = result["mask"]
        masked_signal = result["masked_signal"]
        # All channels should be zero at masked positions
        assert np.all(masked_signal[:, mask] == 0.0)

    def test_masked_positions_are_zero_1d(self):
        result = generate_masked_sample(SIGNAL_1D, mask_ratio=0.15, seed=0)
        mask = result["mask"]
        assert np.all(result["masked_signal"][mask] == 0.0)

    def test_target_is_original(self):
        result = generate_masked_sample(SIGNAL_4CH, seed=0)
        np.testing.assert_array_equal(result["target"], SIGNAL_4CH)

    def test_seed_reproducibility(self):
        a = generate_masked_sample(SIGNAL_4CH, seed=42)
        b = generate_masked_sample(SIGNAL_4CH, seed=42)
        np.testing.assert_array_equal(a["mask"], b["mask"])

    def test_n_masked_is_positive(self):
        result = generate_masked_sample(SIGNAL_4CH, mask_ratio=0.15, seed=0)
        assert result["n_masked"] > 0


# ══════════════════════════════════════════════════════════════════════════════
#  generate_contrastive_pair
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateContrastivePair:
    def test_returns_anchor_positive_negative(self):
        result = generate_contrastive_pair(SIGNAL_4CH, seed=0)
        assert "anchor" in result
        assert "positive" in result
        assert "negative" in result

    def test_anchor_matches_input(self):
        result = generate_contrastive_pair(SIGNAL_4CH, seed=0)
        np.testing.assert_array_equal(result["anchor"], SIGNAL_4CH)

    def test_positive_differs_from_anchor(self):
        result = generate_contrastive_pair(SIGNAL_4CH, seed=0)
        assert not np.allclose(result["positive"], result["anchor"])

    def test_positive_preserves_shape(self):
        result = generate_contrastive_pair(SIGNAL_4CH, seed=0)
        assert result["positive"].shape == SIGNAL_4CH.shape

    def test_with_explicit_negative(self):
        result = generate_contrastive_pair(
            SIGNAL_4CH, negative_signal=SIGNAL_4CH_ALT, seed=0
        )
        np.testing.assert_array_equal(result["negative"], SIGNAL_4CH_ALT)

    def test_augmentations_listed(self):
        result = generate_contrastive_pair(SIGNAL_4CH, seed=0)
        assert len(result["augmentations_applied"]) > 0


# ══════════════════════════════════════════════════════════════════════════════
#  generate_temporal_order_task
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateTemporalOrderTask:
    def test_returns_two_segments(self):
        result = generate_temporal_order_task(SIGNAL_4CH, seed=0)
        assert "segment_a" in result
        assert "segment_b" in result

    def test_label_is_binary(self):
        result = generate_temporal_order_task(SIGNAL_4CH, seed=0)
        assert result["label"] in (0, 1)

    def test_segments_cover_full_signal_2d(self):
        result = generate_temporal_order_task(SIGNAL_4CH, seed=0)
        total = result["segment_a"].shape[-1] + result["segment_b"].shape[-1]
        assert total == SIGNAL_4CH.shape[-1]

    def test_segments_cover_full_signal_1d(self):
        result = generate_temporal_order_task(SIGNAL_1D, seed=0)
        total = len(result["segment_a"]) + len(result["segment_b"])
        assert total == len(SIGNAL_1D)

    def test_split_point_is_valid(self):
        result = generate_temporal_order_task(SIGNAL_4CH, seed=0)
        assert 0 < result["split_point"] < SIGNAL_4CH.shape[-1]

    def test_both_labels_appear(self):
        """Over many seeds, both label=0 and label=1 should appear."""
        labels = set()
        for s in range(50):
            result = generate_temporal_order_task(SIGNAL_4CH, seed=s)
            labels.add(result["label"])
        assert labels == {0, 1}


# ══════════════════════════════════════════════════════════════════════════════
#  create_pretext_task
# ══════════════════════════════════════════════════════════════════════════════

class TestCreatePretextTask:
    def test_masked_prediction_returns_task_type(self):
        result = create_pretext_task(SIGNAL_4CH, task_type="masked_prediction", seed=0)
        assert result["task_type"] == "masked_prediction"

    def test_contrastive_returns_task_type(self):
        result = create_pretext_task(SIGNAL_4CH, task_type="contrastive", seed=0)
        assert result["task_type"] == "contrastive"

    def test_temporal_order_returns_task_type(self):
        result = create_pretext_task(SIGNAL_4CH, task_type="temporal_order", seed=0)
        assert result["task_type"] == "temporal_order"

    def test_invalid_task_type_raises(self):
        with pytest.raises(ValueError, match="Unknown task_type"):
            create_pretext_task(SIGNAL_4CH, task_type="invalid")

    def test_all_supported_tasks(self):
        for task in SUPPORTED_TASKS:
            result = create_pretext_task(SIGNAL_4CH, task_type=task, seed=0)
            assert result["task_type"] == task


# ══════════════════════════════════════════════════════════════════════════════
#  compute_few_shot_config
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeFewShotConfig:
    def test_returns_required_keys(self):
        result = compute_few_shot_config()
        required = {
            "n_samples_per_class", "n_classes", "learning_rate",
            "n_epochs", "batch_size", "freeze_encoder",
            "augment_few_shot", "augmentation_factor",
            "class_names", "total_training_samples",
            "estimated_accuracy_range",
        }
        assert required.issubset(result.keys())

    def test_5_samples_freezes_encoder(self):
        result = compute_few_shot_config(n_samples_per_class=5)
        assert result["freeze_encoder"] is True

    def test_20_samples_unfreezes_encoder(self):
        result = compute_few_shot_config(n_samples_per_class=20)
        assert result["freeze_encoder"] is False

    def test_accuracy_range_is_valid(self):
        result = compute_few_shot_config(n_samples_per_class=5, n_classes=6)
        low, high = result["estimated_accuracy_range"]
        assert 0.0 < low < high <= 1.0

    def test_custom_class_names(self):
        names = ["happy", "sad", "angry"]
        result = compute_few_shot_config(n_classes=3, class_names=names)
        assert result["class_names"] == names

    def test_total_training_samples_with_augmentation(self):
        result = compute_few_shot_config(n_samples_per_class=5, n_classes=6)
        # 5 * 6 = 30 base, augmentation_factor >= 5 => total >= 150
        assert result["total_training_samples"] >= 30


# ══════════════════════════════════════════════════════════════════════════════
#  pretraining_config_to_dict
# ══════════════════════════════════════════════════════════════════════════════

class TestPretrainingConfigToDict:
    def test_returns_dict(self):
        d = pretraining_config_to_dict()
        assert isinstance(d, dict)

    def test_all_keys_present(self):
        d = pretraining_config_to_dict()
        expected = {
            "task_types", "mask_ratio", "n_channels", "fs",
            "epoch_duration_s", "samples_per_epoch",
            "augmentation_params", "n_pretraining_epochs",
            "batch_size", "learning_rate", "temperature",
            "supported_tasks", "emotion_classes",
        }
        assert expected.issubset(d.keys())

    def test_custom_config(self):
        config = PretrainingConfig(mask_ratio=0.25, temperature=0.5)
        d = pretraining_config_to_dict(config)
        assert d["mask_ratio"] == 0.25
        assert d["temperature"] == 0.5

    def test_samples_per_epoch_calculated(self):
        config = PretrainingConfig(fs=256, epoch_duration=4.0)
        d = pretraining_config_to_dict(config)
        assert d["samples_per_epoch"] == 1024

    def test_values_are_native_types(self):
        d = pretraining_config_to_dict()
        assert isinstance(d["mask_ratio"], float)
        assert isinstance(d["n_channels"], int)
        assert isinstance(d["task_types"], list)
