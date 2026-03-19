"""Tests for CNN-KAN DEAP 4-channel training pipeline configuration."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cnn_kan_trainer import (
    CHANNEL_NAMES,
    DEAP_CHANNEL_INDICES,
    DEAP_CHANNEL_MAP,
    DEAP_N_SUBJECTS,
    DEAP_N_TRIALS,
    DEAP_ORIGINAL_FS,
    EMOTION_CLASSES,
    FREQUENCY_BANDS,
    TARGET_FS,
    TARGET_N_CHANNELS,
    TARGET_WINDOW_SAMPLES,
    compute_training_metrics,
    create_training_config,
    preprocess_deap_epoch,
    setup_cross_validation,
    training_report_to_dict,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def deap_epoch():
    """Simulated DEAP epoch: 32 channels x 8064 samples at 128 Hz."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((32, 8064)) * 20.0


@pytest.fixture
def four_ch_epoch():
    """4-channel epoch already extracted."""
    rng = np.random.default_rng(7)
    return rng.standard_normal((4, 1024)) * 20.0


@pytest.fixture
def default_config():
    return create_training_config()


# ── Constants ────────────────────────────────────────────────────────────────


class TestConstants:
    def test_deap_subjects(self):
        assert DEAP_N_SUBJECTS == 32

    def test_deap_trials(self):
        assert DEAP_N_TRIALS == 40

    def test_deap_fs(self):
        assert DEAP_ORIGINAL_FS == 128.0

    def test_target_fs(self):
        assert TARGET_FS == 256.0

    def test_target_channels(self):
        assert TARGET_N_CHANNELS == 4

    def test_channel_names_muse2(self):
        assert CHANNEL_NAMES == ["TP9", "AF7", "AF8", "TP10"]

    def test_channel_map_has_all(self):
        for name in CHANNEL_NAMES:
            assert name in DEAP_CHANNEL_MAP

    def test_emotion_classes(self):
        assert EMOTION_CLASSES == ["positive", "neutral", "negative"]


# ── create_training_config ───────────────────────────────────────────────────


class TestCreateTrainingConfig:
    def test_returns_dict(self, default_config):
        assert isinstance(default_config, dict)

    def test_has_model_section(self, default_config):
        assert "model" in default_config
        assert default_config["model"]["architecture"] == "CNN-KAN"

    def test_has_data_section(self, default_config):
        assert "data" in default_config
        assert default_config["data"]["dataset"] == "DEAP"

    def test_has_training_section(self, default_config):
        assert "training" in default_config
        assert "learning_rate" in default_config["training"]

    def test_has_validation_section(self, default_config):
        assert "validation" in default_config

    def test_default_cv_strategy(self, default_config):
        assert default_config["validation"]["strategy"] == "5-fold"

    def test_loso_strategy(self):
        config = create_training_config(cv_strategy="loso")
        assert config["validation"]["strategy"] == "loso"
        assert config["validation"]["n_folds"] == DEAP_N_SUBJECTS

    def test_custom_learning_rate(self):
        config = create_training_config(learning_rate=0.01)
        assert config["training"]["learning_rate"] == 0.01

    def test_estimated_windows_positive(self, default_config):
        assert default_config["data"]["estimated_total_windows"] > 0

    def test_has_features_section(self, default_config):
        assert "features" in default_config
        assert default_config["features"]["type"] == "pseudo-rgb"


# ── preprocess_deap_epoch ────────────────────────────────────────────────────


class TestPreprocessDeapEpoch:
    def test_output_has_4_channels(self, deap_epoch):
        result = preprocess_deap_epoch(deap_epoch)
        assert result.shape[0] == TARGET_N_CHANNELS

    def test_output_resampled(self, deap_epoch):
        """Output should have ~2x samples due to 128->256 Hz resampling."""
        result = preprocess_deap_epoch(deap_epoch)
        # 8064 samples at 128 Hz -> ~16128 samples at 256 Hz
        expected = int(8064 * (TARGET_FS / DEAP_ORIGINAL_FS))
        assert result.shape[1] == expected

    def test_values_are_finite(self, deap_epoch):
        result = preprocess_deap_epoch(deap_epoch)
        assert np.all(np.isfinite(result))

    def test_4ch_input_passthrough(self, four_ch_epoch):
        """4-channel input at target fs should pass through."""
        result = preprocess_deap_epoch(
            four_ch_epoch,
            original_fs=TARGET_FS,
            target_fs=TARGET_FS,
        )
        assert result.shape[0] == TARGET_N_CHANNELS

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            preprocess_deap_epoch(np.ones(1024))


# ── setup_cross_validation ───────────────────────────────────────────────────


class TestSetupCrossValidation:
    def test_5fold_returns_5_folds(self):
        cv = setup_cross_validation(strategy="5-fold")
        assert cv["n_folds"] == 5
        assert len(cv["folds"]) == 5

    def test_loso_returns_32_folds(self):
        cv = setup_cross_validation(strategy="loso")
        assert cv["n_folds"] == DEAP_N_SUBJECTS
        assert len(cv["folds"]) == DEAP_N_SUBJECTS

    def test_loso_each_fold_holds_out_one(self):
        cv = setup_cross_validation(strategy="loso")
        for fold in cv["folds"]:
            assert fold["n_test_subjects"] == 1
            assert fold["n_train_subjects"] == DEAP_N_SUBJECTS - 1

    def test_5fold_all_subjects_covered(self):
        cv = setup_cross_validation(strategy="5-fold")
        all_test = []
        for fold in cv["folds"]:
            all_test.extend(fold["test_subjects"])
        assert sorted(all_test) == list(range(DEAP_N_SUBJECTS))

    def test_within_subject(self):
        cv = setup_cross_validation(strategy="within-subject")
        assert cv["n_folds"] == DEAP_N_SUBJECTS

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            setup_cross_validation(strategy="invalid")

    def test_reproducible_with_seed(self):
        cv1 = setup_cross_validation(strategy="5-fold", random_seed=42)
        cv2 = setup_cross_validation(strategy="5-fold", random_seed=42)
        for f1, f2 in zip(cv1["folds"], cv2["folds"]):
            assert f1["test_subjects"] == f2["test_subjects"]


# ── compute_training_metrics ─────────────────────────────────────────────────


class TestComputeTrainingMetrics:
    def test_perfect_accuracy(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        metrics = compute_training_metrics(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_zero_accuracy(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        metrics = compute_training_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_confusion_matrix_shape(self):
        y = np.array([0, 1, 2, 0])
        metrics = compute_training_metrics(y, y)
        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape == (3, 3)

    def test_n_samples_count(self):
        y = np.array([0, 1, 2, 0, 1])
        metrics = compute_training_metrics(y, y)
        assert metrics["n_samples"] == 5

    def test_empty_input(self):
        metrics = compute_training_metrics(np.array([]), np.array([]))
        assert metrics["accuracy"] == 0.0
        assert metrics["n_samples"] == 0

    def test_has_per_class_f1(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        metrics = compute_training_metrics(y, y)
        assert "f1_per_class" in metrics
        for cls in EMOTION_CLASSES:
            assert cls in metrics["f1_per_class"]

    def test_has_class_distribution(self):
        y_true = np.array([0, 0, 1, 2])
        y_pred = np.array([0, 0, 1, 2])
        metrics = compute_training_metrics(y_true, y_pred)
        assert metrics["class_distribution"]["positive"] == 2
        assert metrics["class_distribution"]["neutral"] == 1
        assert metrics["class_distribution"]["negative"] == 1


# ── training_report_to_dict ──────────────────────────────────────────────────


class TestTrainingReportToDict:
    def test_empty_metrics_list(self):
        config = create_training_config()
        report = training_report_to_dict(config, [])
        assert report["status"] == "no_results"

    def test_complete_report(self):
        config = create_training_config()
        y = np.array([0, 1, 2, 0, 1, 2])
        fold_metrics = [compute_training_metrics(y, y) for _ in range(5)]
        report = training_report_to_dict(config, fold_metrics)
        assert report["status"] == "complete"
        assert report["summary"]["n_folds"] == 5
        assert report["summary"]["mean_accuracy"] == 1.0

    def test_per_fold_present(self):
        config = create_training_config()
        y = np.array([0, 1, 2])
        fold_metrics = [compute_training_metrics(y, y) for _ in range(3)]
        report = training_report_to_dict(config, fold_metrics)
        assert len(report["per_fold"]) == 3

    def test_per_class_averages(self):
        config = create_training_config()
        y = np.array([0, 1, 2, 0, 1, 2])
        fold_metrics = [compute_training_metrics(y, y)]
        report = training_report_to_dict(config, fold_metrics)
        assert "per_class" in report
        for cls in EMOTION_CLASSES:
            assert cls in report["per_class"]
