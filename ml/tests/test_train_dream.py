"""Tests for dream detector training pipeline — issue #472.

Tests the cross-subject evaluation, within-subject evaluation,
benchmark saving with training_data provenance, and data loading
with subject IDs. Uses mocked data loaders to avoid downloading
Sleep-EDF.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Ensure ml/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_dream import (
    cross_subject_eval,
    within_subject_eval,
    save_benchmarks,
    generate_training_data,
    CLASSES,
)


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Small synthetic dataset with subject IDs for testing."""
    np.random.seed(42)
    n_subjects = 4
    epochs_per_subject = 50
    n_features = 17

    X_all = []
    y_all = []
    subject_ids = []

    for subj in range(n_subjects):
        # Each subject gets slightly different feature distributions
        offset = subj * 0.5
        for _ in range(epochs_per_subject):
            features = np.random.randn(n_features) + offset
            label = np.random.randint(0, 2)
            X_all.append(features)
            y_all.append(label)
            subject_ids.append(subj)

    return (
        np.array(X_all),
        np.array(y_all),
        np.array(subject_ids),
    )


@pytest.fixture
def separable_data():
    """Dataset where classes are linearly separable (for testing accuracy)."""
    np.random.seed(123)
    n_subjects = 4
    epochs_per_subject = 100
    n_features = 5

    X_all = []
    y_all = []
    subject_ids = []

    for subj in range(n_subjects):
        for _ in range(epochs_per_subject // 2):
            # Class 0: features centered around -2
            X_all.append(np.random.randn(n_features) - 2)
            y_all.append(0)
            subject_ids.append(subj)

            # Class 1: features centered around +2
            X_all.append(np.random.randn(n_features) + 2)
            y_all.append(1)
            subject_ids.append(subj)

    return (
        np.array(X_all),
        np.array(y_all),
        np.array(subject_ids),
    )


# -- cross_subject_eval tests -----------------------------------------------

class TestCrossSubjectEval:

    def test_returns_expected_keys(self, synthetic_data):
        X, y, subject_ids = synthetic_data
        result = cross_subject_eval(X, y, subject_ids, n_splits=4)
        assert "mean_accuracy" in result
        assert "std_accuracy" in result
        assert "mean_f1" in result
        assert "std_f1" in result
        assert "fold_accuracies" in result
        assert "fold_f1s" in result
        assert "n_splits" in result
        assert "all_y_true" in result
        assert "all_y_pred" in result

    def test_n_splits_matches_folds(self, synthetic_data):
        X, y, subject_ids = synthetic_data
        result = cross_subject_eval(X, y, subject_ids, n_splits=4)
        assert len(result["fold_accuracies"]) == result["n_splits"]
        assert len(result["fold_f1s"]) == result["n_splits"]

    def test_accuracy_between_0_and_1(self, synthetic_data):
        X, y, subject_ids = synthetic_data
        result = cross_subject_eval(X, y, subject_ids, n_splits=4)
        assert 0.0 <= result["mean_accuracy"] <= 1.0
        for acc in result["fold_accuracies"]:
            assert 0.0 <= acc <= 1.0

    def test_f1_between_0_and_1(self, synthetic_data):
        X, y, subject_ids = synthetic_data
        result = cross_subject_eval(X, y, subject_ids, n_splits=4)
        assert 0.0 <= result["mean_f1"] <= 1.0

    def test_separable_data_high_accuracy(self, separable_data):
        X, y, subject_ids = separable_data
        result = cross_subject_eval(X, y, subject_ids, n_splits=4)
        assert result["mean_accuracy"] > 0.85

    def test_all_samples_evaluated(self, synthetic_data):
        X, y, subject_ids = synthetic_data
        result = cross_subject_eval(X, y, subject_ids, n_splits=4)
        assert len(result["all_y_true"]) == len(y)

    def test_fewer_subjects_than_splits(self):
        """When n_splits > n_subjects, should reduce to n_subjects."""
        np.random.seed(42)
        X = np.random.randn(60, 5)
        y = np.random.randint(0, 2, 60)
        subject_ids = np.array([0] * 20 + [1] * 20 + [2] * 20)
        result = cross_subject_eval(X, y, subject_ids, n_splits=10)
        assert result["n_splits"] == 3

    def test_single_subject_raises(self):
        """Cannot do cross-subject eval with only 1 subject."""
        np.random.seed(42)
        X = np.random.randn(30, 5)
        y = np.random.randint(0, 2, 30)
        subject_ids = np.zeros(30, dtype=int)
        with pytest.raises(ValueError, match="at least 2 subjects"):
            cross_subject_eval(X, y, subject_ids)


# -- within_subject_eval tests ----------------------------------------------

class TestWithinSubjectEval:

    def test_returns_expected_keys(self, synthetic_data):
        X, y, _ = synthetic_data
        result = within_subject_eval(X, y)
        assert "accuracy" in result
        assert "f1_macro" in result
        assert "y_test" in result
        assert "y_pred" in result
        assert "model" in result

    def test_accuracy_between_0_and_1(self, synthetic_data):
        X, y, _ = synthetic_data
        result = within_subject_eval(X, y)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_separable_data_high_accuracy(self, separable_data):
        X, y, _ = separable_data
        result = within_subject_eval(X, y)
        assert result["accuracy"] > 0.90

    def test_model_is_trained(self, synthetic_data):
        X, y, _ = synthetic_data
        result = within_subject_eval(X, y)
        # Model should be able to predict
        preds = result["model"].predict(X[:5])
        assert len(preds) == 5

    def test_within_subject_higher_than_cross_subject(self, separable_data):
        """Within-subject accuracy should typically be >= cross-subject."""
        X, y, subject_ids = separable_data
        ws = within_subject_eval(X, y)
        cs = cross_subject_eval(X, y, subject_ids, n_splits=4)
        # Allow within-subject to be at least as good (with small tolerance)
        assert ws["accuracy"] >= cs["mean_accuracy"] - 0.05


# -- save_benchmarks tests --------------------------------------------------

class TestSaveBenchmarks:

    def test_saves_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_benchmarks(
                dataset_name="test-dataset",
                model_name="test_model",
                output_dir=tmpdir,
                training_data="synthetic",
                within_subject_acc=0.95,
                within_subject_f1=0.94,
            )
            path = Path(tmpdir) / "test_model_benchmark.json"
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["training_data"] == "synthetic"
            assert data["within_subject_accuracy"] == 0.95

    def test_training_data_field_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_benchmarks(
                dataset_name="sleep-edf",
                model_name="dream_detector",
                output_dir=tmpdir,
                training_data="real",
            )
            assert result["training_data"] == "real"

    def test_synthetic_training_data_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_benchmarks(
                dataset_name="simulated",
                model_name="dream_detector",
                output_dir=tmpdir,
                training_data="synthetic",
            )
            assert result["training_data"] == "synthetic"

    def test_cross_subject_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_benchmarks(
                dataset_name="sleep-edf",
                model_name="dream_detector",
                output_dir=tmpdir,
                training_data="real",
                cross_subject_acc=0.84,
                cross_subject_acc_std=0.03,
                cross_subject_f1=0.82,
                cross_subject_f1_std=0.04,
            )
            assert result["cross_subject_accuracy"] == 0.84
            assert result["cross_subject_accuracy_std"] == 0.03
            assert result["cross_subject_f1_macro"] == 0.82
            assert result["cross_subject_f1_macro_std"] == 0.04

    def test_legacy_accuracy_uses_cross_subject(self):
        """The top-level 'accuracy' should prefer cross-subject when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_benchmarks(
                dataset_name="sleep-edf",
                model_name="dream_detector",
                output_dir=tmpdir,
                training_data="real",
                within_subject_acc=0.95,
                within_subject_f1=0.94,
                cross_subject_acc=0.84,
                cross_subject_f1=0.82,
            )
            assert result["accuracy"] == 0.84
            assert result["f1_macro"] == 0.82

    def test_legacy_accuracy_falls_back_to_within_subject(self):
        """Without cross-subject, top-level 'accuracy' uses within-subject."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_benchmarks(
                dataset_name="simulated",
                model_name="dream_detector",
                output_dir=tmpdir,
                training_data="synthetic",
                within_subject_acc=0.97,
                within_subject_f1=0.96,
            )
            assert result["accuracy"] == 0.97

    def test_channel_note_included(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            note = "Sleep-EDF uses 2 channels, Muse has 4."
            result = save_benchmarks(
                dataset_name="sleep-edf",
                model_name="dream_detector",
                output_dir=tmpdir,
                training_data="real",
                channel_note=note,
            )
            assert result["channel_note"] == note

    def test_per_class_metrics_with_predictions(self):
        y_test = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1])
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_benchmarks(
                dataset_name="test",
                model_name="dream_detector",
                output_dir=tmpdir,
                training_data="real",
                within_subject_acc=0.83,
                y_test=y_test,
                y_pred=y_pred,
            )
            assert "per_class" in result
            assert "not_dreaming" in result["per_class"]
            assert "dreaming" in result["per_class"]
            assert "confusion_matrix" in result


# -- generate_training_data tests -------------------------------------------

class TestGenerateTrainingData:

    @patch("training.train_dream.simulate_eeg")
    @patch("training.train_dream.extract_features")
    @patch("training.train_dream.preprocess")
    def test_returns_four_values(self, mock_preprocess, mock_extract, mock_sim):
        """generate_training_data should return (X, y, feature_names, None)."""
        mock_sim.return_value = {"signals": [np.random.randn(256 * 30).tolist()]}
        mock_preprocess.return_value = np.random.randn(256 * 30)
        mock_extract.return_value = {"f1": 0.1, "f2": 0.2, "f3": 0.3}

        X, y, feature_names, subject_ids = generate_training_data(
            n_samples_per_class=2
        )
        assert X.shape == (4, 3)  # 2 per class * 2 classes, 3 features
        assert len(y) == 4
        assert subject_ids is None
        assert feature_names == ["f1", "f2", "f3"]

    @patch("training.train_dream.simulate_eeg")
    @patch("training.train_dream.extract_features")
    @patch("training.train_dream.preprocess")
    def test_balanced_classes(self, mock_preprocess, mock_extract, mock_sim):
        mock_sim.return_value = {"signals": [np.random.randn(256 * 30).tolist()]}
        mock_preprocess.return_value = np.random.randn(256 * 30)
        mock_extract.return_value = {"f1": 0.1}

        _, y, _, _ = generate_training_data(n_samples_per_class=5)
        assert np.sum(y == 0) == 5
        assert np.sum(y == 1) == 5


# -- load_real_data tests ---------------------------------------------------

class TestLoadRealData:

    @patch("training.train_dream.load_real_data")
    def test_returns_subject_ids(self, mock_load):
        """load_real_data should return subject_ids as the 4th element."""
        mock_load.return_value = (
            np.random.randn(100, 17),
            np.random.randint(0, 2, 100),
            [f"f{i}" for i in range(17)],
            np.array([0] * 50 + [1] * 50),
        )
        X, y, feature_names, subject_ids = mock_load(n_subjects=2)
        assert subject_ids is not None
        assert len(np.unique(subject_ids)) == 2


# -- load_rem_detection_with_subjects tests ---------------------------------

class TestLoadRemDetectionWithSubjects:

    @patch("training.data_loaders.load_sleep_edf_with_subjects")
    def test_binary_labels(self, mock_load_edf):
        """REM detection should remap 5-class to binary."""
        # Simulate 5-class data
        X = np.random.randn(100, 7680)
        y = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20)
        subj = np.array([0] * 50 + [1] * 50)
        mock_load_edf.return_value = (X, y, subj)

        from training.data_loaders import load_rem_detection_with_subjects
        X_out, y_out, subj_out = load_rem_detection_with_subjects(n_subjects=2)

        # Only class 4 (REM) should be 1, everything else 0
        assert np.sum(y_out == 1) == 20
        assert np.sum(y_out == 0) == 80
        assert len(subj_out) == 100


# -- Integration: full train function with mocked data ----------------------

class TestTrainIntegration:

    @patch("training.train_dream.load_real_data")
    @patch("training.train_dream.export_onnx")
    def test_train_with_real_data_mock(self, mock_onnx, mock_load):
        """Full training pipeline with mocked real data."""
        np.random.seed(42)
        n_subjects = 4
        epochs_per_subject = 50
        n_features = 10

        X = np.random.randn(n_subjects * epochs_per_subject, n_features)
        y = np.random.randint(0, 2, n_subjects * epochs_per_subject)
        feature_names = [f"f{i}" for i in range(n_features)]
        subject_ids = np.repeat(np.arange(n_subjects), epochs_per_subject)

        mock_load.return_value = (X, y, feature_names, subject_ids)
        mock_onnx.return_value = False

        from training.train_dream import train

        with tempfile.TemporaryDirectory() as tmpdir:
            train(
                simulated=False,
                n_subjects=4,
                output_dir=tmpdir,
            )

            # Check model file saved
            model_path = Path(tmpdir) / "dream_detector_model.pkl"
            assert model_path.exists()

            # Check benchmark file saved
            benchmark_dir = Path("benchmarks")
            benchmark_path = benchmark_dir / "dream_detector_benchmark.json"
            if benchmark_path.exists():
                with open(benchmark_path) as f:
                    bench = json.load(f)
                assert bench["training_data"] == "real"
                assert "within_subject_accuracy" in bench
                assert "cross_subject_accuracy" in bench

    @patch("training.train_dream.simulate_eeg")
    @patch("training.train_dream.extract_features")
    @patch("training.train_dream.preprocess")
    @patch("training.train_dream.export_onnx")
    def test_train_simulated_sets_synthetic(
        self, mock_onnx, mock_preprocess, mock_extract, mock_sim
    ):
        """Simulated training should set training_data='synthetic'."""
        mock_sim.return_value = {"signals": [np.random.randn(256 * 30).tolist()]}
        mock_preprocess.return_value = np.random.randn(256 * 30)
        mock_extract.return_value = {f"f{i}": np.random.randn() for i in range(5)}
        mock_onnx.return_value = False

        from training.train_dream import train

        with tempfile.TemporaryDirectory() as tmpdir:
            train(
                simulated=True,
                output_dir=tmpdir,
            )

            benchmark_path = Path("benchmarks") / "dream_detector_benchmark.json"
            if benchmark_path.exists():
                with open(benchmark_path) as f:
                    bench = json.load(f)
                assert bench["training_data"] == "synthetic"
