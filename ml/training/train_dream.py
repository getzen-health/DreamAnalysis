"""Training script for dream (REM) detection model.

Trains on Sleep-EDF data with binary REM/non-REM labels.
Exports to both sklearn .pkl and ONNX formats with benchmark results.

Supports two evaluation modes:
- Within-subject: random 80/20 split (optimistic, data leakage across subjects)
- Cross-subject: leave-N-subjects-out (realistic, no data leakage)

Usage:
    python -m training.train_dream                     # real data, cross-subject
    python -m training.train_dream --simulated         # synthetic data (for dev)
    python -m training.train_dream --n-subjects 10
    python -m training.train_dream --within-subject    # within-subject split
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.ensemble import GradientBoostingClassifier
import joblib

from processing.eeg_processor import extract_features, preprocess
from simulation.eeg_simulator import simulate_eeg

CLASSES = ["not_dreaming", "dreaming"]
MODEL_DIR = Path("models/saved")
BENCHMARK_DIR = Path("benchmarks")


def generate_training_data(
    n_samples_per_class: int = 300, fs: float = 256.0, epoch_duration: float = 30.0
):
    """Generate synthetic training data for binary REM detection."""
    X = []
    y = []

    # Non-REM states
    non_rem_states = ["rest", "light_sleep", "deep_sleep"]
    for _ in range(n_samples_per_class):
        state = non_rem_states[np.random.randint(len(non_rem_states))]
        result = simulate_eeg(state=state, duration=epoch_duration, fs=fs)
        eeg = np.array(result["signals"][0])
        features = extract_features(preprocess(eeg, fs), fs)
        X.append(list(features.values()))
        y.append(0)

    # REM state
    for _ in range(n_samples_per_class):
        result = simulate_eeg(state="rem", duration=epoch_duration, fs=fs)
        eeg = np.array(result["signals"][0])
        features = extract_features(preprocess(eeg, fs), fs)
        X.append(list(features.values()))
        y.append(1)

    return np.array(X), np.array(y), list(features.keys()), None


def load_real_data(n_subjects: int = 20, fs: float = 256.0):
    """Load real Sleep-EDF data with binary REM labels and subject IDs.

    Returns:
        (X_features, y, feature_names, subject_ids)
    """
    from training.data_loaders import load_rem_detection_with_subjects

    print(f"Loading REM detection data for {n_subjects} subjects...")
    X_raw, y, subject_ids = load_rem_detection_with_subjects(
        n_subjects=n_subjects, target_fs=fs
    )

    print(f"Extracting features from {len(y)} epochs...")
    X_features = []
    feature_names = None

    for i, epoch in enumerate(X_raw):
        features = extract_features(preprocess(epoch, fs), fs)
        X_features.append(list(features.values()))
        if feature_names is None:
            feature_names = list(features.keys())
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(y)} epochs")

    return np.array(X_features), y, feature_names, subject_ids


def export_onnx(model, feature_names, output_path):
    """Export sklearn model to ONNX format."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [
            ("float_input", FloatTensorType([None, len(feature_names)]))
        ]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX model saved to {output_path}")
        return True
    except ImportError:
        print("skl2onnx not installed, skipping ONNX export")
        return False


def cross_subject_eval(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict:
    """Evaluate model using leave-N-subjects-out cross-validation.

    Groups subjects into n_splits folds. Each fold holds out a group of
    subjects for testing --- no data leakage across the train/test boundary.

    Returns:
        Dictionary with per-fold and aggregate accuracy/f1 statistics.
    """
    unique_subjects = np.unique(subject_ids)
    actual_splits = min(n_splits, len(unique_subjects))

    if actual_splits < 2:
        raise ValueError(
            f"Need at least 2 subjects for cross-subject eval, got {len(unique_subjects)}"
        )

    gkf = GroupKFold(n_splits=actual_splits)
    fold_accuracies = []
    fold_f1s = []
    all_y_true = []
    all_y_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=subject_ids)
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_subjects = np.unique(subject_ids[train_idx])
        test_subjects = np.unique(subject_ids[test_idx])

        model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        fold_accuracies.append(acc)
        fold_f1s.append(f1)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(
            f"  Fold {fold_idx + 1}/{actual_splits}: "
            f"acc={acc:.4f}, f1={f1:.4f} "
            f"(train subjects: {list(train_subjects)}, "
            f"test subjects: {list(test_subjects)})"
        )

    return {
        "fold_accuracies": fold_accuracies,
        "fold_f1s": fold_f1s,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "mean_f1": float(np.mean(fold_f1s)),
        "std_f1": float(np.std(fold_f1s)),
        "n_splits": actual_splits,
        "all_y_true": all_y_true,
        "all_y_pred": all_y_pred,
    }


def within_subject_eval(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """Evaluate model using random train/test split (within-subject).

    This is the optimistic evaluation --- epochs from the same subject
    can appear in both train and test sets, inflating accuracy.

    Returns:
        Dictionary with accuracy/f1 statistics and predictions.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "y_test": y_test,
        "y_pred": y_pred,
        "model": model,
    }


def save_benchmarks(
    dataset_name: str,
    model_name: str,
    output_dir,
    training_data: str,
    within_subject_acc: Optional[float] = None,
    within_subject_f1: Optional[float] = None,
    cross_subject_acc: Optional[float] = None,
    cross_subject_acc_std: Optional[float] = None,
    cross_subject_f1: Optional[float] = None,
    cross_subject_f1_std: Optional[float] = None,
    y_test=None,
    y_pred=None,
    channel_note: Optional[str] = None,
) -> Dict:
    """Save benchmark results as JSON with training_data provenance field."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks: Dict = {
        "model_name": model_name,
        "dataset": dataset_name,
        "training_data": training_data,
    }

    # Within-subject results
    if within_subject_acc is not None:
        benchmarks["within_subject_accuracy"] = within_subject_acc
    if within_subject_f1 is not None:
        benchmarks["within_subject_f1_macro"] = within_subject_f1

    # Cross-subject results (the honest number)
    if cross_subject_acc is not None:
        benchmarks["cross_subject_accuracy"] = cross_subject_acc
    if cross_subject_acc_std is not None:
        benchmarks["cross_subject_accuracy_std"] = cross_subject_acc_std
    if cross_subject_f1 is not None:
        benchmarks["cross_subject_f1_macro"] = cross_subject_f1
    if cross_subject_f1_std is not None:
        benchmarks["cross_subject_f1_macro_std"] = cross_subject_f1_std

    # Legacy fields for backward compatibility
    if cross_subject_acc is not None:
        benchmarks["accuracy"] = cross_subject_acc
        benchmarks["f1_macro"] = cross_subject_f1
    elif within_subject_acc is not None:
        benchmarks["accuracy"] = within_subject_acc
        benchmarks["f1_macro"] = within_subject_f1

    # Per-class metrics from final evaluation
    if y_test is not None and y_pred is not None:
        report = classification_report(
            y_test, y_pred, target_names=CLASSES, output_dict=True
        )
        cm = confusion_matrix(y_test, y_pred).tolist()
        benchmarks["per_class"] = {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": report[cls]["support"],
            }
            for cls in CLASSES
            if cls in report
        }
        benchmarks["confusion_matrix"] = cm

    # Channel limitation note
    if channel_note:
        benchmarks["channel_note"] = channel_note

    benchmark_path = output_dir / f"{model_name}_benchmark.json"
    with open(benchmark_path, "w") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"Benchmarks saved to {benchmark_path}")

    return benchmarks


def train(
    simulated: bool = False,
    n_subjects: int = 20,
    output_dir: str = "models/saved",
    within_subject_only: bool = False,
):
    """Train and evaluate the dream detection model.

    By default, uses real Sleep-EDF data and reports both within-subject
    and cross-subject accuracy. The cross-subject number is the honest
    metric --- it reflects how the model performs on unseen individuals.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    if simulated:
        print("Generating simulated training data...")
        X, y, feature_names, subject_ids = generate_training_data(
            n_samples_per_class=300
        )
        dataset_name = "simulated"
        training_data = "synthetic"
    else:
        print("Loading real Sleep-EDF dataset for REM detection...")
        try:
            X, y, feature_names, subject_ids = load_real_data(
                n_subjects=n_subjects
            )
            dataset_name = "sleep-edf-rem"
            training_data = "real"
        except Exception as e:
            print(f"Failed to load real data: {e}")
            print("Falling back to simulated data...")
            X, y, feature_names, subject_ids = generate_training_data(
                n_samples_per_class=300
            )
            dataset_name = "simulated"
            training_data = "synthetic"

    print(f"\nDataset: {dataset_name} (training_data: {training_data})")
    print(f"Total samples: {X.shape[0]}")
    print(f"Features: {len(feature_names)}")
    print(f"Class balance: {np.bincount(y)}")

    if training_data == "real" and subject_ids is not None:
        unique_subjects = np.unique(subject_ids)
        print(f"Subjects: {len(unique_subjects)}")

    # --- Channel limitation note ---
    channel_note = None
    if training_data == "real":
        channel_note = (
            "Sleep-EDF provides 2 EEG channels (Fpz-Cz, Pz-Oz). "
            "Training uses Fpz-Cz only. Muse 2 has 4 channels "
            "(TP9, AF7, AF8, TP10) at different positions --- "
            "expect a domain gap when deploying on Muse hardware."
        )
        print(f"\nChannel note: {channel_note}")

    # --- Within-subject evaluation ---
    print("\n--- Within-subject evaluation (random 80/20 split) ---")
    ws_result = within_subject_eval(X, y)
    ws_acc = ws_result["accuracy"]
    ws_f1 = ws_result["f1_macro"]
    print(f"Within-subject accuracy: {ws_acc:.4f}")
    print(f"Within-subject F1 macro: {ws_f1:.4f}")
    print("\nClassification Report (within-subject):")
    print(
        classification_report(
            ws_result["y_test"], ws_result["y_pred"], target_names=CLASSES
        )
    )

    # --- Cross-subject evaluation ---
    cs_acc = None
    cs_acc_std = None
    cs_f1 = None
    cs_f1_std = None
    cs_y_true = None
    cs_y_pred = None

    if (
        not within_subject_only
        and subject_ids is not None
        and len(np.unique(subject_ids)) >= 2
    ):
        print("\n--- Cross-subject evaluation (leave-N-subjects-out) ---")
        cs_result = cross_subject_eval(X, y, subject_ids)
        cs_acc = cs_result["mean_accuracy"]
        cs_acc_std = cs_result["std_accuracy"]
        cs_f1 = cs_result["mean_f1"]
        cs_f1_std = cs_result["std_f1"]
        cs_y_true = cs_result["all_y_true"]
        cs_y_pred = cs_result["all_y_pred"]

        print(f"\nCross-subject accuracy: {cs_acc:.4f} +/- {cs_acc_std:.4f}")
        print(f"Cross-subject F1 macro: {cs_f1:.4f} +/- {cs_f1_std:.4f}")

        gap = ws_acc - cs_acc
        print(f"\nWithin-vs-cross gap: {gap:.4f} ({gap * 100:.1f} points)")
        if gap > 0.15:
            print(
                "WARNING: Gap > 15 points suggests within-subject accuracy "
                "is inflated by data leakage across subjects."
            )
    elif subject_ids is None:
        print(
            "\nSkipping cross-subject eval: no subject IDs "
            "(synthetic data has no subject structure)."
        )

    # --- Train final model on all data for deployment ---
    print("\n--- Training final model on all data ---")
    final_model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    final_model.fit(X, y)

    # Save sklearn model
    model_path = output_path / "dream_detector_model.pkl"
    joblib.dump({"model": final_model, "feature_names": feature_names}, model_path)
    print(f"Model saved to {model_path}")

    # Export ONNX
    onnx_path = output_path / "dream_detector_model.onnx"
    export_onnx(final_model, feature_names, onnx_path)

    # --- Save benchmarks ---
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    # Use cross-subject predictions for per-class metrics if available
    bench_y_test = cs_y_true if cs_y_true is not None else ws_result["y_test"]
    bench_y_pred = cs_y_pred if cs_y_pred is not None else ws_result["y_pred"]

    benchmarks = save_benchmarks(
        dataset_name=dataset_name,
        model_name="dream_detector",
        output_dir=BENCHMARK_DIR,
        training_data=training_data,
        within_subject_acc=ws_acc,
        within_subject_f1=ws_f1,
        cross_subject_acc=cs_acc,
        cross_subject_acc_std=cs_acc_std,
        cross_subject_f1=cs_f1,
        cross_subject_f1_std=cs_f1_std,
        y_test=bench_y_test,
        y_pred=bench_y_pred,
        channel_note=channel_note,
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DREAM DETECTOR TRAINING SUMMARY")
    print("=" * 60)
    print(f"Training data:          {training_data}")
    print(f"Dataset:                {dataset_name}")
    print(f"Within-subject acc:     {ws_acc:.4f}")
    if cs_acc is not None:
        print(f"Cross-subject acc:      {cs_acc:.4f} +/- {cs_acc_std:.4f}")
        print(f"Cross-subject F1:       {cs_f1:.4f} +/- {cs_f1_std:.4f}")
        print(f"Within-vs-cross gap:    {(ws_acc - cs_acc) * 100:.1f} points")
    if channel_note:
        print(f"Channel limitation:     {channel_note}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dream detection model")
    parser.add_argument(
        "--simulated",
        action="store_true",
        help="Use simulated data instead of real Sleep-EDF",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=20,
        help="Number of subjects to load from Sleep-EDF (max 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/saved",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--within-subject",
        action="store_true",
        help="Only run within-subject evaluation (skip cross-subject)",
    )
    args = parser.parse_args()

    train(
        simulated=args.simulated,
        n_subjects=args.n_subjects,
        output_dir=args.output_dir,
        within_subject_only=args.within_subject,
    )
