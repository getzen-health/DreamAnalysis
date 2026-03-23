"""Retrain dream detector with eye movement features.

Loads Sleep-EDF data via data_loaders.load_rem_detection_with_subjects().
If Sleep-EDF is not available locally, generates enhanced synthetic data
that includes realistic eye movement patterns:
  - REM epochs: inject rapid deflections (saccades) + high theta + low delta
  - Non-REM N3: high delta, no saccades
  - Non-REM N1/N2: moderate features

Extracts standard band powers + eye movement features, trains
GradientBoostingClassifier, reports cross-subject accuracy, saves model.

Usage:
    python -m training.retrain_dream_real
    python -m training.retrain_dream_real --n-subjects 10
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import GradientBoostingClassifier
import joblib

from processing.eeg_processor import (
    extract_features,
    extract_eye_movement_features,
    preprocess,
)

CLASSES = ["not_dreaming", "dreaming"]
MODEL_DIR = Path("models/saved")
BENCHMARK_DIR = Path("benchmarks")


def _generate_enhanced_synthetic_epoch(
    is_rem: bool, fs: float = 256.0, duration: float = 30.0
) -> np.ndarray:
    """Generate a single synthetic EEG epoch with realistic sleep features.

    REM epochs get: high theta, low delta, injected saccade-like deflections.
    Non-REM N3 epochs get: high delta, no saccades.
    Non-REM N1/N2 epochs get: moderate mix.
    """
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs

    if is_rem:
        # REM: theta-dominant + rapid eye movement artifacts
        theta_amp = np.random.uniform(25, 40)
        delta_amp = np.random.uniform(3, 8)
        beta_amp = np.random.uniform(5, 12)

        signal = (
            theta_amp * np.sin(2 * np.pi * np.random.uniform(5, 7) * t)
            + delta_amp * np.sin(2 * np.pi * np.random.uniform(1, 3) * t)
            + beta_amp * np.sin(2 * np.pi * np.random.uniform(15, 25) * t)
            + np.random.randn(n_samples) * 5
        )

        # Inject saccade-like rapid deflections (REMs)
        n_saccades = np.random.randint(8, 25)
        spike_width = int(np.random.uniform(0.03, 0.15) * fs)
        for _ in range(n_saccades):
            pos = np.random.randint(0, max(1, n_samples - spike_width))
            amplitude = np.random.choice([-1, 1]) * np.random.uniform(55, 120)
            end = min(pos + spike_width, n_samples)
            ramp = np.linspace(0, amplitude, end - pos)
            signal[pos:end] += ramp

    else:
        # Non-REM: pick N1/N2 or N3
        stage = np.random.choice(["n1n2", "n3"], p=[0.5, 0.5])

        if stage == "n3":
            # Deep sleep: dominant delta, no saccades
            delta_amp = np.random.uniform(40, 70)
            theta_amp = np.random.uniform(3, 8)
            signal = (
                delta_amp * np.sin(2 * np.pi * np.random.uniform(0.5, 2) * t)
                + theta_amp * np.sin(2 * np.pi * np.random.uniform(5, 7) * t)
                + np.random.randn(n_samples) * 3
            )
        else:
            # N1/N2: moderate features, alpha spindles
            delta_amp = np.random.uniform(15, 30)
            theta_amp = np.random.uniform(8, 15)
            alpha_amp = np.random.uniform(10, 20)
            signal = (
                delta_amp * np.sin(2 * np.pi * np.random.uniform(1, 3) * t)
                + theta_amp * np.sin(2 * np.pi * np.random.uniform(5, 7) * t)
                + alpha_amp * np.sin(2 * np.pi * np.random.uniform(9, 11) * t)
                + np.random.randn(n_samples) * 5
            )

    return signal


def generate_enhanced_synthetic_data(
    n_samples_per_class: int = 300,
    fs: float = 256.0,
    duration: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Generate enhanced synthetic data with eye movement patterns.

    Returns:
        (X_features, y, feature_names, subject_ids)
    """
    np.random.seed(42)

    # Simulate 6 synthetic "subjects" for cross-subject eval
    n_subjects = 6
    epochs_per_subject = n_samples_per_class * 2 // n_subjects

    X_features = []
    y_labels = []
    subject_ids = []
    feature_names = None

    for subj in range(n_subjects):
        for ep in range(epochs_per_subject):
            is_rem = ep < (epochs_per_subject // 2)
            label = 1 if is_rem else 0

            epoch = _generate_enhanced_synthetic_epoch(is_rem, fs, duration)
            processed = preprocess(epoch, fs)

            # Standard features
            feats = extract_features(processed, fs)

            # Eye movement features
            eye_feats = extract_eye_movement_features(epoch, fs)
            feats.update({f"eye_{k}": v for k, v in eye_feats.items()})

            X_features.append(list(feats.values()))
            y_labels.append(label)
            subject_ids.append(subj)

            if feature_names is None:
                feature_names = list(feats.keys())

    return (
        np.array(X_features),
        np.array(y_labels),
        feature_names,
        np.array(subject_ids),
    )


def extract_features_with_eye(epoch: np.ndarray, fs: float) -> Dict[str, float]:
    """Extract standard + eye movement features from a single epoch."""
    processed = preprocess(epoch, fs)
    feats = extract_features(processed, fs)
    eye_feats = extract_eye_movement_features(epoch, fs)
    feats.update({f"eye_{k}": v for k, v in eye_feats.items()})
    return feats


def load_real_data_with_eye_features(
    n_subjects: int = 20, fs: float = 256.0
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Load Sleep-EDF data and extract standard + eye movement features.

    Returns:
        (X_features, y, feature_names, subject_ids)
    """
    from training.data_loaders import load_rem_detection_with_subjects

    print(f"Loading REM detection data for {n_subjects} subjects...")
    X_raw, y, subject_ids = load_rem_detection_with_subjects(
        n_subjects=n_subjects, target_fs=fs
    )

    print(f"Extracting features (standard + eye movement) from {len(y)} epochs...")
    X_features = []
    feature_names = None

    for i, epoch in enumerate(X_raw):
        feats = extract_features_with_eye(epoch, fs)
        X_features.append(list(feats.values()))
        if feature_names is None:
            feature_names = list(feats.keys())
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(y)} epochs")

    return np.array(X_features), y, feature_names, subject_ids


def cross_subject_eval(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_splits: int = 5,
) -> Dict:
    """Leave-N-subjects-out cross-validation."""
    unique_subjects = np.unique(subject_ids)
    actual_splits = min(n_splits, len(unique_subjects))

    if actual_splits < 2:
        raise ValueError(
            f"Need >= 2 subjects for cross-subject eval, got {len(unique_subjects)}"
        )

    gkf = GroupKFold(n_splits=actual_splits)
    fold_accs = []
    fold_f1s = []
    all_y_true = []
    all_y_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=subject_ids)
    ):
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])

        acc = accuracy_score(y[test_idx], y_pred)
        f1 = f1_score(y[test_idx], y_pred, average="macro")
        fold_accs.append(acc)
        fold_f1s.append(f1)
        all_y_true.extend(y[test_idx])
        all_y_pred.extend(y_pred)

        train_subj = np.unique(subject_ids[train_idx])
        test_subj = np.unique(subject_ids[test_idx])
        print(
            f"  Fold {fold_idx + 1}/{actual_splits}: "
            f"acc={acc:.4f}, f1={f1:.4f} "
            f"(train: {list(train_subj)}, test: {list(test_subj)})"
        )

    return {
        "fold_accuracies": fold_accs,
        "fold_f1s": fold_f1s,
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "mean_f1": float(np.mean(fold_f1s)),
        "std_f1": float(np.std(fold_f1s)),
        "n_splits": actual_splits,
        "all_y_true": all_y_true,
        "all_y_pred": all_y_pred,
    }


def train(
    n_subjects: int = 20,
    output_dir: str = "models/saved",
):
    """Train dream detector with eye movement features.

    Tries real Sleep-EDF data first; falls back to enhanced synthetic data.
    Reports both within-subject and cross-subject accuracy.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Try real data first
    training_data = "synthetic"
    dataset_name = "enhanced_synthetic"

    try:
        print("Attempting to load real Sleep-EDF data...")
        X, y, feature_names, subject_ids = load_real_data_with_eye_features(
            n_subjects=n_subjects
        )
        training_data = "real"
        dataset_name = "sleep-edf-rem"
        print(f"Loaded real data: {X.shape[0]} epochs, {X.shape[1]} features")
    except Exception as e:
        print(f"Could not load real data: {e}")
        print("Generating enhanced synthetic data with eye movement patterns...")
        X, y, feature_names, subject_ids = generate_enhanced_synthetic_data(
            n_samples_per_class=300
        )
        print(f"Generated synthetic data: {X.shape[0]} epochs, {X.shape[1]} features")

    print(f"\nDataset: {dataset_name} (training_data: {training_data})")
    print(f"Total samples: {X.shape[0]}")
    print(f"Features: {len(feature_names)}")
    print(f"Class balance: {np.bincount(y)}")
    print(f"Subjects: {len(np.unique(subject_ids))}")

    # Eye movement feature names in the vector
    eye_feat_names = [n for n in feature_names if n.startswith("eye_")]
    print(f"Eye movement features: {eye_feat_names}")

    # Cross-subject evaluation
    print("\n--- Cross-subject evaluation (leave-N-subjects-out) ---")
    cs_result = cross_subject_eval(X, y, subject_ids)
    cs_acc = cs_result["mean_accuracy"]
    cs_acc_std = cs_result["std_accuracy"]
    cs_f1 = cs_result["mean_f1"]
    cs_f1_std = cs_result["std_f1"]

    print(f"\nCross-subject accuracy: {cs_acc:.4f} +/- {cs_acc_std:.4f}")
    print(f"Cross-subject F1 macro: {cs_f1:.4f} +/- {cs_f1_std:.4f}")

    print("\nClassification report (cross-subject, all folds):")
    print(
        classification_report(
            cs_result["all_y_true"],
            cs_result["all_y_pred"],
            target_names=CLASSES,
        )
    )

    # Train final model on all data
    print("\n--- Training final model on all data ---")
    final_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    final_model.fit(X, y)

    # Save model
    model_path = output_path / "dream_detector_model.pkl"
    joblib.dump(
        {
            "model": final_model,
            "feature_names": feature_names,
            "includes_eye_features": True,
        },
        model_path,
    )
    print(f"Model saved to {model_path}")

    # Save benchmarks
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(
        cs_result["all_y_true"], cs_result["all_y_pred"]
    ).tolist()
    report = classification_report(
        cs_result["all_y_true"],
        cs_result["all_y_pred"],
        target_names=CLASSES,
        output_dict=True,
    )

    benchmarks = {
        "model_name": "dream_detector",
        "dataset": dataset_name,
        "training_data": training_data,
        "features": "standard_band_powers + eye_movement",
        "cross_subject_accuracy": cs_acc,
        "cross_subject_accuracy_std": cs_acc_std,
        "cross_subject_f1_macro": cs_f1,
        "cross_subject_f1_macro_std": cs_f1_std,
        "accuracy": cs_acc,
        "f1_macro": cs_f1,
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": report[cls]["support"],
            }
            for cls in CLASSES
            if cls in report
        },
        "confusion_matrix": cm,
        "channel_note": (
            "Sleep-EDF provides 2 EEG channels (Fpz-Cz, Pz-Oz). "
            "Training uses Fpz-Cz only. Muse 2 has 4 channels "
            "(TP9, AF7, AF8, TP10) at different positions -- "
            "expect a domain gap when deploying on Muse hardware."
        ),
    }

    benchmark_path = BENCHMARK_DIR / "dream_detector_benchmark.json"
    with open(benchmark_path, "w") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"Benchmarks saved to {benchmark_path}")

    # Summary
    print("\n" + "=" * 60)
    print("DREAM DETECTOR RETRAIN SUMMARY (with eye movement features)")
    print("=" * 60)
    print(f"Training data:          {training_data}")
    print(f"Dataset:                {dataset_name}")
    print(f"Features:               {len(feature_names)} (incl. {len(eye_feat_names)} eye movement)")
    print(f"Cross-subject acc:      {cs_acc:.4f} +/- {cs_acc_std:.4f}")
    print(f"Cross-subject F1:       {cs_f1:.4f} +/- {cs_f1_std:.4f}")
    print("=" * 60)

    return benchmarks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrain dream detector with eye movement features"
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
    args = parser.parse_args()

    train(n_subjects=args.n_subjects, output_dir=args.output_dir)
