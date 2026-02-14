"""Training script for dream (REM) detection model.

Trains on Sleep-EDF data with binary REM/non-REM labels.
Exports to both sklearn .pkl and ONNX formats with benchmark results.

Usage:
    python -m training.train_dream
    python -m training.train_dream --simulated
    python -m training.train_dream --n-subjects 5
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import joblib

from processing.eeg_processor import extract_features, preprocess
from simulation.eeg_simulator import simulate_eeg

CLASSES = ["not_dreaming", "dreaming"]
MODEL_DIR = Path("models/saved")
BENCHMARK_DIR = Path("benchmarks")


def generate_training_data(n_samples_per_class: int = 300, fs: float = 256.0, epoch_duration: float = 30.0):
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

    return np.array(X), np.array(y), list(features.keys())


def load_real_data(n_subjects: int = 20, fs: float = 256.0):
    """Load real Sleep-EDF data with binary REM labels."""
    from training.data_loaders import load_rem_detection

    print(f"Loading REM detection data for {n_subjects} subjects...")
    X_raw, y = load_rem_detection(n_subjects=n_subjects, target_fs=fs)

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

    return np.array(X_features), y, feature_names


def export_onnx(model, feature_names, output_path):
    """Export sklearn model to ONNX format."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX model saved to {output_path}")
        return True
    except ImportError:
        print("skl2onnx not installed, skipping ONNX export")
        return False


def save_benchmarks(y_test, y_pred, dataset_name, model_name, output_dir):
    """Save benchmark results as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    benchmarks = {
        "model_name": model_name,
        "dataset": dataset_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
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
    }

    benchmark_path = output_dir / f"{model_name}_benchmark.json"
    with open(benchmark_path, "w") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"Benchmarks saved to {benchmark_path}")

    return benchmarks


def train(simulated: bool = False, n_subjects: int = 20, output_dir: str = "models/saved"):
    """Train and save the dream detection model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if simulated:
        print("Generating simulated training data...")
        X, y, feature_names = generate_training_data(n_samples_per_class=300)
        dataset_name = "simulated"
    else:
        print("Loading real Sleep-EDF dataset for REM detection...")
        try:
            X, y, feature_names = load_real_data(n_subjects=n_subjects)
            dataset_name = "sleep-edf-rem"
        except Exception as e:
            print(f"Failed to load real data: {e}")
            print("Falling back to simulated data...")
            X, y, feature_names = generate_training_data(n_samples_per_class=300)
            dataset_name = "simulated"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Dataset: {dataset_name}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {len(feature_names)}")
    print(f"Class balance: {np.bincount(y_train)} (train), {np.bincount(y_test)} (test)")

    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # Save sklearn model
    model_path = output_path / "dream_detector_model.pkl"
    joblib.dump({"model": model, "feature_names": feature_names}, model_path)
    print(f"Model saved to {model_path}")

    # Export ONNX
    onnx_path = output_path / "dream_detector_model.onnx"
    export_onnx(model, feature_names, onnx_path)

    # Save benchmarks
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    benchmarks = save_benchmarks(
        y_test, y_pred, dataset_name, "dream_detector", BENCHMARK_DIR
    )

    print(f"\nFinal accuracy: {benchmarks['accuracy']:.4f}")
    print(f"F1 macro: {benchmarks['f1_macro']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dream detection model")
    parser.add_argument(
        "--simulated", action="store_true",
        help="Use simulated data instead of real Sleep-EDF"
    )
    parser.add_argument(
        "--n-subjects", type=int, default=20,
        help="Number of subjects to load from Sleep-EDF (max 20)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/saved",
        help="Directory to save trained models"
    )
    args = parser.parse_args()

    train(
        simulated=args.simulated,
        n_subjects=args.n_subjects,
        output_dir=args.output_dir,
    )
