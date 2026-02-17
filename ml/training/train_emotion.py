"""Training script for emotion classifier.

Trains on DEAP dataset if available, otherwise uses enhanced simulation.
Exports to both sklearn .pkl and ONNX formats with benchmark results.

Usage:
    python -m training.train_emotion
    python -m training.train_emotion --simulated
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

EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]
EMOTION_TO_STATE = {
    0: "rest",        # happy -> relaxed state with positive valence
    1: "deep_sleep",  # sad -> low arousal, low energy
    2: "stress",      # angry -> high beta, high arousal
    3: "stress",      # fearful -> high beta, high gamma
    4: "meditation",  # relaxed -> high alpha, high theta
    5: "focus",       # focused -> high beta
}
MODEL_DIR = Path("models/saved")
BENCHMARK_DIR = Path("benchmarks")


def generate_training_data(n_samples_per_class: int = 150, fs: float = 256.0, epoch_duration: float = 10.0):
    """Generate synthetic training data."""
    X = []
    y = []

    for emotion_idx, state_name in EMOTION_TO_STATE.items():
        for _ in range(n_samples_per_class):
            result = simulate_eeg(state=state_name, duration=epoch_duration, fs=fs)
            eeg = np.array(result["signals"][0])
            features = extract_features(preprocess(eeg, fs), fs)
            X.append(list(features.values()))
            y.append(emotion_idx)

    return np.array(X), np.array(y), list(features.keys())


def load_real_data(data_dir: str = "data/deap", fs: float = 256.0):
    """Load real EEG data, combining DENS + DEAP when both are available."""
    from training.data_loaders import load_dens, _load_deap_real

    dummy_features = extract_features(preprocess(np.random.randn(int(fs * 10)), fs), fs)
    feature_names = list(dummy_features.keys())

    datasets_loaded = []
    X_parts, y_parts = [], []

    # Try DENS (128-ch, OpenNeuro)
    dens_path = Path("data/dens")
    if dens_path.exists() and list(dens_path.glob("sub-*")):
        try:
            print("Loading DENS dataset (128-ch EEG, OpenNeuro)...")
            X_dens, y_dens = load_dens(data_dir="data/dens", target_fs=fs)
            X_parts.append(X_dens)
            y_parts.append(y_dens)
            datasets_loaded.append(f"dens({len(y_dens)} samples)")
            print(f"  DENS: {len(y_dens)} samples loaded")
        except Exception as e:
            print(f"  DENS failed: {e}")

    # Try DEAP (32-ch, Kaggle)
    deap_path = Path(data_dir)
    if deap_path.exists() and list(deap_path.glob("s*.dat")):
        try:
            print("Loading DEAP dataset (32-ch EEG, Kaggle)...")
            X_deap, y_deap = _load_deap_real(deap_path, fs)
            X_parts.append(X_deap)
            y_parts.append(y_deap)
            datasets_loaded.append(f"deap({len(y_deap)} samples)")
            print(f"  DEAP: {len(y_deap)} samples loaded")
        except Exception as e:
            print(f"  DEAP failed: {e}")

    if X_parts:
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        print(f"Combined dataset: {len(y)} samples from {', '.join(datasets_loaded)}")
        return X, y, feature_names

    # Fallback to cascade loader
    from training.data_loaders import load_deap_or_simulate
    print("No local data found, using cascade loader...")
    X, y = load_deap_or_simulate(data_dir=data_dir, fs=fs)
    return X, y, feature_names


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

    present_classes = sorted(set(y_test) | set(y_pred))
    present_names = [EMOTIONS[i] for i in present_classes if i < len(EMOTIONS)]
    report = classification_report(y_test, y_pred, labels=present_classes, target_names=present_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=present_classes).tolist()

    benchmarks = {
        "model_name": model_name,
        "dataset": dataset_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "per_class": {
            emotion: {
                "precision": report[emotion]["precision"],
                "recall": report[emotion]["recall"],
                "f1": report[emotion]["f1-score"],
                "support": report[emotion]["support"],
            }
            for emotion in present_names
            if emotion in report
        },
        "confusion_matrix": cm,
    }

    benchmark_path = output_dir / f"{model_name}_benchmark.json"
    with open(benchmark_path, "w") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"Benchmarks saved to {benchmark_path}")

    return benchmarks


def train(simulated: bool = False, data_dir: str = "data/deap", output_dir: str = "models/saved"):
    """Train and save the emotion classifier."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if simulated:
        print("Generating simulated training data...")
        X, y, feature_names = generate_training_data(n_samples_per_class=150)
        dataset_name = "simulated"
    else:
        print("Loading emotion dataset...")
        try:
            X, y, feature_names = load_real_data(data_dir=data_dir)
            # Detect which datasets were loaded
            dens_path = Path("data/dens")
            deap_path = Path(data_dir)
            dens_ok = dens_path.exists() and list(dens_path.glob("sub-*"))
            deap_ok = deap_path.exists() and list(deap_path.glob("s*.dat"))
            if dens_ok and deap_ok:
                dataset_name = "dens+deap-combined"
            elif dens_ok:
                dataset_name = "dens-openneuro"
            elif deap_ok:
                dataset_name = "deap-kaggle"
            else:
                dataset_name = "simulated-enhanced"
        except Exception as e:
            print(f"Failed to load data: {e}")
            print("Falling back to simulated data...")
            X, y, feature_names = generate_training_data(n_samples_per_class=150)
            dataset_name = "simulated"

    # Use stratify only if each class has enough samples
    from collections import Counter
    class_counts = Counter(y)
    min_count = min(class_counts.values())
    stratify_param = y if min_count >= 2 else None
    if stratify_param is None:
        print("  Warning: some classes have <2 samples, disabling stratification")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    print(f"Dataset: {dataset_name}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Compute class weights to handle imbalanced data
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.ensemble import RandomForestClassifier

    sample_weights = compute_sample_weight("balanced", y_train)

    gbm = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
    )

    # Train both and pick best, or use ensemble
    print("Training GradientBoosting...")
    gbm.fit(X_train, y_train, sample_weight=sample_weights)
    gbm_acc = accuracy_score(y_test, gbm.predict(X_test))
    print(f"  GBM accuracy: {gbm_acc:.4f}")

    print("Training RandomForest...")
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"  RF accuracy: {rf_acc:.4f}")

    # Use the better model
    if rf_acc > gbm_acc:
        model = rf
        print(f"Selected RandomForest (accuracy: {rf_acc:.4f})")
    else:
        model = gbm
        print(f"Selected GradientBoosting (accuracy: {gbm_acc:.4f})")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    # Only include classes that are present in the data
    present_classes = sorted(set(y_test) | set(y_pred))
    present_names = [EMOTIONS[i] for i in present_classes if i < len(EMOTIONS)]
    print(classification_report(y_test, y_pred, labels=present_classes, target_names=present_names))

    # Save sklearn model
    model_path = output_path / "emotion_classifier_model.pkl"
    joblib.dump({"model": model, "feature_names": feature_names}, model_path)
    print(f"Model saved to {model_path}")

    # Export ONNX
    onnx_path = output_path / "emotion_classifier_model.onnx"
    export_onnx(model, feature_names, onnx_path)

    # Save benchmarks
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    benchmarks = save_benchmarks(
        y_test, y_pred, dataset_name, "emotion_classifier", BENCHMARK_DIR
    )

    print(f"\nFinal accuracy: {benchmarks['accuracy']:.4f}")
    print(f"F1 macro: {benchmarks['f1_macro']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train emotion classifier")
    parser.add_argument(
        "--simulated", action="store_true",
        help="Use simulated data instead of DEAP"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/deap",
        help="Path to DEAP .dat files"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/saved",
        help="Directory to save trained models"
    )
    args = parser.parse_args()

    train(
        simulated=args.simulated,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
