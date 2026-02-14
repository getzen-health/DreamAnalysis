"""Benchmark script for all trained models.

Loads each model from models/saved/, runs against held-out test data,
and outputs benchmark results as JSON to ml/benchmarks/.

Usage:
    python -m training.benchmark
    python -m training.benchmark --simulated
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import joblib

from processing.eeg_processor import extract_features, preprocess
from simulation.eeg_simulator import simulate_eeg

MODEL_DIR = Path("models/saved")
BENCHMARK_DIR = Path("benchmarks")

SLEEP_STAGES = ["Wake", "N1", "N2", "N3", "REM"]
EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]
DREAM_CLASSES = ["not_dreaming", "dreaming"]

SLEEP_STATE_MAP = {0: "rest", 1: "light_sleep", 2: "light_sleep", 3: "deep_sleep", 4: "rem"}
EMOTION_STATE_MAP = {0: "rest", 1: "deep_sleep", 2: "stress", 3: "stress", 4: "meditation", 5: "focus"}


def generate_test_data(state_map, n_per_class=50, fs=256.0, epoch_sec=30.0):
    """Generate test data from simulation."""
    X, y = [], []
    feature_names = None

    for label, state in state_map.items():
        for _ in range(n_per_class):
            result = simulate_eeg(state=state, duration=epoch_sec, fs=fs)
            eeg = np.array(result["signals"][0])
            features = extract_features(preprocess(eeg, fs), fs)
            X.append(list(features.values()))
            y.append(label)
            if feature_names is None:
                feature_names = list(features.keys())

    return np.array(X), np.array(y), feature_names


def benchmark_model(model_path, class_names, state_map, model_name, n_test=50, epoch_sec=30.0):
    """Benchmark a single model and return results dict."""
    print(f"\nBenchmarking {model_name}...")

    # Load model
    data = joblib.load(model_path)
    model = data["model"]
    feature_names = data["feature_names"]

    # Generate test data
    print(f"  Generating {n_test} test samples per class...")
    X_test, y_test, _ = generate_test_data(state_map, n_per_class=n_test, epoch_sec=epoch_sec)

    # Time inference
    start = time.time()
    y_pred = model.predict(X_test)
    elapsed = time.time() - start
    inference_time_ms = (elapsed / len(X_test)) * 1000

    # Compute metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 macro: {f1:.4f}")
    print(f"  Inference time: {inference_time_ms:.2f} ms/sample")

    return {
        "model_name": model_name,
        "dataset": "simulated-benchmark",
        "accuracy": accuracy,
        "f1_macro": f1,
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in class_names
            if name in report
        },
        "confusion_matrix": cm,
        "inference_time_ms": round(inference_time_ms, 3),
        "n_test_samples": len(y_test),
    }


def run_benchmarks(n_test=50):
    """Run benchmarks for all available models."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # Sleep staging
    sleep_path = MODEL_DIR / "sleep_staging_model.pkl"
    if sleep_path.exists():
        bench = benchmark_model(
            sleep_path, SLEEP_STAGES, SLEEP_STATE_MAP,
            "sleep_staging", n_test=n_test, epoch_sec=30.0,
        )
        results["sleep_staging"] = bench
        with open(BENCHMARK_DIR / "sleep_staging_benchmark.json", "w") as f:
            json.dump(bench, f, indent=2)
    else:
        print(f"Skipping sleep staging - model not found at {sleep_path}")

    # Emotion classifier
    emotion_path = MODEL_DIR / "emotion_classifier_model.pkl"
    if emotion_path.exists():
        bench = benchmark_model(
            emotion_path, EMOTIONS, EMOTION_STATE_MAP,
            "emotion_classifier", n_test=n_test, epoch_sec=10.0,
        )
        results["emotion_classifier"] = bench
        with open(BENCHMARK_DIR / "emotion_classifier_benchmark.json", "w") as f:
            json.dump(bench, f, indent=2)
    else:
        print(f"Skipping emotion classifier - model not found at {emotion_path}")

    # Dream detector
    dream_path = MODEL_DIR / "dream_detector_model.pkl"
    if dream_path.exists():
        dream_state_map = {0: "deep_sleep", 1: "rem"}
        bench = benchmark_model(
            dream_path, DREAM_CLASSES, dream_state_map,
            "dream_detector", n_test=n_test, epoch_sec=30.0,
        )
        results["dream_detector"] = bench
        with open(BENCHMARK_DIR / "dream_detector_benchmark.json", "w") as f:
            json.dump(bench, f, indent=2)
    else:
        print(f"Skipping dream detector - model not found at {dream_path}")

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for name, bench in results.items():
        print(f"  {name}: accuracy={bench['accuracy']:.4f}, f1={bench['f1_macro']:.4f}, "
              f"inference={bench['inference_time_ms']:.2f}ms")

    print(f"\nResults saved to {BENCHMARK_DIR}/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark all trained models")
    parser.add_argument(
        "--n-test", type=int, default=50,
        help="Number of test samples per class"
    )
    args = parser.parse_args()

    run_benchmarks(n_test=args.n_test)
