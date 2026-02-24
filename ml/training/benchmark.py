"""Benchmark script for all trained models.

Loads each model from models/saved/, runs against held-out test data,
and outputs benchmark results as JSON to ml/benchmarks/.

Usage:
    python -m training.benchmark
    python -m training.benchmark --dataset simulated
    python -m training.benchmark --dataset deap
    python -m training.benchmark --dataset faced
    python -m training.benchmark --dataset sleep_edf
    python -m training.benchmark --dataset all
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from scipy import stats
import joblib

from processing.eeg_processor import extract_features, preprocess
from simulation.eeg_simulator import simulate_eeg
from training.data_loaders import load_deap_or_simulate, load_faced, load_sleep_edf

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


def wilson_ci(successes, n, alpha=0.05):
    """Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    phat = successes / n
    denom = 1 + (z ** 2) / n
    center = (phat + (z ** 2) / (2 * n)) / denom
    half = (z / denom) * np.sqrt((phat * (1 - phat) + (z ** 2) / (4 * n)) / n)
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=500, alpha=0.05, seed=42):
    """Bootstrap CI for a metric computed from y_true/y_pred."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return (0.0, 0.0)
    stats_samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats_samples.append(metric_fn(y_true[idx], y_pred[idx]))
    lo = float(np.percentile(stats_samples, 100 * (alpha / 2)))
    hi = float(np.percentile(stats_samples, 100 * (1 - alpha / 2)))
    return (lo, hi)


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
    correct = int((y_test == y_pred).sum())
    n = int(len(y_test))
    chance = 1.0 / max(len(class_names), 1)
    p_value = float(stats.binomtest(correct, n, chance, alternative="greater").pvalue)
    acc_ci = wilson_ci(correct, n, alpha=0.05)
    f1_ci = bootstrap_ci(y_test, y_pred, lambda yt, yp: f1_score(yt, yp, average="macro"))
    kappa = float(cohen_kappa_score(y_test, y_pred))

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 macro: {f1:.4f}")
    print(f"  Inference time: {inference_time_ms:.2f} ms/sample")

    return {
        "model_name": model_name,
        "dataset": "simulated-benchmark",
        "accuracy": accuracy,
        "accuracy_ci_95": acc_ci,
        "accuracy_p_value_vs_chance": p_value,
        "f1_macro": f1,
        "f1_macro_ci_95": f1_ci,
        "cohen_kappa": kappa,
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


def _compute_metrics(y_test, y_pred, class_names, model_name, dataset_name, inference_time_ms):
    """Shared metric computation used by both simulated and real-dataset benchmarks."""
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred).tolist()
    correct = int((y_test == y_pred).sum())
    n = int(len(y_test))
    chance = 1.0 / max(len(class_names), 1)
    p_value = float(stats.binomtest(correct, n, chance, alternative="greater").pvalue)
    acc_ci = wilson_ci(correct, n, alpha=0.05)
    f1_ci = bootstrap_ci(
        y_test, y_pred,
        lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
    )
    kappa = float(cohen_kappa_score(y_test, y_pred))

    print(f"  Accuracy: {accuracy:.4f}  (95% CI: {acc_ci[0]:.4f}–{acc_ci[1]:.4f}, p={p_value:.4g})")
    print(f"  F1 macro: {f1:.4f}  (95% CI: {f1_ci[0]:.4f}–{f1_ci[1]:.4f})")
    print(f"  Cohen's κ: {kappa:.4f}")
    print(f"  Inference time: {inference_time_ms:.2f} ms/sample")

    return {
        "model_name": model_name,
        "dataset": dataset_name,
        "accuracy": accuracy,
        "accuracy_ci_95": acc_ci,
        "accuracy_p_value_vs_chance": p_value,
        "f1_macro": f1,
        "f1_macro_ci_95": f1_ci,
        "cohen_kappa": kappa,
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
        "n_test_samples": n,
    }


def benchmark_on_real_dataset(model_path, model_name, dataset_name, class_names,
                               fs=256.0, epoch_sec=30.0):
    """Benchmark a model against a real EEG dataset.

    Supported datasets:
        "deap"      — DEAP (or simulated fallback via load_deap_or_simulate)
        "faced"     — FACED (raises FileNotFoundError if data not downloaded)
        "sleep_edf" — Sleep-EDF (raw signals → feature extraction per epoch)

    Returns:
        dict with benchmark results, or None if dataset unavailable.
    """
    print(f"\nBenchmarking {model_name} on {dataset_name}...")

    # Load model
    data = joblib.load(model_path)
    model = data["model"]

    if dataset_name == "deap":
        print("  Loading DEAP dataset (with simulation fallback)...")
        X, y = load_deap_or_simulate(epoch_sec=epoch_sec)
        dataset_label = "deap-or-simulated"

    elif dataset_name == "faced":
        try:
            print("  Loading FACED dataset...")
            X, y = load_faced(epoch_sec=epoch_sec)
            dataset_label = "faced"
        except FileNotFoundError:
            print(f"  Skipping FACED — data not found in data/faced/. "
                  "Download from https://synapse.org/faced")
            return None

    elif dataset_name == "sleep_edf":
        print("  Loading Sleep-EDF dataset and extracting features per epoch...")
        try:
            X_raw, y = load_sleep_edf(epoch_sec=epoch_sec)
        except FileNotFoundError:
            print("  Skipping Sleep-EDF — data not found. "
                  "Run: python -m training.data_loaders download sleep_edf")
            return None
        # X_raw is (n_epochs, n_samples) — extract features per epoch
        feature_rows = []
        for epoch in X_raw:
            features = extract_features(preprocess(epoch, fs), fs)
            feature_rows.append(list(features.values()))
        X = np.array(feature_rows)
        dataset_label = "sleep-edf"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}. "
                         "Choose from: deap, faced, sleep_edf")

    print(f"  Loaded {len(y)} samples across {len(np.unique(y))} classes.")

    # Align feature count with what the model expects
    n_model_features = model.n_features_in_ if hasattr(model, "n_features_in_") else X.shape[1]
    if X.shape[1] != n_model_features:
        print(f"  WARNING: feature mismatch — dataset has {X.shape[1]}, "
              f"model expects {n_model_features}. Padding/truncating.")
        if X.shape[1] < n_model_features:
            pad = np.zeros((X.shape[0], n_model_features - X.shape[1]))
            X = np.hstack([X, pad])
        else:
            X = X[:, :n_model_features]

    # Time inference
    start = time.time()
    y_pred = model.predict(X)
    elapsed = time.time() - start
    inference_time_ms = (elapsed / max(len(X), 1)) * 1000

    result = _compute_metrics(y, y_pred, class_names, model_name, dataset_label, inference_time_ms)
    result["n_classes_in_data"] = int(len(np.unique(y)))
    return result


def run_benchmarks(n_test=50, dataset="simulated"):
    """Run benchmarks for all available models.

    Args:
        n_test:  Number of test samples per class (simulated mode only).
        dataset: One of "simulated", "deap", "faced", "sleep_edf", or "all".
    """
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    dataset = dataset.lower()

    def _save(key, bench, filename):
        results[key] = bench
        out = BENCHMARK_DIR / filename
        with open(out, "w") as f:
            json.dump(bench, f, indent=2)
        print(f"  Saved → {out}")

    # ------------------------------------------------------------------ #
    # Simulated benchmarks (always available, no download needed)          #
    # ------------------------------------------------------------------ #
    if dataset in ("simulated", "all"):
        sleep_path = MODEL_DIR / "sleep_staging_model.pkl"
        if sleep_path.exists():
            bench = benchmark_model(
                sleep_path, SLEEP_STAGES, SLEEP_STATE_MAP,
                "sleep_staging", n_test=n_test, epoch_sec=30.0,
            )
            _save("sleep_staging_simulated", bench, "sleep_staging_benchmark.json")
        else:
            print(f"Skipping sleep staging — model not found at {sleep_path}")

        emotion_path = MODEL_DIR / "emotion_classifier_model.pkl"
        if emotion_path.exists():
            bench = benchmark_model(
                emotion_path, EMOTIONS, EMOTION_STATE_MAP,
                "emotion_classifier", n_test=n_test, epoch_sec=10.0,
            )
            _save("emotion_classifier_simulated", bench, "emotion_classifier_benchmark.json")
        else:
            print(f"Skipping emotion classifier — model not found at {emotion_path}")

        dream_path = MODEL_DIR / "dream_detector_model.pkl"
        if dream_path.exists():
            dream_state_map = {0: "deep_sleep", 1: "rem"}
            bench = benchmark_model(
                dream_path, DREAM_CLASSES, dream_state_map,
                "dream_detector", n_test=n_test, epoch_sec=30.0,
            )
            _save("dream_detector_simulated", bench, "dream_detector_benchmark.json")
        else:
            print(f"Skipping dream detector — model not found at {dream_path}")

    # ------------------------------------------------------------------ #
    # DEAP (real data with simulation fallback)                            #
    # ------------------------------------------------------------------ #
    if dataset in ("deap", "all"):
        emotion_path = MODEL_DIR / "emotion_classifier_model.pkl"
        if emotion_path.exists():
            bench = benchmark_on_real_dataset(
                emotion_path, "emotion_classifier", "deap", EMOTIONS, epoch_sec=10.0,
            )
            if bench is not None:
                _save("emotion_classifier_deap", bench,
                      "emotion_classifier_deap_benchmark.json")
        else:
            print(f"Skipping DEAP emotion benchmark — model not found at {emotion_path}")

    # ------------------------------------------------------------------ #
    # FACED (requires downloaded data)                                     #
    # ------------------------------------------------------------------ #
    if dataset in ("faced", "all"):
        emotion_path = MODEL_DIR / "emotion_classifier_model.pkl"
        if emotion_path.exists():
            bench = benchmark_on_real_dataset(
                emotion_path, "emotion_classifier", "faced", EMOTIONS, epoch_sec=4.0,
            )
            if bench is not None:
                _save("emotion_classifier_faced", bench,
                      "emotion_classifier_faced_benchmark.json")
        else:
            print(f"Skipping FACED emotion benchmark — model not found at {emotion_path}")

    # ------------------------------------------------------------------ #
    # Sleep-EDF (requires downloaded data)                                 #
    # ------------------------------------------------------------------ #
    if dataset in ("sleep_edf", "all"):
        sleep_path = MODEL_DIR / "sleep_staging_model.pkl"
        if sleep_path.exists():
            bench = benchmark_on_real_dataset(
                sleep_path, "sleep_staging", "sleep_edf", SLEEP_STAGES, epoch_sec=30.0,
            )
            if bench is not None:
                _save("sleep_staging_sleep_edf", bench,
                      "sleep_staging_sleep_edf_benchmark.json")
        else:
            print(f"Skipping Sleep-EDF staging benchmark — model not found at {sleep_path}")

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    if results:
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        for name, bench in results.items():
            print(f"  {name}: accuracy={bench['accuracy']:.4f}, "
                  f"f1={bench['f1_macro']:.4f}, "
                  f"κ={bench['cohen_kappa']:.4f}, "
                  f"inference={bench['inference_time_ms']:.2f}ms "
                  f"[dataset={bench['dataset']}]")
        print(f"\nResults saved to {BENCHMARK_DIR}/")
    else:
        print("\nNo results produced — check model paths and dataset availability.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark all trained models")
    parser.add_argument(
        "--n-test", type=int, default=50,
        help="Number of test samples per class (simulated mode only)",
    )
    parser.add_argument(
        "--dataset",
        choices=["simulated", "deap", "faced", "sleep_edf", "all"],
        default="simulated",
        help=(
            "Dataset to benchmark against. "
            "'simulated' uses EEG simulation (always works). "
            "'deap' uses DEAP with simulation fallback. "
            "'faced' requires data/faced/ downloads. "
            "'sleep_edf' requires Sleep-EDF downloads. "
            "'all' runs every available benchmark."
        ),
    )
    args = parser.parse_args()

    run_benchmarks(n_test=args.n_test, dataset=args.dataset)
