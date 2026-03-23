"""Benchmark against consumer-grade EEG datasets.

Targets the Nature Scientific Data 2026 consumer-EEG dataset and runs
existing AntarAI models against it to measure real-world accuracy on
consumer-grade hardware.

Dataset reference:
    Title: "A consumer-grade EEG dataset for emotion recognition"
    DOI: 10.6084/m9.figshare.30162868
    Published: Nature Scientific Data, 2026
    Device: Consumer-grade EEG headset (comparable to Muse 2)
    Subjects: Multiple subjects with emotion-labeled EEG recordings
    Sampling rate: 256 Hz
    Labels: Valence and arousal (continuous), mapped to discrete emotions

Usage:
    # Check dataset availability
    python -m training.benchmark_consumer_eeg --check

    # Download instructions
    python -m training.benchmark_consumer_eeg --download-instructions

    # Run benchmark (requires downloaded data)
    python -m training.benchmark_consumer_eeg --run

    # Run with specific model
    python -m training.benchmark_consumer_eeg --run --model emotion_mega_lgbm
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/consumer_eeg_2026")
MODEL_DIR = Path("models/saved")
BENCHMARK_DIR = Path("benchmarks")
RESULTS_FILE = BENCHMARK_DIR / "consumer_eeg_benchmark.json"

DATASET_DOI = "10.6084/m9.figshare.30162868"
DATASET_URL = f"https://doi.org/{DATASET_DOI}"

# Expected directory structure after download
EXPECTED_FILES = [
    "README.md",
    "participants.tsv",
]

MODELS_TO_BENCHMARK = [
    {
        "name": "emotion_mega_lgbm",
        "file": "emotion_mega_lgbm.pkl",
        "type": "sklearn",
        "description": "Mega LightGBM — 11 datasets, 187K samples, 71.52% CV",
    },
    {
        "name": "emotion_lgbm_muse_live",
        "file": "emotion_lgbm_muse_live.pkl",
        "type": "sklearn",
        "description": "Muse-native LightGBM — no PCA, 69.25% CV",
    },
    {
        "name": "eegnet_emotion_4ch",
        "file": "eegnet_emotion_4ch.pt",
        "type": "pytorch",
        "description": "EEGNet 4-channel — 85.00% CV, active live path",
    },
    {
        "name": "sleep_staging",
        "file": "sleep_staging_model.pkl",
        "type": "sklearn",
        "description": "Sleep staging LightGBM — 92.98% CV on ISRUC",
    },
]


def check_dataset() -> bool:
    """Check if the consumer EEG dataset is downloaded and accessible."""
    if not DATA_DIR.exists():
        print(f"Dataset directory not found: {DATA_DIR}")
        return False

    files = list(DATA_DIR.rglob("*"))
    if len(files) == 0:
        print(f"Dataset directory is empty: {DATA_DIR}")
        return False

    print(f"Dataset directory found: {DATA_DIR}")
    print(f"  Files found: {len(files)}")
    for f in sorted(files)[:20]:
        print(f"    {f.relative_to(DATA_DIR)}")
    if len(files) > 20:
        print(f"    ... and {len(files) - 20} more")
    return True


def print_download_instructions() -> None:
    """Print instructions for downloading the dataset."""
    print("=" * 70)
    print("Consumer-Grade EEG Dataset — Download Instructions")
    print("=" * 70)
    print()
    print(f"Dataset: Nature Scientific Data 2026")
    print(f"DOI: {DATASET_DOI}")
    print(f"URL: {DATASET_URL}")
    print()
    print("Steps:")
    print(f"  1. Visit {DATASET_URL}")
    print(f"  2. Download all data files from the figshare repository")
    print(f"  3. Extract into: {DATA_DIR.resolve()}")
    print(f"  4. Verify with: python -m training.benchmark_consumer_eeg --check")
    print()
    print("Expected directory structure:")
    print(f"  {DATA_DIR}/")
    print(f"    README.md")
    print(f"    participants.tsv")
    print(f"    sub-01/")
    print(f"      eeg/")
    print(f"        sub-01_task-emotion_eeg.npy (or .csv / .mat)")
    print(f"        sub-01_task-emotion_labels.tsv")
    print(f"    sub-02/")
    print(f"    ...")
    print()
    print("After downloading, run the benchmark:")
    print(f"  python -m training.benchmark_consumer_eeg --run")
    print("=" * 70)


def load_consumer_eeg_data(
    max_subjects: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the consumer EEG dataset.

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Label array (n_samples,)

    Raises:
        FileNotFoundError: If dataset is not downloaded.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Consumer EEG dataset not found at {DATA_DIR}. "
            f"Run with --download-instructions for setup guide."
        )

    # Lazy import heavy dependencies only when actually running
    from processing.eeg_processor import extract_features, preprocess

    subject_dirs = sorted(
        [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    )
    if not subject_dirs:
        raise FileNotFoundError(
            f"No subject directories (sub-XX/) found in {DATA_DIR}. "
            f"Check download instructions."
        )

    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]

    logger.info("Loading %d subjects from %s", len(subject_dirs), DATA_DIR)
    all_features = []
    all_labels = []

    for subj_dir in subject_dirs:
        eeg_dir = subj_dir / "eeg"
        if not eeg_dir.exists():
            logger.warning("Skipping %s — no eeg/ subdirectory", subj_dir.name)
            continue

        # Look for numpy, CSV, or MAT files
        eeg_files = (
            list(eeg_dir.glob("*_eeg.npy"))
            + list(eeg_dir.glob("*_eeg.csv"))
            + list(eeg_dir.glob("*_eeg.mat"))
        )
        label_files = (
            list(eeg_dir.glob("*_labels.tsv"))
            + list(eeg_dir.glob("*_labels.csv"))
        )

        if not eeg_files or not label_files:
            logger.warning(
                "Skipping %s — missing eeg or label files", subj_dir.name
            )
            continue

        eeg_file = eeg_files[0]
        label_file = label_files[0]

        # Load EEG data
        if eeg_file.suffix == ".npy":
            raw = np.load(eeg_file)
        elif eeg_file.suffix == ".csv":
            raw = np.loadtxt(eeg_file, delimiter=",", skiprows=1)
        elif eeg_file.suffix == ".mat":
            import scipy.io
            mat = scipy.io.loadmat(str(eeg_file))
            # Find the main data key (skip __header__, __version__, __globals__)
            data_keys = [k for k in mat.keys() if not k.startswith("__")]
            if not data_keys:
                logger.warning("No data keys in %s", eeg_file)
                continue
            raw = mat[data_keys[0]]
        else:
            continue

        # Load labels
        if label_file.suffix == ".tsv":
            labels = np.loadtxt(label_file, delimiter="\t", skiprows=1, usecols=-1)
        else:
            labels = np.loadtxt(label_file, delimiter=",", skiprows=1, usecols=-1)

        # Extract features per epoch
        # Assume raw shape is (n_epochs, n_channels, n_samples) or (n_epochs, n_samples)
        if raw.ndim == 3:
            # Multichannel: use first channel for feature extraction
            for i, epoch in enumerate(raw):
                if i >= len(labels):
                    break
                features = extract_features(preprocess(epoch[0], 256.0), 256.0)
                all_features.append(list(features.values()))
                all_labels.append(int(labels[i]))
        elif raw.ndim == 2:
            # Single channel epochs: (n_epochs, n_samples)
            for i, epoch in enumerate(raw):
                if i >= len(labels):
                    break
                features = extract_features(preprocess(epoch, 256.0), 256.0)
                all_features.append(list(features.values()))
                all_labels.append(int(labels[i]))
        else:
            logger.warning("Unexpected data shape %s in %s", raw.shape, eeg_file)
            continue

        logger.info(
            "  Loaded %s: %d epochs", subj_dir.name, min(len(raw), len(labels))
        )

    if not all_features:
        raise FileNotFoundError(
            f"No valid EEG data loaded from {DATA_DIR}. Check file format."
        )

    return np.array(all_features), np.array(all_labels)


def run_benchmark(
    model_filter: Optional[str] = None,
    max_subjects: Optional[int] = None,
) -> dict:
    """Run benchmark of existing models against the consumer EEG dataset.

    Args:
        model_filter: If set, only benchmark this model name.
        max_subjects: Limit number of subjects to load (for quick testing).

    Returns:
        Dictionary of benchmark results per model.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
        cohen_kappa_score,
    )

    print("Loading consumer-grade EEG dataset...")
    X, y = load_consumer_eeg_data(max_subjects=max_subjects)
    print(f"  Loaded {len(y)} samples across {len(np.unique(y))} classes")

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for model_info in MODELS_TO_BENCHMARK:
        name = model_info["name"]
        if model_filter and name != model_filter:
            continue

        model_path = MODEL_DIR / model_info["file"]
        if not model_path.exists():
            print(f"\nSkipping {name} — model not found at {model_path}")
            continue

        print(f"\nBenchmarking {name} ({model_info['description']})...")

        try:
            if model_info["type"] == "sklearn":
                import joblib
                data = joblib.load(model_path)
                model = data["model"]

                # Align feature count
                n_model_features = (
                    model.n_features_in_
                    if hasattr(model, "n_features_in_")
                    else X.shape[1]
                )
                X_aligned = X
                if X.shape[1] != n_model_features:
                    print(
                        f"  Feature mismatch: dataset={X.shape[1]}, "
                        f"model={n_model_features}. Padding/truncating."
                    )
                    if X.shape[1] < n_model_features:
                        pad = np.zeros((X.shape[0], n_model_features - X.shape[1]))
                        X_aligned = np.hstack([X, pad])
                    else:
                        X_aligned = X[:, :n_model_features]

                start = time.time()
                y_pred = model.predict(X_aligned)
                elapsed = time.time() - start

            elif model_info["type"] == "pytorch":
                import torch
                model_data = torch.load(model_path, map_location="cpu", weights_only=False)
                print(f"  PyTorch model loaded. Skipping — requires raw EEG input, not features.")
                continue
            else:
                print(f"  Unknown model type: {model_info['type']}")
                continue

            inference_ms = (elapsed / max(len(X_aligned), 1)) * 1000
            accuracy = float(accuracy_score(y, y_pred))
            f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
            kappa = float(cohen_kappa_score(y, y_pred))

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 macro: {f1:.4f}")
            print(f"  Cohen's kappa: {kappa:.4f}")
            print(f"  Inference: {inference_ms:.2f} ms/sample")

            results[name] = {
                "model_name": name,
                "description": model_info["description"],
                "dataset": "consumer_eeg_2026",
                "dataset_doi": DATASET_DOI,
                "accuracy": accuracy,
                "f1_macro": f1,
                "cohen_kappa": kappa,
                "inference_time_ms": round(inference_ms, 3),
                "n_test_samples": int(len(y)),
                "n_classes": int(len(np.unique(y))),
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"model_name": name, "error": str(e)}

    # Save results
    if results:
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {RESULTS_FILE}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark AntarAI models against consumer-grade EEG dataset"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if the dataset is downloaded and accessible",
    )
    parser.add_argument(
        "--download-instructions",
        action="store_true",
        help="Print download instructions for the dataset",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the benchmark (requires downloaded data)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Benchmark only this model (e.g., emotion_mega_lgbm)",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Limit number of subjects to load (for quick testing)",
    )

    args = parser.parse_args()

    if not any([args.check, args.download_instructions, args.run]):
        parser.print_help()
        sys.exit(0)

    if args.check:
        found = check_dataset()
        sys.exit(0 if found else 1)

    if args.download_instructions:
        print_download_instructions()
        sys.exit(0)

    if args.run:
        if not check_dataset():
            print("\nDataset not found. Run with --download-instructions for setup.")
            sys.exit(1)
        run_benchmark(model_filter=args.model, max_subjects=args.max_subjects)


if __name__ == "__main__":
    main()
