"""Training script for the supervised artifact classifier.

Trains on TUH Artifact Corpus (if available) or synthetic data.

Usage:
    python -m training.train_artifact_classifier
    python -m training.train_artifact_classifier --synthetic --n-samples 10000
    python -m training.train_artifact_classifier --data-dir data/tuh_artifact
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path


def load_tuh_artifacts(
    data_dir: str = "data/tuh_artifact",
    target_fs: float = 256.0,
    epoch_sec: float = 1.0,
):
    """Load TUH Artifact Corpus (TUAR) labeled data.

    TUAR provides 310 EEG files with 5 artifact types annotated:
    EYEM (eye movement), CHEW (chewing), SHIV (shivering),
    ELPP (electrode pop), MUSC (muscle).

    Returns:
        (X, y) where X is feature vectors and y is artifact class labels.
    """
    import mne

    mne.set_log_level("WARNING")

    from models.artifact_classifier import extract_artifact_features, ARTIFACT_NAME_TO_ID

    tuh_path = Path(data_dir)
    edf_files = sorted(tuh_path.glob("**/*.edf"))

    if not edf_files:
        raise FileNotFoundError(
            f"No TUH artifact files found in {data_dir}. "
            "Register at https://isip.piconepress.com/projects/tuh_eeg/ to download."
        )

    # TUH label mapping to our classes
    tuh_to_class = {
        "bckg": 0,   # background → clean
        "eyem": 1,   # eye movement → eye_blink
        "chew": 2,   # chewing → muscle_emg
        "shiv": 2,   # shivering → muscle_emg
        "elpp": 3,   # electrode pop → electrode_pop
        "musc": 2,   # muscle → muscle_emg
        "artf": 4,   # generic artifact → motion
    }

    X_all, y_all = [], []
    n_samples_epoch = int(epoch_sec * target_fs)

    for edf_file in edf_files:
        # Look for matching annotation file
        csv_file = edf_file.with_suffix(".csv")
        tse_file = edf_file.with_suffix(".tse")
        lbl_file = edf_file.with_suffix(".lbl")

        annot_file = None
        for candidate in [csv_file, tse_file, lbl_file]:
            if candidate.exists():
                annot_file = candidate
                break

        if annot_file is None:
            continue

        try:
            raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
        except Exception:
            continue

        # Pick first EEG channel
        raw.pick(raw.ch_names[:1])
        if raw.info["sfreq"] != target_fs:
            raw.resample(target_fs)

        data = raw.get_data()[0]

        # Parse annotation file (TSE format: start_sec stop_sec label probability)
        with open(annot_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("version"):
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue

                try:
                    start_sec = float(parts[0])
                    end_sec = float(parts[1])
                    label = parts[2].lower()
                except (ValueError, IndexError):
                    continue

                if label not in tuh_to_class:
                    continue

                artifact_class = tuh_to_class[label]
                start_sample = int(start_sec * target_fs)
                end_sample = int(end_sec * target_fs)

                # Extract fixed-size windows
                for pos in range(start_sample, end_sample - n_samples_epoch, n_samples_epoch):
                    segment = data[pos:pos + n_samples_epoch]
                    if len(segment) == n_samples_epoch:
                        features = extract_artifact_features(segment, target_fs)
                        X_all.append(features)
                        y_all.append(artifact_class)

        if len(y_all) % 1000 == 0 and y_all:
            print(f"  Loaded {len(y_all)} segments...")

    if not X_all:
        raise RuntimeError("No labeled segments extracted from TUH.")

    return np.array(X_all), np.array(y_all)


def train_artifact_classifier(
    data_dir: str = "data/tuh_artifact",
    output_dir: str = "models/saved",
    n_samples: int = 6000,
    use_synthetic: bool = False,
):
    """Train the artifact classifier."""
    from models.artifact_classifier import ArtifactClassifier, ARTIFACT_CLASSES

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "artifact_classifier_model.pkl"

    classifier = ArtifactClassifier()
    start_time = time.time()

    # Try TUH data first
    if not use_synthetic:
        try:
            print("Loading TUH Artifact Corpus...")
            X, y = load_tuh_artifacts(data_dir)
            print(f"  TUH data: {len(y)} samples")
            data_source = "tuh_artifact"
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  TUH not available: {e}")
            use_synthetic = True

    if use_synthetic:
        print(f"Training on synthetic data ({n_samples} samples)...")
        results = classifier.train_on_synthetic(
            n_samples=n_samples, fs=256.0, epoch_sec=1.0
        )
        data_source = "synthetic"
    else:
        results = classifier.train(X, y)

    elapsed = time.time() - start_time

    print(f"\n{'='*50}")
    print(f"Artifact Classifier Training Complete")
    print(f"{'='*50}")
    print(f"  Model type:       {results['model_type']}")
    print(f"  Test accuracy:    {results['accuracy']:.4f}")
    print(f"  CV accuracy:      {results['cv_accuracy']:.4f} +/- {results['cv_std']:.4f}")
    print(f"  Training time:    {elapsed:.1f}s")
    print(f"  Data source:      {data_source}")
    print(f"  Train/test:       {results['n_train']}/{results['n_test']}")

    # Save
    classifier.save(str(model_path))
    print(f"  Model saved to:   {model_path}")

    # Save report
    report = {
        "model": "artifact_classifier",
        "model_type": results["model_type"],
        "accuracy": results["accuracy"],
        "cv_accuracy": results["cv_accuracy"],
        "cv_std": results["cv_std"],
        "n_train": results["n_train"],
        "n_test": results["n_test"],
        "data_source": data_source,
        "training_time_sec": elapsed,
        "classes": ARTIFACT_CLASSES,
    }

    report_path = output_path / "artifact_classifier_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train artifact classifier")
    parser.add_argument("--data-dir", default="data/tuh_artifact")
    parser.add_argument("--output-dir", default="models/saved")
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    train_artifact_classifier(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        use_synthetic=args.synthetic,
    )
