"""Benchmark: YASA vs existing SleepStagingModel on synthetic sleep data.

Generates synthetic sleep data with known stages, runs both the existing
custom model and YASA on the same data, and compares accuracy,
per-stage sensitivity/specificity, and inference time.

GitHub issue: #527

Usage:
    cd ml && python3 -m training.benchmark_sleep_yasa
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic sleep data generator
# ---------------------------------------------------------------------------

STAGE_MAP = {
    "W": 0,
    "WAKE": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4,
    "REM": 4,
}

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def generate_synthetic_epoch(
    stage: str,
    fs: int = 256,
    duration_sec: float = 30.0,
) -> np.ndarray:
    """Generate a single 30-second EEG epoch with characteristics of the given stage.

    Uses frequency band dominance to simulate each stage:
    - Wake: alpha (8-12 Hz) dominant, low delta
    - N1: theta (4-8 Hz) increasing, alpha decreasing
    - N2: theta + spindle bursts (12-15 Hz) + K-complexes
    - N3: delta (0.5-4 Hz) dominant (>75% power)
    - REM: mixed theta, low alpha, occasional rapid eye movements
    """
    rng = np.random.RandomState()
    n = int(fs * duration_sec)
    t = np.arange(n) / fs
    noise = 3.0 * rng.randn(n)

    if stage in ("W", "Wake"):
        # Alpha dominant (10 Hz) + some beta
        signal = (
            25.0 * np.sin(2 * np.pi * 10.0 * t)
            + 8.0 * np.sin(2 * np.pi * 20.0 * t)
            + 3.0 * np.sin(2 * np.pi * 2.0 * t)
            + noise
        )
    elif stage == "N1":
        # Theta increasing, alpha decreasing
        signal = (
            15.0 * np.sin(2 * np.pi * 6.0 * t)
            + 10.0 * np.sin(2 * np.pi * 10.0 * t)
            + 5.0 * np.sin(2 * np.pi * 2.0 * t)
            + noise
        )
    elif stage == "N2":
        # Theta + spindle bursts + K-complexes
        theta = 12.0 * np.sin(2 * np.pi * 5.5 * t)
        delta = 15.0 * np.sin(2 * np.pi * 2.0 * t)
        # Add spindle bursts at 13 Hz
        spindle_env = np.zeros(n)
        for center in np.arange(3, duration_sec - 3, 5):
            sigma = 0.25
            spindle_env += np.exp(-0.5 * ((t - center) / sigma) ** 2)
        spindles = 20.0 * spindle_env * np.sin(2 * np.pi * 13.0 * t)
        signal = theta + delta + spindles + noise
    elif stage == "N3":
        # Delta dominant
        signal = (
            50.0 * np.sin(2 * np.pi * 1.5 * t)
            + 20.0 * np.sin(2 * np.pi * 0.8 * t)
            + 5.0 * np.sin(2 * np.pi * 5.0 * t)
            + noise
        )
    elif stage in ("R", "REM"):
        # Mixed theta, low alpha, some beta
        signal = (
            18.0 * np.sin(2 * np.pi * 5.0 * t)
            + 6.0 * np.sin(2 * np.pi * 10.0 * t)
            + 8.0 * np.sin(2 * np.pi * 18.0 * t)
            + 3.0 * np.sin(2 * np.pi * 1.0 * t)
            + noise
        )
    else:
        signal = noise

    return signal


def generate_synthetic_night(
    fs: int = 256,
    duration_min: float = 30.0,
) -> Tuple[np.ndarray, List[str]]:
    """Generate a synthetic night of sleep with known stage sequence.

    Creates a plausible sleep architecture:
    Wake -> N1 -> N2 -> N3 -> N2 -> REM -> N2 -> N3 -> ... -> Wake

    Returns:
        (eeg_data, true_stages) where eeg_data is a 1D array and
        true_stages is a list of stage labels per 30s epoch.
    """
    n_epochs = int(duration_min * 60 / 30)

    # Create a realistic hypnogram pattern
    # Typical NREM-REM cycles are ~90 min; for 30 min, one partial cycle
    stage_sequence = []
    if n_epochs >= 20:
        # First 2 epochs: Wake/N1 (sleep onset)
        stage_sequence.extend(["W"] * 2)
        stage_sequence.extend(["N1"] * 2)
        # Descent into deep sleep
        stage_sequence.extend(["N2"] * 3)
        stage_sequence.extend(["N3"] * 4)
        # Back up to lighter sleep
        stage_sequence.extend(["N2"] * 3)
        # REM period
        stage_sequence.extend(["REM"] * 3)
        # Fill remaining with N2
        remaining = n_epochs - len(stage_sequence)
        if remaining > 0:
            stage_sequence.extend(["N2"] * remaining)
    else:
        # Short recording: simple pattern
        stages_cycle = ["W", "N1", "N2", "N3", "N2", "REM"]
        for i in range(n_epochs):
            stage_sequence.append(stages_cycle[i % len(stages_cycle)])

    stage_sequence = stage_sequence[:n_epochs]

    # Generate continuous EEG
    eeg_chunks = []
    for stage in stage_sequence:
        epoch = generate_synthetic_epoch(stage, fs=fs)
        eeg_chunks.append(epoch)

    eeg_data = np.concatenate(eeg_chunks)
    return eeg_data, stage_sequence


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------

def _normalize_stage(stage: str) -> int:
    """Convert stage string to integer for comparison."""
    stage_upper = stage.upper().strip()
    return STAGE_MAP.get(stage_upper, -1)


def compute_metrics(
    true_stages: List[str],
    pred_stages: List[str],
) -> Dict:
    """Compute accuracy and per-stage metrics."""
    n = min(len(true_stages), len(pred_stages))
    if n == 0:
        return {"accuracy": 0.0, "n_epochs": 0}

    true_int = [_normalize_stage(s) for s in true_stages[:n]]
    pred_int = [_normalize_stage(s) for s in pred_stages[:n]]

    correct = sum(1 for t, p in zip(true_int, pred_int) if t == p)
    accuracy = correct / n

    # Per-stage metrics
    per_stage = {}
    for idx, name in enumerate(STAGE_NAMES):
        tp = sum(1 for t, p in zip(true_int, pred_int) if t == idx and p == idx)
        fp = sum(1 for t, p in zip(true_int, pred_int) if t != idx and p == idx)
        fn = sum(1 for t, p in zip(true_int, pred_int) if t == idx and p != idx)
        tn = sum(1 for t, p in zip(true_int, pred_int) if t != idx and p != idx)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        per_stage[name] = {
            "sensitivity": round(sensitivity, 3),
            "specificity": round(specificity, 3),
            "f1": round(f1, 3),
            "support": sum(1 for t in true_int if t == idx),
        }

    return {
        "accuracy": round(accuracy, 4),
        "n_epochs": n,
        "per_stage": per_stage,
    }


def benchmark_yasa(eeg_data: np.ndarray, true_stages: List[str], fs: int = 256) -> Dict:
    """Run YASA staging and compute metrics."""
    from models.yasa_sleep import stage_with_yasa

    t0 = time.time()
    result = stage_with_yasa(eeg_data, fs=fs, channel_name="EEG")
    elapsed = time.time() - t0

    if "error" in result:
        return {"error": result["error"], "time_sec": elapsed}

    metrics = compute_metrics(true_stages, result["stages"])
    metrics["time_sec"] = round(elapsed, 3)
    metrics["model"] = "yasa"
    return metrics


def benchmark_existing(eeg_data: np.ndarray, true_stages: List[str], fs: int = 256) -> Dict:
    """Run existing SleepStagingModel and compute metrics."""
    from models.sleep_staging import SleepStagingModel

    model = SleepStagingModel()
    epoch_size = int(30 * fs)  # 30 seconds per epoch
    n_epochs = len(eeg_data) // epoch_size

    t0 = time.time()
    pred_stages = []
    for i in range(n_epochs):
        start = i * epoch_size
        end = start + epoch_size
        epoch = eeg_data[start:end]
        result = model.predict(epoch, fs=float(fs))
        pred_stages.append(result.get("stage", "Wake"))
    elapsed = time.time() - t0

    metrics = compute_metrics(true_stages, pred_stages)
    metrics["time_sec"] = round(elapsed, 3)
    metrics["model"] = "existing_sleep_staging"
    return metrics


def run_benchmark():
    """Run full benchmark comparison."""
    log.info("=" * 60)
    log.info("YASA vs Existing Sleep Staging Model — Benchmark")
    log.info("=" * 60)

    # Generate synthetic data
    log.info("\nGenerating 30 minutes of synthetic sleep data...")
    eeg_data, true_stages = generate_synthetic_night(fs=256, duration_min=30.0)
    log.info(f"  Generated {len(true_stages)} epochs ({len(eeg_data)} samples)")
    log.info(f"  True stage distribution:")
    for stage in STAGE_NAMES:
        count = sum(1 for s in true_stages if _normalize_stage(s) == STAGE_MAP.get(stage[0] if stage != "Wake" else "W", -1))
        log.info(f"    {stage}: {count} epochs")

    # Run YASA
    log.info("\n--- YASA Sleep Staging ---")
    yasa_metrics = benchmark_yasa(eeg_data, true_stages, fs=256)
    if "error" in yasa_metrics:
        log.info(f"  ERROR: {yasa_metrics['error']}")
    else:
        log.info(f"  Accuracy: {yasa_metrics['accuracy']:.1%}")
        log.info(f"  Inference time: {yasa_metrics['time_sec']:.3f}s")
        for name, stats in yasa_metrics.get("per_stage", {}).items():
            log.info(f"  {name:5s}: Sens={stats['sensitivity']:.3f}  "
                     f"Spec={stats['specificity']:.3f}  "
                     f"F1={stats['f1']:.3f}  "
                     f"(n={stats['support']})")

    # Run existing model
    log.info("\n--- Existing SleepStagingModel ---")
    existing_metrics = benchmark_existing(eeg_data, true_stages, fs=256)
    if "error" in existing_metrics:
        log.info(f"  ERROR: {existing_metrics['error']}")
    else:
        log.info(f"  Accuracy: {existing_metrics['accuracy']:.1%}")
        log.info(f"  Inference time: {existing_metrics['time_sec']:.3f}s")
        for name, stats in existing_metrics.get("per_stage", {}).items():
            log.info(f"  {name:5s}: Sens={stats['sensitivity']:.3f}  "
                     f"Spec={stats['specificity']:.3f}  "
                     f"F1={stats['f1']:.3f}  "
                     f"(n={stats['support']})")

    # Summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    if "error" not in yasa_metrics and "error" not in existing_metrics:
        diff = yasa_metrics["accuracy"] - existing_metrics["accuracy"]
        better = "YASA" if diff > 0 else "Existing"
        log.info(f"  YASA accuracy:     {yasa_metrics['accuracy']:.1%}")
        log.info(f"  Existing accuracy: {existing_metrics['accuracy']:.1%}")
        log.info(f"  Difference:        {abs(diff):.1%} in favor of {better}")
        log.info(f"  YASA time:         {yasa_metrics['time_sec']:.3f}s")
        log.info(f"  Existing time:     {existing_metrics['time_sec']:.3f}s")
    log.info("")
    log.info("NOTE: These results are on synthetic data only.")
    log.info("Real Muse 2 data will show different characteristics.")
    log.info("YASA was validated on Muse-S with Kappa=0.76, accuracy 88-96%.")


if __name__ == "__main__":
    run_benchmark()
