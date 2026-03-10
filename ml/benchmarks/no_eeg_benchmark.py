"""No-EEG accuracy benchmark — voice + health emotion detection.

Runs modality ablation across 6 configurations and reports:
  - 6-class accuracy (happy/sad/angry/fear/neutral/calm)
  - Weighted F1
  - Valence MAE (mean absolute error on -1 to +1 scale)
  - Arousal MAE (0 to 1 scale)
  - Stress detection F1 (binary: stressed vs not-stressed)
  - Mood direction accuracy (did valence direction match?)

Usage
-----
# Evaluate existing labeled samples in benchmarks/ground_truth.jsonl:
python benchmarks/no_eeg_benchmark.py

# With custom ground-truth file:
python benchmarks/no_eeg_benchmark.py --gt-file path/to/labels.jsonl

# Print results only (no save):
python benchmarks/no_eeg_benchmark.py --no-save

Ground-truth format (one JSON object per line in ground_truth.jsonl):
{
  "timestamp": 1740000000.0,
  "user_id": "dev",
  "label_emotion": "happy",       # one of: happy/sad/angry/fear/neutral/calm
  "label_valence": 0.6,           # -1 to +1
  "label_arousal": 0.7,           # 0 to 1
  "label_stress": false,          # true = stressed
  "voice_emotion": "happy",       # predicted by voice model
  "voice_valence": 0.55,
  "voice_arousal": 0.65,
  "health_valence": 0.4,
  "health_arousal": 0.5,
  "health_stress": false,
  "combined_emotion": "happy",
  "combined_valence": 0.5,
  "combined_arousal": 0.6,
  "combined_stress": false
}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).parent
_DEFAULT_GT_FILE = _HERE / "ground_truth.jsonl"
_DEFAULT_RESULTS_FILE = _HERE / "no_eeg_benchmark_results.json"

EMOTION_CLASSES = ["happy", "sad", "angry", "fear", "neutral", "calm"]

# ---------------------------------------------------------------------------
# Ablation configurations — which fields to use for each modality config
# ---------------------------------------------------------------------------
ABLATION_CONFIGS = [
    {
        "name": "voice_only",
        "label": "Voice only (emotion2vec+)",
        "emotion_key":  "voice_emotion",
        "valence_key":  "voice_valence",
        "arousal_key":  "voice_arousal",
        "stress_key":   None,           # derived from valence+arousal
    },
    {
        "name": "health_only",
        "label": "Health only (HRV + sleep + activity)",
        "emotion_key":  None,           # health doesn't produce discrete emotion
        "valence_key":  "health_valence",
        "arousal_key":  "health_arousal",
        "stress_key":   "health_stress",
    },
    {
        "name": "voice_plus_health",
        "label": "Voice + Health",
        "emotion_key":  "combined_emotion",
        "valence_key":  "combined_valence",
        "arousal_key":  "combined_arousal",
        "stress_key":   "combined_stress",
    },
    {
        "name": "voice_plus_health_plus_supplements",
        "label": "Voice + Health + Supplements",
        "emotion_key":  "combined_supplement_emotion",
        "valence_key":  "combined_supplement_valence",
        "arousal_key":  "combined_supplement_arousal",
        "stress_key":   "combined_supplement_stress",
    },
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def weighted_f1(y_true: List[str], y_pred: List[str], classes: List[str]) -> float:
    """Macro-averaged F1 weighted by class support."""
    counts: Dict[str, int] = defaultdict(int)
    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)

    for t, p in zip(y_true, y_pred):
        counts[t] += 1
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    total = len(y_true)
    if total == 0:
        return 0.0

    weighted_sum = 0.0
    for cls in classes:
        prec = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        rec  = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = counts[cls]
        weighted_sum += f1 * support

    return weighted_sum / total


def mae(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return float("nan")
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def binary_f1(y_true: List[bool], y_pred: List[bool]) -> float:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def mood_direction_accuracy(
    y_true_valence: List[float],
    y_pred_valence: List[float],
) -> float:
    """Fraction of consecutive pairs where predicted direction matches true direction."""
    if len(y_true_valence) < 2:
        return float("nan")
    n, correct = 0, 0
    for i in range(1, len(y_true_valence)):
        true_dir = y_true_valence[i] - y_true_valence[i - 1]
        pred_dir = y_pred_valence[i] - y_pred_valence[i - 1]
        if true_dir == 0.0:
            continue
        n += 1
        if (true_dir > 0 and pred_dir > 0) or (true_dir < 0 and pred_dir < 0):
            correct += 1
    return correct / n if n > 0 else float("nan")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _load_ground_truth(gt_file: Path) -> List[Dict[str, Any]]:
    if not gt_file.exists():
        return []
    samples = []
    with open(gt_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return samples


def _stress_from_arousal_valence(valence: Optional[float], arousal: Optional[float]) -> bool:
    """Heuristic: stressed = high arousal + negative valence."""
    if valence is None or arousal is None:
        return False
    return arousal > 0.6 and valence < -0.1


def _evaluate_config(
    samples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate one ablation configuration against ground truth labels."""
    y_true_emotion:  List[str]   = []
    y_pred_emotion:  List[str]   = []
    y_true_valence:  List[float] = []
    y_pred_valence:  List[float] = []
    y_true_arousal:  List[float] = []
    y_pred_arousal:  List[float] = []
    y_true_stress:   List[bool]  = []
    y_pred_stress:   List[bool]  = []

    skipped = 0
    for s in samples:
        # Valence
        tv = s.get("label_valence")
        pv = s.get(cfg["valence_key"]) if cfg["valence_key"] else None
        if tv is not None and pv is not None:
            y_true_valence.append(float(tv))
            y_pred_valence.append(float(pv))

        # Arousal
        ta = s.get("label_arousal")
        pa = s.get(cfg["arousal_key"]) if cfg["arousal_key"] else None
        if ta is not None and pa is not None:
            y_true_arousal.append(float(ta))
            y_pred_arousal.append(float(pa))

        # Emotion (discrete)
        te = s.get("label_emotion")
        pe = s.get(cfg["emotion_key"]) if cfg["emotion_key"] else None
        if te is not None and pe is not None:
            y_true_emotion.append(te)
            y_pred_emotion.append(pe)
        elif cfg["emotion_key"] is None:
            pass  # health-only doesn't predict discrete emotion
        else:
            skipped += 1

        # Stress
        ts = s.get("label_stress")
        if cfg["stress_key"]:
            ps = s.get(cfg["stress_key"])
        else:
            # Derive from predicted valence+arousal
            ps = _stress_from_arousal_valence(pv, pa)

        if ts is not None and ps is not None:
            y_true_stress.append(bool(ts))
            y_pred_stress.append(bool(ps))

    result: Dict[str, Any] = {
        "n_samples": len(samples),
        "n_skipped": skipped,
        "n_evaluated_emotion": len(y_true_emotion),
        "n_evaluated_valence": len(y_true_valence),
        "n_evaluated_arousal": len(y_true_arousal),
        "n_evaluated_stress": len(y_true_stress),
    }

    if y_true_emotion:
        result["accuracy_6class"] = round(accuracy(y_true_emotion, y_pred_emotion), 4)
        result["f1_weighted"] = round(weighted_f1(y_true_emotion, y_pred_emotion, EMOTION_CLASSES), 4)
    else:
        result["accuracy_6class"] = None
        result["f1_weighted"] = None

    result["valence_mae"] = round(mae(y_true_valence, y_pred_valence), 4) if y_true_valence else None
    result["arousal_mae"] = round(mae(y_true_arousal, y_pred_arousal), 4) if y_true_arousal else None
    result["stress_f1"]   = round(binary_f1(y_true_stress, y_pred_stress), 4) if y_true_stress else None
    result["mood_direction_accuracy"] = (
        round(mood_direction_accuracy(y_true_valence, y_pred_valence), 4)
        if len(y_true_valence) >= 2 else None
    )

    return result


def run_benchmark(
    gt_file: Path = _DEFAULT_GT_FILE,
    save_results: bool = True,
    results_file: Path = _DEFAULT_RESULTS_FILE,
) -> Dict[str, Any]:
    samples = _load_ground_truth(gt_file)
    if not samples:
        print(f"[no-eeg-benchmark] No ground truth found at {gt_file}")
        print("  → Run collect_ground_truth.py to log labeled check-ins.")
        print("  → Or add samples manually to benchmarks/ground_truth.jsonl")
        return {"error": "no_data", "gt_file": str(gt_file)}

    print(f"[no-eeg-benchmark] Loaded {len(samples)} samples from {gt_file}")
    results: Dict[str, Any] = {}

    for cfg in ABLATION_CONFIGS:
        metrics = _evaluate_config(samples, cfg)
        results[cfg["name"]] = {
            "label": cfg["label"],
            "metrics": metrics,
        }

    _print_table(results)

    output = {
        "n_total_samples": len(samples),
        "gt_file": str(gt_file),
        "configurations": results,
    }

    if save_results:
        with open(results_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n[no-eeg-benchmark] Results saved to {results_file}")

    return output


def _print_table(results: Dict[str, Any]) -> None:
    header = f"{'Config':<40} {'6cls Acc':>9} {'F1':>7} {'Val MAE':>8} {'Aro MAE':>8} {'Str F1':>7} {'Dir%':>6}"
    print()
    print(header)
    print("-" * len(header))
    for name, entry in results.items():
        m = entry["metrics"]
        acc = f"{m['accuracy_6class']*100:.1f}%" if m["accuracy_6class"] is not None else "  N/A  "
        f1  = f"{m['f1_weighted']*100:.1f}%" if m["f1_weighted"] is not None else " N/A "
        vma = f"{m['valence_mae']:.3f}"   if m["valence_mae"] is not None else "  N/A "
        ama = f"{m['arousal_mae']:.3f}"   if m["arousal_mae"] is not None else "  N/A "
        sf1 = f"{m['stress_f1']*100:.1f}%" if m["stress_f1"] is not None else " N/A "
        dir_= f"{m['mood_direction_accuracy']*100:.0f}%" if m["mood_direction_accuracy"] is not None else " N/A"
        print(f"{entry['label']:<40} {acc:>9} {f1:>7} {vma:>8} {ama:>8} {sf1:>7} {dir_:>6}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="No-EEG accuracy benchmark")
    parser.add_argument(
        "--gt-file",
        type=Path,
        default=_DEFAULT_GT_FILE,
        help="Path to ground_truth.jsonl (default: ml/benchmarks/ground_truth.jsonl)",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=_DEFAULT_RESULTS_FILE,
        help="Where to save results JSON",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print results without saving",
    )
    args = parser.parse_args()
    run_benchmark(
        gt_file=args.gt_file,
        save_results=not args.no_save,
        results_file=args.results_file,
    )
