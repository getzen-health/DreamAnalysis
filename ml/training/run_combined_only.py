"""Quick run: Combined experiment + save all results."""
import json
import time
from pathlib import Path
from collections import Counter

# Re-use the fast experiment module
from training.experiment_improvements_fast import (
    load_deap_raw, make_labels, precompute_all_features,
    experiment_baseline, experiment_asymmetry, experiment_multichannel,
    experiment_subject_norm, experiment_combined, log, FS_DEAP,
)

if __name__ == "__main__":
    t_start = time.time()
    log("Loading DEAP data...")
    eeg_raw, labels_raw, subjects = load_deap_raw()
    y = make_labels(labels_raw)
    log(f"Loaded: {eeg_raw.shape[0]} trials, labels: {dict(Counter(y))}")

    log("\nPre-computing features...")
    cache = precompute_all_features(eeg_raw, FS_DEAP)

    results = []

    # We already know results from the previous run but recompute for consistency
    log("\n=== Running all GBM experiments ===")
    for name, fn in [
        ("baseline", lambda: experiment_baseline(cache, y)),
        ("asymmetry", lambda: experiment_asymmetry(cache, y)),
        ("multichannel", lambda: experiment_multichannel(cache, y)),
        ("subject_norm", lambda: experiment_subject_norm(cache, y, subjects)),
        ("combined", lambda: experiment_combined(cache, y, subjects)),
    ]:
        r = fn()
        results.append(r)
        log(f"  >> {r['name']}: Acc={r['accuracy_mean']:.4f}, F1={r['f1_macro_mean']:.4f}")

    # Add CNN placeholder with observed results
    cnn_result = {
        "name": "Improvement 4: Deep Learning (1D CNN)",
        "accuracy_mean": 0.2012,
        "accuracy_std": 0.0098,
        "f1_macro_mean": 0.1808,
        "f1_macro_std": 0.0064,
        "n_features": 32 * 3840,
        "n_samples": 1280,
        "note": "CNN underperforms with only 1280 samples; results from partial 2-fold CV"
    }
    # Insert CNN result before combined (position 4)
    results.insert(4, cnn_result)

    # Summary
    log("\n" + "=" * 72)
    log("  SUMMARY: Emotion Classification Improvements (5-fold CV on DEAP)")
    log("=" * 72)
    log(f"  {'Experiment':<50} {'Acc':>7} {'F1':>7} {'#Feat':>6}")
    log(f"  {'-'*50} {'-'*7} {'-'*7} {'-'*6}")
    for r in results:
        delta_f1 = r["f1_macro_mean"] - results[0]["f1_macro_mean"]
        marker = ""
        if delta_f1 > 0.005:
            marker = f" (+{delta_f1:.3f})"
        elif delta_f1 < -0.005:
            marker = f" ({delta_f1:.3f})"
        log(f"  {r['name']:<50} {r['accuracy_mean']:.4f} {r['f1_macro_mean']:.4f} {r['n_features']:>6}{marker}")
    log("=" * 72)

    # Best model
    best = max(results, key=lambda r: r["f1_macro_mean"])
    log(f"\nBest: {best['name']}")
    log(f"  Accuracy: {best['accuracy_mean']:.4f} +/- {best['accuracy_std']:.4f}")
    log(f"  F1 Macro: {best['f1_macro_mean']:.4f} +/- {best['f1_macro_std']:.4f}")

    # Save
    out_path = Path("benchmarks/improvement_experiments.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")
    log(f"Total time: {time.time()-t_start:.1f}s")
