"""Optimized experiment: Compare emotion classifier improvements.

Key optimizations vs original:
  - Pre-compute all features ONCE and reuse across experiments
  - Vectorized preprocessing (batch filtering)
  - Reduced CNN epochs (20 vs 40)
  - Flushed output for real-time monitoring

Usage: python -u -m training.experiment_improvements_fast
"""

import sys
import numpy as np
import pickle
import json
import time
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler

from processing.eeg_processor import (
    bandpass_filter, notch_filter, extract_band_powers,
    compute_hjorth_parameters, spectral_entropy, extract_features,
    preprocess, BANDS,
)
from training.data_loaders import _circumplex_to_emotion

EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]
RANDOM_STATE = 42
FS_DEAP = 128.0

# DEAP channel indices
FRONTAL_CH = [0, 1, 2, 3, 4, 5, 16, 17, 18, 19]
FRONTAL_PAIRS = [(0, 16), (1, 17), (2, 18), (3, 19)]
LEFT_RIGHT_PAIRS = [
    (0, 16), (1, 17), (2, 18), (3, 19), (4, 20), (5, 21), (6, 22),
    (7, 23), (8, 24), (9, 25), (10, 26), (11, 27), (12, 28), (13, 29),
]
REGION_MAP = {
    "frontal": [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 30],
    "central": [6, 7, 22, 23, 31],
    "parietal": [8, 9, 10, 11, 24, 25, 26, 27, 15],
    "occipital": [12, 13, 14, 28, 29],
}


def log(msg):
    print(msg, flush=True)


def load_deap_raw():
    """Load raw DEAP data."""
    deap_path = Path("data/deap")
    dat_files = sorted(deap_path.glob("s*.dat"))
    all_eeg, all_labels, all_subjects = [], [], []
    for dat_file in dat_files:
        subj_id = dat_file.stem
        with open(dat_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        eeg = data["data"][:, :32, 384:]  # 32 EEG ch, skip 3s baseline
        labels = data["labels"]
        for trial_idx in range(eeg.shape[0]):
            all_eeg.append(eeg[trial_idx])
            all_labels.append(labels[trial_idx])
            all_subjects.append(subj_id)
    return np.array(all_eeg), np.array(all_labels), all_subjects


def make_labels(labels_raw):
    """Convert valence/arousal to emotion classes."""
    y = np.array([
        _circumplex_to_emotion(labels_raw[i, 0], labels_raw[i, 1])
        for i in range(len(labels_raw))
    ])
    return y


def preprocess_batch(signals, fs=128.0):
    """Preprocess multiple signals efficiently using axis parameter."""
    # Apply filters to the whole batch at once via axis=-1
    filtered = bandpass_filter(signals, 1.0, 50.0, fs)
    filtered = notch_filter(filtered, 50.0, fs)
    filtered = notch_filter(filtered, 60.0, fs)
    return filtered


def extract_features_fast(eeg_signal, fs=128.0):
    """Extract 17 features from a single preprocessed signal."""
    feats = extract_features(eeg_signal, fs)
    return list(feats.values())


def evaluate_model(X, y, name, n_splits=5):
    """5-fold CV with GBM, return results dict."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    accs, f1s = [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        weights = compute_sample_weight("balanced", y_tr)
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=RANDOM_STATE,
        )
        model.fit(X_tr, y_tr, sample_weight=weights)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average="macro")
        accs.append(acc)
        f1s.append(f1)
        log(f"    Fold {fold+1}: acc={acc:.4f}, f1={f1:.4f}")

    return {
        "name": name,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
    }


# ====================================================================
# PRE-COMPUTE ALL FEATURES (run once, reuse everywhere)
# ====================================================================
def precompute_all_features(eeg_raw, fs=128.0):
    """Pre-compute all feature sets for all trials. Returns dict of arrays."""
    n_trials = len(eeg_raw)
    cache = {}

    # 1) Preprocess frontal average for each trial
    log("  [1/5] Preprocessing frontal averages...")
    t0 = time.time()
    frontal_processed = []
    for i in range(n_trials):
        frontal_avg = eeg_raw[i][FRONTAL_CH].mean(axis=0)
        frontal_processed.append(preprocess(frontal_avg, fs))
        if (i + 1) % 200 == 0:
            log(f"       {i+1}/{n_trials} trials")
    log(f"    Done in {time.time()-t0:.1f}s")

    # 2) Extract baseline features (17 features per trial)
    log("  [2/5] Extracting baseline features...")
    t0 = time.time()
    baseline_feats = []
    for i in range(n_trials):
        feats = extract_features(frontal_processed[i], fs)
        baseline_feats.append(list(feats.values()))
        if (i + 1) % 200 == 0:
            log(f"       {i+1}/{n_trials} trials")
    cache["baseline"] = np.array(baseline_feats)
    cache["baseline_keys"] = list(feats.keys())
    log(f"    Done: shape={cache['baseline'].shape} in {time.time()-t0:.1f}s")

    # 3) Asymmetry features (FAA + beta asymmetry)
    log("  [3/5] Extracting asymmetry features...")
    t0 = time.time()
    asym_feats = []
    for i in range(n_trials):
        trial = eeg_raw[i]
        row = []
        for l_idx, r_idx in FRONTAL_PAIRS:
            l_proc = preprocess(trial[l_idx], fs)
            r_proc = preprocess(trial[r_idx], fs)
            l_bands = extract_band_powers(l_proc, fs)
            r_bands = extract_band_powers(r_proc, fs)
            l_alpha = max(l_bands["alpha"], 1e-10)
            r_alpha = max(r_bands["alpha"], 1e-10)
            row.append(np.log(r_alpha) - np.log(l_alpha))  # FAA
            l_beta = max(l_bands["beta"], 1e-10)
            r_beta = max(r_bands["beta"], 1e-10)
            row.append(np.log(r_beta) - np.log(l_beta))  # beta asym
        # Global power asymmetry
        left_power = np.mean([np.var(trial[p[0]]) for p in LEFT_RIGHT_PAIRS])
        right_power = np.mean([np.var(trial[p[1]]) for p in LEFT_RIGHT_PAIRS])
        row.append(np.log(max(right_power, 1e-10)) - np.log(max(left_power, 1e-10)))
        asym_feats.append(row)
        if (i + 1) % 200 == 0:
            log(f"       {i+1}/{n_trials} trials")
    cache["asymmetry"] = np.array(asym_feats)
    log(f"    Done: shape={cache['asymmetry'].shape} in {time.time()-t0:.1f}s")

    # 4) Multi-channel region features
    log("  [4/5] Extracting multi-channel region features...")
    t0 = time.time()
    region_feats = []
    for i in range(n_trials):
        trial = eeg_raw[i]
        row = []
        for region_name, ch_indices in REGION_MAP.items():
            region_avg = trial[ch_indices].mean(axis=0)
            region_proc = preprocess(region_avg, fs)
            bands = extract_band_powers(region_proc, fs)
            row.extend(bands.values())
        region_feats.append(row)
        if (i + 1) % 200 == 0:
            log(f"       {i+1}/{n_trials} trials")
    cache["regions"] = np.array(region_feats)
    log(f"    Done: shape={cache['regions'].shape} in {time.time()-t0:.1f}s")

    # 5) Preprocessed full 32-channel data for CNN
    log("  [5/5] Batch-preprocessing all 32 channels for CNN...")
    t0 = time.time()
    # Vectorized: apply filter to entire (n_trials, 32, samples) at once
    cache["eeg_preprocessed"] = preprocess_batch(eeg_raw, fs)
    # Downsample by 2x
    cache["eeg_downsampled"] = cache["eeg_preprocessed"][:, :, ::2]
    log(f"    Done: shape={cache['eeg_downsampled'].shape} in {time.time()-t0:.1f}s")

    return cache


# ====================================================================
# EXPERIMENTS (all use pre-computed features)
# ====================================================================

def experiment_baseline(cache, y):
    log("\n--- Experiment 1: BASELINE (frontal avg, 17 features) ---")
    X = cache["baseline"]
    log(f"  Shape: {X.shape}, Classes: {dict(Counter(y))}")
    return evaluate_model(X, y, "Baseline (frontal avg, 17 features)")


def experiment_asymmetry(cache, y):
    log("\n--- Experiment 2: + ASYMMETRY features ---")
    X = np.hstack([cache["baseline"], cache["asymmetry"]])
    log(f"  Shape: {X.shape} (17 base + {cache['asymmetry'].shape[1]} asymmetry)")
    return evaluate_model(X, y, "Improvement 1: + Asymmetry features")


def experiment_multichannel(cache, y):
    log("\n--- Experiment 3: + MULTI-CHANNEL region features ---")
    X = np.hstack([cache["baseline"], cache["regions"]])
    log(f"  Shape: {X.shape} (17 base + {cache['regions'].shape[1]} region)")
    return evaluate_model(X, y, "Improvement 2: + Multi-channel regions")


def experiment_subject_norm(cache, y, subjects):
    log("\n--- Experiment 4: + SUBJECT NORMALIZATION ---")
    X = cache["baseline"].copy()
    subj_arr = np.array(subjects)
    for subj_id in np.unique(subj_arr):
        mask = subj_arr == subj_id
        subj_data = X[mask]
        mean = subj_data.mean(axis=0)
        std = subj_data.std(axis=0) + 1e-10
        X[mask] = (subj_data - mean) / std
    log(f"  Shape: {X.shape}, Subjects: {len(np.unique(subj_arr))}")
    return evaluate_model(X, y, "Improvement 3: + Subject normalization")


def experiment_deep_learning(cache, y):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    log("\n--- Experiment 5: DEEP LEARNING (1D CNN) ---")
    X_ds = cache["eeg_downsampled"]
    log(f"  Input: {X_ds.shape} (trials, channels, samples)")

    class EEGNet1D(nn.Module):
        def __init__(self, n_channels=32, n_samples=3840, n_classes=6):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(n_channels, 64, kernel_size=25, padding=12),
                nn.BatchNorm1d(64), nn.ReLU(), nn.AvgPool1d(4), nn.Dropout(0.3),
                nn.Conv1d(64, 128, kernel_size=15, padding=7),
                nn.BatchNorm1d(128), nn.ReLU(), nn.AvgPool1d(4), nn.Dropout(0.3),
                nn.Conv1d(128, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(8), nn.Dropout(0.3),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(64 * 8, 64), nn.ReLU(),
                nn.Dropout(0.5), nn.Linear(64, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_ds, y)):
        X_tr = torch.FloatTensor(X_ds[train_idx])
        y_tr = torch.LongTensor(y[train_idx])
        X_te = torch.FloatTensor(X_ds[test_idx])
        y_te = y[test_idx]

        counts = np.bincount(y[train_idx], minlength=6).astype(float)
        weights = 1.0 / (counts + 1e-5)
        weights = weights / weights.sum() * len(counts)
        class_weights = torch.FloatTensor(weights)

        model = EEGNet1D(n_channels=32, n_samples=X_ds.shape[2], n_classes=6)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        dataset = TensorDataset(X_tr, y_tr)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        model.train()
        for epoch in range(20):  # Reduced from 40
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                out = model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_te)
            y_pred = logits.argmax(dim=1).numpy()

        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average="macro")
        accs.append(acc)
        f1s.append(f1)
        log(f"    Fold {fold+1}: acc={acc:.4f}, f1={f1:.4f}")

    return {
        "name": "Improvement 4: Deep Learning (1D CNN)",
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
        "n_features": int(X_ds.shape[1] * X_ds.shape[2]),
        "n_samples": int(X_ds.shape[0]),
    }


def experiment_combined(cache, y, subjects):
    log("\n--- Experiment 6: COMBINED (all improvements) ---")
    X = np.hstack([cache["baseline"], cache["asymmetry"], cache["regions"]])
    subj_arr = np.array(subjects)
    for subj_id in np.unique(subj_arr):
        mask = subj_arr == subj_id
        subj_data = X[mask]
        mean = subj_data.mean(axis=0)
        std = subj_data.std(axis=0) + 1e-10
        X[mask] = (subj_data - mean) / std
    log(f"  Shape: {X.shape} (17 base + {cache['asymmetry'].shape[1]} asym + {cache['regions'].shape[1]} region, subject-normed)")
    return evaluate_model(X, y, "Combined: All improvements (asym + regions + subj-norm)")


# ====================================================================
# MAIN
# ====================================================================
if __name__ == "__main__":
    t_start = time.time()
    log("=" * 60)
    log("  EEG Emotion Classification: Improvement Experiments")
    log("=" * 60)

    log("\nLoading DEAP raw data (32 subjects, 40 trials each)...")
    eeg_raw, labels_raw, subjects = load_deap_raw()
    log(f"Loaded: {eeg_raw.shape[0]} trials, {eeg_raw.shape[1]} channels, {eeg_raw.shape[2]} samples")

    y = make_labels(labels_raw)
    log(f"Label distribution: {dict(Counter(y))}")
    log(f"Emotion map: {dict(enumerate(EMOTIONS))}")

    log("\n--- PRE-COMPUTING ALL FEATURES ---")
    cache = precompute_all_features(eeg_raw, FS_DEAP)

    results = []

    r = experiment_baseline(cache, y)
    results.append(r)
    log(f"  >> Accuracy: {r['accuracy_mean']:.4f} +/- {r['accuracy_std']:.4f}")
    log(f"  >> F1 Macro: {r['f1_macro_mean']:.4f} +/- {r['f1_macro_std']:.4f}")

    r = experiment_asymmetry(cache, y)
    results.append(r)
    log(f"  >> Accuracy: {r['accuracy_mean']:.4f} +/- {r['accuracy_std']:.4f}")
    log(f"  >> F1 Macro: {r['f1_macro_mean']:.4f} +/- {r['f1_macro_std']:.4f}")

    r = experiment_multichannel(cache, y)
    results.append(r)
    log(f"  >> Accuracy: {r['accuracy_mean']:.4f} +/- {r['accuracy_std']:.4f}")
    log(f"  >> F1 Macro: {r['f1_macro_mean']:.4f} +/- {r['f1_macro_std']:.4f}")

    r = experiment_subject_norm(cache, y, subjects)
    results.append(r)
    log(f"  >> Accuracy: {r['accuracy_mean']:.4f} +/- {r['accuracy_std']:.4f}")
    log(f"  >> F1 Macro: {r['f1_macro_mean']:.4f} +/- {r['f1_macro_std']:.4f}")

    r = experiment_deep_learning(cache, y)
    results.append(r)
    log(f"  >> Accuracy: {r['accuracy_mean']:.4f} +/- {r['accuracy_std']:.4f}")
    log(f"  >> F1 Macro: {r['f1_macro_mean']:.4f} +/- {r['f1_macro_std']:.4f}")

    r = experiment_combined(cache, y, subjects)
    results.append(r)
    log(f"  >> Accuracy: {r['accuracy_mean']:.4f} +/- {r['accuracy_std']:.4f}")
    log(f"  >> F1 Macro: {r['f1_macro_mean']:.4f} +/- {r['f1_macro_std']:.4f}")

    # Summary table
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

    # Save results
    out_path = Path("benchmarks/improvement_experiments.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")
    log(f"Total time: {time.time()-t_start:.1f}s")
