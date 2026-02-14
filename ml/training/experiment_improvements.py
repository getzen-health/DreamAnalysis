"""Experiment: Compare emotion classifier improvements one by one.

Baseline: GradientBoosting on 17 hand-crafted features from DENS+DEAP.

Improvements tested:
  1. Asymmetry features (frontal alpha asymmetry, hemispheric ratios)
  2. All 32 DEAP channels (multi-channel feature aggregation)
  3. Subject-specific normalization (z-score per subject)
  4. Deep learning CNN on raw multi-channel EEG

Usage: python -m training.experiment_improvements
"""

import numpy as np
import pickle
import json
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler

from processing.eeg_processor import extract_features, extract_band_powers, preprocess
from training.data_loaders import _circumplex_to_emotion

EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]
RANDOM_STATE = 42

# DEAP channel layout (32 EEG channels, 10-20 system)
DEAP_CH_NAMES = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "F4", "F8", "FC6", "FC2", "C4", "T8",
    "CP6", "CP2", "P4", "P8", "PO4", "O2", "Fz", "Cz",
]
# Hemispheric pairs (left_idx, right_idx)
LEFT_RIGHT_PAIRS = [
    (0, 16),   # Fp1 - Fp2
    (1, 17),   # AF3 - AF4
    (2, 18),   # F3  - F4
    (3, 19),   # F7  - F8
    (4, 20),   # FC5 - FC6
    (5, 21),   # FC1 - FC2
    (6, 22),   # C3  - C4
    (7, 23),   # T7  - T8
    (8, 24),   # CP5 - CP6
    (9, 25),   # CP1 - CP2
    (10, 26),  # P3  - P4
    (11, 27),  # P7  - P8
    (12, 28),  # PO3 - PO4
    (13, 29),  # O1  - O2
]
FRONTAL_PAIRS = [(0, 16), (1, 17), (2, 18), (3, 19)]  # Fp1/2, AF3/4, F3/4, F7/8
FRONTAL_CH = [0, 1, 2, 3, 4, 5, 16, 17, 18, 19]
FS_DEAP = 128.0  # DEAP sampling rate


def load_deap_raw():
    """Load raw DEAP data as (trials, channels, samples) with labels."""
    deap_path = Path("data/deap")
    dat_files = sorted(deap_path.glob("s*.dat"))

    all_eeg, all_labels, all_subjects = [], [], []
    for dat_file in dat_files:
        subj_id = dat_file.stem  # e.g., "s01"
        with open(dat_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        eeg = data["data"][:, :32, 384:]  # 32 EEG ch, skip 3s baseline
        labels = data["labels"]  # (40, 4): valence, arousal, dominance, liking
        for trial_idx in range(eeg.shape[0]):
            all_eeg.append(eeg[trial_idx])
            all_labels.append(labels[trial_idx])
            all_subjects.append(subj_id)

    return np.array(all_eeg), np.array(all_labels), all_subjects


def evaluate_model(X, y, name, n_splits=5):
    """Train and evaluate with cross-validation, return results dict."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    accs, f1s = [], []
    for train_idx, test_idx in skf.split(X_scaled, y):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        weights = compute_sample_weight("balanced", y_tr)
        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=RANDOM_STATE,
        )
        model.fit(X_tr, y_tr, sample_weight=weights)
        y_pred = model.predict(X_te)

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="macro"))

    result = {
        "name": name,
        "accuracy_mean": np.mean(accs),
        "accuracy_std": np.std(accs),
        "f1_macro_mean": np.mean(f1s),
        "f1_macro_std": np.std(f1s),
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
    }
    return result


def print_result(r):
    """Pretty-print one experiment result."""
    print(f"\n{'='*60}")
    print(f"  {r['name']}")
    print(f"{'='*60}")
    print(f"  Samples: {r['n_samples']}, Features: {r['n_features']}")
    print(f"  Accuracy:  {r['accuracy_mean']:.4f} +/- {r['accuracy_std']:.4f}")
    print(f"  F1 Macro:  {r['f1_macro_mean']:.4f} +/- {r['f1_macro_std']:.4f}")


# ====================================================================
# BASELINE: 17 hand-crafted features from frontal channel average
# ====================================================================
def experiment_baseline(eeg_raw, labels_raw, subjects):
    """Baseline: frontal average -> 17 features."""
    print("\n[Baseline] Extracting features from frontal channel average...")
    X_list, y_list = [], []
    for i in range(len(eeg_raw)):
        frontal_avg = eeg_raw[i][FRONTAL_CH].mean(axis=0)
        feats = extract_features(preprocess(frontal_avg, FS_DEAP), FS_DEAP)
        X_list.append(list(feats.values()))
        valence, arousal = labels_raw[i, 0], labels_raw[i, 1]
        y_list.append(_circumplex_to_emotion(valence, arousal))

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"  Shape: {X.shape}, Classes: {dict(Counter(y))}")
    return evaluate_model(X, y, "Baseline (frontal avg, 17 features)")


# ====================================================================
# IMPROVEMENT 1: Add asymmetry features
# ====================================================================
def extract_asymmetry_features(trial_eeg, fs=128.0):
    """Extract hemispheric asymmetry features from multi-channel EEG.

    Returns frontal alpha asymmetry + band asymmetry for all pairs.
    """
    asymmetry = {}

    # Frontal alpha asymmetry (FAA) — key emotion biomarker
    for i, (l_idx, r_idx) in enumerate(FRONTAL_PAIRS):
        l_bands = extract_band_powers(preprocess(trial_eeg[l_idx], fs), fs)
        r_bands = extract_band_powers(preprocess(trial_eeg[r_idx], fs), fs)

        # FAA = ln(right_alpha) - ln(left_alpha)
        # Positive FAA → approach motivation (positive valence)
        l_alpha = max(l_bands["alpha"], 1e-10)
        r_alpha = max(r_bands["alpha"], 1e-10)
        asymmetry[f"faa_{i}"] = np.log(r_alpha) - np.log(l_alpha)

        # Beta asymmetry (arousal lateralization)
        l_beta = max(l_bands["beta"], 1e-10)
        r_beta = max(r_bands["beta"], 1e-10)
        asymmetry[f"beta_asym_{i}"] = np.log(r_beta) - np.log(l_beta)

    # Global hemispheric power asymmetry
    left_chs = [p[0] for p in LEFT_RIGHT_PAIRS]
    right_chs = [p[1] for p in LEFT_RIGHT_PAIRS]
    left_power = np.mean([np.var(trial_eeg[ch]) for ch in left_chs])
    right_power = np.mean([np.var(trial_eeg[ch]) for ch in right_chs])
    asymmetry["global_power_asym"] = np.log(max(right_power, 1e-10)) - np.log(max(left_power, 1e-10))

    return asymmetry


def experiment_asymmetry(eeg_raw, labels_raw, subjects):
    """Improvement 1: Baseline + asymmetry features."""
    print("\n[Improvement 1] Adding asymmetry features...")
    X_list, y_list = [], []
    for i in range(len(eeg_raw)):
        frontal_avg = eeg_raw[i][FRONTAL_CH].mean(axis=0)
        base_feats = extract_features(preprocess(frontal_avg, FS_DEAP), FS_DEAP)
        asym_feats = extract_asymmetry_features(eeg_raw[i], FS_DEAP)

        combined = {**base_feats, **asym_feats}
        X_list.append(list(combined.values()))

        valence, arousal = labels_raw[i, 0], labels_raw[i, 1]
        y_list.append(_circumplex_to_emotion(valence, arousal))

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"  Shape: {X.shape} ({len(base_feats)} base + {len(asym_feats)} asymmetry)")
    return evaluate_model(X, y, "Improvement 1: + Asymmetry features")


# ====================================================================
# IMPROVEMENT 2: All 32 channels (region-based aggregation)
# ====================================================================
REGION_MAP = {
    "frontal": [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 30],  # Fp1..FC2 + Fz
    "central": [6, 7, 22, 23, 31],  # C3, T7, C4, T8, Cz
    "parietal": [8, 9, 10, 11, 24, 25, 26, 27, 15],  # CP5..P8 + Pz
    "occipital": [12, 13, 14, 28, 29],  # PO3, O1, Oz, PO4, O2
}


def extract_multichannel_features(trial_eeg, fs=128.0):
    """Extract features per brain region (frontal, central, parietal, occipital)."""
    region_feats = {}
    for region_name, ch_indices in REGION_MAP.items():
        region_avg = trial_eeg[ch_indices].mean(axis=0)
        bands = extract_band_powers(preprocess(region_avg, fs), fs)
        for band_name, power in bands.items():
            region_feats[f"{region_name}_{band_name}"] = power
    return region_feats


def experiment_multichannel(eeg_raw, labels_raw, subjects):
    """Improvement 2: Region-based multi-channel features."""
    print("\n[Improvement 2] Extracting region-based multi-channel features...")
    X_list, y_list = [], []
    for i in range(len(eeg_raw)):
        frontal_avg = eeg_raw[i][FRONTAL_CH].mean(axis=0)
        base_feats = extract_features(preprocess(frontal_avg, FS_DEAP), FS_DEAP)
        region_feats = extract_multichannel_features(eeg_raw[i], FS_DEAP)

        combined = {**base_feats, **region_feats}
        X_list.append(list(combined.values()))

        valence, arousal = labels_raw[i, 0], labels_raw[i, 1]
        y_list.append(_circumplex_to_emotion(valence, arousal))

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"  Shape: {X.shape} ({len(base_feats)} base + {len(region_feats)} region)")
    return evaluate_model(X, y, "Improvement 2: + Multi-channel regions")


# ====================================================================
# IMPROVEMENT 3: Subject-specific normalization
# ====================================================================
def experiment_subject_norm(eeg_raw, labels_raw, subjects):
    """Improvement 3: Z-score normalize features within each subject."""
    print("\n[Improvement 3] Subject-specific normalization...")
    X_list, y_list, subj_list = [], [], []
    for i in range(len(eeg_raw)):
        frontal_avg = eeg_raw[i][FRONTAL_CH].mean(axis=0)
        feats = extract_features(preprocess(frontal_avg, FS_DEAP), FS_DEAP)
        X_list.append(list(feats.values()))
        valence, arousal = labels_raw[i, 0], labels_raw[i, 1]
        y_list.append(_circumplex_to_emotion(valence, arousal))
        subj_list.append(subjects[i])

    X = np.array(X_list)
    y = np.array(y_list)
    subj_arr = np.array(subj_list)

    # Z-score within each subject
    X_normed = np.zeros_like(X)
    for subj_id in np.unique(subj_arr):
        mask = subj_arr == subj_id
        subj_data = X[mask]
        mean = subj_data.mean(axis=0)
        std = subj_data.std(axis=0) + 1e-10
        X_normed[mask] = (subj_data - mean) / std

    print(f"  Shape: {X_normed.shape}, Subjects: {len(np.unique(subj_arr))}")
    return evaluate_model(X_normed, y, "Improvement 3: + Subject normalization")


# ====================================================================
# IMPROVEMENT 4: Deep Learning (1D CNN on raw multi-channel EEG)
# ====================================================================
def experiment_deep_learning(eeg_raw, labels_raw, subjects):
    """Improvement 4: 1D CNN on raw multi-channel EEG."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print("\n[Improvement 4] Training 1D CNN on raw multi-channel EEG...")

    y_list = []
    for i in range(len(labels_raw)):
        valence, arousal = labels_raw[i, 0], labels_raw[i, 1]
        y_list.append(_circumplex_to_emotion(valence, arousal))
    y = np.array(y_list)

    # Preprocess each channel
    X_processed = np.zeros_like(eeg_raw)
    for i in range(len(eeg_raw)):
        for ch in range(32):
            X_processed[i, ch] = preprocess(eeg_raw[i, ch], FS_DEAP)

    # Downsample to manageable size (take every 2nd sample)
    X_ds = X_processed[:, :, ::2]
    print(f"  Input shape: {X_ds.shape} (trials, channels, samples)")

    class EEGNet1D(nn.Module):
        """Compact 1D CNN for EEG emotion classification."""
        def __init__(self, n_channels=32, n_samples=3840, n_classes=6):
            super().__init__()
            self.features = nn.Sequential(
                # Temporal convolution
                nn.Conv1d(n_channels, 64, kernel_size=25, padding=12),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AvgPool1d(4),
                nn.Dropout(0.3),
                # Deeper conv
                nn.Conv1d(64, 128, kernel_size=15, padding=7),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AvgPool1d(4),
                nn.Dropout(0.3),
                # Final conv
                nn.Conv1d(128, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
                nn.Dropout(0.3),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 8, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_ds, y)):
        X_tr = torch.FloatTensor(X_ds[train_idx])
        y_tr = torch.LongTensor(y[train_idx])
        X_te = torch.FloatTensor(X_ds[test_idx])
        y_te = y[test_idx]

        # Class weights for loss
        counts = np.bincount(y[train_idx], minlength=6).astype(float)
        weights = 1.0 / (counts + 1e-5)
        weights = weights / weights.sum() * len(counts)
        class_weights = torch.FloatTensor(weights)

        model = EEGNet1D(n_channels=32, n_samples=X_ds.shape[2], n_classes=6)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

        dataset = TensorDataset(X_tr, y_tr)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(40):
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
        print(f"  Fold {fold+1}: acc={acc:.4f}, f1={f1:.4f}")

    result = {
        "name": "Improvement 4: Deep Learning (1D CNN)",
        "accuracy_mean": np.mean(accs),
        "accuracy_std": np.std(accs),
        "f1_macro_mean": np.mean(f1s),
        "f1_macro_std": np.std(f1s),
        "n_features": X_ds.shape[1] * X_ds.shape[2],
        "n_samples": X_ds.shape[0],
    }
    return result


# ====================================================================
# COMBINED BEST: Stack all feature improvements
# ====================================================================
def experiment_combined(eeg_raw, labels_raw, subjects):
    """Combined: asymmetry + multi-channel + subject normalization."""
    print("\n[Combined] All feature improvements together...")
    X_list, y_list, subj_list = [], [], []
    for i in range(len(eeg_raw)):
        frontal_avg = eeg_raw[i][FRONTAL_CH].mean(axis=0)
        base_feats = extract_features(preprocess(frontal_avg, FS_DEAP), FS_DEAP)
        asym_feats = extract_asymmetry_features(eeg_raw[i], FS_DEAP)
        region_feats = extract_multichannel_features(eeg_raw[i], FS_DEAP)

        combined = {**base_feats, **asym_feats, **region_feats}
        X_list.append(list(combined.values()))

        valence, arousal = labels_raw[i, 0], labels_raw[i, 1]
        y_list.append(_circumplex_to_emotion(valence, arousal))
        subj_list.append(subjects[i])

    X = np.array(X_list)
    y = np.array(y_list)
    subj_arr = np.array(subj_list)

    # Subject normalization
    X_normed = np.zeros_like(X)
    for subj_id in np.unique(subj_arr):
        mask = subj_arr == subj_id
        subj_data = X[mask]
        mean = subj_data.mean(axis=0)
        std = subj_data.std(axis=0) + 1e-10
        X_normed[mask] = (subj_data - mean) / std

    n_base = len(base_feats)
    n_asym = len(asym_feats)
    n_region = len(region_feats)
    print(f"  Shape: {X_normed.shape} ({n_base} base + {n_asym} asym + {n_region} region, subject-normed)")
    return evaluate_model(X_normed, y, "Combined: All improvements (asym + regions + subj-norm)")


# ====================================================================
# MAIN
# ====================================================================
if __name__ == "__main__":
    print("Loading DEAP raw data (32 subjects, 40 trials each)...")
    eeg_raw, labels_raw, subjects = load_deap_raw()
    print(f"Loaded: {eeg_raw.shape[0]} trials, {eeg_raw.shape[1]} channels, {eeg_raw.shape[2]} samples")

    results = []

    # Baseline
    r = experiment_baseline(eeg_raw, labels_raw, subjects)
    print_result(r)
    results.append(r)

    # Improvement 1: Asymmetry
    r = experiment_asymmetry(eeg_raw, labels_raw, subjects)
    print_result(r)
    results.append(r)

    # Improvement 2: Multi-channel
    r = experiment_multichannel(eeg_raw, labels_raw, subjects)
    print_result(r)
    results.append(r)

    # Improvement 3: Subject normalization
    r = experiment_subject_norm(eeg_raw, labels_raw, subjects)
    print_result(r)
    results.append(r)

    # Improvement 4: Deep learning
    r = experiment_deep_learning(eeg_raw, labels_raw, subjects)
    print_result(r)
    results.append(r)

    # Combined best features
    r = experiment_combined(eeg_raw, labels_raw, subjects)
    print_result(r)
    results.append(r)

    # Summary table
    print("\n" + "=" * 72)
    print("  SUMMARY: Emotion Classification Improvements (5-fold CV)")
    print("=" * 72)
    print(f"  {'Experiment':<50} {'Acc':>7} {'F1':>7} {'#Feat':>6}")
    print(f"  {'-'*50} {'-'*7} {'-'*7} {'-'*6}")
    for r in results:
        delta_acc = r["accuracy_mean"] - results[0]["accuracy_mean"]
        delta_f1 = r["f1_macro_mean"] - results[0]["f1_macro_mean"]
        marker = ""
        if delta_f1 > 0.01:
            marker = f" (+{delta_f1:.3f})"
        elif delta_f1 < -0.01:
            marker = f" ({delta_f1:.3f})"
        print(f"  {r['name']:<50} {r['accuracy_mean']:.4f} {r['f1_macro_mean']:.4f} {r['n_features']:>6}{marker}")
    print("=" * 72)

    # Save results
    out_path = Path("benchmarks/improvement_experiments.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
