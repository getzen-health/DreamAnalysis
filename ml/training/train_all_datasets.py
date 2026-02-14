"""Train emotion classifier on ALL available datasets.

Unified 3-class model: POSITIVE / NEUTRAL / NEGATIVE

Data sources:
  1. DEAP (32 subjects, 1280 trials, 32ch@128Hz) → extract features
  2. GAMEEMO (28 subjects × 4 games, 14ch@128Hz) → extract features
  3. EEG Emotion Recognition (25 subjects × 4 games, 14ch@128Hz) → extract features
  4. SEED (50,910 samples, 5-band DE × 62 channels) → use DE features
  5. Brainwave Emotions (2,132 samples, pre-extracted) → use directly
  6. DENS (OpenNeuro, 128ch@250Hz) → extract features

Usage: PYTHONUNBUFFERED=1 python3 -u -m training.train_all_datasets
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
import glob
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from processing.eeg_processor import (
    preprocess, extract_features, extract_band_powers, BANDS,
)
from training.data_loaders import _circumplex_to_emotion

# 3-class emotion labels
EMOTION_3 = ["positive", "neutral", "negative"]


def log(msg):
    print(msg, flush=True)


def map_6class_to_3class(label_6):
    """Map 6-class emotions to 3 classes: positive(0), neutral(1), negative(2)."""
    # 0=happy→pos, 1=sad→neg, 2=angry→neg, 3=fearful→neg, 4=relaxed→neu, 5=focused→neu
    mapping = {0: 0, 1: 2, 2: 2, 3: 2, 4: 1, 5: 1}
    return mapping.get(label_6, 1)


# ====================================================================
# DATASET 1: DEAP
# ====================================================================
def load_deap_features(fs=128.0):
    """Load DEAP and extract 17 features per trial."""
    deap_path = Path("data/deap")
    dat_files = sorted(deap_path.glob("s*.dat"))
    if not dat_files:
        log("  DEAP: No .dat files found, skipping")
        return None, None

    FRONTAL_CH = [0, 1, 2, 3, 4, 5, 16, 17, 18, 19]
    X_list, y_list = [], []

    for dat_file in dat_files:
        with open(dat_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        eeg = data["data"][:, :32, 384:]  # skip 3s baseline
        labels = data["labels"]

        for trial_idx in range(eeg.shape[0]):
            frontal_avg = eeg[trial_idx][FRONTAL_CH].mean(axis=0)
            processed = preprocess(frontal_avg, fs)
            feats = extract_features(processed, fs)
            X_list.append(list(feats.values()))

            v, a = labels[trial_idx, 0], labels[trial_idx, 1]
            emo_6 = _circumplex_to_emotion(v, a)
            y_list.append(map_6class_to_3class(emo_6))

    return np.array(X_list), np.array(y_list)


# ====================================================================
# DATASET 2: GAMEEMO (28 subjects × 4 games)
# ====================================================================
# Game → emotion mapping (based on game design):
# G1=Boring(LALV)→negative, G2=Horror(HALV)→negative, G3=Funny(HAHV)→positive, G4=Calm(LAHV)→neutral
GAME_LABELS = {"G1": 2, "G2": 2, "G3": 0, "G4": 1}  # pos=0, neu=1, neg=2

GAMEEMO_FRONTAL = ["AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6"]


def load_gameemo_features(epoch_sec=30.0, fs=128.0):
    """Load GAMEEMO preprocessed CSVs, epoch, extract features."""
    gameemo_path = Path(
        "/Users/sravyalu/.cache/kagglehub/datasets/sigfest/"
        "database-for-emotion-recognition-system-gameemo/versions/1/GAMEEMO"
    )
    csv_files = sorted(gameemo_path.glob("*/Preprocessed EEG Data/.csv format/*.csv"))
    if not csv_files:
        log("  GAMEEMO: No CSV files found, skipping")
        return None, None

    epoch_samples = int(epoch_sec * fs)
    X_list, y_list = [], []

    for csv_file in csv_files:
        fname = csv_file.stem  # e.g., "S01G1AllChannels"
        # Extract game number
        game = None
        for g in ["G1", "G2", "G3", "G4"]:
            if g in fname:
                game = g
                break
        if game is None:
            continue

        try:
            df = pd.read_csv(csv_file)
            # Get frontal channels
            frontal_cols = [c for c in GAMEEMO_FRONTAL if c in df.columns]
            if not frontal_cols:
                continue

            frontal_avg = df[frontal_cols].mean(axis=1).values

            # Epoch into windows
            n_epochs = len(frontal_avg) // epoch_samples
            for ep in range(n_epochs):
                start = ep * epoch_samples
                segment = frontal_avg[start:start + epoch_samples]
                if len(segment) < epoch_samples:
                    break

                processed = preprocess(segment, fs)
                feats = extract_features(processed, fs)
                X_list.append(list(feats.values()))
                y_list.append(GAME_LABELS[game])
        except Exception as e:
            continue

    if not X_list:
        return None, None
    return np.array(X_list), np.array(y_list)


# ====================================================================
# DATASET 3: EEG Emotion Recognition (similar to GAMEEMO)
# ====================================================================
def load_eeg_emotion_recognition_features(epoch_sec=30.0, fs=128.0):
    """Load EEG-ER CSVs, epoch, extract features."""
    eer_path = Path(
        "/Users/sravyalu/.cache/kagglehub/datasets/khan1803115/"
        "eeg-dataset-for-emotion-recognition/versions/1/Data"
    )
    csv_files = sorted(eer_path.glob("*.csv"))
    if not csv_files:
        log("  EEG-ER: No CSV files found, skipping")
        return None, None

    epoch_samples = int(epoch_sec * fs)
    X_list, y_list = [], []

    for csv_file in csv_files:
        fname = csv_file.stem
        game = None
        for g in ["G1", "G2", "G3", "G4"]:
            if g in fname:
                game = g
                break
        if game is None:
            continue

        try:
            df = pd.read_csv(csv_file)
            frontal_cols = [c for c in GAMEEMO_FRONTAL if c in df.columns]
            if not frontal_cols:
                continue

            frontal_avg = df[frontal_cols].mean(axis=1).values
            n_epochs = len(frontal_avg) // epoch_samples
            for ep in range(n_epochs):
                start = ep * epoch_samples
                segment = frontal_avg[start:start + epoch_samples]
                if len(segment) < epoch_samples:
                    break

                processed = preprocess(segment, fs)
                feats = extract_features(processed, fs)
                X_list.append(list(feats.values()))
                y_list.append(GAME_LABELS[game])
        except Exception:
            continue

    if not X_list:
        return None, None
    return np.array(X_list), np.array(y_list)


# ====================================================================
# DATASET 4: SEED (pre-extracted DE features)
# ====================================================================
def load_seed_features():
    """Load SEED DE features. Returns (samples, features), labels.

    SEED: (50910, 5, 62) = 5 freq bands × 62 channels.
    We extract: mean DE per band (5), std DE per band (5),
    frontal-mean DE (5), frontal-occipital ratio (5) = 20 features.
    Then add band ratios = 3 more. Total: 23 features.
    Labels: 0=positive, 1=neutral, 2=negative.
    """
    seed_path = Path(
        "/Users/sravyalu/.cache/kagglehub/datasets/daviderusso7/"
        "seed-dataset/versions/1"
    )
    if not seed_path.exists():
        log("  SEED: Dataset not found, skipping")
        return None, None

    data = np.load(seed_path / "DatasetCaricatoNoImage.npz")["arr_0"]  # (50910, 5, 62)
    labels = np.load(seed_path / "LabelsNoImage.npz")["arr_0"]  # (50910,)

    # Extract summary features from the 5×62 DE matrix
    n = len(data)
    feats_list = []

    # Frontal channels in 62-channel cap (approximate: first ~12 are frontal in 10-20)
    frontal_idx = list(range(0, 12))
    occipital_idx = list(range(56, 62))

    for i in range(n):
        de = data[i]  # (5, 62) - 5 bands × 62 channels
        row = []

        # Mean DE per band across all channels (5 features)
        band_means = de.mean(axis=1)
        row.extend(band_means.tolist())

        # Std DE per band (5 features)
        band_stds = de.std(axis=1)
        row.extend(band_stds.tolist())

        # Frontal region mean DE per band (5 features)
        frontal_de = de[:, frontal_idx].mean(axis=1)
        row.extend(frontal_de.tolist())

        # Frontal-to-occipital ratio per band (5 features)
        occipital_de = de[:, occipital_idx].mean(axis=1) + 1e-10
        fo_ratio = frontal_de / occipital_de
        row.extend(fo_ratio.tolist())

        # Band ratios: alpha/beta, theta/beta, alpha/theta (3 features)
        # bands: 0=delta, 1=theta, 2=alpha, 3=beta, 4=gamma
        alpha_m = band_means[2] + 1e-10
        beta_m = band_means[3] + 1e-10
        theta_m = band_means[1] + 1e-10
        row.append(alpha_m / beta_m)
        row.append(theta_m / beta_m)
        row.append(alpha_m / theta_m)

        feats_list.append(row)

    X = np.array(feats_list)
    return X, labels


# ====================================================================
# DATASET 5: Brainwave Emotions (pre-extracted features)
# ====================================================================
def load_brainwave_emotions(n_components=50):
    """Load Brainwave Emotions CSV. PCA to n_components to speed up training."""
    csv_path = Path(
        "/Users/sravyalu/.cache/kagglehub/datasets/birdy654/"
        "eeg-brainwave-dataset-feeling-emotions/versions/1/emotions.csv"
    )
    if not csv_path.exists():
        log("  Brainwave: Dataset not found, skipping")
        return None, None

    df = pd.read_csv(csv_path)
    label_col = df.columns[-1]
    feature_cols = df.columns[:-1]

    X_raw = df[feature_cols].values
    label_map = {"POSITIVE": 0, "NEUTRAL": 1, "NEGATIVE": 2}
    y = df[label_col].map(label_map).values

    # PCA to reduce 2548 features → n_components (speeds up GBM massively)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    # Handle NaN/Inf from scaling
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=n_components, random_state=42)
    X = pca.fit_transform(X_scaled)
    log(f"  PCA: {X_raw.shape[1]} features → {n_components} components "
        f"({pca.explained_variance_ratio_.sum():.1%} variance)")

    return X, y


# ====================================================================
# DATASET 6: DENS
# ====================================================================
def load_dens_features(fs=250.0):
    """Load DENS data and extract features."""
    dens_path = Path("data/dens")
    if not dens_path.exists():
        log("  DENS: Data directory not found, skipping")
        return None, None

    try:
        from training.data_loaders import load_dens
        X_raw, y_raw = load_dens(str(dens_path))
        if X_raw is not None and len(X_raw) > 0:
            # y_raw is already 6-class, map to 3
            y_3 = np.array([map_6class_to_3class(int(label)) for label in y_raw])
            return X_raw, y_3
    except Exception as e:
        log(f"  DENS: Error loading: {e}")

    return None, None


# ====================================================================
# EVALUATION
# ====================================================================
def evaluate_model(X, y, name, n_splits=5):
    """5-fold CV with GBM, return results dict."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accs, f1s = [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        weights = compute_sample_weight("balanced", y_tr)

        # Train both GBM and RF, pick best
        gbm = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )
        gbm.fit(X_tr, y_tr, sample_weight=weights)

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced", random_state=42,
        )
        rf.fit(X_tr, y_tr)

        gbm_acc = accuracy_score(y_te, gbm.predict(X_te))
        rf_acc = accuracy_score(y_te, rf.predict(X_te))
        best_model = gbm if gbm_acc >= rf_acc else rf

        y_pred = best_model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average="macro")
        accs.append(acc)
        f1s.append(f1)
        log(f"    Fold {fold+1}: acc={acc:.4f}, f1={f1:.4f} ({'GBM' if best_model is gbm else 'RF'})")

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
# MAIN
# ====================================================================
if __name__ == "__main__":
    t_start = time.time()
    log("=" * 72)
    log("  MULTI-DATASET EMOTION CLASSIFIER (3-class: positive/neutral/negative)")
    log("=" * 72)

    all_results = []
    datasets_loaded = {}

    # --- Load all raw EEG datasets ---
    log("\n[1/6] Loading DEAP...")
    t = time.time()
    X_deap, y_deap = load_deap_features()
    if X_deap is not None:
        log(f"  DEAP: {X_deap.shape[0]} samples, {X_deap.shape[1]} features ({time.time()-t:.1f}s)")
        log(f"  Labels: {dict(Counter(y_deap))}")
        datasets_loaded["DEAP"] = (X_deap, y_deap)

    log("\n[2/6] Loading GAMEEMO...")
    t = time.time()
    X_gameemo, y_gameemo = load_gameemo_features()
    if X_gameemo is not None:
        log(f"  GAMEEMO: {X_gameemo.shape[0]} samples, {X_gameemo.shape[1]} features ({time.time()-t:.1f}s)")
        log(f"  Labels: {dict(Counter(y_gameemo))}")
        datasets_loaded["GAMEEMO"] = (X_gameemo, y_gameemo)

    log("\n[3/6] Loading EEG Emotion Recognition...")
    t = time.time()
    X_eer, y_eer = load_eeg_emotion_recognition_features()
    if X_eer is not None:
        log(f"  EEG-ER: {X_eer.shape[0]} samples, {X_eer.shape[1]} features ({time.time()-t:.1f}s)")
        log(f"  Labels: {dict(Counter(y_eer))}")
        datasets_loaded["EEG-ER"] = (X_eer, y_eer)

    log("\n[4/6] Loading DENS...")
    t = time.time()
    X_dens, y_dens = load_dens_features()
    if X_dens is not None:
        log(f"  DENS: {X_dens.shape[0]} samples, {X_dens.shape[1]} features ({time.time()-t:.1f}s)")
        log(f"  Labels: {dict(Counter(y_dens))}")
        datasets_loaded["DENS"] = (X_dens, y_dens)

    log("\n[5/6] Loading SEED...")
    t = time.time()
    X_seed, y_seed = load_seed_features()
    if X_seed is not None:
        log(f"  SEED: {X_seed.shape[0]} samples, {X_seed.shape[1]} features ({time.time()-t:.1f}s)")
        log(f"  Labels: {dict(Counter(y_seed))}")
        datasets_loaded["SEED"] = (X_seed, y_seed)

    log("\n[6/6] Loading Brainwave Emotions...")
    t = time.time()
    X_bw, y_bw = load_brainwave_emotions()
    if X_bw is not None:
        log(f"  Brainwave: {X_bw.shape[0]} samples, {X_bw.shape[1]} features ({time.time()-t:.1f}s)")
        log(f"  Labels: {dict(Counter(y_bw))}")
        datasets_loaded["Brainwave"] = (X_bw, y_bw)

    total_samples = sum(X.shape[0] for X, y in datasets_loaded.values())
    log(f"\n>>> Total datasets loaded: {len(datasets_loaded)}, Total samples: {total_samples}")

    # --- Evaluate individual datasets ---
    log("\n" + "=" * 72)
    log("  INDIVIDUAL DATASET RESULTS")
    log("=" * 72)

    for ds_name, (X, y) in datasets_loaded.items():
        if len(X) < 50:
            log(f"\n{ds_name}: Too few samples ({len(X)}), skipping evaluation")
            continue
        log(f"\n--- {ds_name} ({X.shape[0]} samples, {X.shape[1]} features) ---")
        r = evaluate_model(X, y, ds_name)
        all_results.append(r)
        log(f"  >> Accuracy: {r['accuracy_mean']:.4f} +/- {r['accuracy_std']:.4f}")
        log(f"  >> F1 Macro: {r['f1_macro_mean']:.4f} +/- {r['f1_macro_std']:.4f}")

    # --- Combined raw EEG datasets (same 17-feature space) ---
    raw_eeg_datasets = ["DEAP", "GAMEEMO", "EEG-ER", "DENS"]
    raw_parts_X, raw_parts_y = [], []
    for ds in raw_eeg_datasets:
        if ds in datasets_loaded:
            X, y = datasets_loaded[ds]
            if X.shape[1] == 17:  # same feature space
                raw_parts_X.append(X)
                raw_parts_y.append(y)

    if len(raw_parts_X) > 1:
        X_combined_raw = np.vstack(raw_parts_X)
        y_combined_raw = np.concatenate(raw_parts_y)
        combined_names = [ds for ds in raw_eeg_datasets if ds in datasets_loaded and datasets_loaded[ds][0].shape[1] == 17]
        log(f"\n--- COMBINED RAW EEG ({'+'.join(combined_names)}) ---")
        log(f"  Total: {X_combined_raw.shape[0]} samples, {X_combined_raw.shape[1]} features")
        log(f"  Labels: {dict(Counter(y_combined_raw))}")
        r = evaluate_model(X_combined_raw, y_combined_raw, f"Combined: {'+'.join(combined_names)}")
        all_results.append(r)
        log(f"  >> Accuracy: {r['accuracy_mean']:.4f} +/- {r['accuracy_std']:.4f}")
        log(f"  >> F1 Macro: {r['f1_macro_mean']:.4f} +/- {r['f1_macro_std']:.4f}")

    # --- Summary ---
    log("\n" + "=" * 72)
    log("  FINAL SUMMARY: All Models (3-class: positive/neutral/negative)")
    log("=" * 72)
    log(f"  {'Model':<45} {'Acc':>8} {'F1':>8} {'Samples':>8} {'Feats':>6}")
    log(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for r in sorted(all_results, key=lambda x: -x["f1_macro_mean"]):
        log(f"  {r['name']:<45} {r['accuracy_mean']:.4f}  {r['f1_macro_mean']:.4f}  {r['n_samples']:>8} {r['n_features']:>6}")
    log("=" * 72)

    best = max(all_results, key=lambda r: r["f1_macro_mean"])
    log(f"\nBest: {best['name']}")
    log(f"  Accuracy: {best['accuracy_mean']:.4f} +/- {best['accuracy_std']:.4f}")
    log(f"  F1 Macro: {best['f1_macro_mean']:.4f} +/- {best['f1_macro_std']:.4f}")
    log(f"  Samples:  {best['n_samples']}")

    # Save
    out_path = Path("benchmarks/all_datasets_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to {out_path}")
    log(f"Total time: {time.time()-t_start:.1f}s")
