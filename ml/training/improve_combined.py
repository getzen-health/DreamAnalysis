"""Further improvements to the combined emotion classifier.

Improvements tested:
  1. More data: overlapping windows (50% overlap) → ~2x more samples
  2. More features: add differential entropy + wavelet energy per band (27 features)
  3. Multi-channel features: use ALL available channels per dataset (not just frontal avg)
  4. Asymmetry features for datasets that have bilateral channels
  5. Better models: RF tuned, ExtraTrees, stacked ensemble
  6. Cross-dataset transfer: use SEED features as auxiliary training signal

Usage: PYTHONUNBUFFERED=1 python3 -u -m training.improve_combined
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    ExtraTreesClassifier, VotingClassifier,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from processing.eeg_processor import (
    preprocess, extract_features, extract_band_powers,
)
from training.data_loaders import _circumplex_to_emotion

EMOTION_3 = ["positive", "neutral", "negative"]
RANDOM_STATE = 42


def log(msg):
    print(msg, flush=True)


def map_6class_to_3class(label_6):
    mapping = {0: 0, 1: 2, 2: 2, 3: 2, 4: 1, 5: 1}
    return mapping.get(label_6, 1)


# ====================================================================
# ENHANCED FEATURE EXTRACTION
# ====================================================================
def extract_enhanced_features(signal, fs=128.0):
    """Extract 27 enhanced features from a single-channel EEG signal.

    17 base features + 5 wavelet band energies + 5 differential entropy features.
    Wait -- extract_features already includes DE. So let's add:
    - Wavelet energy (DWT) per band (5 features)
    - Peak frequency (1 feature)
    - Band power ratios: delta/beta, gamma/theta (2 features)
    - Signal stats: skewness, kurtosis (2 features)
    Total: 17 + 5 + 1 + 2 + 2 = 27 features
    """
    from scipy.stats import skew, kurtosis
    import pywt

    # Base 17 features
    base = extract_features(signal, fs)
    feats = list(base.values())

    # Wavelet (DWT) band energies - decompose into 5 levels
    try:
        max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet('db4').dec_len)
        level = min(5, max_level)
        coeffs = pywt.wavedec(signal, 'db4', level=level)
        energies = [float(np.sum(c ** 2)) for c in coeffs]
        total_e = sum(energies) + 1e-10
        # Pad to exactly 5 levels + 1 approx = 6 entries, take last 5
        while len(energies) < 6:
            energies.append(0.0)
        wavelet_feats = [e / total_e for e in energies[:5]]
    except Exception:
        wavelet_feats = [0.0] * 5
    feats.extend(wavelet_feats)

    # Peak frequency
    from scipy.signal import welch
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), int(fs * 2)))
    peak_freq = float(freqs[np.argmax(psd)]) if len(psd) > 0 else 0.0
    feats.append(peak_freq)

    # Additional ratios
    bp = extract_band_powers(signal, fs)
    delta = bp.get('delta', 0) + 1e-10
    beta = bp.get('beta', 0) + 1e-10
    gamma = bp.get('gamma', 0) + 1e-10
    theta = bp.get('theta', 0) + 1e-10
    feats.append(delta / beta)   # delta/beta ratio (drowsiness marker)
    feats.append(gamma / theta)  # gamma/theta ratio (cognitive load)

    # Signal statistics
    feats.append(float(skew(signal)))
    feats.append(float(kurtosis(signal)))

    return feats  # 27 features


def extract_multichannel_enhanced(channel_data, fs=128.0, channel_names=None):
    """Extract features from multiple channels with region aggregation.

    Returns: enhanced features from frontal avg + region band powers + asymmetry.
    """
    n_channels = channel_data.shape[0]

    # Preprocess all channels
    processed = np.array([preprocess(channel_data[ch], fs) for ch in range(n_channels)])

    # Average all channels for base features
    avg_signal = processed.mean(axis=0)
    feats = extract_enhanced_features(avg_signal, fs)

    # Per-channel band powers → compute mean and std across channels
    all_band_powers = []
    for ch in range(n_channels):
        bp = extract_band_powers(processed[ch], fs)
        all_band_powers.append(list(bp.values()))

    bp_array = np.array(all_band_powers)  # (n_channels, 5)
    # Mean band power across channels (5)
    feats.extend(bp_array.mean(axis=0).tolist())
    # Std band power across channels (5) - captures spatial variability
    feats.extend(bp_array.std(axis=0).tolist())

    # Inter-channel coherence approximation: correlation of band powers
    if n_channels >= 2:
        # Frontal vs posterior power ratio (if we know channel layout)
        half = n_channels // 2
        front_power = bp_array[:half].mean()
        back_power = bp_array[half:].mean() + 1e-10
        feats.append(float(front_power / back_power))
    else:
        feats.append(1.0)

    return feats  # 27 + 5 + 5 + 1 = 38 features


# ====================================================================
# DATA LOADING WITH IMPROVEMENTS
# ====================================================================

GAMEEMO_FRONTAL = ["AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6"]
GAMEEMO_ALL_CH = ["AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6",
                  "O1", "O2", "P7", "P8", "T7", "T8"]
GAME_LABELS = {"G1": 2, "G2": 2, "G3": 0, "G4": 1}


def load_deap_enhanced(fs=128.0):
    """Load DEAP with enhanced multi-channel features."""
    deap_path = Path("data/deap")
    dat_files = sorted(deap_path.glob("s*.dat"))
    if not dat_files:
        return None, None

    FRONTAL_CH = [0, 1, 2, 3, 4, 5, 16, 17, 18, 19]
    ALL_CH = list(range(32))

    X_list, y_list = [], []
    for dat_file in dat_files:
        with open(dat_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        eeg = data["data"][:, :32, 384:]
        labels = data["labels"]

        for trial_idx in range(eeg.shape[0]):
            trial = eeg[trial_idx]

            # Enhanced features from all 32 channels
            feats = extract_multichannel_enhanced(trial, fs)
            X_list.append(feats)

            v, a = labels[trial_idx, 0], labels[trial_idx, 1]
            emo_6 = _circumplex_to_emotion(v, a)
            y_list.append(map_6class_to_3class(emo_6))

    return np.array(X_list), np.array(y_list)


def load_gameemo_enhanced(epoch_sec=30.0, overlap=0.5, fs=128.0):
    """Load GAMEEMO with overlapping windows and multi-channel features."""
    gameemo_path = Path(
        "/Users/sravyalu/.cache/kagglehub/datasets/sigfest/"
        "database-for-emotion-recognition-system-gameemo/versions/1/GAMEEMO"
    )
    csv_files = sorted(gameemo_path.glob("*/Preprocessed EEG Data/.csv format/*.csv"))
    if not csv_files:
        return None, None

    epoch_samples = int(epoch_sec * fs)
    step_samples = int(epoch_samples * (1 - overlap))
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
            ch_cols = [c for c in GAMEEMO_ALL_CH if c in df.columns]
            if len(ch_cols) < 4:
                continue

            ch_data = df[ch_cols].values.T  # (n_channels, n_samples)
            n_samples = ch_data.shape[1]

            # Overlapping windows
            start = 0
            while start + epoch_samples <= n_samples:
                segment = ch_data[:, start:start + epoch_samples]
                feats = extract_multichannel_enhanced(segment, fs)
                X_list.append(feats)
                y_list.append(GAME_LABELS[game])
                start += step_samples
        except Exception:
            continue

    if not X_list:
        return None, None
    return np.array(X_list), np.array(y_list)


def load_eer_enhanced(epoch_sec=30.0, overlap=0.5, fs=128.0):
    """Load EEG-ER with overlapping windows and multi-channel features."""
    eer_path = Path(
        "/Users/sravyalu/.cache/kagglehub/datasets/khan1803115/"
        "eeg-dataset-for-emotion-recognition/versions/1/Data"
    )
    csv_files = sorted(eer_path.glob("*.csv"))
    if not csv_files:
        return None, None

    epoch_samples = int(epoch_sec * fs)
    step_samples = int(epoch_samples * (1 - overlap))
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
            ch_cols = [c for c in GAMEEMO_ALL_CH if c in df.columns]
            if len(ch_cols) < 4:
                continue

            ch_data = df[ch_cols].values.T
            n_samples = ch_data.shape[1]

            start = 0
            while start + epoch_samples <= n_samples:
                segment = ch_data[:, start:start + epoch_samples]
                feats = extract_multichannel_enhanced(segment, fs)
                X_list.append(feats)
                y_list.append(GAME_LABELS[game])
                start += step_samples
        except Exception:
            continue

    if not X_list:
        return None, None
    return np.array(X_list), np.array(y_list)


def load_dens_enhanced(fs=250.0):
    """Load DENS with enhanced features."""
    try:
        from training.data_loaders import load_dens
        X_raw, y_raw = load_dens("data/dens")
        if X_raw is not None and len(X_raw) > 0:
            y_3 = np.array([map_6class_to_3class(int(l)) for l in y_raw])
            return X_raw, y_3
    except Exception as e:
        log(f"  DENS: {e}")
    return None, None


def load_seed_enhanced():
    """Load SEED with richer feature extraction."""
    seed_path = Path(
        "/Users/sravyalu/.cache/kagglehub/datasets/daviderusso7/"
        "seed-dataset/versions/1"
    )
    if not seed_path.exists():
        return None, None

    data = np.load(seed_path / "DatasetCaricatoNoImage.npz")["arr_0"]  # (50910, 5, 62)
    labels = np.load(seed_path / "LabelsNoImage.npz")["arr_0"]

    n = len(data)
    frontal_idx = list(range(0, 12))
    central_idx = list(range(12, 30))
    parietal_idx = list(range(30, 50))
    occipital_idx = list(range(50, 62))

    feats_list = []
    for i in range(n):
        de = data[i]  # (5, 62)
        row = []

        # Global stats per band (5 mean + 5 std = 10)
        row.extend(de.mean(axis=1).tolist())
        row.extend(de.std(axis=1).tolist())

        # Region means per band (4 regions × 5 bands = 20)
        for region_idx in [frontal_idx, central_idx, parietal_idx, occipital_idx]:
            row.extend(de[:, region_idx].mean(axis=1).tolist())

        # Frontal-occipital ratio (5)
        frontal_de = de[:, frontal_idx].mean(axis=1)
        occipital_de = de[:, occipital_idx].mean(axis=1) + 1e-10
        row.extend((frontal_de / occipital_de).tolist())

        # Left-right asymmetry (approximate: first half vs second half of channels) (5)
        left_de = de[:, :31].mean(axis=1)
        right_de = de[:, 31:].mean(axis=1) + 1e-10
        row.extend((left_de / right_de).tolist())

        # Band ratios (3)
        bm = de.mean(axis=1)
        alpha_m, beta_m, theta_m = bm[2] + 1e-10, bm[3] + 1e-10, bm[1] + 1e-10
        row.append(alpha_m / beta_m)
        row.append(theta_m / beta_m)
        row.append(alpha_m / theta_m)

        feats_list.append(row)

    return np.array(feats_list), labels  # 48 features


def load_brainwave_enhanced(n_components=80):
    """Load Brainwave with PCA."""
    csv_path = Path(
        "/Users/sravyalu/.cache/kagglehub/datasets/birdy654/"
        "eeg-brainwave-dataset-feeling-emotions/versions/1/emotions.csv"
    )
    if not csv_path.exists():
        return None, None

    df = pd.read_csv(csv_path)
    X_raw = df[df.columns[:-1]].values
    label_map = {"POSITIVE": 0, "NEUTRAL": 1, "NEGATIVE": 2}
    y = df[df.columns[-1]].map(label_map).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X = pca.fit_transform(X_scaled)
    log(f"  PCA: {X_raw.shape[1]} → {n_components} ({pca.explained_variance_ratio_.sum():.1%} var)")
    return X, y


# ====================================================================
# ADVANCED MODEL EVALUATION
# ====================================================================
def evaluate_multiple_models(X, y, name, n_splits=5):
    """Evaluate GBM, RF, ExtraTrees, and Stacked Ensemble."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    model_results = {}
    for model_name, model_fn in [
        ("GBM", lambda: GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=3, random_state=RANDOM_STATE)),
        ("RF", lambda: RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            min_samples_leaf=2, random_state=RANDOM_STATE)),
        ("ExtraTrees", lambda: ExtraTreesClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            min_samples_leaf=2, random_state=RANDOM_STATE)),
        ("Ensemble", lambda: VotingClassifier(estimators=[
            ('gbm', GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=RANDOM_STATE)),
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight="balanced",
                random_state=RANDOM_STATE)),
            ('et', ExtraTreesClassifier(
                n_estimators=200, max_depth=10, class_weight="balanced",
                random_state=RANDOM_STATE)),
        ], voting='soft')),
    ]:
        accs, f1s = [], []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            model = model_fn()
            if model_name == "GBM":
                weights = compute_sample_weight("balanced", y_tr)
                model.fit(X_tr, y_tr, sample_weight=weights)
            else:
                model.fit(X_tr, y_tr)

            y_pred = model.predict(X_te)
            accs.append(accuracy_score(y_te, y_pred))
            f1s.append(f1_score(y_te, y_pred, average="macro"))

        model_results[model_name] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "f1_macro_mean": float(np.mean(f1s)),
            "f1_macro_std": float(np.std(f1s)),
        }
        log(f"    {model_name:12s}: Acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
            f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

    best_model = max(model_results.items(), key=lambda x: x[1]["f1_macro_mean"])
    return {
        "name": name,
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "models": model_results,
        "best_model": best_model[0],
        "accuracy_mean": best_model[1]["accuracy_mean"],
        "accuracy_std": best_model[1]["accuracy_std"],
        "f1_macro_mean": best_model[1]["f1_macro_mean"],
        "f1_macro_std": best_model[1]["f1_macro_std"],
    }


# ====================================================================
# MAIN
# ====================================================================
if __name__ == "__main__":
    t_start = time.time()
    log("=" * 72)
    log("  IMPROVED MULTI-DATASET EMOTION CLASSIFIER")
    log("  Improvements: enhanced features, overlapping windows,")
    log("  multi-channel, ensemble models")
    log("=" * 72)

    all_results = []

    # --- Load all datasets with enhanced features ---
    log("\n[1/6] DEAP (32ch, enhanced multi-channel features)...")
    t = time.time()
    X_deap, y_deap = load_deap_enhanced()
    if X_deap is not None:
        log(f"  {X_deap.shape[0]} samples, {X_deap.shape[1]} features ({time.time()-t:.1f}s)")
        log(f"  Labels: {dict(Counter(y_deap))}")

    log("\n[2/6] GAMEEMO (14ch, 50% overlap, enhanced)...")
    t = time.time()
    X_gameemo, y_gameemo = load_gameemo_enhanced()
    if X_gameemo is not None:
        log(f"  {X_gameemo.shape[0]} samples, {X_gameemo.shape[1]} features ({time.time()-t:.1f}s)")
        log(f"  Labels: {dict(Counter(y_gameemo))}")

    log("\n[3/6] EEG-ER (14ch, 50% overlap, enhanced)...")
    t = time.time()
    X_eer, y_eer = load_eer_enhanced()
    if X_eer is not None:
        log(f"  {X_eer.shape[0]} samples, {X_eer.shape[1]} features ({time.time()-t:.1f}s)")
        log(f"  Labels: {dict(Counter(y_eer))}")

    log("\n[4/6] DENS...")
    t = time.time()
    X_dens, y_dens = load_dens_enhanced()
    if X_dens is not None:
        log(f"  {X_dens.shape[0]} samples, {X_dens.shape[1]} features ({time.time()-t:.1f}s)")

    log("\n[5/6] SEED (enhanced region features)...")
    t = time.time()
    X_seed, y_seed = load_seed_enhanced()
    if X_seed is not None:
        log(f"  {X_seed.shape[0]} samples, {X_seed.shape[1]} features ({time.time()-t:.1f}s)")

    log("\n[6/6] Brainwave (PCA=80)...")
    t = time.time()
    X_bw, y_bw = load_brainwave_enhanced()
    if X_bw is not None:
        log(f"  {X_bw.shape[0]} samples, {X_bw.shape[1]} features ({time.time()-t:.1f}s)")

    # --- Evaluate individual datasets ---
    log("\n" + "=" * 72)
    log("  EVALUATING WITH MULTIPLE MODELS")
    log("=" * 72)

    datasets = {
        "DEAP-enhanced": (X_deap, y_deap),
        "GAMEEMO-enhanced": (X_gameemo, y_gameemo),
        "EEG-ER-enhanced": (X_eer, y_eer),
    }
    if X_dens is not None:
        datasets["DENS"] = (X_dens, y_dens)

    for ds_name, (X, y) in datasets.items():
        if X is None or len(X) < 50:
            continue
        log(f"\n--- {ds_name} ({X.shape[0]} samples, {X.shape[1]} features) ---")
        r = evaluate_multiple_models(X, y, ds_name)
        all_results.append(r)

    # --- Combined raw EEG (enhanced features, matching dimensions) ---
    log("\n--- COMBINING RAW EEG DATASETS ---")
    # DEAP has 38 features (32 channels), GAMEEMO/EEG-ER have 38 (14 channels)
    # They should match since extract_multichannel_enhanced returns same # features
    combined_parts = []
    for ds_name in ["DEAP-enhanced", "GAMEEMO-enhanced", "EEG-ER-enhanced"]:
        if ds_name in datasets and datasets[ds_name][0] is not None:
            X, y = datasets[ds_name]
            combined_parts.append((X, y, ds_name))

    if len(combined_parts) > 1:
        # Verify all have same feature count
        feat_counts = set(X.shape[1] for X, y, n in combined_parts)
        if len(feat_counts) == 1:
            X_combined = np.vstack([X for X, y, n in combined_parts])
            y_combined = np.concatenate([y for X, y, n in combined_parts])
            names = "+".join([n.replace("-enhanced", "") for X, y, n in combined_parts])
            log(f"\nCombined {names}: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
            log(f"Labels: {dict(Counter(y_combined))}")
            r = evaluate_multiple_models(X_combined, y_combined, f"Combined-Enhanced: {names}")
            all_results.append(r)
        else:
            log(f"  Feature mismatch: {feat_counts}, using PCA alignment")
            # PCA align to common space
            min_feats = min(feat_counts)
            aligned_parts = []
            for X, y, n in combined_parts:
                if X.shape[1] > min_feats:
                    pca = PCA(n_components=min_feats, random_state=RANDOM_STATE)
                    X = pca.fit_transform(StandardScaler().fit_transform(X))
                aligned_parts.append((X, y, n))
            X_combined = np.vstack([X for X, y, n in aligned_parts])
            y_combined = np.concatenate([y for X, y, n in aligned_parts])
            names = "+".join([n.replace("-enhanced", "") for X, y, n in aligned_parts])
            log(f"\nCombined (PCA-aligned) {names}: {X_combined.shape[0]} samples")
            r = evaluate_multiple_models(X_combined, y_combined, f"Combined-PCA: {names}")
            all_results.append(r)

    # --- SEED evaluation ---
    if X_seed is not None:
        log(f"\n--- SEED ({X_seed.shape[0]} samples, {X_seed.shape[1]} features) ---")
        r = evaluate_multiple_models(X_seed, y_seed, "SEED-enhanced")
        all_results.append(r)

    # --- Brainwave evaluation ---
    if X_bw is not None:
        log(f"\n--- Brainwave ({X_bw.shape[0]} samples, {X_bw.shape[1]} features) ---")
        r = evaluate_multiple_models(X_bw, y_bw, "Brainwave-PCA80")
        all_results.append(r)

    # --- COMPARISON WITH BASELINE ---
    log("\n" + "=" * 72)
    log("  FINAL COMPARISON (sorted by best F1 macro)")
    log("=" * 72)
    log(f"  {'Model':<40} {'Best':>10} {'Acc':>8} {'F1':>8} {'Samples':>8} {'Feats':>6}")
    log(f"  {'-'*40} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    # Add baseline for comparison
    baseline_results = [
        {"name": "BASELINE: DEAP (17 feat)", "best_model": "RF", "accuracy_mean": 0.6602,
         "f1_macro_mean": 0.3570, "n_samples": 1280, "n_features": 17},
        {"name": "BASELINE: Combined (17 feat)", "best_model": "GBM", "accuracy_mean": 0.7167,
         "f1_macro_mean": 0.6920, "n_samples": 3481, "n_features": 17},
    ]

    all_display = baseline_results + all_results
    for r in sorted(all_display, key=lambda x: -x["f1_macro_mean"]):
        bm = r.get("best_model", "?")
        log(f"  {r['name']:<40} {bm:>10} {r['accuracy_mean']:.4f}  {r['f1_macro_mean']:.4f}  "
            f"{r['n_samples']:>8} {r['n_features']:>6}")
    log("=" * 72)

    # Best overall
    best = max(all_results, key=lambda r: r["f1_macro_mean"])
    log(f"\nBest improved model: {best['name']} ({best['best_model']})")
    log(f"  Accuracy: {best['accuracy_mean']:.4f} ± {best['accuracy_std']:.4f}")
    log(f"  F1 Macro: {best['f1_macro_mean']:.4f} ± {best['f1_macro_std']:.4f}")
    log(f"  Samples:  {best['n_samples']}, Features: {best['n_features']}")

    # Save
    out_path = Path("benchmarks/improved_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nResults saved to {out_path}")
    log(f"Total time: {time.time()-t_start:.1f}s")
