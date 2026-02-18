"""Train 6-class emotion classifier on DEAP using Muse 2-equivalent channels.

Extracts 4 channels from DEAP's 32-channel layout that match Muse 2 positions:
  AF7 → AF3 (idx 1), AF8 → AF4 (idx 17), TP9 → T7 (idx 7), TP10 → T8 (idx 23)

Uses 4-second sliding windows (2s overlap) for ~15× more training samples than
single-trial extraction. Trains LightGBM with 5-fold stratified CV, exports to
ONNX, and stores benchmark. Also supports incremental learning from real Muse 2
data collected during live sessions.

Usage:
    cd ml && python -m training.train_deap_muse
    cd ml && python -m training.train_deap_muse --online-data data/muse2_collected
"""

import argparse
import json
import pickle
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from processing.eeg_processor import preprocess, extract_features, extract_band_powers

# ── Constants ──────────────────────────────────────────────────────────────

EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]

# DEAP BioSemi 32-channel indices matching Muse 2 electrode positions
# Muse 2: TP9 (left temporal), AF7 (left frontal), AF8 (right frontal), TP10 (right temporal)
MUSE_CHANNEL_MAP = {
    "AF7": 1,   # AF3 in DEAP
    "AF8": 17,  # AF4 in DEAP
    "TP9": 7,   # T7 in DEAP
    "TP10": 23, # T8 in DEAP
}
MUSE_CH_INDICES = list(MUSE_CHANNEL_MAP.values())  # [1, 17, 7, 23]
MUSE_CH_NAMES = list(MUSE_CHANNEL_MAP.keys())

DEAP_FS = 128.0  # DEAP preprocessed sampling rate
BASELINE_SAMPLES = 384  # 3s baseline at 128 Hz
EPOCH_SEC = 4.0  # 4-second windows (512 samples at 128 Hz)
OVERLAP = 0.5    # 50% overlap


# ── Valence-Arousal → 6-class emotion mapping ─────────────────────────────

def circumplex_to_emotion(valence: float, arousal: float) -> int:
    """Map DEAP valence-arousal (0-9 scale) to 6 emotion classes.

    Uses data-driven thresholds based on DEAP's actual distribution
    (median V=3.08, A=3.31). Splits each quadrant to get 6 classes.
    """
    # DEAP medians as split points
    v_mid = 3.08
    a_mid = 3.31

    v_high = valence >= v_mid
    a_high = arousal >= a_mid

    if v_high and a_high:
        # Positive + activated: happy (strong) or focused (moderate)
        if valence >= 4.5 or arousal >= 4.5:
            return 0  # happy
        return 5  # focused
    elif not v_high and not a_high:
        # Negative + deactivated: sad
        return 1  # sad
    elif not v_high and a_high:
        # Negative + activated: angry (high arousal) or fearful (moderate)
        if arousal >= 4.5:
            return 2  # angry
        return 3  # fearful
    else:
        # Positive + deactivated: relaxed
        return 4  # relaxed


# ── Feature extraction (per-channel + cross-channel) ──────────────────────

def extract_multichannel_features(
    channels: np.ndarray, fs: float
) -> Dict[str, float]:
    """Extract features from 4-channel EEG (Muse 2 layout).

    Per-channel: 17 features × 4 channels = 68
    Cross-channel: asymmetry (frontal, temporal), coherence = 6
    Aggregate: mean/std of band powers across channels = 10
    Total: ~84 features
    """
    features = {}
    all_band_powers = []
    all_per_channel = []

    for ch_idx, ch_name in enumerate(MUSE_CH_NAMES):
        sig = channels[ch_idx]
        if len(sig) < 34:
            continue
        processed = preprocess(sig, fs)
        ch_features = extract_features(processed, fs)
        bands = extract_band_powers(processed, fs)
        all_band_powers.append(bands)
        all_per_channel.append(ch_features)

        # Per-channel features with channel prefix
        for k, v in ch_features.items():
            features[f"{ch_name}_{k}"] = v

    if len(all_band_powers) < 2:
        return features

    # Cross-channel: frontal asymmetry (AF7 vs AF8)
    if len(all_band_powers) >= 2:
        af7_bands = all_band_powers[0]  # AF7
        af8_bands = all_band_powers[1]  # AF8
        for band in ["alpha", "beta", "theta", "gamma"]:
            left = af7_bands.get(band, 0)
            right = af8_bands.get(band, 0)
            features[f"frontal_asym_{band}"] = (right - left) / max(right + left, 1e-10)

    # Cross-channel: temporal asymmetry (TP9 vs TP10)
    if len(all_band_powers) >= 4:
        tp9_bands = all_band_powers[2]   # TP9
        tp10_bands = all_band_powers[3]  # TP10
        for band in ["alpha", "beta"]:
            left = tp9_bands.get(band, 0)
            right = tp10_bands.get(band, 0)
            features[f"temporal_asym_{band}"] = (right - left) / max(right + left, 1e-10)

    # Aggregate: mean and std of band powers across channels
    for band in ["delta", "theta", "alpha", "beta", "gamma"]:
        vals = [bp.get(band, 0) for bp in all_band_powers]
        features[f"mean_{band}"] = float(np.mean(vals))
        features[f"std_{band}"] = float(np.std(vals))

    return features


# ── DEAP data loading ─────────────────────────────────────────────────────

def load_deap_muse_channels(
    data_dir: str = "data/deap",
    epoch_sec: float = EPOCH_SEC,
    overlap: float = OVERLAP,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load DEAP data using only Muse 2-equivalent channels with sliding windows.

    Returns:
        X: (n_samples, n_features) feature matrix
        y: (n_samples,) emotion labels 0-5
        feature_names: list of feature names
    """
    deap_path = Path(data_dir)
    dat_files = sorted(deap_path.glob("s*.dat"))

    if not dat_files:
        raise FileNotFoundError(f"No DEAP .dat files found in {data_dir}")

    print(f"Loading {len(dat_files)} DEAP subjects with Muse 2 channels...")
    print(f"  Channels: {MUSE_CH_NAMES} → DEAP indices {MUSE_CH_INDICES}")
    print(f"  Window: {epoch_sec}s, overlap: {overlap*100:.0f}%")

    epoch_samples = int(epoch_sec * DEAP_FS)
    step_samples = int(epoch_samples * (1 - overlap))

    X_all, y_all = [], []
    feature_names = None
    class_counts = {i: 0 for i in range(6)}

    for dat_file in dat_files:
        subj = dat_file.stem
        with open(dat_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        eeg = data["data"][:, :32, :]  # (40 trials, 32 ch, 8064 samples)
        labels = data["labels"]         # (40 trials, 4) [valence, arousal, dominance, liking]

        subj_count = 0
        for trial_idx in range(eeg.shape[0]):
            valence = labels[trial_idx, 0]
            arousal = labels[trial_idx, 1]
            emotion = circumplex_to_emotion(valence, arousal)

            # Extract Muse 2-equivalent channels, skip 3s baseline
            trial_data = eeg[trial_idx, MUSE_CH_INDICES, BASELINE_SAMPLES:]
            # trial_data shape: (4, 7680) — 4 channels, 60s at 128 Hz

            # Sliding windows
            n_total = trial_data.shape[1]
            pos = 0
            while pos + epoch_samples <= n_total:
                window = trial_data[:, pos:pos + epoch_samples]

                try:
                    features = extract_multichannel_features(window, DEAP_FS)
                    if feature_names is None:
                        feature_names = sorted(features.keys())
                    feat_vec = [features.get(k, 0.0) for k in feature_names]
                    X_all.append(feat_vec)
                    y_all.append(emotion)
                    class_counts[emotion] += 1
                    subj_count += 1
                except Exception:
                    pass

                pos += step_samples

        print(f"  {subj}: {subj_count} epochs")

    X = np.array(X_all)
    y = np.array(y_all)

    print(f"\nTotal: {len(y)} samples, {len(feature_names)} features")
    print("Class distribution:")
    for i, name in enumerate(EMOTIONS):
        print(f"  {name:10s}: {class_counts[i]:5d} ({class_counts[i]/len(y)*100:.1f}%)")

    return X, y, feature_names


# ── Online learning data ──────────────────────────────────────────────────

def load_muse2_collected(
    data_dir: str = "data/muse2_collected",
    epoch_sec: float = EPOCH_SEC,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load real Muse 2 data collected during live sessions.

    Expects JSON files with format:
    {
        "signals": [[ch0], [ch1], [ch2], [ch3]],
        "sample_rate": 256,
        "label": "happy" | "sad" | ... | 0-5,
        "timestamp": "..."
    }
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return np.array([]), np.array([]), []

    json_files = sorted(data_path.glob("*.json"))
    if not json_files:
        return np.array([]), np.array([]), []

    label_map = {name: idx for idx, name in enumerate(EMOTIONS)}
    epoch_samples = int(epoch_sec * 256)

    X_all, y_all = [], []
    feature_names = None

    for jf in json_files:
        try:
            with open(jf) as f:
                rec = json.load(f)
            signals = np.array(rec["signals"])
            fs = rec.get("sample_rate", 256)
            label = rec["label"]
            if isinstance(label, str):
                label = label_map.get(label, -1)
            if label < 0 or label > 5:
                continue

            # Extract windows
            n_total = signals.shape[1]
            step = epoch_samples // 2
            pos = 0
            while pos + epoch_samples <= n_total:
                window = signals[:4, pos:pos + epoch_samples]
                features = extract_multichannel_features(window, fs)
                if feature_names is None:
                    feature_names = sorted(features.keys())
                feat_vec = [features.get(k, 0.0) for k in feature_names]
                X_all.append(feat_vec)
                y_all.append(label)
                pos += step

        except Exception as e:
            print(f"  Skipping {jf.name}: {e}")

    if X_all:
        print(f"Loaded {len(y_all)} Muse 2 samples from {data_dir}")

    return np.array(X_all) if X_all else np.array([]), np.array(y_all) if y_all else np.array([]), feature_names or []


# ── Training ──────────────────────────────────────────────────────────────

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_folds: int = 5,
) -> dict:
    """Train LightGBM with stratified k-fold cross-validation.

    Returns dict with model, scaler, scores, and feature importance.
    """
    import lightgbm as lgb

    print(f"\nTraining LightGBM ({n_folds}-fold CV)...")
    print(f"  Samples: {len(y)}, Features: {X.shape[1]}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Replace NaN/Inf with 0
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_f1s = []
    best_model = None
    best_acc = 0.0
    all_y_true, all_y_pred = [], []

    params = {
        "objective": "multiclass",
        "num_class": 6,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "max_depth": 8,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")

        fold_accuracies.append(acc)
        fold_f1s.append(f1)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

        print(f"  Fold {fold+1}: Accuracy={acc:.4f}, F1={f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    mean_acc = np.mean(fold_accuracies)
    mean_f1 = np.mean(fold_f1s)

    print(f"\n{'='*50}")
    print(f"Mean Accuracy: {mean_acc:.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"Mean F1 Macro: {mean_f1:.4f} ± {np.std(fold_f1s):.4f}")
    print(f"{'='*50}")

    # Full classification report
    present_labels = sorted(set(all_y_true) | set(all_y_pred))
    present_names = [EMOTIONS[i] for i in present_labels]
    print("\nClassification Report:")
    print(classification_report(
        all_y_true, all_y_pred,
        labels=present_labels, target_names=present_names, digits=3
    ))

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Feature importance
    importance = best_model.feature_importances_
    top_features = sorted(
        zip(feature_names, importance), key=lambda x: x[1], reverse=True
    )[:15]
    print("\nTop 15 features:")
    for fname, imp in top_features:
        print(f"  {fname:40s} {imp:6.0f}")

    # Retrain best model on full data for production
    print("\nRetraining on full dataset for production model...")
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X_scaled, y)

    return {
        "model": final_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "mean_accuracy": float(mean_acc),
        "mean_f1": float(mean_f1),
        "fold_accuracies": [float(a) for a in fold_accuracies],
        "fold_f1s": [float(f) for f in fold_f1s],
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y),
    }


# ── Export ────────────────────────────────────────────────────────────────

def save_model(result: dict, output_dir: str = "models/saved"):
    """Save trained model as sklearn .pkl and attempt ONNX export."""
    import joblib

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save sklearn model
    pkl_path = out / "emotion_deap_muse.pkl"
    joblib.dump({
        "model": result["model"],
        "scaler": result["scaler"],
        "feature_names": result["feature_names"],
    }, pkl_path)
    print(f"\nSaved sklearn model: {pkl_path} ({pkl_path.stat().st_size / 1024:.0f} KB)")

    # Try ONNX export (use onnxmltools for LightGBM support)
    onnx_path = out / "emotion_deap_muse.onnx"
    try:
        import onnxmltools
        from onnxmltools.convert.lightgbm.shape_calculators import calculate_linear_classifier_output_shapes  # noqa: F401
        from skl2onnx.common.data_types import FloatTensorType

        n_features = len(result["feature_names"])
        initial_type = [("input", FloatTensorType([None, n_features]))]
        onnx_model = onnxmltools.convert_lightgbm(
            result["model"], initial_types=initial_type, target_opset=11
        )

        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Saved ONNX model: {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB)")
    except Exception as e:
        print(f"ONNX export failed: {e}")

    # Save benchmark
    bench_path = Path("benchmarks/emotion_classifier_benchmark.json")
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark = {
        "model_name": "emotion_deap_muse",
        "dataset": f"DEAP-32subj-4ch-muse",
        "accuracy": result["mean_accuracy"],
        "f1_macro": result["mean_f1"],
        "n_samples": result["n_samples"],
        "n_features": len(result["feature_names"]),
        "fold_accuracies": result["fold_accuracies"],
        "per_class": {
            EMOTIONS[i]: {
                "support": int(sum(1 for y in result["confusion_matrix"][i]))
            } for i in range(6)
        },
        "confusion_matrix": result["confusion_matrix"],
        "channels": MUSE_CH_NAMES,
        "epoch_sec": EPOCH_SEC,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    bench_path.write_text(json.dumps(benchmark, indent=2))
    print(f"Saved benchmark: {bench_path}")

    return pkl_path, onnx_path


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train emotion classifier on DEAP (Muse 2 channels)")
    parser.add_argument("--data-dir", default="data/deap", help="DEAP data directory")
    parser.add_argument("--online-data", default=None, help="Directory with collected Muse 2 data")
    parser.add_argument("--epoch-sec", type=float, default=EPOCH_SEC, help="Epoch duration (seconds)")
    parser.add_argument("--overlap", type=float, default=OVERLAP, help="Sliding window overlap (0-1)")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    start = time.time()

    # Load DEAP data
    X, y, feature_names = load_deap_muse_channels(
        data_dir=args.data_dir,
        epoch_sec=args.epoch_sec,
        overlap=args.overlap,
    )

    # Load any real Muse 2 data
    if args.online_data:
        X_muse, y_muse, muse_features = load_muse2_collected(args.online_data)
        if len(X_muse) > 0:
            # Align features (DEAP at 128 Hz may produce slightly different features)
            # Use intersection of feature names
            common = sorted(set(feature_names) & set(muse_features))
            if len(common) >= 20:
                deap_idx = [feature_names.index(f) for f in common]
                muse_idx = [muse_features.index(f) for f in common]
                X = np.vstack([X[:, deap_idx], X_muse[:, muse_idx]])
                y = np.concatenate([y, y_muse])
                feature_names = common
                print(f"Combined: {len(y)} total samples ({len(X_muse)} from Muse 2)")

    # Train
    result = train_model(X, y, feature_names, n_folds=args.folds)

    # Save
    pkl_path, onnx_path = save_model(result)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Accuracy: {result['mean_accuracy']*100:.1f}%  F1: {result['mean_f1']*100:.1f}%")

    # Check if model meets the 60% threshold for production use
    if result["mean_accuracy"] >= 0.60:
        print(f"\n✓ Model exceeds 60% accuracy threshold — will be used for live inference!")
    else:
        print(f"\n✗ Model below 60% threshold — feature-based classifier will be used instead.")
        print("  To improve: collect more Muse 2 data with --online-data flag.")


if __name__ == "__main__":
    main()
