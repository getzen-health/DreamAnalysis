#!/usr/bin/env python3
"""Train a LightGBM emotion classifier on Muse-Subconscious 80-feature data.

Features: 4 stats (mean / std / median / IQR) × 5 bands × 4 channels = 80 features.
Labels  : 0 = positive (mellow > 0.6), 1 = neutral (concentration > 0.6), 2 = negative.

This model is designed for *live* Muse 2 inference.  It uses raw 80-dimensional
feature vectors (no PCA) so that the same feature extraction can be reproduced on
incoming EEG without requiring a saved PCA transform.

Usage (run from the ml/ directory):
    python training/train_muse_live_model.py
"""

import json
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths — relative to ml/
# ---------------------------------------------------------------------------
ZIP_PATH   = Path("data/Muse_Subconscious.zip")
MODEL_OUT  = Path("models/saved/emotion_lgbm_muse_live.pkl")
BENCH_OUT  = Path("benchmarks/emotion_lgbm_muse_live_benchmark.json")

# ---------------------------------------------------------------------------
# Feature extraction (mirrors mega_trainer.load_muse_subconscious)
# ---------------------------------------------------------------------------

BAND_NAMES = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
CH_NAMES   = ["TP9", "AF7", "AF8", "TP10"]
BAND_COLS  = [f"{b}_{c}" for b in BAND_NAMES for c in CH_NAMES]  # 20 columns

WIN  = 4 * 256   # 4-second window at 256 Hz = 1 024 samples
STEP = 2 * 256   # 50 % overlap = 2-second hop


def _window_stats(col: np.ndarray) -> list:
    """Return [mean, std, median, IQR] for one column of band powers."""
    col = col[np.isfinite(col)]
    if len(col) < 10:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(col)),
        float(np.std(col)),
        float(np.median(col)),
        float(np.percentile(col, 75) - np.percentile(col, 25)),
    ]


def load_muse_subconscious_features():
    """Load Muse-Subconscious recordings and return X (n, 80), y, groups."""
    if not ZIP_PATH.exists():
        sys.exit(f"[ERROR] Dataset not found: {ZIP_PATH.resolve()}")

    X, y, groups = [], [], []
    n_subjects = 0

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        csv_files = [f for f in zf.namelist() if f.endswith(".csv") and "/Muse/" in f]
        print(f"  Found {len(csv_files)} recordings in zip")

        for csv_name in csv_files:
            try:
                subject_id = Path(csv_name).stem
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f)

                feat_cols = [c for c in BAND_COLS if c in df.columns]
                if len(feat_cols) < 10:
                    continue
                if "Mellow" not in df.columns or "Concentration" not in df.columns:
                    continue

                data   = df[feat_cols].values
                mellow = pd.to_numeric(df["Mellow"],        errors="coerce").values
                conc   = pd.to_numeric(df["Concentration"], errors="coerce").values

                for start in range(0, len(data) - WIN, STEP):
                    window = data[start : start + WIN]         # (1024, ≤20)
                    m_val  = float(np.nanmean(mellow[start : start + WIN]))
                    c_val  = float(np.nanmean(conc  [start : start + WIN]))

                    if np.isnan(m_val) and np.isnan(c_val):
                        continue
                    m_val = 0.0 if np.isnan(m_val) else m_val
                    c_val = 0.0 if np.isnan(c_val) else c_val

                    # 0 = positive, 1 = neutral, 2 = negative
                    label = 0 if m_val > 0.6 else (1 if c_val > 0.6 else 2)

                    feat = []
                    for col_idx in range(window.shape[1]):
                        feat.extend(_window_stats(window[:, col_idx]))

                    # Pad to exactly 80 features if fewer band columns were found
                    while len(feat) < 80:
                        feat.extend([0.0, 0.0, 0.0, 0.0])

                    X.append(feat[:80])
                    y.append(label)
                    groups.append(subject_id)

                n_subjects += 1
            except Exception as exc:
                print(f"    [warn] skipping {csv_name}: {exc}")
                continue

    if len(X) == 0:
        sys.exit("[ERROR] No samples extracted — check zip structure")

    X = np.array(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y, dtype=np.int32)

    print(f"  Loaded {len(X)} samples, {X.shape[1]} features, {n_subjects} subjects")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, np.array(groups)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Fit scaler + LightGBM, run LOSO cross-validation, return (model, scaler, cv_acc)."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        sys.exit("[ERROR] LightGBM not installed. Run: pip install lightgbm")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LGBMClassifier(
        n_estimators=400,
        max_depth=6,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    # Leave-one-subject-out cross-validation
    n_subjects = len(np.unique(groups))
    n_splits   = min(n_subjects, 10)  # cap at 10 folds to keep it fast
    gkf = GroupKFold(n_splits=n_splits)

    print(f"\n  Running {n_splits}-fold group cross-validation ({n_subjects} subjects) …")
    y_pred_cv = cross_val_predict(model, X_scaled, y, groups=groups, cv=gkf)
    cv_acc = float(accuracy_score(y, y_pred_cv))
    cv_f1  = float(f1_score(y, y_pred_cv, average="macro", zero_division=0))

    print(f"\n  CV Accuracy : {cv_acc:.4f} ({cv_acc*100:.2f}%)")
    print(f"  CV F1 macro : {cv_f1:.4f}")
    print("\n" + classification_report(y, y_pred_cv,
                                       target_names=["positive", "neutral", "negative"],
                                       zero_division=0))

    # Final fit on all data
    model.fit(X_scaled, y)

    return model, scaler, cv_acc, cv_f1


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_model(model, scaler, cv_acc: float, cv_f1: float):
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    BENCH_OUT.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model":         model,
        "scaler":        scaler,
        "feature_names": [
            f"{b}_{c}_{s}"
            for b in BAND_NAMES
            for c in CH_NAMES
            for s in ["mean", "std", "median", "iqr"]
        ],
        "n_classes":     3,
        "class_names":   ["positive", "neutral", "negative"],
        "n_features":    80,
        "model_type":    "lgbm-muse-live",
    }
    joblib.dump(payload, MODEL_OUT, compress=3)
    print(f"\n  Model saved → {MODEL_OUT}")

    bench = {
        "model_name":  "emotion_lgbm_muse_live",
        "dataset":     "Muse-Subconscious",
        "cv_method":   "leave-one-subject-out (GroupKFold)",
        "accuracy":    round(cv_acc, 4),
        "f1_macro":    round(cv_f1, 4),
        "n_classes":   3,
        "class_names": ["positive", "neutral", "negative"],
        "notes": (
            f"LOSO cross-subject accuracy {cv_acc*100:.2f}%. "
            "80-feature (4-stat × 5-band × 4-channel) LightGBM, no PCA — "
            "features reproducible from live Muse 2 EEG."
        ),
    }
    BENCH_OUT.write_text(json.dumps(bench, indent=2))
    print(f"  Benchmark  → {BENCH_OUT}")
    return cv_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Muse Live LightGBM Emotion Classifier — Training")
    print("=" * 60)

    print("\n[1/3] Loading Muse-Subconscious dataset …")
    X, y, groups = load_muse_subconscious_features()

    print("\n[2/3] Training LightGBM …")
    model, scaler, cv_acc, cv_f1 = train(X, y, groups)

    print("\n[3/3] Saving model …")
    save_model(model, scaler, cv_acc, cv_f1)

    print("\n" + "=" * 60)
    if cv_acc >= 0.60:
        print(f"  SUCCESS — CV accuracy {cv_acc*100:.2f}% ≥ 60% threshold.")
        print("  The model will be auto-loaded by EmotionClassifier.")
    else:
        print(f"  WARNING — CV accuracy {cv_acc*100:.2f}% < 60% threshold.")
        print("  EmotionClassifier will NOT use this model automatically.")
        print("  Consider collecting more labeled data or adjusting hyperparameters.")
    print("=" * 60)
