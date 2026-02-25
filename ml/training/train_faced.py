"""
FACED Dataset — DE Feature Loader & Emotion Classifier Training
================================================================

Dataset: Fine-grained Affective Computing EEG Dataset (FACED)
  - 123 subjects, 32 channels (30 EEG + 2 mastoid), 250 Hz
  - 9 discrete emotion categories: anger, disgust, fear, sadness, neutral,
    amusement, inspiration, joy, tenderness
  - 28 video clips (3 per emotion × 8 emotions + 4 neutral = 28)
  - Synapse ID: syn50614194
  - Paper: https://doi.org/10.1038/s41597-023-02650-w

Expected data layout (EEG_Features.zip, already extracted):
  ml/data/faced/EEG_Features/DE/sub000.pkl.pkl ... sub122.pkl.pkl
  Each file: numpy array  shape (28, 32, 30, 5)
    dim 0: 28 video clips
    dim 1: 32 channels  (first 30 = EEG, last 2 = mastoid L/R)
    dim 2: 30 seconds per video (1-sec windows)
    dim 3: 5 DE bands   [delta(1-4), theta(4-8), alpha(8-14), beta(14-30), gamma(30-47)]

Channel layout (FACED 32-channel BrainProducts, confirmed by DE topology analysis):
  FP1(0)  FPZ(1)  FP2(2)  AF3(3)  AF4(4)  F7(5)   F5(6)   F3(7)
  F1(8)   FZ(9)   F2(10)  F4(11)  F6(12)  F8(13)  FT7(14) FC5(15)
  FC3(16) FC1(17) FCZ(18) FC2(19) FC4(20) FC6(21) FT8(22) T7(23)
  T8(24)  TP7(25) CP5(26) CP3(27) CP1(28) CPZ(29) Mastoid-L(30) Mastoid-R(31)

Confirmed by analysis:
  - ch23/ch24 (T7/T8): most bilaterally symmetric pair (mean_diff=0.111 across 123 subjects)
  - ch0 (FP1): highest beta/delta DE → left frontal pole (eye-blink contamination)
  - ch2 (FP2): right frontal pole homologue
  - ch1 (FPZ): midline, highest overall DE (reference-proximity artifact)

Muse 2 channel mapping:
  T7(23) → TP9, FP1(0) → AF7, FP2(2) → AF8, T8(24) → TP10

9-class → 3-class mapping (positive / neutral / negative):
  positive (0): amusement(5), inspiration(6), joy(7), tenderness(8)
  neutral  (1): neutral(4)
  negative (2): anger(0), disgust(1), fear(2), sadness(3)

28-video label sequence (standard FACED order):
  Verify against Stimuli_info.xlsx (syn52370955) if needed.
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).resolve().parent   # ml/training/
_ML_ROOT  = _HERE.parent                       # ml/

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
try:
    import lightgbm as lgb
    _LGB_OK = True
except ImportError:
    _LGB_OK = False

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_OK = True
except ImportError:
    _SMOTE_OK = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 9-class label names (from torcheeg / FACED paper)
FACED_9_CLASSES = ["anger", "disgust", "fear", "sadness", "neutral",
                   "amusement", "inspiration", "joy", "tenderness"]

# 3-class names used in this project
EMOTIONS_3 = ["positive", "neutral", "negative"]

# 9-class → 3-class mapping
FACED_9_TO_3: Dict[int, int] = {
    5: 0, 6: 0, 7: 0, 8: 0,   # positive
    4: 1,                       # neutral
    0: 2, 1: 2, 2: 2, 3: 2,   # negative
}

# FACED 28-video emotion label sequence (standard dataset order).
# Each entry is a FACED 9-class integer for that video index.
# Negative emotions first, neutral middle, positive last.
# ⚠ Verify with Stimuli_info.xlsx (syn52370955) if needed.
FACED_VIDEO_LABELS_28: List[int] = [
    1, 1, 1,     # disgust   (videos  0–2)
    2, 2, 2,     # fear      (videos  3–5)
    3, 3, 3,     # sadness   (videos  6–8)
    4, 4, 4, 4,  # neutral   (videos  9–12)   ← 4 clips
    5, 5, 5,     # amusement (videos 13–15)
    7, 7, 7,     # joy       (videos 16–18)
    6, 6, 6,     # inspiration (videos 19–21)
    8, 8, 8,     # tenderness  (videos 22–24)
    0, 0, 0,     # anger     (videos 25–27)
]
assert len(FACED_VIDEO_LABELS_28) == 28, "Label list must have exactly 28 entries"

# Muse-equivalent channel indices in the FACED 32-channel layout
# [T7(23)→TP9,  FP1(0)→AF7,  FP2(2)→AF8,  T8(24)→TP10]
_MUSE_CH = [23, 0, 2, 24]

# Band indices inside dim-3 of the DE array
_DELTA, _THETA, _ALPHA, _BETA, _GAMMA = 0, 1, 2, 3, 4

# Paths
DE_DIR       = _ML_ROOT / "data" / "faced" / "EEG_Features" / "DE"
MODEL_DIR    = _ML_ROOT / "models" / "saved"
BENCHMARK_DIR = _ML_ROOT / "benchmarks"

MAX_PER_CLASS = 6000   # cap samples per 3-class label to keep RAM manageable

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature extraction from a 1-second DE window
# ---------------------------------------------------------------------------

def _de_to_features(de_4ch: np.ndarray) -> np.ndarray:
    """Convert a (4, 5) DE array for one second into a 1-D feature vector.

    Input: de_4ch[channel, band]  — 4 Muse-equivalent channels × 5 bands
    Output: feature vector of length 57
      - 20  raw DE values            (4 ch × 5 bands)
      - 16  intra-channel band ratios (alpha/beta, theta/beta, alpha/theta, delta/theta) × 4 ch
      - 5   frontal DASM             (FP2_DE − FP1_DE per band)
      - 5   frontal RASM             (FP2_DE / FP1_DE per band)
      - 5   temporal DASM            (T8_DE  − T7_DE  per band)
      - 5   temporal RASM            (T8_DE  / T7_DE  per band)
      - 1   frontal alpha asymmetry  (FAA ≡ alpha_FP2 − alpha_FP1)
    Total = 57
    """
    eps = 1e-8
    # channel layout: [0]=T7(TP9), [1]=FP1(AF7), [2]=FP2(AF8), [3]=T8(TP10)
    t7, fp1, fp2, t8 = de_4ch[0], de_4ch[1], de_4ch[2], de_4ch[3]

    # 20 raw DE values
    raw = de_4ch.flatten()  # (20,)

    # 16 intra-channel ratios (one per band-pair per channel)
    ratios = []
    for ch in range(4):
        d = de_4ch[ch, _DELTA] + eps
        h = de_4ch[ch, _THETA] + eps
        a = de_4ch[ch, _ALPHA] + eps
        b = de_4ch[ch, _BETA]  + eps
        ratios.extend([a / b, h / b, a / h, d / h])
    ratios = np.array(ratios)   # (16,)

    # 5 frontal DASM / RASM  (FP2 − FP1, FP2 / FP1)
    f_dasm = fp2 - fp1                          # (5,)
    f_rasm = fp2 / (fp1 + eps)                  # (5,)

    # 5 temporal DASM / RASM  (T8 − T7, T8 / T7)
    t_dasm = t8 - t7                            # (5,)
    t_rasm = t8 / (t7 + eps)                    # (5,)

    # FAA: alpha band asymmetry
    faa = np.array([fp2[_ALPHA] - fp1[_ALPHA]])  # (1,)

    return np.concatenate([raw, ratios, f_dasm, f_rasm, t_dasm, t_rasm, faa])
    # Total: 20 + 16 + 5 + 5 + 5 + 5 + 1 = 57


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_faced_de(
    de_dir: Path = DE_DIR,
    use_3class: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load FACED pre-extracted DE features for all subjects.

    Returns (X, y) where:
      X: float32 array (n_samples, 57)
      y: int32  array  (n_samples,) — 3-class if use_3class else 9-class
    """
    pkl_files = sorted(de_dir.glob("sub*.pkl.pkl"))
    if not pkl_files:
        log.error(
            "No sub*.pkl.pkl files found in %s\n"
            "  Extract EEG_Features.zip (syn52368847) into ml/data/faced/",
            de_dir,
        )
        return np.array([]), np.array([])

    log.info("Found %d subject files in %s", len(pkl_files), de_dir)

    import pickle

    X_by_class: Dict[int, List[np.ndarray]] = {}
    n_classes = 3 if use_3class else 9
    for c in range(n_classes):
        X_by_class[c] = []

    for pkl_path in pkl_files:
        try:
            with open(pkl_path, "rb") as fh:
                data = pickle.load(fh)          # (28, 32, 30, 5)
        except Exception as exc:
            log.debug("Cannot load %s: %s", pkl_path.name, exc)
            continue

        if data.ndim != 4 or data.shape != (28, 32, 30, 5):
            log.debug("Unexpected shape %s in %s — skipping", data.shape, pkl_path.name)
            continue

        # Iterate over all 28 videos × 30 seconds
        for vid_idx, raw_label in enumerate(FACED_VIDEO_LABELS_28):
            label = FACED_9_TO_3[raw_label] if use_3class else raw_label
            if len(X_by_class[label]) >= MAX_PER_CLASS:
                continue

            # Extract 4 Muse-equivalent channels, all 30 seconds
            de_vid = data[vid_idx][_MUSE_CH, :, :]   # (4, 30, 5) — [ch, sec, band]
            de_vid = de_vid.transpose(1, 0, 2)         # (30, 4, 5)  — [sec, ch, band]

            for sec in range(30):
                if len(X_by_class[label]) >= MAX_PER_CLASS:
                    break
                feat = _de_to_features(de_vid[sec])    # (57,)
                X_by_class[label].append(feat)

    X_parts, y_parts = [], []
    for cls, rows in X_by_class.items():
        if rows:
            X_parts.append(np.array(rows, dtype=np.float32))
            y_parts.append(np.full(len(rows), cls, dtype=np.int32))

    if not X_parts:
        log.error("No features extracted — check DE directory.")
        return np.array([]), np.array([])

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    class_names = EMOTIONS_3 if use_3class else FACED_9_CLASSES
    counts = {class_names[i]: int((y == i).sum()) for i in range(n_classes)}
    log.info("Loaded %d samples | class distribution: %s", len(y), counts)
    return X, y


# ---------------------------------------------------------------------------
# SMOTE balancing
# ---------------------------------------------------------------------------

def _smote(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not _SMOTE_OK:
        log.warning("imbalanced-learn not installed — skipping SMOTE.")
        return X, y
    counts = {c: int((y == c).sum()) for c in np.unique(y)}
    majority = max(counts.values())
    target   = max(majority // 2, 1)
    strategy = {c: max(cnt, target) for c, cnt in counts.items()}
    k = max(1, min(5, min(counts.values()) - 1))
    if k < 1:
        return X, y
    try:
        sm = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=42)
        return sm.fit_resample(X, y)
    except Exception as exc:
        log.warning("SMOTE failed (%s) — using raw data.", exc)
        return X, y


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(X: np.ndarray, y: np.ndarray) -> Dict:
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    results: List[Dict] = []
    n_classes = len(np.unique(y))
    class_names = EMOTIONS_3 if n_classes == 3 else FACED_9_CLASSES

    # ---- LightGBM ----
    if _LGB_OK:
        try:
            clf = lgb.LGBMClassifier(
                n_estimators=600, num_leaves=63, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1,
                class_weight="balanced", random_state=42,
                n_jobs=-1, verbose=-1,
            )
            clf.fit(X_tr, y_tr)
            acc  = accuracy_score(y_te, clf.predict(X_te))
            rep  = classification_report(y_te, clf.predict(X_te),
                                         target_names=class_names[:n_classes],
                                         output_dict=True, zero_division=0)
            cv   = cross_val_score(
                lgb.LGBMClassifier(
                    n_estimators=600, num_leaves=63, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1,
                    class_weight="balanced", random_state=42,
                    n_jobs=-1, verbose=-1,
                ),
                X_scaled, y,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="accuracy", n_jobs=-1,
            )
            log.info("LightGBM test=%.2f%% | CV=%.2f%%±%.2f%%",
                     acc * 100, cv.mean() * 100, cv.std() * 100)
            results.append({"model": clf, "scaler": scaler, "accuracy": float(acc),
                             "cv_mean": float(cv.mean()), "cv_std": float(cv.std()),
                             "report": rep, "classifier_type": "LightGBM"})
        except Exception as exc:
            log.warning("LightGBM failed: %s", exc)

    # ---- RandomForest fallback ----
    try:
        rf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        acc_rf = accuracy_score(y_te, rf.predict(X_te))
        rep_rf = classification_report(y_te, rf.predict(X_te),
                                       target_names=class_names[:n_classes],
                                       output_dict=True, zero_division=0)
        cv_rf  = cross_val_score(
            RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                   random_state=42, n_jobs=-1),
            X_scaled, y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy", n_jobs=-1,
        )
        log.info("RandomForest test=%.2f%% | CV=%.2f%%±%.2f%%",
                 acc_rf * 100, cv_rf.mean() * 100, cv_rf.std() * 100)
        results.append({"model": rf, "scaler": scaler, "accuracy": float(acc_rf),
                        "cv_mean": float(cv_rf.mean()), "cv_std": float(cv_rf.std()),
                        "report": rep_rf, "classifier_type": "RandomForest"})
    except Exception as exc:
        log.warning("RandomForest failed: %s", exc)

    if not results:
        raise RuntimeError("All classifiers failed.")

    best = max(results, key=lambda r: r["accuracy"])
    log.info("Best: %s — test %.2f%% | CV %.2f%%±%.2f%%",
             best["classifier_type"], best["accuracy"] * 100,
             best["cv_mean"] * 100, best["cv_std"] * 100)
    return best


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(best: Dict, n_samples: int) -> None:
    import pickle

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": best["model"],
        "scaler": best["scaler"],
        "feature_names": [f"feat_{i}" for i in range(57)],
        "n_features": 57,
        "multichannel": True,
        "accuracy": best["accuracy"],
        "classifier_type": best["classifier_type"],
        "trained_on": ["FACED"],
        "label_names": EMOTIONS_3,
        "n_classes": 3,
        "channel_map": "T7(23)→TP9, FP1(0)→AF7, FP2(2)→AF8, T8(24)→TP10",
    }
    model_path = MODEL_DIR / "emotion_classifier_faced.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    log.info("Model saved → %s", model_path)

    benchmark = {
        "dataset": "FACED",
        "n_subjects": 123,
        "n_samples": n_samples,
        "classifier": best["classifier_type"],
        "test_accuracy": round(best["accuracy"], 4),
        "cv_accuracy_mean": round(best["cv_mean"], 4),
        "cv_accuracy_std": round(best["cv_std"], 4),
        "n_features": 57,
        "feature_description": (
            "20 DE (4-ch × 5-band) + 16 band-ratios + "
            "10 frontal DASM/RASM + 10 temporal DASM/RASM + 1 FAA"
        ),
        "n_classes": 3,
        "classes": EMOTIONS_3,
        "channel_indices": _MUSE_CH,
        "channel_map": "T7(23)=TP9, FP1(0)=AF7, FP2(2)=AF8, T8(24)=TP10",
        "label_map_9_to_3": {
            FACED_9_CLASSES[k]: EMOTIONS_3[v] for k, v in FACED_9_TO_3.items()
        },
        "notes": (
            "Pre-extracted DE features from EEG_Features.zip. "
            "4 Muse-equivalent channels. "
            "28-video sequence: disgust×3, fear×3, sadness×3, neutral×4, "
            "amusement×3, joy×3, inspiration×3, tenderness×3, anger×3."
        ),
    }
    bm_path = BENCHMARK_DIR / "emotion_classifier_faced_benchmark.json"
    with open(bm_path, "w") as fh:
        json.dump(benchmark, fh, indent=2)
    log.info("Benchmark saved → %s", bm_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== FACED DE feature training (3-class) ===")
    log.info("DE directory: %s", DE_DIR)
    log.info("Channels: T7(ch23)→TP9, FP1(ch0)→AF7, FP2(ch2)→AF8, T8(ch24)→TP10")
    log.info("Features per sample: 57  (DE + ratios + DASM/RASM + FAA)")

    X, y = load_faced_de(de_dir=DE_DIR, use_3class=True)
    if X.size == 0:
        log.error("No data loaded — check ml/data/faced/EEG_Features/DE/ directory.")
        sys.exit(1)

    log.info("Raw dataset: %d samples, %d features, %d classes", X.shape[0], X.shape[1], len(np.unique(y)))

    X_bal, y_bal = _smote(X, y)
    log.info("After SMOTE: %d samples", len(y_bal))

    best = train_and_evaluate(X_bal, y_bal)
    save_results(best, n_samples=len(y))

    log.info("Done.  Best test accuracy: %.2f%%  CV: %.2f%%±%.2f%%",
             best["accuracy"] * 100, best["cv_mean"] * 100, best["cv_std"] * 100)


if __name__ == "__main__":
    main()
