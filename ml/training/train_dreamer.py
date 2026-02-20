"""Train improved emotion classifier: DEAP (32 subjects) + EmoKey (Muse 2) + DREAMER (optional).

Key improvements over the previous DEAP-only model:
  - Full 32-subject DEAP with frontal 4-channel subset (AF3/AF4/T7/T8 ≈ Muse AF7/AF8/TP9/TP10)
  - Adaptive median-split labeling (corrects for DEAP's 0-9 skewed distribution)
  - EmoKey: real Muse S recordings with clear emotion labels (HAPPINESS/ANGER/FEAR/SADNESS)
  - DREAMER: 14-ch Emotiv EPOC (closest consumer device to Muse) — if DREAMER.mat available
  - Mastoid re-referencing applied before feature extraction (corrects Fpz contamination)
  - Multichannel features: DASM/RASM asymmetry + FAA + FMT (41 features total)
  - LightGBM classifier with class weights (fast, ~60 sec on 30K samples)
  - Per-class cap before SMOTE to prevent majority-class dominance

Usage (run from ml/ directory):
    python -m training.train_dreamer

DREAMER access:
    File must be at data/DREAMER.mat.
    Request access at https://zenodo.org/records/546113
    (Stamos.Katsigiannis@durham.ac.uk or Naeem.Ramzan@uws.ac.uk)
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

from processing.eeg_processor import extract_features_multichannel, rereference_to_mastoid

EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]
DEAP_FS   = 128.0   # DEAP pre-processed at 128 Hz
EPOCH_SEC = 4.0     # 4-sec windows (Welch PSD stability threshold from 2024 literature)
HOP_SEC   = 2.0     # 50% overlap
MAX_PER_CLASS = 4000  # cap before SMOTE to avoid majority-class dominance
MODEL_DIR     = Path("models/saved")
BENCHMARK_DIR = Path("benchmarks")

# DEAP channel indices for 4-channel Muse-equivalent subset (0-indexed):
#   ch7  = T7  → TP9  analog (left temporal)
#   ch1  = AF3 → AF7  analog (left frontal)
#   ch29 = AF4 → AF8  analog (right frontal)
#   ch12 = T8  → TP10 analog (right temporal)
_DEAP_CH = [7, 1, 29, 12]

# Emotiv EPOC channel order (DREAMER — 0-indexed):
# AF3(0), F7(1), F3(2), FC5(3), T7(4), P7(5), O1(6), O2(7), P8(8), T8(9), FC6(10), F4(11), F8(12), AF4(13)
_EPOC_CH = [4, 0, 13, 9]    # T7=TP9, AF3=AF7, AF4=AF8, T8=TP10


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── Label mapping ────────────────────────────────────────────────────────────

def _circumplex_label(valence: float, arousal: float, dominance: float,
                      v_med: float, a_med: float, d_med: float) -> int:
    """Map valence/arousal/dominance to 6-class emotion using dataset-specific medians.

    Uses median-split so the mapping adapts to any rating scale (DEAP 0-9,
    DREAMER 1-5) without assuming a fixed neutral point.

    'focused' is assigned to samples near the center of both axes.
    """
    # Samples close to both medians → focused/neutral
    if abs(valence - v_med) <= 0.3 * v_med and abs(arousal - a_med) <= 0.3 * a_med:
        return 5   # focused / neutral

    high_v = valence > v_med
    high_a = arousal > a_med
    high_d = dominance > d_med

    if high_v and high_a:
        return 0   # happy
    if high_v and not high_a:
        return 4   # relaxed
    if not high_v and high_a and high_d:
        return 2   # angry   (high D = approach motivation)
    if not high_v and high_a and not high_d:
        return 3   # fearful (low D = withdrawal)
    return 1       # sad     (low V, low A)


# ─── DEAP ─────────────────────────────────────────────────────────────────────

def _deap_medians(data_dir: str) -> tuple:
    """Compute global valence/arousal/dominance medians across all DEAP subjects."""
    vals, aros, doms = [], [], []
    for dat_path in sorted(Path(data_dir).glob("s*.dat")):
        try:
            with open(dat_path, "rb") as fh:
                subj = pickle.load(fh, encoding="latin1")
            vals.extend(subj["labels"][:, 0].tolist())
            aros.extend(subj["labels"][:, 1].tolist())
            doms.extend(subj["labels"][:, 2].tolist())
        except Exception:
            continue
    if not vals:
        return 4.5, 4.5, 4.5   # fallback for 0-9 scale
    return float(np.median(vals)), float(np.median(aros)), float(np.median(doms))


def load_deap_multichannel(data_dir: str = "data/deap") -> tuple:
    """Load all 32 DEAP .dat files, extract 4-ch Muse-equivalent multichannel features."""
    dat_files = sorted(Path(data_dir).glob("s*.dat"))
    if not dat_files:
        _log("  DEAP: no .dat files found, skipping")
        return np.array([]), np.array([])

    _log(f"  DEAP: {len(dat_files)} subjects — computing label medians...")
    v_med, a_med, d_med = _deap_medians(data_dir)
    _log(f"    medians: V={v_med:.2f}  A={a_med:.2f}  D={d_med:.2f}")

    win = int(EPOCH_SEC * DEAP_FS)
    hop = int(HOP_SEC * DEAP_FS)
    X_by_class: dict = {i: [] for i in range(6)}

    for dat_path in dat_files:
        try:
            with open(dat_path, "rb") as fh:
                subj = pickle.load(fh, encoding="latin1")
            eeg_all = subj["data"]       # (40, 40, 8064)
            labels  = subj["labels"]     # (40, 4)

            for trial_idx in range(eeg_all.shape[0]):
                v, a, d = labels[trial_idx][0], labels[trial_idx][1], labels[trial_idx][2]
                label = _circumplex_label(v, a, d, v_med, a_med, d_med)

                # Skip if this class is already at capacity
                if len(X_by_class[label]) >= MAX_PER_CLASS:
                    continue

                eeg_4ch = eeg_all[trial_idx][_DEAP_CH, :]  # (4, 8064)
                eeg_4ch = rereference_to_mastoid(eeg_4ch, left_mastoid_ch=0, right_mastoid_ch=3)

                for start in range(0, eeg_4ch.shape[1] - win, hop):
                    if len(X_by_class[label]) >= MAX_PER_CLASS:
                        break
                    seg   = eeg_4ch[:, start:start + win]
                    feats = extract_features_multichannel(seg, DEAP_FS)
                    X_by_class[label].append(list(feats.values()))

        except Exception as e:
            _log(f"    {dat_path.name}: {e}")

    X_parts, y_parts = [], []
    for cls, rows in X_by_class.items():
        if rows:
            X_parts.extend(rows)
            y_parts.extend([cls] * len(rows))

    if not X_parts:
        return np.array([]), np.array([])

    X_arr = np.array(X_parts, dtype=np.float32)
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
    counts = np.bincount(np.array(y_parts), minlength=6).tolist()
    _log(f"  DEAP done: {len(y_parts)} samples, {X_arr.shape[1]} features")
    _log(f"    label dist: {dict(zip(EMOTIONS, counts))}")
    return X_arr, np.array(y_parts, dtype=np.int32)


# ─── EmoKey ───────────────────────────────────────────────────────────────────

def load_emokey_multichannel() -> tuple:
    """Load EmoKey Muse S recordings: 4 emotions × 45 subjects, real Muse 2 hardware."""
    base = Path(
        "data/emokey/EmoKey Moments EEG Dataset (EKM-ED)"
        "/muse_wearable_data/preprocessed/clean-signals/0.0078125S"
    )
    if not base.exists():
        _log("  EmoKey: not found, skipping")
        return np.array([]), np.array([])

    emotion_map = {
        "HAPPINESS":         0,   # happy
        "SADNESS":           1,   # sad
        "ANGER":             2,   # angry
        "FEAR":              3,   # fearful
        "NEUTRAL_HAPPINESS": 4,   # relaxed
        "NEUTRAL_SADNESS":   5,   # focused
        "NEUTRAL_ANGER":     5,   # focused
        "NEUTRAL_FEAR":      5,   # focused
    }
    raw_cols = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
    SFREQ = 128.0
    win = int(EPOCH_SEC * SFREQ)
    hop = int(HOP_SEC * SFREQ)

    subjects = sorted([d for d in base.iterdir() if d.is_dir()])
    _log(f"  EmoKey: {len(subjects)} subjects...")
    X_by_class: dict = {i: [] for i in range(6)}

    for subj_dir in subjects:
        for csv_file in subj_dir.glob("*.csv"):
            emo_name = csv_file.stem
            if emo_name not in emotion_map:
                continue
            label = emotion_map[emo_name]
            if len(X_by_class[label]) >= MAX_PER_CLASS:
                continue
            try:
                df = pd.read_csv(csv_file)
                if not all(c in df.columns for c in raw_cols):
                    continue
                raw = df[raw_cols].values.T.astype(np.float32)   # (4, n_samples)
                raw -= raw.mean(axis=1, keepdims=True)
                raw = rereference_to_mastoid(raw, left_mastoid_ch=0, right_mastoid_ch=3)

                for start in range(0, raw.shape[1] - win, hop):
                    if len(X_by_class[label]) >= MAX_PER_CLASS:
                        break
                    seg = raw[:, start:start + win]
                    feats = extract_features_multichannel(seg, SFREQ)
                    X_by_class[label].append(list(feats.values()))
            except Exception:
                continue

    X_parts, y_parts = [], []
    for cls, rows in X_by_class.items():
        if rows:
            X_parts.extend(rows)
            y_parts.extend([cls] * len(rows))

    if not X_parts:
        return np.array([]), np.array([])

    X_arr = np.array(X_parts, dtype=np.float32)
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
    counts = np.bincount(np.array(y_parts), minlength=6).tolist()
    _log(f"  EmoKey done: {len(y_parts)} samples, {X_arr.shape[1]} features")
    _log(f"    label dist: {dict(zip(EMOTIONS, counts))}")
    return X_arr, np.array(y_parts, dtype=np.int32)


# ─── DREAMER (optional) ───────────────────────────────────────────────────────

def load_dreamer_multichannel() -> tuple:
    """Load DREAMER if data/DREAMER.mat is present.

    DREAMER: 23 subjects × 18 film clips, 14-ch Emotiv EPOC @ 128 Hz.
    Valence/arousal/dominance labels on 1-5 scale.

    To obtain: https://zenodo.org/records/546113
    """
    mat_path = Path("data/DREAMER.mat")
    if not mat_path.exists():
        _log("  DREAMER: data/DREAMER.mat not found — skipping")
        _log("    Request access at zenodo.org/records/546113")
        return np.array([]), np.array([])

    try:
        import h5py
    except ImportError:
        _log("  DREAMER: h5py not installed — pip install h5py")
        return np.array([]), np.array([])

    _log("  DREAMER: loading...")
    SFREQ = 128.0
    N_CH  = 14
    win   = int(EPOCH_SEC * SFREQ)
    hop   = int(HOP_SEC * SFREQ)
    X, y  = [], []

    try:
        with h5py.File(str(mat_path), "r") as f:
            data_ref = f["DREAMER"]["Data"]
            n_subjects = data_ref.shape[1]
            # Compute DREAMER medians first
            all_v, all_a, all_d = [], [], []
            for si in range(n_subjects):
                subj = f[data_ref[0, si]]
                for ti in range(subj["EEG"]["stimuli"].shape[1]):
                    all_v.append(float(np.array(f[subj["ScoreValence"][0, ti]]).flat[0]))
                    all_a.append(float(np.array(f[subj["ScoreArousal"][0, ti]]).flat[0]))
                    all_d.append(float(np.array(f[subj["ScoreDominance"][0, ti]]).flat[0]))
            v_med, a_med, d_med = float(np.median(all_v)), float(np.median(all_a)), float(np.median(all_d))
            _log(f"    DREAMER medians: V={v_med:.2f} A={a_med:.2f} D={d_med:.2f}")

            X_by_class: dict = {i: [] for i in range(6)}
            for si in range(n_subjects):
                subj      = f[data_ref[0, si]]
                stim_data = subj["EEG"]["stimuli"]
                val_data  = subj["ScoreValence"]
                aro_data  = subj["ScoreArousal"]
                dom_data  = subj["ScoreDominance"]

                for ti in range(stim_data.shape[1]):
                    trial_eeg = np.array(f[stim_data[0, ti]])
                    valence   = float(np.array(f[val_data[0, ti]]).flat[0])
                    arousal   = float(np.array(f[aro_data[0, ti]]).flat[0])
                    dominance = float(np.array(f[dom_data[0, ti]]).flat[0])
                    label     = _circumplex_label(valence, arousal, dominance, v_med, a_med, d_med)

                    if trial_eeg.shape[1] == N_CH:
                        eeg = trial_eeg.T
                    elif trial_eeg.shape[0] == N_CH:
                        eeg = trial_eeg
                    else:
                        eeg = trial_eeg[:N_CH, :] if trial_eeg.shape[0] < trial_eeg.shape[1] \
                              else trial_eeg[:, :N_CH].T

                    eeg_4ch = eeg[_EPOC_CH, :]
                    eeg_4ch = rereference_to_mastoid(eeg_4ch, left_mastoid_ch=0, right_mastoid_ch=3)

                    for start in range(0, eeg_4ch.shape[1] - win, hop):
                        if len(X_by_class[label]) >= MAX_PER_CLASS:
                            break
                        seg = eeg_4ch[:, start:start + win]
                        feats = extract_features_multichannel(seg, SFREQ)
                        X_by_class[label].append(list(feats.values()))

    except Exception as e:
        _log(f"  DREAMER failed: {e}")
        return np.array([]), np.array([])

    X_parts, y_parts = [], []
    for cls, rows in X_by_class.items():
        if rows:
            X_parts.extend(rows)
            y_parts.extend([cls] * len(rows))

    if not X_parts:
        return np.array([]), np.array([])

    X_arr  = np.array(X_parts, dtype=np.float32)
    X_arr  = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
    counts = np.bincount(np.array(y_parts), minlength=6).tolist()
    _log(f"  DREAMER done: {len(y_parts)} samples")
    _log(f"    label dist: {dict(zip(EMOTIONS, counts))}")
    return X_arr, np.array(y_parts, dtype=np.int32)


# ─── Training ─────────────────────────────────────────────────────────────────

def _smote_balance(X: np.ndarray, y: np.ndarray) -> tuple:
    """SMOTE-oversample minority classes to ≥50% of majority class size."""
    if not HAS_SMOTE:
        return X, y
    label_dist = np.bincount(y, minlength=len(EMOTIONS))
    max_count  = int(label_dist.max())
    target     = max(10, int(max_count * 0.5))
    strategy   = {cls: target for cls, cnt in enumerate(label_dist)
                  if 0 < cnt < target}
    if not strategy:
        return X, y
    k = max(1, min(5, min(label_dist[label_dist > 0]) - 1))
    try:
        smote  = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=42)
        X2, y2 = smote.fit_resample(X, y)
        _log(f"  After SMOTE: {len(y2)} samples (was {len(y)})")
        return X2, y2
    except Exception as e:
        _log(f"  SMOTE skipped: {e}")
        return X, y


def train_and_evaluate(X: np.ndarray, y: np.ndarray) -> dict:
    """Train LightGBM (primary) + RandomForest (fallback). Return best result dict."""
    _log(f"\nTraining: {len(y)} samples × {X.shape[1]} features")
    _log(f"Label dist: {dict(zip(EMOTIONS, np.bincount(y, minlength=6).tolist()))}")

    X, y   = _smote_balance(X, y)
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=42)
    sw = compute_sample_weight("balanced", ytr)

    best = {"accuracy": 0.0, "model": None, "scaler": scaler, "name": ""}

    # LightGBM — primary (fast, strong on tabular)
    if HAS_LGBM:
        _log("  Fitting LightGBM...")
        try:
            clf = lgb.LGBMClassifier(
                n_estimators=600, num_leaves=63, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1,
                class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1,
            )
            clf.fit(Xtr, ytr, sample_weight=sw)
            acc = accuracy_score(yte, clf.predict(Xte))
            f1  = f1_score(yte, clf.predict(Xte), average="weighted")
            _log(f"    LightGBM: acc={acc:.4f}  f1={f1:.4f}")
            if acc > best["accuracy"]:
                best.update({"accuracy": acc, "model": clf, "name": "LightGBM"})
        except Exception as e:
            _log(f"    LightGBM failed: {e}")

    # RandomForest — fallback (no sample_weight needed, class_weight handles it)
    _log("  Fitting RandomForest...")
    try:
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=42,
        )
        clf.fit(Xtr, ytr)
        acc = accuracy_score(yte, clf.predict(Xte))
        f1  = f1_score(yte, clf.predict(Xte), average="weighted")
        _log(f"    RandomForest: acc={acc:.4f}  f1={f1:.4f}")
        if acc > best["accuracy"]:
            best.update({"accuracy": acc, "model": clf, "name": "RandomForest"})
    except Exception as e:
        _log(f"    RandomForest failed: {e}")

    if best["model"] is None:
        return best

    preds  = best["model"].predict(Xte)
    report = classification_report(yte, preds, target_names=EMOTIONS, output_dict=True)
    _log(f"\nBest: {best['name']}  accuracy={best['accuracy']:.4f}")
    _log(classification_report(yte, preds, target_names=EMOTIONS))
    best["report"] = report
    return best


def save_model(best: dict, n_features: int):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    model_data = {
        "model":           best["model"],
        "scaler":          best["scaler"],
        "feature_names":   [f"mc_feat_{i}" for i in range(n_features)],
        "n_features":      n_features,
        "multichannel":    True,
        "accuracy":        float(best["accuracy"]),
        "classifier_type": best["name"],
        "trained_on":      ["DEAP", "EmoKey", "DREAMER (if available)"],
        "label_names":     EMOTIONS,
    }
    out = MODEL_DIR / "emotion_classifier_model.pkl"
    with open(out, "wb") as fh:
        pickle.dump(model_data, fh)
    _log(f"Saved → {out}")

    bench = {
        "accuracy":     float(best["accuracy"]),
        "classifier":   best["name"],
        "n_features":   n_features,
        "multichannel": True,
        "per_class":    {k: v for k, v in best.get("report", {}).items() if isinstance(v, dict)},
    }
    bench_out = BENCHMARK_DIR / "emotion_classifier_benchmark.json"
    with open(bench_out, "w") as fh:
        json.dump(bench, fh, indent=2)
    _log(f"Saved benchmark → {bench_out}")


def main():
    _log("=" * 60)
    _log("Multichannel Emotion Classifier — Training")
    _log("Datasets: DEAP (32 subj) + EmoKey (Muse 2) + DREAMER (optional)")
    _log("Features: 41 multichannel (DASM/RASM + FAA + FMT + mastoid reref)")
    _log("=" * 60)

    X_deap,  y_deap  = load_deap_multichannel()
    X_emo,   y_emo   = load_emokey_multichannel()
    X_dream, y_dream = load_dreamer_multichannel()

    parts_X = [x for x in [X_deap, X_emo, X_dream] if len(x) > 0]
    parts_y = [y for y, x in zip([y_deap, y_emo, y_dream], [X_deap, X_emo, X_dream]) if len(x) > 0]

    if not parts_X:
        _log("ERROR: No training data found. Check data/ directory.")
        sys.exit(1)

    min_feats = min(x.shape[1] for x in parts_X)
    X = np.vstack([x[:, :min_feats] for x in parts_X])
    y = np.concatenate(parts_y)

    _log(f"\nTotal: {len(y)} samples × {X.shape[1]} features")
    _log(f"Overall: {dict(zip(EMOTIONS, np.bincount(y, minlength=6).tolist()))}")

    best = train_and_evaluate(X, y)

    if best["model"] is None:
        _log("ERROR: All classifiers failed.")
        sys.exit(1)

    acc = best["accuracy"]
    save_model(best, n_features=X.shape[1])

    if acc >= 0.60:
        _log(f"\nSUCCESS: {acc:.4f} >= 60% threshold — model will be used in live inference.")
    else:
        _log(f"\nNOTE: {acc:.4f} < 60% threshold — model saved but live path uses heuristics.")
        _log("  Cross-subject 6-class from 4-ch EEG realistically lands at 55-72%.")
        _log("  Personalization via BaselineCalibrator adds +15-29%.")
    _log("Done.")


if __name__ == "__main__":
    main()
