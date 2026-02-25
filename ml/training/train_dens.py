"""
DENS Dataset — EEG Emotion Classifier Training
================================================

Dataset: Dataset on Emotion with Naturalistic Stimuli (DENS)
  - 40 subjects, 128-channel EGI HydroCel EEG + ECG + EMG (132 total channels)
  - 250 Hz, Cz reference
  - 11 video stimuli per subject (~60 sec each)
  - Continuous valence/arousal labels (1–9 SAM scale) per trial
  - BIDS format (.set/.fdt EEGLAB) — OpenNeuro doi:10.18112/openneuro.ds003751
  - Data path: ml/data/dens/

Channel mapping (EGI 128-ch HydroCel, EEGLAB coordinate system +X=nose, +Y=left):
  AF7  (left frontal pole)   → E32  (ch31)  Y=+6.6
  AF8  (right frontal pole)  → E1   (ch0)   Y=-6.6
  TP9  (left temporal)       → E48  (ch47)  Y=+6.8
  TP10 (right temporal)      → E119 (ch118) Y=-6.8
  Mastoid reference: average of TP9 + TP10 channels

Valence → 3-class label:
  >= 6.0  → positive  (0)
  <= 4.0  → negative  (2)
  4.0–6.0 → neutral   (1)
"""
from __future__ import annotations

import csv
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).resolve().parent
_ML_ROOT = _HERE.parent

for _p in [str(_ML_ROOT), str(_ML_ROOT / "processing")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    from processing.eeg_processor import (
        extract_features_multichannel,
        rereference_to_mastoid,
    )
except ImportError:
    from eeg_processor import (
        extract_features_multichannel,
        rereference_to_mastoid,
    )

try:
    import scipy.io as sio
    _SCI_OK = True
except ImportError:
    _SCI_OK = False
    print("ERROR: scipy not installed. Run: pip install scipy")
    sys.exit(1)

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
DENS_DATA_DIR = _ML_ROOT / "data" / "dens"
MODEL_DIR     = _ML_ROOT / "models" / "saved"
BENCHMARK_DIR = _ML_ROOT / "benchmarks"

DENS_FS          = 250          # Hz
EPOCH_SEC        = 4.0          # seconds per sliding window
HOP_SEC          = 2.0          # overlap hop
MAX_TRIAL_SEC    = 60.0         # max seconds to extract per trial
MAX_PER_CLASS    = 3000         # cap samples per class

EMOTIONS_3 = ["positive", "neutral", "negative"]

# Muse 2 equivalent channel indices in the 132-channel DENS layout
# (verified by 3D coordinate matching: E1=AF8, E32=AF7, E48=TP9, E119=TP10)
# Channel order for feature extraction: [TP9, AF7, AF8, TP10]
_CH_TP9  = 47    # E48  — left temporal-mastoid
_CH_AF7  = 31    # E32  — left frontal pole
_CH_AF8  = 0     # E1   — right frontal pole
_CH_TP10 = 118   # E119 — right temporal-mastoid
_MUSE_CH_IDX = [_CH_TP9, _CH_AF7, _CH_AF8, _CH_TP10]

VALENCE_POS_THRESH = 6.0
VALENCE_NEG_THRESH = 4.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FDT binary loader
# ---------------------------------------------------------------------------

def _load_fdt(fdt_path: Path, n_channels: int, n_samples: int) -> Optional[np.ndarray]:
    """Read EEGLAB .fdt binary (float32, channel-major C order)."""
    expected = n_channels * n_samples
    try:
        raw = np.fromfile(fdt_path, dtype=np.float32)
        if raw.size != expected:
            # Try Fortran order (older EEGLAB versions)
            if raw.size == expected:
                data = raw.reshape((n_channels, n_samples))
            else:
                log.debug("FDT size mismatch: got %d, expected %d", raw.size, expected)
                return None
        data = raw.reshape((n_channels, n_samples))
        return data     # (132, n_samples) in µV
    except Exception as exc:
        log.debug("FDT load failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Stimulus event & label extraction
# ---------------------------------------------------------------------------

def _get_trials(
    eeg_set,          # loaded scipy mat struct
    beh_path: Path,
) -> List[Dict]:
    """Return list of {onset_sample, duration_samples, valence, label_3class}."""
    # ---- read beh file for valence labels ----
    try:
        with open(beh_path, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            beh_rows = list(reader)
    except Exception as exc:
        log.debug("Cannot read beh file %s: %s", beh_path, exc)
        return []

    # ---- find stimulus onset events ----
    events = eeg_set.event if hasattr(eeg_set.event, "__len__") else []
    stm_latencies = [
        int(float(ev.latency))
        for ev in events
        if str(getattr(ev, "type", "")) == "stm"
    ]
    # Also find end-of-trial marker (vlnc = valence rating appears after each trial)
    vlnc_latencies = sorted([
        int(float(ev.latency))
        for ev in events
        if str(getattr(ev, "type", "")) == "vlnc"
    ])

    if not stm_latencies:
        log.debug("No 'stm' events found.")
        return []

    trials = []
    for i, onset in enumerate(sorted(stm_latencies)):
        # Find nearest vlnc after this stm → trial end
        end_sample = onset + int(MAX_TRIAL_SEC * DENS_FS)
        for vl in vlnc_latencies:
            if vl > onset:
                end_sample = min(vl, onset + int(MAX_TRIAL_SEC * DENS_FS))
                break

        duration = max(0, end_sample - onset)
        if duration < int(EPOCH_SEC * DENS_FS):
            continue

        # Match to beh row (same order as events)
        if i < len(beh_rows):
            try:
                valence = float(beh_rows[i]["valence"])
            except (KeyError, ValueError):
                continue
        else:
            continue

        # 3-class label from valence
        if valence >= VALENCE_POS_THRESH:
            label = 0   # positive
        elif valence <= VALENCE_NEG_THRESH:
            label = 2   # negative
        else:
            label = 1   # neutral

        trials.append({
            "onset":    onset,
            "duration": duration,
            "valence":  valence,
            "label":    label,
        })

    return trials


# ---------------------------------------------------------------------------
# Per-subject feature extraction
# ---------------------------------------------------------------------------

def _process_subject(sub_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for one subject."""
    eeg_dir = sub_dir / "eeg"
    beh_dir = sub_dir / "beh"

    # Find .set file
    set_files = list(eeg_dir.glob("*_eeg.set"))
    beh_files = list(beh_dir.glob("*_beh.tsv"))
    if not set_files or not beh_files:
        return np.array([]), np.array([])

    set_path = set_files[0]
    beh_path = beh_files[0]
    fdt_path = set_path.with_suffix(".fdt")
    if not fdt_path.exists():
        # Try alternate capitalisation
        alt = eeg_dir / (set_path.stem.replace("_eeg", "_eeg").replace(".set", ".fdt") + "")
        fdt_candidates = list(eeg_dir.glob("*.fdt"))
        if fdt_candidates:
            fdt_path = fdt_candidates[0]
        else:
            log.debug("No .fdt file for %s", sub_dir.name)
            return np.array([]), np.array([])

    # Load .set metadata
    try:
        mat = sio.loadmat(str(set_path), struct_as_record=False, squeeze_me=True)
        eeg_struct = mat["EEG"]
    except Exception as exc:
        log.debug("Cannot load .set for %s: %s", sub_dir.name, exc)
        return np.array([]), np.array([])

    n_ch   = int(eeg_struct.nbchan)
    n_samp = int(eeg_struct.pnts)

    # Sanity check channel indices
    max_idx = max(_MUSE_CH_IDX)
    if max_idx >= n_ch:
        log.debug("%s: only %d channels, need %d — skipping", sub_dir.name, n_ch, max_idx + 1)
        return np.array([]), np.array([])

    # Load binary EEG
    data = _load_fdt(fdt_path, n_ch, n_samp)
    if data is None:
        return np.array([]), np.array([])

    # Validate data range (should be ±500 µV for raw EEG)
    ch_sample = data[_CH_AF7, :1000]
    if np.abs(ch_sample).max() < 0.001 or np.abs(ch_sample).max() > 1e6:
        log.debug("%s: unexpected data range (%.3f µV) — check channel scaling",
                  sub_dir.name, np.abs(ch_sample).max())

    # Get trial onsets + labels
    trials = _get_trials(eeg_struct, beh_path)
    if not trials:
        log.debug("%s: no trials extracted", sub_dir.name)
        return np.array([]), np.array([])

    win  = int(EPOCH_SEC * DENS_FS)
    hop  = int(HOP_SEC   * DENS_FS)

    X_list: List[np.ndarray] = []
    y_list: List[int]         = []

    for trial in trials:
        onset    = trial["onset"]
        duration = trial["duration"]
        label    = trial["label"]
        end_samp = min(onset + duration, n_samp)

        # Select 4 Muse-equivalent channels, re-reference to linked mastoid (TP9+TP10)
        seg_4ch = data[_MUSE_CH_IDX, onset:end_samp]   # (4, T)
        try:
            seg_4ch = rereference_to_mastoid(
                seg_4ch, left_mastoid_ch=0, right_mastoid_ch=3
            )
        except Exception:
            pass

        # Sliding windows
        n_seg = seg_4ch.shape[1]
        start = 0
        while start + win <= n_seg:
            epoch = seg_4ch[:, start: start + win]
            try:
                feats = extract_features_multichannel(epoch, DENS_FS)
                X_list.append(np.array(list(feats.values()), dtype=np.float32))
                y_list.append(label)
            except Exception:
                pass
            start += hop

    if not X_list:
        return np.array([]), np.array([])

    return np.array(X_list), np.array(y_list, dtype=np.int32)


# ---------------------------------------------------------------------------
# Full dataset loader
# ---------------------------------------------------------------------------

def load_dens(data_dir: Path = DENS_DATA_DIR) -> Tuple[np.ndarray, np.ndarray]:
    sub_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ])
    if not sub_dirs:
        log.error("No sub-* directories found in %s", data_dir)
        return np.array([]), np.array([])

    log.info("Found %d subject directories", len(sub_dirs))

    X_by_class: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}

    for sub_dir in sub_dirs:
        X_sub, y_sub = _process_subject(sub_dir)
        if X_sub.size == 0:
            log.debug("  %s: no data", sub_dir.name)
            continue

        for label in range(3):
            mask = y_sub == label
            needed = MAX_PER_CLASS - len(X_by_class[label])
            if needed <= 0:
                continue
            rows = X_sub[mask][:needed]
            X_by_class[label].extend(rows.tolist())

        counts = {EMOTIONS_3[i]: int((y_sub == i).sum()) for i in range(3)}
        log.info("  %s: %d samples | %s", sub_dir.name, len(y_sub), counts)

    X_parts, y_parts = [], []
    for cls, rows in X_by_class.items():
        if rows:
            X_parts.append(np.array(rows, dtype=np.float32))
            y_parts.append(np.full(len(rows), cls, dtype=np.int32))

    if not X_parts:
        log.error("No features extracted — check data directory.")
        return np.array([]), np.array([])

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    counts_total = {EMOTIONS_3[i]: int((y == i).sum()) for i in range(3)}
    log.info("Total: %d samples | %s", len(y), counts_total)
    return X, y


# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------

def _smote(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not _SMOTE_OK:
        log.warning("imbalanced-learn not installed — skipping SMOTE.")
        return X, y
    counts = {c: int((y == c).sum()) for c in np.unique(y)}
    if min(counts.values()) < 2:
        return X, y
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
            acc = accuracy_score(y_te, clf.predict(X_te))
            rep = classification_report(y_te, clf.predict(X_te),
                                        target_names=EMOTIONS_3,
                                        output_dict=True, zero_division=0)
            cv  = cross_val_score(
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

    try:
        rf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        acc_rf = accuracy_score(y_te, rf.predict(X_te))
        rep_rf = classification_report(y_te, rf.predict(X_te),
                                       target_names=EMOTIONS_3,
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

    n_feat = best["model"].n_features_in_
    payload = {
        "model": best["model"], "scaler": best["scaler"],
        "feature_names": [f"feat_{i}" for i in range(n_feat)],
        "n_features": n_feat,
        "multichannel": True,
        "accuracy": best["accuracy"],
        "classifier_type": best["classifier_type"],
        "trained_on": ["DENS"],
        "label_names": EMOTIONS_3,
        "n_classes": 3,
        "channel_map": "E48(ch47)→TP9, E32(ch31)→AF7, E1(ch0)→AF8, E119(ch118)→TP10",
    }
    model_path = MODEL_DIR / "emotion_classifier_dens.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    log.info("Model saved → %s", model_path)

    benchmark = {
        "dataset": "DENS",
        "n_subjects": 40,
        "n_samples": n_samples,
        "classifier": best["classifier_type"],
        "test_accuracy": round(best["accuracy"], 4),
        "cv_accuracy_mean": round(best["cv_mean"], 4),
        "cv_accuracy_std": round(best["cv_std"], 4),
        "n_features": n_feat,
        "n_classes": 3,
        "classes": EMOTIONS_3,
        "channel_map": {
            "TP9": "E48(ch47)", "AF7": "E32(ch31)",
            "AF8": "E1(ch0)",   "TP10": "E119(ch118)",
        },
        "label_rule": {
            "positive": "valence >= 6.0",
            "neutral":  "4.0 < valence < 6.0",
            "negative": "valence <= 4.0",
        },
        "notes": (
            "128-ch EGI HydroCel, 4 Muse-equivalent channels selected by 3D coord matching. "
            "4-sec sliding windows (50% overlap). Mastoid re-reference applied. "
            "Valence threshold: pos>=6, neg<=4."
        ),
    }
    bm_path = BENCHMARK_DIR / "emotion_classifier_dens_benchmark.json"
    with open(bm_path, "w") as fh:
        json.dump(benchmark, fh, indent=2)
    log.info("Benchmark saved → %s", bm_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== DENS EEG emotion training ===")
    log.info("Data dir  : %s", DENS_DATA_DIR)
    log.info("Channels  : TP9=E48(ch47), AF7=E32(ch31), AF8=E1(ch0), TP10=E119(ch118)")
    log.info("Labels    : valence>=6→positive, <=4→negative, else neutral")
    log.info("Window    : %.1f sec, hop %.1f sec", EPOCH_SEC, HOP_SEC)

    X, y = load_dens(data_dir=DENS_DATA_DIR)
    if X.size == 0:
        log.error("No data loaded.")
        sys.exit(1)

    log.info("Dataset: %d samples, %d features, %d classes", X.shape[0], X.shape[1], len(np.unique(y)))

    n_unique_classes = len(np.unique(y))
    if n_unique_classes < 2:
        log.error("Only %d class found — adjust valence thresholds.", n_unique_classes)
        sys.exit(1)

    X_bal, y_bal = _smote(X, y)
    log.info("After SMOTE: %d samples", len(y_bal))

    best = train_and_evaluate(X_bal, y_bal)
    save_results(best, n_samples=len(y))
    log.info("Done. Best test: %.2f%%  CV: %.2f%%±%.2f%%",
             best["accuracy"] * 100, best["cv_mean"] * 100, best["cv_std"] * 100)


if __name__ == "__main__":
    main()
