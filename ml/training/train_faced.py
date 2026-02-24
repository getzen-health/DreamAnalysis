"""
FACED Dataset — Multichannel EEG Emotion Classifier Training
=============================================================

Dataset: Fine-grained Affective Computing EEG Dataset (FACED)
  - 123 subjects, 30 EEG channels (10-20 system), 250 Hz
  - 9 discrete emotion categories (28 video clips per emotion)
  - Synapse ID: syn50614194
  - Paper: https://doi.org/10.1038/s41597-023-02650-w

Download instructions (one-time setup):
    pip install synapseclient
    synapse get -r syn50614194 --downloadLocation data/faced/
  OR manually download the 3.3 GB archive and extract to:
    ml/data/faced/   (sub_001.mat, sub_002.mat, … sub_123.mat)

FACED emotion labels (integer):
    0 = amusement   1 = inspiration   2 = joy          3 = tenderness
    4 = anger       5 = fear          6 = disgust       7 = sadness
    8 = neutral

Mapped to 6-class system used by this project:
    happy(0)   sad(1)   angry(2)   fearful(3)   relaxed(4)   focused(5)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or from ml/ directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # ml/training/
_ML_ROOT = _HERE.parent                           # ml/
_REPO_ROOT = _ML_ROOT.parent                      # repo root

for _p in [str(_ML_ROOT), str(_ML_ROOT / "processing")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Imports (optional heavy deps — handled gracefully)
# ---------------------------------------------------------------------------
try:
    from processing.eeg_processor import extract_features_multichannel, rereference_to_mastoid
except ImportError:
    try:
        from eeg_processor import extract_features_multichannel, rereference_to_mastoid
    except ImportError as e:
        raise ImportError(
            "Cannot import eeg_processor. Run from repo root or ml/. "
            f"Original error: {e}"
        )

try:
    import scipy.io as sio
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

try:
    import h5py
    _H5PY_OK = True
except ImportError:
    _H5PY_OK = False

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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]

FACED_FS = 250.0          # Hz — FACED native sample rate
EPOCH_SEC = 4.0           # seconds per sliding window
HOP_SEC = 2.0             # step between windows
MAX_PER_CLASS = 4000      # cap per emotion class to keep RAM reasonable

# FACED 30-channel 10-20 layout (0-indexed):
# Fp1(0) Fp2(1) F7(2) F3(3) Fz(4) F4(5) F8(6) FT7(7) FC3(8) FCz(9) FC4(10)
# FT8(11) T7(12) C3(13) Cz(14) C4(15) T8(16) TP7(17) CP3(18) CPz(19) CP4(20)
# TP8(21) P7(22) P3(23) Pz(24) P4(25) P8(26) O1(27) Oz(28) O2(29)
#
# Muse channel order: [TP9(≈T7), AF7(≈Fp1), AF8(≈Fp2), TP10(≈T8)]
# → select FACED indices [12, 0, 1, 16] to get [T7, Fp1, Fp2, T8]
_FACED_CH = [12, 0, 1, 16]  # T7(→TP9), Fp1(→AF7), Fp2(→AF8), T8(→TP10)

# Map FACED 9-class integers → 6-class project emotion integers
FACED_LABEL_MAP: Dict[int, int] = {
    0: 0,   # amusement  → happy
    1: 4,   # inspiration → relaxed
    2: 0,   # joy        → happy
    3: 4,   # tenderness → relaxed
    4: 2,   # anger      → angry
    5: 3,   # fear       → fearful
    6: 3,   # disgust    → fearful
    7: 1,   # sadness    → sad
    8: 5,   # neutral    → focused
}

MODEL_DIR = _ML_ROOT / "models" / "saved"
BENCHMARK_DIR = _ML_ROOT / "benchmarks"
FACED_DATA_DIR = _ML_ROOT / "data" / "faced"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _win_hop() -> Tuple[int, int]:
    """Return (window_samples, hop_samples) for FACED_FS."""
    return int(FACED_FS * EPOCH_SEC), int(FACED_FS * HOP_SEC)


def _smote_balance(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """SMOTE-oversample minority classes to ≥50 % of majority class size."""
    if not _SMOTE_OK:
        log.warning("imbalanced-learn not installed — skipping SMOTE.")
        return X, y

    counts = {cls: int((y == cls).sum()) for cls in np.unique(y)}
    majority = max(counts.values())
    target = max(majority // 2, 1)

    strategy = {cls: max(cnt, target) for cls, cnt in counts.items()}
    k = max(1, min(5, min(counts.values()) - 1))
    if k < 1:
        return X, y

    try:
        sm = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res
    except Exception as exc:
        log.warning("SMOTE failed (%s) — using raw data.", exc)
        return X, y

# ---------------------------------------------------------------------------
# FACED loader
# ---------------------------------------------------------------------------

def _load_mat_v7(path: Path) -> Optional[np.ndarray]:
    """Load a MATLAB v7.3 file via h5py (returns raw array or None)."""
    if not _H5PY_OK:
        return None
    try:
        with h5py.File(path, "r") as f:
            # FACED files typically store the EEG matrix under key 'de_lds'
            # or as the first dataset found.
            for key in ("de_lds", "eeg", "EEG", "data"):
                if key in f:
                    return np.array(f[key])
            # Fallback: first dataset
            for key in f.keys():
                obj = f[key]
                if hasattr(obj, "shape") and len(obj.shape) >= 2:
                    return np.array(obj)
    except Exception as exc:
        log.debug("h5py load failed for %s: %s", path, exc)
    return None


def _load_mat_legacy(path: Path) -> Optional[dict]:
    """Load a MATLAB v5/v6 .mat file via scipy."""
    if not _SCIPY_OK:
        return None
    try:
        return sio.loadmat(str(path))
    except Exception as exc:
        log.debug("scipy.io.loadmat failed for %s: %s", path, exc)
    return None


def load_faced_multichannel(
    data_dir: Path = FACED_DATA_DIR,
    skip_existing: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load FACED .mat files, extract multichannel features, return (X, y).

    Expects one of these layouts in each sub_XXX.mat:
      Option A — raw EEG: array shaped (n_trials, n_channels, n_samples)
      Option B — preprocessed segment: (n_channels, n_samples) per trial

    If the .mat appears to contain pre-extracted feature matrices rather than
    raw EEG, the file is skipped with a warning.

    Parameters
    ----------
    data_dir : Path
        Directory containing sub_*.mat files (or sub_*.npy).
    skip_existing : bool
        If True and a checkpoint .npy exists, load that instead of re-processing.
    """
    if not data_dir.exists():
        log.warning(
            "FACED data directory not found: %s\n"
            "  Download with:\n"
            "    pip install synapseclient\n"
            "    synapse get -r syn50614194 --downloadLocation %s",
            data_dir, data_dir,
        )
        return np.array([]), np.array([])

    # Optional checkpoint (saves ~15 min on re-runs)
    ckpt_X = data_dir / "_faced_X.npy"
    ckpt_y = data_dir / "_faced_y.npy"
    if skip_existing and ckpt_X.exists() and ckpt_y.exists():
        log.info("Loading FACED features from checkpoint …")
        return np.load(ckpt_X), np.load(ckpt_y)

    mat_files = sorted(data_dir.glob("sub_*.mat"))
    npy_files = sorted(data_dir.glob("sub_*.npy"))

    if not mat_files and not npy_files:
        log.warning("No sub_*.mat or sub_*.npy files found in %s", data_dir)
        return np.array([]), np.array([])

    win, hop = _win_hop()

    # Per-class accumulator (capped at MAX_PER_CLASS)
    X_by_class: Dict[int, List[List[float]]] = {i: [] for i in range(6)}

    # ------------------------------------------------------------------ .npy
    for npy_path in npy_files:
        try:
            arr = np.load(npy_path, allow_pickle=True)
            # Expected: dict with keys "eeg" (n_trials, 30, n_samples) and "labels" (n_trials,)
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                d = arr.item()
                eeg_all = d.get("eeg", None)
                labels_raw = d.get("labels", None)
            else:
                log.debug("Unexpected .npy layout in %s — skipping.", npy_path)
                continue

            if eeg_all is None or labels_raw is None:
                continue

            for trial_idx in range(eeg_all.shape[0]):
                raw_label = int(labels_raw[trial_idx])
                label = FACED_LABEL_MAP.get(raw_label, None)
                if label is None:
                    continue
                if len(X_by_class[label]) >= MAX_PER_CLASS:
                    continue

                eeg_trial = eeg_all[trial_idx]          # (30, n_samples)
                eeg_4ch = eeg_trial[_FACED_CH, :]       # (4, n_samples)
                eeg_4ch = rereference_to_mastoid(eeg_4ch,
                                                 left_mastoid_ch=0,
                                                 right_mastoid_ch=3)

                for start in range(0, eeg_4ch.shape[1] - win, hop):
                    if len(X_by_class[label]) >= MAX_PER_CLASS:
                        break
                    seg = eeg_4ch[:, start: start + win]
                    try:
                        feats = extract_features_multichannel(seg, FACED_FS)
                        X_by_class[label].append(list(feats.values()))
                    except Exception:
                        pass

        except Exception as exc:
            log.debug("Failed to load %s: %s", npy_path, exc)

    # ------------------------------------------------------------------ .mat
    for mat_path in mat_files:
        subject_id = mat_path.stem  # e.g. sub_001

        # Try v7.3 first (FACED typically uses this format)
        raw = _load_mat_v7(mat_path)
        mat_dict: Optional[dict] = None

        if raw is None:
            mat_dict = _load_mat_legacy(mat_path)
            if mat_dict is None:
                log.debug("Cannot open %s — skipping.", mat_path)
                continue

        # ---- parse label array ----
        labels_raw: Optional[np.ndarray] = None

        if mat_dict is not None:
            # scipy dict keys
            for lkey in ("labels", "label", "emotion_label", "y"):
                if lkey in mat_dict:
                    labels_raw = np.array(mat_dict[lkey]).flatten().astype(int)
                    break
            eeg_key = None
            for ek in ("EEG", "eeg", "data", "signal"):
                if ek in mat_dict:
                    eeg_key = ek
                    break
            if eeg_key is None:
                log.debug("%s: no recognised EEG key — skipping.", subject_id)
                continue
            eeg_block = np.array(mat_dict[eeg_key])
        else:
            # h5py path: raw is an ndarray; labels live in a sibling key
            # We re-open to grab labels
            labels_raw = None
            if _H5PY_OK:
                try:
                    with h5py.File(mat_path, "r") as f:
                        for lkey in ("labels", "label", "emotion_label", "y"):
                            if lkey in f:
                                labels_raw = np.array(f[lkey]).flatten().astype(int)
                                break
                except Exception:
                    pass
            eeg_block = raw  # type: ignore[assignment]

        if labels_raw is None or eeg_block is None:
            log.debug("%s: missing labels or EEG block — skipping.", subject_id)
            continue

        # ---- normalise EEG shape ----
        # Expected: (n_trials, n_channels, n_samples)
        # Some files may be transposed or 2-D
        if eeg_block.ndim == 2:
            # (n_channels, n_samples) — single-trial file
            eeg_block = eeg_block[np.newaxis, ...]  # → (1, ch, samples)

        if eeg_block.ndim == 3:
            n_trials, n_ch, n_samp = eeg_block.shape
            # If the shape looks transposed (samples × channels × trials), fix it
            if n_ch > n_samp:
                eeg_block = eeg_block.transpose(0, 2, 1)
                n_trials, n_ch, n_samp = eeg_block.shape

            if n_ch < max(_FACED_CH) + 1:
                log.debug(
                    "%s: only %d channels (need ≥17) — not raw EEG, skipping.",
                    subject_id, n_ch,
                )
                continue

            if len(labels_raw) != n_trials:
                log.debug(
                    "%s: label count %d ≠ trial count %d — skipping.",
                    subject_id, len(labels_raw), n_trials,
                )
                continue

            for trial_idx in range(n_trials):
                raw_label = int(labels_raw[trial_idx])
                label = FACED_LABEL_MAP.get(raw_label, None)
                if label is None:
                    continue
                if len(X_by_class[label]) >= MAX_PER_CLASS:
                    continue

                eeg_trial = eeg_block[trial_idx]        # (n_ch, n_samp)
                eeg_4ch = eeg_trial[_FACED_CH, :]       # (4, n_samp)
                eeg_4ch = rereference_to_mastoid(
                    eeg_4ch, left_mastoid_ch=0, right_mastoid_ch=3
                )

                for start in range(0, eeg_4ch.shape[1] - win, hop):
                    if len(X_by_class[label]) >= MAX_PER_CLASS:
                        break
                    seg = eeg_4ch[:, start: start + win]
                    try:
                        feats = extract_features_multichannel(seg, FACED_FS)
                        X_by_class[label].append(list(feats.values()))
                    except Exception:
                        pass
        else:
            log.debug(
                "%s: unexpected EEG block shape %s — skipping.",
                subject_id, eeg_block.shape,
            )

    # ---- assemble final arrays ----
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for cls, rows in X_by_class.items():
        if rows:
            X_parts.append(np.array(rows, dtype=np.float32))
            y_parts.append(np.full(len(rows), cls, dtype=np.int32))

    if not X_parts:
        log.warning("FACED: no features extracted — check data directory.")
        return np.array([]), np.array([])

    X_out = np.vstack(X_parts)
    y_out = np.concatenate(y_parts)

    counts = {EMOTIONS[i]: int((y_out == i).sum()) for i in range(6)}
    log.info(
        "FACED loaded: %d samples | class dist: %s",
        len(y_out),
        {k: v for k, v in counts.items() if v > 0},
    )

    # Save checkpoint for fast re-runs
    try:
        np.save(ckpt_X, X_out)
        np.save(ckpt_y, y_out)
    except Exception:
        pass

    return X_out, y_out

# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
) -> Dict:
    """Train LightGBM (+ RandomForest fallback) and return best result dict."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    results: List[Dict] = []

    # ---- LightGBM ----
    if _LGB_OK:
        try:
            clf_lgb = lgb.LGBMClassifier(
                n_estimators=600,
                num_leaves=63,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            clf_lgb.fit(X_train, y_train)
            y_pred = clf_lgb.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(
                y_test, y_pred,
                target_names=EMOTIONS,
                output_dict=True,
                zero_division=0,
            )
            log.info("LightGBM test accuracy: %.2f%%", acc * 100)

            # 5-fold CV
            cv_scores = cross_val_score(
                lgb.LGBMClassifier(
                    n_estimators=600, num_leaves=63, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1,
                    class_weight="balanced", random_state=42,
                    n_jobs=-1, verbose=-1,
                ),
                X_scaled, y,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="accuracy",
                n_jobs=-1,
            )
            log.info(
                "LightGBM CV: %.2f%% ± %.2f%%",
                cv_scores.mean() * 100,
                cv_scores.std() * 100,
            )

            results.append({
                "model": clf_lgb,
                "scaler": scaler,
                "accuracy": float(acc),
                "cv_mean": float(cv_scores.mean()),
                "cv_std": float(cv_scores.std()),
                "report": report,
                "classifier_type": "LightGBM",
            })
        except Exception as exc:
            log.warning("LightGBM training failed: %s", exc)

    # ---- RandomForest fallback ----
    try:
        clf_rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf_rf.fit(X_train, y_train)
        y_pred_rf = clf_rf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        report_rf = classification_report(
            y_test, y_pred_rf,
            target_names=EMOTIONS,
            output_dict=True,
            zero_division=0,
        )
        log.info("RandomForest test accuracy: %.2f%%", acc_rf * 100)

        cv_rf = cross_val_score(
            RandomForestClassifier(
                n_estimators=400, class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
            X_scaled, y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy",
            n_jobs=-1,
        )
        results.append({
            "model": clf_rf,
            "scaler": scaler,
            "accuracy": float(acc_rf),
            "cv_mean": float(cv_rf.mean()),
            "cv_std": float(cv_rf.std()),
            "report": report_rf,
            "classifier_type": "RandomForest",
        })
    except Exception as exc:
        log.warning("RandomForest training failed: %s", exc)

    if not results:
        raise RuntimeError("All classifiers failed — cannot save model.")

    best = max(results, key=lambda r: r["accuracy"])
    log.info(
        "Best classifier: %s — test %.2f%% | CV %.2f%%±%.2f%%",
        best["classifier_type"],
        best["accuracy"] * 100,
        best["cv_mean"] * 100,
        best["cv_std"] * 100,
    )
    return best


# ---------------------------------------------------------------------------
# Save model + benchmark
# ---------------------------------------------------------------------------

def save_model(best: Dict, n_features: int, trained_on: List[str]) -> None:
    """Persist model pickle and benchmark JSON."""
    import pickle

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = [f"feat_{i}" for i in range(n_features)]

    payload = {
        "model": best["model"],
        "scaler": best["scaler"],
        "feature_names": feature_names,
        "n_features": n_features,
        "multichannel": True,
        "accuracy": best["accuracy"],
        "classifier_type": best["classifier_type"],
        "trained_on": trained_on,
        "label_names": EMOTIONS,
    }

    model_path = MODEL_DIR / "emotion_classifier_model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    log.info("Model saved → %s", model_path)

    benchmark = {
        "dataset": "FACED",
        "classifier": best["classifier_type"],
        "test_accuracy": round(best["accuracy"], 4),
        "cv_accuracy_mean": round(best["cv_mean"], 4),
        "cv_accuracy_std": round(best["cv_std"], 4),
        "n_features": n_features,
        "n_samples_trained": int(best.get("n_samples", 0)),
        "classes": EMOTIONS,
        "trained_on": trained_on,
        "label_map": {
            "FACED_amusement(0)": "happy",
            "FACED_inspiration(1)": "relaxed",
            "FACED_joy(2)": "happy",
            "FACED_tenderness(3)": "relaxed",
            "FACED_anger(4)": "angry",
            "FACED_fear(5)": "fearful",
            "FACED_disgust(6)": "fearful",
            "FACED_sadness(7)": "sad",
            "FACED_neutral(8)": "focused",
        },
        "notes": (
            "Multichannel features from FACED channels [T7, Fp1, Fp2, T8] "
            "remapped to Muse [TP9, AF7, AF8, TP10]. "
            f"Sliding windows: {EPOCH_SEC}s @ {HOP_SEC}s hop, {FACED_FS}Hz."
        ),
    }

    bm_path = BENCHMARK_DIR / "emotion_classifier_benchmark.json"
    with open(bm_path, "w") as fh:
        json.dump(benchmark, fh, indent=2)
    log.info("Benchmark saved → %s", bm_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(skip_existing_cache: bool = True) -> None:
    log.info("=== FACED multichannel emotion training ===")
    log.info("Data dir : %s", FACED_DATA_DIR)
    log.info("Window   : %.1f s | Hop: %.1f s | Max/class: %d", EPOCH_SEC, HOP_SEC, MAX_PER_CLASS)

    X_faced, y_faced = load_faced_multichannel(
        data_dir=FACED_DATA_DIR,
        skip_existing=skip_existing_cache,
    )

    if X_faced.size == 0:
        log.error(
            "No FACED data loaded.\n\n"
            "  Download the dataset:\n"
            "    pip install synapseclient\n"
            "    mkdir -p %s\n"
            "    synapse get -r syn50614194 --downloadLocation %s\n\n"
            "  Then re-run this script.",
            FACED_DATA_DIR, FACED_DATA_DIR,
        )
        sys.exit(1)

    # SMOTE balancing
    X_bal, y_bal = _smote_balance(X_faced, y_faced)
    log.info("Post-SMOTE: %d samples", len(y_bal))

    best = train_and_evaluate(X_bal, y_bal)
    best["n_samples"] = len(y_bal)

    if best["accuracy"] < 0.35:
        log.warning(
            "Test accuracy %.1f%% is very low — consider checking data quality "
            "or increasing MAX_PER_CLASS.",
            best["accuracy"] * 100,
        )

    save_model(best, n_features=X_bal.shape[1], trained_on=["FACED"])

    log.info("Done.  Model test accuracy: %.2f%%", best["accuracy"] * 100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train emotion classifier on FACED dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=FACED_DATA_DIR,
        help="Path to directory containing sub_*.mat files.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore existing feature cache and re-extract from raw .mat files.",
    )
    args = parser.parse_args()

    # Allow overriding data dir at CLI
    FACED_DATA_DIR = args.data_dir  # type: ignore[misc]
    main(skip_existing_cache=not args.no_cache)
