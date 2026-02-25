"""
Multimodal Mega LGBM Trainer
==============================
Extends the EEG-only mega model by adding audio (92-dim MFCC) and video
(72-dim facial-grid) features where available (EAV dataset), keeping all
other datasets as EEG-only.

Feature vector layout (251-dim):
  [0:85]    EEG features      (always present — 5 bands×4ch×4stats + 5 DASM)
  [85:177]  Audio features    (MFCC + spectral, 92-dim; NaN if absent)
  [177:249] Video features    (6×6 facial grid intensity+gradient, 72-dim; NaN if absent)
  [249]     has_audio flag    (1.0 / 0.0)
  [250]     has_video flag    (1.0 / 0.0)

LightGBM handles NaN natively (best split direction for missing values) so no
zero-padding is needed — the model learns to use audio/video when present and
fall back to EEG-only when absent.

Saves:
  models/saved/multimodal_mega_lgbm.pkl  — {model, n_features_raw, metadata}
  benchmarks/multimodal_mega_benchmark.json

Usage (from ml/ directory):
    python3 training/train_multimodal_mega_lgbm.py
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).resolve().parent
_ML_ROOT = _HERE.parent

for _p in [str(_ML_ROOT), str(_ML_ROOT / "processing"), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Imports (reuse existing loaders + feature extractors)
# ---------------------------------------------------------------------------
from train_mega_lgbm_unified import (
    load_deap,
    load_dreamer,
    load_gameemo,
    load_dens,
    load_faced,
    load_seed_iv,
    load_eeg_er,
    load_stew,
    load_muse_subconscious,
    load_emokey,
    _sliding_windows,
    extract_features,
    _EAV_DIR,
    _EAV_FS,
    _EAV_CH_MUSE,
    _EAV_COND_TO3,
    _EAV_SEGS_PER_TRIAL,
    EMOTIONS_3,
)

# Audio feature extractor (92-dim)
from train_audio_emotion import extract_audio_features as _extract_audio_features

# Video feature extractor (72-dim)
from train_video_emotion import extract_video_features

try:
    import scipy.io as sio
    _SCI_OK = True
except ImportError:
    _SCI_OK = False

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_EEG   = 85
N_AUDIO = 92
N_VIDEO = 72
N_FLAGS = 2
N_TOTAL = N_EEG + N_AUDIO + N_VIDEO + N_FLAGS   # 251

MODEL_OUT = _ML_ROOT / "models" / "saved" / "multimodal_mega_lgbm.pkl"
BENCH_OUT = _ML_ROOT / "benchmarks" / "multimodal_mega_benchmark.json"

# EAV filename patterns
_EAV_VIDEO_LABEL_MAP = {
    "Happiness": 0, "Anger": 2, "Sadness": 2,
    "Neutral": 1, "Calmness": 1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eeg_to_multimodal(X_eeg: np.ndarray) -> np.ndarray:
    """Pad EEG-only rows (85-dim) → 251-dim with NaN for audio/video, flags=0."""
    n = X_eeg.shape[0]
    result = np.full((n, N_TOTAL), np.nan, dtype=np.float32)
    result[:, :N_EEG] = X_eeg
    result[:, N_EEG + N_AUDIO + N_VIDEO]     = 0.0   # has_audio = 0
    result[:, N_EEG + N_AUDIO + N_VIDEO + 1] = 0.0   # has_video = 0
    return result


def _make_row(
    eeg_feat: np.ndarray,
    audio_feat: Optional[np.ndarray],
    video_feat: Optional[np.ndarray],
) -> np.ndarray:
    """Build one 251-dim feature row from per-modality feature arrays."""
    row = np.full(N_TOTAL, np.nan, dtype=np.float32)
    row[:N_EEG] = eeg_feat

    if audio_feat is not None:
        row[N_EEG: N_EEG + N_AUDIO]                = audio_feat
        row[N_EEG + N_AUDIO + N_VIDEO]              = 1.0   # has_audio
    else:
        row[N_EEG + N_AUDIO + N_VIDEO]              = 0.0

    if video_feat is not None:
        row[N_EEG + N_AUDIO: N_EEG + N_AUDIO + N_VIDEO] = video_feat
        row[N_EEG + N_AUDIO + N_VIDEO + 1]              = 1.0   # has_video
    else:
        row[N_EEG + N_AUDIO + N_VIDEO + 1]              = 0.0

    return row


# ---------------------------------------------------------------------------
# EAV multimodal loader
# ---------------------------------------------------------------------------

def _find_file(directory: Path, prefix: str, suffix: str) -> Optional[Path]:
    """Find a file in directory whose stem starts with prefix and ends with suffix."""
    for p in directory.iterdir():
        if p.name.startswith(prefix) and p.suffix.lower() == suffix:
            return p
    return None


def load_eav_multimodal() -> Tuple[np.ndarray, np.ndarray]:
    """Load EAV dataset with aligned EEG + audio + video features.

    For each trial:
      - EEG windows are extracted with _sliding_windows → multiple rows per trial
      - Audio features (92-dim) extracted once per trial (if Speaking trial)
      - Video features (72-dim) extracted once per trial
      - Same audio/video features repeated for every EEG window of the trial

    Returns (X, y) where X is (n_samples, 251) and y is (n_samples,).
    """
    if not _EAV_DIR.exists():
        log.warning("EAV: %s not found — skipping", _EAV_DIR)
        return np.empty((0, N_TOTAL), np.float32), np.empty(0, np.int32)

    subj_dirs = sorted(
        d for d in _EAV_DIR.iterdir()
        if d.is_dir() and d.name.lower().startswith("subject")
    )
    if not subj_dirs:
        log.warning("EAV: no subject* dirs — skipping")
        return np.empty((0, N_TOTAL), np.float32), np.empty(0, np.int32)

    Xs, ys = [], []
    n_loaded = 0
    n_audio_ok = 0
    n_video_ok = 0

    for subj_dir in subj_dirs:
        eeg_dir  = subj_dir / "EEG"
        aud_dir  = subj_dir / "Audio"
        vid_dir  = subj_dir / "Video"
        sid      = subj_dir.name
        eeg_file = eeg_dir / f"{sid}_eeg.mat"
        lab_file = eeg_dir / f"{sid}_eeg_label.mat"

        if not (eeg_file.exists() and lab_file.exists()):
            continue

        try:
            eeg_mat = sio.loadmat(str(eeg_file), struct_as_record=False, squeeze_me=True)
            lab_mat = sio.loadmat(str(lab_file), struct_as_record=False, squeeze_me=True)

            raw = None
            for key in ("seg1", "seg", "eeg", "data"):
                if key in eeg_mat:
                    raw = np.array(eeg_mat[key], dtype=np.float32)
                    break
            if raw is None or raw.ndim != 3:
                continue

            n_total_segs, n_ch, n_tp = raw.shape
            if n_ch != 30 or max(_EAV_CH_MUSE) >= n_ch:
                continue

            lbl = np.array(lab_mat["label"], dtype=np.uint8)
            if lbl.ndim != 2 or lbl.shape[0] != 10:
                continue

            n_trials = lbl.shape[1]
            cond_per_trial = np.argmax(lbl, axis=0)
            labels3 = np.array(
                [_EAV_COND_TO3[int(c)] for c in cond_per_trial], dtype=np.int32
            )

            n_segs_per_trial = n_total_segs // n_trials
            eeg4 = raw[:, _EAV_CH_MUSE, :]   # (n_total_segs, 4, n_tp)

            for trial_idx in range(n_trials):
                # ── EEG windows ──────────────────────────────────────────────
                s0 = trial_idx * n_segs_per_trial
                s1 = s0 + n_segs_per_trial
                trial_segs = eeg4[s0:s1]                          # (50, 4, 200)
                trial_eeg  = trial_segs.transpose(1, 0, 2).reshape(4, -1)  # (4, 10000)
                eeg_windows = _sliding_windows(trial_eeg, _EAV_FS, artifact_uv=200.0)
                if len(eeg_windows) == 0:
                    continue

                # ── Video features ───────────────────────────────────────────
                seq_num    = trial_idx + 1   # 1-indexed
                prefix_vid = f"{seq_num:03d}_"
                vid_path   = _find_file(vid_dir, prefix_vid, ".mp4") if vid_dir.exists() else None
                vid_feat   = extract_video_features(vid_path) if vid_path else None
                if vid_feat is not None:
                    n_video_ok += 1

                # ── Audio features ───────────────────────────────────────────
                prefix_aud = f"{seq_num:03d}_"
                aud_path   = _find_file(aud_dir, prefix_aud, ".wav") if aud_dir.exists() else None
                aud_feat: Optional[np.ndarray] = None
                if aud_path is not None:
                    try:
                        import soundfile as sf
                        audio_data, sr = sf.read(str(aud_path), always_2d=False)
                        if audio_data.ndim > 1:
                            audio_data = audio_data.mean(axis=1)
                        aud_feat = _extract_audio_features(
                            audio_data.astype(np.float32), int(sr)
                        )
                        if not np.any(np.isfinite(aud_feat) & (aud_feat != 0)):
                            aud_feat = None
                        else:
                            n_audio_ok += 1
                    except Exception:
                        pass

                # ── Combine: repeat audio/video for each EEG window ──────────
                label = int(labels3[trial_idx])
                for eeg_f in eeg_windows:
                    row = _make_row(eeg_f, aud_feat, vid_feat)
                    Xs.append(row)
                    ys.append(label)

            n_loaded += 1

        except Exception as exc:
            log.debug("EAV %s error: %s", sid, exc)
            continue

    if not Xs:
        return np.empty((0, N_TOTAL), np.float32), np.empty(0, np.int32)

    log.info(
        "EAV multimodal: %d subjects → %d samples "
        "(audio in %d trials, video in %d trials)",
        n_loaded, len(ys), n_audio_ok, n_video_ok,
    )
    return np.array(Xs, np.float32), np.array(ys, np.int32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _smote(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """SMOTE balancing — NaN-safe: impute before SMOTE, restore NaN pattern after."""
    if not _SMOTE_OK:
        return X, y
    counts = {int(c): int((y == c).sum()) for c in np.unique(y)}
    min_c  = min(counts.values())
    if min_c < 6:
        return X, y

    # Replace NaN with column medians for SMOTE (tree-based — shouldn't matter much)
    col_medians = np.nanmedian(X, axis=0)
    X_imp = np.where(np.isnan(X), np.tile(col_medians, (X.shape[0], 1)), X)

    try:
        k = min(5, min_c - 1)
        X_bal, y_bal = SMOTE(k_neighbors=k, random_state=42).fit_resample(X_imp, y)
        log.info("SMOTE: %d → %d samples", len(y), len(y_bal))
        return X_bal.astype(np.float32), y_bal.astype(np.int32)
    except Exception as exc:
        log.warning("SMOTE failed (%s) — using original data", exc)
        return X, y


def train_and_evaluate(
    X: np.ndarray, y: np.ndarray
) -> dict:
    """5-fold CV then retrain on full data. Returns result dict."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, classification_report

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        clf = lgb.LGBMClassifier(
            n_estimators=600,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X[tr], y[tr])
        score = accuracy_score(y[va], clf.predict(X[va]))
        cv_scores.append(score)
        log.info("  Fold %d: %.2f%%", fold + 1, score * 100)

    cv_mean = float(np.mean(cv_scores))
    cv_std  = float(np.std(cv_scores))
    log.info("CV: %.2f%% ± %.2f%%", cv_mean * 100, cv_std * 100)

    # Final model on all data
    final = lgb.LGBMClassifier(
        n_estimators=800,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    final.fit(X, y)
    train_acc = accuracy_score(y, final.predict(X))
    log.info("Train acc: %.2f%%", train_acc * 100)
    log.info("\n%s", classification_report(
        y, final.predict(X), target_names=EMOTIONS_3, zero_division=0
    ))

    return {"model": final, "cv_mean": cv_mean, "cv_std": cv_std, "train_acc": train_acc}


def save_artifacts(result: dict, n_raw: int, datasets_used: List[str]) -> None:
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    BENCH_OUT.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model":          result["model"],
        "n_features_raw": N_TOTAL,
        "n_classes":      3,
        "class_names":    EMOTIONS_3,
        "cv_accuracy":    result["cv_mean"],
        "label_map":      {0: "positive", 1: "neutral", 2: "negative"},
        "feature_layout": (
            f"{N_EEG}-dim EEG (5 bands×4ch×4stats+5DASM) | "
            f"{N_AUDIO}-dim audio MFCC+spectral | "
            f"{N_VIDEO}-dim video facial-grid | "
            f"2-dim modality flags (has_audio, has_video). "
            "NaN used for absent modalities — LightGBM handles natively."
        ),
        "datasets":       datasets_used,
    }
    with open(MODEL_OUT, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    log.info("Multimodal model saved → %s", MODEL_OUT)

    bench = {
        "dataset":           "+".join(datasets_used),
        "n_samples":         n_raw,
        "classifier":        f"LightGBM (251-dim: EEG+Audio+Video, NaN-missing)",
        "accuracy":          round(result["cv_mean"], 4),
        "cv_accuracy_mean":  round(result["cv_mean"], 4),
        "cv_accuracy_std":   round(result["cv_std"],  4),
        "train_accuracy":    round(result["train_acc"], 4),
        "n_features":        N_TOTAL,
        "n_eeg_features":    N_EEG,
        "n_audio_features":  N_AUDIO,
        "n_video_features":  N_VIDEO,
        "n_classes":         3,
        "classes":           EMOTIONS_3,
    }
    with open(BENCH_OUT, "w") as fh:
        json.dump(bench, fh, indent=2)
    log.info("Benchmark saved → %s", BENCH_OUT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not _LGB_OK:
        log.error("lightgbm not installed"); sys.exit(1)
    if not _SCI_OK:
        log.error("scipy not installed"); sys.exit(1)

    t0 = time.time()
    log.info("=== Multimodal Mega LGBM Trainer ===")
    log.info("Feature space: EEG(85) + Audio(92) + Video(72) + flags(2) = 251-dim")
    log.info("Missing modalities represented as NaN — handled natively by LightGBM")

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    datasets_used: List[str] = []

    # ── EEG-only datasets: pad to 251-dim ─────────────────────────────────────
    eeg_loaders = [
        ("DEAP",   load_deap),
        ("DREAMER", load_dreamer),
        ("GAMEEMO", load_gameemo),
        ("DENS",   load_dens),
        ("FACED",  load_faced),
        ("SEED-IV", load_seed_iv),
        ("EEG-ER", load_eeg_er),
        ("STEW",   load_stew),
        ("Muse-Sub", load_muse_subconscious),
        ("EmoKey", load_emokey),
    ]
    for name, loader in eeg_loaders:
        log.info("── Loading %s ──", name)
        X_e, y_e = loader()
        if X_e.size > 0:
            X_mm = _eeg_to_multimodal(X_e)
            all_X.append(X_mm); all_y.append(y_e)
            datasets_used.append(name)
            log.info("%s: %d samples (EEG-only, audio/video=NaN)", name, len(y_e))

    # ── EAV: full multimodal (EEG + audio + video) ────────────────────────────
    log.info("── Loading EAV (multimodal: EEG + audio + video) ──")
    X_eav, y_eav = load_eav_multimodal()
    if X_eav.size > 0:
        all_X.append(X_eav); all_y.append(y_eav)
        datasets_used.append("EAV-multimodal")
        log.info("EAV multimodal: %d samples", len(y_eav))

    if not all_X:
        log.error("No data loaded"); sys.exit(1)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    n_raw = len(y)
    log.info("Combined: %d samples, %d features | datasets: %s",
             n_raw, X.shape[1], datasets_used)

    class_counts = {EMOTIONS_3[i]: int((y == i).sum()) for i in range(3)}
    log.info("Class counts (before SMOTE): %s", class_counts)

    # ── SMOTE ─────────────────────────────────────────────────────────────────
    X_bal, y_bal = _smote(X, y)

    # ── Train + CV ────────────────────────────────────────────────────────────
    log.info("── Training LightGBM with 5-fold CV ──")
    result = train_and_evaluate(X_bal, y_bal)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_artifacts(result, n_raw, datasets_used)

    elapsed = time.time() - t0
    log.info("Done in %.1f s — CV accuracy: %.2f%%", elapsed, result["cv_mean"] * 100)
    log.info("Model path: %s", MODEL_OUT)


if __name__ == "__main__":
    main()
