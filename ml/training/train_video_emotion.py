"""Video facial emotion model trainer — EAV .mp4 files.

Samples frames from each video, detects face with Haar cascade,
extracts 72-dim features (6×6 grid: mean intensity + gradient magnitude),
averages over frames → 1 feature vector per video trial.

Labels from filename: Happiness→positive(0), Anger/Sadness→negative(2),
                      Neutral/Calmness→neutral(1).

Features (72-dim per trial — temporal average of per-frame features):
  [0:36]  6×6 grid of 8×8 blocks — mean grayscale intensity
  [36:72] 6×6 grid of 8×8 blocks — mean Sobel gradient magnitude

Usage (from ml/ directory):
    python3 training/train_video_emotion.py
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

_HERE    = Path(__file__).resolve().parent
_ML_ROOT = _HERE.parent
for _p in [str(_ML_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

try:
    import lightgbm as lgb
    _LGB_OK = True
except ImportError:
    _LGB_OK = False

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_EAV_DIR      = _ML_ROOT / "data" / "eav" / "EAV"
MODEL_OUT     = _ML_ROOT / "models" / "saved" / "video_emotion_lgbm.pkl"
BENCHMARK_OUT = _ML_ROOT / "benchmarks" / "video_emotion_benchmark.json"
N_FEATURES    = 72
FACE_SIZE     = 48    # resize face crop to 48×48
GRID          = 6     # 6×6 grid of 8×8 blocks
BLOCK         = FACE_SIZE // GRID  # 8

# Filename keyword → 3-class label
_VIDEO_LABEL_MAP = {
    "Happiness": 0,
    "Anger":     2,
    "Sadness":   2,
    "Neutral":   1,
    "Calmness":  1,
}
EMOTIONS_3 = ["positive", "neutral", "negative"]


def _init_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def extract_frame_features(gray_frame: np.ndarray) -> Optional[np.ndarray]:
    """Extract 72-dim features from a single grayscale frame.

    Returns None if no face detected.
    """
    face_cascade = _init_face_detector()
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )
    if len(faces) == 0:
        return None

    # Use largest detected face
    x, y, w, h = sorted(faces, key=lambda f: -f[2])[0]
    face = cv2.resize(gray_frame[y:y+h, x:x+w], (FACE_SIZE, FACE_SIZE))
    face_f = face.astype(np.float32) / 255.0

    # Gradient magnitude
    gx  = cv2.Sobel(face_f, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(face_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)

    # 6×6 grid of 8×8 blocks
    intensity_grid = face_f.reshape(GRID, BLOCK, GRID, BLOCK).mean(axis=(1, 3))
    gradient_grid  = mag.reshape(GRID, BLOCK, GRID, BLOCK).mean(axis=(1, 3))

    return np.concatenate([
        intensity_grid.flatten(),
        gradient_grid.flatten(),
    ]).astype(np.float32)   # (72,)


def extract_video_features(
    mp4_path: Path,
    n_samples: int = 6,
    min_face_frames: int = 2,
) -> Optional[np.ndarray]:
    """Extract 72-dim feature vector for an entire video (temporal average).

    Jumps directly to `n_samples` evenly-spaced frame positions using
    cap.set(CAP_PROP_POS_FRAMES, ...) — much faster than sequential read.

    Returns None if fewer than `min_face_frames` faces detected.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    sample_positions = np.linspace(
        total_frames * 0.1,   # skip first 10% (settling)
        total_frames * 0.9,   # skip last 10%
        n_samples, dtype=int
    )

    frame_feats: List[np.ndarray] = []
    for pos in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        feat = extract_frame_features(gray)
        if feat is not None:
            frame_feats.append(feat)

    cap.release()

    if len(frame_feats) < min_face_frames:
        return None

    return np.mean(frame_feats, axis=0).astype(np.float32)   # (72,)


def load_eav_video() -> Tuple[np.ndarray, np.ndarray]:
    """Load all EAV video files and extract features."""
    subj_dirs = sorted(
        d for d in _EAV_DIR.iterdir()
        if d.is_dir() and d.name.lower().startswith("subject")
    )
    if not subj_dirs:
        log.error("No subject dirs in %s", _EAV_DIR)
        return np.empty((0, N_FEATURES), np.float32), np.empty(0, np.int32)

    Xs, ys = [], []
    n_videos = 0
    n_total  = 0

    for subj_dir in subj_dirs:
        video_dir = subj_dir / "Video"
        if not video_dir.exists():
            continue
        for mp4_path in sorted(video_dir.glob("*.mp4")):
            # Filename: 001_Trial_01_Listening_Neutral.mp4
            stem = mp4_path.stem
            label = None
            for keyword, lbl in _VIDEO_LABEL_MAP.items():
                if keyword in stem:
                    label = lbl
                    break
            if label is None:
                continue
            n_total += 1
            feat = extract_video_features(mp4_path, n_samples=6)
            if feat is not None:
                Xs.append(feat)
                ys.append(label)
                n_videos += 1

    log.info("Video: %d/%d files yielded face features", n_videos, n_total)
    if not Xs:
        return np.empty((0, N_FEATURES), np.float32), np.empty(0, np.int32)

    return np.array(Xs, np.float32), np.array(ys, np.int32)


def train(X: np.ndarray, y: np.ndarray) -> dict:
    """Scale → PCA → 5-fold CV LightGBM → retrain on full data."""
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    n_comp = min(50, X.shape[1], X.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=42)
    X_pca  = pca.fit_transform(X_sc)
    log.info("PCA: %d → %d dims", X.shape[1], n_comp)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for fold, (tr, va) in enumerate(skf.split(X_pca, y)):
        clf = lgb.LGBMClassifier(
            n_estimators=400, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
        clf.fit(X_pca[tr], y[tr])
        score = accuracy_score(y[va], clf.predict(X_pca[va]))
        cv_scores.append(score)
        log.info("  Fold %d: %.2f%%", fold + 1, score * 100)

    cv_mean = float(np.mean(cv_scores))
    cv_std  = float(np.std(cv_scores))
    log.info("CV: %.2f%% ± %.2f%%", cv_mean * 100, cv_std * 100)

    final = lgb.LGBMClassifier(
        n_estimators=600, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
    )
    final.fit(X_pca, y)
    log.info("Train acc: %.2f%%", accuracy_score(y, final.predict(X_pca)) * 100)
    log.info("\n%s", classification_report(y, final.predict(X_pca),
             target_names=EMOTIONS_3, zero_division=0))

    return {"model": final, "scaler": scaler, "pca": pca,
            "cv_mean": cv_mean, "cv_std": cv_std}


def save(result: dict, n_samples: int) -> None:
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_OUT.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model":          result["model"],
        "scaler":         result["scaler"],
        "pca":            result["pca"],
        "n_features_raw": N_FEATURES,
        "n_classes":      3,
        "class_names":    EMOTIONS_3,
        "cv_accuracy":    result["cv_mean"],
        "label_map":      {0: "positive", 1: "neutral", 2: "negative"},
        "face_size":      FACE_SIZE,
        "grid":           GRID,
        "feature_layout": (
            "72-dim: 6×6 grid of 8×8 blocks × 2 (mean intensity + "
            "mean Sobel gradient), temporal-averaged over sampled frames"
        ),
    }
    with open(MODEL_OUT, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    log.info("Video model saved → %s", MODEL_OUT)

    bench = {
        "dataset":          "EAV-Video",
        "n_samples":        n_samples,
        "classifier":       "LightGBM (PCA 72→50)",
        "accuracy":         round(result["cv_mean"], 4),
        "cv_accuracy_mean": round(result["cv_mean"], 4),
        "cv_accuracy_std":  round(result["cv_std"],  4),
        "n_features":       N_FEATURES,
        "n_classes":        3,
        "classes":          EMOTIONS_3,
    }
    with open(BENCHMARK_OUT, "w") as fh:
        json.dump(bench, fh, indent=2)
    log.info("Benchmark saved → %s", BENCHMARK_OUT)


def main() -> None:
    if not _LGB_OK:
        log.error("lightgbm not installed"); sys.exit(1)
    if not _CV2_OK:
        log.error("opencv-python not installed"); sys.exit(1)

    log.info("=== Video Facial Emotion Trainer (EAV) ===")
    log.info("Extracting features from EAV Video/ files (this takes a few minutes)...")
    X, y = load_eav_video()
    if X.size == 0:
        log.error("No video data loaded — check EAV path %s", _EAV_DIR)
        sys.exit(1)

    counts = {EMOTIONS_3[i]: int((y == i).sum()) for i in range(3)}
    log.info("Class counts: %s", counts)

    result = train(X, y)
    save(result, len(y))
    log.info("Video model CV accuracy: %.2f%%", result["cv_mean"] * 100)


if __name__ == "__main__":
    main()
