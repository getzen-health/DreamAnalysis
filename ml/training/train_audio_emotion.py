"""Audio emotion model trainer — EAV speaking-trial .wav files.

Extracts librosa MFCC + spectral features from EAV Audio/ .wav files.
Labels from filename: Happiness→positive(0), Anger/Sadness→negative(2),
                      Neutral/Calmness→neutral(1).

Features (92-dim):
  [0:80]   40 MFCCs × 2 stats (mean, std)
  [80:82]  Spectral centroid mean + std
  [82:84]  Spectral bandwidth mean + std
  [84:86]  Spectral rolloff mean + std
  [86:88]  Zero crossing rate mean + std
  [88:90]  RMS energy mean + std
  [90:92]  Spectral flatness mean + std

Usage (from ml/ directory):
    python3 training/train_audio_emotion.py
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
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False

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
MODEL_OUT     = _ML_ROOT / "models" / "saved" / "audio_emotion_lgbm.pkl"
BENCHMARK_OUT = _ML_ROOT / "benchmarks" / "audio_emotion_benchmark.json"
_SR           = 22050  # resample all audio to 22 kHz
_N_MFCC       = 40
N_FEATURES    = 92

# Filename keyword → 3-class label
_AUDIO_LABEL_MAP = {
    "Happiness": 0,   # positive
    "Anger":     2,   # negative
    "Sadness":   2,   # negative
    "Neutral":   1,   # neutral
    "Calmness":  1,   # neutral (calm ≈ neutral/relaxed)
}
EMOTIONS_3 = ["positive", "neutral", "negative"]


def extract_audio_features(y: np.ndarray, sr: int = _SR) -> np.ndarray:
    """Extract 92-dim feature vector from a raw audio waveform."""
    if len(y) < sr // 4:   # < 0.25 sec — skip
        return np.zeros(N_FEATURES, dtype=np.float32)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=_N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)    # (40,)
    mfcc_std  = mfcc.std(axis=1)     # (40,)

    # Spectral features
    sc  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sb  = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    sr2 = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    sf  = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    def ms(x: np.ndarray) -> List[float]:
        return [float(x.mean()), float(x.std())]

    feat = np.concatenate([
        mfcc_mean, mfcc_std,     # 80
        ms(sc), ms(sb), ms(sr2), # 6
        ms(zcr), ms(rms),        # 4
        ms(sf),                  # 2
    ]).astype(np.float32)        # total: 92

    feat = np.where(np.isfinite(feat), feat, 0.0)
    return feat


def load_eav_audio() -> Tuple[np.ndarray, np.ndarray]:
    """Load all EAV speaking-trial audio files and extract features."""
    if not _LIBROSA_OK:
        log.error("librosa not installed — cannot extract audio features")
        return np.empty((0, N_FEATURES), np.float32), np.empty(0, np.int32)

    subj_dirs = sorted(
        d for d in _EAV_DIR.iterdir()
        if d.is_dir() and d.name.lower().startswith("subject")
    )
    if not subj_dirs:
        log.error("No subject dirs in %s", _EAV_DIR)
        return np.empty((0, N_FEATURES), np.float32), np.empty(0, np.int32)

    Xs, ys = [], []
    n_files = 0

    for subj_dir in subj_dirs:
        audio_dir = subj_dir / "Audio"
        if not audio_dir.exists():
            continue
        for wav_path in sorted(audio_dir.glob("*.wav")):
            # Filename pattern: 064_Trial_04_Speaking_Happiness_aud.wav
            stem = wav_path.stem  # e.g. "064_Trial_04_Speaking_Happiness_aud"
            label = None
            for keyword, lbl in _AUDIO_LABEL_MAP.items():
                if keyword in stem:
                    label = lbl
                    break
            if label is None:
                continue
            try:
                y, sr = librosa.load(str(wav_path), sr=_SR, mono=True)
                feat = extract_audio_features(y, sr)
                Xs.append(feat)
                ys.append(label)
                n_files += 1
            except Exception as exc:
                log.debug("Audio load error %s: %s", wav_path.name, exc)

    if not Xs:
        return np.empty((0, N_FEATURES), np.float32), np.empty(0, np.int32)

    log.info("Audio: %d files → %d samples", n_files, len(ys))
    return np.array(Xs, np.float32), np.array(ys, np.int32)


def train(X: np.ndarray, y: np.ndarray) -> dict:
    """Scale → PCA → 5-fold CV LightGBM → retrain on full data."""
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    n_comp = min(60, X.shape[1], X.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=42)
    X_pca  = pca.fit_transform(X_sc)
    log.info("PCA: %d → %d dims", X.shape[1], n_comp)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for fold, (tr, va) in enumerate(skf.split(X_pca, y)):
        clf = lgb.LGBMClassifier(
            n_estimators=600, num_leaves=31, learning_rate=0.05,
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
        n_estimators=800, num_leaves=31, learning_rate=0.05,
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
        "sr":             _SR,
        "n_mfcc":         _N_MFCC,
        "feature_layout": (
            "92-dim: 40 MFCCs×2 (mean+std)=80, spectral_centroid×2=2, "
            "bandwidth×2=2, rolloff×2=2, ZCR×2=2, RMS×2=2, flatness×2=2"
        ),
    }
    with open(MODEL_OUT, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    log.info("Audio model saved → %s", MODEL_OUT)

    bench = {
        "dataset":          "EAV-Audio",
        "n_samples":        n_samples,
        "classifier":       "LightGBM (PCA 92→60)",
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
    if not _LIBROSA_OK:
        log.error("librosa not installed"); sys.exit(1)

    log.info("=== Audio Emotion Trainer (EAV) ===")
    X, y = load_eav_audio()
    if X.size == 0:
        log.error("No audio data loaded — check EAV path %s", _EAV_DIR)
        sys.exit(1)

    counts = {EMOTIONS_3[i]: int((y == i).sum()) for i in range(3)}
    log.info("Class counts: %s", counts)

    result = train(X, y)
    save(result, len(y))
    log.info("Audio model CV accuracy: %.2f%%", result["cv_mean"] * 100)


if __name__ == "__main__":
    main()
