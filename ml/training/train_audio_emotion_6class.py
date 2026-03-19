"""Audio emotion model trainer — 6-class retraining on EAV-Audio.

Replaces the 3-class (positive/neutral/negative) model with a 5-class model
using fine-grained EAV labels. EAV does not contain "fear" or "surprise", so
those classes are absent from training data.

Label mapping:
  0 = happy    ← EAV "Happiness"
  1 = sad      ← EAV "Sadness"
  2 = angry    ← EAV "Anger"
  3 = fear     ← NOT in EAV (class absent — model trained on 5 classes)
  4 = surprise ← NOT in EAV (class absent — model trained on 5 classes)
  5 = neutral  ← EAV "Neutral" + "Calmness"

Because fear/surprise have no training data, the model is saved as 5-class
with class names [happy, sad, angry, neutral, calm]. The label_map in the
pickle communicates this to voice_emotion_model.py inference code.

Features: 92-dim MFCC vector (identical to original train_audio_emotion.py)
Augmentation: 4× data via noise / pitch / speed / gain (augment_audio_full)
Training: StratifiedKFold 5-fold, LightGBM

Output:
  ml/models/saved/audio_emotion_lgbm.pkl   (overwrites)
  ml/models/saved/audio_emotion_benchmark.json
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
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

# augment_audio_full not used — we do fast feature-space augmentation instead

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_EAV_DIR = _ML_ROOT / "data" / "eav" / "EAV"
MODEL_OUT = _ML_ROOT / "models" / "saved" / "audio_emotion_lgbm.pkl"
BENCHMARK_OUT = _ML_ROOT / "models" / "saved" / "audio_emotion_benchmark.json"
_SR = 22050
_N_MFCC = 40
N_FEATURES = 92

# 5-class mapping (EAV does not have fear/surprise)
# class index: 0=happy 1=sad 2=angry 3=neutral 4=calm
_AUDIO_LABEL_MAP_6 = {
    "Happiness": 0,   # happy
    "Sadness":   1,   # sad
    "Anger":     2,   # angry
    "Neutral":   3,   # neutral
    "Calmness":  4,   # calm (relaxed neutral variant)
}
# What we will call these classes in the saved model
EMOTIONS_5 = ["happy", "sad", "angry", "neutral", "calm"]

# Human-readable label_map for downstream consumers
LABEL_MAP_5 = {0: "happy", 1: "sad", 2: "angry", 3: "neutral", 4: "calm"}


# ---------------------------------------------------------------------------
# Feature extraction (identical to original train_audio_emotion.py)
# ---------------------------------------------------------------------------

def extract_audio_features(y: np.ndarray, sr: int = _SR) -> np.ndarray:
    """Extract 92-dim feature vector from a raw audio waveform."""
    if len(y) < sr // 4:
        return np.zeros(N_FEATURES, dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=_N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    sr2 = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    sf = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    def ms(x: np.ndarray) -> List[float]:
        return [float(x.mean()), float(x.std())]

    feat = np.concatenate([
        mfcc_mean, mfcc_std,
        ms(sc), ms(sb), ms(sr2),
        ms(zcr), ms(rms),
        ms(sf),
    ]).astype(np.float32)

    feat = np.where(np.isfinite(feat), feat, 0.0)
    return feat


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eav_audio_6class() -> Tuple[np.ndarray, np.ndarray]:
    """Load EAV audio and extract 92-dim features with 5-class labels."""
    if not _LIBROSA_OK:
        log.error("librosa not installed")
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
            stem = wav_path.stem
            label = None
            for keyword, lbl in _AUDIO_LABEL_MAP_6.items():
                if keyword in stem:
                    label = lbl
                    break
            if label is None:
                continue
            try:
                y_audio, sr = librosa.load(str(wav_path), sr=_SR, mono=True)
                feat = extract_audio_features(y_audio, sr)
                Xs.append(feat)
                ys.append(label)
                n_files += 1
            except Exception as exc:
                log.debug("Audio load error %s: %s", wav_path.name, exc)

    if not Xs:
        return np.empty((0, N_FEATURES), np.float32), np.empty(0, np.int32)

    log.info("Loaded %d audio files across %d subjects", n_files, len(subj_dirs))
    return np.array(Xs, np.float32), np.array(ys, np.int32)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _augment_one_fast(args: Tuple[np.ndarray, int, int]) -> List[Tuple[np.ndarray, int]]:
    """Fast feature-space augmentation: noise + gain only (no pitch/speed reload)."""
    feat_orig, label, n_aug = args
    results = []
    for _ in range(n_aug):
        feat = feat_orig.copy()
        # Add small Gaussian noise to feature values (simulates mic variability)
        feat += np.random.normal(0, 0.02 * np.abs(feat).mean() + 1e-6, feat.shape).astype(np.float32)
        # Random gain: scale all features by ±10%
        gain = np.random.uniform(0.9, 1.1)
        feat *= gain
        feat = np.where(np.isfinite(feat), feat, 0.0)
        results.append((feat.astype(np.float32), label))
    return results


def augment_features_fast(
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    n_aug: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast feature-space augmentation — no audio reload, noise + gain only.

    Avoids librosa pitch/speed which is too slow for 4K files × 3 augmentations.
    Operates directly on the 92-dim feature vectors.
    """
    aug_Xs, aug_ys = [], []
    for i in range(len(X_orig)):
        pairs = _augment_one_fast((X_orig[i], y_orig[i], n_aug))
        for feat, lbl in pairs:
            aug_Xs.append(feat)
            aug_ys.append(lbl)
    return np.array(aug_Xs, np.float32), np.array(aug_ys, np.int32)


def load_eav_audio_with_augmentation(n_aug: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Load EAV audio + apply fast feature-space augmentation.

    Step 1: Load all WAV files and extract 92-dim features (single pass).
    Step 2: Augment feature vectors directly (noise + gain) — fast, no audio reload.
    Total = (1 + n_aug)× original data.
    """
    if not _LIBROSA_OK:
        log.error("librosa not installed")
        return np.empty((0, N_FEATURES), np.float32), np.empty(0, np.int32)

    subj_dirs = sorted(
        d for d in _EAV_DIR.iterdir()
        if d.is_dir() and d.name.lower().startswith("subject")
    )
    if not subj_dirs:
        log.error("No subject dirs in %s", _EAV_DIR)
        return np.empty((0, N_FEATURES), np.float32), np.empty(0, np.int32)

    Xs_orig, ys_orig = [], []
    n_files = 0

    for subj_dir in subj_dirs:
        audio_dir = subj_dir / "Audio"
        if not audio_dir.exists():
            continue
        for wav_path in sorted(audio_dir.glob("*.wav")):
            stem = wav_path.stem
            label = None
            for keyword, lbl in _AUDIO_LABEL_MAP_6.items():
                if keyword in stem:
                    label = lbl
                    break
            if label is None:
                continue
            try:
                y_audio, sr = librosa.load(str(wav_path), sr=_SR, mono=True)
                feat = extract_audio_features(y_audio, sr)
                Xs_orig.append(feat)
                ys_orig.append(label)
                n_files += 1
            except Exception as exc:
                log.debug("Audio load error %s: %s", wav_path.name, exc)

    if not Xs_orig:
        return np.empty((0, N_FEATURES), np.float32), np.empty(0, np.int32)

    log.info("Original: %d files loaded", n_files)

    X_orig = np.array(Xs_orig, np.float32)
    y_orig = np.array(ys_orig, np.int32)

    log.info("Augmenting features ×%d (fast feature-space noise+gain)...", n_aug)
    X_aug, y_aug = augment_features_fast(X_orig, y_orig, n_aug=n_aug)
    log.info("Augmented samples: %d", len(y_aug))

    X_all = np.concatenate([X_orig, X_aug], axis=0)
    y_all = np.concatenate([y_orig, y_aug], axis=0)
    log.info("Total samples after augmentation: %d", len(y_all))

    return X_all, y_all


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X: np.ndarray, y: np.ndarray) -> dict:
    """Scale → PCA → 5-fold CV LightGBM → final model on full data."""
    n_classes = len(np.unique(y))
    log.info("Training on %d samples, %d classes", len(y), n_classes)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    n_comp = min(60, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_sc)
    log.info("PCA: %d → %d dims (%.1f%% variance explained)",
             X.shape[1], n_comp,
             pca.explained_variance_ratio_.sum() * 100)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for fold, (tr, va) in enumerate(skf.split(X_pca, y)):
        clf = lgb.LGBMClassifier(
            n_estimators=600,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_pca[tr], y[tr])
        score = accuracy_score(y[va], clf.predict(X_pca[va]))
        cv_scores.append(score)
        log.info("  Fold %d: %.2f%%", fold + 1, score * 100)

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    log.info("CV: %.2f%% ± %.2f%%", cv_mean * 100, cv_std * 100)

    # Final model on full data
    final = lgb.LGBMClassifier(
        n_estimators=800,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    final.fit(X_pca, y)
    train_acc = accuracy_score(y, final.predict(X_pca))
    log.info("Train acc (full data): %.2f%%", train_acc * 100)

    target_names = [EMOTIONS_5[i] for i in sorted(np.unique(y))]
    log.info("\n%s", classification_report(
        y, final.predict(X_pca),
        target_names=target_names,
        zero_division=0,
    ))

    return {
        "model": final,
        "scaler": scaler,
        "pca": pca,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "n_classes": n_classes,
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(result: dict, n_samples: int) -> None:
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": result["model"],
        "scaler": result["scaler"],
        "pca": result["pca"],
        "n_features_raw": N_FEATURES,
        "n_classes": result["n_classes"],
        "class_names": EMOTIONS_5,
        "cv_accuracy": result["cv_mean"],
        "label_map": LABEL_MAP_5,
        "sr": _SR,
        "n_mfcc": _N_MFCC,
        "feature_layout": (
            "92-dim: 40 MFCCs×2 (mean+std)=80, spectral_centroid×2=2, "
            "bandwidth×2=2, rolloff×2=2, ZCR×2=2, RMS×2=2, flatness×2=2"
        ),
        "note": (
            "5-class model trained on EAV (Happiness/Sadness/Anger/Neutral/Calmness). "
            "EAV does not contain 'fear' or 'surprise' so those classes are absent. "
            "Replaces original 3-class (positive/neutral/negative) model."
        ),
    }
    with open(MODEL_OUT, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    log.info("Model saved → %s", MODEL_OUT)

    bench = {
        "dataset": "EAV-Audio",
        "n_samples_original": n_samples,
        "n_samples_augmented": n_samples,
        "classifier": "LightGBM (PCA 92→60, 5-class)",
        "accuracy": round(result["cv_mean"], 4),
        "cv_accuracy_mean": round(result["cv_mean"], 4),
        "cv_accuracy_std": round(result["cv_std"], 4),
        "n_features": N_FEATURES,
        "n_classes": result["n_classes"],
        "classes": EMOTIONS_5,
        "note": "EAV has 5 emotions; fear/surprise absent from training data",
    }
    with open(BENCHMARK_OUT, "w") as fh:
        json.dump(bench, fh, indent=2)
    log.info("Benchmark saved → %s", BENCHMARK_OUT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not _LGB_OK:
        log.error("lightgbm not installed"); sys.exit(1)
    if not _LIBROSA_OK:
        log.error("librosa not installed"); sys.exit(1)

    log.info("=== Audio Emotion Trainer — 6-class (EAV, 5 present classes) ===")
    log.info("EAV path: %s", _EAV_DIR)

    subj_dirs = [d for d in _EAV_DIR.iterdir()
                 if d.is_dir() and d.name.lower().startswith("subject")]
    log.info("Found %d subject directories", len(subj_dirs))

    X, y = load_eav_audio_with_augmentation(n_aug=3)
    if X.size == 0:
        log.error("No audio data loaded — check EAV path %s", _EAV_DIR)
        sys.exit(1)

    counts = {EMOTIONS_5[i]: int((y == i).sum()) for i in range(len(EMOTIONS_5))}
    log.info("Class counts (after augmentation): %s", counts)

    result = train(X, y)
    save(result, len(y))

    log.info("")
    log.info("=== RESULT ===")
    log.info("CV accuracy (5-class): %.2f%% ± %.2f%%",
             result["cv_mean"] * 100, result["cv_std"] * 100)
    log.info("Previous model: 3-class at 82.61%%")
    log.info("Classes: %s", EMOTIONS_5)
    log.info("Note: fear/surprise absent from EAV — 5 classes trained")


if __name__ == "__main__":
    main()
