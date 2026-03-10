"""Train the voice biomarker ensemble for subtle emotion detection.

Builds a LightGBM classifier on top of the 10-dimensional extended
biomarker feature vector + 6 deep-model probabilities = 16 total features.

Usage::

    python training/train_voice_ensemble.py \\
        --data-dir /path/to/labeled_audio \\
        --output models/saved/voice_ensemble_lgbm.pkl

Directory layout expected::

    labeled_audio/
        happy/       *.wav  (or .mp3, .flac)
        sad/
        angry/
        fear/
        surprise/
        neutral/

The script extracts features for each file, trains with 5-fold CV, and
saves a dict {"scaler": ..., "clf": ..., "label_encoder": ...}.

Requirements: lightgbm, librosa, scikit-learn
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

log = logging.getLogger(__name__)

_EMOTIONS_6 = ["happy", "sad", "angry", "fear", "surprise", "neutral"]


def _extract_features_for_file(
    path: Path, sr: int = 16000
) -> Tuple[np.ndarray, bool]:
    """Return (feature_vector, success_flag)."""
    try:
        import librosa  # type: ignore
    except ImportError:
        raise RuntimeError("librosa is required for feature extraction")

    try:
        audio, orig_sr = librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:
        log.warning("Cannot load %s: %s", path, exc)
        return np.zeros(16, dtype=np.float32), False

    if len(audio) < sr * 1.5:
        log.debug("Skipping %s — too short", path.name)
        return np.zeros(16, dtype=np.float32), False

    # Extended biomarker features (10-dim)
    from models.voice_biomarker_ensemble import _extract_extended_biomarkers

    bio = _extract_extended_biomarkers(audio, sr)
    if bio is None:
        log.debug("Biomarker extraction failed for %s", path.name)
        bio = np.zeros(10, dtype=np.float32)

    # Deep model probabilities (6-dim) — use gracefully if not available
    deep_probs = np.full(6, 1.0 / 6, dtype=np.float32)
    try:
        from models.voice_emotion_model import VoiceEmotionModel  # type: ignore

        model = VoiceEmotionModel()
        result = model.predict(audio, sample_rate=sr)
        if result and "probabilities" in result:
            deep_probs = np.array(
                [result["probabilities"].get(e, 1.0 / 6) for e in _EMOTIONS_6],
                dtype=np.float32,
            )
    except Exception:
        pass  # keep uniform priors if deep model unavailable

    features = np.concatenate([bio, deep_probs])  # (16,)
    return features.astype(np.float32), True


def load_dataset(data_dir: str, sr: int = 16000):
    """Walk data_dir/<emotion>/*.wav and return (X, y) arrays."""
    X: List[np.ndarray] = []
    y: List[int] = []

    data_path = Path(data_dir)
    for label_idx, emotion in enumerate(_EMOTIONS_6):
        class_dir = data_path / emotion
        if not class_dir.is_dir():
            log.warning("Missing class directory: %s", class_dir)
            continue
        files = sorted(
            [
                p
                for p in class_dir.iterdir()
                if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}
            ]
        )
        log.info("%s: %d files", emotion, len(files))
        for path in files:
            feat, ok = _extract_features_for_file(path, sr)
            if ok:
                X.append(feat)
                y.append(label_idx)

    if not X:
        raise ValueError(f"No valid audio files found in {data_dir}")

    return np.stack(X), np.array(y, dtype=np.int32)


def train(data_dir: str, output_path: str, sr: int = 16000) -> None:
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError:
        raise RuntimeError("lightgbm is required: pip install lightgbm")

    from sklearn.model_selection import StratifiedKFold, cross_val_score  # type: ignore
    from sklearn.preprocessing import LabelEncoder, StandardScaler  # type: ignore

    log.info("Loading dataset from %s", data_dir)
    X, y_raw = load_dataset(data_dir, sr=sr)

    le = LabelEncoder()
    le.fit(_EMOTIONS_6)
    y = le.transform([_EMOTIONS_6[i] for i in y_raw])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
    log.info(
        "5-fold CV accuracy: %.3f ± %.3f", scores.mean(), scores.std()
    )

    # Final fit on all data
    clf.fit(X_scaled, y)

    # Per-class report
    from sklearn.metrics import classification_report  # type: ignore
    y_pred = clf.predict(X_scaled)
    log.info(
        "Train report:\n%s",
        classification_report(y, y_pred, target_names=_EMOTIONS_6),
    )

    bundle = {"scaler": scaler, "clf": clf, "label_encoder": le, "cv_mean": float(scores.mean())}
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Saved ensemble model to %s (CV %.2f%%)", output_path, scores.mean() * 100)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Train voice biomarker ensemble"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root directory with <emotion>/*.wav subdirs",
    )
    parser.add_argument(
        "--output",
        default="models/saved/voice_ensemble_lgbm.pkl",
        help="Output .pkl path (default: models/saved/voice_ensemble_lgbm.pkl)",
    )
    parser.add_argument(
        "--sr", type=int, default=16000, help="Resample rate (default 16000)"
    )
    args = parser.parse_args()
    train(args.data_dir, args.output, sr=args.sr)


if __name__ == "__main__":
    main()
