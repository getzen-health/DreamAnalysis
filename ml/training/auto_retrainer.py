"""Auto-retrainer: rebuilds the personal model from all saved session data.

Runs twice daily (every 12 hours) via the scheduler in main.py.
Uses pre-computed band_powers + emotion features stored in analysis_timeline
so it doesn't need to re-run the expensive EEG signal-processing pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"
USER_MODELS_DIR = Path(__file__).parent.parent / "user_models"

# Feature keys extracted from each timeline entry
_BAND_KEYS = ["delta", "theta", "alpha", "beta", "gamma"]
_EMOTION_KEYS = ["stress_index", "focus_index", "relaxation_index", "valence", "arousal"]


def _entry_to_row(entry: dict, feature_names: List[str]) -> Optional[List[float]]:
    """Convert one analysis_timeline entry to a feature row.

    Returns None if the entry lacks required data.
    """
    emotion = entry.get("emotions") or {}
    band = entry.get("band_powers") or {}
    feats = entry.get("features") or {}

    label = emotion.get("emotion", "")
    if not label or label in ("unknown", "—", ""):
        return None

    # Build feature dict
    feat: Dict[str, float] = {}
    for k in _BAND_KEYS:
        feat[k] = float(band.get(k, 0.0))
    for k in _EMOTION_KEYS:
        feat[k] = float(emotion.get(k, 0.0))
    for k, v in feats.items():
        if isinstance(v, (int, float)):
            feat[f"f_{k}"] = float(v)

    # Align to known feature_names (fill missing with 0)
    return [feat.get(n, 0.0) for n in feature_names]


def _build_feature_names(sample_entry: dict) -> List[str]:
    """Derive ordered feature names from one timeline entry."""
    emotion = sample_entry.get("emotions") or {}
    band = sample_entry.get("band_powers") or {}
    feats = sample_entry.get("features") or {}

    names = _BAND_KEYS + _EMOTION_KEYS
    for k, v in feats.items():
        if isinstance(v, (int, float)):
            names.append(f"f_{k}")
    return names


def collect_training_data(min_samples: int = 20) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load features + labels from every saved session JSON.

    Returns:
        X: feature matrix  (n_samples × n_features) — empty if not enough data
        y: emotion label strings (n_samples,)
        feature_names: ordered feature name list
    """
    all_rows: List[List[float]] = []
    all_labels: List[str] = []
    feature_names: Optional[List[str]] = None

    for json_file in sorted(SESSIONS_DIR.glob("*.json")):
        try:
            with open(json_file) as f:
                meta = json.load(f)
        except Exception:
            continue

        timeline = meta.get("analysis_timeline") or []
        for entry in timeline:
            label = (entry.get("emotions") or {}).get("emotion", "")
            if not label or label in ("unknown", "—", ""):
                continue

            # Determine feature_names from first usable entry
            if feature_names is None:
                feature_names = _build_feature_names(entry)

            row = _entry_to_row(entry, feature_names)
            if row is None:
                continue

            all_rows.append(row)
            all_labels.append(label)

    feature_names = feature_names or (_BAND_KEYS + _EMOTION_KEYS)

    if len(all_labels) < min_samples:
        logger.info(
            f"Auto-retrain: only {len(all_labels)} labeled frames found "
            f"(need ≥ {min_samples}) — skipping."
        )
        return np.array([]), [], feature_names

    return np.array(all_rows, dtype=np.float32), all_labels, feature_names


def retrain_personal_model(user_id: str = "default") -> dict:
    """Retrain the personal SGD classifier from all accumulated session data.

    Saves the model + metadata to user_models/{user_id}_personal.pkl/.json.

    Returns:
        Dict describing the training result.
    """
    try:
        import joblib
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import LabelEncoder
    except ImportError as exc:
        return {"trained": False, "reason": f"Missing dependency: {exc}"}

    X, y_labels, feature_names = collect_training_data()
    if len(y_labels) == 0:
        return {"trained": False, "reason": "Not enough labeled session data"}

    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    classes = list(le.classes_)

    model = SGDClassifier(
        loss="log_loss",
        warm_start=True,
        max_iter=200,
        random_state=42,
        n_iter_no_change=10,
    )
    model.fit(X, y)
    train_acc = float(np.mean(model.predict(X) == y))

    # Persist
    USER_MODELS_DIR.mkdir(exist_ok=True)
    model_path = USER_MODELS_DIR / f"{user_id}_personal.pkl"
    meta_path = USER_MODELS_DIR / f"{user_id}_personal_meta.json"

    joblib.dump(model, model_path)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "classes": classes,
                "feature_names": feature_names,
                "n_samples": len(y_labels),
                "train_accuracy": round(train_acc, 4),
            },
            f,
            indent=2,
        )

    logger.info(
        f"[auto-retrain] user={user_id} samples={len(y_labels)} "
        f"classes={classes} acc={train_acc:.3f}"
    )

    return {
        "trained": True,
        "n_samples": len(y_labels),
        "classes": classes,
        "train_accuracy": round(train_acc, 3),
    }
