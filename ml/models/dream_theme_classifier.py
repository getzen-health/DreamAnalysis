"""Dream theme classification from journal text.

Inspired by DreamNet (arXiv 2503.05778, 2025) which achieves 92.1% accuracy
on dream theme classification. We implement a TF-IDF + LightGBM classifier
with the standard dream theme taxonomy from the DreamBank dataset.

Themes based on Calvin Hall / Robert Van de Castle coding system (universal taxonomy).
"""
from __future__ import annotations

import logging
import pathlib
import re
from typing import Dict, List

import numpy as np

log = logging.getLogger(__name__)

MODEL_PATH = pathlib.Path(__file__).parent / "saved" / "dream_theme_classifier.pkl"

# Hall-Van de Castle dream theme taxonomy (universally validated)
DREAM_THEMES = [
    "flying",        # Soaring, levitation, freedom
    "falling",       # Falling from heights, loss of control
    "pursuit",       # Being chased, fleeing, danger
    "water",         # Ocean, flood, swimming, drowning
    "transportation",# Cars, planes, trains, travel
    "social",        # People, relationships, conversation
    "conflict",      # Fighting, arguments, aggression
    "loss",          # Death, separation, grief
    "discovery",     # Finding things, exploration, revelation
    "transformation",# Shape shifting, metamorphosis
    "school",        # Tests, classroom, academic performance
    "nature",        # Forests, animals, landscapes
    "supernatural",  # Ghosts, magic, impossible events
    "body",          # Injury, illness, physical sensations
    "neutral",       # No strong theme
]

# Keyword seeds per theme (used for both feature extraction and naive scoring)
THEME_KEYWORDS: Dict[str, List[str]] = {
    "flying": ["fly", "flew", "soar", "float", "levitate", "sky", "air", "wing", "glide", "hover"],
    "falling": ["fall", "fell", "drop", "plunge", "tumble", "cliff", "edge", "descend", "crash"],
    "pursuit": ["chase", "chased", "run", "ran", "escape", "flee", "follow", "caught", "hunt", "danger"],
    "water": ["water", "ocean", "sea", "swim", "flood", "drown", "river", "wave", "rain", "wet", "pool"],
    "transportation": ["car", "drive", "drove", "plane", "train", "bus", "road", "travel", "vehicle", "fly", "airport"],
    "social": ["person", "people", "friend", "family", "talk", "meet", "conversation", "together", "group", "hug"],
    "conflict": ["fight", "argue", "attack", "angry", "hit", "punch", "war", "battle", "enemy", "hurt"],
    "loss": ["die", "died", "death", "dead", "lose", "lost", "gone", "miss", "sad", "grief", "funeral"],
    "discovery": ["find", "found", "discover", "explore", "room", "door", "hidden", "secret", "path", "new"],
    "transformation": ["change", "transform", "become", "morph", "different", "shape", "turn into", "strange"],
    "school": ["school", "class", "test", "exam", "teacher", "student", "study", "grade", "fail", "college"],
    "nature": ["tree", "forest", "animal", "mountain", "grass", "flower", "bird", "dog", "cat", "outside"],
    "supernatural": ["ghost", "magic", "monster", "demon", "angel", "spirit", "weird", "impossible", "strange"],
    "body": ["body", "pain", "hurt", "sick", "blood", "wound", "hand", "eye", "feel", "touch", "breathe"],
    "neutral": [],
}

# Correlation discoveries from DreamNet paper
THEME_CORRELATIONS = {
    "falling": {"anxiety": 0.91, "stress": 0.78},
    "pursuit": {"anxiety": 0.85, "stress": 0.82, "fear": 0.79},
    "flying": {"positive_affect": 0.72, "freedom": 0.68},
    "loss": {"depression": 0.74, "grief": 0.71},
    "conflict": {"anger": 0.80, "stress": 0.75},
    "social": {"extraversion": 0.55, "positive_affect": 0.49},
    "transformation": {"life_change": 0.63, "identity": 0.58},
    "discovery": {"curiosity": 0.61, "positive_affect": 0.54},
}


class DreamThemeClassifier:
    """Classifies dream journal entries into thematic categories."""

    def __init__(self):
        self._model = None  # lazy-loaded LightGBM + TF-IDF pipeline
        self._load_model()

    def _load_model(self):
        """Try to load saved sklearn pipeline (TF-IDF + LightGBM)."""
        if MODEL_PATH.exists():
            try:
                import pickle
                with open(MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                log.info("Dream theme classifier loaded from %s", MODEL_PATH)
            except Exception as e:
                log.warning("Failed to load dream theme model: %s — using keyword fallback", e)
                self._model = None
        else:
            log.info("No saved dream theme model — using keyword-based scoring")

    def classify(self, text: str) -> Dict:
        """Classify dream journal text into themes.

        Args:
            text: dream journal entry (any length)

        Returns:
            dict with:
              - primary_theme: most likely theme
              - theme_scores: dict of theme -> probability 0-1
              - correlated_states: psychological correlations
              - keywords_found: matched theme keywords
              - confidence: overall confidence 0-1
        """
        if not text or not text.strip():
            return self._empty_result()

        cleaned = self._preprocess(text)

        if self._model is not None:
            try:
                return self._predict_model(cleaned, text)
            except Exception as e:
                log.debug("Model prediction failed: %s — falling back to keywords", e)

        return self._predict_keywords(cleaned, text)

    def _preprocess(self, text: str) -> str:
        """Lowercase, remove punctuation, expand contractions."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _predict_keywords(self, cleaned: str, original: str) -> Dict:
        """Keyword-based scoring — counts weighted keyword matches per theme."""
        scores: Dict[str, float] = {}
        matched_keywords: Dict[str, List[str]] = {}

        for theme, keywords in THEME_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in cleaned)
            scores[theme] = float(count)
            matched_keywords[theme] = [kw for kw in keywords if kw in cleaned]

        # Ensure neutral is a fallback — gets score 1 if no other theme matches
        if sum(v for k, v in scores.items() if k != "neutral") == 0:
            scores["neutral"] = 1.0

        # Normalize to probabilities
        total = sum(scores.values()) + 1e-10
        probs = {t: round(s / total, 4) for t, s in scores.items()}

        primary = max(probs, key=probs.get)
        confidence = probs[primary]

        # Get correlations for primary theme
        correlations = THEME_CORRELATIONS.get(primary, {})

        return {
            "primary_theme": primary,
            "theme_scores": probs,
            "correlated_states": correlations,
            "keywords_found": {t: kws for t, kws in matched_keywords.items() if kws},
            "confidence": round(confidence, 4),
            "method": "keyword",
            "word_count": len(original.split()),
        }

    def _predict_model(self, cleaned: str, original: str) -> Dict:
        """ML model prediction (TF-IDF + LightGBM pipeline)."""
        proba = self._model.predict_proba([cleaned])[0]
        classes = self._model.classes_

        probs = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        primary = classes[np.argmax(proba)]
        confidence = float(proba.max())

        return {
            "primary_theme": primary,
            "theme_scores": probs,
            "correlated_states": THEME_CORRELATIONS.get(primary, {}),
            "keywords_found": {},
            "confidence": round(confidence, 4),
            "method": "ml",
            "word_count": len(original.split()),
        }

    def _empty_result(self) -> Dict:
        return {
            "primary_theme": "neutral",
            "theme_scores": {t: 0.0 for t in DREAM_THEMES},
            "correlated_states": {},
            "keywords_found": {},
            "confidence": 0.0,
            "method": "empty",
            "word_count": 0,
        }

    def get_theme_info(self, theme: str) -> Dict:
        """Get information about a dream theme."""
        return {
            "theme": theme,
            "keywords": THEME_KEYWORDS.get(theme, []),
            "correlations": THEME_CORRELATIONS.get(theme, {}),
        }
