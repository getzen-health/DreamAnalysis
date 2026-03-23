"""Wrapper for emotion2vec speech emotion recognition model.

emotion2vec (ACL 2024, funaudiollm) is a universal speech emotion
representation model that achieves state-of-the-art performance across
multiple emotion recognition benchmarks.

Model variants:
    - emotion2vec_plus_large (300M params, best accuracy)
      - 71.8% weighted accuracy on IEMOCAP
      - 9-class: angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown
    - emotion2vec_plus_base (faster, lower memory)
    - emotion2vec_base (original, smaller)

Integration path:
    1. Install: pip install funasr modelscope
    2. Model auto-downloads from HuggingFace (~350MB for plus_large)
    3. First call triggers download; subsequent calls use cached model
    4. Lazy loading: model only loads when predict() is first called

Output format (6-class, matching EEG pipeline):
    emotion: "happy"|"sad"|"angry"|"fear"|"surprise"|"neutral"
    probabilities: {emotion: 0.0-1.0, ...}
    valence: -1.0 to 1.0
    arousal: 0.0 to 1.0
    confidence: 0.0 to 1.0
    model_type: "voice_emotion2vec"

Requirements:
    pip install funasr modelscope  # ~350MB model download on first use

Reference:
    Ma, Z. et al. (2024) "emotion2vec: Self-Supervised Pre-Training for
    Speech Emotion Representation" ACL 2024
    https://github.com/ddlBoJack/emotion2vec
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# emotion2vec 9-class output labels
E2V_LABELS: List[str] = [
    "angry", "disgusted", "fearful", "happy", "neutral",
    "other", "sad", "surprised", "unknown",
]

# Mapping from emotion2vec 9-class to our 6-class system
E2V_TO_6CLASS: Dict[str, str] = {
    "angry":     "angry",
    "disgusted": "angry",      # merge disgusted -> angry
    "fearful":   "fear",
    "happy":     "happy",
    "neutral":   "neutral",
    "other":     "neutral",    # merge other -> neutral
    "sad":       "sad",
    "surprised": "surprise",
    "unknown":   "neutral",    # merge unknown -> neutral
}

# 6-class label set
EMOTIONS_6: List[str] = ["happy", "sad", "angry", "fear", "surprise", "neutral"]

# Valence mapping for each emotion (used when only class label available)
_VALENCE_MAP: Dict[str, float] = {
    "happy": 0.7, "sad": -0.6, "angry": -0.5,
    "fear": -0.4, "surprise": 0.2, "neutral": 0.0,
}

# Arousal mapping for each emotion
_AROUSAL_MAP: Dict[str, float] = {
    "happy": 0.6, "sad": 0.2, "angry": 0.8,
    "fear": 0.7, "surprise": 0.7, "neutral": 0.3,
}

_MIN_SAMPLES = 8000  # ~0.5s at 16kHz


class Emotion2vecWrapper:
    """Wrapper for emotion2vec speech emotion recognition.

    Uses lazy loading: model is not downloaded/loaded until predict()
    is first called. This avoids the 350MB download at import time.
    """

    def __init__(self, model_name: str = "iic/emotion2vec_plus_large"):
        self._model_name = model_name
        self._model = None
        self._available: Optional[bool] = None  # None = not yet checked
        self._load_attempted = False

    def _try_load(self):
        """Attempt to load emotion2vec model. Lazy — called on first predict()."""
        if self._load_attempted:
            return
        self._load_attempted = True

        try:
            from funasr import AutoModel  # type: ignore

            log.info("Loading emotion2vec model: %s (this may download ~350MB)", self._model_name)
            self._model = AutoModel(model=self._model_name)
            self._available = True
            log.info("emotion2vec model loaded successfully")
        except ImportError:
            log.info(
                "funasr not installed — emotion2vec unavailable. "
                "Install with: pip install funasr modelscope"
            )
            self._available = False
        except Exception as e:
            log.warning("emotion2vec model failed to load: %s", e)
            self._available = False

    @property
    def available(self) -> bool:
        """Whether the model is loaded and ready for inference."""
        if self._available is None:
            self._try_load()
        return self._available or False

    def predict(
        self,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> Optional[Dict]:
        """Predict emotion from audio waveform.

        Args:
            audio: 1-D float32 waveform (mono).
            sr: Sample rate in Hz.

        Returns:
            Dict with emotion, probabilities, valence, arousal, confidence,
            model_type. Returns None if model unavailable or audio too short.
        """
        if not self.available:
            return None

        if audio is None or len(audio) < _MIN_SAMPLES:
            return None

        try:
            result = self._model.generate(input=audio, output_dir=None, granularity="utterance")
            return self._parse_result(result)
        except Exception as e:
            log.warning("emotion2vec inference failed: %s", e)
            return None

    def _parse_result(self, raw_result) -> Optional[Dict]:
        """Parse raw emotion2vec output into our 6-class format.

        Args:
            raw_result: Raw output from emotion2vec model.

        Returns:
            Standardized emotion dict or None on parse failure.
        """
        try:
            # emotion2vec returns list of dicts with 'labels' and 'scores'
            if not raw_result or not isinstance(raw_result, list):
                return None

            entry = raw_result[0]
            labels = entry.get("labels", E2V_LABELS)
            scores = entry.get("scores", [])

            if not scores:
                return None

            # Build 9-class probability dict
            raw_probs = {}
            for label, score in zip(labels, scores):
                raw_probs[label] = float(score)

            # Merge to 6-class
            probs_6 = {e: 0.0 for e in EMOTIONS_6}
            for label_9, label_6 in E2V_TO_6CLASS.items():
                probs_6[label_6] += raw_probs.get(label_9, 0.0)

            # Normalize
            total = sum(probs_6.values())
            if total > 0:
                probs_6 = {k: v / total for k, v in probs_6.items()}

            # Winner
            emotion = max(probs_6, key=probs_6.get)  # type: ignore
            confidence = probs_6[emotion]

            # Valence and arousal from probability-weighted mapping
            valence = sum(
                probs_6[e] * _VALENCE_MAP[e] for e in EMOTIONS_6
            )
            arousal = sum(
                probs_6[e] * _AROUSAL_MAP[e] for e in EMOTIONS_6
            )

            return {
                "emotion": emotion,
                "probabilities": probs_6,
                "valence": float(np.clip(valence, -1.0, 1.0)),
                "arousal": float(np.clip(arousal, 0.0, 1.0)),
                "confidence": float(confidence),
                "model_type": "voice_emotion2vec",
            }

        except Exception as e:
            log.warning("Failed to parse emotion2vec result: %s", e)
            return None

    @staticmethod
    def map_9class_to_6class(label_9: str) -> str:
        """Map a 9-class emotion2vec label to our 6-class system."""
        return E2V_TO_6CLASS.get(label_9.lower(), "neutral")

    @staticmethod
    def get_model_info() -> Dict:
        """Return model metadata for status endpoints."""
        return {
            "name": "emotion2vec_plus_large",
            "source": "funaudiollm/emotion2vec",
            "paper": "ACL 2024",
            "classes": 9,
            "mapped_classes": 6,
            "size_mb": 350,
            "benchmark_iemocap_wa": 71.8,
            "requires": ["funasr", "modelscope"],
        }
