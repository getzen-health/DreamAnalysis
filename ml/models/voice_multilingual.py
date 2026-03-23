"""Multilingual voice emotion detection with SenseVoice integration.

Documents the integration path for FunAudioLLM's SenseVoice model,
which supports 50+ languages with emotion detection, speech recognition,
and audio event classification in a single model.

SenseVoice (Alibaba, 2024):
    - Model: FunAudioLLM/SenseVoiceSmall (funasr library)
    - Languages: 50+ (Mandarin, English, Cantonese, Japanese, Korean, etc.)
    - Output: ASR text + emotion tags (<|HAPPY|>, <|SAD|>, <|ANGRY|>, <|NEUTRAL|>)
    - Size: ~234MB (SenseVoiceSmall)
    - Inference: ~100ms per utterance on CPU

Integration status:
    - Language detection: stub (uses simple heuristic, full integration
      requires funasr or langid library)
    - Cultural context: implemented (reuses multilingual_emotion.py profiles)
    - SenseVoice model loading: stub (requires funasr pip install + download)

Requirements:
    pip install funasr  # For full SenseVoice integration
    # Model auto-downloads on first use (~234MB)

Reference:
    FunAudioLLM/SenseVoice: https://github.com/FunAudioLLM/SenseVoice
    An, Z. et al. (2024) "FunAudioLLM: Voice Understanding and Generation
    Foundation Models for Natural Interaction Between Humans and LLMs"
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Supported language codes (ISO 639-1) with SenseVoice coverage
SUPPORTED_LANGUAGES: List[str] = [
    "zh", "en", "yue", "ja", "ko",  # Primary SenseVoice languages
    "fr", "de", "es", "it", "pt",   # European languages
    "ru", "ar", "hi", "th", "vi",   # Other languages
    "te", "ta", "bn", "ml", "kn",   # Indian languages
]

# Cultural expression profiles — how emotions manifest in voice across cultures
# Based on 7-country study (2024, n=5,900): collectivist cultures express
# emotions less vocally, especially negative emotions.
CULTURAL_VOICE_PROFILES: Dict[str, Dict] = {
    "ja": {
        "group": "collectivist",
        "expression_intensity": 0.65,
        "description": "Japanese speakers use subtle pitch modulation; "
                       "anger often expressed through silence rather than volume",
        "vocal_cues": {
            "anger": "lowered volume, clipped speech, formality increase",
            "sadness": "breathy voice, slower pace, trailing sentences",
            "happiness": "raised pitch, faster pace, particle usage (ne, yo)",
        },
    },
    "zh": {
        "group": "collectivist",
        "expression_intensity": 0.70,
        "description": "Mandarin tonal system complicates F0-based emotion detection; "
                       "cultural restraint in formal contexts",
        "vocal_cues": {
            "anger": "tone flattening, increased volume, shorter utterances",
            "sadness": "lower register, slower speech rate",
            "happiness": "wider pitch range, faster tempo",
        },
    },
    "ko": {
        "group": "collectivist",
        "expression_intensity": 0.70,
        "description": "Korean honorific levels affect vocal expression; "
                       "age/status differences modulate emotional display",
        "vocal_cues": {
            "anger": "formal speech level shift, volume increase",
            "sadness": "nasalized quality, slower pace",
            "happiness": "raised pitch, aegyo (cute speech) in informal",
        },
    },
    "hi": {
        "group": "collectivist",
        "expression_intensity": 0.75,
        "description": "Hindi speakers show more vocal expressiveness than East Asian "
                       "collectivist cultures but less than Western individualist",
        "vocal_cues": {
            "anger": "increased volume and speech rate",
            "sadness": "lower pitch, slower pace, breathy quality",
            "happiness": "wider pitch range, faster tempo, laughter",
        },
    },
    "te": {
        "group": "collectivist",
        "expression_intensity": 0.75,
        "description": "Telugu speakers show similar patterns to Hindi; "
                       "regional variation between Telangana and Andhra dialects",
        "vocal_cues": {
            "anger": "increased volume, emphatic consonants",
            "sadness": "lower register, elongated vowels",
            "happiness": "higher pitch, faster pace",
        },
    },
    "en": {
        "group": "individualist",
        "expression_intensity": 1.0,
        "description": "English (baseline) — most emotion recognition models "
                       "are trained on English data",
        "vocal_cues": {
            "anger": "increased volume, higher pitch, faster rate",
            "sadness": "lower pitch, slower rate, breathy quality",
            "happiness": "higher pitch, wider range, faster rate",
        },
    },
    "ar": {
        "group": "collectivist",
        "expression_intensity": 0.85,
        "description": "Arabic speakers show high vocal expressiveness in social "
                       "contexts but restraint in formal/religious settings",
        "vocal_cues": {
            "anger": "emphatic consonants, increased volume",
            "sadness": "nasalized quality, slower pace",
            "happiness": "wider pitch range, animated gestures",
        },
    },
}


def detect_language(audio: np.ndarray, sr: int = 16000) -> str:
    """Detect the spoken language from audio.

    Stub implementation: returns "en" as default.
    Full implementation requires funasr or langid library.

    Args:
        audio: 1-D float32 waveform (mono).
        sr: Sample rate in Hz.

    Returns:
        ISO 639-1 language code (e.g., "en", "ja", "zh").
    """
    # TODO: Integrate SenseVoice language detection
    # Full implementation:
    #   from funasr import AutoModel
    #   model = AutoModel(model="FunAudioLLM/SenseVoiceSmall")
    #   result = model.generate(input=audio, language="auto")
    #   detected_lang = result[0]["language"]
    log.info("Language detection stub called — returning 'en' (default)")
    return "en"


def get_cultural_profile(language: str) -> Dict:
    """Get the cultural voice expression profile for a language.

    Args:
        language: ISO 639-1 language code.

    Returns:
        Dict with expression_intensity, group, description, vocal_cues.
    """
    lang = language.lower().strip()[:2]
    if lang in CULTURAL_VOICE_PROFILES:
        return CULTURAL_VOICE_PROFILES[lang]

    # Default: moderate profile for unknown languages
    return {
        "group": "unknown",
        "expression_intensity": 0.85,
        "description": f"No specific profile for language '{lang}'; using moderate defaults",
        "vocal_cues": {},
    }


def adjust_for_culture(
    predictions: Dict[str, float],
    language: str,
) -> Dict[str, float]:
    """Adjust emotion predictions based on cultural context.

    Cultures with lower vocal expression intensity may have genuine
    emotions under-detected. This function amplifies predictions for
    low-expression cultures and attenuates for high-expression cultures.

    Args:
        predictions: Dict of emotion -> probability (e.g., {"happy": 0.3, ...}).
        language: ISO 639-1 language code of the speaker.

    Returns:
        Adjusted predictions (re-normalized to sum to 1.0).
    """
    profile = get_cultural_profile(language)
    intensity = profile["expression_intensity"]

    if intensity >= 1.0:
        return predictions  # No adjustment needed for baseline culture

    adjusted = {}
    for emotion, prob in predictions.items():
        if emotion == "neutral":
            # Reduce neutral boost for suppressive cultures
            # (what appears neutral may actually be mild emotion)
            adjusted[emotion] = prob * intensity
        else:
            # Amplify non-neutral emotions to compensate for suppression
            boost = 1.0 + (1.0 - intensity) * 0.5
            adjusted[emotion] = prob * boost

    # Re-normalize to sum to 1.0
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted


class SenseVoiceMultilingual:
    """Multilingual voice emotion detection using SenseVoice.

    Wraps FunAudioLLM's SenseVoice model with cultural calibration.
    Falls back gracefully if funasr is not installed.
    """

    def __init__(self):
        self._model = None
        self._available = False
        self._try_load()

    def _try_load(self):
        """Attempt to load SenseVoice model. Fail gracefully."""
        try:
            from funasr import AutoModel  # type: ignore
            self._model = AutoModel(model="FunAudioLLM/SenseVoiceSmall")
            self._available = True
            log.info("SenseVoice model loaded successfully")
        except ImportError:
            log.info(
                "funasr not installed — SenseVoice unavailable. "
                "Install with: pip install funasr"
            )
        except Exception as e:
            log.warning("SenseVoice model failed to load: %s", e)

    @property
    def available(self) -> bool:
        return self._available

    def predict(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        language: Optional[str] = None,
    ) -> Optional[Dict]:
        """Predict emotion from audio with cultural calibration.

        Args:
            audio: 1-D float32 waveform (mono).
            sr: Sample rate in Hz.
            language: ISO 639-1 code. If None, auto-detect.

        Returns:
            Dict with emotion, probabilities, language, culture_group,
            adjustments_applied. Returns None if model unavailable.
        """
        if not self._available:
            return None

        if audio is None or len(audio) < sr:
            return None

        # Detect language if not provided
        lang = language or detect_language(audio, sr)

        # Run SenseVoice inference
        # TODO: Replace with actual model inference
        # result = self._model.generate(input=audio, language=lang)
        # raw_emotion = parse_emotion_tag(result)
        # raw_probs = extract_probabilities(result)

        # Placeholder: return None until model is integrated
        log.info(
            "SenseVoice predict called (language=%s) — "
            "full inference not yet integrated", lang,
        )
        return None

    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return SUPPORTED_LANGUAGES.copy()

    def get_cultural_context(self, language: str) -> Dict:
        """Get cultural context for a specific language.

        Useful for UI display of cultural notes.
        """
        profile = get_cultural_profile(language)
        return {
            "language": language,
            "culture_group": profile["group"],
            "expression_intensity": profile["expression_intensity"],
            "description": profile["description"],
            "vocal_cues": profile.get("vocal_cues", {}),
        }
