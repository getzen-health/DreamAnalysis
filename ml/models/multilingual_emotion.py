"""Cross-cultural voice emotion recognition with culture-aware calibration.

Wraps the existing emotion2vec / SenseVoice pipeline with a cultural
calibration layer that adjusts predictions based on detected or specified
language/culture, accounting for display rules and expression intensity
differences.

Key findings (2024-2025):
- English-trained models drop 17-23 points on East Asian/African samples
- Cross-lingual accuracy degradation: 15-25 points (Han et al. 2024)
- Suppression is adaptive in collectivist cultures but maladaptive in
  individualist cultures (7-country study, n=5,900, 2024)
- In-group emotion recognition advantage: +9.3 percentage points
  (Elfenbein & Ambady meta-analysis)

References:
    emotion2vec (ACL 2024) — 71.8% WA on IEMOCAP
    EmoBox (INTERSPEECH 2024) — 32 datasets, 14 languages
    SenseVoice (Alibaba 2024) — multilingual voice understanding
    7-country study (2024) — collectivist vs individualist suppression
"""

import logging
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# Language-to-culture-group mapping
_COLLECTIVIST_LANGS = {"ja", "zh", "ko", "ar", "hi", "th", "vi", "id", "ms"}
_INDIVIDUALIST_LANGS = {"en", "de", "fr", "es", "it", "pt", "nl", "sv", "da", "no"}
_MIXED_LANGS = {"ru", "tr", "pl", "ro", "hu"}

# Culture-specific calibration parameters
_CULTURE_PROFILES = {
    "collectivist": {
        "expression_intensity_scale": 0.7,  # restrained expression norm
        "suppression_penalty": 0.0,         # suppression is adaptive
        "neutral_prior_boost": 0.15,        # higher neutral baseline
        "anger_suppression": 0.3,           # anger expressed less openly
        "description": "Emotional restraint valued; social harmony prioritized",
    },
    "individualist": {
        "expression_intensity_scale": 1.0,  # full expression norm
        "suppression_penalty": 0.2,         # suppression = maladaptive
        "neutral_prior_boost": 0.0,
        "anger_suppression": 0.0,
        "description": "Open emotional expression valued; authenticity prioritized",
    },
    "mixed": {
        "expression_intensity_scale": 0.85,
        "suppression_penalty": 0.1,
        "neutral_prior_boost": 0.07,
        "anger_suppression": 0.15,
        "description": "Mixed cultural expression norms",
    },
}

# Language-agnostic features (most universal across cultures)
LANGUAGE_AGNOSTIC_FEATURES = [
    "f0_mean",
    "f0_std",
    "energy_mean",
    "energy_std",
    "speech_rate_relative",
    "pause_duration_mean",
]


def get_culture_group(language: str) -> str:
    """Map ISO 639-1 language code to cultural group.

    Args:
        language: ISO 639-1 code (e.g., 'en', 'ja', 'zh').

    Returns:
        'collectivist', 'individualist', or 'mixed'.
    """
    lang = language.lower().strip()[:2]
    if lang in _COLLECTIVIST_LANGS:
        return "collectivist"
    if lang in _INDIVIDUALIST_LANGS:
        return "individualist"
    if lang in _MIXED_LANGS:
        return "mixed"
    # Default to individualist (most models trained on Western data)
    return "individualist"


class CulturalCalibrator:
    """Adjusts emotion predictions for cultural context.

    Takes raw emotion probabilities and calibrates them based on the
    cultural group's expected expression patterns.
    """

    EMOTIONS_6 = ["happy", "sad", "angry", "fear", "surprise", "neutral"]

    def calibrate(
        self,
        probabilities: Dict[str, float],
        culture_group: str,
        valence: float = 0.0,
        arousal: float = 0.5,
    ) -> Dict:
        """Apply culture-aware calibration to emotion probabilities.

        Args:
            probabilities: Raw per-emotion probabilities.
            culture_group: 'collectivist', 'individualist', or 'mixed'.
            valence: Raw valence score (-1 to 1).
            arousal: Raw arousal score (0 to 1).

        Returns:
            Dict with calibrated probabilities, valence, arousal,
            detected adjustments, and culture metadata.
        """
        profile = _CULTURE_PROFILES.get(culture_group, _CULTURE_PROFILES["individualist"])

        # Copy probabilities
        cal_probs = {e: probabilities.get(e, 0.0) for e in self.EMOTIONS_6}

        # 1. Boost neutral prior for collectivist cultures
        neutral_boost = profile["neutral_prior_boost"]
        if neutral_boost > 0:
            cal_probs["neutral"] += neutral_boost

        # 2. Suppress anger detection for collectivist cultures
        anger_suppress = profile["anger_suppression"]
        if anger_suppress > 0 and cal_probs.get("angry", 0) > 0:
            redistributed = cal_probs["angry"] * anger_suppress
            cal_probs["angry"] -= redistributed
            # Redistribute to neutral and sad (anger often masked as these)
            cal_probs["neutral"] += redistributed * 0.6
            cal_probs["sad"] += redistributed * 0.4

        # 3. Scale expression intensity
        scale = profile["expression_intensity_scale"]
        if scale < 1.0:
            for e in self.EMOTIONS_6:
                if e != "neutral":
                    deviation = cal_probs[e] - (1.0 / len(self.EMOTIONS_6))
                    cal_probs[e] = (1.0 / len(self.EMOTIONS_6)) + deviation / scale

        # Re-normalize
        total = sum(max(v, 0) for v in cal_probs.values())
        if total > 0:
            cal_probs = {e: max(v, 0) / total for e, v in cal_probs.items()}

        # Calibrate valence/arousal
        cal_valence = valence
        cal_arousal = arousal
        if culture_group == "collectivist":
            # Collectivist cultures may underexpress — amplify detected signals
            if valence < 0:
                cal_valence = valence * 1.2  # amplify negative signals
                cal_valence = max(cal_valence, -1.0)
            # Arousal may be suppressed
            cal_arousal = min(arousal * 1.15, 1.0)

        # Determine top emotion
        top_emotion = max(cal_probs, key=cal_probs.get)

        adjustments = []
        if neutral_boost > 0:
            adjustments.append("neutral_prior_boosted")
        if anger_suppress > 0:
            adjustments.append("anger_expression_adjusted")
        if scale < 1.0:
            adjustments.append("intensity_rescaled")

        return {
            "emotion": top_emotion,
            "probabilities": {e: round(v, 4) for e, v in cal_probs.items()},
            "valence": round(float(cal_valence), 4),
            "arousal": round(float(cal_arousal), 4),
            "culture_group": culture_group,
            "culture_description": profile["description"],
            "adjustments_applied": adjustments,
            "calibrated": True,
        }


class CulturalEIAdapter:
    """Adjust EI scoring for cultural context.

    Bar-On EQ-i validated in 30+ languages but British adults score higher
    than Chinese on global trait EI. This adapter normalizes EI subscores
    against culture-specific norms rather than global population norms.
    """

    # Approximate normative adjustments (from TEIQue meta-analysis, N=67,734)
    _NORM_OFFSETS = {
        "collectivist": {
            "self_perception": -3.0,      # lower self-assessed scores
            "self_expression": -5.0,      # suppression is normative
            "interpersonal": +2.0,        # stronger social orientation
            "decision_making": 0.0,
            "stress_management": +1.0,    # better group coping
        },
        "individualist": {
            "self_perception": 0.0,
            "self_expression": 0.0,
            "interpersonal": 0.0,
            "decision_making": 0.0,
            "stress_management": 0.0,
        },
        "mixed": {
            "self_perception": -1.5,
            "self_expression": -2.5,
            "interpersonal": +1.0,
            "decision_making": 0.0,
            "stress_management": +0.5,
        },
    }

    def adjust_ei_scores(
        self, scores: Dict[str, float], culture_group: str
    ) -> Dict:
        """Adjust EI dimension scores for cultural norms.

        Args:
            scores: Dict with EI dimension names → raw scores (0-100).
            culture_group: Cultural context.

        Returns:
            Dict with adjusted scores and adjustment metadata.
        """
        offsets = self._NORM_OFFSETS.get(culture_group, self._NORM_OFFSETS["individualist"])
        adjusted = {}
        for dim, raw in scores.items():
            offset = offsets.get(dim, 0.0)
            adjusted[dim] = round(float(np.clip(raw + offset, 0, 100)), 1)

        return {
            "adjusted_scores": adjusted,
            "raw_scores": scores,
            "culture_group": culture_group,
            "adjustments": {k: v for k, v in offsets.items() if v != 0},
        }
