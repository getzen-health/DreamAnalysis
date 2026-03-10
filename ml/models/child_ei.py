"""Child/Adolescent Emotional Intelligence Scoring — #286.

Age-adapted Bar-On EQ-i:YV (Youth Version) implementation for digital tracking.
Adjusts voice F0 thresholds, EI composite weights, and attention-span-aware
calibration windows for ages 6-17.

Key differences from adult EI:
  - Voice F0 range: 87-437 Hz (children) vs 85-255 Hz (adults)
  - EEG dominant band: theta 3-9 Hz (children) vs alpha 8-12 Hz (adults)
  - Adult-to-child SER transfer: F-score ~0.47 (domain gap)
  - Shorter calibration protocol (2 min vs 5 min)

Age bands:
  young (6-9):   visual/game-based, 4 core emotions, 3-min max sessions
  middle (10-12): reflective check-ins, 6 emotions, 5-min sessions
  teen (13-17):  full EI profile, journal-style, up to 10-min sessions

Evidence:
  - Bar-On EQ-i:YV (2000) — validated for ages 8-18
  - Mightier (HR biofeedback + games): 87% families see improvement in 90 days
  - Child SER with age-matched data: ~83% (4-class)

References:
  - Bar-On & Parker (2000) — Bar-On Emotional Quotient Inventory: Youth Version
  - Mightier clinical outcomes 2023
  - ACM CHI 2024 — child emotion recognition from voice (age normalization)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ── Age band classification ────────────────────────────────────────────────────

def get_age_band(age: int) -> str:
    if age < 6:
        raise ValueError(f"Minimum supported age is 6, got {age}")
    if age <= 9:
        return "young"
    if age <= 12:
        return "middle"
    return "teen"


# ── Emotion vocabulary by age band ────────────────────────────────────────────

EMOTION_VOCABULARY: Dict[str, List[str]] = {
    "young":  ["happy", "sad", "angry", "scared"],
    "middle": ["happy", "sad", "angry", "scared", "surprised", "worried"],
    "teen":   ["happy", "sad", "angry", "fearful", "surprised", "calm", "anxious", "proud"],
}

# ── Bar-On EQ-i:YV 5-factor structure (adapted for digital tracking) ─────────
# Original subscales: Intrapersonal, Interpersonal, Stress Management,
# Adaptability, General Mood
#
# Digital proxies for each subscale:

SUBSCALE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "young": {
        "intrapersonal": 0.20,     # self-awareness (fewer items for younger)
        "interpersonal": 0.30,     # empathy/peer relations (emphasis for young)
        "stress_management": 0.25,
        "adaptability": 0.15,
        "general_mood": 0.10,
    },
    "middle": {
        "intrapersonal": 0.22,
        "interpersonal": 0.25,
        "stress_management": 0.22,
        "adaptability": 0.18,
        "general_mood": 0.13,
    },
    "teen": {
        "intrapersonal": 0.25,
        "interpersonal": 0.22,
        "stress_management": 0.20,
        "adaptability": 0.18,
        "general_mood": 0.15,
    },
}

# ── Voice F0 thresholds (age-adjusted) ────────────────────────────────────────
# Children's F0 is higher and ranges more than adults.
# Source: ACM CHI 2024 child voice emotion recognition paper.

F0_RANGES: Dict[str, Tuple[float, float]] = {
    "young":  (200.0, 437.0),   # Hz  — prepubertal, higher pitch
    "middle": (160.0, 380.0),   # Hz  — early puberty overlap
    "teen":   (87.0, 300.0),    # Hz  — post-pubertal, closer to adult
}

# ── Session length caps (attention span) ─────────────────────────────────────

MAX_SESSION_SECONDS: Dict[str, int] = {
    "young":  180,   # 3 min
    "middle": 300,   # 5 min
    "teen":   600,   # 10 min
}

# ── Calibration protocol (shorter than adult 5 min) ──────────────────────────

CALIBRATION_FRAMES: Dict[str, int] = {
    "young":  15,   # 15 sec resting
    "middle": 20,   # ~20 sec
    "teen":   30,   # same as adult min
}


# ── EI Scorer ─────────────────────────────────────────────────────────────────

class ChildEIScorer:
    """Age-adapted EI composite score for children 6-17."""

    def score(
        self,
        age: int,
        valence: float,
        arousal: float,
        stress_index: float,
        focus_index: float,
        emotion: str,
        session_count: int = 1,
        peer_interaction_rating: Optional[float] = None,   # 0-1 parent-reported
        emotion_labeling_accuracy: Optional[float] = None,  # 0-1 from game
    ) -> Dict[str, Any]:
        """Compute age-adapted EI composite score.

        Parameters
        ----------
        age : int                        Child's age in years (6-17)
        valence : float                  Voice/EEG valence -1 to 1
        arousal : float                  Arousal 0-1
        stress_index : float             Stress 0-1
        focus_index : float              Focus 0-1
        emotion : str                    Detected emotion label
        session_count : int              Number of completed sessions
        peer_interaction_rating : float  Parent-reported 0-1 (optional)
        emotion_labeling_accuracy : float Game accuracy 0-1 (optional)

        Returns
        -------
        dict with composite_score, subscales, band, age_tips, badges
        """
        band = get_age_band(age)
        weights = SUBSCALE_WEIGHTS[band]

        # ── Subscale proxies ──────────────────────────────────────────────────
        # Intrapersonal: ability to identify own emotions
        intrapersonal = self._intrapersonal(valence, emotion, emotion_labeling_accuracy)

        # Interpersonal: empathy + peer relations (needs parent input)
        interpersonal = self._interpersonal(valence, peer_interaction_rating)

        # Stress management: inverse of stress + adaptability signals
        stress_mgmt = self._stress_management(stress_index, arousal)

        # Adaptability: focus + recovery across sessions
        adaptability = self._adaptability(focus_index, session_count)

        # General mood: overall positive affect
        general_mood = max(0.0, min(1.0, (valence + 1) / 2))

        subscales = {
            "intrapersonal": round(intrapersonal * 100, 1),
            "interpersonal": round(interpersonal * 100, 1),
            "stress_management": round(stress_mgmt * 100, 1),
            "adaptability": round(adaptability * 100, 1),
            "general_mood": round(general_mood * 100, 1),
        }

        composite = (
            intrapersonal * weights["intrapersonal"]
            + interpersonal * weights["interpersonal"]
            + stress_mgmt * weights["stress_management"]
            + adaptability * weights["adaptability"]
            + general_mood * weights["general_mood"]
        )
        composite_score = round(composite * 100, 1)

        badges = self._compute_badges(band, composite, subscales, session_count)
        tips = self._age_tips(band, subscales)

        return {
            "composite_score": composite_score,
            "subscales": subscales,
            "age_band": band,
            "age": age,
            "emotions_available": EMOTION_VOCABULARY[band],
            "max_session_seconds": MAX_SESSION_SECONDS[band],
            "calibration_frames_needed": CALIBRATION_FRAMES[band],
            "badges": badges,
            "tips": tips,
            "f0_range_hz": {"min": F0_RANGES[band][0], "max": F0_RANGES[band][1]},
        }

    # ── Subscale helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _intrapersonal(valence: float, emotion: str, accuracy: Optional[float]) -> float:
        base = (valence + 1) / 2 * 0.5
        if accuracy is not None:
            return min(1.0, base + accuracy * 0.5)
        # Without game data, use valence as partial proxy
        return min(1.0, base + 0.3)

    @staticmethod
    def _interpersonal(valence: float, peer_rating: Optional[float]) -> float:
        if peer_rating is not None:
            return min(1.0, 0.4 * max(0.0, (valence + 1) / 2) + 0.6 * peer_rating)
        return min(1.0, 0.5 + valence * 0.3)

    @staticmethod
    def _stress_management(stress: float, arousal: float) -> float:
        inv_stress = 1.0 - stress
        calm_arousal = 1.0 - abs(arousal - 0.4)   # optimal ~0.4
        return min(1.0, 0.6 * inv_stress + 0.4 * calm_arousal)

    @staticmethod
    def _adaptability(focus: float, sessions: int) -> float:
        session_bonus = min(0.2, sessions * 0.02)
        return min(1.0, 0.8 * focus + session_bonus)

    @staticmethod
    def _compute_badges(
        band: str, composite: float, subscales: Dict[str, float], sessions: int
    ) -> List[str]:
        badges = []
        if sessions >= 5:
            badges.append("5-day streak")
        if sessions >= 10:
            badges.append("EI Explorer")
        if sessions >= 30:
            badges.append("Emotion Master")
        if composite >= 0.75:
            badges.append("High EI")
        if subscales.get("stress_management", 0) >= 80:
            badges.append("Calm Champion")
        if subscales.get("interpersonal", 0) >= 80:
            badges.append("Empathy Star")
        if band == "young" and composite >= 0.6:
            badges.append("Feeling Smart!")
        return badges

    @staticmethod
    def _age_tips(band: str, subscales: Dict[str, float]) -> List[str]:
        tips = []
        sm = subscales.get("stress_management", 50)
        inter = subscales.get("interpersonal", 50)
        intra = subscales.get("intrapersonal", 50)

        if band == "young":
            if sm < 50:
                tips.append("Try the breathing bubble game to calm down when feeling upset.")
            if intra < 50:
                tips.append("Play the emotion matching game to practice naming your feelings.")
            if inter < 50:
                tips.append("Think of one kind thing you can do for a friend today.")
        elif band == "middle":
            if sm < 50:
                tips.append("4-7-8 breathing (in 4 counts, hold 7, out 8) helps with stress.")
            if intra < 50:
                tips.append("Try the emotion journal — write 3 feelings you noticed today.")
            if inter < 50:
                tips.append("Active listening practice: repeat back what a friend says before responding.")
        else:  # teen
            if sm < 50:
                tips.append("Cognitive reappraisal: when stressed, ask 'what would I tell a friend in this situation?'")
            if intra < 50:
                tips.append("Emotion granularity: try to name specific feelings (not just 'bad'). Are you anxious, disappointed, or overwhelmed?")
            if inter < 50:
                tips.append("Perspective-taking exercise: imagine the situation from the other person's point of view.")

        if not tips:
            tips.append("Keep up the great emotional awareness work!")

        return tips


_scorer: Optional[ChildEIScorer] = None


def get_child_ei_scorer() -> ChildEIScorer:
    global _scorer
    if _scorer is None:
        _scorer = ChildEIScorer()
    return _scorer
