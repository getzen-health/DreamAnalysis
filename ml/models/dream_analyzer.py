"""Dream Narrative Analyzer — Phase 1-2 of #287.

Heuristic NLP analysis of dream journal text without an LLM dependency.
Extracts:
  - Emotional content (valence, arousal, intensity)
  - Recurring archetypes / symbols
  - Nightmare severity (0-1)
  - Imagery Rehearsal Therapy (IRT) recommendation when nightmares detected
  - Lucid dreaming probability markers in text
  - Correlations with mood state for multi-day dashboard

References:
  - DreamNet 2025 (arXiv 2503.05778) — 92.1% dream text classification accuracy
  - Northwestern TLR 2024 — audio cues triple lucid dream frequency
  - IRT meta-analysis — d=0.85-1.24 for nightmare reduction
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ── Emotion keyword lexicons ──────────────────────────────────────────────────

_POSITIVE_WORDS = {
    "joy", "happy", "happiness", "love", "beautiful", "peaceful", "calm",
    "warm", "light", "bright", "wonderful", "amazing", "safe", "comfort",
    "serene", "playful", "laugh", "laughter", "delight", "joyful", "bliss",
    "excited", "elated", "grateful", "hope", "flying", "free", "soar",
    "dance", "hug", "embrace", "reunion", "celebrate",
}

_NEGATIVE_WORDS = {
    "fear", "scared", "terror", "nightmare", "dark", "shadow", "danger",
    "attack", "chase", "chased", "run", "running", "fall", "falling",
    "death", "dying", "dead", "monster", "threat", "hurt", "pain", "cry",
    "crying", "trapped", "stuck", "lost", "alone", "abandoned", "angry",
    "rage", "violent", "violence", "blood", "wound", "injury", "disaster",
    "flood", "fire", "crash", "drown", "sink", "suffocate", "scream",
    "anxious", "anxiety", "panic", "helpless", "hopeless", "grief",
}

_HIGH_AROUSAL_WORDS = {
    "chase", "run", "scream", "fight", "attack", "rush", "panic", "terror",
    "explode", "crash", "storm", "fly", "soar", "race", "excitement",
    "wild", "intense", "vivid", "electricity", "surge",
}

_LOW_AROUSAL_WORDS = {
    "calm", "peaceful", "quiet", "still", "slow", "gentle", "drift",
    "float", "sleep", "rest", "hazy", "foggy", "dim", "soft", "faint",
}

# ── Archetype / symbol lexicon ────────────────────────────────────────────────

_ARCHETYPES: Dict[str, List[str]] = {
    "shadow":        ["shadow", "dark figure", "stranger", "monster", "enemy", "villain"],
    "anima/animus":  ["lover", "partner", "mysterious figure", "romantic", "beautiful woman", "handsome man"],
    "self":          ["mirror", "reflection", "twin", "double", "my self", "myself"],
    "trickster":     ["clown", "fool", "joker", "shapeshifter", "deceive", "trick"],
    "hero":          ["hero", "warrior", "rescue", "save", "champion", "mission"],
    "wise elder":    ["teacher", "guide", "mentor", "grandmother", "grandfather", "wise"],
    "journey":       ["road", "path", "bridge", "door", "gate", "travel", "journey", "map"],
    "transformation":["change", "transform", "metamorphosis", "butterfly", "shed", "rebirth"],
    "water":         ["ocean", "sea", "river", "lake", "flood", "rain", "swim", "drown"],
    "home":          ["house", "home", "room", "childhood", "family", "mother", "father"],
    "pursuit":       ["chase", "chased", "run from", "escape", "flee", "caught"],
    "falling":       ["fall", "falling", "plunge", "drop", "abyss"],
    "flying":        ["fly", "flying", "soar", "float", "levitate", "wings"],
    "teeth":         ["teeth", "tooth", "dental", "crumble", "fall out"],
    "exam/test":     ["exam", "test", "school", "late", "unprepared", "fail", "class"],
}

# ── Nightmare markers ─────────────────────────────────────────────────────────

_NIGHTMARE_WORDS = {
    "nightmare", "terror", "horrified", "horrifying", "monster", "demon",
    "attack", "murder", "killed", "drown", "suffocate", "trapped", "paralyzed",
    "evil", "scream", "helpless", "death", "dying", "blood", "violence",
}

_NIGHTMARE_INTENSIFIERS = {"very", "extremely", "totally", "completely", "terrifying"}

# ── Lucid dream markers ───────────────────────────────────────────────────────

_LUCID_MARKERS = {
    "realized i was dreaming", "knew i was dreaming", "lucid", "conscious in the dream",
    "aware i was dreaming", "woke up within", "controlled the dream", "became aware",
}


# ── IRT protocol ──────────────────────────────────────────────────────────────

IRT_PROTOCOL = {
    "description": (
        "Imagery Rehearsal Therapy (IRT) for nightmare reduction. "
        "Evidence: d=0.85-1.24 (nightmare frequency), d=1.00-1.53 for PTSD symptoms."
    ),
    "steps": [
        "Write down the nightmare in detail.",
        "Choose a turning point — identify the moment the dream became distressing.",
        "Rewrite the nightmare with a new, positive ending of your choice.",
        "Rehearse the new version mentally for 10-20 minutes before sleep.",
        "Practice the new narrative every night for at least 3 weeks.",
        "Track nightmare frequency weekly — expect 50-80% reduction within 4-6 weeks.",
    ],
    "reference": "Krakow & Zadra (2006), Imagery Rehearsal Therapy meta-analysis 2023",
}


# ── Main analyzer ─────────────────────────────────────────────────────────────

class DreamAnalyzer:
    """Analyzes dream journal text using heuristic NLP."""

    def analyze(self, text: str) -> Dict[str, Any]:
        """Full dream analysis pipeline.

        Parameters
        ----------
        text : str  Dream journal entry (free text, any length).

        Returns
        -------
        dict with keys:
          emotional_valence   (-1 to 1)
          emotional_arousal   (0 to 1)
          emotional_intensity (0 to 1)
          nightmare_score     (0 to 1)
          is_nightmare        bool
          archetypes          list of matched archetype names
          emotions_detected   list of {word, polarity}
          lucid_probability   (0 to 1)
          word_count          int
          irt_recommended     bool
          irt_protocol        dict | None
          insights            list[str]
          morning_mood_prediction  str
        """
        if not text or not text.strip():
            return self._empty_result()

        words = self._tokenize(text)
        word_set = set(words)
        lower = text.lower()

        valence, arousal, intensity, pos_count, neg_count = self._compute_affect(words, word_set)
        nightmare_score = self._nightmare_score(words, word_set, lower, intensity)
        archetypes = self._detect_archetypes(lower)
        lucid_prob = self._lucid_probability(lower)
        emotions = self._list_emotions(words, word_set)
        insights = self._generate_insights(
            valence, arousal, nightmare_score, archetypes, lucid_prob, pos_count, neg_count
        )
        morning_mood = self._predict_morning_mood(valence, nightmare_score, lucid_prob)
        irt_recommended = nightmare_score >= 0.4

        return {
            "emotional_valence": round(valence, 3),
            "emotional_arousal": round(arousal, 3),
            "emotional_intensity": round(intensity, 3),
            "nightmare_score": round(nightmare_score, 3),
            "is_nightmare": nightmare_score >= 0.5,
            "archetypes": archetypes,
            "emotions_detected": emotions,
            "lucid_probability": round(lucid_prob, 3),
            "word_count": len(words),
            "irt_recommended": irt_recommended,
            "irt_protocol": IRT_PROTOCOL if irt_recommended else None,
            "insights": insights,
            "morning_mood_prediction": morning_mood,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z]+", text.lower())

    @staticmethod
    def _compute_affect(
        words: List[str], word_set: set
    ) -> Tuple[float, float, float, int, int]:
        pos = sum(1 for w in words if w in _POSITIVE_WORDS)
        neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
        total = pos + neg or 1
        valence = (pos - neg) / total
        valence = max(-1.0, min(1.0, valence))

        high_a = sum(1 for w in words if w in _HIGH_AROUSAL_WORDS)
        low_a = sum(1 for w in words if w in _LOW_AROUSAL_WORDS)
        arousal_raw = (high_a - low_a) / max(high_a + low_a, 1)
        arousal = max(0.0, min(1.0, (arousal_raw + 1) / 2))

        intensity = min(1.0, (pos + neg) / max(len(words) * 0.08, 1))

        return valence, arousal, intensity, pos, neg

    @staticmethod
    def _nightmare_score(
        words: List[str], word_set: set, lower: str, intensity: float
    ) -> float:
        base = sum(1 for w in words if w in _NIGHTMARE_WORDS)
        # Intensifier bonus
        bonus = sum(0.2 for w in words if w in _NIGHTMARE_INTENSIFIERS)
        raw = (base + bonus) / max(len(words) * 0.05, 1)
        return min(1.0, raw * intensity * 3.0)

    @staticmethod
    def _detect_archetypes(lower: str) -> List[str]:
        found = []
        for archetype, keywords in _ARCHETYPES.items():
            for kw in keywords:
                if kw in lower:
                    found.append(archetype)
                    break
        return found

    @staticmethod
    def _lucid_probability(lower: str) -> float:
        for marker in _LUCID_MARKERS:
            if marker in lower:
                return 0.9
        # Softer signals
        soft = sum(1 for w in ["lucid", "aware", "conscious", "control"] if w in lower)
        return min(0.6, soft * 0.2)

    @staticmethod
    def _list_emotions(words: List[str], word_set: set) -> List[Dict[str, str]]:
        results = []
        seen: set = set()
        for w in words:
            if w in seen:
                continue
            if w in _POSITIVE_WORDS:
                results.append({"word": w, "polarity": "positive"})
                seen.add(w)
            elif w in _NEGATIVE_WORDS:
                results.append({"word": w, "polarity": "negative"})
                seen.add(w)
        return results[:20]

    @staticmethod
    def _generate_insights(
        valence: float,
        arousal: float,
        nightmare_score: float,
        archetypes: List[str],
        lucid_prob: float,
        pos: int,
        neg: int,
    ) -> List[str]:
        insights = []

        if nightmare_score >= 0.5:
            insights.append(
                "This dream contains nightmare patterns. IRT (Imagery Rehearsal Therapy) "
                "can reduce nightmare frequency by 50-80% within 4-6 weeks."
            )
        elif nightmare_score >= 0.25:
            insights.append("Mild distress detected. Journaling about this dream may reduce its recurrence.")

        if valence > 0.3:
            insights.append("Emotionally positive dream — often linked to next-day mood lift and creative problem-solving.")
        elif valence < -0.3:
            insights.append("Negative emotional tone detected. Negative dream reappraisal (reframing as information) can improve morning mood.")

        if arousal > 0.65:
            insights.append("High arousal dream — vivid and emotionally intense dreams often reflect daytime stress or REM pressure.")

        if lucid_prob >= 0.7:
            insights.append("Lucid dreaming episode detected! Audio reality-checking cues during REM can triple lucid dream frequency (Northwestern TLR 2024).")

        if "pursuit" in archetypes:
            insights.append("Chase/pursuit archetype: often reflects avoidance of an emotion or situation in waking life.")
        if "flying" in archetypes:
            insights.append("Flying archetype: associated with high sense of agency and creative expansion.")
        if "falling" in archetypes:
            insights.append("Falling archetype: commonly linked to anxiety, insecurity, or a sudden loss of control.")
        if "transformation" in archetypes:
            insights.append("Transformation archetype: suggests psychological growth or a life transition underway.")
        if "teeth" in archetypes:
            insights.append("Teeth archetype: one of the most universal dream symbols — linked to communication anxiety or concerns about appearance.")

        if not insights:
            insights.append("Neutral dream content captured. Patterns emerge after 7+ journal entries.")

        return insights

    @staticmethod
    def _predict_morning_mood(
        valence: float, nightmare_score: float, lucid_prob: float
    ) -> str:
        score = valence - nightmare_score * 0.6 + lucid_prob * 0.3
        if score > 0.4:
            return "positive"
        elif score > 0.0:
            return "slightly positive"
        elif score > -0.3:
            return "neutral"
        elif score > -0.6:
            return "slightly negative"
        else:
            return "negative"

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "emotional_valence": 0.0,
            "emotional_arousal": 0.5,
            "emotional_intensity": 0.0,
            "nightmare_score": 0.0,
            "is_nightmare": False,
            "archetypes": [],
            "emotions_detected": [],
            "lucid_probability": 0.0,
            "word_count": 0,
            "irt_recommended": False,
            "irt_protocol": None,
            "insights": ["Enter your dream journal to see analysis."],
            "morning_mood_prediction": "neutral",
        }


# Module-level singleton
_analyzer: Optional[DreamAnalyzer] = None


def get_dream_analyzer() -> DreamAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = DreamAnalyzer()
    return _analyzer
