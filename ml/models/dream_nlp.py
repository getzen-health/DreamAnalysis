"""DreamNLP — unified NLP analysis of dream journal text.

Implements issue #48: DreamNet NLP for dream journal analysis.

Combines theme classification (Hall/Van de Castle taxonomy) with sentiment
analysis, symbol extraction, and lucidity detection — all without heavy
transformer dependencies. Uses TF-IDF keyword scoring, regex pattern
matching, and lexicon-based affect analysis.

References:
  - Hall & Van de Castle (1966): Content Analysis of Dreams
  - DreamBank coding system: 15+ validated theme categories
  - Domhoff (2003): The Scientific Study of Dreams
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


# ── Symbol lexicon (Hall/Van de Castle + common Jungian symbols) ──────────────

DREAM_SYMBOLS: Dict[str, Dict[str, Any]] = {
    # ── Motion / action symbols ──────────────────────────────────────────────
    "flying": {
        "keywords": [r"\bfl(y|ew|ying)\b", r"\bsoar(ing)?\b", r"\blevitat\w*\b",
                     r"\bglid(e|ing)\b", r"\bhovering\b", r"\bairborne\b"],
        "meaning": "autonomy, freedom, transcendence of limits",
        "valence": "positive",
        "category": "motion",
    },
    "falling": {
        "keywords": [r"\bfall(ing|en|s)?\b", r"\bdropp?(ing|ed)?\b", r"\bplunge\w*\b",
                     r"\btumbl\w*\b", r"\bprecipice\b", r"\babyss\b"],
        "meaning": "loss of control, anxiety, insecurity",
        "valence": "negative",
        "category": "motion",
    },
    "running": {
        "keywords": [r"\brunning\b", r"\bran\b", r"\bchase[sd]?\b", r"\bflee(ing)?\b",
                     r"\bescap\w*\b", r"\bpursuit\b"],
        "meaning": "avoidance, urgency, stress response",
        "valence": "negative",
        "category": "motion",
    },
    "swimming": {
        "keywords": [r"\bswim(ming|s)?\b", r"\bswam\b", r"\bfloat(ing)?\b", r"\bdrift(ing)?\b"],
        "meaning": "navigating emotions, resilience",
        "valence": "neutral",
        "category": "motion",
    },
    # ── People / relationship symbols ────────────────────────────────────────
    "stranger": {
        "keywords": [r"\bstranger\b", r"\bunknown person\b", r"\bdark figure\b",
                     r"\bshadow figure\b", r"\bmysterious (person|figure|man|woman)\b"],
        "meaning": "the shadow self, unknown aspects of psyche",
        "valence": "neutral",
        "category": "people",
    },
    "mother": {
        "keywords": [r"\bmother\b", r"\bmom\b", r"\bmama\b", r"\bmaternal\b"],
        "meaning": "nurturing, protection, origins",
        "valence": "neutral",
        "category": "people",
    },
    "father": {
        "keywords": [r"\bfather\b", r"\bdad\b", r"\bpaternal\b", r"\bold man\b"],
        "meaning": "authority, guidance, discipline",
        "valence": "neutral",
        "category": "people",
    },
    "child": {
        "keywords": [r"\bchild(ren)?\b", r"\bbaby\b", r"\binfant\b", r"\blittle (boy|girl)\b",
                     r"\byoung(er)? self\b"],
        "meaning": "innocence, new beginnings, inner child",
        "valence": "positive",
        "category": "people",
    },
    # ── Places / environments ────────────────────────────────────────────────
    "house": {
        "keywords": [r"\bhouse\b", r"\bhome\b", r"\building\b", r"\broom\b",
                     r"\bcorridor\b", r"\bbasement\b", r"\battic\b"],
        "meaning": "the self, psyche's structure, life situation",
        "valence": "neutral",
        "category": "place",
    },
    "school": {
        "keywords": [r"\bschool\b", r"\bclassroom\b", r"\bexam\b", r"\btest\b",
                     r"\bteacher\b", r"\bstudent\b", r"\buniversity\b", r"\bcollege\b",
                     r"\bgrade[sd]?\b", r"\bunprepared\b"],
        "meaning": "performance anxiety, unresolved learning, life challenges",
        "valence": "negative",
        "category": "place",
    },
    "forest": {
        "keywords": [r"\bforest\b", r"\bwoods\b", r"\btrees?\b", r"\bjungle\b",
                     r"\bwilderness\b"],
        "meaning": "the unconscious, mystery, natural self",
        "valence": "neutral",
        "category": "place",
    },
    "ocean": {
        "keywords": [r"\bocean\b", r"\bsea\b", r"\bdeep water\b", r"\bwaves?\b",
                     r"\btide\b", r"\bbreach\b"],
        "meaning": "the unconscious mind, emotional depth, the vast unknown",
        "valence": "neutral",
        "category": "place",
    },
    # ── Objects / artifacts ───────────────────────────────────────────────────
    "teeth": {
        "keywords": [r"\bteeth\b", r"\btooth\b", r"\bteeth fall(ing)?\b",
                     r"\bteeth crumbl\w*\b", r"\bdental\b"],
        "meaning": "communication anxiety, appearance concerns, fear of loss",
        "valence": "negative",
        "category": "object",
    },
    "door": {
        "keywords": [r"\bdoor(way)?\b", r"\bgate\b", r"\bentrance\b", r"\bportal\b",
                     r"\bthreshold\b"],
        "meaning": "opportunity, transition, access to new situations",
        "valence": "positive",
        "category": "object",
    },
    "mirror": {
        "keywords": [r"\bmirror\b", r"\breflection\b", r"\breflect\w*\b", r"\bdouble\b"],
        "meaning": "self-examination, identity, the observing self",
        "valence": "neutral",
        "category": "object",
    },
    "key": {
        "keywords": [r"\bkey(s)?\b", r"\bunlock(ing)?\b", r"\bopen(ing)?\b"],
        "meaning": "solution, access, unlocking potential",
        "valence": "positive",
        "category": "object",
    },
    "money": {
        "keywords": [r"\bmoney\b", r"\bcash\b", r"\bgold\b", r"\btreasure\b",
                     r"\bwealth\b", r"\briches?\b"],
        "meaning": "value, self-worth, power, security",
        "valence": "positive",
        "category": "object",
    },
    # ── Nature / elements ─────────────────────────────────────────────────────
    "fire": {
        "keywords": [r"\bfire\b", r"\bflame[sd]?\b", r"\bburn(ing)?\b", r"\binferno\b",
                     r"\bblaze\b"],
        "meaning": "transformation, passion, destruction, purification",
        "valence": "neutral",
        "category": "nature",
    },
    "water": {
        "keywords": [r"\bwater\b", r"\brain\b", r"\bflood\b", r"\briver\b",
                     r"\blake\b", r"\bstream\b"],
        "meaning": "emotions, the unconscious, cleansing",
        "valence": "neutral",
        "category": "nature",
    },
    "snake": {
        "keywords": [r"\bsnake[sd]?\b", r"\bserpent\b", r"\bcoil(ing)?\b", r"\bvenom\b"],
        "meaning": "transformation, hidden threat, primal energy, healing",
        "valence": "negative",
        "category": "nature",
    },
    "dog": {
        "keywords": [r"\bdogs?\b", r"\bpuppy\b", r"\bcanine\b"],
        "meaning": "loyalty, instinct, domesticity",
        "valence": "positive",
        "category": "nature",
    },
    "cat": {
        "keywords": [r"\bcats?\b", r"\bkitten\b", r"\bfeline\b"],
        "meaning": "independence, mystery, feminine energy",
        "valence": "neutral",
        "category": "nature",
    },
    "bird": {
        "keywords": [r"\bbirds?\b", r"\bflyng\b", r"\braven\b", r"\beagle\b",
                     r"\bdove\b", r"\browl\b"],
        "meaning": "freedom, aspiration, messages from the unconscious",
        "valence": "positive",
        "category": "nature",
    },
    # ── Emotional / state symbols ─────────────────────────────────────────────
    "darkness": {
        "keywords": [r"\bdark(ness)?\b", r"\bblackness\b", r"\bshadow\b",
                     r"\bdim(ness)?\b", r"\bnight\b"],
        "meaning": "fear of the unknown, unconscious content, isolation",
        "valence": "negative",
        "category": "emotional",
    },
    "light": {
        "keywords": [r"\bright light\b", r"\bglowing\b", r"\bsunlight\b", r"\bbeam\b",
                     r"\billuminat\w*\b", r"\bdawn\b"],
        "meaning": "insight, hope, clarity, spiritual awareness",
        "valence": "positive",
        "category": "emotional",
    },
    "being_naked": {
        "keywords": [r"\bnaked\b", r"\bnudity\b", r"\bno clothes\b", r"\bundressed\b",
                     r"\bexposed\b"],
        "meaning": "vulnerability, authenticity, social anxiety",
        "valence": "negative",
        "category": "emotional",
    },
}


# ── Sentiment lexicons ────────────────────────────────────────────────────────

_POSITIVE = frozenset({
    "joy", "happy", "happiness", "love", "beautiful", "peaceful", "calm",
    "warm", "wonderful", "amazing", "safe", "comfort", "serene", "playful",
    "laugh", "laughter", "delight", "joyful", "bliss", "excited", "elated",
    "grateful", "hope", "free", "dance", "hug", "embrace", "reunion",
    "celebrate", "glad", "thrilled", "cheerful", "pleasant", "content",
    "relieved", "proud", "inspired", "creative", "magical",
})

_NEGATIVE = frozenset({
    "fear", "scared", "terror", "nightmare", "dark", "danger", "attack",
    "death", "dying", "dead", "monster", "threat", "hurt", "pain", "cry",
    "crying", "trapped", "stuck", "lost", "alone", "abandoned", "angry",
    "rage", "violent", "violence", "blood", "wound", "injury", "disaster",
    "flood", "fire", "crash", "drown", "sink", "suffocate", "scream",
    "anxious", "anxiety", "panic", "helpless", "hopeless", "grief",
    "terrified", "horrified", "disturbing", "unsettling", "ominous",
    "dread", "despair", "nightmare", "torment", "agony",
})

_EMOTIONAL_TONES = {
    "anxious":  ["anxious", "anxiety", "panic", "worried", "nervous", "dread", "apprehensive"],
    "peaceful": ["peaceful", "calm", "serene", "tranquil", "still", "gentle", "relaxed"],
    "exciting": ["excited", "thrilling", "adventure", "exhilarating", "vivid", "intense", "wild"],
    "sad":      ["sad", "grief", "loss", "cry", "crying", "tears", "mourning", "melancholy"],
    "fearful":  ["fear", "scared", "terrified", "horrified", "dread", "terror", "frightened"],
    "joyful":   ["happy", "joy", "joyful", "bliss", "delight", "elated", "wonderful"],
    "confusing":["confused", "strange", "weird", "bizarre", "uncertain", "lost", "unclear"],
    "romantic": ["love", "romantic", "kiss", "embrace", "partner", "passion", "desire"],
}

# ── Lucidity markers ──────────────────────────────────────────────────────────

_LUCID_STRONG = [
    r"i realized i was dreaming",
    r"knew i was dreaming",
    r"became lucid",
    r"conscious (in|within) the dream",
    r"aware (that )?i was dreaming",
    r"woke up (inside|within) the dream",
    r"controlled the dream",
    r"took control",
    r"i could (control|change|fly|do) (the dream|anything|everything)",
]

_LUCID_WEAK = [
    r"\blucid\b",
    r"\baware\b",
    r"\bconscious\b",
    r"\bcontrol(led)?\b",
    r"this is a dream",
    r"must be dreaming",
]


class DreamNLP:
    """Unified NLP analysis of dream journal text.

    Implements issue #48 (DreamNet NLP). No large transformer dependencies —
    uses TF-IDF keyword scoring, regex pattern matching, and lexicon-based
    affect analysis.

    Usage:
        nlp = DreamNLP()
        result = nlp.analyze("I was flying over the ocean when suddenly I fell...")
    """

    def analyze(self, text: str) -> Dict[str, Any]:
        """Full dream NLP analysis pipeline.

        Parameters
        ----------
        text : str
            Raw dream journal entry.

        Returns
        -------
        dict with:
          themes         — list of detected theme names (ordered by score)
          theme_scores   — {theme: score 0-1}
          primary_theme  — highest-scoring theme
          sentiment      — {valence: float -1..1, label: str, tone: str}
          symbols        — list of {name, meaning, valence, category, count}
          lucidity       — {score: 0-1, label: str, markers_found: list}
          word_count     — int
          emotional_words — list of {word, polarity}
          insights       — list[str] actionable observations
        """
        if not text or not text.strip():
            return self._empty_result()

        lower = text.lower()
        words = self._tokenize(lower)
        word_set = set(words)

        themes = self._score_themes(lower, words)
        sentiment = self._analyze_sentiment(words, word_set)
        symbols = self._extract_symbols(lower)
        lucidity = self._detect_lucidity(lower)
        emotional_words = self._list_emotional_words(words, word_set)
        insights = self._generate_insights(themes, sentiment, symbols, lucidity)

        return {
            "themes": [t for t, _ in themes[:5]],
            "theme_scores": {t: round(s, 4) for t, s in themes},
            "primary_theme": themes[0][0] if themes else "neutral",
            "sentiment": sentiment,
            "symbols": symbols,
            "lucidity": lucidity,
            "word_count": len(words),
            "emotional_words": emotional_words,
            "insights": insights,
        }

    def analyze_corpus(self, entries: List[str]) -> Dict[str, Any]:
        """Analyze multiple dream journal entries for recurring patterns.

        Parameters
        ----------
        entries : list of str
            Multiple dream journal entries (e.g., past 30 days).

        Returns
        -------
        dict with:
          recurring_symbols  — symbols appearing in 2+ entries
          dominant_themes    — top themes across all entries
          sentiment_trend    — list of per-entry valence values
          lucidity_rate      — fraction of entries with lucidity markers
          symbol_frequency   — {symbol: count_of_entries}
          theme_frequency    — {theme: count_of_entries}
          total_entries      — int
        """
        if not entries:
            return {"total_entries": 0, "recurring_symbols": [], "dominant_themes": []}

        all_symbols: Counter = Counter()
        all_themes: Counter = Counter()
        valence_trend: List[float] = []
        lucid_count = 0

        per_entry_symbols: List[List[str]] = []

        for text in entries:
            if not text or not text.strip():
                continue
            result = self.analyze(text)

            # Themes
            primary = result.get("primary_theme", "neutral")
            all_themes[primary] += 1

            # Symbols
            symbols_found = [s["name"] for s in result.get("symbols", [])]
            per_entry_symbols.append(symbols_found)
            all_symbols.update(set(symbols_found))  # count entries, not occurrences

            # Sentiment
            valence_trend.append(result["sentiment"]["valence"])

            # Lucidity
            if result["lucidity"]["score"] >= 0.5:
                lucid_count += 1

        total = len(entries)
        recurring = [
            {"symbol": sym, "entry_count": cnt, "frequency": round(cnt / total, 3)}
            for sym, cnt in all_symbols.most_common()
            if cnt >= 2
        ]

        return {
            "total_entries": total,
            "recurring_symbols": recurring,
            "dominant_themes": [
                {"theme": t, "count": c, "frequency": round(c / total, 3)}
                for t, c in all_themes.most_common(5)
            ],
            "sentiment_trend": [round(v, 3) for v in valence_trend],
            "mean_valence": round(sum(valence_trend) / len(valence_trend), 3) if valence_trend else 0.0,
            "lucidity_rate": round(lucid_count / total, 3),
            "symbol_frequency": dict(all_symbols.most_common(10)),
            "theme_frequency": dict(all_themes.most_common()),
        }

    # ── Theme scoring ─────────────────────────────────────────────────────────

    # Hall/Van de Castle 15-category taxonomy mapped to keyword sets
    _THEME_KEYWORDS: Dict[str, List[str]] = {
        "flying":         ["fly", "flew", "soar", "float", "levitate", "sky", "air", "wing", "glide", "hover", "airborne"],
        "falling":        ["fall", "fell", "drop", "plunge", "tumble", "cliff", "edge", "descend", "crash", "abyss"],
        "pursuit":        ["chase", "chased", "run", "ran", "escape", "flee", "follow", "caught", "hunt", "danger", "running"],
        "water":          ["water", "ocean", "sea", "swim", "flood", "drown", "river", "wave", "rain", "wet", "pool", "lake"],
        "transportation": ["car", "drive", "drove", "plane", "train", "bus", "road", "travel", "vehicle", "airport", "boat"],
        "social":         ["person", "people", "friend", "family", "talk", "meet", "conversation", "together", "group", "hug", "party"],
        "conflict":       ["fight", "argue", "attack", "angry", "hit", "punch", "war", "battle", "enemy", "hurt", "violence"],
        "loss":           ["die", "died", "death", "dead", "lose", "lost", "gone", "miss", "sad", "grief", "funeral", "disappeared"],
        "discovery":      ["find", "found", "discover", "explore", "room", "door", "hidden", "secret", "path", "new", "reveal"],
        "transformation": ["change", "transform", "become", "morph", "different", "shape", "turn", "strange", "butterfly", "shift"],
        "school":         ["school", "class", "test", "exam", "teacher", "student", "study", "grade", "fail", "college", "late"],
        "nature":         ["tree", "forest", "animal", "mountain", "grass", "flower", "bird", "dog", "cat", "outside", "field"],
        "supernatural":   ["ghost", "magic", "monster", "demon", "angel", "spirit", "weird", "impossible", "strange", "haunted"],
        "body":           ["body", "pain", "hurt", "sick", "blood", "wound", "hand", "eye", "feel", "touch", "breathe", "naked"],
        "neutral":        [],
    }

    def _score_themes(self, lower: str, words: List[str]) -> List[Tuple[str, float]]:
        """Score all themes and return sorted list of (theme, score)."""
        scores: Dict[str, float] = {}
        for theme, kws in self._THEME_KEYWORDS.items():
            hits = sum(1 for kw in kws if kw in lower)
            scores[theme] = float(hits)

        # Ensure neutral is a floor when nothing matches
        if sum(v for k, v in scores.items() if k != "neutral") == 0:
            scores["neutral"] = 1.0

        total = sum(scores.values()) + 1e-10
        normalized = sorted(
            [(t, s / total) for t, s in scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        return normalized

    # ── Sentiment analysis ────────────────────────────────────────────────────

    def _analyze_sentiment(self, words: List[str], word_set: set) -> Dict[str, Any]:
        """Compute valence, label, and emotional tone."""
        pos = sum(1 for w in words if w in _POSITIVE)
        neg = sum(1 for w in words if w in _NEGATIVE)
        total = pos + neg or 1

        valence = float((pos - neg) / total)
        valence = max(-1.0, min(1.0, valence))

        if valence > 0.3:
            label = "positive"
        elif valence > 0.05:
            label = "slightly positive"
        elif valence > -0.05:
            label = "neutral"
        elif valence > -0.3:
            label = "slightly negative"
        else:
            label = "negative"

        tone = self._detect_tone(word_set)

        return {
            "valence": round(valence, 3),
            "label": label,
            "tone": tone,
            "positive_word_count": pos,
            "negative_word_count": neg,
        }

    @staticmethod
    def _detect_tone(word_set: set) -> str:
        """Detect dominant emotional tone from word set."""
        tone_scores: Dict[str, int] = {}
        for tone, tone_words in _EMOTIONAL_TONES.items():
            tone_scores[tone] = sum(1 for w in tone_words if w in word_set)

        if not any(tone_scores.values()):
            return "neutral"

        return max(tone_scores, key=tone_scores.get)

    # ── Symbol extraction ─────────────────────────────────────────────────────

    def _extract_symbols(self, lower: str) -> List[Dict[str, Any]]:
        """Extract dream symbols using regex pattern matching."""
        found: List[Dict[str, Any]] = []

        for symbol_name, info in DREAM_SYMBOLS.items():
            count = 0
            matched: List[str] = []
            for pattern in info["keywords"]:
                for m in re.finditer(pattern, lower):
                    count += 1
                    text = m.group()
                    if text not in matched:
                        matched.append(text)

            if count > 0:
                found.append({
                    "name": symbol_name,
                    "meaning": info["meaning"],
                    "valence": info["valence"],
                    "category": info["category"],
                    "count": count,
                    "matched_text": matched[:3],  # cap at 3 examples
                })

        # Sort by count descending
        found.sort(key=lambda x: x["count"], reverse=True)
        return found

    # ── Lucidity detection ────────────────────────────────────────────────────

    def _detect_lucidity(self, lower: str) -> Dict[str, Any]:
        """Detect lucid dreaming markers in text."""
        strong_markers: List[str] = []
        for pattern in _LUCID_STRONG:
            if re.search(pattern, lower):
                strong_markers.append(pattern)

        weak_markers: List[str] = []
        if not strong_markers:
            for pattern in _LUCID_WEAK:
                if re.search(pattern, lower):
                    weak_markers.append(pattern)

        if strong_markers:
            score = 0.9
            label = "highly likely"
        elif len(weak_markers) >= 3:
            score = 0.6
            label = "possible"
        elif len(weak_markers) >= 1:
            score = 0.3
            label = "weak indicators"
        else:
            score = 0.0
            label = "not detected"

        return {
            "score": round(score, 3),
            "label": label,
            "markers_found": strong_markers if strong_markers else weak_markers,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z]+", text.lower())

    @staticmethod
    def _list_emotional_words(words: List[str], word_set: set) -> List[Dict[str, str]]:
        results = []
        seen: set = set()
        for w in words:
            if w in seen:
                continue
            if w in _POSITIVE:
                results.append({"word": w, "polarity": "positive"})
                seen.add(w)
            elif w in _NEGATIVE:
                results.append({"word": w, "polarity": "negative"})
                seen.add(w)
        return results[:20]

    @staticmethod
    def _generate_insights(
        themes: List[Tuple[str, float]],
        sentiment: Dict[str, Any],
        symbols: List[Dict[str, Any]],
        lucidity: Dict[str, Any],
    ) -> List[str]:
        insights: List[str] = []
        primary = themes[0][0] if themes else "neutral"

        # Lucidity
        if lucidity["score"] >= 0.7:
            insights.append(
                "Lucid dream detected. Audio reality-checking cues during REM can triple "
                "lucid dream frequency (Northwestern TLR 2024)."
            )

        # Nightmare indicators
        if sentiment["tone"] == "fearful" and sentiment["valence"] < -0.3:
            insights.append(
                "High-fear content detected. Imagery Rehearsal Therapy (IRT) reduces nightmare "
                "frequency by 50-80% within 4-6 weeks — try rewriting this dream with a new ending."
            )
        elif sentiment["tone"] == "anxious":
            insights.append(
                "Anxiety-tinged dream. Recurring anxiety dreams often reflect unresolved daytime stress."
            )

        # Theme insights
        theme_insights = {
            "pursuit":        "Chase theme: often reflects avoidance of a situation or emotion in waking life.",
            "flying":         "Flying theme: associated with high sense of agency and creative expansion.",
            "falling":        "Falling theme: commonly linked to anxiety, insecurity, or sudden loss of control.",
            "loss":           "Loss/death theme: usually symbolic — may represent endings, transitions, or fear of change rather than literal death.",
            "school":         "School/exam theme: one of the most common adult dreams — often signals performance anxiety or perfectionism.",
            "transformation": "Transformation theme: suggests psychological growth or a significant life transition.",
            "discovery":      "Discovery theme: curiosity and openness are active — your unconscious may be processing new information.",
            "social":         "Social theme: high relationship content suggests current interpersonal dynamics are prominent in your mind.",
            "supernatural":   "Supernatural theme: vivid, magical dreams are associated with high REM activity and creative thinking.",
        }
        if primary in theme_insights:
            insights.append(theme_insights[primary])

        # Symbol-specific insights
        symbol_names = {s["name"] for s in symbols}
        if "teeth" in symbol_names:
            insights.append("Teeth symbol: one of the most universal dream symbols — linked to communication anxiety or appearance concerns.")
        if "mirror" in symbol_names:
            insights.append("Mirror symbol: often signals self-reflection or a need to examine how you present yourself.")
        if "snake" in symbol_names:
            insights.append("Snake symbol: ambivalent — can represent transformation and healing, or hidden fear and threat.")

        # Positive note
        if sentiment["valence"] > 0.3:
            insights.append("Emotionally positive dream — positive dreams are linked to next-day mood improvement and creative problem-solving.")

        if not insights:
            insights.append("Neutral dream content. Patterns become visible after 7+ journal entries.")

        return insights[:5]  # cap at 5 insights

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "themes": ["neutral"],
            "theme_scores": {"neutral": 1.0},
            "primary_theme": "neutral",
            "sentiment": {
                "valence": 0.0,
                "label": "neutral",
                "tone": "neutral",
                "positive_word_count": 0,
                "negative_word_count": 0,
            },
            "symbols": [],
            "lucidity": {"score": 0.0, "label": "not detected", "markers_found": []},
            "word_count": 0,
            "emotional_words": [],
            "insights": ["Enter your dream journal to see analysis."],
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_instance: Optional[DreamNLP] = None


def get_dream_nlp() -> DreamNLP:
    global _instance
    if _instance is None:
        _instance = DreamNLP()
    return _instance
