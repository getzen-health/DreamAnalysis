"""Emotion Archaeology — reconstruct past emotional landscapes from digital artifacts.

Analyzes digital footprints (photos, journal entries, music listening history,
calendar events) to reconstruct what a person's emotional life likely looked
like over a given period. Each artifact type contributes a different emotional
signal, and the system stitches them together into a unified emotional timeline.

Artifact channels:
    Photo metadata  -> emotional context from timestamps, locations, people count,
                       event type (wedding=joy, hospital=anxiety, sunset=calm, etc.)
    Journal/text    -> sentiment analysis via keyword scoring, emotional theme
                       detection, narrative arc tracking
    Music history   -> valence/energy mapping from genres and listening patterns
                       (sad ballads=low valence, dance=high energy+positive)
    Calendar events -> likely emotional states from event types
                       (meeting=stress, vacation=joy, deadline=anxiety, etc.)

Output: A per-period emotional landscape with dominant emotions, emotional
diversity score, and a natural-language archaeology report.

Issue #453.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & mappings
# ---------------------------------------------------------------------------

# Sentiment word lists (simplified but functional)
_POSITIVE_WORDS = frozenset({
    "happy", "joy", "love", "great", "wonderful", "amazing", "beautiful",
    "excited", "grateful", "thankful", "blessed", "peace", "calm", "relaxed",
    "proud", "hopeful", "inspired", "cheerful", "delighted", "content",
    "fantastic", "excellent", "brilliant", "sunshine", "laugh", "smile",
    "celebration", "triumph", "success", "accomplish", "fun", "enjoy",
    "warm", "kind", "gentle", "comfort", "safe", "bliss", "elated",
    "optimistic", "thrilled", "ecstatic",
})

_NEGATIVE_WORDS = frozenset({
    "sad", "angry", "fear", "anxious", "depressed", "lonely", "hurt",
    "pain", "cry", "stressed", "overwhelmed", "frustrated", "disappointed",
    "worried", "scared", "terrified", "grief", "loss", "broken", "empty",
    "exhausted", "hopeless", "miserable", "regret", "guilt", "shame",
    "bitter", "resentful", "jealous", "panic", "dread", "despair",
    "heartbreak", "suffering", "torment", "anguish", "rage", "fury",
    "anxious", "nervous", "upset",
})

# Emotional themes detected in text
_THEME_PATTERNS: Dict[str, List[str]] = {
    "gratitude": ["grateful", "thankful", "blessed", "appreciate", "gratitude"],
    "anxiety": ["anxious", "worried", "nervous", "panic", "dread", "fear"],
    "grief": ["grief", "loss", "mourning", "miss", "gone", "passed"],
    "growth": ["learn", "grow", "progress", "improve", "better", "evolve"],
    "love": ["love", "adore", "cherish", "partner", "together", "romance"],
    "loneliness": ["lonely", "alone", "isolated", "nobody", "solitude"],
    "anger": ["angry", "furious", "rage", "mad", "frustrated", "annoyed"],
    "hope": ["hope", "dream", "future", "believe", "faith", "aspire"],
    "nostalgia": ["remember", "memories", "past", "childhood", "those days"],
    "achievement": ["accomplish", "success", "achieve", "won", "completed", "goal"],
}

# Music genre -> (valence, energy) mappings on -1..1 scale
_GENRE_EMOTION_MAP: Dict[str, Tuple[float, float]] = {
    "pop": (0.5, 0.6),
    "rock": (0.1, 0.7),
    "metal": (-0.2, 0.9),
    "jazz": (0.3, 0.3),
    "classical": (0.2, 0.2),
    "blues": (-0.3, 0.3),
    "country": (0.1, 0.4),
    "hip_hop": (0.2, 0.7),
    "rap": (0.0, 0.8),
    "electronic": (0.3, 0.8),
    "dance": (0.6, 0.9),
    "r_and_b": (0.3, 0.4),
    "soul": (0.1, 0.3),
    "folk": (0.1, 0.2),
    "indie": (0.0, 0.4),
    "punk": (-0.1, 0.9),
    "reggae": (0.4, 0.4),
    "ambient": (0.2, 0.1),
    "lo_fi": (0.1, 0.2),
    "sad_ballad": (-0.6, 0.1),
    "worship": (0.4, 0.3),
    "soundtrack": (0.1, 0.4),
    "unknown": (0.0, 0.4),
}

# Calendar event type -> (valence, arousal, dominant_emotion)
_EVENT_EMOTION_MAP: Dict[str, Tuple[float, float, str]] = {
    "meeting": (-0.1, 0.5, "stress"),
    "deadline": (-0.3, 0.7, "anxiety"),
    "vacation": (0.7, 0.6, "joy"),
    "holiday": (0.6, 0.5, "joy"),
    "birthday": (0.6, 0.6, "joy"),
    "wedding": (0.8, 0.7, "joy"),
    "funeral": (-0.7, 0.3, "grief"),
    "doctor": (-0.3, 0.5, "anxiety"),
    "dentist": (-0.4, 0.5, "anxiety"),
    "therapy": (0.1, 0.3, "neutral"),
    "exercise": (0.4, 0.7, "positive"),
    "gym": (0.3, 0.6, "positive"),
    "date": (0.5, 0.6, "excitement"),
    "interview": (-0.2, 0.7, "anxiety"),
    "exam": (-0.4, 0.8, "anxiety"),
    "presentation": (-0.2, 0.7, "stress"),
    "concert": (0.6, 0.8, "joy"),
    "movie": (0.3, 0.4, "relaxation"),
    "dinner": (0.4, 0.4, "contentment"),
    "lunch": (0.3, 0.3, "neutral"),
    "travel": (0.5, 0.5, "excitement"),
    "moving": (-0.1, 0.6, "stress"),
    "breakup": (-0.7, 0.5, "grief"),
    "layoff": (-0.6, 0.6, "anxiety"),
    "promotion": (0.7, 0.7, "joy"),
    "graduation": (0.8, 0.7, "joy"),
    "surgery": (-0.5, 0.4, "anxiety"),
    "hospital": (-0.5, 0.5, "anxiety"),
    "meditation": (0.3, 0.1, "calm"),
    "yoga": (0.4, 0.3, "calm"),
    "default": (0.0, 0.3, "neutral"),
}

# Photo context -> (valence, arousal, dominant_emotion)
_PHOTO_CONTEXT_MAP: Dict[str, Tuple[float, float, str]] = {
    "wedding": (0.8, 0.7, "joy"),
    "party": (0.6, 0.7, "excitement"),
    "birthday": (0.6, 0.6, "joy"),
    "graduation": (0.7, 0.7, "pride"),
    "travel": (0.5, 0.5, "excitement"),
    "nature": (0.4, 0.2, "calm"),
    "sunset": (0.4, 0.2, "calm"),
    "beach": (0.5, 0.3, "relaxation"),
    "family": (0.5, 0.4, "warmth"),
    "friends": (0.5, 0.5, "joy"),
    "pet": (0.5, 0.3, "warmth"),
    "food": (0.3, 0.3, "contentment"),
    "selfie": (0.3, 0.4, "neutral"),
    "hospital": (-0.4, 0.4, "anxiety"),
    "funeral": (-0.6, 0.3, "grief"),
    "work": (0.0, 0.4, "neutral"),
    "concert": (0.6, 0.8, "excitement"),
    "sport": (0.4, 0.7, "excitement"),
    "default": (0.1, 0.3, "neutral"),
}

# Emotion categories used in the final timeline
EMOTION_CATEGORIES = [
    "joy", "calm", "excitement", "pride", "warmth", "contentment",
    "neutral", "stress", "anxiety", "grief", "anger", "loneliness",
    "relaxation", "positive",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TextSentiment:
    """Result of analyzing a single text entry."""
    text_preview: str
    positive_score: float
    negative_score: float
    net_sentiment: float  # -1 to 1
    themes: List[str]
    word_count: int


@dataclass
class MusicEmotionSignal:
    """Emotional signal from a music listening entry."""
    track_or_genre: str
    valence: float  # -1 to 1
    energy: float   # 0 to 1
    inferred_mood: str


@dataclass
class CalendarEmotionSignal:
    """Emotional signal from a calendar event."""
    event_type: str
    event_name: str
    valence: float
    arousal: float
    dominant_emotion: str


@dataclass
class PhotoEmotionSignal:
    """Emotional signal from photo metadata."""
    event_type: str
    people_count: int
    valence: float
    arousal: float
    dominant_emotion: str
    timestamp: Optional[float] = None
    location: Optional[str] = None


@dataclass
class EmotionalPeriod:
    """Emotional summary for a single time period."""
    period_label: str
    dominant_emotion: str
    average_valence: float
    average_arousal: float
    emotion_distribution: Dict[str, float]
    emotional_diversity: float  # 0..1, higher = more diverse
    artifact_count: int
    themes: List[str]


@dataclass
class ArchaeologyReport:
    """Full emotional archaeology report."""
    user_id: str
    generated_at: float
    period_start: str
    period_end: str
    total_artifacts_analyzed: int
    timeline: List[EmotionalPeriod]
    overall_valence: float
    overall_arousal: float
    dominant_themes: List[str]
    narrative_summary: str
    emotional_trajectory: str  # "improving", "declining", "stable", "volatile"
    recommendations: List[str]


# ---------------------------------------------------------------------------
# Text sentiment analysis
# ---------------------------------------------------------------------------


def analyze_text_sentiment(text: str) -> TextSentiment:
    """Analyze sentiment and emotional themes in a text entry.

    Uses keyword-based scoring. Each positive word adds +1, each negative
    word adds -1. Net sentiment is normalized to [-1, 1].
    Emotional themes are detected via pattern matching.

    Parameters
    ----------
    text : str
        Journal entry, message, or any text to analyze.

    Returns
    -------
    TextSentiment
        Sentiment scores, detected themes, and text stats.
    """
    if not text or not text.strip():
        return TextSentiment(
            text_preview="",
            positive_score=0.0,
            negative_score=0.0,
            net_sentiment=0.0,
            themes=[],
            word_count=0,
        )

    words = re.findall(r"[a-zA-Z]+", text.lower())
    word_count = len(words)

    if word_count == 0:
        return TextSentiment(
            text_preview=text[:80],
            positive_score=0.0,
            negative_score=0.0,
            net_sentiment=0.0,
            themes=[],
            word_count=0,
        )

    pos_count = sum(1 for w in words if w in _POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in _NEGATIVE_WORDS)

    total_sentiment = pos_count + neg_count
    positive_score = pos_count / word_count
    negative_score = neg_count / word_count

    if total_sentiment > 0:
        net_sentiment = (pos_count - neg_count) / total_sentiment
    else:
        net_sentiment = 0.0

    # Clamp
    net_sentiment = max(-1.0, min(1.0, net_sentiment))

    # Detect themes
    text_lower = text.lower()
    detected_themes: List[str] = []
    for theme, keywords in _THEME_PATTERNS.items():
        if any(kw in text_lower for kw in keywords):
            detected_themes.append(theme)

    preview = text[:80] + ("..." if len(text) > 80 else "")

    return TextSentiment(
        text_preview=preview,
        positive_score=round(positive_score, 4),
        negative_score=round(negative_score, 4),
        net_sentiment=round(net_sentiment, 4),
        themes=detected_themes,
        word_count=word_count,
    )


# ---------------------------------------------------------------------------
# Music history analysis
# ---------------------------------------------------------------------------


def analyze_music_history(
    entries: List[Dict[str, Any]],
) -> List[MusicEmotionSignal]:
    """Map music listening history to emotional signals.

    Each entry should have at minimum a ``genre`` key (str). Optional keys:
    ``track`` (str), ``valence`` (float, 0-1), ``energy`` (float, 0-1).

    If ``valence``/``energy`` are provided directly (e.g. from Spotify audio
    features), those override the genre lookup.

    Parameters
    ----------
    entries : list of dict
        Music listening entries.

    Returns
    -------
    list of MusicEmotionSignal
    """
    signals: List[MusicEmotionSignal] = []

    for entry in entries:
        genre = entry.get("genre", "unknown").lower().replace(" ", "_").replace("-", "_")
        track = entry.get("track", genre)

        # Use provided valence/energy if available and non-None, else fall back to genre map
        if entry.get("valence") is not None and entry.get("energy") is not None:
            valence = float(entry["valence"]) * 2 - 1  # 0..1 -> -1..1
            energy = float(entry["energy"])
        else:
            default = _GENRE_EMOTION_MAP.get("unknown", (0.0, 0.4))
            valence, energy = _GENRE_EMOTION_MAP.get(genre, default)

        # Determine mood label
        if valence > 0.3 and energy > 0.5:
            mood = "energetic_positive"
        elif valence > 0.3 and energy <= 0.5:
            mood = "calm_positive"
        elif valence < -0.3 and energy > 0.5:
            mood = "agitated_negative"
        elif valence < -0.3 and energy <= 0.5:
            mood = "melancholic"
        elif energy > 0.6:
            mood = "high_energy_neutral"
        else:
            mood = "neutral"

        signals.append(MusicEmotionSignal(
            track_or_genre=track,
            valence=round(valence, 3),
            energy=round(energy, 3),
            inferred_mood=mood,
        ))

    return signals


# ---------------------------------------------------------------------------
# Calendar analysis
# ---------------------------------------------------------------------------


def analyze_calendar_events(
    events: List[Dict[str, Any]],
) -> List[CalendarEmotionSignal]:
    """Map calendar events to likely emotional states.

    Each event should have ``event_type`` (str) and ``name`` (str).

    Parameters
    ----------
    events : list of dict
        Calendar events, each with ``event_type`` and ``name``.

    Returns
    -------
    list of CalendarEmotionSignal
    """
    signals: List[CalendarEmotionSignal] = []

    for event in events:
        event_type = event.get("event_type", "default").lower().replace(" ", "_")
        name = event.get("name", event_type)

        default = _EVENT_EMOTION_MAP["default"]
        valence, arousal, emotion = _EVENT_EMOTION_MAP.get(event_type, default)

        signals.append(CalendarEmotionSignal(
            event_type=event_type,
            event_name=name,
            valence=round(valence, 3),
            arousal=round(arousal, 3),
            dominant_emotion=emotion,
        ))

    return signals


# ---------------------------------------------------------------------------
# Photo metadata analysis
# ---------------------------------------------------------------------------


def analyze_photo_metadata(
    photos: List[Dict[str, Any]],
) -> List[PhotoEmotionSignal]:
    """Extract emotional context from photo metadata.

    Each photo dict should have at minimum ``event_type`` (str).
    Optional: ``people_count`` (int), ``timestamp`` (float), ``location`` (str).

    Parameters
    ----------
    photos : list of dict
        Photo metadata entries.

    Returns
    -------
    list of PhotoEmotionSignal
    """
    signals: List[PhotoEmotionSignal] = []

    for photo in photos:
        event_type = photo.get("event_type", "default").lower().replace(" ", "_")
        people_count = int(photo.get("people_count", 0))
        ts = photo.get("timestamp")
        location = photo.get("location")

        default = _PHOTO_CONTEXT_MAP["default"]
        valence, arousal, emotion = _PHOTO_CONTEXT_MAP.get(event_type, default)

        # More people in photos tends to indicate social/positive context
        if people_count > 3:
            valence = min(1.0, valence + 0.1)
            arousal = min(1.0, arousal + 0.05)
        elif people_count == 0:
            # Solo photos are neutral — don't adjust much
            pass

        signals.append(PhotoEmotionSignal(
            event_type=event_type,
            people_count=people_count,
            valence=round(valence, 3),
            arousal=round(arousal, 3),
            dominant_emotion=emotion,
            timestamp=ts,
            location=location,
        ))

    return signals


# ---------------------------------------------------------------------------
# Emotional timeline reconstruction
# ---------------------------------------------------------------------------


def _compute_emotional_diversity(distribution: Dict[str, float]) -> float:
    """Compute Shannon entropy-based diversity score (0..1).

    Higher values = more emotionally diverse period.
    """
    values = [v for v in distribution.values() if v > 0]
    if len(values) <= 1:
        return 0.0

    total = sum(values)
    if total <= 0:
        return 0.0

    probs = [v / total for v in values]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(values))

    if max_entropy <= 0:
        return 0.0

    return round(min(1.0, entropy / max_entropy), 3)


def _infer_trajectory(periods: List[EmotionalPeriod]) -> str:
    """Infer overall emotional trajectory from period summaries."""
    if len(periods) < 2:
        return "stable"

    valences = [p.average_valence for p in periods]

    # Linear trend via differences
    diffs = [valences[i + 1] - valences[i] for i in range(len(valences) - 1)]
    avg_diff = sum(diffs) / len(diffs)

    # Volatility
    if len(valences) >= 3:
        variance = sum((v - sum(valences) / len(valences)) ** 2 for v in valences) / len(valences)
        volatility = math.sqrt(variance)
    else:
        volatility = 0.0

    if volatility > 0.4:
        return "volatile"
    elif avg_diff > 0.1:
        return "improving"
    elif avg_diff < -0.1:
        return "declining"
    else:
        return "stable"


def reconstruct_emotional_timeline(
    artifacts: Dict[str, Any],
    period_labels: Optional[List[str]] = None,
) -> List[EmotionalPeriod]:
    """Stitch together emotional signals from multiple artifact types
    into a unified emotional timeline.

    Parameters
    ----------
    artifacts : dict
        Keys: ``texts`` (list of str), ``music`` (list of dict),
        ``calendar`` (list of dict), ``photos`` (list of dict).
        Each list contributes emotional signals. All are optional.
    period_labels : list of str, optional
        Labels for time periods. If None, a single "overall" period is used.

    Returns
    -------
    list of EmotionalPeriod
        One per period label.
    """
    if period_labels is None:
        period_labels = ["overall"]

    # Analyze each artifact channel
    text_sentiments: List[TextSentiment] = []
    for text in artifacts.get("texts", []):
        text_sentiments.append(analyze_text_sentiment(text))

    music_signals = analyze_music_history(artifacts.get("music", []))
    calendar_signals = analyze_calendar_events(artifacts.get("calendar", []))
    photo_signals = analyze_photo_metadata(artifacts.get("photos", []))

    total_artifacts = (
        len(text_sentiments) + len(music_signals) +
        len(calendar_signals) + len(photo_signals)
    )

    # Collect all valence/arousal readings and emotions
    all_valences: List[float] = []
    all_arousals: List[float] = []
    all_emotions: List[str] = []
    all_themes: List[str] = []

    for ts in text_sentiments:
        all_valences.append(ts.net_sentiment)
        all_arousals.append(0.3 + abs(ts.net_sentiment) * 0.4)  # proxy arousal
        all_themes.extend(ts.themes)
        if ts.net_sentiment > 0.2:
            all_emotions.append("joy")
        elif ts.net_sentiment < -0.2:
            all_emotions.append("grief")
        else:
            all_emotions.append("neutral")

    for ms in music_signals:
        all_valences.append(ms.valence)
        all_arousals.append(ms.energy)
        all_emotions.append(ms.inferred_mood)

    for cs in calendar_signals:
        all_valences.append(cs.valence)
        all_arousals.append(cs.arousal)
        all_emotions.append(cs.dominant_emotion)

    for ps in photo_signals:
        all_valences.append(ps.valence)
        all_arousals.append(ps.arousal)
        all_emotions.append(ps.dominant_emotion)

    # Build periods — distribute artifacts evenly across periods
    n_periods = len(period_labels)
    n_signals = len(all_valences)
    periods: List[EmotionalPeriod] = []

    if n_signals == 0:
        for label in period_labels:
            periods.append(EmotionalPeriod(
                period_label=label,
                dominant_emotion="neutral",
                average_valence=0.0,
                average_arousal=0.0,
                emotion_distribution={"neutral": 1.0},
                emotional_diversity=0.0,
                artifact_count=0,
                themes=[],
            ))
        return periods

    chunk_size = max(1, n_signals // n_periods)

    for i, label in enumerate(period_labels):
        start_idx = i * chunk_size
        if i == n_periods - 1:
            # Last period gets the rest
            end_idx = n_signals
        else:
            end_idx = min(start_idx + chunk_size, n_signals)

        if start_idx >= n_signals:
            # More periods than data — fill with neutral
            periods.append(EmotionalPeriod(
                period_label=label,
                dominant_emotion="neutral",
                average_valence=0.0,
                average_arousal=0.0,
                emotion_distribution={"neutral": 1.0},
                emotional_diversity=0.0,
                artifact_count=0,
                themes=[],
            ))
            continue

        chunk_valences = all_valences[start_idx:end_idx]
        chunk_arousals = all_arousals[start_idx:end_idx]
        chunk_emotions = all_emotions[start_idx:end_idx]

        avg_v = sum(chunk_valences) / len(chunk_valences)
        avg_a = sum(chunk_arousals) / len(chunk_arousals)

        # Build emotion distribution
        emotion_counts: Dict[str, int] = {}
        for emo in chunk_emotions:
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
        total_emo = sum(emotion_counts.values())
        emotion_dist = {k: round(v / total_emo, 3) for k, v in emotion_counts.items()}

        dominant = max(emotion_counts, key=emotion_counts.get)  # type: ignore[arg-type]
        diversity = _compute_emotional_diversity(emotion_dist)

        # Collect themes for this period
        period_theme_start = min(start_idx, len(all_themes))
        period_theme_end = min(end_idx, len(all_themes))
        period_themes = list(set(all_themes[period_theme_start:period_theme_end]))

        periods.append(EmotionalPeriod(
            period_label=label,
            dominant_emotion=dominant,
            average_valence=round(avg_v, 3),
            average_arousal=round(avg_a, 3),
            emotion_distribution=emotion_dist,
            emotional_diversity=diversity,
            artifact_count=end_idx - start_idx,
            themes=period_themes,
        ))

    return periods


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _generate_narrative(
    timeline: List[EmotionalPeriod],
    trajectory: str,
    overall_valence: float,
) -> str:
    """Generate a natural-language narrative summary of the emotional landscape."""
    if not timeline:
        return "Insufficient data to reconstruct emotional landscape."

    parts: List[str] = []

    # Opening
    if overall_valence > 0.3:
        parts.append("Overall, this period reflects a predominantly positive emotional landscape.")
    elif overall_valence < -0.3:
        parts.append("Overall, this period reflects a challenging emotional landscape with significant negative affect.")
    else:
        parts.append("Overall, this period reflects a mixed emotional landscape with both positive and difficult moments.")

    # Per-period notes
    for period in timeline:
        if period.dominant_emotion in ("joy", "excitement", "calm", "pride", "warmth", "positive"):
            tone = "positive"
        elif period.dominant_emotion in ("grief", "anxiety", "stress", "anger", "loneliness"):
            tone = "difficult"
        else:
            tone = "neutral"

        parts.append(
            f"During '{period.period_label}', the dominant emotional tone was "
            f"{tone} (primary: {period.dominant_emotion}, "
            f"valence: {period.average_valence:+.2f}, "
            f"diversity: {period.emotional_diversity:.2f})."
        )

    # Trajectory
    trajectory_text = {
        "improving": "The emotional trajectory shows improvement over time.",
        "declining": "The emotional trajectory shows a decline, suggesting increasing difficulty.",
        "stable": "The emotional landscape remained relatively stable throughout.",
        "volatile": "The emotional landscape was volatile, with significant swings between periods.",
    }
    parts.append(trajectory_text.get(trajectory, ""))

    return " ".join(parts)


def _generate_recommendations(
    trajectory: str,
    overall_valence: float,
    themes: List[str],
) -> List[str]:
    """Generate actionable recommendations based on the emotional landscape."""
    recs: List[str] = []

    if overall_valence < -0.2:
        recs.append(
            "Consider reaching out to a mental health professional if negative patterns persist."
        )

    if "anxiety" in themes:
        recs.append(
            "Anxiety patterns detected — mindfulness or breathing exercises may help."
        )

    if "grief" in themes:
        recs.append(
            "Grief signals present — allow space for processing and consider grief support groups."
        )

    if "loneliness" in themes:
        recs.append(
            "Loneliness indicators detected — scheduling regular social activities may improve wellbeing."
        )

    if trajectory == "declining":
        recs.append(
            "Emotional trajectory is declining — monitoring mental health and seeking support early is advised."
        )

    if trajectory == "volatile":
        recs.append(
            "High emotional volatility detected — journaling or mood tracking may help identify triggers."
        )

    if overall_valence > 0.3:
        recs.append(
            "Positive emotional baseline detected — continue nurturing activities that bring joy."
        )

    if "growth" in themes:
        recs.append(
            "Growth themes detected — personal development efforts appear to be making a positive impact."
        )

    if not recs:
        recs.append(
            "Emotional landscape appears balanced — continue current self-care practices."
        )

    return recs


def generate_archaeology_report(
    user_id: str,
    artifacts: Dict[str, Any],
    period_start: str = "unknown",
    period_end: str = "unknown",
    period_labels: Optional[List[str]] = None,
) -> ArchaeologyReport:
    """Generate a complete emotional archaeology report.

    Parameters
    ----------
    user_id : str
        User identifier.
    artifacts : dict
        Digital artifacts. Keys: ``texts``, ``music``, ``calendar``, ``photos``.
    period_start, period_end : str
        Human-readable period boundaries (e.g. "2024-01", "2024-12").
    period_labels : list of str, optional
        Labels for sub-periods. If None, defaults to ["overall"].

    Returns
    -------
    ArchaeologyReport
    """
    timeline = reconstruct_emotional_timeline(artifacts, period_labels)

    total_artifacts = (
        len(artifacts.get("texts", []))
        + len(artifacts.get("music", []))
        + len(artifacts.get("calendar", []))
        + len(artifacts.get("photos", []))
    )

    # Overall stats
    if timeline:
        overall_valence = sum(p.average_valence for p in timeline) / len(timeline)
        overall_arousal = sum(p.average_arousal for p in timeline) / len(timeline)
    else:
        overall_valence = 0.0
        overall_arousal = 0.0

    trajectory = _infer_trajectory(timeline)

    # Collect all themes
    all_themes: List[str] = []
    for period in timeline:
        all_themes.extend(period.themes)
    # Count and sort
    theme_counts: Dict[str, int] = {}
    for t in all_themes:
        theme_counts[t] = theme_counts.get(t, 0) + 1
    dominant_themes = sorted(theme_counts, key=theme_counts.get, reverse=True)[:5]  # type: ignore[arg-type]

    narrative = _generate_narrative(timeline, trajectory, overall_valence)
    recommendations = _generate_recommendations(trajectory, overall_valence, dominant_themes)

    return ArchaeologyReport(
        user_id=user_id,
        generated_at=time.time(),
        period_start=period_start,
        period_end=period_end,
        total_artifacts_analyzed=total_artifacts,
        timeline=timeline,
        overall_valence=round(overall_valence, 3),
        overall_arousal=round(overall_arousal, 3),
        dominant_themes=dominant_themes,
        narrative_summary=narrative,
        emotional_trajectory=trajectory,
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def report_to_dict(report: ArchaeologyReport) -> Dict[str, Any]:
    """Serialize an ArchaeologyReport to a plain dict for JSON responses."""
    return {
        "user_id": report.user_id,
        "generated_at": report.generated_at,
        "period_start": report.period_start,
        "period_end": report.period_end,
        "total_artifacts_analyzed": report.total_artifacts_analyzed,
        "timeline": [
            {
                "period_label": p.period_label,
                "dominant_emotion": p.dominant_emotion,
                "average_valence": p.average_valence,
                "average_arousal": p.average_arousal,
                "emotion_distribution": p.emotion_distribution,
                "emotional_diversity": p.emotional_diversity,
                "artifact_count": p.artifact_count,
                "themes": p.themes,
            }
            for p in report.timeline
        ],
        "overall_valence": report.overall_valence,
        "overall_arousal": report.overall_arousal,
        "dominant_themes": report.dominant_themes,
        "narrative_summary": report.narrative_summary,
        "emotional_trajectory": report.emotional_trajectory,
        "recommendations": report.recommendations,
    }
