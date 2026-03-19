"""Dream-sleep architecture fusion model — issue #437.

Correlates dream content/themes with sleep stage physiology to answer:
  - Which sleep patterns produce which types of dreams?
  - Does high REM + stress predict nightmares?
  - What recurring dream themes cluster with specific physiological conditions?

Science basis:
  - Hobson & McCarley (1977) — activation-synthesis: REM brainstem activity
    generates dream imagery, cortex "interprets" it into narrative.
  - Nir & Tononi (2010) — dreaming and consciousness share neural substrates;
    sleep architecture quality directly shapes dream phenomenology.
  - Schredl (2018) — dream recall is highest after REM awakenings (>80%),
    dream emotionality correlates with pre-sleep stress and REM density.
  - Scarpelli et al. (2019) — nightmares are more frequent when REM% is
    elevated AND sleep efficiency is low (fragmented sleep).

No ML weights required — all computation is deterministic heuristic scoring.
"""
from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# -- Emotion keyword lexicons (compact, tuned for dream text) ----------------

_EMOTION_LEXICON: Dict[str, List[str]] = {
    "joy": [
        "happy", "joy", "joyful", "elated", "bliss", "wonderful", "amazing",
        "celebrate", "laugh", "laughter", "delight", "excited", "grateful",
    ],
    "fear": [
        "fear", "scared", "terrified", "terror", "horror", "horrified",
        "panic", "dread", "fright", "frightened", "alarmed",
    ],
    "sadness": [
        "sad", "grief", "crying", "cry", "tears", "mourning", "loss",
        "melancholy", "depressed", "heartbroken", "lonely", "abandoned",
    ],
    "anger": [
        "angry", "rage", "furious", "violent", "aggression", "fight",
        "hostile", "irritated", "frustrated", "resentment",
    ],
    "anxiety": [
        "anxious", "anxiety", "worried", "nervous", "uneasy", "tense",
        "stressed", "restless", "overwhelmed", "helpless", "trapped",
    ],
    "peace": [
        "peaceful", "calm", "serene", "tranquil", "safe", "warm",
        "gentle", "soothing", "relaxed", "comfort", "embrace",
    ],
    "wonder": [
        "beautiful", "magical", "mysterious", "surreal", "vivid",
        "luminous", "ethereal", "transcendent", "awe", "cosmic",
    ],
}

_THEME_KEYWORDS: Dict[str, List[str]] = {
    "flying": ["fly", "flying", "soar", "soaring", "float", "levitate", "wings", "glide"],
    "falling": ["fall", "falling", "plunge", "drop", "plummeting", "abyss"],
    "pursuit": ["chase", "chased", "chasing", "run", "running", "escape", "flee", "pursued"],
    "water": ["ocean", "sea", "river", "lake", "flood", "swim", "drown", "wave", "water"],
    "teeth": ["teeth", "tooth", "dental", "crumble", "jaw"],
    "exam": ["exam", "test", "school", "unprepared", "late", "class", "fail"],
    "death": ["death", "dying", "dead", "funeral", "grave", "coffin", "killed"],
    "transformation": ["transform", "change", "metamorphosis", "butterfly", "rebirth", "morph"],
    "lucid": ["lucid", "aware", "conscious", "realized dreaming", "controlled dream"],
    "social": ["party", "crowd", "people", "friend", "family", "gathering", "wedding"],
    "nature": ["forest", "mountain", "garden", "tree", "flower", "sun", "moon", "star"],
    "travel": ["road", "journey", "travel", "car", "train", "plane", "bridge", "path"],
    "nightmare": ["monster", "demon", "evil", "scream", "attack", "dark", "shadow", "nightmare"],
}


# -- Dataclasses -------------------------------------------------------------

@dataclass
class DreamEntry:
    """A single dream journal entry."""
    text: str
    date: str = ""
    emotions: List[str] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    lucidity_score: float = 0.0


@dataclass
class SleepMetrics:
    """Sleep architecture metrics for a single night."""
    rem_pct: float = 0.22          # REM fraction 0-1
    deep_pct: float = 0.20         # N3 deep sleep fraction 0-1
    efficiency: float = 0.85       # time-asleep / time-in-bed 0-1
    duration_h: float = 7.5        # total sleep hours
    awakenings: int = 0            # number of awakenings during the night


@dataclass
class DreamSleepCorrelation:
    """Correlation result between a dream entry and sleep metrics."""
    dream_date: str
    primary_emotion: str
    primary_theme: str
    emotions: Dict[str, float]
    themes: List[str]
    sleep_quality_score: float     # 0-100
    correlation_scores: Dict[str, float]
    insights: List[str]
    nightmare_risk: float          # 0-1
    lucidity_score: float          # 0-1


@dataclass
class DreamSleepProfile:
    """Aggregated profile across multiple dream-sleep pairs."""
    total_entries: int
    dominant_emotion: str
    dominant_theme: str
    avg_sleep_quality: float
    avg_nightmare_risk: float
    theme_frequency: Dict[str, int]
    emotion_frequency: Dict[str, int]
    theme_sleep_correlations: Dict[str, Dict[str, float]]
    insights: List[str]
    dream_type_predictions: Dict[str, float]


# -- Reference constants for sleep scoring -----------------------------------

_REF_REM_PCT = 0.22
_REF_DEEP_PCT = 0.20
_REF_EFFICIENCY = 0.85
_REF_DURATION_H = 8.0


# -- Core analysis functions -------------------------------------------------

def analyze_dream_content(text: str) -> Dict[str, Any]:
    """Extract themes and emotions from dream journal text.

    Parameters
    ----------
    text : str
        Free-text dream journal entry.

    Returns
    -------
    dict with keys:
        emotions     — dict[str, float] mapping emotion name to intensity 0-1
        themes       — list of detected theme names
        primary_emotion — highest-scoring emotion
        primary_theme   — highest-scoring theme (or "unclassified")
        lucidity_score  — 0-1 probability of lucid dreaming markers
        word_count      — int
    """
    if not text or not text.strip():
        return {
            "emotions": {},
            "themes": [],
            "primary_emotion": "neutral",
            "primary_theme": "unclassified",
            "lucidity_score": 0.0,
            "word_count": 0,
        }

    lower = text.lower()
    words = re.findall(r"[a-z]+", lower)
    word_count = len(words)
    word_set = set(words)

    # -- Emotion scoring --
    emotions: Dict[str, float] = {}
    for emotion, keywords in _EMOTION_LEXICON.items():
        hits = sum(1 for kw in keywords if kw in word_set)
        if hits > 0:
            intensity = min(1.0, hits / max(word_count * 0.03, 1.0))
            emotions[emotion] = round(intensity, 3)

    primary_emotion = max(emotions, key=emotions.get) if emotions else "neutral"

    # -- Theme scoring --
    theme_scores: Dict[str, float] = {}
    for theme, keywords in _THEME_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits > 0:
            score = min(1.0, hits / max(len(keywords) * 0.4, 1.0))
            theme_scores[theme] = score

    themes_detected = sorted(theme_scores, key=theme_scores.get, reverse=True)
    primary_theme = themes_detected[0] if themes_detected else "unclassified"

    # -- Lucidity scoring --
    lucid_markers = [
        "realized i was dreaming", "knew i was dreaming", "lucid dream",
        "conscious in the dream", "aware i was dreaming", "controlled the dream",
        "became aware i was dreaming",
    ]
    lucidity_score = 0.0
    for marker in lucid_markers:
        if marker in lower:
            lucidity_score = 0.9
            break
    if lucidity_score == 0.0:
        soft = sum(1 for w in ["lucid", "aware", "conscious", "control"] if w in word_set)
        lucidity_score = min(0.6, soft * 0.2)

    return {
        "emotions": emotions,
        "themes": themes_detected,
        "primary_emotion": primary_emotion,
        "primary_theme": primary_theme,
        "lucidity_score": round(lucidity_score, 3),
        "word_count": word_count,
    }


def _compute_sleep_quality(metrics: SleepMetrics) -> float:
    """Compute a 0-100 sleep quality score from sleep metrics."""
    # Duration component (optimal 7-9h)
    if 7.0 <= metrics.duration_h <= 9.0:
        dur_score = 1.0
    elif metrics.duration_h < 7.0:
        dur_score = max(0.0, metrics.duration_h / 7.0)
    else:
        dur_score = max(0.0, 1.0 - (metrics.duration_h - 9.0) / 3.0)

    # Deep sleep component
    deep_score = min(1.0, metrics.deep_pct / _REF_DEEP_PCT)

    # REM component
    rem_score = min(1.0, metrics.rem_pct / _REF_REM_PCT)

    # Efficiency component
    eff_score = max(0.0, min(1.0, (metrics.efficiency - 0.70) / 0.30))

    # Awakenings penalty
    awk_score = max(0.0, 1.0 - metrics.awakenings / 8.0)

    composite = (
        0.25 * dur_score
        + 0.25 * deep_score
        + 0.20 * rem_score
        + 0.20 * eff_score
        + 0.10 * awk_score
    )
    return round(max(0.0, min(100.0, composite * 100.0)), 1)


def _compute_nightmare_risk(
    emotions: Dict[str, float],
    metrics: SleepMetrics,
) -> float:
    """Estimate nightmare risk from dream emotions + sleep architecture.

    High REM + low efficiency + negative emotions = elevated nightmare risk.
    Based on Scarpelli et al. (2019): fragmented REM sleep with high emotional
    load predicts nightmare frequency.
    """
    # Emotional negativity component
    neg_emotions = emotions.get("fear", 0.0) + emotions.get("anger", 0.0) + emotions.get("anxiety", 0.0)
    neg_score = min(1.0, neg_emotions / 1.5)

    # Elevated REM above reference -> more dream activity
    rem_excess = max(0.0, (metrics.rem_pct - _REF_REM_PCT) / _REF_REM_PCT)

    # Poor efficiency -> fragmented sleep -> nightmare-prone
    eff_deficit = max(0.0, (_REF_EFFICIENCY - metrics.efficiency) / _REF_EFFICIENCY)

    # Frequent awakenings compound risk
    awk_factor = min(1.0, metrics.awakenings / 6.0)

    risk = 0.40 * neg_score + 0.25 * rem_excess + 0.20 * eff_deficit + 0.15 * awk_factor
    return round(max(0.0, min(1.0, risk)), 3)


def correlate_dream_sleep(
    dream: DreamEntry,
    sleep: SleepMetrics,
) -> DreamSleepCorrelation:
    """Correlate a single dream entry with its night's sleep metrics.

    Parameters
    ----------
    dream : DreamEntry
        Dream journal entry (text + optional metadata).
    sleep : SleepMetrics
        Sleep architecture from the same night.

    Returns
    -------
    DreamSleepCorrelation with scores and insights.
    """
    content = analyze_dream_content(dream.text)
    emotions = content["emotions"]
    themes = content["themes"]
    primary_emotion = content["primary_emotion"]
    primary_theme = content["primary_theme"]
    lucidity = max(content["lucidity_score"], dream.lucidity_score)

    sleep_quality = _compute_sleep_quality(sleep)
    nightmare_risk = _compute_nightmare_risk(emotions, sleep)

    # Correlation scores: how each sleep metric relates to dream characteristics
    correlations: Dict[str, float] = {}

    # REM correlates with dream vividness and emotional intensity
    total_emotion_intensity = sum(emotions.values()) if emotions else 0.0
    if sleep.rem_pct > 0:
        correlations["rem_emotion_correlation"] = round(
            min(1.0, total_emotion_intensity * (sleep.rem_pct / _REF_REM_PCT)), 3
        )
    else:
        correlations["rem_emotion_correlation"] = 0.0

    # Deep sleep inversely correlates with dream recall (fewer dreams in N3)
    correlations["deep_sleep_dream_suppression"] = round(
        min(1.0, sleep.deep_pct / _REF_DEEP_PCT), 3
    )

    # Sleep efficiency correlates with dream pleasantness
    positive_emotions = emotions.get("joy", 0.0) + emotions.get("peace", 0.0) + emotions.get("wonder", 0.0)
    correlations["efficiency_pleasantness"] = round(
        sleep.efficiency * min(1.0, positive_emotions + 0.3), 3
    )

    # Awakenings correlate with dream fragmentation / nightmare frequency
    correlations["awakening_nightmare_link"] = round(
        min(1.0, (nightmare_risk + min(1.0, sleep.awakenings / 5.0)) / 2.0), 3
    )

    # Generate insights
    insights = _generate_correlation_insights(
        primary_emotion=primary_emotion,
        primary_theme=primary_theme,
        emotions=emotions,
        sleep=sleep,
        sleep_quality=sleep_quality,
        nightmare_risk=nightmare_risk,
        lucidity=lucidity,
    )

    return DreamSleepCorrelation(
        dream_date=dream.date,
        primary_emotion=primary_emotion,
        primary_theme=primary_theme,
        emotions=emotions,
        themes=themes,
        sleep_quality_score=sleep_quality,
        correlation_scores=correlations,
        insights=insights,
        nightmare_risk=nightmare_risk,
        lucidity_score=round(lucidity, 3),
    )


def _generate_correlation_insights(
    primary_emotion: str,
    primary_theme: str,
    emotions: Dict[str, float],
    sleep: SleepMetrics,
    sleep_quality: float,
    nightmare_risk: float,
    lucidity: float,
) -> List[str]:
    """Generate human-readable insights from dream-sleep correlation."""
    insights: List[str] = []

    # REM-based insights
    if sleep.rem_pct > 0.28:
        insights.append(
            f"REM sleep is elevated ({sleep.rem_pct:.0%}). High REM increases "
            "dream vividness and emotional intensity (Hobson & McCarley, 1977)."
        )
    elif sleep.rem_pct < 0.15:
        insights.append(
            f"REM sleep is low ({sleep.rem_pct:.0%}). Reduced REM may impair "
            "emotional processing and dream recall."
        )

    # Nightmare risk
    if nightmare_risk > 0.5:
        insights.append(
            f"Elevated nightmare risk ({nightmare_risk:.0%}). Fragmented sleep "
            "with negative emotional content is a known nightmare predictor "
            "(Scarpelli et al., 2019)."
        )

    # Deep sleep + dream suppression
    if sleep.deep_pct > 0.25 and not emotions:
        insights.append(
            "High deep sleep with minimal dream content — N3 (slow-wave) sleep "
            "suppresses dream generation. Dreams are most vivid during REM."
        )

    # Efficiency + pleasantness
    if sleep.efficiency >= 0.90 and primary_emotion in ("joy", "peace", "wonder"):
        insights.append(
            "Good sleep efficiency with positive dream content. Consolidated sleep "
            "is associated with more pleasant dream experiences."
        )
    elif sleep.efficiency < 0.75:
        insights.append(
            f"Sleep efficiency is low ({sleep.efficiency:.0%}). Fragmented sleep "
            "tends to produce more negative and anxiety-laden dream content."
        )

    # Theme-specific
    if primary_theme == "pursuit" and sleep.awakenings >= 3:
        insights.append(
            "Chase/pursuit dreams with frequent awakenings — this pattern often "
            "reflects unresolved daytime anxiety disrupting sleep architecture."
        )
    if primary_theme == "flying" and sleep.rem_pct > 0.22:
        insights.append(
            "Flying dreams during high-REM nights — associated with a sense of "
            "agency and emotional liberation."
        )

    # Lucidity
    if lucidity > 0.5:
        insights.append(
            "Lucid dreaming detected. Lucid dreams are most common during "
            "late-cycle REM periods with high cortical activation."
        )

    if not insights:
        insights.append(
            "Dream content and sleep architecture are within normal ranges. "
            "Patterns become more meaningful after 7+ paired entries."
        )

    return insights


# -- Dream type prediction ---------------------------------------------------

# Mapping from sleep conditions to likely dream types with base probabilities.
# Each condition adjusts probabilities up or down.
_DREAM_TYPE_BASE: Dict[str, float] = {
    "vivid_positive": 0.20,
    "vivid_negative": 0.10,
    "nightmare": 0.05,
    "lucid": 0.05,
    "mundane": 0.30,
    "abstract": 0.15,
    "no_recall": 0.15,
}


def predict_dream_type(
    sleep: SleepMetrics,
    recent_stress: float = 0.0,
    recent_emotions: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Predict dream likelihood and type distribution from tonight's sleep data.

    Parameters
    ----------
    sleep : SleepMetrics
        Current night's sleep architecture.
    recent_stress : float
        Recent stress level 0-1 (from EEG stress detector or self-report).
    recent_emotions : dict, optional
        Recent emotional state (e.g., from pre-sleep mood check-in).

    Returns
    -------
    dict with:
        dream_probabilities — dict[str, float] summing to ~1.0
        predicted_type      — most likely dream type
        nightmare_risk      — 0-1
        recall_probability  — 0-1 (chance of remembering)
        insights            — list of strings
    """
    if recent_emotions is None:
        recent_emotions = {}

    probs = dict(_DREAM_TYPE_BASE)

    # -- REM adjustments --
    rem_ratio = sleep.rem_pct / _REF_REM_PCT
    if rem_ratio > 1.2:
        # High REM -> more vivid dreams, less no-recall
        probs["vivid_positive"] += 0.10
        probs["vivid_negative"] += 0.05
        probs["no_recall"] -= 0.10
        probs["lucid"] += 0.03
    elif rem_ratio < 0.7:
        # Low REM -> fewer vivid dreams, more no-recall
        probs["vivid_positive"] -= 0.10
        probs["vivid_negative"] -= 0.05
        probs["no_recall"] += 0.15
        probs["mundane"] += 0.05

    # -- Efficiency adjustments --
    if sleep.efficiency < 0.75:
        # Fragmented sleep -> more nightmares, less positive
        probs["nightmare"] += 0.10
        probs["vivid_negative"] += 0.08
        probs["vivid_positive"] -= 0.08
    elif sleep.efficiency >= 0.90:
        probs["vivid_positive"] += 0.05
        probs["nightmare"] -= 0.03

    # -- Stress adjustments --
    if recent_stress > 0.6:
        probs["nightmare"] += 0.12
        probs["vivid_negative"] += 0.08
        probs["vivid_positive"] -= 0.05
        probs["mundane"] -= 0.05
    elif recent_stress < 0.2:
        probs["vivid_positive"] += 0.05
        probs["nightmare"] -= 0.03

    # -- Deep sleep adjustments --
    if sleep.deep_pct > 0.30:
        # Very high deep sleep -> fewer dreams recalled
        probs["no_recall"] += 0.10
        probs["mundane"] += 0.05
        probs["vivid_positive"] -= 0.05

    # -- Awakenings --
    if sleep.awakenings >= 4:
        probs["nightmare"] += 0.05
        probs["vivid_negative"] += 0.05
        probs["no_recall"] -= 0.05

    # -- Recent emotions --
    neg_load = (
        recent_emotions.get("fear", 0.0)
        + recent_emotions.get("anger", 0.0)
        + recent_emotions.get("anxiety", 0.0)
    )
    pos_load = (
        recent_emotions.get("joy", 0.0)
        + recent_emotions.get("peace", 0.0)
    )
    if neg_load > 0.5:
        probs["nightmare"] += 0.08
        probs["vivid_negative"] += 0.05
    if pos_load > 0.5:
        probs["vivid_positive"] += 0.08
        probs["nightmare"] -= 0.03

    # -- Clamp and normalize --
    for k in probs:
        probs[k] = max(0.01, probs[k])  # floor at 1%
    total = sum(probs.values())
    probs = {k: round(v / total, 3) for k, v in probs.items()}

    predicted_type = max(probs, key=probs.get)

    # Nightmare risk (dedicated score)
    nightmare_risk = round(
        min(1.0, probs.get("nightmare", 0.0) + probs.get("vivid_negative", 0.0) * 0.3),
        3,
    )

    # Recall probability
    recall_prob = round(min(1.0, max(0.1, 0.3 + (sleep.rem_pct / _REF_REM_PCT - 0.5) * 0.5)), 3)

    # Insights
    insights = _generate_prediction_insights(sleep, recent_stress, predicted_type, nightmare_risk)

    return {
        "dream_probabilities": probs,
        "predicted_type": predicted_type,
        "nightmare_risk": nightmare_risk,
        "recall_probability": recall_prob,
        "insights": insights,
    }


def _generate_prediction_insights(
    sleep: SleepMetrics,
    stress: float,
    predicted_type: str,
    nightmare_risk: float,
) -> List[str]:
    """Generate insights for dream type prediction."""
    insights: List[str] = []

    if nightmare_risk > 0.3:
        insights.append(
            "Elevated nightmare risk tonight. Consider a brief relaxation "
            "exercise or Imagery Rehearsal Therapy (IRT) before sleep."
        )

    if sleep.rem_pct > 0.28 and stress > 0.5:
        insights.append(
            "High REM combined with elevated stress increases vivid negative "
            "dream probability. Pre-sleep stress reduction recommended."
        )

    if predicted_type == "lucid":
        insights.append(
            "Conditions favor lucid dreaming. If desired, practice reality "
            "checks before sleep to increase lucid dream probability."
        )

    if sleep.duration_h < 6.0:
        insights.append(
            "Short sleep duration reduces REM proportion (REM is concentrated "
            "in later cycles). Dream recall will likely be low."
        )

    if sleep.efficiency >= 0.90 and stress < 0.3:
        insights.append(
            "Good sleep quality with low stress — conditions favor pleasant, "
            "restorative dream content."
        )

    if not insights:
        insights.append(
            "Sleep conditions are average. Dream type prediction improves "
            "with more historical dream-sleep pairs."
        )

    return insights


# -- Profile computation (multi-entry aggregation) ---------------------------

def compute_dream_sleep_profile(
    dreams: List[DreamEntry],
    sleep_data: List[SleepMetrics],
) -> DreamSleepProfile:
    """Compute an aggregated dream-sleep profile from multiple paired entries.

    Parameters
    ----------
    dreams : list[DreamEntry]
        Dream journal entries (one per night).
    sleep_data : list[SleepMetrics]
        Matching sleep metrics (same length as dreams).

    Returns
    -------
    DreamSleepProfile with aggregated statistics and correlations.
    """
    n = min(len(dreams), len(sleep_data))
    if n == 0:
        return DreamSleepProfile(
            total_entries=0,
            dominant_emotion="neutral",
            dominant_theme="unclassified",
            avg_sleep_quality=0.0,
            avg_nightmare_risk=0.0,
            theme_frequency={},
            emotion_frequency={},
            theme_sleep_correlations={},
            insights=["No entries provided. Log dream-sleep pairs to build your profile."],
            dream_type_predictions={},
        )

    theme_freq: Dict[str, int] = defaultdict(int)
    emotion_freq: Dict[str, int] = defaultdict(int)
    sleep_qualities: List[float] = []
    nightmare_risks: List[float] = []

    # Per-theme sleep metric accumulators for correlation
    theme_sleep_rem: Dict[str, List[float]] = defaultdict(list)
    theme_sleep_deep: Dict[str, List[float]] = defaultdict(list)
    theme_sleep_eff: Dict[str, List[float]] = defaultdict(list)

    correlations: List[DreamSleepCorrelation] = []

    for i in range(n):
        corr = correlate_dream_sleep(dreams[i], sleep_data[i])
        correlations.append(corr)

        sleep_qualities.append(corr.sleep_quality_score)
        nightmare_risks.append(corr.nightmare_risk)

        for theme in corr.themes:
            theme_freq[theme] += 1
            theme_sleep_rem[theme].append(sleep_data[i].rem_pct)
            theme_sleep_deep[theme].append(sleep_data[i].deep_pct)
            theme_sleep_eff[theme].append(sleep_data[i].efficiency)

        for emotion in corr.emotions:
            emotion_freq[emotion] += 1

    # Dominant emotion and theme
    dominant_emotion = max(emotion_freq, key=emotion_freq.get) if emotion_freq else "neutral"
    dominant_theme = max(theme_freq, key=theme_freq.get) if theme_freq else "unclassified"

    # Theme-sleep correlations (average sleep metrics per theme)
    theme_sleep_corrs: Dict[str, Dict[str, float]] = {}
    for theme in theme_freq:
        rem_vals = theme_sleep_rem[theme]
        deep_vals = theme_sleep_deep[theme]
        eff_vals = theme_sleep_eff[theme]
        theme_sleep_corrs[theme] = {
            "avg_rem_pct": round(sum(rem_vals) / len(rem_vals), 3) if rem_vals else 0.0,
            "avg_deep_pct": round(sum(deep_vals) / len(deep_vals), 3) if deep_vals else 0.0,
            "avg_efficiency": round(sum(eff_vals) / len(eff_vals), 3) if eff_vals else 0.0,
            "occurrence_count": theme_freq[theme],
        }

    # Aggregate dream type prediction from average sleep
    avg_sleep = SleepMetrics(
        rem_pct=sum(s.rem_pct for s in sleep_data[:n]) / n,
        deep_pct=sum(s.deep_pct for s in sleep_data[:n]) / n,
        efficiency=sum(s.efficiency for s in sleep_data[:n]) / n,
        duration_h=sum(s.duration_h for s in sleep_data[:n]) / n,
        awakenings=round(sum(s.awakenings for s in sleep_data[:n]) / n),
    )
    type_pred = predict_dream_type(avg_sleep)

    # Profile insights
    insights = _generate_profile_insights(
        n=n,
        dominant_emotion=dominant_emotion,
        dominant_theme=dominant_theme,
        avg_quality=sum(sleep_qualities) / n,
        avg_nightmare=sum(nightmare_risks) / n,
        theme_sleep_corrs=theme_sleep_corrs,
    )

    return DreamSleepProfile(
        total_entries=n,
        dominant_emotion=dominant_emotion,
        dominant_theme=dominant_theme,
        avg_sleep_quality=round(sum(sleep_qualities) / n, 1),
        avg_nightmare_risk=round(sum(nightmare_risks) / n, 3),
        theme_frequency=dict(theme_freq),
        emotion_frequency=dict(emotion_freq),
        theme_sleep_correlations=theme_sleep_corrs,
        insights=insights,
        dream_type_predictions=type_pred["dream_probabilities"],
    )


def _generate_profile_insights(
    n: int,
    dominant_emotion: str,
    dominant_theme: str,
    avg_quality: float,
    avg_nightmare: float,
    theme_sleep_corrs: Dict[str, Dict[str, float]],
) -> List[str]:
    """Generate profile-level insights."""
    insights: List[str] = []

    insights.append(
        f"Profile based on {n} dream-sleep pairs. "
        f"Dominant emotion: {dominant_emotion}, dominant theme: {dominant_theme}."
    )

    if avg_quality >= 80:
        insights.append("Overall sleep quality is good. Dream content tends to reflect this positively.")
    elif avg_quality < 60:
        insights.append(
            "Sleep quality is below average. Improving sleep hygiene may reduce "
            "negative dream content and nightmare frequency."
        )

    if avg_nightmare > 0.3:
        insights.append(
            f"Average nightmare risk is elevated ({avg_nightmare:.0%}). "
            "Consider stress reduction techniques and IRT before bed."
        )

    # Theme-specific sleep correlations
    for theme, corrs in theme_sleep_corrs.items():
        if corrs["occurrence_count"] >= 3:
            if corrs["avg_rem_pct"] > 0.28:
                insights.append(
                    f"'{theme}' dreams consistently occur on high-REM nights "
                    f"(avg {corrs['avg_rem_pct']:.0%} REM)."
                )
            if corrs["avg_efficiency"] < 0.80 and theme in ("nightmare", "pursuit", "falling"):
                insights.append(
                    f"'{theme}' dreams correlate with poor sleep efficiency "
                    f"(avg {corrs['avg_efficiency']:.0%}). Improving sleep consolidation may help."
                )

    return insights


def profile_to_dict(profile: DreamSleepProfile) -> Dict[str, Any]:
    """Convert a DreamSleepProfile dataclass to a JSON-serializable dict."""
    return {
        "total_entries": profile.total_entries,
        "dominant_emotion": profile.dominant_emotion,
        "dominant_theme": profile.dominant_theme,
        "avg_sleep_quality": profile.avg_sleep_quality,
        "avg_nightmare_risk": profile.avg_nightmare_risk,
        "theme_frequency": profile.theme_frequency,
        "emotion_frequency": profile.emotion_frequency,
        "theme_sleep_correlations": profile.theme_sleep_correlations,
        "insights": profile.insights,
        "dream_type_predictions": profile.dream_type_predictions,
    }


# -- Module-level singleton --------------------------------------------------

class DreamSleepFusionModel:
    """Facade class for the dream-sleep fusion system."""

    def analyze_content(self, text: str) -> Dict[str, Any]:
        return analyze_dream_content(text)

    def correlate(
        self, dream: DreamEntry, sleep: SleepMetrics
    ) -> DreamSleepCorrelation:
        return correlate_dream_sleep(dream, sleep)

    def predict_type(
        self,
        sleep: SleepMetrics,
        recent_stress: float = 0.0,
        recent_emotions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        return predict_dream_type(sleep, recent_stress, recent_emotions)

    def compute_profile(
        self, dreams: List[DreamEntry], sleep_data: List[SleepMetrics]
    ) -> DreamSleepProfile:
        return compute_dream_sleep_profile(dreams, sleep_data)


_model: Optional[DreamSleepFusionModel] = None


def get_dream_sleep_fusion_model() -> DreamSleepFusionModel:
    """Return (or lazily create) the module-level DreamSleepFusionModel."""
    global _model
    if _model is None:
        _model = DreamSleepFusionModel()
        log.info("DreamSleepFusionModel initialized (heuristic-based, no weights)")
    return _model
