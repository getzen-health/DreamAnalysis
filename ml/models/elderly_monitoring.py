"""Elderly cognitive-emotional monitoring for early MCI/dementia detection.

Tracks emotional processing decline as an early marker of mild cognitive
impairment (MCI) and dementia.  Emotional changes often precede measurable
cognitive decline by months or years.

Key markers tracked:
  - Emotional processing speed (reaction time to emotional stimuli)
  - Emotional flattening (narrowing emotional range over time)
  - Cognitive-emotional coupling (co-decline of cognition and emotion)
  - Emotional memory advantage (emotionally salient events remembered better)
  - Social engagement decline (early dementia marker)
  - Composite MCI risk score combining all indicators

References:
    Henry et al. (2009) -- Emotion processing in MCI and early AD
    Bediou et al. (2009) -- Facial emotion recognition in MCI
    Sturm et al. (2013) -- Emotional flattening in neurodegenerative disease
    Carstensen (2006) -- Socioemotional selectivity and aging
    Kensinger (2008) -- Emotional memory advantage across the lifespan
    Ismail et al. (2016) -- Neuropsychiatric symptoms as early MCI markers

Issue #448.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Processing speed norms (milliseconds) -- healthy elderly adults
_HEALTHY_REACTION_TIME_MS = 850.0       # mean RT to emotional faces
_HEALTHY_IDENTIFICATION_TIME_MS = 1200.0  # mean time to label an emotion

# Emotional range norms (number of distinct emotions per week)
_HEALTHY_EMOTION_RANGE = 6.0    # typical distinct emotions reported
_MIN_RANGE_FOR_CONCERN = 3.0    # below this = flattening signal

# Social engagement norms (interactions per week)
_HEALTHY_SOCIAL_INTERACTIONS = 10.0
_LOW_SOCIAL_THRESHOLD = 4.0

# Emotional memory advantage norms
_HEALTHY_EMA_RATIO = 1.35  # emotional items recalled 35% better than neutral

# Composite MCI risk thresholds
_LOW_RISK = 0.25
_MILD_RISK = 0.45
_MODERATE_RISK = 0.65

# Weights for composite score
_W_PROCESSING_SPEED = 0.20
_W_FLATTENING = 0.20
_W_COUPLING = 0.20
_W_MEMORY = 0.20
_W_SOCIAL = 0.20

_CLINICAL_DISCLAIMER = (
    "This is a screening tool for research and wellness purposes only. "
    "It does NOT constitute a clinical diagnosis of MCI, dementia, or any "
    "neurological condition. Always consult a qualified healthcare professional "
    "for clinical evaluation."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ProcessingSpeedReading:
    """Single emotional processing speed measurement."""
    reaction_time_ms: float          # RT to emotional stimulus
    identification_time_ms: float    # time to identify the emotion
    stimulus_type: str = "face"      # "face", "voice", "scene"
    emotion_presented: str = "neutral"
    correct: bool = True
    timestamp: float = 0.0


@dataclass
class EmotionRangeReading:
    """Snapshot of emotional range over a time window."""
    distinct_emotions: int           # count of distinct emotions experienced
    dominant_emotion: str = "neutral"
    valence_range: float = 0.0       # max_valence - min_valence observed
    arousal_range: float = 0.0       # max_arousal - min_arousal observed
    window_days: int = 7
    timestamp: float = 0.0


@dataclass
class SocialEngagementReading:
    """Social interaction metrics over a time window."""
    interactions_count: int          # total social interactions
    unique_contacts: int = 0         # distinct people interacted with
    initiated_count: int = 0         # self-initiated interactions
    avg_duration_minutes: float = 0.0
    window_days: int = 7
    timestamp: float = 0.0


@dataclass
class EmotionalMemoryReading:
    """Emotional memory advantage measurement."""
    emotional_items_recalled: int    # out of emotional items presented
    neutral_items_recalled: int      # out of neutral items presented
    emotional_items_total: int = 10
    neutral_items_total: int = 10
    delay_minutes: int = 30          # delay between encoding and recall
    timestamp: float = 0.0


@dataclass
class CognitiveReading:
    """Cognitive performance snapshot."""
    attention_score: float = 0.5     # 0-1 attention index
    memory_score: float = 0.5        # 0-1 memory index
    executive_score: float = 0.5     # 0-1 executive function index
    processing_speed_score: float = 0.5  # 0-1 cognitive processing speed
    timestamp: float = 0.0


@dataclass
class MonitoringProfile:
    """Full elderly cognitive-emotional monitoring profile."""
    processing_speed_score: float = 0.0
    flattening_score: float = 0.0
    coupling_score: float = 0.0
    emotional_memory_score: float = 0.0
    social_engagement_score: float = 0.0
    mci_risk_score: float = 0.0
    risk_category: str = "low_risk"
    processing_speed_detail: Dict[str, Any] = field(default_factory=dict)
    flattening_detail: Dict[str, Any] = field(default_factory=dict)
    coupling_detail: Dict[str, Any] = field(default_factory=dict)
    memory_detail: Dict[str, Any] = field(default_factory=dict)
    social_detail: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    disclaimer: str = _CLINICAL_DISCLAIMER


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------


def compute_processing_speed(
    readings: List[ProcessingSpeedReading],
) -> Dict[str, Any]:
    """Compute emotional processing speed score from reaction-time data.

    Slower reaction times and identification times to emotional stimuli
    are early markers of MCI (Henry et al., 2009).

    Returns:
        Dict with score (0-1, higher = more concern), mean_rt_ms,
        mean_id_time_ms, accuracy, and flag list.
    """
    if not readings:
        return {
            "score": 0.0,
            "mean_rt_ms": 0.0,
            "mean_id_time_ms": 0.0,
            "accuracy": 1.0,
            "flags": [],
            "n_readings": 0,
        }

    rts = [r.reaction_time_ms for r in readings]
    id_times = [r.identification_time_ms for r in readings]
    accuracy = sum(1 for r in readings if r.correct) / len(readings)

    mean_rt = float(np.mean(rts))
    mean_id = float(np.mean(id_times))

    # Score: how much slower than healthy norm, 0-1
    rt_deviation = max(0.0, mean_rt - _HEALTHY_REACTION_TIME_MS) / _HEALTHY_REACTION_TIME_MS
    id_deviation = max(0.0, mean_id - _HEALTHY_IDENTIFICATION_TIME_MS) / _HEALTHY_IDENTIFICATION_TIME_MS

    # Accuracy penalty
    accuracy_penalty = max(0.0, 0.8 - accuracy)  # penalty kicks in below 80%

    score = float(np.clip(
        0.40 * min(rt_deviation, 1.0)
        + 0.35 * min(id_deviation, 1.0)
        + 0.25 * (accuracy_penalty / 0.8),
        0.0, 1.0,
    ))

    flags = []
    if mean_rt > _HEALTHY_REACTION_TIME_MS * 1.5:
        flags.append("markedly_slow_reaction_time")
    elif mean_rt > _HEALTHY_REACTION_TIME_MS * 1.2:
        flags.append("elevated_reaction_time")
    if mean_id > _HEALTHY_IDENTIFICATION_TIME_MS * 1.5:
        flags.append("markedly_slow_identification")
    if accuracy < 0.6:
        flags.append("low_accuracy")

    return {
        "score": round(score, 4),
        "mean_rt_ms": round(mean_rt, 1),
        "mean_id_time_ms": round(mean_id, 1),
        "accuracy": round(accuracy, 3),
        "flags": flags,
        "n_readings": len(readings),
    }


def detect_emotional_flattening(
    readings: List[EmotionRangeReading],
) -> Dict[str, Any]:
    """Detect emotional flattening -- narrowing emotional range over time.

    Progressive loss of emotional range is a hallmark of frontotemporal
    dementia and can appear in MCI (Sturm et al., 2013).

    Returns:
        Dict with score (0-1), current_range, trend, and flags.
    """
    if not readings:
        return {
            "score": 0.0,
            "current_range": 0,
            "valence_span": 0.0,
            "arousal_span": 0.0,
            "trend": "insufficient_data",
            "flags": [],
            "n_readings": 0,
        }

    # Use latest reading as current state
    latest = readings[-1]
    current_range = latest.distinct_emotions
    valence_span = latest.valence_range
    arousal_span = latest.arousal_range

    # Range deficit score
    range_deficit = max(0.0, _HEALTHY_EMOTION_RANGE - current_range) / _HEALTHY_EMOTION_RANGE
    valence_deficit = max(0.0, 1.0 - valence_span)  # full range = 2.0 (-1 to 1), normalized
    arousal_deficit = max(0.0, 0.8 - arousal_span)   # typical healthy range ~0.8

    score = float(np.clip(
        0.45 * range_deficit
        + 0.30 * valence_deficit
        + 0.25 * (arousal_deficit / 0.8),
        0.0, 1.0,
    ))

    # Trend detection: compare first half vs second half
    trend = "insufficient_data"
    if len(readings) >= 4:
        mid = len(readings) // 2
        early_ranges = [r.distinct_emotions for r in readings[:mid]]
        late_ranges = [r.distinct_emotions for r in readings[mid:]]
        early_mean = float(np.mean(early_ranges))
        late_mean = float(np.mean(late_ranges))
        diff = late_mean - early_mean
        if diff < -1.0:
            trend = "declining"
        elif diff > 1.0:
            trend = "improving"
        else:
            trend = "stable"

    flags = []
    if current_range <= _MIN_RANGE_FOR_CONCERN:
        flags.append("critically_narrow_range")
    elif current_range <= _HEALTHY_EMOTION_RANGE * 0.6:
        flags.append("reduced_emotional_range")
    if trend == "declining":
        flags.append("declining_range_trend")

    return {
        "score": round(score, 4),
        "current_range": current_range,
        "valence_span": round(valence_span, 3),
        "arousal_span": round(arousal_span, 3),
        "trend": trend,
        "flags": flags,
        "n_readings": len(readings),
    }


def compute_cognitive_emotional_coupling(
    cognitive_readings: List[CognitiveReading],
    emotion_range_readings: List[EmotionRangeReading],
) -> Dict[str, Any]:
    """Compute cognitive-emotional coupling -- do they decline together?

    In healthy aging, cognitive and emotional capacities can diverge.
    In MCI/dementia, they often co-decline, particularly emotional
    processing and executive function (Henry et al., 2009).

    Returns:
        Dict with score (0-1), coupling_strength, both_declining flag.
    """
    if not cognitive_readings or not emotion_range_readings:
        return {
            "score": 0.0,
            "coupling_strength": 0.0,
            "cognitive_trend": "insufficient_data",
            "emotional_trend": "insufficient_data",
            "both_declining": False,
            "flags": [],
        }

    # Compute cognitive composite over time
    cog_scores = [
        (r.attention_score + r.memory_score + r.executive_score + r.processing_speed_score) / 4.0
        for r in cognitive_readings
    ]

    # Compute emotional range over time (normalized)
    emo_scores = [
        r.distinct_emotions / _HEALTHY_EMOTION_RANGE
        for r in emotion_range_readings
    ]

    # Determine trends
    def _trend(values: List[float]) -> str:
        if len(values) < 2:
            return "insufficient_data"
        first_half = float(np.mean(values[: len(values) // 2]))
        second_half = float(np.mean(values[len(values) // 2:]))
        diff = second_half - first_half
        if diff < -0.1:
            return "declining"
        elif diff > 0.1:
            return "improving"
        return "stable"

    cog_trend = _trend(cog_scores)
    emo_trend = _trend(emo_scores)

    # Coupling: correlation between cognitive and emotional scores
    min_len = min(len(cog_scores), len(emo_scores))
    if min_len >= 3:
        cog_arr = np.array(cog_scores[:min_len])
        emo_arr = np.array(emo_scores[:min_len])
        # Avoid division by zero in correlation
        if np.std(cog_arr) > 1e-9 and np.std(emo_arr) > 1e-9:
            coupling = float(np.corrcoef(cog_arr, emo_arr)[0, 1])
        else:
            coupling = 0.0
    else:
        coupling = 0.0

    both_declining = cog_trend == "declining" and emo_trend == "declining"

    # Score: high coupling + both declining = concern
    score_base = max(0.0, coupling) * 0.5  # positive coupling matters
    if both_declining:
        score_base += 0.4
    elif cog_trend == "declining" or emo_trend == "declining":
        score_base += 0.2

    score = float(np.clip(score_base, 0.0, 1.0))

    flags = []
    if both_declining:
        flags.append("cognitive_emotional_co_decline")
    if coupling > 0.7:
        flags.append("strong_coupling")

    return {
        "score": round(score, 4),
        "coupling_strength": round(coupling, 4),
        "cognitive_trend": cog_trend,
        "emotional_trend": emo_trend,
        "both_declining": both_declining,
        "flags": flags,
    }


def assess_emotional_memory(
    readings: List[EmotionalMemoryReading],
) -> Dict[str, Any]:
    """Assess emotional memory advantage -- emotional items remembered better.

    In healthy brains, emotional events are remembered better than neutral
    ones (emotional memory advantage, EMA). Loss of this advantage is an
    early sign of amygdala-hippocampal circuit dysfunction in MCI
    (Kensinger, 2008).

    Returns:
        Dict with score (0-1), ema_ratio, trend, and flags.
    """
    if not readings:
        return {
            "score": 0.0,
            "ema_ratio": _HEALTHY_EMA_RATIO,
            "emotional_recall_rate": 0.0,
            "neutral_recall_rate": 0.0,
            "advantage_preserved": True,
            "trend": "insufficient_data",
            "flags": [],
            "n_readings": 0,
        }

    # Compute EMA ratio for each reading
    ratios = []
    for r in readings:
        emo_rate = r.emotional_items_recalled / max(1, r.emotional_items_total)
        neu_rate = r.neutral_items_recalled / max(1, r.neutral_items_total)
        ratio = emo_rate / max(0.01, neu_rate)
        ratios.append(ratio)

    latest = readings[-1]
    emo_rate = latest.emotional_items_recalled / max(1, latest.emotional_items_total)
    neu_rate = latest.neutral_items_recalled / max(1, latest.neutral_items_total)
    current_ratio = ratios[-1]

    # Score: how much EMA is diminished compared to healthy norm
    advantage_loss = max(0.0, _HEALTHY_EMA_RATIO - current_ratio) / _HEALTHY_EMA_RATIO
    advantage_preserved = current_ratio >= 1.1  # at least 10% advantage

    score = float(np.clip(advantage_loss, 0.0, 1.0))

    # Trend
    trend = "insufficient_data"
    if len(ratios) >= 4:
        mid = len(ratios) // 2
        early_mean = float(np.mean(ratios[:mid]))
        late_mean = float(np.mean(ratios[mid:]))
        diff = late_mean - early_mean
        if diff < -0.1:
            trend = "declining"
        elif diff > 0.1:
            trend = "improving"
        else:
            trend = "stable"

    flags = []
    if current_ratio < 1.0:
        flags.append("emotional_memory_advantage_lost")
    elif current_ratio < 1.1:
        flags.append("emotional_memory_advantage_marginal")
    if trend == "declining":
        flags.append("declining_ema_trend")

    return {
        "score": round(score, 4),
        "ema_ratio": round(current_ratio, 3),
        "emotional_recall_rate": round(emo_rate, 3),
        "neutral_recall_rate": round(neu_rate, 3),
        "advantage_preserved": advantage_preserved,
        "trend": trend,
        "flags": flags,
        "n_readings": len(readings),
    }


def _compute_social_engagement(
    readings: List[SocialEngagementReading],
) -> Dict[str, Any]:
    """Compute social engagement score.

    Declining social interaction is an early marker of dementia, even
    after controlling for depression (Ismail et al., 2016).

    Returns:
        Dict with score (0-1), current metrics, trend, and flags.
    """
    if not readings:
        return {
            "score": 0.0,
            "interactions_per_week": 0.0,
            "unique_contacts": 0,
            "initiated_ratio": 0.0,
            "trend": "insufficient_data",
            "flags": [],
            "n_readings": 0,
        }

    latest = readings[-1]
    interactions = latest.interactions_count
    unique = latest.unique_contacts
    initiated_ratio = latest.initiated_count / max(1, interactions)

    # Deficit from healthy norm
    interaction_deficit = max(0.0, _HEALTHY_SOCIAL_INTERACTIONS - interactions) / _HEALTHY_SOCIAL_INTERACTIONS
    diversity_deficit = max(0.0, 1.0 - unique / max(1.0, _HEALTHY_SOCIAL_INTERACTIONS * 0.5))

    score = float(np.clip(
        0.50 * interaction_deficit
        + 0.30 * diversity_deficit
        + 0.20 * max(0.0, 0.3 - initiated_ratio),  # low initiative = concern
        0.0, 1.0,
    ))

    # Trend
    trend = "insufficient_data"
    if len(readings) >= 4:
        mid = len(readings) // 2
        early = [r.interactions_count for r in readings[:mid]]
        late = [r.interactions_count for r in readings[mid:]]
        diff = float(np.mean(late)) - float(np.mean(early))
        if diff < -2.0:
            trend = "declining"
        elif diff > 2.0:
            trend = "improving"
        else:
            trend = "stable"

    flags = []
    if interactions <= _LOW_SOCIAL_THRESHOLD:
        flags.append("critically_low_social_engagement")
    elif interactions <= _HEALTHY_SOCIAL_INTERACTIONS * 0.6:
        flags.append("reduced_social_engagement")
    if initiated_ratio < 0.2 and interactions > 0:
        flags.append("low_self_initiated_interactions")
    if trend == "declining":
        flags.append("declining_social_trend")

    return {
        "score": round(score, 4),
        "interactions_per_week": interactions,
        "unique_contacts": unique,
        "initiated_ratio": round(initiated_ratio, 3),
        "trend": trend,
        "flags": flags,
        "n_readings": len(readings),
    }


def compute_mci_risk_score(
    processing_speed: Dict[str, Any],
    flattening: Dict[str, Any],
    coupling: Dict[str, Any],
    emotional_memory: Dict[str, Any],
    social_engagement: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute composite MCI risk score from all sub-scores.

    Weighted combination of five emotional-cognitive domains.
    Higher score = greater concern.

    Returns:
        Dict with mci_risk_score (0-1), risk_category, sub_scores,
        contributing_factors, and disclaimer.
    """
    ps_score = processing_speed.get("score", 0.0)
    fl_score = flattening.get("score", 0.0)
    co_score = coupling.get("score", 0.0)
    em_score = emotional_memory.get("score", 0.0)
    se_score = social_engagement.get("score", 0.0)

    composite = (
        _W_PROCESSING_SPEED * ps_score
        + _W_FLATTENING * fl_score
        + _W_COUPLING * co_score
        + _W_MEMORY * em_score
        + _W_SOCIAL * se_score
    )
    composite = float(np.clip(composite, 0.0, 1.0))

    if composite < _LOW_RISK:
        category = "low_risk"
    elif composite < _MILD_RISK:
        category = "mild_concern"
    elif composite < _MODERATE_RISK:
        category = "moderate_concern"
    else:
        category = "high_risk"

    # Identify top contributing factors
    sub_scores = {
        "processing_speed": round(ps_score, 4),
        "emotional_flattening": round(fl_score, 4),
        "cognitive_emotional_coupling": round(co_score, 4),
        "emotional_memory": round(em_score, 4),
        "social_engagement": round(se_score, 4),
    }
    contributing = sorted(sub_scores.items(), key=lambda x: x[1], reverse=True)
    top_factors = [k for k, v in contributing if v > 0.3]

    # Aggregate all flags
    all_flags = []
    for detail in [processing_speed, flattening, coupling, emotional_memory, social_engagement]:
        all_flags.extend(detail.get("flags", []))

    return {
        "mci_risk_score": round(composite, 4),
        "risk_category": category,
        "sub_scores": sub_scores,
        "contributing_factors": top_factors,
        "flags": all_flags,
        "disclaimer": _CLINICAL_DISCLAIMER,
    }


def compute_monitoring_profile(
    processing_speed_readings: Optional[List[ProcessingSpeedReading]] = None,
    emotion_range_readings: Optional[List[EmotionRangeReading]] = None,
    cognitive_readings: Optional[List[CognitiveReading]] = None,
    emotional_memory_readings: Optional[List[EmotionalMemoryReading]] = None,
    social_engagement_readings: Optional[List[SocialEngagementReading]] = None,
) -> MonitoringProfile:
    """Compute full monitoring profile from all available data.

    Any input can be None or empty -- the profile gracefully handles
    missing data by assigning a 0.0 score to that domain.
    """
    ps_result = compute_processing_speed(processing_speed_readings or [])
    fl_result = detect_emotional_flattening(emotion_range_readings or [])
    co_result = compute_cognitive_emotional_coupling(
        cognitive_readings or [], emotion_range_readings or [],
    )
    em_result = assess_emotional_memory(emotional_memory_readings or [])
    se_result = _compute_social_engagement(social_engagement_readings or [])

    risk_result = compute_mci_risk_score(
        ps_result, fl_result, co_result, em_result, se_result,
    )

    return MonitoringProfile(
        processing_speed_score=ps_result["score"],
        flattening_score=fl_result["score"],
        coupling_score=co_result["score"],
        emotional_memory_score=em_result["score"],
        social_engagement_score=se_result["score"],
        mci_risk_score=risk_result["mci_risk_score"],
        risk_category=risk_result["risk_category"],
        processing_speed_detail=ps_result,
        flattening_detail=fl_result,
        coupling_detail=co_result,
        memory_detail=em_result,
        social_detail=se_result,
        flags=risk_result["flags"],
    )


def profile_to_dict(profile: MonitoringProfile) -> Dict[str, Any]:
    """Serialize a MonitoringProfile to a plain dict."""
    return {
        "processing_speed_score": profile.processing_speed_score,
        "flattening_score": profile.flattening_score,
        "coupling_score": profile.coupling_score,
        "emotional_memory_score": profile.emotional_memory_score,
        "social_engagement_score": profile.social_engagement_score,
        "mci_risk_score": profile.mci_risk_score,
        "risk_category": profile.risk_category,
        "processing_speed_detail": profile.processing_speed_detail,
        "flattening_detail": profile.flattening_detail,
        "coupling_detail": profile.coupling_detail,
        "memory_detail": profile.memory_detail,
        "social_detail": profile.social_detail,
        "flags": profile.flags,
        "disclaimer": profile.disclaimer,
    }
