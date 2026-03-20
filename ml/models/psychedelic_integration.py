"""EEG-guided psychedelic integration companion.

Monitors therapy sessions through six phases (preparation, onset, peak,
plateau, comedown, integration) using EEG spectral features to detect
phase transitions, track emotional processing, assess readiness, compute
integration scores, and flag safety concerns.

Phase detection from EEG:
- Preparation: baseline alpha, normal beta, low theta — calm wakefulness
- Onset:       alpha decrease, theta increase, rising gamma — transition begins
- Peak:        theta dominance, high gamma bursts, ego dissolution markers,
               maximum spectral entropy — full altered state
- Plateau:     sustained theta, moderate alpha recovery, stable entropy
- Comedown:    alpha recovery, theta decrease, beta normalization
- Integration: return to near-baseline with elevated theta (processing marker)

Emotional processing tracking:
- Surfaces which emotions appear (valence/arousal trajectory)
- Measures processing depth from spectral entropy + theta coherence
- Identifies unresolved themes via sustained negative valence without
  arousal reduction

Safety monitoring:
- Distress detection: extreme arousal + negative valence
- Grounding guidance when distress thresholds exceeded
- Continuous safety level assessment

References:
    Carhart-Harris et al. (2016) — Neural correlates of the psychedelic state
    Tagliazucchi et al. (2014) — Increased repertoire of brain dynamical states
    Barrett et al. (2020) — Emotional breakthrough and psychedelic therapy
    Roseman et al. (2019) — Emotional face recall and psilocybin

CLINICAL DISCLAIMER: This is a research and educational tool only.
It is NOT a substitute for professional therapeutic guidance. Psychedelic
therapy should only be conducted under qualified clinical supervision.

Issue #446.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clinical disclaimer & safety resources
# ---------------------------------------------------------------------------

_CLINICAL_DISCLAIMER = (
    "Wellness disclaimer: This psychedelic integration companion is a research "
    "and educational wellness tool only, not a medical device. It is NOT a "
    "substitute for professional guidance. Psychedelic-assisted experiences "
    "should only be conducted under qualified supervision. For any mental "
    "health crisis, contact a licensed professional or call the 988 Suicide "
    "& Crisis Lifeline."
)

_SAFETY_RESOURCES = (
    "If you or someone you know is in crisis: "
    "988 Suicide & Crisis Lifeline (US): call or text 988. "
    "Crisis Text Line: text HOME to 741741. "
    "Fireside Project (psychedelic support): call or text 62-FIRESIDE (623-473-7433)."
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SessionPhase(str, Enum):
    """Psychedelic therapy session phases."""
    PREPARATION = "preparation"
    ONSET = "onset"
    PEAK = "peak"
    PLATEAU = "plateau"
    COMEDOWN = "comedown"
    INTEGRATION = "integration"
    UNKNOWN = "unknown"


class SafetyLevel(str, Enum):
    """Session safety assessment level."""
    SAFE = "safe"
    MONITOR = "monitor"
    CONCERN = "concern"
    DISTRESS = "distress"


class ReadinessLevel(str, Enum):
    """Pre-session readiness assessment."""
    READY = "ready"
    CAUTION = "caution"
    NOT_READY = "not_ready"


class EmotionCategory(str, Enum):
    """Broad emotion categories tracked during sessions."""
    JOY = "joy"
    SADNESS = "sadness"
    FEAR = "fear"
    ANGER = "anger"
    AWE = "awe"
    GRIEF = "grief"
    LOVE = "love"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

# Band power fraction thresholds for phase detection
_ONSET_ALPHA_DROP = 0.15       # alpha fraction below this = onset transition
_ONSET_THETA_RISE = 0.30       # theta fraction above this = onset confirmed
_PEAK_THETA_DOMINANCE = 0.40   # theta dominant = peak territory
_PEAK_GAMMA_BURST = 0.15       # gamma fraction above this = ego dissolution marker
_PEAK_ENTROPY_HIGH = 0.65      # spectral entropy above this = peak state
_PLATEAU_THETA_SUSTAINED = 0.30  # theta sustained but not dominant
_COMEDOWN_ALPHA_RECOVERY = 0.20  # alpha recovering above this
_INTEGRATION_THETA_ELEVATED = 0.20  # theta still elevated = processing

# Readiness thresholds
_READINESS_STRESS_CEILING = 0.5   # must be below this to be ready
_READINESS_ANXIETY_CEILING = 0.5  # must be below this
_READINESS_AROUSAL_RANGE = (0.2, 0.6)  # should be moderate, not extreme

# Safety thresholds
_DISTRESS_AROUSAL = 0.85       # extreme arousal component
_DISTRESS_VALENCE = -0.6       # strongly negative valence component
_CONCERN_AROUSAL = 0.75        # elevated arousal concern
_CONCERN_VALENCE = -0.4        # negative valence concern

# Integration scoring
_INTEGRATION_MIN_SESSIONS = 1   # minimum sessions for scoring
_POSITIVE_SHIFT_THRESHOLD = 0.15  # valence improvement threshold

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EEGReading:
    """A single EEG spectral reading for phase detection."""
    theta_fraction: float = 0.0
    alpha_fraction: float = 0.0
    beta_fraction: float = 0.0
    gamma_fraction: float = 0.0
    spectral_entropy: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theta_fraction": round(self.theta_fraction, 4),
            "alpha_fraction": round(self.alpha_fraction, 4),
            "beta_fraction": round(self.beta_fraction, 4),
            "gamma_fraction": round(self.gamma_fraction, 4),
            "spectral_entropy": round(self.spectral_entropy, 4),
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class EmotionalEvent:
    """An emotional event surfaced during a session."""
    category: EmotionCategory
    intensity: float = 0.0  # 0-1
    valence: float = 0.0
    arousal: float = 0.0
    processing_depth: float = 0.0  # 0-1: how deeply processed
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "intensity": round(self.intensity, 4),
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "processing_depth": round(self.processing_depth, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class SessionProfile:
    """Complete psychedelic session profile."""
    phase: SessionPhase
    phase_confidence: float
    phase_indicators: List[str]
    safety: SafetyLevel
    safety_notes: List[str]
    emotional_events: List[EmotionalEvent]
    integration_score: Optional[float]
    processing_depth: float
    unresolved_themes: List[str]
    grounding_needed: bool
    clinical_disclaimer: str = _CLINICAL_DISCLAIMER
    safety_resources: str = _SAFETY_RESOURCES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "phase_confidence": round(self.phase_confidence, 4),
            "phase_indicators": self.phase_indicators,
            "safety": self.safety.value,
            "safety_notes": self.safety_notes,
            "emotional_events": [e.to_dict() for e in self.emotional_events],
            "integration_score": (
                round(self.integration_score, 4)
                if self.integration_score is not None
                else None
            ),
            "processing_depth": round(self.processing_depth, 4),
            "unresolved_themes": self.unresolved_themes,
            "grounding_needed": self.grounding_needed,
            "clinical_disclaimer": self.clinical_disclaimer,
            "safety_resources": self.safety_resources,
        }


# ---------------------------------------------------------------------------
# Core engine: phase detection
# ---------------------------------------------------------------------------


def detect_session_phase(reading: EEGReading) -> Dict[str, Any]:
    """Detect the current session phase from EEG spectral features.

    Uses a priority-scored approach: each phase gets a confidence score
    based on how well the reading matches that phase's EEG profile.

    Args:
        reading: Current EEG spectral state.

    Returns:
        Dict with detected phase, confidence, indicators, and disclaimer.
    """
    scores: Dict[SessionPhase, float] = {}
    indicators: Dict[SessionPhase, List[str]] = {
        p: [] for p in SessionPhase if p != SessionPhase.UNKNOWN
    }

    # --- Preparation: normal baseline ---
    prep_score = 0.0
    if reading.alpha_fraction >= 0.20:
        prep_score += 0.30
        indicators[SessionPhase.PREPARATION].append("normal_alpha")
    if reading.beta_fraction >= 0.20:
        prep_score += 0.25
        indicators[SessionPhase.PREPARATION].append("normal_beta")
    if reading.theta_fraction < _INTEGRATION_THETA_ELEVATED:
        # Preparation has truly low theta — below the elevated range
        prep_score += 0.25
        indicators[SessionPhase.PREPARATION].append("low_theta")
    if reading.spectral_entropy < 0.50:
        prep_score += 0.20
        indicators[SessionPhase.PREPARATION].append("low_entropy")
    scores[SessionPhase.PREPARATION] = prep_score

    # --- Onset: alpha drops, theta rises ---
    onset_score = 0.0
    if reading.alpha_fraction < _ONSET_ALPHA_DROP:
        onset_score += 0.35
        indicators[SessionPhase.ONSET].append("alpha_decrease")
    if reading.theta_fraction >= _ONSET_THETA_RISE:
        onset_score += 0.35
        indicators[SessionPhase.ONSET].append("theta_increase")
    if reading.gamma_fraction >= 0.08:
        onset_score += 0.15
        indicators[SessionPhase.ONSET].append("gamma_rising")
    if 0.40 <= reading.spectral_entropy <= 0.65:
        onset_score += 0.15
        indicators[SessionPhase.ONSET].append("entropy_increasing")
    scores[SessionPhase.ONSET] = onset_score

    # --- Peak: theta dominance + high gamma + high entropy ---
    peak_score = 0.0
    if reading.theta_fraction >= _PEAK_THETA_DOMINANCE:
        peak_score += 0.30
        indicators[SessionPhase.PEAK].append("theta_dominance")
    if reading.gamma_fraction >= _PEAK_GAMMA_BURST:
        peak_score += 0.25
        indicators[SessionPhase.PEAK].append("ego_dissolution_markers")
    if reading.spectral_entropy >= _PEAK_ENTROPY_HIGH:
        peak_score += 0.25
        indicators[SessionPhase.PEAK].append("high_entropy")
    if reading.alpha_fraction < 0.10:
        peak_score += 0.20
        indicators[SessionPhase.PEAK].append("alpha_suppressed")
    scores[SessionPhase.PEAK] = peak_score

    # --- Plateau: sustained theta, moderate alpha recovery ---
    plateau_score = 0.0
    if _PLATEAU_THETA_SUSTAINED <= reading.theta_fraction < _PEAK_THETA_DOMINANCE:
        plateau_score += 0.35
        indicators[SessionPhase.PLATEAU].append("sustained_theta")
    if 0.10 <= reading.alpha_fraction < 0.20:
        plateau_score += 0.25
        indicators[SessionPhase.PLATEAU].append("partial_alpha_recovery")
    if 0.45 <= reading.spectral_entropy < _PEAK_ENTROPY_HIGH:
        plateau_score += 0.20
        indicators[SessionPhase.PLATEAU].append("moderate_entropy")
    if reading.gamma_fraction < _PEAK_GAMMA_BURST:
        plateau_score += 0.20
        indicators[SessionPhase.PLATEAU].append("gamma_reduced")
    scores[SessionPhase.PLATEAU] = plateau_score

    # --- Comedown: alpha recovery, theta decreasing ---
    comedown_score = 0.0
    if reading.alpha_fraction >= _COMEDOWN_ALPHA_RECOVERY:
        comedown_score += 0.35
        indicators[SessionPhase.COMEDOWN].append("alpha_recovery")
    if reading.theta_fraction < _PLATEAU_THETA_SUSTAINED:
        comedown_score += 0.30
        indicators[SessionPhase.COMEDOWN].append("theta_decreasing")
    if reading.beta_fraction >= 0.15:
        comedown_score += 0.20
        indicators[SessionPhase.COMEDOWN].append("beta_normalizing")
    if reading.spectral_entropy < 0.45:
        comedown_score += 0.15
        indicators[SessionPhase.COMEDOWN].append("entropy_normalizing")
    scores[SessionPhase.COMEDOWN] = comedown_score

    # --- Integration: near-baseline but theta still elevated ---
    integration_score = 0.0
    if reading.alpha_fraction >= 0.20:
        integration_score += 0.30
        indicators[SessionPhase.INTEGRATION].append("alpha_restored")
    if _INTEGRATION_THETA_ELEVATED <= reading.theta_fraction < _ONSET_THETA_RISE:
        integration_score += 0.30
        indicators[SessionPhase.INTEGRATION].append("elevated_theta")
    if reading.beta_fraction >= 0.20:
        integration_score += 0.20
        indicators[SessionPhase.INTEGRATION].append("beta_restored")
    if reading.spectral_entropy < 0.50:
        integration_score += 0.20
        indicators[SessionPhase.INTEGRATION].append("low_entropy")
    scores[SessionPhase.INTEGRATION] = integration_score

    # Find the best phase
    best_phase = max(scores, key=lambda p: scores[p])
    best_score = scores[best_phase]

    if best_score < 0.25:
        best_phase = SessionPhase.UNKNOWN
        best_score = 0.0

    return {
        "phase": best_phase.value,
        "confidence": round(best_score, 4),
        "indicators": indicators.get(best_phase, []),
        "all_scores": {p.value: round(v, 4) for p, v in scores.items()},
        "reading": reading.to_dict(),
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Readiness assessment
# ---------------------------------------------------------------------------


def assess_readiness(
    valence: float,
    arousal: float,
    stress_index: float = 0.0,
    anxiety_index: float = 0.0,
    sleep_quality: float = 0.5,
    intention_clarity: float = 0.5,
) -> Dict[str, Any]:
    """Assess pre-session emotional readiness.

    Evaluates whether the user's current emotional and physiological
    state is appropriate for a psychedelic therapy session.

    Args:
        valence: Current emotional valence (-1 to 1).
        arousal: Current arousal level (0 to 1).
        stress_index: Current stress level (0 to 1).
        anxiety_index: Current anxiety level (0 to 1).
        sleep_quality: Recent sleep quality (0 to 1).
        intention_clarity: Clarity of therapeutic intention (0 to 1).

    Returns:
        Dict with readiness level, score, concerns, and recommendations.
    """
    score = 0.0
    max_score = 0.0
    concerns: List[str] = []
    recommendations: List[str] = []

    # Stress check (25% weight)
    max_score += 0.25
    if stress_index <= _READINESS_STRESS_CEILING:
        score += 0.25
    else:
        concerns.append("elevated_stress")
        recommendations.append(
            "Your stress level is elevated. Consider a calming practice "
            "(breathing exercises, gentle music) before the session."
        )

    # Anxiety check (25% weight)
    max_score += 0.25
    if anxiety_index <= _READINESS_ANXIETY_CEILING:
        score += 0.25
    else:
        concerns.append("elevated_anxiety")
        recommendations.append(
            "Anxiety is above the recommended threshold. Grounding "
            "exercises may help: 5-4-3-2-1 sensory technique."
        )

    # Arousal in healthy range (15% weight)
    max_score += 0.15
    if _READINESS_AROUSAL_RANGE[0] <= arousal <= _READINESS_AROUSAL_RANGE[1]:
        score += 0.15
    else:
        concerns.append("arousal_out_of_range")
        if arousal < _READINESS_AROUSAL_RANGE[0]:
            recommendations.append(
                "Your arousal is very low. Light movement or fresh air "
                "may help you reach a more alert baseline."
            )
        else:
            recommendations.append(
                "Your arousal is elevated. Try slow breathing to bring "
                "your activation to a calmer level."
            )

    # Sleep quality (15% weight)
    max_score += 0.15
    if sleep_quality >= 0.4:
        score += 0.15
    else:
        concerns.append("poor_sleep")
        recommendations.append(
            "Poor recent sleep can amplify difficult experiences. "
            "Consider postponing if you are severely sleep-deprived."
        )

    # Intention clarity (10% weight)
    max_score += 0.10
    if intention_clarity >= 0.4:
        score += 0.10
    else:
        concerns.append("unclear_intention")
        recommendations.append(
            "Having a clear personal intention improves outcomes. "
            "Spend a few minutes journaling about what you hope to explore."
        )

    # Valence baseline (10% weight)
    max_score += 0.10
    if valence >= -0.3:
        score += 0.10
    else:
        concerns.append("negative_baseline")
        recommendations.append(
            "Your emotional baseline is quite negative. While difficult "
            "emotions can be worked with, extreme negativity may indicate "
            "this is not the right time. Consult your therapist."
        )

    readiness_score = score / max_score if max_score > 0 else 0.0

    if readiness_score >= 0.70 and len(concerns) <= 1:
        level = ReadinessLevel.READY
    elif readiness_score >= 0.45:
        level = ReadinessLevel.CAUTION
    else:
        level = ReadinessLevel.NOT_READY

    return {
        "readiness": level.value,
        "score": round(readiness_score, 4),
        "concerns": concerns,
        "recommendations": recommendations,
        "set_assessment": {
            "stress": round(stress_index, 4),
            "anxiety": round(anxiety_index, 4),
            "arousal": round(arousal, 4),
            "valence": round(valence, 4),
            "sleep_quality": round(sleep_quality, 4),
            "intention_clarity": round(intention_clarity, 4),
        },
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Emotional processing tracker
# ---------------------------------------------------------------------------


def track_emotional_processing(
    readings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Track emotional processing across a session.

    Analyzes a sequence of emotional readings to determine which emotions
    surfaced, how deeply they were processed, and what themes remain
    unresolved.

    Args:
        readings: List of dicts with 'valence', 'arousal', 'theta_fraction',
            'spectral_entropy', and optional 'timestamp'.

    Returns:
        Dict with emotional events, processing depth, unresolved themes,
        and session summary.
    """
    if not readings:
        return {
            "events": [],
            "processing_depth": 0.0,
            "unresolved_themes": [],
            "emotions_surfaced": [],
            "total_events": 0,
            "session_summary": "No data to analyze.",
            "clinical_disclaimer": _CLINICAL_DISCLAIMER,
        }

    events: List[EmotionalEvent] = []
    total_depth = 0.0
    unresolved: List[str] = []

    for r in readings:
        valence = r.get("valence", 0.0)
        arousal = r.get("arousal", 0.0)
        theta = r.get("theta_fraction", 0.0)
        entropy = r.get("spectral_entropy", 0.0)
        ts = r.get("timestamp", time.time())

        # Classify emotion category from valence/arousal
        category = _classify_emotion(valence, arousal)
        intensity = min(1.0, abs(valence) + arousal * 0.5)

        # Processing depth from theta + entropy (higher = deeper processing)
        depth = min(1.0, theta * 1.2 + entropy * 0.5)
        total_depth += depth

        event = EmotionalEvent(
            category=category,
            intensity=round(intensity, 4),
            valence=valence,
            arousal=arousal,
            processing_depth=round(depth, 4),
            timestamp=ts,
        )
        events.append(event)

    avg_depth = total_depth / len(readings) if readings else 0.0

    # Detect unresolved themes: sustained negative valence without depth
    neg_events = [e for e in events if e.valence < -0.3]
    if neg_events:
        shallow_neg = [e for e in neg_events if e.processing_depth < 0.3]
        if len(shallow_neg) > len(neg_events) * 0.5:
            unresolved.append("unprocessed_negative_emotions")

    # Check for persistent fear without resolution
    fear_events = [e for e in events if e.category == EmotionCategory.FEAR]
    if fear_events and len(fear_events) > len(events) * 0.3:
        # If fear persists through >30% of the session
        late_fear = [e for e in fear_events[-max(1, len(fear_events) // 2):]
                     if e.intensity > 0.5]
        if late_fear:
            unresolved.append("persistent_fear")

    # Check for grief not reaching depth
    grief_events = [e for e in events if e.category == EmotionCategory.GRIEF]
    if grief_events:
        avg_grief_depth = sum(e.processing_depth for e in grief_events) / len(grief_events)
        if avg_grief_depth < 0.3:
            unresolved.append("surface_level_grief")

    # Gather unique emotions surfaced
    emotions_surfaced = list(set(e.category.value for e in events))

    return {
        "events": [e.to_dict() for e in events],
        "processing_depth": round(avg_depth, 4),
        "unresolved_themes": unresolved,
        "emotions_surfaced": emotions_surfaced,
        "total_events": len(events),
        "session_summary": _summarize_processing(events, avg_depth, unresolved),
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }


def _classify_emotion(valence: float, arousal: float) -> EmotionCategory:
    """Classify broad emotion category from valence and arousal."""
    if abs(valence) < 0.15 and arousal < 0.3:
        return EmotionCategory.NEUTRAL
    if valence > 0.3 and arousal > 0.5:
        return EmotionCategory.JOY
    if valence > 0.2 and arousal < 0.4:
        return EmotionCategory.LOVE
    if valence > 0.3 and arousal > 0.3:
        return EmotionCategory.AWE
    if valence < -0.3 and arousal > 0.6:
        return EmotionCategory.ANGER
    if valence < -0.3 and arousal > 0.4:
        return EmotionCategory.FEAR
    if valence < -0.2 and arousal < 0.3:
        return EmotionCategory.GRIEF
    if valence < -0.1:
        return EmotionCategory.SADNESS
    return EmotionCategory.NEUTRAL


def _summarize_processing(
    events: List[EmotionalEvent],
    avg_depth: float,
    unresolved: List[str],
) -> str:
    """Generate a human-readable processing summary."""
    n = len(events)
    if n == 0:
        return "No emotional events recorded."

    unique = set(e.category.value for e in events)
    depth_label = (
        "deep" if avg_depth >= 0.6
        else "moderate" if avg_depth >= 0.3
        else "surface-level"
    )

    summary = (
        f"Session included {n} emotional reading(s) spanning "
        f"{len(unique)} emotion category/categories. "
        f"Average processing depth: {depth_label} ({avg_depth:.2f})."
    )

    if unresolved:
        summary += (
            f" Unresolved themes detected: {', '.join(unresolved)}. "
            "Consider revisiting these in integration work."
        )

    return summary


# ---------------------------------------------------------------------------
# Integration scoring
# ---------------------------------------------------------------------------


def compute_integration_score(
    pre_session_valence: float,
    post_session_valence: float,
    processing_depth: float = 0.0,
    unresolved_count: int = 0,
    days_since_session: int = 0,
    emotional_stability: float = 0.5,
) -> Dict[str, Any]:
    """Compute integration score: did the session produce lasting change?

    Measures how well insights from the session have been integrated into
    the user's emotional baseline.

    Args:
        pre_session_valence: Emotional valence before session (-1 to 1).
        post_session_valence: Current emotional valence (-1 to 1).
        processing_depth: Average processing depth from the session (0 to 1).
        unresolved_count: Number of unresolved themes.
        days_since_session: Days elapsed since the session.
        emotional_stability: Stability of emotional baseline (0 to 1).

    Returns:
        Dict with integration score, components, and recommendations.
    """
    # Valence shift component (40% weight)
    valence_shift = post_session_valence - pre_session_valence
    valence_component = float(np.clip(valence_shift / 0.5, -1.0, 1.0))
    # Normalize to 0-1 range
    valence_score = (valence_component + 1.0) / 2.0

    # Processing depth component (25% weight)
    depth_score = float(np.clip(processing_depth, 0.0, 1.0))

    # Resolution component (20% weight) — fewer unresolved = better
    resolution_score = max(0.0, 1.0 - unresolved_count * 0.25)

    # Stability component (15% weight)
    stability_score = float(np.clip(emotional_stability, 0.0, 1.0))

    # Weighted integration score
    raw_score = (
        0.40 * valence_score
        + 0.25 * depth_score
        + 0.20 * resolution_score
        + 0.15 * stability_score
    )

    # Time decay — integration fading is a concern signal
    if days_since_session > 14:
        decay = max(0.8, 1.0 - (days_since_session - 14) * 0.005)
        raw_score *= decay

    score = float(np.clip(raw_score, 0.0, 1.0))

    # Generate assessment
    if score >= 0.70:
        assessment = "strong"
        message = (
            "Strong integration: the session appears to have produced "
            "lasting positive shifts in your emotional baseline."
        )
    elif score >= 0.45:
        assessment = "moderate"
        message = (
            "Moderate integration: some positive shifts are present but "
            "there may be unresolved material worth exploring further."
        )
    else:
        assessment = "needs_attention"
        message = (
            "Integration needs attention: the session's insights may not "
            "have fully settled. Consider integration practices such as "
            "journaling, therapy, or mindfulness to deepen the work."
        )

    recommendations: List[str] = []
    if unresolved_count > 0:
        recommendations.append(
            f"{unresolved_count} unresolved theme(s) from the session. "
            "Consider revisiting with a therapist."
        )
    if valence_shift < _POSITIVE_SHIFT_THRESHOLD:
        recommendations.append(
            "No significant positive shift in emotional baseline. "
            "Integration practices (journaling, art, movement) may help."
        )
    if emotional_stability < 0.3:
        recommendations.append(
            "Emotional stability is low. Grounding and routine are important "
            "during the integration period."
        )

    return {
        "integration_score": round(score, 4),
        "assessment": assessment,
        "message": message,
        "components": {
            "valence_shift": round(valence_score, 4),
            "processing_depth": round(depth_score, 4),
            "resolution": round(resolution_score, 4),
            "stability": round(stability_score, 4),
        },
        "valence_change": round(valence_shift, 4),
        "days_since_session": days_since_session,
        "recommendations": recommendations,
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Safety monitoring
# ---------------------------------------------------------------------------


def check_session_safety(
    reading: EEGReading,
    recent_readings: Optional[List[EEGReading]] = None,
) -> Dict[str, Any]:
    """Check session safety from current and recent EEG/emotional data.

    Monitors for distress (extreme arousal + negative valence) and
    provides grounding guidance when needed.

    Args:
        reading: Current EEG/emotional reading.
        recent_readings: Optional recent history for pattern detection.

    Returns:
        Dict with safety level, grounding guidance, and resources.
    """
    safety_level = SafetyLevel.SAFE
    notes: List[str] = []
    grounding_needed = False
    recent = recent_readings or []

    # Distress detection: extreme arousal + negative valence
    if reading.arousal >= _DISTRESS_AROUSAL and reading.valence <= _DISTRESS_VALENCE:
        safety_level = SafetyLevel.DISTRESS
        grounding_needed = True
        notes.append("Acute distress: extreme arousal with negative valence")

    elif reading.arousal >= _CONCERN_AROUSAL and reading.valence <= _CONCERN_VALENCE:
        safety_level = SafetyLevel.CONCERN
        grounding_needed = True
        notes.append("Elevated concern: high arousal with negative affect")

    elif reading.arousal >= _CONCERN_AROUSAL:
        if safety_level == SafetyLevel.SAFE:
            safety_level = SafetyLevel.MONITOR
        notes.append("High arousal detected")

    elif reading.valence <= _DISTRESS_VALENCE:
        if safety_level == SafetyLevel.SAFE:
            safety_level = SafetyLevel.MONITOR
        notes.append("Significantly negative valence")

    # Check sustained distress in recent history
    if len(recent) >= 3:
        recent_distress = sum(
            1 for r in recent[-3:]
            if r.arousal >= _CONCERN_AROUSAL and r.valence <= _CONCERN_VALENCE
        )
        if recent_distress >= 2:
            if safety_level in (SafetyLevel.SAFE, SafetyLevel.MONITOR):
                safety_level = SafetyLevel.CONCERN
            grounding_needed = True
            notes.append("Sustained elevated distress across multiple readings")

    # Grounding guidance
    guidance_map = {
        SafetyLevel.SAFE: (
            "Session is proceeding safely. Continue at your own pace."
        ),
        SafetyLevel.MONITOR: (
            "Monitoring elevated signals. Remember: you are safe, this "
            "experience is temporary, and you can return to your body "
            "at any time."
        ),
        SafetyLevel.CONCERN: (
            "Your distress level is elevated. Try grounding: feel your "
            "feet on the floor, squeeze something solid, breathe slowly. "
            "Count 5 things you can see. You are here. You are safe."
        ),
        SafetyLevel.DISTRESS: (
            "GROUNDING NEEDED. Focus on your breath. Breathe in for 4 "
            "counts, hold for 4, out for 6. Feel the ground under you. "
            "Squeeze your hands. Say your name out loud. You are safe. "
            "This will pass. Your guide is here with you."
        ),
    }

    return {
        "safety_level": safety_level.value,
        "grounding_needed": grounding_needed,
        "notes": notes,
        "guidance": guidance_map[safety_level],
        "safety_resources": _SAFETY_RESOURCES,
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Composite session profile
# ---------------------------------------------------------------------------


def compute_session_profile(
    reading: EEGReading,
    emotional_readings: Optional[List[Dict[str, Any]]] = None,
    recent_eeg: Optional[List[EEGReading]] = None,
    pre_session_valence: Optional[float] = None,
) -> SessionProfile:
    """Compute a comprehensive session profile.

    Combines phase detection, safety monitoring, emotional processing,
    and integration scoring into a single coherent profile.

    Args:
        reading: Current EEG spectral reading.
        emotional_readings: Optional emotional reading history for processing.
        recent_eeg: Optional recent EEG readings for safety context.
        pre_session_valence: Optional pre-session valence for integration.

    Returns:
        SessionProfile with all assessments.
    """
    # Phase detection
    phase_result = detect_session_phase(reading)
    phase = SessionPhase(phase_result["phase"])
    phase_confidence = phase_result["confidence"]
    phase_indicators = phase_result["indicators"]

    # Safety check
    safety_result = check_session_safety(reading, recent_eeg)
    safety = SafetyLevel(safety_result["safety_level"])
    safety_notes = safety_result["notes"]
    grounding_needed = safety_result["grounding_needed"]

    # Emotional processing
    emotional_data = emotional_readings or []
    processing_result = track_emotional_processing(emotional_data)
    events = [
        EmotionalEvent(
            category=EmotionCategory(e["category"]),
            intensity=e["intensity"],
            valence=e["valence"],
            arousal=e["arousal"],
            processing_depth=e["processing_depth"],
            timestamp=e.get("timestamp", time.time()),
        )
        for e in processing_result["events"]
    ]
    processing_depth = processing_result["processing_depth"]
    unresolved_themes = processing_result["unresolved_themes"]

    # Integration score (only if post-session and pre_session_valence given)
    integration_score_val = None
    if pre_session_valence is not None and phase == SessionPhase.INTEGRATION:
        integration_result = compute_integration_score(
            pre_session_valence=pre_session_valence,
            post_session_valence=reading.valence,
            processing_depth=processing_depth,
            unresolved_count=len(unresolved_themes),
        )
        integration_score_val = integration_result["integration_score"]

    return SessionProfile(
        phase=phase,
        phase_confidence=phase_confidence,
        phase_indicators=phase_indicators,
        safety=safety,
        safety_notes=safety_notes,
        emotional_events=events,
        integration_score=integration_score_val,
        processing_depth=processing_depth,
        unresolved_themes=unresolved_themes,
        grounding_needed=grounding_needed,
    )


def profile_to_dict(profile: SessionProfile) -> Dict[str, Any]:
    """Convert a SessionProfile to a serializable dict.

    Args:
        profile: SessionProfile instance.

    Returns:
        Dict representation of the profile.
    """
    return profile.to_dict()
