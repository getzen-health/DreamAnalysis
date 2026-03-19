"""EEG-guided grief and loss processing companion.

Provides stage-aware support for grief processing using the Kubler-Ross
five-stage model combined with Worden's four tasks of mourning. Detects
the user's current grief stage from emotional data, tracks trajectory
over time, selects stage-appropriate support interventions, and monitors
for anniversary effects and safety concerns.

Grief stage detection:
- Denial:      flat affect + low arousal + emotional numbness
- Anger:       high arousal + negative valence + elevated anger index
- Bargaining:  moderate arousal + oscillating valence + rumination markers
- Depression:  low arousal + sustained negative valence + withdrawal
- Acceptance:  moderate arousal + neutral-positive valence + stability

Worden's tasks of mourning (parallel framework):
1. Accept the reality of the loss
2. Process the pain of grief
3. Adjust to a world without the deceased
4. Find an enduring connection while embarking on a new life

Safety rails:
- Suicidal ideation markers: prolonged flat affect + isolation + hopelessness
- Always provide crisis line information
- Never minimize grief or rush the process
- Escalate when safety thresholds are exceeded

References:
    Kubler-Ross (1969) — On Death and Dying
    Worden (2009) — Grief Counseling and Grief Therapy, 4th ed.
    Shear et al. (2011) — Complicated grief and related bereavement issues
    Bonanno (2004) — Loss, trauma, and human resilience

CLINICAL DISCLAIMER: This is a research and educational tool only.
It is NOT a substitute for professional grief counseling or clinical
care. For any genuine mental health crisis, contact a licensed
professional or emergency services immediately.

Issue #424.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clinical disclaimer & crisis resources
# ---------------------------------------------------------------------------

_CLINICAL_DISCLAIMER = (
    "Clinical disclaimer: This grief companion is a research and educational "
    "tool only. It is NOT a substitute for professional grief counseling or "
    "therapy. For any genuine mental health crisis, contact a licensed mental "
    "health professional or call the 988 Suicide & Crisis Lifeline (call or "
    "text 988 in the US), or your local equivalent."
)

_CRISIS_RESOURCES = (
    "If you or someone you know is in crisis: "
    "988 Suicide & Crisis Lifeline (US): call or text 988. "
    "Crisis Text Line: text HOME to 741741. "
    "International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/"
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GriefStage(str, Enum):
    """Kubler-Ross grief stages."""
    DENIAL = "denial"
    ANGER = "anger"
    BARGAINING = "bargaining"
    DEPRESSION = "depression"
    ACCEPTANCE = "acceptance"
    UNKNOWN = "unknown"


class WordenTask(str, Enum):
    """Worden's four tasks of mourning."""
    ACCEPT_REALITY = "accept_reality"
    PROCESS_PAIN = "process_pain"
    ADJUST_WORLD = "adjust_world"
    FIND_CONNECTION = "find_connection"


class TrajectoryTrend(str, Enum):
    """Overall trajectory direction."""
    PROGRESSING = "progressing"
    STUCK = "stuck"
    REGRESSING = "regressing"
    OSCILLATING = "oscillating"
    INSUFFICIENT_DATA = "insufficient_data"


class SafetyLevel(str, Enum):
    """Safety assessment level."""
    SAFE = "safe"
    MONITOR = "monitor"
    CONCERN = "concern"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

# Denial: emotional numbness / flat affect
_DENIAL_AROUSAL_CEILING = 0.30
_DENIAL_VALENCE_RANGE = 0.20  # flat = valence close to zero

# Anger: high arousal + negative + anger
_ANGER_AROUSAL_THRESHOLD = 0.65
_ANGER_VALENCE_CEILING = -0.15
_ANGER_INDEX_THRESHOLD = 0.50

# Bargaining: moderate arousal + valence oscillation
_BARGAINING_AROUSAL_RANGE = (0.30, 0.65)
_BARGAINING_VALENCE_INSTABILITY = 0.25  # high variance in recent readings

# Depression: low arousal + sustained negative valence
_DEPRESSION_AROUSAL_CEILING = 0.35
_DEPRESSION_VALENCE_CEILING = -0.20

# Acceptance: moderate arousal + neutral-to-positive valence + stability
_ACCEPTANCE_AROUSAL_RANGE = (0.25, 0.60)
_ACCEPTANCE_VALENCE_FLOOR = -0.10

# Safety: prolonged flat affect + isolation markers
_SAFETY_FLAT_AFFECT_DURATION = 5  # consecutive readings
_SAFETY_DEPRESSION_SEVERITY = 0.80  # high depression score
_SAFETY_HOPELESSNESS_THRESHOLD = 0.75

# Anniversary detection: emotional dip within window of significant date
_ANNIVERSARY_WINDOW_DAYS = 7
_ANNIVERSARY_DIP_THRESHOLD = 0.25  # valence drop relative to recent baseline

# Trajectory: minimum readings for trend calculation
_MIN_READINGS_FOR_TRAJECTORY = 3
_STUCK_THRESHOLD_READINGS = 5  # same stage for this many readings = "stuck"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GriefReading:
    """A single emotional state reading in the grief context."""
    valence: float = 0.0
    arousal: float = 0.0
    stress_index: float = 0.0
    anger_index: float = 0.0
    focus_index: float = 0.5
    isolation_index: float = 0.0  # 0=socially connected, 1=isolated
    hopelessness_index: float = 0.0  # 0=hopeful, 1=hopeless
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "stress_index": round(self.stress_index, 4),
            "anger_index": round(self.anger_index, 4),
            "focus_index": round(self.focus_index, 4),
            "isolation_index": round(self.isolation_index, 4),
            "hopelessness_index": round(self.hopelessness_index, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class SupportIntervention:
    """A stage-appropriate support intervention."""
    stage: GriefStage
    worden_task: WordenTask
    name: str
    description: str
    guidance: str
    duration_minutes: int = 10
    intensity: str = "gentle"  # gentle, moderate, active

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "worden_task": self.worden_task.value,
            "name": self.name,
            "description": self.description,
            "guidance": self.guidance,
            "duration_minutes": self.duration_minutes,
            "intensity": self.intensity,
        }


@dataclass
class GriefProfile:
    """Complete grief assessment profile."""
    stage: GriefStage
    stage_confidence: float
    stage_indicators: List[str]
    worden_task: WordenTask
    trajectory: TrajectoryTrend
    safety: SafetyLevel
    safety_notes: List[str]
    intervention: Optional[SupportIntervention]
    anniversary_detected: bool
    readings_count: int
    clinical_disclaimer: str = _CLINICAL_DISCLAIMER
    crisis_resources: str = _CRISIS_RESOURCES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "stage_confidence": round(self.stage_confidence, 4),
            "stage_indicators": self.stage_indicators,
            "worden_task": self.worden_task.value,
            "trajectory": self.trajectory.value,
            "safety": self.safety.value,
            "safety_notes": self.safety_notes,
            "intervention": self.intervention.to_dict() if self.intervention else None,
            "anniversary_detected": self.anniversary_detected,
            "readings_count": self.readings_count,
            "clinical_disclaimer": self.clinical_disclaimer,
            "crisis_resources": self.crisis_resources,
        }


# ---------------------------------------------------------------------------
# Intervention library
# ---------------------------------------------------------------------------

INTERVENTION_LIBRARY: List[SupportIntervention] = [
    # Denial stage interventions
    SupportIntervention(
        stage=GriefStage.DENIAL,
        worden_task=WordenTask.ACCEPT_REALITY,
        name="Gentle Reality Acknowledgment",
        description=(
            "A soft approach to gently acknowledging the loss without "
            "forcing emotional confrontation. Respects the protective "
            "function of denial while creating small openings."
        ),
        guidance=(
            "You do not need to feel everything right now. Denial is your "
            "mind's way of protecting you from overwhelming pain. When you "
            "are ready, try saying the loss out loud, just once: 'This happened.' "
            "Then breathe. You can stop there. There is no rush."
        ),
        duration_minutes=5,
        intensity="gentle",
    ),
    SupportIntervention(
        stage=GriefStage.DENIAL,
        worden_task=WordenTask.ACCEPT_REALITY,
        name="Grounding in the Present",
        description=(
            "Use sensory grounding to reconnect with reality. Denial often "
            "manifests as dissociation from the present moment."
        ),
        guidance=(
            "Place both feet on the ground. Feel the temperature of the air. "
            "Name three things you can see right now. You are here. This moment "
            "is real. You do not need to process everything at once."
        ),
        duration_minutes=5,
        intensity="gentle",
    ),
    # Anger stage interventions
    SupportIntervention(
        stage=GriefStage.ANGER,
        worden_task=WordenTask.PROCESS_PAIN,
        name="Anger Validation and Containment",
        description=(
            "Validates anger as a natural grief response while providing "
            "safe containment. Anger in grief is often directed at the "
            "deceased, at oneself, at God, or at the universe."
        ),
        guidance=(
            "Your anger is valid. It is a natural part of grief. You are "
            "allowed to feel it. Try naming what you are angry about: "
            "'I am angry because...' Let yourself say it. Then take three "
            "slow breaths. The anger does not need to be resolved right now "
            "-- it just needs to be heard."
        ),
        duration_minutes=10,
        intensity="moderate",
    ),
    SupportIntervention(
        stage=GriefStage.ANGER,
        worden_task=WordenTask.PROCESS_PAIN,
        name="Physical Release",
        description=(
            "Channel anger energy through safe physical movement. "
            "Anger carries physiological activation that benefits from "
            "motor discharge."
        ),
        guidance=(
            "Your body is holding this anger. Give it an outlet: squeeze "
            "a pillow hard, press your palms together as hard as you can "
            "for 10 seconds, or take a brisk walk. The goal is not to get "
            "rid of the anger but to give the energy somewhere to go."
        ),
        duration_minutes=10,
        intensity="active",
    ),
    # Bargaining stage interventions
    SupportIntervention(
        stage=GriefStage.BARGAINING,
        worden_task=WordenTask.PROCESS_PAIN,
        name="Meaning-Making Reflection",
        description=(
            "Supports the natural meaning-making process without rushing "
            "past the 'what if' and 'if only' thoughts. Bargaining is the "
            "mind trying to regain control."
        ),
        guidance=(
            "The 'what if' thoughts are your mind trying to find a way "
            "this could have been different. That is normal. Instead of "
            "fighting these thoughts, try writing one down. Then ask: "
            "'What would I say to a friend having this thought?' Offer "
            "yourself that same compassion."
        ),
        duration_minutes=15,
        intensity="moderate",
    ),
    SupportIntervention(
        stage=GriefStage.BARGAINING,
        worden_task=WordenTask.ADJUST_WORLD,
        name="Letting Go of Control",
        description=(
            "Gently acknowledges the limits of control. Bargaining often "
            "involves magical thinking about what could have prevented the loss."
        ),
        guidance=(
            "Some things are beyond our control. That is one of the hardest "
            "truths in grief. Try placing your hands palm-up on your lap -- "
            "a posture of openness. Breathe in and think: 'I did what I could.' "
            "Breathe out and think: 'Some things I cannot change.' Repeat slowly."
        ),
        duration_minutes=10,
        intensity="gentle",
    ),
    # Depression stage interventions
    SupportIntervention(
        stage=GriefStage.DEPRESSION,
        worden_task=WordenTask.PROCESS_PAIN,
        name="Compassionate Presence",
        description=(
            "Sits with the sadness without trying to fix it. Depression in "
            "grief is not clinical depression -- it is the appropriate "
            "emotional response to significant loss."
        ),
        guidance=(
            "This sadness is a measure of how much this person or thing "
            "mattered to you. You do not need to feel better right now. "
            "If you can, place a hand on your heart. Breathe slowly. "
            "Say to yourself: 'This hurts because it mattered.' "
            "You are not broken. You are grieving."
        ),
        duration_minutes=10,
        intensity="gentle",
    ),
    SupportIntervention(
        stage=GriefStage.DEPRESSION,
        worden_task=WordenTask.ADJUST_WORLD,
        name="Gentle Activation",
        description=(
            "Small behavioral activation to prevent complete withdrawal. "
            "Not about 'cheering up' but about maintaining basic self-care "
            "and one tiny connection to the world."
        ),
        guidance=(
            "Grief can make everything feel impossible. You do not need to "
            "do everything. Just one thing. Can you drink a glass of water? "
            "Open a window? Send one text to someone who cares about you? "
            "That is enough. That is more than enough."
        ),
        duration_minutes=5,
        intensity="gentle",
    ),
    # Acceptance stage interventions
    SupportIntervention(
        stage=GriefStage.ACCEPTANCE,
        worden_task=WordenTask.FIND_CONNECTION,
        name="Integration and Legacy",
        description=(
            "Supports the integration of loss into a new life narrative. "
            "Acceptance does not mean the grief is over -- it means the "
            "loss has found a place to live."
        ),
        guidance=(
            "Acceptance is not about being okay with the loss. It is about "
            "finding a way to carry it. What did this person or experience "
            "teach you? How did it change who you are? These are not easy "
            "questions. Take your time. The answers may come in pieces."
        ),
        duration_minutes=15,
        intensity="moderate",
    ),
    SupportIntervention(
        stage=GriefStage.ACCEPTANCE,
        worden_task=WordenTask.FIND_CONNECTION,
        name="Continuing Bonds",
        description=(
            "Based on Klass et al.'s continuing bonds theory -- maintaining "
            "a healthy ongoing relationship with what was lost, rather than "
            "severing all connection."
        ),
        guidance=(
            "You do not have to let go completely. Many people find comfort "
            "in maintaining a connection: a ritual, a letter, a place you "
            "visit, a tradition you keep alive. What connection would feel "
            "meaningful to you? There is no wrong answer."
        ),
        duration_minutes=10,
        intensity="gentle",
    ),
]

# Index by stage for fast lookup
_INTERVENTIONS_BY_STAGE: Dict[GriefStage, List[SupportIntervention]] = {}
for _intervention in INTERVENTION_LIBRARY:
    _INTERVENTIONS_BY_STAGE.setdefault(_intervention.stage, []).append(_intervention)


# ---------------------------------------------------------------------------
# Stage-to-Worden mapping
# ---------------------------------------------------------------------------

_STAGE_TO_WORDEN: Dict[GriefStage, WordenTask] = {
    GriefStage.DENIAL: WordenTask.ACCEPT_REALITY,
    GriefStage.ANGER: WordenTask.PROCESS_PAIN,
    GriefStage.BARGAINING: WordenTask.PROCESS_PAIN,
    GriefStage.DEPRESSION: WordenTask.ADJUST_WORLD,
    GriefStage.ACCEPTANCE: WordenTask.FIND_CONNECTION,
    GriefStage.UNKNOWN: WordenTask.ACCEPT_REALITY,
}


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


def detect_grief_stage(reading: GriefReading) -> Dict[str, Any]:
    """Detect the current grief stage from an emotional state reading.

    Uses a priority-scored approach: each stage gets a confidence score
    based on how well the reading matches that stage's profile. The
    highest-scoring stage wins.

    Args:
        reading: Current emotional/physiological state.

    Returns:
        Dict with detected stage, confidence, indicators, worden_task,
        and clinical_disclaimer.
    """
    scores: Dict[GriefStage, float] = {}
    indicators: Dict[GriefStage, List[str]] = {s: [] for s in GriefStage if s != GriefStage.UNKNOWN}

    # --- Denial ---
    denial_score = 0.0
    if reading.arousal <= _DENIAL_AROUSAL_CEILING:
        denial_score += 0.35
        indicators[GriefStage.DENIAL].append("low_arousal")
    if abs(reading.valence) <= _DENIAL_VALENCE_RANGE:
        denial_score += 0.35
        indicators[GriefStage.DENIAL].append("flat_affect")
    if reading.focus_index < 0.3:
        denial_score += 0.15
        indicators[GriefStage.DENIAL].append("low_focus")
    if reading.stress_index < 0.3:
        denial_score += 0.15
        indicators[GriefStage.DENIAL].append("low_stress")
    scores[GriefStage.DENIAL] = denial_score

    # --- Anger ---
    anger_score = 0.0
    if reading.arousal >= _ANGER_AROUSAL_THRESHOLD:
        anger_score += 0.30
        indicators[GriefStage.ANGER].append("high_arousal")
    if reading.valence <= _ANGER_VALENCE_CEILING:
        anger_score += 0.25
        indicators[GriefStage.ANGER].append("negative_valence")
    if reading.anger_index >= _ANGER_INDEX_THRESHOLD:
        anger_score += 0.35
        indicators[GriefStage.ANGER].append("elevated_anger")
    if reading.stress_index >= 0.5:
        anger_score += 0.10
        indicators[GriefStage.ANGER].append("elevated_stress")
    scores[GriefStage.ANGER] = anger_score

    # --- Bargaining ---
    bargaining_score = 0.0
    arousal_in_range = _BARGAINING_AROUSAL_RANGE[0] <= reading.arousal <= _BARGAINING_AROUSAL_RANGE[1]
    if arousal_in_range:
        bargaining_score += 0.25
        indicators[GriefStage.BARGAINING].append("moderate_arousal")
    if reading.stress_index >= 0.4:
        bargaining_score += 0.25
        indicators[GriefStage.BARGAINING].append("rumination_stress")
    # Bargaining often shows mixed valence (neither clearly positive nor deeply negative)
    if -0.40 <= reading.valence <= 0.10:
        bargaining_score += 0.25
        indicators[GriefStage.BARGAINING].append("mixed_valence")
    if reading.focus_index >= 0.5:
        bargaining_score += 0.25
        indicators[GriefStage.BARGAINING].append("cognitive_engagement")
    scores[GriefStage.BARGAINING] = bargaining_score

    # --- Depression ---
    depression_score = 0.0
    if reading.arousal <= _DEPRESSION_AROUSAL_CEILING:
        depression_score += 0.30
        indicators[GriefStage.DEPRESSION].append("low_arousal")
    if reading.valence <= _DEPRESSION_VALENCE_CEILING:
        depression_score += 0.30
        indicators[GriefStage.DEPRESSION].append("negative_valence")
    if reading.isolation_index >= 0.5:
        depression_score += 0.20
        indicators[GriefStage.DEPRESSION].append("social_withdrawal")
    if reading.hopelessness_index >= 0.4:
        depression_score += 0.20
        indicators[GriefStage.DEPRESSION].append("hopelessness")
    scores[GriefStage.DEPRESSION] = depression_score

    # --- Acceptance ---
    acceptance_score = 0.0
    acceptance_arousal = _ACCEPTANCE_AROUSAL_RANGE[0] <= reading.arousal <= _ACCEPTANCE_AROUSAL_RANGE[1]
    if acceptance_arousal:
        acceptance_score += 0.30
        indicators[GriefStage.ACCEPTANCE].append("balanced_arousal")
    if reading.valence >= _ACCEPTANCE_VALENCE_FLOOR:
        acceptance_score += 0.30
        indicators[GriefStage.ACCEPTANCE].append("neutral_positive_valence")
    if reading.stress_index < 0.4:
        acceptance_score += 0.20
        indicators[GriefStage.ACCEPTANCE].append("low_stress")
    if reading.hopelessness_index < 0.3:
        acceptance_score += 0.20
        indicators[GriefStage.ACCEPTANCE].append("hopefulness")
    scores[GriefStage.ACCEPTANCE] = acceptance_score

    # Find the best stage
    best_stage = max(scores, key=lambda s: scores[s])
    best_score = scores[best_stage]

    # If no stage has a meaningful score, mark as unknown
    if best_score < 0.25:
        best_stage = GriefStage.UNKNOWN
        best_score = 0.0

    worden_task = _STAGE_TO_WORDEN[best_stage]

    return {
        "stage": best_stage.value,
        "confidence": round(best_score, 4),
        "indicators": indicators.get(best_stage, []),
        "worden_task": worden_task.value,
        "all_scores": {s.value: round(v, 4) for s, v in scores.items()},
        "reading": reading.to_dict(),
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }


def track_grief_trajectory(
    readings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Track grief trajectory over multiple readings.

    Analyzes a sequence of grief stage detections to determine whether
    the user is progressing through grief, stuck in one stage, regressing,
    or oscillating between stages.

    Args:
        readings: List of dicts, each with at least "stage" and "timestamp" keys.
            These are previous detect_grief_stage() results.

    Returns:
        Dict with trajectory trend, stage history, time in each stage,
        and recommendations.
    """
    if len(readings) < _MIN_READINGS_FOR_TRAJECTORY:
        return {
            "trend": TrajectoryTrend.INSUFFICIENT_DATA.value,
            "readings_analyzed": len(readings),
            "minimum_required": _MIN_READINGS_FOR_TRAJECTORY,
            "message": (
                "Not enough readings to determine a trajectory. "
                "Continue checking in regularly."
            ),
            "clinical_disclaimer": _CLINICAL_DISCLAIMER,
        }

    # Extract stage sequence
    stages = [r.get("stage", "unknown") for r in readings]
    stage_order = {
        "denial": 0,
        "anger": 1,
        "bargaining": 2,
        "depression": 3,
        "acceptance": 4,
        "unknown": -1,
    }

    # Count time in each stage
    stage_counts: Dict[str, int] = {}
    for s in stages:
        stage_counts[s] = stage_counts.get(s, 0) + 1

    current_stage = stages[-1]
    dominant_stage = max(stage_counts, key=lambda s: stage_counts[s])

    # Detect trend
    # Check if stuck: same stage for many consecutive readings
    recent = stages[-_STUCK_THRESHOLD_READINGS:] if len(stages) >= _STUCK_THRESHOLD_READINGS else stages
    all_same = len(set(recent)) == 1

    # Check direction: are stages generally moving forward?
    valid_stages = [s for s in stages if s in stage_order and stage_order[s] >= 0]
    if len(valid_stages) >= 2:
        stage_nums = [stage_order[s] for s in valid_stages]
        first_half = stage_nums[: len(stage_nums) // 2]
        second_half = stage_nums[len(stage_nums) // 2:]
        avg_first = sum(first_half) / len(first_half) if first_half else 0
        avg_second = sum(second_half) / len(second_half) if second_half else 0

        unique_stages = len(set(stages[-5:] if len(stages) >= 5 else stages))

        if all_same and len(recent) >= _STUCK_THRESHOLD_READINGS:
            trend = TrajectoryTrend.STUCK
        elif unique_stages >= 3:
            trend = TrajectoryTrend.OSCILLATING
        elif avg_second > avg_first + 0.3:
            trend = TrajectoryTrend.PROGRESSING
        elif avg_second < avg_first - 0.3:
            trend = TrajectoryTrend.REGRESSING
        elif all_same:
            trend = TrajectoryTrend.STUCK
        else:
            trend = TrajectoryTrend.PROGRESSING
    else:
        trend = TrajectoryTrend.INSUFFICIENT_DATA

    # Generate message based on trend
    messages = {
        TrajectoryTrend.PROGRESSING: (
            "Your grief is moving through its natural stages. This does not "
            "mean it hurts less -- it means you are processing. Be patient "
            "with yourself."
        ),
        TrajectoryTrend.STUCK: (
            f"You have been in the {current_stage} stage for a while. "
            "Being stuck does not mean you are failing. Sometimes grief "
            "needs more time in one place. But if this feels overwhelming, "
            "consider talking to a grief counselor."
        ),
        TrajectoryTrend.REGRESSING: (
            "Moving backward in grief is completely normal. Grief is not "
            "linear. A setback does not erase your progress. Be gentle "
            "with yourself."
        ),
        TrajectoryTrend.OSCILLATING: (
            "Your grief is moving between different stages. This is very "
            "common and normal. Grief rarely follows a straight line. "
            "Each stage teaches you something."
        ),
        TrajectoryTrend.INSUFFICIENT_DATA: (
            "Not enough data to determine a trend yet. Keep checking in."
        ),
    }

    return {
        "trend": trend.value,
        "current_stage": current_stage,
        "dominant_stage": dominant_stage,
        "stage_counts": stage_counts,
        "readings_analyzed": len(readings),
        "message": messages[trend],
        "recommendation": _trajectory_recommendation(trend, current_stage),
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }


def _trajectory_recommendation(trend: TrajectoryTrend, current_stage: str) -> str:
    """Generate recommendation based on trajectory and current stage."""
    if trend == TrajectoryTrend.STUCK:
        if current_stage == "depression":
            return (
                "Prolonged depression stage warrants professional support. "
                "Consider reaching out to a grief counselor or therapist."
            )
        return (
            "Consider trying a new approach or activity that might gently "
            "shift your perspective. Small changes can help."
        )
    if trend == TrajectoryTrend.REGRESSING:
        return (
            "Regression often happens around triggers: anniversaries, "
            "holidays, or unexpected reminders. This is temporary."
        )
    if trend == TrajectoryTrend.OSCILLATING:
        return (
            "Oscillation is the normal pattern of grief. Focus on "
            "self-care and maintaining your daily routine."
        )
    return "Continue at your own pace. There is no timeline for grief."


def select_support_intervention(
    stage: str,
    intensity_preference: str = "gentle",
) -> Dict[str, Any]:
    """Select a stage-appropriate support intervention.

    Matches interventions to the detected grief stage and filters by
    intensity preference.

    Args:
        stage: Detected grief stage (string value).
        intensity_preference: "gentle", "moderate", or "active".

    Returns:
        Dict with the selected intervention and alternatives.
    """
    try:
        grief_stage = GriefStage(stage)
    except ValueError:
        return {
            "selected": False,
            "reason": "unknown_stage",
            "message": f"Unknown grief stage: {stage}",
            "clinical_disclaimer": _CLINICAL_DISCLAIMER,
        }

    if grief_stage == GriefStage.UNKNOWN:
        return {
            "selected": False,
            "reason": "stage_unclear",
            "message": (
                "Your current grief stage is not clearly identifiable. "
                "That is okay -- grief does not always fit neatly into stages. "
                "Try a general self-care activity and check in again later."
            ),
            "clinical_disclaimer": _CLINICAL_DISCLAIMER,
        }

    candidates = _INTERVENTIONS_BY_STAGE.get(grief_stage, [])
    if not candidates:
        return {
            "selected": False,
            "reason": "no_interventions_available",
            "message": "No interventions available for this stage.",
            "clinical_disclaimer": _CLINICAL_DISCLAIMER,
        }

    # Prefer matching intensity, but fall back to any
    preferred = [c for c in candidates if c.intensity == intensity_preference]
    selected = preferred[0] if preferred else candidates[0]
    alternatives = [c for c in candidates if c.name != selected.name]

    return {
        "selected": True,
        "intervention": selected.to_dict(),
        "alternatives": [a.to_dict() for a in alternatives],
        "stage": grief_stage.value,
        "worden_task": _STAGE_TO_WORDEN[grief_stage].value,
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
        "crisis_resources": _CRISIS_RESOURCES,
    }


def detect_anniversary_effect(
    current_reading: GriefReading,
    recent_readings: List[GriefReading],
    significant_dates: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Detect anniversary effects -- emotional dips around significant dates.

    Anniversary reactions are well-documented in grief literature: emotional
    regression around dates of the loss, birthdays, holidays, etc.

    Args:
        current_reading: Current emotional state.
        recent_readings: Recent readings for baseline comparison.
        significant_dates: Optional list of dicts with "date" (ISO format)
            and "description" keys.

    Returns:
        Dict with anniversary detection result and guidance.
    """
    # Compute recent valence baseline
    if len(recent_readings) < 3:
        baseline_valence = 0.0
    else:
        baseline_valence = sum(r.valence for r in recent_readings) / len(recent_readings)

    valence_dip = baseline_valence - current_reading.valence
    dip_detected = valence_dip >= _ANNIVERSARY_DIP_THRESHOLD

    # Check if near a significant date
    near_significant_date = False
    date_info = None
    if significant_dates:
        now = time.time()
        for sd in significant_dates:
            date_str = sd.get("date", "")
            desc = sd.get("description", "significant date")
            # Simple check: if a timestamp is provided in the date dict
            sd_timestamp = sd.get("timestamp")
            if sd_timestamp is not None:
                days_diff = abs(now - sd_timestamp) / 86400
                if days_diff <= _ANNIVERSARY_WINDOW_DAYS:
                    near_significant_date = True
                    date_info = desc
                    break

    anniversary_detected = dip_detected and (near_significant_date or len(recent_readings) >= 3)

    guidance = ""
    if anniversary_detected and near_significant_date:
        guidance = (
            f"It looks like you may be experiencing an anniversary reaction "
            f"near {date_info}. This is very common in grief. The body and "
            f"mind remember, even when we are not consciously thinking about "
            f"it. Be extra gentle with yourself around these times."
        )
    elif dip_detected:
        guidance = (
            "There has been a noticeable dip in your emotional state compared "
            "to recent readings. This could be an anniversary reaction or "
            "simply a difficult day. Either way, it is okay to not be okay."
        )
    else:
        guidance = "No significant anniversary effect detected at this time."

    return {
        "anniversary_detected": anniversary_detected,
        "emotional_dip_detected": dip_detected,
        "valence_dip": round(valence_dip, 4),
        "baseline_valence": round(baseline_valence, 4),
        "current_valence": round(current_reading.valence, 4),
        "near_significant_date": near_significant_date,
        "date_info": date_info,
        "guidance": guidance,
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }


def check_safety(
    current_reading: GriefReading,
    recent_readings: Optional[List[GriefReading]] = None,
) -> Dict[str, Any]:
    """Check safety markers for suicidal ideation and severe distress.

    Monitors for:
    - Prolonged flat affect (emotional numbness that may mask despair)
    - High isolation combined with hopelessness
    - Severe depression indicators
    - Explicit hopelessness markers

    ALWAYS provides crisis resources regardless of assessment level.

    Args:
        current_reading: Current emotional state.
        recent_readings: Optional recent history for pattern detection.

    Returns:
        Dict with safety level, notes, crisis resources, and guidance.
    """
    safety_level = SafetyLevel.SAFE
    notes: List[str] = []
    recent = recent_readings or []

    # Check current reading markers
    if current_reading.hopelessness_index >= _SAFETY_HOPELESSNESS_THRESHOLD:
        safety_level = SafetyLevel.CONCERN
        notes.append("High hopelessness detected")

    if current_reading.isolation_index >= 0.7 and current_reading.hopelessness_index >= 0.5:
        safety_level = SafetyLevel.CRITICAL
        notes.append("High isolation combined with hopelessness")

    if current_reading.valence <= -0.7 and current_reading.arousal <= 0.15:
        if safety_level.value in ("safe", "monitor"):
            safety_level = SafetyLevel.CONCERN
        notes.append("Very negative valence with very low arousal")

    # Check prolonged flat affect in history
    if len(recent) >= _SAFETY_FLAT_AFFECT_DURATION:
        last_n = recent[-_SAFETY_FLAT_AFFECT_DURATION:]
        all_flat = all(
            abs(r.valence) <= 0.15 and r.arousal <= 0.25
            for r in last_n
        )
        if all_flat:
            if safety_level != SafetyLevel.CRITICAL:
                safety_level = SafetyLevel.CONCERN
            notes.append("Prolonged flat affect detected across multiple readings")

    # Check for sustained depression pattern
    if len(recent) >= 3:
        recent_depressed = sum(
            1 for r in recent[-3:]
            if r.valence <= _DEPRESSION_VALENCE_CEILING
            and r.arousal <= _DEPRESSION_AROUSAL_CEILING
        )
        if recent_depressed >= 3:
            if safety_level == SafetyLevel.SAFE:
                safety_level = SafetyLevel.MONITOR
            notes.append("Sustained depression pattern in recent readings")

    # Generate safety guidance
    guidance_map = {
        SafetyLevel.SAFE: (
            "No immediate safety concerns detected. Continue processing "
            "your grief at your own pace. Remember: asking for help is "
            "always okay."
        ),
        SafetyLevel.MONITOR: (
            "Some patterns suggest you may benefit from additional support. "
            "Consider reaching out to a trusted friend, family member, or "
            "grief counselor."
        ),
        SafetyLevel.CONCERN: (
            "Your current state shows markers that warrant attention. "
            "Please consider reaching out to a mental health professional. "
            "You do not have to go through this alone."
        ),
        SafetyLevel.CRITICAL: (
            "Please reach out for help right now. You are not alone, and "
            "people care about you. Call 988 (Suicide & Crisis Lifeline) "
            "or text HOME to 741741 (Crisis Text Line). If you are in "
            "immediate danger, call 911 or go to your nearest emergency room."
        ),
    }

    return {
        "safety_level": safety_level.value,
        "notes": notes,
        "guidance": guidance_map[safety_level],
        "crisis_resources": _CRISIS_RESOURCES,
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }


def compute_grief_profile(
    current_reading: GriefReading,
    recent_readings: Optional[List[GriefReading]] = None,
    stage_history: Optional[List[Dict[str, Any]]] = None,
    significant_dates: Optional[List[Dict[str, Any]]] = None,
) -> GriefProfile:
    """Compute a comprehensive grief profile from current and historical data.

    Combines stage detection, trajectory tracking, safety checking,
    anniversary detection, and intervention selection into a single
    coherent profile.

    Args:
        current_reading: Current emotional state.
        recent_readings: Optional recent reading history.
        stage_history: Optional list of previous stage detection results.
        significant_dates: Optional significant dates for anniversary detection.

    Returns:
        GriefProfile with all assessments combined.
    """
    recent = recent_readings or []
    history = stage_history or []

    # Detect stage
    stage_result = detect_grief_stage(current_reading)
    stage = GriefStage(stage_result["stage"])
    confidence = stage_result["confidence"]
    stage_indicators = stage_result["indicators"]

    # Track trajectory
    if history:
        trajectory_result = track_grief_trajectory(history)
        trajectory = TrajectoryTrend(trajectory_result["trend"])
    else:
        trajectory = TrajectoryTrend.INSUFFICIENT_DATA

    # Check safety
    safety_result = check_safety(current_reading, recent)
    safety = SafetyLevel(safety_result["safety_level"])
    safety_notes = safety_result["notes"]

    # Detect anniversary effects
    anniversary_result = detect_anniversary_effect(
        current_reading, recent, significant_dates
    )
    anniversary_detected = anniversary_result["anniversary_detected"]

    # Select intervention
    intervention_result = select_support_intervention(stage.value)
    intervention = None
    if intervention_result.get("selected"):
        inv_data = intervention_result["intervention"]
        intervention = SupportIntervention(
            stage=GriefStage(inv_data["stage"]),
            worden_task=WordenTask(inv_data["worden_task"]),
            name=inv_data["name"],
            description=inv_data["description"],
            guidance=inv_data["guidance"],
            duration_minutes=inv_data["duration_minutes"],
            intensity=inv_data["intensity"],
        )

    worden_task = _STAGE_TO_WORDEN[stage]

    return GriefProfile(
        stage=stage,
        stage_confidence=confidence,
        stage_indicators=stage_indicators,
        worden_task=worden_task,
        trajectory=trajectory,
        safety=safety,
        safety_notes=safety_notes,
        intervention=intervention,
        anniversary_detected=anniversary_detected,
        readings_count=len(recent) + 1,
    )


def profile_to_dict(profile: GriefProfile) -> Dict[str, Any]:
    """Convert a GriefProfile to a serializable dict.

    Args:
        profile: GriefProfile instance.

    Returns:
        Dict representation of the profile.
    """
    return profile.to_dict()
