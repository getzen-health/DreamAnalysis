"""Micro-intervention engine: just-in-time 30-second interventions.

Selects and times brief interventions (breathing, grounding, movement,
cognitive) at the biologically optimal moment based on emotion stream
analysis.  Monitors for stress spikes, focus decline, energy crashes,
and pre-meeting anxiety.  Tracks per-user effectiveness and learns
optimal timing across circadian phase, stress level, and context.

Avoids intervention fatigue through cooldown periods, daily caps,
and flow-state detection.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COOLDOWN_SECONDS = 900          # 15-minute cooldown between interventions
MAX_INTERVENTIONS_PER_DAY = 8   # prevent fatigue
FLOW_STATE_THRESHOLD = 0.70     # don't interrupt if flow >= this
STRESS_SPIKE_VALENCE_DROP = 0.25
STRESS_SPIKE_AROUSAL_MIN = 0.65
FOCUS_DECLINE_THRESHOLD = 0.30
ENERGY_CRASH_AROUSAL_MAX = 0.25
ENERGY_CRASH_VALENCE_MAX = 0.30
PRE_MEETING_MINUTES = 10        # anxiety window before a meeting
OUTCOME_WINDOW_SECONDS = 300    # 5 min to measure post-intervention change

# Circadian phase bins (hours of day -> phase label)
CIRCADIAN_PHASES = [
    (0, 6, "night"),
    (6, 10, "morning"),
    (10, 14, "midday"),
    (14, 17, "afternoon"),
    (17, 21, "evening"),
    (21, 24, "night"),
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InterventionCategory(str, Enum):
    BREATHING = "breathing"
    GROUNDING = "grounding"
    MOVEMENT = "movement"
    COGNITIVE = "cognitive"


class TriggerType(str, Enum):
    STRESS_SPIKE = "stress_spike"
    FOCUS_DECLINE = "focus_decline"
    ENERGY_CRASH = "energy_crash"
    PRE_MEETING_ANXIETY = "pre_meeting_anxiety"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EmotionState:
    """Snapshot of a user's emotional state at a point in time."""

    valence: float          # -1.0 to 1.0 (negative to positive)
    arousal: float          # 0.0 to 1.0 (calm to energetic)
    stress_index: float     # 0.0 to 1.0
    focus_index: float      # 0.0 to 1.0
    flow_index: float = 0.0  # 0.0 to 1.0
    timestamp: float = 0.0
    context: str = ""       # e.g. "working", "meeting", "break"

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class Intervention:
    """A single intervention from the library."""

    id: str
    name: str
    category: InterventionCategory
    duration_seconds: int = 30
    description: str = ""
    instructions: str = ""
    evidence: str = ""
    intensity_range: Tuple[float, float] = (0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "duration_seconds": self.duration_seconds,
            "description": self.description,
            "instructions": self.instructions,
            "evidence": self.evidence,
            "intensity_range": list(self.intensity_range),
        }


@dataclass
class InterventionOutcome:
    """Recorded outcome after an intervention was delivered."""

    user_id: str
    intervention_id: str
    trigger_type: str
    valence_before: float
    arousal_before: float
    valence_after: float
    arousal_after: float
    stress_before: float = 0.0
    stress_after: float = 0.0
    felt_helpful: Optional[bool] = None
    circadian_phase: str = ""
    context: str = ""
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def valence_delta(self) -> float:
        """Positive = improvement (valence went up)."""
        return self.valence_after - self.valence_before

    @property
    def arousal_delta(self) -> float:
        """Negative = calming (arousal went down after stress spike)."""
        return self.arousal_after - self.arousal_before

    @property
    def stress_delta(self) -> float:
        """Positive = improvement (stress went down)."""
        return self.stress_before - self.stress_after

    @property
    def effective(self) -> bool:
        """Did the intervention measurably help?"""
        if self.felt_helpful is True:
            return True
        # Valence improved OR stress dropped meaningfully
        return self.valence_delta > 0.05 or self.stress_delta > 0.05


@dataclass
class TriggerEvent:
    """Record of when and why a trigger fired."""

    trigger_type: TriggerType
    emotion_state: EmotionState
    intervention_id: Optional[str] = None
    timestamp: float = 0.0
    suppressed: bool = False       # True if blocked by cooldown/flow/cap
    suppression_reason: str = ""

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# ---------------------------------------------------------------------------
# Intervention Library
# ---------------------------------------------------------------------------

INTERVENTION_LIBRARY: List[Intervention] = [
    # --- Breathing ---
    Intervention(
        id="box_breathing",
        name="Box Breathing",
        category=InterventionCategory.BREATHING,
        duration_seconds=30,
        description="4-count box breathing pattern",
        instructions="Inhale 4s, hold 4s, exhale 4s, hold 4s. Repeat.",
        evidence="Ma et al. (2017): Diaphragmatic breathing reduces cortisol. Navy SEALs protocol.",
        intensity_range=(0.3, 0.8),
    ),
    Intervention(
        id="breathing_478",
        name="4-7-8 Breathing",
        category=InterventionCategory.BREATHING,
        duration_seconds=30,
        description="Relaxation breathing: inhale 4, hold 7, exhale 8",
        instructions="Inhale through nose 4s, hold 7s, exhale through mouth 8s. Two rounds.",
        evidence="Weil (2015): Activates parasympathetic nervous system. Best for acute anxiety.",
        intensity_range=(0.5, 1.0),
    ),
    Intervention(
        id="physiological_sigh",
        name="Physiological Sigh",
        category=InterventionCategory.BREATHING,
        duration_seconds=30,
        description="Double inhale followed by extended exhale",
        instructions="Double inhale through nose (quick-quick), long exhale through mouth. Repeat 3x.",
        evidence="Balban et al. (2023, Cell Reports Medicine): Outperformed mindfulness for mood improvement in RCT.",
        intensity_range=(0.6, 1.0),
    ),
    # --- Grounding ---
    Intervention(
        id="grounding_54321",
        name="5-4-3-2-1 Senses",
        category=InterventionCategory.GROUNDING,
        duration_seconds=30,
        description="Sensory grounding exercise",
        instructions="Name 5 things you see, 4 hear, 3 touch, 2 smell, 1 taste.",
        evidence="Evidence-informed for anxiety and dissociation. Sensory modality engagement.",
        intensity_range=(0.5, 1.0),
    ),
    # --- Movement ---
    Intervention(
        id="desk_stretch",
        name="Quick Desk Stretch",
        category=InterventionCategory.MOVEMENT,
        duration_seconds=30,
        description="30-second seated stretch routine",
        instructions="Neck rolls 10s, shoulder shrugs 10s, wrist circles 10s.",
        evidence="Workplace ergonomics: micro-breaks reduce musculoskeletal strain and mental fatigue.",
        intensity_range=(0.2, 0.6),
    ),
    Intervention(
        id="walk_break",
        name="Walk Break",
        category=InterventionCategory.MOVEMENT,
        duration_seconds=30,
        description="Stand and walk for 30 seconds",
        instructions="Stand up, walk to a window or around the room. Focus on your footsteps.",
        evidence="Kang et al. (2020): 5-min nature walks reduce perceived stress.",
        intensity_range=(0.3, 0.7),
    ),
    # --- Cognitive ---
    Intervention(
        id="cognitive_reframe",
        name="Quick Reframe",
        category=InterventionCategory.COGNITIVE,
        duration_seconds=30,
        description="Cognitive reappraisal micro-exercise",
        instructions="Name the stressor. Ask: 'What is another way to see this?' Write one alternative.",
        evidence="30+ years of research on cognitive reappraisal. Most effective at low-to-moderate intensity.",
        intensity_range=(0.0, 0.5),
    ),
    Intervention(
        id="gratitude_pause",
        name="Gratitude Pause",
        category=InterventionCategory.COGNITIVE,
        duration_seconds=30,
        description="Name three things you are grateful for",
        instructions="Pause. Name 3 things you appreciate right now. Be specific.",
        evidence="Emmons & McCullough (2003): Gratitude exercises improve well-being and reduce stress.",
        intensity_range=(0.0, 0.4),
    ),
]

_LIBRARY_BY_ID: Dict[str, Intervention] = {i.id: i for i in INTERVENTION_LIBRARY}
_LIBRARY_BY_CATEGORY: Dict[InterventionCategory, List[Intervention]] = {}
for _iv in INTERVENTION_LIBRARY:
    _LIBRARY_BY_CATEGORY.setdefault(_iv.category, []).append(_iv)


# ---------------------------------------------------------------------------
# Helper: circadian phase from hour of day
# ---------------------------------------------------------------------------

def _get_circadian_phase(hour: float) -> str:
    """Map hour of day (0-24) to a circadian phase label."""
    hour = hour % 24
    for start, end, label in CIRCADIAN_PHASES:
        if start <= hour < end:
            return label
    return "night"


# ---------------------------------------------------------------------------
# InterventionEngine
# ---------------------------------------------------------------------------

class InterventionEngine:
    """Main engine: trigger detection, intervention selection, outcome tracking.

    Thread-safe via simple dict-per-user pattern (no shared mutable state
    across users).
    """

    def __init__(self) -> None:
        # user_id -> list of outcomes
        self._outcomes: Dict[str, List[InterventionOutcome]] = {}
        # user_id -> list of trigger events
        self._triggers: Dict[str, List[TriggerEvent]] = {}
        # user_id -> previous emotion state (for delta detection)
        self._prev_state: Dict[str, EmotionState] = {}
        # user_id -> timestamp of last delivered intervention
        self._last_intervention_ts: Dict[str, float] = {}
        # user_id -> count of interventions today (keyed by day string)
        self._daily_counts: Dict[str, Dict[str, int]] = {}

    # ── Trigger Detection ─────────────────────────────────────────────────

    def check_trigger(
        self,
        user_id: str,
        state: EmotionState,
        upcoming_meeting_minutes: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Check if the current emotion state warrants an intervention.

        Returns a dict with:
          triggered: bool
          trigger_type: str or None
          suppressed: bool (True if trigger fired but was blocked)
          suppression_reason: str
        """
        trigger_type: Optional[TriggerType] = None

        prev = self._prev_state.get(user_id)
        self._prev_state[user_id] = state

        # --- Stress spike: valence drops + arousal spikes ---
        if prev is not None:
            valence_drop = prev.valence - state.valence
            if (
                valence_drop >= STRESS_SPIKE_VALENCE_DROP
                and state.arousal >= STRESS_SPIKE_AROUSAL_MIN
            ):
                trigger_type = TriggerType.STRESS_SPIKE

        # High absolute stress also counts
        if trigger_type is None and state.stress_index >= 0.75:
            trigger_type = TriggerType.STRESS_SPIKE

        # --- Focus decline ---
        if trigger_type is None and state.focus_index <= FOCUS_DECLINE_THRESHOLD:
            trigger_type = TriggerType.FOCUS_DECLINE

        # --- Energy crash ---
        if (
            trigger_type is None
            and state.arousal <= ENERGY_CRASH_AROUSAL_MAX
            and state.valence <= ENERGY_CRASH_VALENCE_MAX
        ):
            trigger_type = TriggerType.ENERGY_CRASH

        # --- Pre-meeting anxiety ---
        if (
            trigger_type is None
            and upcoming_meeting_minutes is not None
            and upcoming_meeting_minutes <= PRE_MEETING_MINUTES
            and state.stress_index >= 0.50
        ):
            trigger_type = TriggerType.PRE_MEETING_ANXIETY

        if trigger_type is None:
            return {
                "triggered": False,
                "trigger_type": None,
                "suppressed": False,
                "suppression_reason": "",
            }

        # --- Suppression checks ---
        suppressed, reason = self._check_suppression(user_id, state)

        event = TriggerEvent(
            trigger_type=trigger_type,
            emotion_state=state,
            suppressed=suppressed,
            suppression_reason=reason,
        )
        self._triggers.setdefault(user_id, []).append(event)

        return {
            "triggered": not suppressed,
            "trigger_type": trigger_type.value,
            "suppressed": suppressed,
            "suppression_reason": reason,
        }

    def _check_suppression(
        self, user_id: str, state: EmotionState
    ) -> Tuple[bool, str]:
        """Check if intervention should be suppressed."""
        now = time.time()

        # Flow state protection
        if state.flow_index >= FLOW_STATE_THRESHOLD:
            return True, "flow_state"

        # Cooldown period
        last_ts = self._last_intervention_ts.get(user_id, 0.0)
        if now - last_ts < COOLDOWN_SECONDS:
            return True, "cooldown"

        # Daily cap
        day_key = time.strftime("%Y-%m-%d")
        day_counts = self._daily_counts.setdefault(user_id, {})
        if day_counts.get(day_key, 0) >= MAX_INTERVENTIONS_PER_DAY:
            return True, "daily_cap"

        return False, ""

    # ── Intervention Selection ────────────────────────────────────────────

    def select_intervention(
        self,
        user_id: str,
        state: EmotionState,
        trigger_type: str,
        hour_of_day: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Select the best intervention for the current state and trigger.

        Uses effectiveness history to personalize, falls back to
        evidence-based defaults when no history exists.
        """
        now = time.time()
        if hour_of_day is None:
            hour_of_day = float(time.localtime().tm_hour) + time.localtime().tm_min / 60.0

        phase = _get_circadian_phase(hour_of_day)
        intensity = state.stress_index

        # Filter candidates by category preference for this trigger
        candidates = self._candidates_for_trigger(trigger_type)

        # Filter by intensity range
        candidates = [
            c for c in candidates
            if c.intensity_range[0] <= intensity <= c.intensity_range[1]
        ]

        if not candidates:
            # Fallback: any breathing exercise
            candidates = _LIBRARY_BY_CATEGORY.get(
                InterventionCategory.BREATHING, INTERVENTION_LIBRARY[:1]
            )

        # Score candidates using effectiveness history
        scored = self._score_candidates(user_id, candidates, phase, intensity)

        # Pick the best
        best = scored[0]

        # Record delivery
        self._last_intervention_ts[user_id] = now
        day_key = time.strftime("%Y-%m-%d")
        day_counts = self._daily_counts.setdefault(user_id, {})
        day_counts[day_key] = day_counts.get(day_key, 0) + 1

        return {
            "intervention": best.to_dict(),
            "trigger_type": trigger_type,
            "circadian_phase": phase,
            "selection_scores": {
                c.id: round(self._effectiveness_score(user_id, c.id, phase, intensity), 3)
                for c in scored[:5]
            },
        }

    def _candidates_for_trigger(self, trigger_type: str) -> List[Intervention]:
        """Return category-filtered candidates appropriate for a trigger type."""
        category_map: Dict[str, List[InterventionCategory]] = {
            TriggerType.STRESS_SPIKE.value: [
                InterventionCategory.BREATHING,
                InterventionCategory.GROUNDING,
            ],
            TriggerType.FOCUS_DECLINE.value: [
                InterventionCategory.MOVEMENT,
                InterventionCategory.COGNITIVE,
            ],
            TriggerType.ENERGY_CRASH.value: [
                InterventionCategory.MOVEMENT,
                InterventionCategory.BREATHING,
            ],
            TriggerType.PRE_MEETING_ANXIETY.value: [
                InterventionCategory.BREATHING,
                InterventionCategory.COGNITIVE,
            ],
        }
        categories = category_map.get(trigger_type, list(InterventionCategory))
        result: List[Intervention] = []
        for cat in categories:
            result.extend(_LIBRARY_BY_CATEGORY.get(cat, []))
        return result

    def _score_candidates(
        self,
        user_id: str,
        candidates: List[Intervention],
        phase: str,
        intensity: float,
    ) -> List[Intervention]:
        """Sort candidates by effectiveness score (descending)."""
        return sorted(
            candidates,
            key=lambda c: self._effectiveness_score(user_id, c.id, phase, intensity),
            reverse=True,
        )

    def _effectiveness_score(
        self,
        user_id: str,
        intervention_id: str,
        phase: str,
        intensity: float,
    ) -> float:
        """Compute a score for how effective this intervention is for this user.

        Combines:
          - Historical success rate (weighted most heavily)
          - Phase-specific success rate bonus
          - Recency bonus (more recent outcomes weighted higher)
          - Default score of 0.5 when no history (neutral prior)
        """
        outcomes = self._outcomes.get(user_id, [])
        relevant = [o for o in outcomes if o.intervention_id == intervention_id]

        if not relevant:
            return 0.5  # neutral prior

        now = time.time()
        total_weight = 0.0
        weighted_success = 0.0
        phase_bonus = 0.0
        phase_count = 0

        for o in relevant:
            # Recency weight: half-life of 7 days
            age_days = (now - o.timestamp) / 86400.0
            weight = math.exp(-0.1 * age_days)
            total_weight += weight
            if o.effective:
                weighted_success += weight

            # Phase-specific bonus
            if o.circadian_phase == phase and o.effective:
                phase_bonus += 0.1
                phase_count += 1

        base_score = weighted_success / total_weight if total_weight > 0 else 0.5
        phase_adj = min(0.2, phase_bonus)  # cap phase bonus at 0.2
        return min(1.0, base_score + phase_adj)

    # ── Outcome Recording ─────────────────────────────────────────────────

    def record_outcome(
        self,
        user_id: str,
        intervention_id: str,
        trigger_type: str,
        valence_before: float,
        arousal_before: float,
        valence_after: float,
        arousal_after: float,
        stress_before: float = 0.0,
        stress_after: float = 0.0,
        felt_helpful: Optional[bool] = None,
        hour_of_day: Optional[float] = None,
        context: str = "",
    ) -> Dict[str, Any]:
        """Record intervention outcome for learning."""
        if hour_of_day is None:
            hour_of_day = float(time.localtime().tm_hour) + time.localtime().tm_min / 60.0

        phase = _get_circadian_phase(hour_of_day)

        outcome = InterventionOutcome(
            user_id=user_id,
            intervention_id=intervention_id,
            trigger_type=trigger_type,
            valence_before=valence_before,
            arousal_before=arousal_before,
            valence_after=valence_after,
            arousal_after=arousal_after,
            stress_before=stress_before,
            stress_after=stress_after,
            felt_helpful=felt_helpful,
            circadian_phase=phase,
            context=context,
        )
        self._outcomes.setdefault(user_id, []).append(outcome)

        return {
            "recorded": True,
            "effective": outcome.effective,
            "valence_delta": round(outcome.valence_delta, 3),
            "arousal_delta": round(outcome.arousal_delta, 3),
            "stress_delta": round(outcome.stress_delta, 3),
            "circadian_phase": phase,
        }

    # ── Effectiveness Analytics ────────────────────────────────────────────

    def compute_effectiveness(
        self,
        user_id: str,
        intervention_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute effectiveness stats for a user, optionally filtered by intervention."""
        outcomes = self._outcomes.get(user_id, [])
        if intervention_id:
            outcomes = [o for o in outcomes if o.intervention_id == intervention_id]

        if not outcomes:
            return {
                "user_id": user_id,
                "total_outcomes": 0,
                "success_rate": 0.0,
                "avg_valence_delta": 0.0,
                "avg_stress_delta": 0.0,
                "by_intervention": {},
                "by_phase": {},
                "by_trigger": {},
            }

        total = len(outcomes)
        successes = sum(1 for o in outcomes if o.effective)
        avg_valence = sum(o.valence_delta for o in outcomes) / total
        avg_stress = sum(o.stress_delta for o in outcomes) / total

        # Breakdown by intervention
        by_intervention: Dict[str, Dict[str, Any]] = {}
        for o in outcomes:
            entry = by_intervention.setdefault(o.intervention_id, {
                "count": 0, "successes": 0, "valence_deltas": [], "stress_deltas": [],
            })
            entry["count"] += 1
            if o.effective:
                entry["successes"] += 1
            entry["valence_deltas"].append(o.valence_delta)
            entry["stress_deltas"].append(o.stress_delta)

        by_intervention_summary: Dict[str, Dict[str, Any]] = {}
        for iid, data in by_intervention.items():
            by_intervention_summary[iid] = {
                "count": data["count"],
                "success_rate": round(data["successes"] / data["count"], 3),
                "avg_valence_delta": round(sum(data["valence_deltas"]) / data["count"], 3),
                "avg_stress_delta": round(sum(data["stress_deltas"]) / data["count"], 3),
            }

        # Breakdown by circadian phase
        by_phase: Dict[str, Dict[str, Any]] = {}
        for o in outcomes:
            entry = by_phase.setdefault(o.circadian_phase or "unknown", {
                "count": 0, "successes": 0,
            })
            entry["count"] += 1
            if o.effective:
                entry["successes"] += 1

        by_phase_summary: Dict[str, Dict[str, Any]] = {}
        for phase, data in by_phase.items():
            by_phase_summary[phase] = {
                "count": data["count"],
                "success_rate": round(data["successes"] / data["count"], 3),
            }

        # Breakdown by trigger type
        by_trigger: Dict[str, Dict[str, Any]] = {}
        for o in outcomes:
            entry = by_trigger.setdefault(o.trigger_type or "unknown", {
                "count": 0, "successes": 0,
            })
            entry["count"] += 1
            if o.effective:
                entry["successes"] += 1

        by_trigger_summary: Dict[str, Dict[str, Any]] = {}
        for tt, data in by_trigger.items():
            by_trigger_summary[tt] = {
                "count": data["count"],
                "success_rate": round(data["successes"] / data["count"], 3),
            }

        return {
            "user_id": user_id,
            "total_outcomes": total,
            "success_rate": round(successes / total, 3),
            "avg_valence_delta": round(avg_valence, 3),
            "avg_stress_delta": round(avg_stress, 3),
            "by_intervention": by_intervention_summary,
            "by_phase": by_phase_summary,
            "by_trigger": by_trigger_summary,
        }

    def get_intervention_stats(self, user_id: str) -> Dict[str, Any]:
        """High-level stats: counts, daily usage, recent triggers."""
        outcomes = self._outcomes.get(user_id, [])
        triggers = self._triggers.get(user_id, [])

        day_key = time.strftime("%Y-%m-%d")
        day_counts = self._daily_counts.get(user_id, {})
        today_count = day_counts.get(day_key, 0)

        last_ts = self._last_intervention_ts.get(user_id, 0.0)
        now = time.time()
        cooldown_remaining = max(0.0, COOLDOWN_SECONDS - (now - last_ts)) if last_ts > 0 else 0.0

        return {
            "user_id": user_id,
            "total_outcomes": len(outcomes),
            "total_triggers": len(triggers),
            "suppressed_triggers": sum(1 for t in triggers if t.suppressed),
            "today_count": today_count,
            "max_per_day": MAX_INTERVENTIONS_PER_DAY,
            "cooldown_remaining_seconds": round(cooldown_remaining, 1),
            "library_size": len(INTERVENTION_LIBRARY),
        }

    # ── Serialization ─────────────────────────────────────────────────────

    def engine_to_dict(self) -> Dict[str, Any]:
        """Serialize engine state for debugging / persistence."""
        return {
            "users": list(self._outcomes.keys() | self._triggers.keys()),
            "total_outcomes": sum(len(v) for v in self._outcomes.values()),
            "total_triggers": sum(len(v) for v in self._triggers.values()),
            "library": [i.to_dict() for i in INTERVENTION_LIBRARY],
        }
