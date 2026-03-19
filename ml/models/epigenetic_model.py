"""Epigenetic emotional inheritance model.

Surfaces intergenerational stress patterns by comparing a user's longitudinal
emotional data against reported family history.  Inspired by emerging research
on epigenetic transmission of stress reactivity (Yehuda et al., 2016; Bowers &
Yehuda, 2016) and the Adverse Childhood Experiences (ACE) framework.

Core concepts:
  - Family emotional history intake: structured questionnaire about
    parents/grandparents (anxiety, depression, trauma, addiction, resilience).
  - Intergenerational pattern detection: compare user's emotional patterns
    to reported family patterns.
  - Stress inheritance markers: heightened baseline stress, specific trigger
    sensitivities matching family history.
  - Protective factor detection: where has the user broken family patterns?
  - Risk amplification vs attenuation scoring: which inherited patterns are
    getting stronger vs weaker across generations.
  - Generational healing tracking: is the user's emotional profile healthier
    than their parents' reported profile?

Usage:
    engine = EpigeneticEngine()
    engine.intake_family_history(user_id, [member1, member2, ...])
    engine.add_emotional_snapshot(user_id, date, valence, arousal, stress, ...)
    patterns  = engine.detect_inherited_patterns(user_id)
    risk      = engine.compute_risk_attenuation(user_id)
    factors   = engine.detect_protective_factors(user_id)
    healing   = engine.track_generational_healing(user_id)
    profile   = engine.compute_epigenetic_profile(user_id)
    serialized = engine.profile_to_dict(user_id)

GitHub issue: #449
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

CONDITION_TYPES = [
    "anxiety",
    "depression",
    "trauma",
    "addiction",
    "resilience",
    "anger",
    "grief",
    "phobia",
]

RELATION_TYPES = [
    "mother",
    "father",
    "maternal_grandmother",
    "maternal_grandfather",
    "paternal_grandmother",
    "paternal_grandfather",
    "sibling",
]

# Generational weight: closer relatives have stronger expected influence.
_RELATION_WEIGHTS: Dict[str, float] = {
    "mother": 1.0,
    "father": 1.0,
    "maternal_grandmother": 0.5,
    "maternal_grandfather": 0.5,
    "paternal_grandmother": 0.5,
    "paternal_grandfather": 0.5,
    "sibling": 0.3,
}

_MAX_FAMILY_MEMBERS = 20
_MAX_SNAPSHOTS_PER_USER = 1000


# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class FamilyMember:
    """A single family member's emotional history."""
    relation: str               # one of RELATION_TYPES
    conditions: List[str]       # subset of CONDITION_TYPES
    severity: float             # 0-10 overall severity rating
    notes: str = ""
    timestamp: float = 0.0


@dataclass
class EmotionalSnapshot:
    """A single point-in-time emotional reading for the user."""
    date: str                   # YYYY-MM-DD
    valence: float              # -1 to 1
    arousal: float              # 0 to 1
    stress_level: float         # 0 to 1
    trigger: str = ""           # optional trigger label
    timestamp: float = 0.0


@dataclass
class InheritedPattern:
    """A detected intergenerational emotional pattern."""
    condition: str              # e.g. "anxiety"
    family_prevalence: float    # 0-1 how common in family
    user_match_score: float     # 0-1 how strongly user shows this
    direction: str              # "amplified", "attenuated", "stable"
    contributing_relations: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class EpigeneticProfile:
    """Full epigenetic emotional inheritance profile for a user."""
    user_id: str
    inherited_patterns: List[InheritedPattern]
    risk_score: float           # 0-1 overall inherited risk
    attenuation_score: float    # 0-1 how much risk is being reduced
    protective_factors: List[str]
    healing_score: float        # 0-1 generational healing progress
    assessment_date: str
    timestamp: float = 0.0


# ── Core engine ────────────────────────────────────────────────────────────

class EpigeneticEngine:
    """Surface intergenerational stress patterns from family history +
    longitudinal emotional data.

    Stateful: maintains per-user family history and emotional snapshots.
    """

    def __init__(self) -> None:
        self._family_history: Dict[str, List[FamilyMember]] = defaultdict(list)
        self._emotional_data: Dict[str, List[EmotionalSnapshot]] = defaultdict(list)

    # ── Family history intake ──────────────────────────────────────────────

    def intake_family_history(
        self,
        user_id: str,
        members: List[FamilyMember],
    ) -> List[FamilyMember]:
        """Record family member emotional history for a user.

        Validates relation types and condition types, clamps severity.
        Replaces any existing history for this user.

        Args:
            user_id: User identifier.
            members: List of FamilyMember dataclass instances.

        Returns:
            The validated, stored list of FamilyMember objects.
        """
        validated: List[FamilyMember] = []
        for m in members[:_MAX_FAMILY_MEMBERS]:
            relation = m.relation if m.relation in RELATION_TYPES else "mother"
            conditions = [c for c in m.conditions if c in CONDITION_TYPES]
            severity = max(0.0, min(10.0, float(m.severity)))
            validated.append(FamilyMember(
                relation=relation,
                conditions=conditions,
                severity=severity,
                notes=m.notes,
                timestamp=time.time(),
            ))
        self._family_history[user_id] = validated
        return validated

    # ── Emotional data ingestion ───────────────────────────────────────────

    def add_emotional_snapshot(
        self,
        user_id: str,
        date: str,
        valence: float,
        arousal: float,
        stress_level: float,
        trigger: str = "",
    ) -> EmotionalSnapshot:
        """Add a single emotional data point for longitudinal tracking."""
        snapshot = EmotionalSnapshot(
            date=date,
            valence=max(-1.0, min(1.0, float(valence))),
            arousal=max(0.0, min(1.0, float(arousal))),
            stress_level=max(0.0, min(1.0, float(stress_level))),
            trigger=trigger,
            timestamp=time.time(),
        )
        snapshots = self._emotional_data[user_id]
        snapshots.append(snapshot)
        if len(snapshots) > _MAX_SNAPSHOTS_PER_USER:
            self._emotional_data[user_id] = snapshots[-_MAX_SNAPSHOTS_PER_USER:]
        return snapshot

    # ── Pattern detection ──────────────────────────────────────────────────

    def detect_inherited_patterns(self, user_id: str) -> List[InheritedPattern]:
        """Compare user's emotional patterns to reported family patterns.

        For each condition present in the family, compute:
          - family_prevalence: weighted fraction of family members with it.
          - user_match_score: how strongly the user's emotional data
            correlates with that condition's signature.
          - direction: amplified / attenuated / stable.

        Returns:
            List of InheritedPattern objects (one per family condition).
        """
        family = self._family_history.get(user_id, [])
        data = self._emotional_data.get(user_id, [])

        if not family:
            return []

        # Collect which conditions appear and their weighted prevalence
        condition_evidence: Dict[str, Dict] = {}
        for condition in CONDITION_TYPES:
            weighted_sum = 0.0
            total_weight = 0.0
            relations: List[str] = []
            for member in family:
                weight = _RELATION_WEIGHTS.get(member.relation, 0.3)
                total_weight += weight
                if condition in member.conditions:
                    weighted_sum += weight * (member.severity / 10.0)
                    relations.append(member.relation)
            if total_weight > 0 and weighted_sum > 0:
                prevalence = weighted_sum / total_weight
                condition_evidence[condition] = {
                    "prevalence": min(1.0, prevalence),
                    "relations": relations,
                }

        if not condition_evidence:
            return []

        # Compute user match scores from emotional data
        user_stress = _mean_stress(data) if data else 0.0
        user_valence = _mean_valence(data) if data else 0.0
        user_arousal = _mean_arousal(data) if data else 0.5

        patterns: List[InheritedPattern] = []
        for condition, evidence in condition_evidence.items():
            match_score = _compute_condition_match(
                condition, user_stress, user_valence, user_arousal,
            )
            prevalence = evidence["prevalence"]

            # Direction: compare user match to family prevalence
            delta = match_score - prevalence
            if delta > 0.10:
                direction = "amplified"
            elif delta < -0.10:
                direction = "attenuated"
            else:
                direction = "stable"

            confidence = min(1.0, len(data) / 20.0) * 0.5 + 0.5 * prevalence

            patterns.append(InheritedPattern(
                condition=condition,
                family_prevalence=round(prevalence, 4),
                user_match_score=round(match_score, 4),
                direction=direction,
                contributing_relations=evidence["relations"],
                confidence=round(min(1.0, confidence), 4),
            ))

        return patterns

    # ── Risk attenuation ───────────────────────────────────────────────────

    def compute_risk_attenuation(self, user_id: str) -> Dict:
        """Score which inherited patterns are getting stronger vs weaker.

        Returns:
            Dict with risk_score (overall inherited risk 0-1),
            attenuation_score (0-1 how much risk is being reduced),
            per-condition breakdown, and summary.
        """
        patterns = self.detect_inherited_patterns(user_id)

        if not patterns:
            return {
                "risk_score": 0.0,
                "attenuation_score": 0.0,
                "amplified": [],
                "attenuated": [],
                "stable": [],
                "summary": "No family history data available.",
            }

        amplified = [p for p in patterns if p.direction == "amplified"]
        attenuated = [p for p in patterns if p.direction == "attenuated"]
        stable = [p for p in patterns if p.direction == "stable"]

        # Risk score: weighted average of family prevalence * user match
        total_risk = 0.0
        total_weight = 0.0
        for p in patterns:
            w = p.family_prevalence
            total_risk += w * p.user_match_score
            total_weight += w

        risk_score = total_risk / total_weight if total_weight > 0 else 0.0
        risk_score = max(0.0, min(1.0, risk_score))

        # Attenuation: fraction of patterns that are attenuated, weighted
        attenuation_numerator = sum(p.family_prevalence for p in attenuated)
        attenuation_score = (
            attenuation_numerator / total_weight if total_weight > 0 else 0.0
        )
        attenuation_score = max(0.0, min(1.0, attenuation_score))

        return {
            "risk_score": round(risk_score, 4),
            "attenuation_score": round(attenuation_score, 4),
            "amplified": [p.condition for p in amplified],
            "attenuated": [p.condition for p in attenuated],
            "stable": [p.condition for p in stable],
            "summary": _risk_summary(risk_score, attenuation_score, amplified, attenuated),
        }

    # ── Protective factors ─────────────────────────────────────────────────

    def detect_protective_factors(self, user_id: str) -> Dict:
        """Detect where the user has broken family patterns.

        A protective factor exists when:
          - A condition is prevalent in the family (prevalence >= 0.3)
          - But the user's match score is low (< 0.3)

        Also looks for positive-only patterns: resilience in user data
        that exceeds family resilience.

        Returns:
            Dict with protective_factors list, broken_patterns,
            resilience_indicators, and overall protective_score.
        """
        patterns = self.detect_inherited_patterns(user_id)
        data = self._emotional_data.get(user_id, [])
        family = self._family_history.get(user_id, [])

        broken: List[str] = []
        for p in patterns:
            if p.family_prevalence >= 0.3 and p.user_match_score < 0.3:
                broken.append(p.condition)

        # Resilience indicators from emotional data
        resilience_indicators: List[str] = []
        if len(data) >= 4:
            mid = len(data) // 2
            early = data[:mid]
            recent = data[mid:]

            early_stress = _mean_stress(early)
            recent_stress = _mean_stress(recent)
            if recent_stress < early_stress - 0.05:
                resilience_indicators.append("stress_reduction_over_time")

            early_valence = _mean_valence(early)
            recent_valence = _mean_valence(recent)
            if recent_valence > early_valence + 0.05:
                resilience_indicators.append("valence_improvement_over_time")

            recent_arousal_var = _variance([s.arousal for s in recent])
            early_arousal_var = _variance([s.arousal for s in early])
            if recent_arousal_var < early_arousal_var - 0.01:
                resilience_indicators.append("emotional_stability_improvement")

        # Family resilience presence
        family_has_resilience = any(
            "resilience" in m.conditions for m in family
        )
        if family_has_resilience:
            resilience_indicators.append("family_resilience_heritage")

        protective_factors = broken + resilience_indicators

        protective_score = 0.0
        if patterns:
            broken_weight = len(broken) / max(1, len(patterns))
            resilience_weight = len(resilience_indicators) / 4.0
            protective_score = 0.6 * broken_weight + 0.4 * min(1.0, resilience_weight)
            protective_score = max(0.0, min(1.0, protective_score))

        return {
            "protective_factors": protective_factors,
            "broken_patterns": broken,
            "resilience_indicators": resilience_indicators,
            "protective_score": round(protective_score, 4),
            "total_family_conditions": len(patterns),
        }

    # ── Generational healing tracking ──────────────────────────────────────

    def track_generational_healing(self, user_id: str) -> Dict:
        """Is the user's emotional profile healthier than their parents?

        Computes a healing score by comparing user's emotional metrics
        against an estimated parental baseline derived from family history.

        Returns:
            Dict with healing_score (0-1), parental_burden,
            user_burden, improvement delta, and interpretation.
        """
        family = self._family_history.get(user_id, [])
        data = self._emotional_data.get(user_id, [])

        if not family:
            return {
                "healing_score": 0.0,
                "parental_burden": None,
                "user_burden": None,
                "delta": None,
                "interpretation": "No family history available.",
            }

        # Parental burden: weighted severity of conditions in parents
        parents = [m for m in family if m.relation in ("mother", "father")]
        if not parents:
            # Fall back to all family members
            parents = family

        parental_burden = sum(
            m.severity / 10.0 * len(m.conditions) / max(1, len(CONDITION_TYPES))
            for m in parents
        ) / len(parents)
        parental_burden = max(0.0, min(1.0, parental_burden))

        # User burden: derived from emotional data
        if data:
            user_stress = _mean_stress(data)
            user_valence = _mean_valence(data)
            # Higher stress, lower valence = higher burden
            user_burden = 0.5 * user_stress + 0.5 * max(0.0, -user_valence + 0.5)
            user_burden = max(0.0, min(1.0, user_burden))
        else:
            user_burden = 0.5  # unknown = neutral

        delta = parental_burden - user_burden  # positive = healing
        healing_score = max(0.0, min(1.0, 0.5 + delta))

        if delta > 0.15:
            interpretation = (
                "Significant generational healing detected. The user's emotional "
                "profile is meaningfully healthier than the parental generation."
            )
        elif delta > 0.0:
            interpretation = (
                "Modest generational healing. The user shows slight improvement "
                "over the parental generation's emotional burden."
            )
        elif delta > -0.15:
            interpretation = (
                "Generational emotional burden is approximately stable. "
                "No significant change from parental patterns."
            )
        else:
            interpretation = (
                "User may be carrying increased emotional burden relative to "
                "the parental generation. Consider targeted support."
            )

        return {
            "healing_score": round(healing_score, 4),
            "parental_burden": round(parental_burden, 4),
            "user_burden": round(user_burden, 4),
            "delta": round(delta, 4),
            "interpretation": interpretation,
        }

    # ── Full profile ───────────────────────────────────────────────────────

    def compute_epigenetic_profile(self, user_id: str) -> Dict:
        """Compute a comprehensive epigenetic profile combining all metrics.

        Aggregates: inherited patterns, risk/attenuation, protective factors,
        and generational healing into a single profile dict.

        Returns:
            Dict with all sub-assessments and overall summary.
        """
        patterns = self.detect_inherited_patterns(user_id)
        risk = self.compute_risk_attenuation(user_id)
        protective = self.detect_protective_factors(user_id)
        healing = self.track_generational_healing(user_id)

        family = self._family_history.get(user_id, [])
        data = self._emotional_data.get(user_id, [])

        from datetime import datetime, timezone
        assessment_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

        return {
            "user_id": user_id,
            "inherited_patterns": [
                {
                    "condition": p.condition,
                    "family_prevalence": p.family_prevalence,
                    "user_match_score": p.user_match_score,
                    "direction": p.direction,
                    "contributing_relations": p.contributing_relations,
                    "confidence": p.confidence,
                }
                for p in patterns
            ],
            "risk_attenuation": risk,
            "protective_factors": protective,
            "generational_healing": healing,
            "family_members_count": len(family),
            "emotional_datapoints": len(data),
            "assessment_date": assessment_date,
        }

    # ── Serialization ──────────────────────────────────────────────────────

    def profile_to_dict(self, user_id: str) -> Dict:
        """Serialize the full user state to a dict for API responses.

        Returns user's family history summary, emotional snapshot count,
        and condition definitions.
        """
        family = self._family_history.get(user_id, [])
        data = self._emotional_data.get(user_id, [])

        return {
            "user_id": user_id,
            "family_members": [
                {
                    "relation": m.relation,
                    "conditions": m.conditions,
                    "severity": m.severity,
                    "notes": m.notes,
                }
                for m in family
            ],
            "emotional_snapshots": len(data),
            "condition_types": CONDITION_TYPES,
            "relation_types": RELATION_TYPES,
        }


# ── Helper functions ───────────────────────────────────────────────────────

def _mean_stress(snapshots: List[EmotionalSnapshot]) -> float:
    """Average stress level across snapshots."""
    if not snapshots:
        return 0.0
    return sum(s.stress_level for s in snapshots) / len(snapshots)


def _mean_valence(snapshots: List[EmotionalSnapshot]) -> float:
    """Average valence across snapshots."""
    if not snapshots:
        return 0.0
    return sum(s.valence for s in snapshots) / len(snapshots)


def _mean_arousal(snapshots: List[EmotionalSnapshot]) -> float:
    """Average arousal across snapshots."""
    if not snapshots:
        return 0.5
    return sum(s.arousal for s in snapshots) / len(snapshots)


def _variance(values: List[float]) -> float:
    """Population variance. Returns 0.0 for fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _compute_condition_match(
    condition: str,
    user_stress: float,
    user_valence: float,
    user_arousal: float,
) -> float:
    """Estimate how strongly the user's emotional profile matches a condition.

    Uses heuristic mapping from emotional metrics to condition signatures.
    Returns 0-1 match score.
    """
    # Condition emotional signatures (approximate)
    signatures: Dict[str, Dict[str, float]] = {
        "anxiety": {"stress_weight": 0.5, "arousal_weight": 0.3, "neg_valence_weight": 0.2},
        "depression": {"stress_weight": 0.2, "arousal_weight": -0.3, "neg_valence_weight": 0.5},
        "trauma": {"stress_weight": 0.5, "arousal_weight": 0.2, "neg_valence_weight": 0.3},
        "addiction": {"stress_weight": 0.3, "arousal_weight": 0.4, "neg_valence_weight": 0.3},
        "resilience": {"stress_weight": -0.4, "arousal_weight": 0.1, "neg_valence_weight": -0.5},
        "anger": {"stress_weight": 0.3, "arousal_weight": 0.5, "neg_valence_weight": 0.2},
        "grief": {"stress_weight": 0.2, "arousal_weight": -0.2, "neg_valence_weight": 0.6},
        "phobia": {"stress_weight": 0.4, "arousal_weight": 0.4, "neg_valence_weight": 0.2},
    }

    sig = signatures.get(condition, {"stress_weight": 0.33, "arousal_weight": 0.33, "neg_valence_weight": 0.34})
    neg_valence = max(0.0, -user_valence + 0.5) / 1.5  # normalize to 0-1ish

    score = (
        sig["stress_weight"] * user_stress
        + sig["arousal_weight"] * user_arousal
        + sig["neg_valence_weight"] * neg_valence
    )
    # Resilience has negative weights, can go negative
    return max(0.0, min(1.0, score))


def _risk_summary(
    risk_score: float,
    attenuation_score: float,
    amplified: List[InheritedPattern],
    attenuated: List[InheritedPattern],
) -> str:
    """Generate human-readable risk summary."""
    parts: List[str] = []

    if risk_score >= 0.6:
        parts.append("Elevated inherited emotional risk detected.")
    elif risk_score >= 0.3:
        parts.append("Moderate inherited emotional risk detected.")
    else:
        parts.append("Low inherited emotional risk.")

    if amplified:
        names = ", ".join(p.condition for p in amplified)
        parts.append(f"Amplified patterns: {names}.")

    if attenuated:
        names = ", ".join(p.condition for p in attenuated)
        parts.append(f"Attenuated (improving) patterns: {names}.")

    if attenuation_score > 0.4:
        parts.append("Significant pattern breaking is occurring.")

    return " ".join(parts)


# ── Module-level singleton ─────────────────────────────────────────────────

_engine: Optional[EpigeneticEngine] = None


def get_epigenetic_engine() -> EpigeneticEngine:
    global _engine
    if _engine is None:
        _engine = EpigeneticEngine()
    return _engine
