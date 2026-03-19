"""Post-traumatic growth (PTG) tracker.

Based on: Tedeschi & Calhoun (1996, 2004) — Post-Traumatic Growth Inventory (PTGI).

Core idea: Adversity can catalyze positive psychological transformation across
five domains. PTG is distinct from resilience: resilience = return to baseline,
growth = exceeding pre-adversity baseline functioning.

Five PTG domains (Tedeschi & Calhoun):
  1. Relating to Others — deeper, more meaningful relationships
  2. New Possibilities — new paths, interests, or opportunities
  3. Personal Strength — greater sense of inner strength
  4. Spiritual Change — deepened existential or spiritual understanding
  5. Appreciation of Life — enhanced gratitude and presence

Growth indicators (detectable from longitudinal emotional data):
  - Emotional range expansion post-trauma (wider valence spread)
  - Increased positive affect baseline (higher mean valence)
  - Improved emotional regulation (faster stress recovery)
  - Increased social engagement (more interaction sessions)

Usage:
    tracker = PTGTracker()
    score = tracker.compute_ptg_score(domain_ratings, emotional_history)
    indicators = tracker.detect_growth_indicators(pre_data, post_data)
    trajectory = tracker.track_growth_trajectory(user_id)
    distinction = tracker.distinguish_resilience_vs_growth(pre_data, post_data)
    profile = tracker.compute_growth_profile(user_id)
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── PTG domain definitions ──────────────────────────────────────────────────

PTG_DOMAINS: Dict[str, Dict] = {
    "relating_to_others": {
        "label": "Relating to Others",
        "description": "Deeper, more meaningful relationships after adversity",
        "ptgi_items": 7,  # Items 6,8,9,15,16,20,21 in PTGI
        "max_score": 35,
    },
    "new_possibilities": {
        "label": "New Possibilities",
        "description": "New paths, interests, or opportunities discovered",
        "ptgi_items": 5,  # Items 3,7,11,14,17
        "max_score": 25,
    },
    "personal_strength": {
        "label": "Personal Strength",
        "description": "Greater sense of inner strength and self-reliance",
        "ptgi_items": 4,  # Items 4,10,12,19
        "max_score": 20,
    },
    "spiritual_change": {
        "label": "Spiritual Change",
        "description": "Deepened existential or spiritual understanding",
        "ptgi_items": 2,  # Items 5,18
        "max_score": 10,
    },
    "appreciation_of_life": {
        "label": "Appreciation of Life",
        "description": "Enhanced gratitude, presence, and life priorities",
        "ptgi_items": 3,  # Items 1,2,13
        "max_score": 15,
    },
}

ADVERSITY_TYPES = ["loss", "illness", "trauma", "relationship", "career"]

_MAX_ASSESSMENTS_PER_USER = 365


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class AdversityEvent:
    """A recorded adversity event."""
    date: str                       # YYYY-MM-DD
    severity: float                 # 0-10 scale
    adversity_type: str             # one of ADVERSITY_TYPES
    description: str = ""
    timestamp: float = 0.0


@dataclass
class EmotionalSnapshot:
    """A single point-in-time emotional reading."""
    date: str                       # YYYY-MM-DD
    valence: float                  # -1 to 1
    arousal: float                  # 0 to 1
    stress_level: float             # 0 to 1
    social_engagement: float        # 0 to 1 (interaction frequency)
    timestamp: float = 0.0


@dataclass
class PTGAssessment:
    """A full PTG assessment result."""
    date: str
    domain_scores: Dict[str, float]     # domain -> 0-1 normalized
    ptg_total: float                    # 0-1 overall
    growth_indicators: Dict[str, bool]
    classification: str                 # "growth", "resilience", "struggling", "declining"
    confidence: float                   # 0-1
    timestamp: float = 0.0


# ── Core tracker ────────────────────────────────────────────────────────────

class PTGTracker:
    """Track post-traumatic growth across the five Tedeschi & Calhoun domains.

    Stateful: maintains per-user adversity events, emotional history,
    self-report domain ratings, and assessment history.
    """

    def __init__(self) -> None:
        self._adversity_events: Dict[str, List[AdversityEvent]] = defaultdict(list)
        self._emotional_history: Dict[str, List[EmotionalSnapshot]] = defaultdict(list)
        self._domain_ratings: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._assessments: Dict[str, List[PTGAssessment]] = defaultdict(list)

    # ── Adversity management ────────────────────────────────────────────────

    def record_adversity(
        self,
        user_id: str,
        date: str,
        severity: float,
        adversity_type: str,
        description: str = "",
    ) -> AdversityEvent:
        """Record an adversity event for a user.

        Args:
            user_id: User identifier.
            date: Date string (YYYY-MM-DD).
            severity: 0-10 scale of adversity severity.
            adversity_type: One of ADVERSITY_TYPES.
            description: Optional free-text description.

        Returns:
            The recorded AdversityEvent.
        """
        severity = max(0.0, min(10.0, float(severity)))
        if adversity_type not in ADVERSITY_TYPES:
            adversity_type = "trauma"  # default fallback

        event = AdversityEvent(
            date=date,
            severity=severity,
            adversity_type=adversity_type,
            description=description,
            timestamp=time.time(),
        )
        self._adversity_events[user_id].append(event)
        return event

    def add_emotional_snapshot(
        self,
        user_id: str,
        date: str,
        valence: float,
        arousal: float,
        stress_level: float,
        social_engagement: float,
    ) -> EmotionalSnapshot:
        """Add a single emotional data point for longitudinal tracking."""
        snapshot = EmotionalSnapshot(
            date=date,
            valence=max(-1.0, min(1.0, float(valence))),
            arousal=max(0.0, min(1.0, float(arousal))),
            stress_level=max(0.0, min(1.0, float(stress_level))),
            social_engagement=max(0.0, min(1.0, float(social_engagement))),
            timestamp=time.time(),
        )
        self._emotional_history[user_id].append(snapshot)
        return snapshot

    def set_domain_ratings(
        self, user_id: str, ratings: Dict[str, float]
    ) -> Dict[str, float]:
        """Set self-report PTG domain ratings (0-5 Likert scale per domain).

        Ratings are normalized to 0-1 internally.
        """
        normalized: Dict[str, float] = {}
        for domain in PTG_DOMAINS:
            raw = ratings.get(domain, 0.0)
            normalized[domain] = max(0.0, min(1.0, float(raw) / 5.0))
        self._domain_ratings[user_id] = normalized
        return normalized

    # ── Core computation functions ──────────────────────────────────────────

    def compute_ptg_score(
        self,
        user_id: str,
        domain_ratings: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """Compute overall PTG score combining self-report domains and emotional data.

        If domain_ratings is provided, updates stored ratings first.
        Blends self-report (60%) with emotional indicators (40%) when both available.

        Returns:
            Dict with ptg_total (0-1), domain_scores, component weights,
            and interpretation label.
        """
        if domain_ratings is not None:
            self.set_domain_ratings(user_id, domain_ratings)

        stored_ratings = self._domain_ratings.get(user_id, {})
        emotional_data = self._emotional_history.get(user_id, [])

        # Self-report component (normalized domain scores)
        if stored_ratings:
            self_report_score = sum(stored_ratings.values()) / len(stored_ratings)
            domain_scores = dict(stored_ratings)
        else:
            self_report_score = 0.0
            domain_scores = {d: 0.0 for d in PTG_DOMAINS}

        # Emotional data component
        emotional_score = 0.0
        if len(emotional_data) >= 2:
            # Use recent data vs earliest data as proxy for growth
            n = len(emotional_data)
            early = emotional_data[: max(1, n // 3)]
            recent = emotional_data[-max(1, n // 3):]

            early_valence = sum(s.valence for s in early) / len(early)
            recent_valence = sum(s.valence for s in recent) / len(recent)
            valence_improvement = max(0.0, (recent_valence - early_valence + 1.0) / 2.0)

            early_stress = sum(s.stress_level for s in early) / len(early)
            recent_stress = sum(s.stress_level for s in recent) / len(recent)
            stress_reduction = max(0.0, (early_stress - recent_stress + 0.5) / 1.0)
            stress_reduction = min(1.0, stress_reduction)

            recent_engagement = sum(s.social_engagement for s in recent) / len(recent)

            emotional_score = (
                0.4 * valence_improvement
                + 0.3 * stress_reduction
                + 0.3 * recent_engagement
            )
            emotional_score = max(0.0, min(1.0, emotional_score))

        # Blend: 60% self-report, 40% emotional (when both available)
        if stored_ratings and len(emotional_data) >= 2:
            ptg_total = 0.6 * self_report_score + 0.4 * emotional_score
        elif stored_ratings:
            ptg_total = self_report_score
        elif len(emotional_data) >= 2:
            ptg_total = emotional_score
        else:
            ptg_total = 0.0

        ptg_total = max(0.0, min(1.0, ptg_total))

        interpretation = _interpret_ptg_score(ptg_total)

        return {
            "ptg_total": round(ptg_total, 4),
            "domain_scores": {k: round(v, 4) for k, v in domain_scores.items()},
            "self_report_score": round(self_report_score, 4),
            "emotional_score": round(emotional_score, 4),
            "interpretation": interpretation,
            "has_self_report": bool(stored_ratings),
            "emotional_datapoints": len(emotional_data),
        }

    def detect_growth_indicators(
        self,
        user_id: str,
    ) -> Dict:
        """Detect post-traumatic growth indicators from longitudinal emotional data.

        Checks for:
          1. Emotional range expansion (wider valence spread post-trauma)
          2. Increased positive affect baseline (higher mean valence)
          3. Improved emotional regulation (faster stress recovery)
          4. Increased social engagement

        Returns:
            Dict with indicator booleans, numeric values, and overall count.
        """
        data = self._emotional_history.get(user_id, [])
        adversity = self._adversity_events.get(user_id, [])

        if len(data) < 4:
            return {
                "indicators": {
                    "emotional_range_expansion": False,
                    "increased_positive_affect": False,
                    "improved_emotional_regulation": False,
                    "increased_social_engagement": False,
                },
                "details": {},
                "indicators_present": 0,
                "sufficient_data": False,
            }

        # Split into early/recent halves
        mid = len(data) // 2
        early = data[:mid]
        recent = data[mid:]

        # 1. Emotional range expansion: wider valence spread
        early_valences = [s.valence for s in early]
        recent_valences = [s.valence for s in recent]
        early_range = max(early_valences) - min(early_valences) if early_valences else 0.0
        recent_range = max(recent_valences) - min(recent_valences) if recent_valences else 0.0
        range_expansion = recent_range > early_range + 0.1

        # 2. Increased positive affect baseline
        early_mean_valence = sum(early_valences) / len(early_valences) if early_valences else 0.0
        recent_mean_valence = sum(recent_valences) / len(recent_valences) if recent_valences else 0.0
        positive_affect_increase = recent_mean_valence > early_mean_valence + 0.05

        # 3. Improved emotional regulation (lower stress variability in recent)
        early_stress = [s.stress_level for s in early]
        recent_stress = [s.stress_level for s in recent]
        early_stress_var = _variance(early_stress)
        recent_stress_var = _variance(recent_stress)
        early_mean_stress = sum(early_stress) / len(early_stress) if early_stress else 0.0
        recent_mean_stress = sum(recent_stress) / len(recent_stress) if recent_stress else 0.0
        regulation_improved = (
            recent_mean_stress < early_mean_stress - 0.03
            or recent_stress_var < early_stress_var - 0.005
        )

        # 4. Increased social engagement
        early_engagement = sum(s.social_engagement for s in early) / len(early) if early else 0.0
        recent_engagement = sum(s.social_engagement for s in recent) / len(recent) if recent else 0.0
        engagement_increased = recent_engagement > early_engagement + 0.05

        indicators = {
            "emotional_range_expansion": range_expansion,
            "increased_positive_affect": positive_affect_increase,
            "improved_emotional_regulation": regulation_improved,
            "increased_social_engagement": engagement_increased,
        }

        details = {
            "early_valence_range": round(early_range, 4),
            "recent_valence_range": round(recent_range, 4),
            "early_mean_valence": round(early_mean_valence, 4),
            "recent_mean_valence": round(recent_mean_valence, 4),
            "early_mean_stress": round(early_mean_stress, 4),
            "recent_mean_stress": round(recent_mean_stress, 4),
            "early_social_engagement": round(early_engagement, 4),
            "recent_social_engagement": round(recent_engagement, 4),
        }

        return {
            "indicators": indicators,
            "details": details,
            "indicators_present": sum(1 for v in indicators.values() if v),
            "sufficient_data": True,
        }

    def track_growth_trajectory(self, user_id: str) -> Dict:
        """Track PTG trajectory over time from stored assessments.

        Computes slope of PTG total scores to determine if user is showing
        growth, plateau, or decline.

        Returns:
            Dict with trajectory classification, slope, assessment count,
            and recent scores.
        """
        assessments = self._assessments.get(user_id, [])

        if len(assessments) < 2:
            return {
                "trajectory": "insufficient_data",
                "slope": 0.0,
                "assessments": len(assessments),
                "recent_scores": [a.ptg_total for a in assessments],
                "interpretation": "Need at least 2 assessments to compute trajectory.",
            }

        scores = [a.ptg_total for a in assessments]
        slope = _theil_sen_slope(scores)

        if slope > 0.02:
            trajectory = "growing"
            interpretation = "User is showing post-traumatic growth over time."
        elif slope < -0.02:
            trajectory = "declining"
            interpretation = "User shows declining trajectory. Consider intervention."
        else:
            trajectory = "plateau"
            interpretation = "Growth has plateaued. May need new challenges or support."

        return {
            "trajectory": trajectory,
            "slope": round(slope, 4),
            "assessments": len(assessments),
            "recent_scores": [round(s, 4) for s in scores[-5:]],
            "interpretation": interpretation,
        }

    def distinguish_resilience_vs_growth(self, user_id: str) -> Dict:
        """Distinguish between resilience (return to baseline) and growth (exceeding baseline).

        Resilience: post-adversity functioning returns to pre-adversity levels.
        Growth: post-adversity functioning exceeds pre-adversity levels.

        Requires adversity events and emotional data spanning before and after.

        Returns:
            Dict with classification ("growth", "resilience", "struggling"),
            numeric comparison, and explanation.
        """
        adversity = self._adversity_events.get(user_id, [])
        data = self._emotional_history.get(user_id, [])

        if not adversity or len(data) < 4:
            return {
                "classification": "insufficient_data",
                "pre_adversity_baseline": None,
                "post_adversity_level": None,
                "delta": None,
                "explanation": "Need adversity event and emotional data to distinguish.",
            }

        # Use earliest adversity event as reference point
        earliest_adversity = min(adversity, key=lambda a: a.date)
        adversity_date = earliest_adversity.date

        # Split emotional data around adversity date
        pre = [s for s in data if s.date < adversity_date]
        post = [s for s in data if s.date >= adversity_date]

        if not pre or not post:
            # If no clear split, use first/second half
            mid = len(data) // 2
            pre = data[:mid]
            post = data[mid:]

        pre_valence = sum(s.valence for s in pre) / len(pre)
        post_valence = sum(s.valence for s in post) / len(post)
        pre_engagement = sum(s.social_engagement for s in pre) / len(pre)
        post_engagement = sum(s.social_engagement for s in post) / len(post)

        # Composite functioning metric: 60% valence + 40% engagement
        pre_functioning = 0.6 * pre_valence + 0.4 * pre_engagement
        post_functioning = 0.6 * post_valence + 0.4 * post_engagement
        delta = post_functioning - pre_functioning

        if delta > 0.05:
            classification = "growth"
            explanation = (
                "Post-adversity functioning exceeds pre-adversity baseline. "
                "This indicates genuine post-traumatic growth."
            )
        elif delta >= -0.05:
            classification = "resilience"
            explanation = (
                "Post-adversity functioning has returned to pre-adversity levels. "
                "This indicates resilience — a return to baseline."
            )
        else:
            classification = "struggling"
            explanation = (
                "Post-adversity functioning remains below pre-adversity baseline. "
                "Additional support may be beneficial."
            )

        return {
            "classification": classification,
            "pre_adversity_baseline": round(pre_functioning, 4),
            "post_adversity_level": round(post_functioning, 4),
            "delta": round(delta, 4),
            "adversity_date": adversity_date,
            "adversity_type": earliest_adversity.adversity_type,
            "adversity_severity": earliest_adversity.severity,
            "explanation": explanation,
        }

    def compute_growth_profile(self, user_id: str) -> Dict:
        """Compute a comprehensive growth profile combining all PTG metrics.

        Aggregates: PTG score, growth indicators, trajectory, and
        resilience-vs-growth distinction into a single profile dict.

        Returns:
            Dict with all sub-assessments and overall summary.
        """
        ptg_score = self.compute_ptg_score(user_id)
        indicators = self.detect_growth_indicators(user_id)
        trajectory = self.track_growth_trajectory(user_id)
        distinction = self.distinguish_resilience_vs_growth(user_id)

        adversity_events = self._adversity_events.get(user_id, [])

        # Store assessment
        assessment = PTGAssessment(
            date=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
            domain_scores=ptg_score["domain_scores"],
            ptg_total=ptg_score["ptg_total"],
            growth_indicators={k: v for k, v in indicators["indicators"].items()},
            classification=distinction["classification"],
            confidence=min(1.0, indicators["indicators_present"] / 4.0 * 0.5 + 0.5)
            if indicators["sufficient_data"]
            else 0.2,
            timestamp=time.time(),
        )
        user_assessments = self._assessments[user_id]
        user_assessments.append(assessment)
        if len(user_assessments) > _MAX_ASSESSMENTS_PER_USER:
            self._assessments[user_id] = user_assessments[-_MAX_ASSESSMENTS_PER_USER:]

        return {
            "user_id": user_id,
            "ptg_score": ptg_score,
            "growth_indicators": indicators,
            "trajectory": trajectory,
            "resilience_vs_growth": distinction,
            "adversity_events": len(adversity_events),
            "assessment_date": assessment.date,
            "overall_classification": distinction["classification"],
        }

    def profile_to_dict(self, user_id: str) -> Dict:
        """Serialize the full user state to a dict for API responses.

        Returns user's adversity history, emotional snapshot count,
        domain ratings, and assessment count.
        """
        adversity = self._adversity_events.get(user_id, [])
        emotional = self._emotional_history.get(user_id, [])
        ratings = self._domain_ratings.get(user_id, {})
        assessments = self._assessments.get(user_id, [])

        return {
            "user_id": user_id,
            "adversity_events": [
                {
                    "date": e.date,
                    "severity": e.severity,
                    "type": e.adversity_type,
                    "description": e.description,
                }
                for e in adversity
            ],
            "emotional_snapshots": len(emotional),
            "domain_ratings": dict(ratings),
            "assessments_count": len(assessments),
            "domains": {
                k: {
                    "label": v["label"],
                    "description": v["description"],
                    "current_score": ratings.get(k),
                }
                for k, v in PTG_DOMAINS.items()
            },
        }


# ── Math helpers ────────────────────────────────────────────────────────────

def _variance(values: List[float]) -> float:
    """Compute population variance. Returns 0.0 for fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _theil_sen_slope(values: List[float]) -> float:
    """Compute Theil-Sen slope (median of all pairwise slopes).

    Pure Python -- no scipy dependency.
    Returns 0.0 for fewer than 2 values.
    """
    n = len(values)
    if n < 2:
        return 0.0

    slopes: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = j - i
            if dx != 0:
                slopes.append((values[j] - values[i]) / dx)

    if not slopes:
        return 0.0

    slopes.sort()
    mid = len(slopes) // 2
    if len(slopes) % 2 == 1:
        return slopes[mid]
    return (slopes[mid - 1] + slopes[mid]) / 2.0


def _interpret_ptg_score(score: float) -> str:
    """Map PTG total score to human-readable interpretation."""
    if score >= 0.75:
        return "significant_growth"
    elif score >= 0.50:
        return "moderate_growth"
    elif score >= 0.25:
        return "emerging_growth"
    else:
        return "minimal_growth"


# ── Module-level singleton ──────────────────────────────────────────────────

_tracker: Optional[PTGTracker] = None


def get_ptg_tracker() -> PTGTracker:
    global _tracker
    if _tracker is None:
        _tracker = PTGTracker()
    return _tracker
