"""
HRV Biofeedback — resonance frequency breathing guide + coherence scoring.

Evidence base: d=0.81-0.83 for anxiety/stress (24 studies, 484 participants).
Resonance frequency: ~6 breaths/min (range 4.5-6.5 bpm), 10s per breath cycle.

This module provides:
  - RespirationPattern: dataclass describing a breathing pattern
  - score_session: compute coherence score for a completed session
  - prescribe_pattern: select optimal pattern from voice-derived stress level
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BreathPhase:
    label: str
    duration_s: float
    direction: str  # "in" | "hold" | "out"


@dataclass
class RespirationPattern:
    id: str
    name: str
    tagline: str
    science: str
    phases: List[BreathPhase]
    cycle_duration_s: float = field(init=False)
    breaths_per_min: float = field(init=False)
    evidence_grade: str = "A"

    def __post_init__(self) -> None:
        self.cycle_duration_s = sum(p.duration_s for p in self.phases)
        self.breaths_per_min = 60.0 / self.cycle_duration_s


# ── Evidence-based patterns ──────────────────────────────────────────────────

PATTERNS: Dict[str, RespirationPattern] = {
    "resonance": RespirationPattern(
        id="resonance",
        name="Resonance Breathing",
        tagline="6 breaths/min — strongest HRV evidence (d=0.81)",
        science=(
            "Resonance frequency breathing (~0.1 Hz / 6 BPM) maximally amplifies "
            "heart rate oscillations via baroreflex coupling. 24 RCTs, 484 participants: "
            "d=0.81 for anxiety reduction, d=0.83 for stress. Gold standard for HRV training."
        ),
        phases=[
            BreathPhase("Inhale", 5.0, "in"),
            BreathPhase("Exhale", 5.0, "out"),
        ],
        evidence_grade="A",
    ),
    "box": RespirationPattern(
        id="box",
        name="Box Breathing",
        tagline="Military-derived, 4-4-4-4 pattern",
        science=(
            "Equal-ratio breathing with breath-holds synchronises autonomic nervous "
            "system. Used by US Navy SEALs for acute stress regulation. Activates "
            "prefrontal cortex inhibition of amygdala response."
        ),
        phases=[
            BreathPhase("Inhale", 4.0, "in"),
            BreathPhase("Hold in", 4.0, "hold"),
            BreathPhase("Exhale", 4.0, "out"),
            BreathPhase("Hold out", 4.0, "hold"),
        ],
        evidence_grade="B",
    ),
    "478": RespirationPattern(
        id="478",
        name="4-7-8 Breathing",
        tagline="Extended exhale activates parasympathetic",
        science=(
            "Long exhale (8s) relative to inhale (4s) drives vagal tone via "
            "pulmonary stretch receptors. Dr Andrew Weil derived from pranayama. "
            "Particularly effective for pre-sleep anxiety reduction."
        ),
        phases=[
            BreathPhase("Inhale", 4.0, "in"),
            BreathPhase("Hold", 7.0, "hold"),
            BreathPhase("Exhale", 8.0, "out"),
        ],
        evidence_grade="B",
    ),
    "cyclic_sigh": RespirationPattern(
        id="cyclic_sigh",
        name="Cyclic Sighing",
        tagline="Stanford 2023: greatest acute positive affect improvement",
        science=(
            "Double inhale (nasal, then top-up) followed by long exhale. Stanford 2023 "
            "RCT: superior to mindfulness and other breathwork patterns for real-time "
            "stress reduction and next-day positive affect (p<0.001, n=108)."
        ),
        phases=[
            BreathPhase("Inhale (nose)", 1.5, "in"),
            BreathPhase("Top-up inhale", 0.5, "in"),
            BreathPhase("Long exhale (mouth)", 5.0, "out"),
        ],
        evidence_grade="A",
    ),
}


# ── Coherence scoring ────────────────────────────────────────────────────────

def score_session(
    session_duration_s: float,
    pattern_id: str,
    stress_before: Optional[float],
    stress_after: Optional[float],
    completed_cycles: int,
) -> Dict[str, float]:
    """
    Compute a coherence quality score (0-100) and improvement metrics
    for a completed breathing session.

    Parameters
    ----------
    session_duration_s : float
        Actual session duration in seconds.
    pattern_id : str
        The breathing pattern used.
    stress_before : float | None
        Stress index before session (0-1 scale), or None if not measured.
    stress_after : float | None
        Stress index after session (0-1 scale), or None if not measured.
    completed_cycles : int
        Number of full breathing cycles completed.

    Returns
    -------
    dict with keys: coherence_score, stress_delta, effectiveness, grade
    """
    pattern = PATTERNS.get(pattern_id)
    ideal_cycles = session_duration_s / (pattern.cycle_duration_s if pattern else 10.0)
    cycle_adherence = min(1.0, completed_cycles / max(1, ideal_cycles))

    # Duration bonus: 4 min minimum for physiological effect, 20 min optimal
    duration_factor = min(1.0, (session_duration_s - 60) / (20 * 60 - 60))
    duration_factor = max(0.0, duration_factor)

    # Evidence weight: resonance and cyclic_sigh have strongest evidence
    evidence_bonus = 1.15 if pattern_id in ("resonance", "cyclic_sigh") else 1.0

    coherence = min(100, round(cycle_adherence * 70 + duration_factor * 30) * evidence_bonus)

    stress_delta = 0.0
    effectiveness = "unknown"
    if stress_before is not None and stress_after is not None:
        stress_delta = stress_before - stress_after  # positive = improvement
        if stress_delta > 0.15:
            effectiveness = "strong"
        elif stress_delta > 0.05:
            effectiveness = "moderate"
        elif stress_delta >= 0:
            effectiveness = "mild"
        else:
            effectiveness = "none"

    grade = "A" if coherence >= 80 else "B" if coherence >= 60 else "C"

    return {
        "coherence_score": round(coherence, 1),
        "stress_delta": round(stress_delta, 3),
        "effectiveness": effectiveness,
        "grade": grade,
        "completed_cycles": completed_cycles,
        "session_duration_s": round(session_duration_s, 1),
    }


def prescribe_pattern(stress_index: float, time_of_day_hour: int = 12) -> str:
    """
    Prescribe the optimal breathing pattern based on current stress level
    and time of day.

    Parameters
    ----------
    stress_index : float
        Current stress 0-1.
    time_of_day_hour : int
        Hour 0-23.

    Returns
    -------
    str : pattern_id
    """
    # Pre-sleep (after 20:00): 4-7-8 or resonance
    if time_of_day_hour >= 20:
        return "478" if stress_index > 0.5 else "resonance"
    # High acute stress: cyclic sighing (Stanford: fastest effect)
    if stress_index > 0.65:
        return "cyclic_sigh"
    # Moderate stress: resonance (strongest long-term evidence)
    if stress_index > 0.35:
        return "resonance"
    # Low stress / focus: box breathing
    return "box"
