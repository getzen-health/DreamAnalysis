"""CBT-based AI Emotion Coach — rule-based intervention ranking.

Selects evidence-based emotional regulation techniques based on the user's
current emotional state (valence/arousal/stress/HRV/focus) and ranks them
by the user's historical effectiveness data.

Techniques implemented:
- 4-7-8 breathing (high stress, high arousal)
- Box breathing (moderate stress, focus needed)
- 5-4-3-2-1 grounding (anxiety, fear)
- Cognitive reframe (negative valence, rumination)
- Body scan (moderate stress, body tension)
- Progressive muscle relaxation (high stress, prolonged)
- Mindful observation (mild stress, mind-wandering)
- Positive self-talk (low valence, low self-efficacy cue)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Technique catalogue ───────────────────────────────────────────────────────

_TECHNIQUES: List[Dict[str, Any]] = [
    {
        "name": "4-7-8 breathing",
        "technique": "diaphragmatic_breathing",
        "description": (
            "Inhale for 4 counts, hold for 7, exhale for 8. "
            "Activates the parasympathetic nervous system and rapidly reduces "
            "cortisol. Best for acute high-stress, high-arousal states."
        ),
        "duration_seconds": 300,
        "rationale": "High stress + high arousal → rapid parasympathetic activation needed.",
        "base_effectiveness": 0.82,
    },
    {
        "name": "box breathing",
        "technique": "diaphragmatic_breathing",
        "description": (
            "Inhale 4 counts, hold 4, exhale 4, hold 4. "
            "Used by Navy SEALs to maintain focus under pressure. "
            "Balances arousal while improving attention."
        ),
        "duration_seconds": 240,
        "rationale": "Moderate stress or focus deficit → balanced arousal regulation.",
        "base_effectiveness": 0.76,
    },
    {
        "name": "5-4-3-2-1 grounding",
        "technique": "grounding",
        "description": (
            "Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, "
            "1 you taste. Anchors attention to the present and interrupts "
            "anxiety or dissociative spirals."
        ),
        "duration_seconds": 180,
        "rationale": "Anxiety or fear → interrupt the spiral by grounding in sensory reality.",
        "base_effectiveness": 0.74,
    },
    {
        "name": "cognitive reframe",
        "technique": "cognitive_restructuring",
        "description": (
            "Identify the automatic negative thought, evaluate evidence for "
            "and against it, then generate a more balanced alternative. "
            "Core CBT technique for rumination and negative valence states."
        ),
        "duration_seconds": 420,
        "rationale": "Negative valence + elevated arousal → challenge distorted thinking.",
        "base_effectiveness": 0.78,
    },
    {
        "name": "body scan",
        "technique": "somatic_awareness",
        "description": (
            "Systematically bring attention from toes to crown, noticing "
            "tension without judgment. Reduces somatic stress responses "
            "and improves interoceptive awareness."
        ),
        "duration_seconds": 600,
        "rationale": "Moderate stress + somatic tension → release held physical stress.",
        "base_effectiveness": 0.71,
    },
    {
        "name": "progressive muscle relaxation",
        "technique": "somatic_awareness",
        "description": (
            "Tense each muscle group for 5 seconds, then release for 30. "
            "Works top-down or bottom-up. Most effective for chronic or "
            "prolonged high stress where breathing alone is insufficient."
        ),
        "duration_seconds": 720,
        "rationale": "Sustained high stress → deep somatic release.",
        "base_effectiveness": 0.79,
    },
    {
        "name": "mindful observation",
        "technique": "mindfulness",
        "description": (
            "Choose one object and observe it with full curiosity for several "
            "minutes — colour, texture, weight, temperature. Trains "
            "sustained attention and interrupts mind-wandering."
        ),
        "duration_seconds": 300,
        "rationale": "Mild stress or mind-wandering → gently redirect attention.",
        "base_effectiveness": 0.67,
    },
    {
        "name": "positive self-talk",
        "technique": "cognitive_restructuring",
        "description": (
            "Identify one specific strength you demonstrated recently. "
            "Repeat three 'I can' or 'I have' statements aloud. "
            "Counters learned helplessness and low self-efficacy signals."
        ),
        "duration_seconds": 180,
        "rationale": "Low valence + low arousal → rebuild motivational self-concept.",
        "base_effectiveness": 0.69,
    },
]

# Index for O(1) lookup by name
_TECHNIQUE_BY_NAME: Dict[str, Dict[str, Any]] = {t["name"]: t for t in _TECHNIQUES}

# ── Persistence helpers ───────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent / "data" / "emotion_coach"


def _user_path(user_id: str) -> Path:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR / f"{user_id}.json"


def _load_user(user_id: str) -> Dict[str, Any]:
    p = _user_path(user_id)
    if p.exists():
        try:
            with open(p, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"user_id": user_id, "sessions": []}


def _save_user(user_id: str, data: Dict[str, Any]) -> None:
    p = _user_path(user_id)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


# ── Effectiveness computation ─────────────────────────────────────────────────

def _historical_effectiveness(technique_name: str, sessions: List[Dict]) -> Optional[float]:
    """Average stress delta (stress_after - stress_before) for a technique.

    Returns None if no history for this technique (fewer than 2 sessions).
    A more negative delta = better — technique is normalised so higher = better.
    """
    relevant = [
        s for s in sessions
        if s.get("intervention") == technique_name
        and s.get("stress_before") is not None
        and s.get("stress_after") is not None
    ]
    if len(relevant) < 2:
        return None
    deltas = [s["stress_after"] - s["stress_before"] for s in relevant]
    avg_delta = sum(deltas) / len(deltas)
    # stress_delta is negative when technique reduces stress.
    # Convert to 0-1 effectiveness score: best possible = stress drops by 1.0.
    effectiveness = max(0.0, min(1.0, 0.5 - avg_delta))
    return round(effectiveness, 4)


# ── EmotionCoach ─────────────────────────────────────────────────────────────

class EmotionCoach:
    """Rule-based CBT intervention selector.

    No ML model files required — pure logic + historical personalisation.
    """

    def get_interventions(
        self,
        state: Dict[str, Any],
        user_history: List[Dict],
    ) -> List[Dict[str, Any]]:
        """Return top-3 ranked interventions for the current emotional state.

        Parameters
        ----------
        state:
            Keys: valence (-1 to 1), arousal (0-1), stress_index (0-1),
            focus_index (0-1). Optional: hrv_ms (float).
        user_history:
            List of session dicts from the last 30 sessions, each with
            keys: intervention, stress_before, stress_after, duration_seconds.

        Returns
        -------
        List of up to 3 dicts with keys:
            name, technique, description, duration_seconds, rationale,
            estimated_effectiveness.
        """
        valence = float(state.get("valence", 0.0))
        arousal = float(state.get("arousal", 0.5))
        stress = float(state.get("stress_index", 0.3))
        focus = float(state.get("focus_index", 0.5))

        # ── Rule-based primary recommendation ────────────────────────────────
        # Priority order: specific high-activation states first, catch-all last.
        if stress > 0.7 and arousal > 0.7:
            primary = "4-7-8 breathing"
        elif valence < -0.3 and arousal < 0.4:
            primary = "positive self-talk"
        elif valence < -0.2 and arousal > 0.5:
            primary = "cognitive reframe"
        elif stress > 0.5 and focus < 0.4:
            primary = "5-4-3-2-1 grounding"
        elif stress > 0.6:
            primary = "progressive muscle relaxation"
        else:
            primary = "box breathing"

        # ── Score every technique ─────────────────────────────────────────────
        scored: List[Dict[str, Any]] = []
        for t in _TECHNIQUES:
            hist_eff = _historical_effectiveness(t["name"], user_history)
            if hist_eff is not None:
                estimated = round(0.4 * t["base_effectiveness"] + 0.6 * hist_eff, 4)
            else:
                estimated = t["base_effectiveness"]

            # Promote primary rule-based recommendation
            if t["name"] == primary:
                estimated = min(1.0, estimated + 0.15)

            entry = dict(t)
            entry["estimated_effectiveness"] = estimated
            scored.append(entry)

        scored.sort(key=lambda x: x["estimated_effectiveness"], reverse=True)
        return scored[:3]

    def log_session(
        self,
        user_id: str,
        intervention: str,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Persist a completed coaching session.

        Parameters
        ----------
        user_id:    Unique user identifier.
        intervention: Name of the technique used.
        state_before: Emotional state before the session.
        state_after:  Emotional state after the session.

        Returns
        -------
        dict with keys: logged, sessions_count, effectiveness_delta.
        """
        data = _load_user(user_id)
        stress_before = float(state_before.get("stress_index", 0.0))
        stress_after = float(state_after.get("stress_index", 0.0))
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "intervention": intervention,
            "stress_before": round(stress_before, 4),
            "stress_after": round(stress_after, 4),
            "valence_before": round(float(state_before.get("valence", 0.0)), 4),
            "valence_after": round(float(state_after.get("valence", 0.0)), 4),
            "arousal_before": round(float(state_before.get("arousal", 0.5)), 4),
            "arousal_after": round(float(state_after.get("arousal", 0.5)), 4),
        }
        data["sessions"].append(record)
        _save_user(user_id, data)
        effectiveness_delta = round(stress_after - stress_before, 4)
        return {
            "logged": True,
            "sessions_count": len(data["sessions"]),
            "effectiveness_delta": effectiveness_delta,
        }

    def get_history(self, user_id: str) -> Dict[str, Any]:
        """Load user history and summarise top interventions.

        Returns
        -------
        dict: user_id, sessions (list), top_interventions (list of {name, avg_delta, count}).
        """
        data = _load_user(user_id)
        sessions = data.get("sessions", [])

        # Aggregate per technique
        from collections import defaultdict
        counts: Dict[str, int] = defaultdict(int)
        delta_sums: Dict[str, float] = defaultdict(float)
        for s in sessions:
            name = s.get("intervention", "unknown")
            counts[name] += 1
            if s.get("stress_before") is not None and s.get("stress_after") is not None:
                delta_sums[name] += s["stress_after"] - s["stress_before"]

        top = []
        for name, count in counts.items():
            avg_delta = round(delta_sums[name] / count, 4) if count else 0.0
            top.append({"name": name, "avg_stress_delta": avg_delta, "count": count})
        top.sort(key=lambda x: x["avg_stress_delta"])  # most negative delta first (best)

        return {
            "user_id": user_id,
            "sessions": sessions,
            "top_interventions": top,
        }

    def get_insight(self, user_id: str) -> str:
        """Generate a human-readable insight from the user's coaching history.

        Returns a plain-text sentence describing the most effective technique,
        or a generic onboarding message if insufficient data exists.
        """
        data = _load_user(user_id)
        sessions = data.get("sessions", [])

        if len(sessions) < 3:
            return (
                "Complete at least 3 coaching sessions to receive personalised insights. "
                "Start with box breathing — it works well as a general-purpose technique."
            )

        # Find best technique (most negative average stress delta, >= 2 uses)
        from collections import defaultdict
        counts: Dict[str, int] = defaultdict(int)
        delta_sums: Dict[str, float] = defaultdict(float)
        for s in sessions:
            name = s.get("intervention", "unknown")
            counts[name] += 1
            if s.get("stress_before") is not None and s.get("stress_after") is not None:
                delta_sums[name] += s["stress_after"] - s["stress_before"]

        candidates = [
            (name, delta_sums[name] / counts[name], counts[name])
            for name, count in counts.items()
            if count >= 2
        ]
        if not candidates:
            return (
                "You have sessions recorded but each technique was only used once. "
                "Repeat your favourite technique to build up comparison data."
            )

        candidates.sort(key=lambda x: x[1])  # lowest (most negative) delta first
        best_name, best_avg_delta, best_count = candidates[0]
        pct = round(abs(best_avg_delta) * 100)

        if best_avg_delta < 0:
            return (
                f"{best_name.capitalize()} reduces your stress index by "
                f"approximately {pct}% on average across {best_count} sessions. "
                "It is your most effective technique so far."
            )
        else:
            return (
                f"None of your techniques have shown a clear stress reduction yet. "
                f"Try sticking with one technique for at least 5 sessions before evaluating. "
                f"Your most-used technique is {best_name}."
            )
