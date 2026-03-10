"""Longitudinal emotional intelligence growth tracker.

Tracks EI growth over weeks/months using:
  - Daily behavioral proxies (valence variance, emotion regulation speed)
  - Periodic formal assessment (5-domain Bar-On proxy via behavioral data)
  - Trend analysis with effect size (Cohen's d) vs baseline

Based on:
- Mattingly & Kraiger (2019): d=0.46 average from CBT-based EI interventions
- Bar-On EQ-i 2.0: 5 domains (intrapersonal, interpersonal, stress management,
  adaptability, general mood)
- Keng et al. (2011): mindfulness d=0.36-0.72 on emotion regulation

Storage: JSON file at data/ei_growth/<user_id>.json
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

_DATA_DIR = Path("data/ei_growth")
_LOCK = threading.Lock()

# Domain names (Bar-On EQ-i 2.0 proxy)
_DOMAINS = [
    "intrapersonal",      # self-awareness, emotional insight
    "interpersonal",      # empathy, social awareness (voice prosody)
    "stress_management",  # stress recovery speed, regulation
    "adaptability",       # emotional flexibility, reframing
    "general_mood",       # baseline positive affect, optimism
]


def _load_user(user_id: str) -> Dict:
    path = _DATA_DIR / f"{user_id}.json"
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "user_id": user_id,
            "sessions": [],      # list of session snapshots
            "weekly_scores": [], # aggregated weekly domain scores
            "baseline": None,    # first-week average per domain
            "created_at": time.time(),
        }


def _save_user(user_id: str, data: Dict) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    path = _DATA_DIR / f"{user_id}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _compute_domain_scores(session: Dict) -> Dict[str, float]:
    """Derive 5 EI domain scores from a session's emotion/health metrics.

    Scores are in [0, 1] and represent the estimated EI level for that domain
    based on observable behavioral proxies from this session.
    """
    # --- Intrapersonal (self-awareness) ---
    # High valence accuracy (user correct about their state) + low variance
    valence = session.get("valence", 0.0)
    stress_idx = session.get("stress_index", 0.5)
    emotion_conf = session.get("confidence", 0.5)
    intrapersonal = float(np.clip(0.4 * emotion_conf + 0.3 * (1 - abs(valence)) + 0.3 * (1 - stress_idx), 0, 1))

    # --- Interpersonal (social / empathy proxy) ---
    # Positive valence + low anger + high relaxation correlate with empathy
    anger_idx = session.get("anger_index", 0.3)
    relax_idx = session.get("relaxation_index", 0.5)
    interpersonal = float(np.clip(0.4 * max(0, valence) + 0.3 * relax_idx + 0.3 * (1 - anger_idx), 0, 1))

    # --- Stress management ---
    # Low stress index + high HRV proxy (focus_index inversely relates to worry)
    focus_idx = session.get("focus_index", 0.5)
    stress_mgmt = float(np.clip(0.5 * (1 - stress_idx) + 0.3 * focus_idx + 0.2 * relax_idx, 0, 1))

    # --- Adaptability ---
    # Emotion diversity (entropy of emotion probabilities) × regulation speed
    probs = session.get("probabilities", {})
    if probs and len(probs) == 6:
        p = np.array(list(probs.values()), dtype=float)
        p = p / (p.sum() + 1e-9)
        entropy = float(-np.sum(p * np.log(p + 1e-9)) / np.log(6))  # normalized
    else:
        entropy = 0.5
    adaptability = float(np.clip(0.5 * entropy + 0.3 * (1 - stress_idx) + 0.2 * focus_idx, 0, 1))

    # --- General mood (positive affect baseline) ---
    arousal = session.get("arousal", 0.5)
    general_mood = float(np.clip(0.5 * max(0, valence) + 0.3 * (1 - stress_idx) + 0.2 * arousal, 0, 1))

    return {
        "intrapersonal": intrapersonal,
        "interpersonal": interpersonal,
        "stress_management": stress_mgmt,
        "adaptability": adaptability,
        "general_mood": general_mood,
    }


class EIGrowthTracker:
    """Track emotional intelligence growth over time for a user.

    Call ``log_session()`` after each emotion analysis to accumulate data.
    Call ``get_report()`` to get trend analysis and growth metrics.
    """

    def log_session(self, user_id: str, session_data: Dict) -> Dict:
        """Record a new session and return current EI snapshot.

        Args:
            user_id: Unique user identifier.
            session_data: Dict with emotion/health metrics (valence, arousal,
                stress_index, focus_index, relaxation_index, anger_index,
                confidence, probabilities).

        Returns:
            Dict with current domain scores, trend, and data count.
        """
        with _LOCK:
            data = _load_user(user_id)

            domain_scores = _compute_domain_scores(session_data)
            snapshot = {
                "timestamp": time.time(),
                "domain_scores": domain_scores,
                "composite": float(np.mean(list(domain_scores.values()))),
                "raw": {
                    k: session_data.get(k)
                    for k in ("valence", "arousal", "stress_index", "focus_index",
                              "relaxation_index", "anger_index", "confidence")
                },
            }
            data["sessions"].append(snapshot)

            # Establish baseline from first 7 sessions
            if data["baseline"] is None and len(data["sessions"]) >= 7:
                first_week = data["sessions"][:7]
                data["baseline"] = {
                    d: float(np.mean([s["domain_scores"][d] for s in first_week]))
                    for d in _DOMAINS
                }

            # Aggregate weekly score every 7 sessions
            if len(data["sessions"]) % 7 == 0:
                week_sessions = data["sessions"][-7:]
                weekly = {
                    "week": len(data["weekly_scores"]) + 1,
                    "timestamp": time.time(),
                    "domain_scores": {
                        d: float(np.mean([s["domain_scores"][d] for s in week_sessions]))
                        for d in _DOMAINS
                    },
                    "composite": float(
                        np.mean([s["composite"] for s in week_sessions])
                    ),
                }
                data["weekly_scores"].append(weekly)

            _save_user(user_id, data)

            return {
                "domain_scores": domain_scores,
                "composite": snapshot["composite"],
                "session_count": len(data["sessions"]),
                "weeks_tracked": len(data["weekly_scores"]),
            }

    def get_report(self, user_id: str) -> Dict:
        """Return full EI growth report with trends and effect sizes.

        Returns:
            Dict with: current_scores, baseline_scores, growth (Cohen's d
            per domain), trend, weeks_tracked, insight, session_count.
        """
        with _LOCK:
            data = _load_user(user_id)

        sessions = data["sessions"]
        if not sessions:
            return {"error": "no_data", "message": "No sessions recorded yet"}

        # Current scores = average of last 7 sessions (or all)
        recent = sessions[-min(7, len(sessions)):]
        current = {
            d: float(np.mean([s["domain_scores"][d] for s in recent]))
            for d in _DOMAINS
        }
        current_composite = float(np.mean(list(current.values())))

        baseline = data.get("baseline")
        growth: Dict[str, Any] = {}
        effect_sizes: List[float] = []

        if baseline and len(sessions) >= 14:
            # Cohen's d per domain: (current - baseline) / pooled_sd
            # Use within-domain session std as pooled estimate
            for d in _DOMAINS:
                all_scores = [s["domain_scores"][d] for s in sessions]
                sd = float(np.std(all_scores)) or 0.1
                d_score = (current[d] - baseline[d]) / sd
                growth[d] = {
                    "baseline": round(baseline[d], 3),
                    "current": round(current[d], 3),
                    "change": round(current[d] - baseline[d], 3),
                    "effect_size_d": round(d_score, 2),
                    "magnitude": _effect_label(d_score),
                }
                effect_sizes.append(d_score)

        # Trend over weekly composites
        weekly_scores = data.get("weekly_scores", [])
        trend = "insufficient_data"
        if len(weekly_scores) >= 2:
            composites = [w["composite"] for w in weekly_scores]
            t = np.linspace(0, 1, len(composites))
            slope = float(np.polyfit(t, composites, 1)[0])
            if slope > 0.02:
                trend = "improving"
            elif slope < -0.02:
                trend = "declining"
            else:
                trend = "stable"

        # Strongest and weakest domain
        strongest = max(current, key=current.get) if current else None
        weakest = min(current, key=current.get) if current else None

        overall_effect = float(np.mean(effect_sizes)) if effect_sizes else None
        insight = _generate_insight(trend, strongest, weakest, overall_effect, len(sessions))

        return {
            "user_id": user_id,
            "session_count": len(sessions),
            "weeks_tracked": len(weekly_scores),
            "current_scores": {d: round(v, 3) for d, v in current.items()},
            "current_composite": round(current_composite, 3),
            "baseline_scores": {d: round(v, 3) for d, v in baseline.items()} if baseline else None,
            "growth": growth,
            "overall_effect_size_d": round(overall_effect, 2) if overall_effect is not None else None,
            "trend": trend,
            "strongest_domain": strongest,
            "weakest_domain": weakest,
            "insight": insight,
            "weekly_progression": [
                {"week": w["week"], "composite": round(w["composite"], 3)}
                for w in weekly_scores
            ],
        }

    def get_domain_history(self, user_id: str, domain: str, limit: int = 30) -> Dict:
        """Return time-series for one EI domain (last N sessions)."""
        if domain not in _DOMAINS:
            return {"error": f"Unknown domain. Valid: {_DOMAINS}"}
        with _LOCK:
            data = _load_user(user_id)
        sessions = data["sessions"][-limit:]
        return {
            "domain": domain,
            "values": [
                {
                    "timestamp": s["timestamp"],
                    "score": round(s["domain_scores"][domain], 3),
                }
                for s in sessions
            ],
        }

    def reset(self, user_id: str) -> Dict:
        with _LOCK:
            _save_user(user_id, {
                "user_id": user_id,
                "sessions": [],
                "weekly_scores": [],
                "baseline": None,
                "created_at": time.time(),
            })
        return {"status": "reset"}


def _effect_label(d: float) -> str:
    if d >= 0.8:
        return "large"
    if d >= 0.5:
        return "medium"
    if d >= 0.2:
        return "small"
    if d > -0.2:
        return "negligible"
    return "negative"


def _generate_insight(
    trend: str,
    strongest: Optional[str],
    weakest: Optional[str],
    overall_d: Optional[float],
    n_sessions: int,
) -> str:
    if n_sessions < 7:
        remaining = 7 - n_sessions
        return f"Keep going — {remaining} more session{'s' if remaining > 1 else ''} needed to establish your baseline."

    parts = []
    if trend == "improving":
        parts.append("Your emotional intelligence is trending upward.")
    elif trend == "declining":
        parts.append("Your scores have dipped recently — consistent practice helps.")
    else:
        parts.append("Your EI scores are stable.")

    if overall_d is not None:
        if overall_d >= 0.5:
            parts.append(f"Overall growth vs baseline: d={overall_d:.2f} (medium-large effect — clinically meaningful).")
        elif overall_d >= 0.2:
            parts.append(f"Overall growth vs baseline: d={overall_d:.2f} (small-medium effect).")

    if strongest:
        label = strongest.replace("_", " ")
        parts.append(f"Your strongest domain is {label}.")
    if weakest and weakest != strongest:
        label = weakest.replace("_", " ")
        parts.append(f"Focus area: {label}.")

    return " ".join(parts)


# Module-level singleton
_tracker: Optional[EIGrowthTracker] = None


def get_tracker() -> EIGrowthTracker:
    global _tracker
    if _tracker is None:
        _tracker = EIGrowthTracker()
    return _tracker
