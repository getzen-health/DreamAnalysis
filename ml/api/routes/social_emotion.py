"""Social emotion tracking — dyadic voice analysis and interpersonal EI.

Tracks emotional dynamics between people (couples, teams, therapist-client)
through voice conversation analysis and interpersonal EI scoring.

POST /social-emotion/analyze-conversation — analyze two-speaker conversation
POST /social-emotion/log-session/{user_id} — log interpersonal session data
GET  /social-emotion/report/{user_id}      — interpersonal EI report
GET  /social-emotion/status                — health check

Research basis:
- Borelli et al. (2019): Pitch entrainment predicts relationship satisfaction (r=0.31-0.45)
- Gottman (2000): Negative vocal affect predicts divorce with 93.6% accuracy (acoustic)
- Emotional contagion: Speaker A arousal predicts Speaker B arousal 2-5s later
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["Social Emotion"])

_DATA_DIR = Path("data/social_emotion")


# ── Pydantic models ───────────────────────────────────────────────────────────

class SpeakerState(BaseModel):
    """Emotional state snapshot for one speaker."""
    speaker_id: str = Field(..., description="'A' or 'B' (or any label)")
    valence: float = Field(0.0, ge=-1.0, le=1.0)
    arousal: float = Field(0.5, ge=0.0, le=1.0)
    stress_index: float = Field(0.5, ge=0.0, le=1.0)
    f0_mean: Optional[float] = Field(None, description="Mean fundamental frequency (Hz)")
    f0_std: Optional[float] = Field(None, description="F0 standard deviation (Hz)")
    speech_rate: Optional[float] = Field(None, description="Words/min or syllables/s")
    pause_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="Fraction of silence")


class ConversationRequest(BaseModel):
    speaker_a: SpeakerState
    speaker_b: SpeakerState
    duration_seconds: Optional[float] = Field(None, ge=0.0)
    context: Optional[str] = Field(None, description="e.g. 'couple', 'team', 'therapy'")


class SessionData(BaseModel):
    partner_id: Optional[str] = Field(None, description="ID of the other person (optional)")
    own_valence: float = Field(0.0, ge=-1.0, le=1.0)
    own_arousal: float = Field(0.5, ge=0.0, le=1.0)
    own_stress: float = Field(0.5, ge=0.0, le=1.0)
    partner_valence: Optional[float] = Field(None, ge=-1.0, le=1.0)
    partner_arousal: Optional[float] = Field(None, ge=0.0, le=1.0)
    context: Optional[str] = Field(None)
    date: Optional[str] = Field(None)


# ── Dyadic analysis ───────────────────────────────────────────────────────────

def _emotional_synchrony(a: SpeakerState, b: SpeakerState) -> float:
    """Compute emotional synchrony (0-1) between two speakers.

    Based on Borelli (2019): arousal synchrony predicts relationship quality.
    Higher synchrony = stronger emotional attunement.
    """
    # Valence agreement (0-1)
    valence_diff = abs(a.valence - b.valence)
    valence_sync = max(0.0, 1.0 - valence_diff)

    # Arousal synchrony
    arousal_diff = abs(a.arousal - b.arousal)
    arousal_sync = max(0.0, 1.0 - arousal_diff)

    # F0 entrainment (if both have pitch data)
    pitch_sync = 0.5  # neutral if no data
    if a.f0_mean is not None and b.f0_mean is not None and a.f0_mean > 0 and b.f0_mean > 0:
        ratio = min(a.f0_mean, b.f0_mean) / max(a.f0_mean, b.f0_mean)
        pitch_sync = ratio  # 1.0 = identical pitch, 0 = maximally different

    return round(0.40 * arousal_sync + 0.35 * valence_sync + 0.25 * pitch_sync, 3)


def _conflict_risk(a: SpeakerState, b: SpeakerState) -> float:
    """Estimate conflict risk (0-1) from vocal emotional states.

    Gottman (2000) markers: negative valence + high arousal (fight-or-flight) +
    stress asymmetry (one person stressed, other calm → mismatch).
    """
    # Both negative = risk
    avg_valence = (a.valence + b.valence) / 2
    neg_score = max(0.0, -avg_valence)  # 0 → 1 as valence goes −1

    # High combined arousal
    avg_arousal = (a.arousal + b.arousal) / 2
    arousal_score = max(0.0, avg_arousal - 0.5) * 2  # 0.5→0, 1.0→1

    # Stress asymmetry (one calm, one stressed = frustration signal)
    stress_diff = abs(a.stress_index - b.stress_index)
    asymmetry_score = stress_diff  # 0→0, 1→1

    risk = 0.45 * neg_score + 0.30 * arousal_score + 0.25 * asymmetry_score
    return round(min(1.0, max(0.0, risk)), 3)


def _emotional_contagion(a: SpeakerState, b: SpeakerState) -> Dict:
    """Detect potential emotional contagion direction.

    High-arousal speaker tends to raise the other's arousal within 2-5s.
    Returns: direction ('A→B', 'B→A', 'mutual', 'none') + strength.
    """
    diff = a.arousal - b.arousal
    if abs(diff) < 0.10:
        return {"direction": "mutual", "strength": round(1.0 - abs(diff) * 5, 2)}
    if diff > 0.20:
        return {"direction": "A→B", "strength": round(min(1.0, diff * 2), 2)}
    if diff < -0.20:
        return {"direction": "B→A", "strength": round(min(1.0, abs(diff) * 2), 2)}
    return {"direction": "none", "strength": 0.0}


def _relationship_quality_hint(synchrony: float, conflict_risk: float) -> str:
    if synchrony > 0.70 and conflict_risk < 0.25:
        return "Strong emotional attunement — high synchrony, low conflict markers."
    if conflict_risk > 0.60:
        return (
            "Elevated conflict markers — significant negative valence or stress asymmetry. "
            "Consider: active listening, de-escalation breathing, brief break."
        )
    if synchrony < 0.35:
        return (
            "Low emotional synchrony — speakers appear emotionally misaligned. "
            "This is normal in task-focused conversations."
        )
    return "Moderate emotional alignment — typical for mixed-context conversations."


# ── Session persistence ───────────────────────────────────────────────────────

def _load_user(user_id: str) -> Dict:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = _DATA_DIR / f"{user_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"user_id": user_id, "sessions": [], "interpersonal_scores": []}


def _save_user(user_id: str, data: Dict) -> None:
    (_DATA_DIR / f"{user_id}.json").write_text(json.dumps(data, indent=2, default=str))


def _interpersonal_score(sessions: List[Dict]) -> float:
    """Bar-On interpersonal domain score (0-1) from session history."""
    if not sessions:
        return 0.5
    # Interpersonal EI: positive valence during interactions, low conflict
    avg_own = sum(s.get("own_valence", 0.0) for s in sessions[-10:]) / min(10, len(sessions))
    avg_sync = sum(s.get("synchrony", 0.5) for s in sessions[-10:]) / min(10, len(sessions))
    score = 0.5 * (avg_own + 1.0) / 2.0 + 0.5 * avg_sync
    return round(min(1.0, max(0.0, score)), 3)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/social-emotion/analyze-conversation")
def analyze_conversation(req: ConversationRequest) -> dict:
    """Analyze two-speaker conversation emotional dynamics.

    Computes:
    - Emotional synchrony (Borelli 2019: pitch + arousal entrainment)
    - Conflict risk (Gottman 2000: negative affect + arousal markers)
    - Emotional contagion direction (who is influencing whom)
    - Relationship quality hint

    Does NOT require audio — works from pre-computed emotion states.
    For raw audio analysis, run voice emotion on each speaker first.
    """
    try:
        a, b = req.speaker_a, req.speaker_b
        synchrony = _emotional_synchrony(a, b)
        conflict = _conflict_risk(a, b)
        contagion = _emotional_contagion(a, b)
        quality = _relationship_quality_hint(synchrony, conflict)

        return {
            "speakers": {
                a.speaker_id: {"valence": a.valence, "arousal": a.arousal, "stress": a.stress_index},
                b.speaker_id: {"valence": b.valence, "arousal": b.arousal, "stress": b.stress_index},
            },
            "emotional_synchrony": synchrony,
            "conflict_risk": conflict,
            "conflict_level": (
                "low" if conflict < 0.30
                else "moderate" if conflict < 0.60
                else "high"
            ),
            "emotional_contagion": contagion,
            "relationship_quality_hint": quality,
            "context": req.context,
            "research_notes": {
                "synchrony": "r=0.31-0.45 predicts relationship satisfaction (Borelli 2019)",
                "conflict": "Negative vocal affect predicts conflict (Gottman 2000)",
                "contagion": "Arousal contagion occurs within 2-5 seconds between speakers",
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/social-emotion/log-session/{user_id}")
def log_session(user_id: str, session: SessionData) -> dict:
    """Log an interpersonal session for longitudinal EI tracking.

    Updates the user's interpersonal EI domain score based on:
    - Own valence during interaction
    - Emotional synchrony with partner (if partner state provided)
    """
    try:
        data = _load_user(user_id)

        # Compute synchrony if partner state provided
        synchrony = None
        if session.partner_valence is not None and session.partner_arousal is not None:
            valence_sync = max(0.0, 1.0 - abs(session.own_valence - session.partner_valence))
            arousal_sync = max(0.0, 1.0 - abs(session.own_arousal - session.partner_arousal))
            synchrony = round(0.5 * valence_sync + 0.5 * arousal_sync, 3)

        record = {
            "date": session.date or date.today().isoformat(),
            "partner_id": session.partner_id,
            "own_valence": session.own_valence,
            "own_arousal": session.own_arousal,
            "own_stress": session.own_stress,
            "partner_valence": session.partner_valence,
            "partner_arousal": session.partner_arousal,
            "synchrony": synchrony,
            "context": session.context,
        }
        data["sessions"].append(record)

        interpersonal = _interpersonal_score(data["sessions"])
        data["interpersonal_scores"].append({
            "date": record["date"],
            "score": interpersonal,
        })

        _save_user(user_id, data)
        return {
            "logged": True,
            "session_count": len(data["sessions"]),
            "interpersonal_ei_score": interpersonal,
            "synchrony": synchrony,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/social-emotion/report/{user_id}")
def get_report(user_id: str) -> dict:
    """Return longitudinal interpersonal EI report.

    Includes:
    - Interpersonal EI domain score (Bar-On model)
    - Trend over last 30 sessions
    - Average synchrony with partners
    - Conflict frequency estimate
    """
    try:
        data = _load_user(user_id)
        sessions = data.get("sessions", [])
        scores = data.get("interpersonal_scores", [])

        interpersonal = _interpersonal_score(sessions)

        # Trend: last 10 vs prior 10
        trend = "stable"
        if len(scores) >= 20:
            recent_avg = sum(s["score"] for s in scores[-10:]) / 10
            prior_avg = sum(s["score"] for s in scores[-20:-10]) / 10
            if recent_avg - prior_avg > 0.05:
                trend = "improving"
            elif prior_avg - recent_avg > 0.05:
                trend = "declining"

        # Synchrony stats
        sync_values = [s["synchrony"] for s in sessions if s.get("synchrony") is not None]
        avg_sync = sum(sync_values) / len(sync_values) if sync_values else None

        return {
            "user_id": user_id,
            "sessions_logged": len(sessions),
            "interpersonal_ei_score": interpersonal,
            "trend": trend,
            "average_synchrony": round(avg_sync, 3) if avg_sync is not None else None,
            "score_history": scores[-30:],
            "insight": _interpersonal_insight(interpersonal, trend, avg_sync),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _interpersonal_insight(score: float, trend: str, avg_sync: Optional[float]) -> str:
    if score > 0.70:
        return "Strong interpersonal EI — positive interactions with high emotional attunement."
    if trend == "improving":
        return "Interpersonal EI is improving across sessions — keep logging to track progress."
    if avg_sync is not None and avg_sync < 0.40:
        return (
            "Low emotional synchrony in interactions. "
            "Practice: active listening, reflecting partner's emotional language."
        )
    if score < 0.40:
        return (
            "Low interpersonal EI score. Common patterns: stress bleeds into interactions, "
            "emotional mismatch with others. Consider: stress reduction before social interactions."
        )
    return "Moderate interpersonal EI. Building consistent positive interactions increases score."


@router.get("/social-emotion/status")
def status() -> dict:
    return {
        "status": "ready",
        "capabilities": [
            "emotional_synchrony_analysis",
            "conflict_risk_detection",
            "emotional_contagion_direction",
            "interpersonal_ei_tracking",
        ],
        "privacy_note": "No audio stored. Analysis uses pre-computed emotion features only.",
        "research_basis": [
            "Borelli et al. (2019): Pitch entrainment r=0.31-0.45",
            "Gottman (2000): Vocal affect conflict prediction 93.6%",
        ],
    }
