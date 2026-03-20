"""
Breathing + HRV Biofeedback API routes.

Endpoints:
  GET  /breathing/patterns              — list evidence-based breathing patterns
  POST /breathing/prescribe             — prescribe pattern from stress/time
  POST /breathing/session/complete      — score a completed session
  POST /breathing/stress/pre            — record pre-session voice stress
  POST /breathing/stress/post           — record post-session voice stress + compute delta

References:
  - #279: HRV biofeedback resonance breathing (d=0.81)
  - #281: Breath-based stress detection from microphone
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.hrv_biofeedback import PATTERNS, prescribe_pattern, score_session

log = logging.getLogger(__name__)
router = APIRouter(prefix="/breathing", tags=["breathing"])

_WELLNESS_DISCLAIMER = (
    "Wellness tool only — not medical advice. Breathing pattern suggestions are based "
    "on published research but are not a substitute for professional medical guidance. "
    "If you have a respiratory or cardiovascular condition, consult your physician."
)

# In-memory store for pre-session stress snapshots (keyed by user_id)
_pre_stress: Dict[str, Dict[str, Any]] = {}


# ── Schemas ──────────────────────────────────────────────────────────────────

class PrescribeRequest(BaseModel):
    stress_index: float = Field(..., ge=0.0, le=1.0, description="Current stress 0-1")
    hour: int = Field(12, ge=0, le=23, description="Current hour 0-23")


class SessionCompleteRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    pattern_id: str = Field("resonance")
    duration_s: float = Field(..., gt=0, description="Actual session duration seconds")
    completed_cycles: int = Field(0, ge=0)
    stress_after: Optional[float] = Field(None, ge=0.0, le=1.0)


class StressSnapshotRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    stress_index: float = Field(..., ge=0.0, le=1.0)
    source: str = Field("voice", description="'voice' | 'eeg' | 'manual'")


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/patterns")
def list_patterns() -> List[Dict[str, Any]]:
    """Return all evidence-based breathing patterns."""
    return [
        {
            "id": p.id,
            "name": p.name,
            "tagline": p.tagline,
            "science": p.science,
            "phases": [
                {"label": ph.label, "duration_s": ph.duration_s, "direction": ph.direction}
                for ph in p.phases
            ],
            "cycle_duration_s": p.cycle_duration_s,
            "breaths_per_min": round(p.breaths_per_min, 2),
            "evidence_grade": p.evidence_grade,
        }
        for p in PATTERNS.values()
    ]


@router.post("/prescribe")
def prescribe(req: PrescribeRequest) -> Dict[str, Any]:
    """Recommend a breathing pattern based on stress level and time of day.

    Note: 'prescribe' is used in the API path for backward compatibility.
    This is a wellness recommendation, not a medical prescription.
    """
    pattern_id = prescribe_pattern(req.stress_index, req.hour)
    pattern = PATTERNS[pattern_id]
    reason = (
        "Cyclic sighing is fastest for acute stress (Stanford 2023)" if pattern_id == "cyclic_sigh"
        else "Resonance breathing has strongest long-term evidence (d=0.81)" if pattern_id == "resonance"
        else "4-7-8 breathing is optimal for pre-sleep relaxation" if pattern_id == "478"
        else "Box breathing supports calm focus at low stress levels"
    )
    return {
        "recommended_pattern_id": pattern_id,
        "name": pattern.name,
        "reason": reason,
        "breaths_per_min": round(pattern.breaths_per_min, 2),
        "evidence_grade": pattern.evidence_grade,
        "disclaimer": _WELLNESS_DISCLAIMER,
    }


@router.post("/stress/pre")
def record_pre_stress(req: StressSnapshotRequest) -> Dict[str, Any]:
    """Record pre-session stress baseline (from voice or EEG)."""
    _pre_stress[req.user_id] = {
        "stress_index": req.stress_index,
        "source": req.source,
        "ts": time.time(),
    }
    recommended_pattern = prescribe_pattern(
        req.stress_index,
        int(time.strftime("%H")),
    )
    return {
        "status": "recorded",
        "stress_index": req.stress_index,
        "recommended_pattern": recommended_pattern,
    }


@router.post("/session/complete")
def complete_session(req: SessionCompleteRequest) -> Dict[str, Any]:
    """Score a completed breathing session and compute stress delta."""
    pre = _pre_stress.get(req.user_id)
    stress_before = pre["stress_index"] if pre else None

    result = score_session(
        session_duration_s=req.duration_s,
        pattern_id=req.pattern_id,
        stress_before=stress_before,
        stress_after=req.stress_after,
        completed_cycles=req.completed_cycles,
    )

    # Clean up pre-session snapshot
    _pre_stress.pop(req.user_id, None)

    # Auto-log to supplement tracker for cross-modal correlation
    try:
        from api.routes.supplement_tracker import get_tracker
        tracker = get_tracker()
        tracker.log_brain_state(
            user_id=req.user_id,
            timestamp=time.time(),
            emotion_data={
                "stress_index": req.stress_after or 0.3,
                "source": "breathing_session",
                "session_type": req.pattern_id,
                "coherence_score": result["coherence_score"],
            },
        )
    except Exception as exc:
        log.debug("Supplement tracker log skipped: %s", exc)

    return result
