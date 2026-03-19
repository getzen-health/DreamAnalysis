"""Perinatal emotional intelligence API routes — issue #451.

EEG-guided pregnancy & postpartum mood tracking with PPD early warning.

Endpoints:
  POST /perinatal/assess    -- assess current perinatal emotional state
  POST /perinatal/ppd-risk  -- compute PPD risk score
  GET  /perinatal/status    -- model availability

Issue #451.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.perinatal_model import (
    BluesVsPPD,
    BondingLevel,
    HormonalPhase,
    PPDRiskLevel,
    PerinatalPhase,
    PerinatalReading,
    PerinatalState,
    _CLINICAL_DISCLAIMER,
    _CRISIS_RESOURCES,
    compute_perinatal_profile,
    compute_trimester_baseline,
    profile_to_dict,
    score_ppd_risk,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/perinatal", tags=["perinatal"])


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class PerinatalReadingInput(BaseModel):
    valence: float = Field(..., ge=-1.0, le=1.0, description="Emotional valence (-1 to 1)")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Emotional arousal (0 to 1)")
    stress_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Stress index")
    anxiety_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Anxiety index")
    irritability_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Irritability index")
    fatigue_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Fatigue index")
    tearfulness_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Tearfulness index")
    bonding_response: float = Field(default=0.5, ge=0.0, le=1.0, description="Baby stimulus response")
    partner_valence: Optional[float] = Field(default=None, ge=-1.0, le=1.0, description="Partner valence")
    partner_stress: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Partner stress")


class AssessRequest(BaseModel):
    reading: PerinatalReadingInput = Field(..., description="Current emotional reading")
    weeks_pregnant: Optional[int] = Field(default=None, ge=0, le=42, description="Weeks pregnant (None if postpartum)")
    weeks_postpartum: Optional[int] = Field(default=None, ge=0, le=52, description="Weeks postpartum (None if pregnant)")
    recent_readings: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Previous readings for trend analysis",
    )


class PPDRiskRequest(BaseModel):
    reading: PerinatalReadingInput = Field(..., description="Current emotional reading")
    weeks_postpartum: int = Field(..., ge=0, le=52, description="Weeks postpartum")
    recent_readings: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Previous readings for persistence detection",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_domain_reading(inp: PerinatalReadingInput) -> PerinatalReading:
    """Convert Pydantic input to domain dataclass."""
    return PerinatalReading(
        valence=inp.valence,
        arousal=inp.arousal,
        stress_index=inp.stress_index,
        anxiety_index=inp.anxiety_index,
        irritability_index=inp.irritability_index,
        fatigue_index=inp.fatigue_index,
        tearfulness_index=inp.tearfulness_index,
        bonding_response=inp.bonding_response,
        partner_valence=inp.partner_valence,
        partner_stress=inp.partner_stress,
    )


def _parse_recent_readings(raw: Optional[List[Dict[str, Any]]]) -> Optional[List[PerinatalReading]]:
    """Parse raw recent readings dicts into domain objects."""
    if not raw:
        return None
    readings: List[PerinatalReading] = []
    for r in raw:
        readings.append(PerinatalReading(
            valence=r.get("valence", 0.0),
            arousal=r.get("arousal", 0.0),
            stress_index=r.get("stress_index", 0.0),
            anxiety_index=r.get("anxiety_index", 0.0),
            irritability_index=r.get("irritability_index", 0.0),
            fatigue_index=r.get("fatigue_index", 0.0),
            tearfulness_index=r.get("tearfulness_index", 0.0),
            bonding_response=r.get("bonding_response", 0.5),
            partner_valence=r.get("partner_valence"),
            partner_stress=r.get("partner_stress"),
            timestamp=r.get("timestamp", 0.0),
        ))
    return readings


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/assess")
async def assess_perinatal(req: AssessRequest) -> Dict[str, Any]:
    """Assess current perinatal emotional state.

    Computes the full perinatal profile: trimester baseline, PPD risk,
    blues-vs-PPD distinction, hormonal context, bonding readiness, and
    partner support assessment.
    """
    reading = _to_domain_reading(req.reading)
    state = PerinatalState(
        weeks_pregnant=req.weeks_pregnant,
        weeks_postpartum=req.weeks_postpartum,
    )
    recent = _parse_recent_readings(req.recent_readings)

    profile = compute_perinatal_profile(reading, state, recent)
    return profile_to_dict(profile)


@router.post("/ppd-risk")
async def compute_ppd_risk(req: PPDRiskRequest) -> Dict[str, Any]:
    """Compute PPD risk score from emotional reading.

    Returns an EPDS-compatible proxy score (0-30) with risk classification.
    This is NOT a replacement for the clinical EPDS questionnaire.
    """
    reading = _to_domain_reading(req.reading)
    state = PerinatalState(weeks_postpartum=req.weeks_postpartum)
    recent = _parse_recent_readings(req.recent_readings)

    risk = score_ppd_risk(reading, state, recent)

    return {
        "epds_proxy_score": risk.epds_proxy_score,
        "risk_level": risk.risk_level.value,
        "confidence": risk.confidence,
        "indicators": risk.indicators,
        "recommendation": risk.recommendation,
        "clinical_disclaimer": risk.clinical_disclaimer,
        "crisis_resources": _CRISIS_RESOURCES,
    }


@router.get("/status")
async def status() -> Dict[str, Any]:
    """Model availability and perinatal module info."""
    return {
        "status": "ok",
        "model": "perinatal_emotional_intelligence",
        "version": "1.0.0",
        "phases": [p.value for p in PerinatalPhase if p != PerinatalPhase.UNKNOWN],
        "risk_levels": [r.value for r in PPDRiskLevel],
        "blues_vs_ppd_classes": [b.value for b in BluesVsPPD],
        "hormonal_phases": [h.value for h in HormonalPhase if h != HormonalPhase.UNKNOWN],
        "bonding_levels": [b.value for b in BondingLevel],
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
        "crisis_resources": _CRISIS_RESOURCES,
    }
