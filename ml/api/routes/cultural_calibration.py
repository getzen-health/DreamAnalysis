"""Cultural emotion calibration API routes (issue #436).

POST /cultural/calibrate  -- apply cultural calibration to emotion data
GET  /cultural/profiles   -- list available cultural profiles
GET  /cultural/status     -- health check
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/cultural", tags=["cultural-calibration"])


# -- Pydantic schemas --------------------------------------------------------


class CalibrateRequest(BaseModel):
    culture: str = Field(
        ...,
        min_length=1,
        description="Cultural cluster name (e.g. 'east_asian', 'latin_american')",
    )
    valence: float = Field(..., ge=-1.0, le=1.0, description="Measured valence")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Measured arousal")
    reported_valence: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Self-reported valence (optional)"
    )
    reported_arousal: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Self-reported arousal (optional)"
    )


class DisplayRuleCorrectionOut(BaseModel):
    original_valence: float
    original_arousal: float
    corrected_valence: float
    corrected_arousal: float
    valence_adjustment: float
    arousal_adjustment: float
    correction_rationale: str


class SelfReportCalibrationOut(BaseModel):
    original_valence: float
    original_arousal: float
    calibrated_valence: float
    calibrated_arousal: float
    bias_corrections_applied: List[str]


class AffectValuationOut(BaseModel):
    ideal_arousal: float
    ideal_valence_type: str
    current_arousal: float
    current_valence: float
    alignment_score: float
    interpretation: str


class CalibrateResponse(BaseModel):
    culture: str
    display_rule_correction: DisplayRuleCorrectionOut
    self_report_calibration: SelfReportCalibrationOut
    affect_valuation: AffectValuationOut
    adapted_thresholds: Dict[str, float]
    profile_summary: Dict[str, Any]


class ProfileOut(BaseModel):
    cluster_name: str
    description: str
    display_rules: Dict[str, float]
    affect_valuation: Dict[str, Any]
    self_report_biases: Dict[str, float]


# -- Endpoints ---------------------------------------------------------------


@router.post("/calibrate", response_model=CalibrateResponse)
def cultural_calibrate(req: CalibrateRequest) -> Dict[str, Any]:
    """Apply cultural calibration to emotion data.

    Corrects measured emotion readings for cultural display rules, calibrates
    self-report values for known response biases, computes affect valuation
    alignment, and adapts detection thresholds.
    """
    try:
        from models.cultural_calibration import calibrate
    except ImportError as exc:
        log.error("cultural_calibration import failed: %s", exc)
        raise HTTPException(503, "Cultural calibration model unavailable") from exc

    try:
        result = calibrate(
            valence=req.valence,
            arousal=req.arousal,
            culture=req.culture,
            reported_valence=req.reported_valence,
            reported_arousal=req.reported_arousal,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        log.exception("Cultural calibration failed: %s", exc)
        raise HTTPException(500, f"Calibration error: {exc}") from exc

    # Serialise dataclasses to dicts for Pydantic
    return {
        "culture": result.culture,
        "display_rule_correction": {
            "original_valence": result.display_rule_correction.original_valence,
            "original_arousal": result.display_rule_correction.original_arousal,
            "corrected_valence": result.display_rule_correction.corrected_valence,
            "corrected_arousal": result.display_rule_correction.corrected_arousal,
            "valence_adjustment": result.display_rule_correction.valence_adjustment,
            "arousal_adjustment": result.display_rule_correction.arousal_adjustment,
            "correction_rationale": result.display_rule_correction.correction_rationale,
        },
        "self_report_calibration": {
            "original_valence": result.self_report_calibration.original_valence,
            "original_arousal": result.self_report_calibration.original_arousal,
            "calibrated_valence": result.self_report_calibration.calibrated_valence,
            "calibrated_arousal": result.self_report_calibration.calibrated_arousal,
            "bias_corrections_applied": result.self_report_calibration.bias_corrections_applied,
        },
        "affect_valuation": {
            "ideal_arousal": result.affect_valuation.ideal_arousal,
            "ideal_valence_type": result.affect_valuation.ideal_valence_type,
            "current_arousal": result.affect_valuation.current_arousal,
            "current_valence": result.affect_valuation.current_valence,
            "alignment_score": result.affect_valuation.alignment_score,
            "interpretation": result.affect_valuation.interpretation,
        },
        "adapted_thresholds": result.adapted_thresholds,
        "profile_summary": result.profile_summary,
    }


@router.get("/profiles")
def list_cultural_profiles() -> Dict[str, Any]:
    """List all available cultural profiles."""
    try:
        from models.cultural_calibration import CULTURAL_PROFILES, profile_to_dict
    except ImportError as exc:
        log.error("cultural_calibration import failed: %s", exc)
        raise HTTPException(503, "Cultural calibration model unavailable") from exc

    profiles = {
        name: profile_to_dict(p) for name, p in CULTURAL_PROFILES.items()
    }
    return {
        "count": len(profiles),
        "profiles": profiles,
    }


@router.get("/status")
def cultural_status() -> Dict[str, Any]:
    """Health check for the cultural calibration module."""
    try:
        from models.cultural_calibration import CULTURAL_PROFILES  # noqa: F401
        available = True
    except ImportError:
        available = False

    return {
        "available": available,
        "model_type": "cultural display-rule calibration",
        "description": (
            "Adapts emotion recognition to cultural display rules, affect valuation "
            "differences, and self-report response biases across 8 major cultural "
            "clusters. Corrects measured emotion for suppression/amplification norms "
            "and adjusts detection thresholds."
        ),
        "cultural_clusters": [
            "east_asian", "south_asian", "latin_american", "nordic",
            "north_american", "middle_eastern", "sub_saharan_african",
            "western_european",
        ],
    }
