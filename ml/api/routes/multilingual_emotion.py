"""Cross-cultural voice emotion recognition endpoints.

POST /voice-emotion/multilingual   — Culture-aware voice emotion prediction
GET  /voice-emotion/languages      — List supported languages + culture groups
POST /voice-emotion/set-culture    — Override auto-detected culture group
POST /voice-emotion/calibrate      — Calibrate existing emotion predictions
POST /ei/cultural-adjust           — Adjust EI scores for cultural context
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.multilingual_emotion import (
    CulturalCalibrator,
    CulturalEIAdapter,
    get_culture_group,
    _COLLECTIVIST_LANGS,
    _INDIVIDUALIST_LANGS,
    _MIXED_LANGS,
    _CULTURE_PROFILES,
)

log = logging.getLogger(__name__)
router = APIRouter(tags=["multilingual-emotion"])

_calibrator = CulturalCalibrator()
_ei_adapter = CulturalEIAdapter()

# Per-user culture overrides
_user_culture_overrides: Dict[str, str] = {}


# ── Request models ────────────────────────────────────────────────────────────


class CalibrateRequest(BaseModel):
    probabilities: Dict[str, float] = Field(
        ..., description="Raw emotion probabilities (6-class)"
    )
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.5, ge=0.0, le=1.0)
    language: str = Field(
        default="en",
        description="ISO 639-1 language code (e.g., 'en', 'ja', 'zh')",
    )
    user_id: str = Field(..., description="User identifier")


class SetCultureRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    culture_group: str = Field(
        ..., description="One of: collectivist, individualist, mixed"
    )


class EIAdjustRequest(BaseModel):
    scores: Dict[str, float] = Field(
        ..., description="EI dimension scores (0-100)"
    )
    culture_group: Optional[str] = Field(
        default=None,
        description="Override culture group (auto-detected from language if omitted)",
    )
    language: str = Field(default="en")


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/voice-emotion/calibrate")
async def calibrate_emotion(req: CalibrateRequest):
    """Apply culture-aware calibration to emotion predictions.

    Takes raw emotion probabilities and adjusts them based on the
    cultural group's expected expression patterns.
    """
    # Check for user override
    culture = _user_culture_overrides.get(req.user_id)
    if not culture:
        culture = get_culture_group(req.language)

    result = _calibrator.calibrate(
        req.probabilities, culture, req.valence, req.arousal
    )
    return result


@router.get("/voice-emotion/languages")
async def list_languages():
    """List supported languages and their cultural group mappings."""
    all_langs = {}
    for lang in sorted(_COLLECTIVIST_LANGS):
        all_langs[lang] = "collectivist"
    for lang in sorted(_INDIVIDUALIST_LANGS):
        all_langs[lang] = "individualist"
    for lang in sorted(_MIXED_LANGS):
        all_langs[lang] = "mixed"

    return {
        "languages": all_langs,
        "culture_groups": {
            name: {
                "description": profile["description"],
                "expression_intensity_scale": profile["expression_intensity_scale"],
                "n_languages": sum(
                    1 for v in all_langs.values() if v == name
                ),
            }
            for name, profile in _CULTURE_PROFILES.items()
        },
        "total_languages": len(all_langs),
    }


@router.post("/voice-emotion/set-culture")
async def set_culture(req: SetCultureRequest):
    """Override auto-detected culture group for a user."""
    if req.culture_group not in _CULTURE_PROFILES:
        return {
            "ok": False,
            "error": f"Invalid culture group. Must be one of: {list(_CULTURE_PROFILES.keys())}",
        }
    _user_culture_overrides[req.user_id] = req.culture_group
    return {
        "ok": True,
        "user_id": req.user_id,
        "culture_group": req.culture_group,
    }


@router.post("/ei/cultural-adjust")
async def adjust_ei(req: EIAdjustRequest):
    """Adjust EI dimension scores for cultural context.

    Normalizes scores against culture-specific norms so that
    collectivist cultures are not penalized for normatively
    lower self-expression scores.
    """
    culture = req.culture_group or get_culture_group(req.language)
    return _ei_adapter.adjust_ei_scores(req.scores, culture)
