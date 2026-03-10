"""Supplement/Medication/Vitamin Tracker API endpoints.

Endpoints
---------
POST /supplements/log
    Log a supplement intake event.

GET  /supplements/log/{user_id}
    Retrieve supplement log entries, optionally filtered by name.

POST /supplements/brain-state
    Log a brain state snapshot (called internally from /analyze-eeg).

GET  /supplements/correlations/{user_id}
    Analyze correlations between a supplement and brain state changes.

GET  /supplements/report/{user_id}
    Full supplement report with per-supplement correlation verdicts.

GET  /supplements/active/{user_id}
    List supplements taken in the last N hours.

DELETE /supplements/reset/{user_id}
    Clear all supplement data for a user.

GET  /supplements/knowledge/{supplement_name}
    Return evidence-based entry for a supplement (CANMAT 2022 + 2024-2025 lit).

GET  /supplements/interactions
    Check synergies/cautions for a set of supplements (?names=omega-3,caffeine).

GET  /supplements/compare/{user_id}/{supplement_name}
    Compare personal observed effects against population-average expected effects.
"""
from __future__ import annotations

import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from models.supplement_tracker import SupplementTracker, VALID_SUPPLEMENT_TYPES
import models.supplement_knowledge as _kb

router = APIRouter(prefix="/supplements", tags=["Supplement Tracker"])

# Module-level singleton
_tracker = SupplementTracker()


def get_tracker() -> SupplementTracker:
    """Return the module-level SupplementTracker singleton."""
    return _tracker


# ── Request models ──────────────────────────────────────────────────


class LogSupplementRequest(BaseModel):
    """Request body for logging a supplement intake."""
    user_id: str = Field(..., description="User identifier")
    name: str = Field(..., description="Supplement name (e.g. 'Omega-3', 'Vitamin D')")
    type: str = Field(
        ...,
        description="Type: vitamin, supplement, medication, or food_supplement",
    )
    dosage: float = Field(..., description="Amount taken")
    unit: str = Field(..., description="Unit of measurement (mg, IU, mcg, etc.)")
    timestamp: Optional[float] = Field(
        default=None,
        description="Unix timestamp. Defaults to current time.",
    )
    notes: Optional[str] = Field(
        default="",
        description="Optional notes about this intake.",
    )


class LogBrainStateRequest(BaseModel):
    """Request body for logging a brain state snapshot."""
    user_id: str = Field(..., description="User identifier")
    timestamp: Optional[float] = Field(
        default=None,
        description="Unix timestamp. Defaults to current time.",
    )
    valence: float = Field(0.0, description="Emotional valence (-1 to 1)")
    arousal: float = Field(0.0, description="Arousal level (0 to 1)")
    stress_index: float = Field(0.0, description="Stress index (0 to 1)")
    focus_index: float = Field(0.0, description="Focus index (0 to 1)")
    alpha_beta_ratio: float = Field(0.0, description="Alpha/beta ratio (relaxation)")
    theta_power: float = Field(0.0, description="Theta power (creativity/drowsiness)")
    faa: float = Field(0.0, description="Frontal alpha asymmetry")


# ── Endpoints ───────────────────────────────────────────────────────


@router.post("/log")
async def log_supplement(req: LogSupplementRequest):
    """Log a supplement, vitamin, or medication intake event.

    Records the intake with timestamp for later correlation analysis
    against EEG-derived brain state metrics.
    """
    if req.type not in VALID_SUPPLEMENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid supplement type '{req.type}'. "
                f"Valid types: {sorted(VALID_SUPPLEMENT_TYPES)}"
            ),
        )

    ts = req.timestamp if req.timestamp is not None else time.time()

    entry_id = _tracker.log_supplement(
        user_id=req.user_id,
        name=req.name,
        supplement_type=req.type,
        dosage=req.dosage,
        unit=req.unit,
        timestamp=ts,
        notes=req.notes or "",
    )
    return {"entry_id": entry_id, "logged_at": ts}


@router.get("/log/{user_id}")
async def get_supplement_log(
    user_id: str,
    last_n: int = 50,
    supplement_name: Optional[str] = None,
):
    """Retrieve supplement log entries for a user.

    Optionally filter by supplement name (case-insensitive).
    Returns the most recent `last_n` entries.
    """
    entries = _tracker.get_log(
        user_id=user_id,
        last_n=last_n,
        supplement_name=supplement_name,
    )
    return {"user_id": user_id, "count": len(entries), "entries": entries}


@router.post("/brain-state")
async def log_brain_state(req: LogBrainStateRequest):
    """Log a brain state snapshot for supplement correlation.

    Called internally by the EEG analysis pipeline to build a
    timeline of brain states for correlation with supplement intake.
    """
    ts = req.timestamp if req.timestamp is not None else time.time()

    _tracker.log_brain_state(
        user_id=req.user_id,
        timestamp=ts,
        emotion_data={
            "valence": req.valence,
            "arousal": req.arousal,
            "stress_index": req.stress_index,
            "focus_index": req.focus_index,
            "alpha_beta_ratio": req.alpha_beta_ratio,
            "theta_power": req.theta_power,
            "faa": req.faa,
        },
    )
    return {"stored": True, "timestamp": ts}


@router.get("/correlations/{user_id}")
async def get_correlations(
    user_id: str,
    supplement_name: str,
    window_hours: float = 4.0,
):
    """Analyze correlations between a supplement and brain state changes.

    Compares brain states in the hours after taking the supplement
    versus control periods when it was not taken. Returns average
    shifts in valence, arousal, stress, focus, and EEG-specific
    metrics (alpha/beta ratio, theta, FAA).
    """
    result = _tracker.analyze_correlations(
        user_id=user_id,
        supplement_name=supplement_name,
        window_hours=window_hours,
    )
    return result


@router.get("/report/{user_id}")
async def get_supplement_report(user_id: str):
    """Generate a full supplement correlation report.

    For each supplement the user takes, computes the overall
    correlation verdict (positive/negative/neutral/insufficient_data)
    against EEG-derived brain metrics.
    """
    return _tracker.get_supplement_report(user_id)


@router.get("/active/{user_id}")
async def get_active_supplements(user_id: str, hours: float = 24.0):
    """List supplements taken within the last N hours.

    Useful for showing which supplements are currently active and
    may be influencing brain state readings.
    """
    active = _tracker.get_active_supplements(user_id, hours=hours)
    return {"user_id": user_id, "hours": hours, "count": len(active), "supplements": active}


@router.delete("/reset/{user_id}")
async def reset_supplement_data(user_id: str):
    """Clear all supplement and brain state data for a user."""
    _tracker.reset(user_id)
    return {"user_id": user_id, "status": "reset"}


# ── Evidence-based knowledge base endpoints ─────────────────────────


@router.get("/knowledge/{supplement_name}")
async def get_supplement_knowledge(supplement_name: str):
    """Return evidence-based entry for a supplement from the knowledge base.

    Provides immediate, research-backed guidance before the user has
    accumulated enough personal EEG data for reliable correlations.
    Source: CANMAT 2022 + 2024-2025 systematic reviews.

    Returns:
        Evidence grade (A/B/C/W), expected effects on valence/stress/focus,
        onset timeline, mechanism, synergies, cautions, and references.
    """
    entry = _kb.lookup(supplement_name)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Supplement '{supplement_name}' not found in knowledge base. "
                f"Known supplements: {sorted(_kb.SUPPLEMENT_DB.keys())}"
            ),
        )
    return {
        "supplement": supplement_name,
        "knowledge": entry,
        "source": "CANMAT 2022 + 2024-2025 systematic reviews",
        "note": (
            "Population-average expected effects. "
            "Use /supplements/compare/{user_id}/{supplement_name} "
            "once you have ≥7 days of EEG data after taking this supplement."
        ),
    }


@router.get("/interactions")
async def check_supplement_interactions(
    names: str = Query(
        ...,
        description="Comma-separated supplement names, e.g. 'omega-3,caffeine,l-theanine'",
    ),
):
    """Check synergies and cautions for a set of supplements taken together.

    Pass a comma-separated list of supplement names via the `names` query param.
    Returns interaction rules that apply when all listed supplements are present.

    Example: GET /supplements/interactions?names=caffeine,l-theanine
    """
    name_list = [n.strip() for n in names.split(",") if n.strip()]
    if len(name_list) < 2:
        raise HTTPException(
            status_code=422,
            detail="Provide at least 2 supplement names separated by commas.",
        )
    result = _kb.check_interactions(name_list)
    return {
        "supplements": name_list,
        "canonical_names": result["canonical_names"],
        "interactions": result["interactions"],
        "interaction_count": result["interaction_count"],
        "count": result["interaction_count"],
        "has_cautions": result["has_cautions"],
        "has_synergies": result["has_synergies"],
    }


@router.get("/compare/{user_id}/{supplement_name}")
async def compare_personal_vs_population(user_id: str, supplement_name: str):
    """Compare personal observed effects vs population-average expected effects.

    Requires at least one supplement log entry and sufficient brain state
    snapshots (≥7 days recommended) to compute a meaningful correlation.

    Returns per-metric comparison (valence, stress_index, focus_index) with:
    - personal: your observed average shift
    - population_avg: clinical average from CANMAT 2022 research
    - direction: above_average / average / below_average
    - note: human-readable interpretation
    """
    # Pull personal correlation data
    correlation = _tracker.analyze_correlations(
        user_id=user_id,
        supplement_name=supplement_name,
    )

    if correlation.get("verdict") == "insufficient_data":
        return {
            "user_id": user_id,
            "supplement": supplement_name,
            "status": "insufficient_data",
            "reason": correlation.get("reason", "not_enough_data"),
            "sample_count_post": correlation.get("sample_count_post", 0),
            "knowledge_base": _kb.lookup(supplement_name),
        }

    # Build personal effects dict from flat correlation result
    personal_effects: dict = {
        "valence": correlation.get("avg_valence_shift", 0.0),
        "stress_index": correlation.get("avg_stress_shift", 0.0),
        "focus_index": correlation.get("avg_focus_shift", 0.0),
    }

    comparison = _kb.population_vs_personal(
        personal_effects, supplement_name=supplement_name
    )

    entry = _kb.lookup(supplement_name)

    return {
        "user_id": user_id,
        "supplement": supplement_name,
        "status": "ok",
        "personal_data_points": correlation.get("sample_count_post", 0),
        "personal_verdict": correlation.get("verdict"),
        "population_comparison": comparison,
        "population_reference": entry.get("expected_effects") if entry else None,
        "evidence_grade": entry.get("evidence_grade") if entry else None,
        "note": (
            "Personal effects are averaged over EEG sessions within the "
            "supplement's onset window. Requires ≥7 days for reliable estimates."
        ),
    }
