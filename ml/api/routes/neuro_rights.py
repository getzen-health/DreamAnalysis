"""Neuro-rights governance API routes.

Endpoints:
  POST /neuro-rights/audit              -- audit rights compliance for a user
  POST /neuro-rights/consent            -- log a consent grant or revocation
  GET  /neuro-rights/inventory/{user_id} -- data sovereignty inventory
  GET  /neuro-rights/status             -- framework availability check

GitHub issue: #443
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.neuro_rights import (
    CONSENT_ACTIONS,
    NEURAL_DATA_TYPES,
    NEURO_RIGHTS_DESCRIPTIONS,
    NeuroRight,
    audit_rights_compliance,
    check_data_minimization,
    compute_data_inventory,
    compute_deletion_impact,
    compute_governance_report,
    generate_explanation,
    log_consent,
    report_to_dict,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/neuro-rights", tags=["neuro-rights"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class AuditRequest(BaseModel):
    user_id: str = Field(..., description="User to audit")
    data_flows: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "Optional explicit data flows to audit. "
            "If omitted, uses internally tracked flows."
        ),
    )


class ConsentRequest(BaseModel):
    user_id: str = Field(..., description="User granting or revoking consent")
    data_type: str = Field(
        ...,
        description=f"Neural data type. One of: {sorted(NEURAL_DATA_TYPES)}",
    )
    action: str = Field(
        ...,
        description="Consent action: 'grant' or 'revoke'",
    )
    purpose: str = Field(
        "analysis",
        description="Purpose of the consent grant/revocation",
    )
    granted_to: str = Field(
        "system",
        description="Entity receiving (or losing) consent",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/audit")
async def audit_rights(request: AuditRequest):
    """Audit neuro-rights compliance for a user.

    Checks all five neuro-rights (mental privacy, cognitive liberty,
    mental integrity, psychological continuity, fair access) against
    the user's data flows and consent state.

    Returns a compliance report with per-right status and violations.
    """
    try:
        report = audit_rights_compliance(
            request.user_id,
            data_flows=request.data_flows,
        )
        return report_to_dict(report)
    except Exception as exc:
        log.exception("Rights audit failed for user %s", request.user_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/consent")
async def record_consent(request: ConsentRequest):
    """Log a consent grant or revocation.

    Records an immutable entry in the consent ledger with a SHA-256 hash
    for integrity verification. Updates the active consent state for
    the user and data type.
    """
    if request.action not in CONSENT_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action: {request.action}. Must be one of: {sorted(CONSENT_ACTIONS)}",
        )
    if request.data_type not in NEURAL_DATA_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown data type: {request.data_type}. Valid types: {sorted(NEURAL_DATA_TYPES)}",
        )

    try:
        entry = log_consent(
            user_id=request.user_id,
            data_type=request.data_type,
            action=request.action,
            purpose=request.purpose,
            granted_to=request.granted_to,
        )
        return {
            "status": "ok",
            "consent_entry": entry,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Consent logging failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/inventory/{user_id}")
async def get_inventory(user_id: str):
    """Get the data sovereignty inventory for a user.

    Returns what neural data exists, who has accessed it, when it was
    last accessed, and the current consent status for each data type.
    """
    try:
        inventory = compute_data_inventory(user_id)
        return inventory
    except Exception as exc:
        log.exception("Inventory computation failed for user %s", user_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/status")
async def get_status():
    """Check neuro-rights governance framework availability.

    Returns framework version, supported rights, and valid data types.
    """
    return {
        "status": "available",
        "framework": "Neuro-Rights Governance (Yuste & Goering 2017)",
        "version": "1.0.0",
        "rights": {r.value: NEURO_RIGHTS_DESCRIPTIONS[r] for r in NeuroRight},
        "supported_data_types": sorted(NEURAL_DATA_TYPES),
        "consent_actions": sorted(CONSENT_ACTIONS),
        "endpoints": [
            "POST /neuro-rights/audit",
            "POST /neuro-rights/consent",
            "GET /neuro-rights/inventory/{user_id}",
            "GET /neuro-rights/status",
        ],
    }
