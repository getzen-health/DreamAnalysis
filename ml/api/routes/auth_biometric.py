"""EEG biometric authentication API endpoints.

Endpoints:
    POST /auth/eeg-enroll     — Add an EEG enrollment segment for a user
    POST /auth/eeg-verify     — Verify EEG matches enrolled template
    POST /auth/eeg-identify   — 1-of-N: find best-matching enrolled user
    DELETE /auth/eeg-template — Remove stored templates (right-to-erasure)
    GET  /auth/eeg-status     — Enrolled users and template readiness
"""

import logging
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["EEG Biometric Auth"])


# ── Request models ─────────────────────────────────────────────────────────────

class BiometricEEGRequest(BaseModel):
    eeg: List[List[float]] = Field(
        ..., description="EEG data: shape [n_channels, n_samples] or [n_samples]"
    )
    fs: float = Field(default=256.0, gt=0, description="Sampling rate in Hz")
    user_id: str = Field(..., min_length=1, description="User identifier")


class BiometricEnrollRequest(BiometricEEGRequest):
    consent_given: bool = Field(
        ...,
        description=(
            "Must be true. Confirms the user understands that EEG spectral "
            "templates will be stored for identity verification."
        ),
    )


class DeleteTemplateRequest(BaseModel):
    user_id: str = Field(..., description="User ID whose template to delete")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/auth/eeg-enroll")
async def eeg_enroll(request: BiometricEnrollRequest):
    """Add an EEG enrollment segment for a user.

    Call ≥3 times with resting-state EEG (eyes closed, 10+ seconds each).
    After 3 segments the template is finalized and verification is available.
    Only frequency-domain templates are stored — raw EEG is never persisted.

    Requires `consent_given: true` — explicit acknowledgement that EEG spectral
    templates will be stored for identity verification.
    """
    if not request.consent_given:
        raise HTTPException(
            status_code=400,
            detail=(
                "Explicit consent is required for EEG biometric enrollment. "
                "Set consent_given=true to confirm you understand that EEG spectral "
                "templates will be stored for identity verification."
            ),
        )
    try:
        from models.eeg_authenticator import get_eeg_authenticator
        eeg = np.array(request.eeg, dtype=np.float32)
        result = get_eeg_authenticator().enroll(eeg, fs=request.fs, user_id=request.user_id)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/auth/eeg-verify")
async def eeg_verify(request: BiometricEEGRequest):
    """Verify whether the incoming EEG matches the enrolled template.

    Returns match (bool), similarity score (0-1), and the decision threshold.
    Requires at least 3 prior enrollments for the user_id.
    """
    try:
        from models.eeg_authenticator import get_eeg_authenticator
        eeg = np.array(request.eeg, dtype=np.float32)
        result = get_eeg_authenticator().verify(eeg, fs=request.fs, user_id=request.user_id)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/auth/eeg-identify")
async def eeg_identify(request: BiometricEEGRequest):
    """1-of-N identification: find the best-matching enrolled user.

    Returns identified_user (or None if no match above threshold),
    similarity score, and a ranked list of all candidates.
    """
    try:
        from models.eeg_authenticator import get_eeg_authenticator
        eeg = np.array(request.eeg, dtype=np.float32)
        result = get_eeg_authenticator().identify(eeg, fs=request.fs)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/auth/eeg-template")
async def delete_eeg_template(request: DeleteTemplateRequest):
    """Remove all stored biometric templates for the given user (GDPR erasure).

    After deletion the user must re-enroll before verification is possible.
    """
    try:
        from models.eeg_authenticator import get_eeg_authenticator
        result = get_eeg_authenticator().delete_template(request.user_id)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/auth/eeg-status")
async def eeg_auth_status():
    """Return list of enrolled users and which have finalized templates."""
    try:
        from models.eeg_authenticator import get_eeg_authenticator
        return get_eeg_authenticator().get_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/auth/eeg-audit")
async def get_biometric_audit(user_id: Optional[str] = None, limit: int = 100):
    """Return biometric system audit log.

    Pass `user_id` to filter to a specific user.
    `limit` caps how many of the most-recent events are returned (default 100).
    """
    from models.eeg_authenticator import _biometric_audit_log

    log = _biometric_audit_log
    if user_id:
        log = [e for e in log if e["user_id"] == user_id]
    return {"events": log[-limit:], "total": len(log)}
