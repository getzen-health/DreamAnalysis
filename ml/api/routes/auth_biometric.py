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
    user_id: str = Field(default="default", description="User identifier")


class DeleteTemplateRequest(BaseModel):
    user_id: str = Field(..., description="User ID whose template to delete")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/auth/eeg-enroll")
async def eeg_enroll(request: BiometricEEGRequest):
    """Add an EEG enrollment segment for a user.

    Call ≥3 times with resting-state EEG (eyes closed, 10+ seconds each).
    After 3 segments the template is finalized and verification is available.
    Only frequency-domain templates are stored — raw EEG is never persisted.
    """
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
