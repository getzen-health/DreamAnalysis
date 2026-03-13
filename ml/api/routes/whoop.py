"""Whoop API v2 integration routes.

GET  /whoop/auth/url             — build OAuth2 authorization URL
POST /whoop/auth/exchange        — exchange auth code for tokens
POST /whoop/auth/refresh         — refresh an access token
GET  /whoop/sync/{user_id}       — pull latest metrics and return biometric payload
GET  /whoop/recovery/{user_id}   — latest recovery record (parsed)
GET  /whoop/sleep/{user_id}      — latest sleep record (parsed)
GET  /whoop/cycle/{user_id}      — latest cycle/strain record (parsed)
GET  /whoop/status               — check OAuth env var configuration
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/whoop", tags=["whoop"])


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class ExchangeRequest(BaseModel):
    code: str = Field(..., description="Authorization code from Whoop OAuth2 callback")
    redirect_uri: Optional[str] = Field(
        None, description="Must match the URI used during authorization"
    )


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., description="Whoop refresh token")


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    token_type: Optional[str] = None
    scope: Optional[str] = None


class BiometricPayload(BaseModel):
    user_id: str
    source: str = "whoop"
    hrv_rmssd: Optional[float] = None
    resting_heart_rate: Optional[float] = None
    recovery_score: Optional[float] = None
    sleep_performance_pct: Optional[float] = None
    sleep_total_hours: Optional[float] = None
    sleep_efficiency: Optional[float] = None
    sleep_rem_hours: Optional[float] = None
    sleep_deep_hours: Optional[float] = None
    strain_score: Optional[float] = None
    active_energy_kcal: Optional[float] = None
    skin_temperature_deviation: Optional[float] = None
    fetched_at: Optional[float] = None


# ── Helper ─────────────────────────────────────────────────────────────────────

def _get_whoop_client(access_token: str):
    try:
        from health.whoop_sync import WhoopClient
        return WhoopClient(access_token)
    except Exception as exc:
        raise HTTPException(500, f"Failed to initialize Whoop client: {exc}")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/status")
def whoop_status() -> Dict[str, Any]:
    """Return whether Whoop OAuth credentials are configured."""
    client_id_set = bool(os.environ.get("WHOOP_CLIENT_ID"))
    client_secret_set = bool(os.environ.get("WHOOP_CLIENT_SECRET"))
    redirect_uri_set = bool(os.environ.get("WHOOP_REDIRECT_URI"))
    return {
        "configured": client_id_set and client_secret_set and redirect_uri_set,
        "client_id_set": client_id_set,
        "client_secret_set": client_secret_set,
        "redirect_uri_set": redirect_uri_set,
    }


@router.get("/auth/url")
def whoop_auth_url(
    state: Optional[str] = Query(None, description="CSRF state token"),
) -> Dict[str, str]:
    """Build the Whoop OAuth2 authorization URL.

    Redirect the user's browser to the returned URL to initiate login.
    """
    try:
        from health.whoop_sync import authorization_url
        url = authorization_url(state=state)
        return {"authorization_url": url}
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Failed to build authorization URL: {exc}")


@router.post("/auth/exchange", response_model=TokenResponse)
def whoop_exchange_code(req: ExchangeRequest) -> Dict[str, Any]:
    """Exchange a Whoop authorization code for access + refresh tokens."""
    try:
        from health.whoop_sync import exchange_code
        tokens = exchange_code(code=req.code, redirect_uri=req.redirect_uri)
        return tokens
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Token exchange failed: {exc}")


@router.post("/auth/refresh", response_model=TokenResponse)
def whoop_refresh_token(req: RefreshRequest) -> Dict[str, Any]:
    """Refresh an expired Whoop access token."""
    try:
        from health.whoop_sync import refresh_tokens
        tokens = refresh_tokens(req.refresh_token)
        return tokens
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Token refresh failed: {exc}")


@router.get("/sync/{user_id}", response_model=BiometricPayload)
def whoop_sync(
    user_id: str,
    access_token: str = Query(..., description="Whoop access token"),
) -> Dict[str, Any]:
    """Pull all available Whoop metrics for a user and return a biometric payload.

    The payload is in the same format as POST /biometrics/update so it can
    be passed directly to the MultimodalEmotionFusion pipeline.
    """
    client = _get_whoop_client(access_token)
    try:
        payload = client.build_biometric_payload(user_id)
        return payload
    except RuntimeError as exc:
        raise HTTPException(502, f"Whoop API error: {exc}")
    except Exception as exc:
        log.error("Whoop sync failed for user %s: %s", user_id, exc)
        raise HTTPException(500, f"Whoop sync failed: {exc}")


@router.get("/recovery/{user_id}")
def whoop_recovery(
    user_id: str,
    access_token: str = Query(..., description="Whoop access token"),
) -> Dict[str, Any]:
    """Return the latest Whoop recovery record for a user (parsed metrics)."""
    client = _get_whoop_client(access_token)
    try:
        from health.whoop_sync import parse_recovery
        record = client.get_latest_recovery()
        if record is None:
            return {"user_id": user_id, "data": None, "message": "No recovery data"}
        return {"user_id": user_id, "data": parse_recovery(record)}
    except RuntimeError as exc:
        raise HTTPException(502, f"Whoop API error: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Recovery fetch failed: {exc}")


@router.get("/sleep/{user_id}")
def whoop_sleep(
    user_id: str,
    access_token: str = Query(..., description="Whoop access token"),
) -> Dict[str, Any]:
    """Return the latest Whoop sleep record for a user (parsed metrics)."""
    client = _get_whoop_client(access_token)
    try:
        from health.whoop_sync import parse_sleep
        record = client.get_latest_sleep()
        if record is None:
            return {"user_id": user_id, "data": None, "message": "No sleep data"}
        return {"user_id": user_id, "data": parse_sleep(record)}
    except RuntimeError as exc:
        raise HTTPException(502, f"Whoop API error: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Sleep fetch failed: {exc}")


@router.get("/cycle/{user_id}")
def whoop_cycle(
    user_id: str,
    access_token: str = Query(..., description="Whoop access token"),
) -> Dict[str, Any]:
    """Return the latest Whoop cycle/strain record for a user (parsed metrics)."""
    client = _get_whoop_client(access_token)
    try:
        from health.whoop_sync import parse_cycle
        record = client.get_latest_cycle()
        if record is None:
            return {"user_id": user_id, "data": None, "message": "No cycle data"}
        return {"user_id": user_id, "data": parse_cycle(record)}
    except RuntimeError as exc:
        raise HTTPException(502, f"Whoop API error: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Cycle fetch failed: {exc}")
