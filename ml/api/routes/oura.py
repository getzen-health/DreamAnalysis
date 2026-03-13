"""Oura Ring API v2 integration routes.

GET  /oura/auth/url              — build OAuth2 authorization URL
POST /oura/auth/exchange         — exchange auth code for tokens
POST /oura/auth/refresh          — refresh an access token
GET  /oura/sync/{user_id}        — pull latest metrics and return biometric payload
GET  /oura/readiness/{user_id}   — latest readiness record (parsed)
GET  /oura/sleep/{user_id}       — latest sleep records (parsed)
GET  /oura/activity/{user_id}    — latest activity record (parsed)
GET  /oura/status                — check OAuth env var configuration
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/oura", tags=["oura"])


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class ExchangeRequest(BaseModel):
    code: str = Field(..., description="Authorization code from Oura OAuth2 callback")
    redirect_uri: Optional[str] = Field(
        None, description="Must match the URI used during authorization"
    )


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., description="Oura refresh token")


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    token_type: Optional[str] = None


class BiometricPayload(BaseModel):
    user_id: str
    source: str = "oura"
    readiness_score: Optional[int] = None
    sleep_score: Optional[int] = None
    activity_score: Optional[int] = None
    hrv_rmssd: Optional[float] = None
    hrv_balance_index: Optional[float] = None
    resting_heart_rate: Optional[float] = None
    skin_temperature_deviation: Optional[float] = None
    respiratory_rate: Optional[float] = None
    sleep_total_hours: Optional[float] = None
    sleep_rem_hours: Optional[float] = None
    sleep_deep_hours: Optional[float] = None
    sleep_efficiency: Optional[float] = None
    steps_today: Optional[int] = None
    active_energy_kcal: Optional[float] = None
    fetched_at: Optional[float] = None


# ── Helper ─────────────────────────────────────────────────────────────────────

def _get_oura_client(access_token: str):
    try:
        from health.oura_sync import OuraClient
        return OuraClient(access_token)
    except Exception as exc:
        raise HTTPException(500, f"Failed to initialize Oura client: {exc}")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/status")
def oura_status() -> Dict[str, Any]:
    """Return whether Oura OAuth credentials are configured."""
    client_id_set = bool(os.environ.get("OURA_CLIENT_ID"))
    client_secret_set = bool(os.environ.get("OURA_CLIENT_SECRET"))
    redirect_uri_set = bool(os.environ.get("OURA_REDIRECT_URI"))
    return {
        "configured": client_id_set and client_secret_set and redirect_uri_set,
        "client_id_set": client_id_set,
        "client_secret_set": client_secret_set,
        "redirect_uri_set": redirect_uri_set,
    }


@router.get("/auth/url")
def oura_auth_url(
    state: Optional[str] = Query(None, description="CSRF state token"),
) -> Dict[str, str]:
    """Build the Oura OAuth2 authorization URL.

    Redirect the user's browser to the returned URL to initiate login.
    """
    try:
        from health.oura_sync import authorization_url
        url = authorization_url(state=state)
        return {"authorization_url": url}
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Failed to build authorization URL: {exc}")


@router.post("/auth/exchange", response_model=TokenResponse)
def oura_exchange_code(req: ExchangeRequest) -> Dict[str, Any]:
    """Exchange an Oura authorization code for access + refresh tokens."""
    try:
        from health.oura_sync import exchange_code
        tokens = exchange_code(code=req.code, redirect_uri=req.redirect_uri)
        return tokens
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Token exchange failed: {exc}")


@router.post("/auth/refresh", response_model=TokenResponse)
def oura_refresh_token(req: RefreshRequest) -> Dict[str, Any]:
    """Refresh an expired Oura access token."""
    try:
        from health.oura_sync import refresh_tokens
        tokens = refresh_tokens(req.refresh_token)
        return tokens
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Token refresh failed: {exc}")


@router.get("/sync/{user_id}", response_model=BiometricPayload)
def oura_sync(
    user_id: str,
    access_token: str = Query(..., description="Oura access token"),
) -> Dict[str, Any]:
    """Pull all available Oura metrics for a user and return a biometric payload.

    The payload format matches POST /biometrics/update so it can be passed
    directly to the MultimodalEmotionFusion pipeline.
    """
    client = _get_oura_client(access_token)
    try:
        payload = client.build_biometric_payload(user_id)
        return payload
    except RuntimeError as exc:
        raise HTTPException(502, f"Oura API error: {exc}")
    except Exception as exc:
        log.error("Oura sync failed for user %s: %s", user_id, exc)
        raise HTTPException(500, f"Oura sync failed: {exc}")


@router.get("/readiness/{user_id}")
def oura_readiness(
    user_id: str,
    access_token: str = Query(..., description="Oura access token"),
    days: int = Query(1, description="Number of recent days to return (max 7)"),
) -> Dict[str, Any]:
    """Return recent Oura daily readiness records for a user (parsed metrics)."""
    client = _get_oura_client(access_token)
    days = min(max(1, days), 7)
    try:
        from health.oura_sync import parse_readiness
        records = client.get_daily_readiness(
            start_date=client._n_days_ago_str(days - 1),
            end_date=client._today_str(),
        )
        parsed = [parse_readiness(r) for r in records]
        return {"user_id": user_id, "count": len(parsed), "data": parsed}
    except RuntimeError as exc:
        raise HTTPException(502, f"Oura API error: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Readiness fetch failed: {exc}")


@router.get("/sleep/{user_id}")
def oura_sleep(
    user_id: str,
    access_token: str = Query(..., description="Oura access token"),
    days: int = Query(1, description="Number of recent days to return (max 7)"),
) -> Dict[str, Any]:
    """Return recent Oura sleep session records for a user (parsed metrics).

    Returns detailed per-session metrics including HRV, resting HR, and
    temperature — not just the daily summary score.
    """
    client = _get_oura_client(access_token)
    days = min(max(1, days), 7)
    try:
        from health.oura_sync import parse_sleep_session
        sessions = client.get_sleep_sessions(
            start_date=client._n_days_ago_str(days),
            end_date=client._today_str(),
        )
        parsed = [parse_sleep_session(s) for s in sessions]
        return {"user_id": user_id, "count": len(parsed), "data": parsed}
    except RuntimeError as exc:
        raise HTTPException(502, f"Oura API error: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Sleep fetch failed: {exc}")


@router.get("/activity/{user_id}")
def oura_activity(
    user_id: str,
    access_token: str = Query(..., description="Oura access token"),
    days: int = Query(1, description="Number of recent days to return (max 7)"),
) -> Dict[str, Any]:
    """Return recent Oura daily activity records for a user (parsed metrics)."""
    client = _get_oura_client(access_token)
    days = min(max(1, days), 7)
    try:
        from health.oura_sync import parse_activity
        records = client.get_daily_activity(
            start_date=client._n_days_ago_str(days - 1),
            end_date=client._today_str(),
        )
        parsed = [parse_activity(r) for r in records]
        return {"user_id": user_id, "count": len(parsed), "data": parsed}
    except RuntimeError as exc:
        raise HTTPException(502, f"Oura API error: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Activity fetch failed: {exc}")
