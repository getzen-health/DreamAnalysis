"""Garmin Connect API integration routes.

GET  /garmin/status                — check OAuth 1.0a env var configuration
GET  /garmin/auth/url              — get OAuth 1.0a request token + authorization URL
POST /garmin/auth/exchange         — exchange OAuth verifier for access token
POST /garmin/webhook               — receive Garmin data push (webhook)
GET  /garmin/sync/{user_id}        — manual sync using stored tokens
GET  /garmin/body-battery/{user_id} — latest body battery reading
GET  /garmin/stress/{user_id}      — latest stress data
GET  /garmin/sleep/{user_id}       — latest sleep summary
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/garmin", tags=["garmin"])


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class RequestTokenResponse(BaseModel):
    oauth_token: str
    oauth_token_secret: str
    authorization_url: str


class ExchangeRequest(BaseModel):
    oauth_token: str = Field(..., description="Temporary request token from /garmin/auth/url")
    oauth_verifier: str = Field(..., description="Verifier code from Garmin callback")
    oauth_token_secret: str = Field(..., description="Request token secret from /garmin/auth/url")


class AccessTokenResponse(BaseModel):
    oauth_token: str
    oauth_token_secret: str


class GarminBiometricPayload(BaseModel):
    user_id: str
    source: str = "garmin"
    body_battery_highest: Optional[int] = None
    body_battery_lowest: Optional[int] = None
    body_battery_charged: Optional[int] = None
    body_battery_drained: Optional[int] = None
    average_stress: Optional[int] = None
    max_stress: Optional[int] = None
    resting_heart_rate: Optional[float] = None
    hrv_last_night: Optional[int] = None
    hrv_weekly_average: Optional[int] = None
    hrv_status: Optional[str] = None
    sleep_total_hours: Optional[float] = None
    sleep_deep_hours: Optional[float] = None
    sleep_light_hours: Optional[float] = None
    sleep_rem_hours: Optional[float] = None
    sleep_score: Optional[int] = None
    average_spo2: Optional[float] = None
    average_respiration: Optional[float] = None
    steps_today: Optional[int] = None
    active_energy_kcal: Optional[float] = None
    fetched_at: Optional[float] = None


# ── Helper ─────────────────────────────────────────────────────────────────────

def _get_garmin_client(access_token: str, access_token_secret: str):
    try:
        from health.garmin_sync import GarminClient
        return GarminClient(
            access_token=access_token,
            access_token_secret=access_token_secret,
        )
    except (ValueError, ImportError) as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Failed to initialize Garmin client: {exc}")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/status")
def garmin_status() -> Dict[str, Any]:
    """Return whether Garmin OAuth 1.0a credentials are configured."""
    consumer_key_set = bool(os.environ.get("GARMIN_CONSUMER_KEY"))
    consumer_secret_set = bool(os.environ.get("GARMIN_CONSUMER_SECRET"))
    callback_url_set = bool(os.environ.get("GARMIN_CALLBACK_URL"))
    return {
        "configured": consumer_key_set and consumer_secret_set,
        "consumer_key_set": consumer_key_set,
        "consumer_secret_set": consumer_secret_set,
        "callback_url_set": callback_url_set,
        "oauth_version": "1.0a",
    }


@router.get("/auth/url", response_model=RequestTokenResponse)
def garmin_auth_url() -> Dict[str, Any]:
    """Fetch a Garmin OAuth 1.0a request token and return the authorization URL.

    Step 1 of the OAuth 1.0a flow. The caller should:
      1. Store oauth_token and oauth_token_secret from the response.
      2. Redirect the user to authorization_url.
      3. After the user authorizes, Garmin calls back with oauth_token + oauth_verifier.
      4. Exchange those via POST /garmin/auth/exchange.
    """
    try:
        from health.garmin_sync import get_request_token, get_authorization_url
        request_token = get_request_token()
        auth_url = get_authorization_url(oauth_token=request_token["oauth_token"])
        return {
            "oauth_token": request_token["oauth_token"],
            "oauth_token_secret": request_token["oauth_token_secret"],
            "authorization_url": auth_url,
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except ImportError as exc:
        raise HTTPException(500, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Failed to get Garmin request token: {exc}")


@router.post("/auth/exchange", response_model=AccessTokenResponse)
def garmin_exchange_verifier(req: ExchangeRequest) -> Dict[str, Any]:
    """Exchange a Garmin OAuth verifier for a permanent access token.

    Step 3 of the OAuth 1.0a flow. Store the returned oauth_token and
    oauth_token_secret — these are the permanent credentials used for
    all subsequent API calls and webhook syncs.
    """
    try:
        from health.garmin_sync import exchange_verifier
        tokens = exchange_verifier(
            oauth_token=req.oauth_token,
            oauth_verifier=req.oauth_verifier,
            oauth_token_secret=req.oauth_token_secret,
        )
        return {
            "oauth_token": tokens["oauth_token"],
            "oauth_token_secret": tokens["oauth_token_secret"],
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Garmin verifier exchange failed: {exc}")


@router.post("/webhook")
async def garmin_webhook(request: Request) -> Dict[str, Any]:
    """Receive and parse a Garmin data push webhook.

    Garmin pushes health data (dailies, sleeps, stress, HRV, etc.) to this
    endpoint rather than requiring polling. Register this URL in the Garmin
    Health API developer console as your webhook callback.

    Returns a summary of the data types received and record counts.
    """
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(400, f"Invalid JSON in webhook payload: {exc}")

    try:
        from health.garmin_sync import parse_webhook_payload
        normalized = parse_webhook_payload(body)
    except Exception as exc:
        log.error("Garmin webhook parsing failed: %s", exc)
        raise HTTPException(500, f"Webhook parsing failed: {exc}")

    # Build a summary of what was received
    summary: Dict[str, int] = {}
    for key, value in normalized.items():
        if isinstance(value, list):
            summary[key] = len(value)
        else:
            summary[key] = 1

    log.info("Garmin webhook received: %s", summary)
    return {
        "status": "accepted",
        "received": summary,
        "data": normalized,
    }


@router.get("/sync/{user_id}", response_model=GarminBiometricPayload)
def garmin_sync(
    user_id: str,
    access_token: str = Query(..., description="Garmin OAuth 1.0a access token"),
    access_token_secret: str = Query(..., description="Garmin OAuth 1.0a access token secret"),
) -> Dict[str, Any]:
    """Pull all available Garmin metrics for a user and return a biometric payload.

    Requires a valid OAuth 1.0a access token + secret obtained via the
    /garmin/auth/url → /garmin/auth/exchange flow.

    The payload format matches POST /biometrics/update so it can be passed
    directly to the MultimodalEmotionFusion pipeline.
    """
    client = _get_garmin_client(access_token, access_token_secret)
    try:
        payload = client.build_biometric_payload(user_id)
        return payload
    except RuntimeError as exc:
        raise HTTPException(502, f"Garmin API error: {exc}")
    except Exception as exc:
        log.error("Garmin sync failed for user %s: %s", user_id, exc)
        raise HTTPException(500, f"Garmin sync failed: {exc}")


@router.get("/body-battery/{user_id}")
def garmin_body_battery(
    user_id: str,
    access_token: str = Query(..., description="Garmin OAuth 1.0a access token"),
    access_token_secret: str = Query(..., description="Garmin OAuth 1.0a access token secret"),
) -> Dict[str, Any]:
    """Return the latest body battery reading for a user.

    Body battery is Garmin's proprietary energy metric (0-100) derived from
    HRV, sleep, stress, and activity. High values indicate full energy reserve;
    low values indicate depletion.
    """
    client = _get_garmin_client(access_token, access_token_secret)
    try:
        battery = client.get_body_battery()
        if battery is None:
            return {"user_id": user_id, "data": None, "message": "No body battery data"}
        return {"user_id": user_id, "data": battery}
    except RuntimeError as exc:
        raise HTTPException(502, f"Garmin API error: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Body battery fetch failed: {exc}")


@router.get("/stress/{user_id}")
def garmin_stress(
    user_id: str,
    access_token: str = Query(..., description="Garmin OAuth 1.0a access token"),
    access_token_secret: str = Query(..., description="Garmin OAuth 1.0a access token secret"),
) -> Dict[str, Any]:
    """Return recent stress detail records for a user.

    Garmin provides continuous stress monitoring (0-100) based on HRV analysis.
    Ranges: 0-25 = rest, 26-50 = low, 51-75 = medium, 76-100 = high stress.
    """
    client = _get_garmin_client(access_token, access_token_secret)
    try:
        stress_records = client.get_stress_data()
        return {
            "user_id": user_id,
            "count": len(stress_records),
            "data": stress_records,
        }
    except RuntimeError as exc:
        raise HTTPException(502, f"Garmin API error: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Stress fetch failed: {exc}")


@router.get("/sleep/{user_id}")
def garmin_sleep(
    user_id: str,
    access_token: str = Query(..., description="Garmin OAuth 1.0a access token"),
    access_token_secret: str = Query(..., description="Garmin OAuth 1.0a access token secret"),
) -> Dict[str, Any]:
    """Return recent sleep summary records for a user (parsed metrics).

    Includes sleep stages (deep/light/REM), SpO2, respiration rate, average
    stress during sleep, and the Garmin sleep score (0-100).
    """
    client = _get_garmin_client(access_token, access_token_secret)
    try:
        from health.garmin_sync import parse_sleep
        raw_sleeps = client.get_sleep_data()
        parsed = [parse_sleep(s) for s in raw_sleeps]
        return {"user_id": user_id, "count": len(parsed), "data": parsed}
    except RuntimeError as exc:
        raise HTTPException(502, f"Garmin API error: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Sleep fetch failed: {exc}")
