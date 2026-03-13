"""Whoop API v2 integration — OAuth2 + health metric sync.

Pulls recovery score, strain, sleep performance, HRV, and resting HR
from the Whoop v2 API and maps them to the BiometricPayload schema used
by the ML backend.

OAuth2 flow:
  1. Direct user to authorization_url() to get an authorization code.
  2. Exchange code for tokens via exchange_code().
  3. Use WhoopClient(access_token) for data requests.
  4. Refresh tokens automatically via refresh_tokens().

API base: https://api.prod.whoop.com/developer/v1
Docs: https://developer.whoop.com/api

Environment variables:
  WHOOP_CLIENT_ID     — OAuth2 client ID
  WHOOP_CLIENT_SECRET — OAuth2 client secret
  WHOOP_REDIRECT_URI  — Redirect URI registered with Whoop
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

log = logging.getLogger(__name__)

_WHOOP_AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
_WHOOP_TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
_WHOOP_API_BASE = "https://api.prod.whoop.com/developer/v1"

# Scopes required for the metrics we pull
_REQUIRED_SCOPES = [
    "read:recovery",
    "read:cycles",
    "read:sleep",
    "read:profile",
    "read:body_measurement",
    "offline",  # needed for refresh tokens
]


# ── Internal health metric format ─────────────────────────────────────────────

def _empty_whoop_payload(user_id: str) -> Dict[str, Any]:
    """Return skeleton BiometricPayload with whoop provenance."""
    return {
        "user_id": user_id,
        "source": "whoop",
        # All metric fields are optional — only set when data is available
    }


# ── OAuth2 helpers ─────────────────────────────────────────────────────────────

def authorization_url(
    client_id: Optional[str] = None,
    redirect_uri: Optional[str] = None,
    state: Optional[str] = None,
) -> str:
    """Build the Whoop OAuth2 authorization URL.

    Args:
        client_id: Whoop client ID (falls back to WHOOP_CLIENT_ID env var).
        redirect_uri: Redirect URI (falls back to WHOOP_REDIRECT_URI env var).
        state: CSRF state token (optional but recommended).

    Returns:
        Full authorization URL to redirect the user to.

    Raises:
        ValueError: if client_id or redirect_uri cannot be resolved.
    """
    cid = client_id or os.environ.get("WHOOP_CLIENT_ID")
    ruri = redirect_uri or os.environ.get("WHOOP_REDIRECT_URI")
    if not cid:
        raise ValueError("WHOOP_CLIENT_ID is not set")
    if not ruri:
        raise ValueError("WHOOP_REDIRECT_URI is not set")

    params: Dict[str, str] = {
        "client_id": cid,
        "redirect_uri": ruri,
        "response_type": "code",
        "scope": " ".join(_REQUIRED_SCOPES),
    }
    if state:
        params["state"] = state

    return f"{_WHOOP_AUTH_URL}?{urlencode(params)}"


def exchange_code(
    code: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    redirect_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Exchange an authorization code for access + refresh tokens.

    Args:
        code: Authorization code from the OAuth2 callback.
        client_id, client_secret, redirect_uri: Override env vars.

    Returns:
        Token response dict: {access_token, refresh_token, expires_in, ...}

    Raises:
        RuntimeError: if the token request fails.
    """
    import urllib.request
    import json as _json

    cid = client_id or os.environ.get("WHOOP_CLIENT_ID")
    csecret = client_secret or os.environ.get("WHOOP_CLIENT_SECRET")
    ruri = redirect_uri or os.environ.get("WHOOP_REDIRECT_URI")

    if not cid or not csecret:
        raise ValueError("WHOOP_CLIENT_ID and WHOOP_CLIENT_SECRET must be set")
    if not ruri:
        raise ValueError("WHOOP_REDIRECT_URI must be set")

    body = urlencode({
        "grant_type": "authorization_code",
        "client_id": cid,
        "client_secret": csecret,
        "redirect_uri": ruri,
        "code": code,
    }).encode()

    req = urllib.request.Request(
        _WHOOP_TOKEN_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return _json.loads(resp.read())
    except Exception as exc:
        raise RuntimeError(f"Whoop token exchange failed: {exc}") from exc


def refresh_tokens(
    refresh_token: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> Dict[str, Any]:
    """Refresh an expired access token using the refresh token.

    Returns:
        New token response dict with updated access_token + refresh_token.

    Raises:
        RuntimeError: if the refresh request fails.
    """
    import urllib.request
    import json as _json

    cid = client_id or os.environ.get("WHOOP_CLIENT_ID")
    csecret = client_secret or os.environ.get("WHOOP_CLIENT_SECRET")

    if not cid or not csecret:
        raise ValueError("WHOOP_CLIENT_ID and WHOOP_CLIENT_SECRET must be set")

    body = urlencode({
        "grant_type": "refresh_token",
        "client_id": cid,
        "client_secret": csecret,
        "refresh_token": refresh_token,
        "scope": " ".join(_REQUIRED_SCOPES),
    }).encode()

    req = urllib.request.Request(
        _WHOOP_TOKEN_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return _json.loads(resp.read())
    except Exception as exc:
        raise RuntimeError(f"Whoop token refresh failed: {exc}") from exc


# ── API client ─────────────────────────────────────────────────────────────────

class WhoopClient:
    """Authenticated client for Whoop API v1 (v2 public release).

    Usage:
        client = WhoopClient(access_token="...")
        recovery = client.get_latest_recovery()
        payload = client.build_biometric_payload(user_id="u1")
    """

    def __init__(self, access_token: str) -> None:
        if not access_token:
            raise ValueError("access_token must not be empty")
        self._token = access_token

    # ── Low-level HTTP helper ──────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[Dict[str, str]] = None) -> Any:
        """GET {_WHOOP_API_BASE}/{path} with Bearer auth.

        Raises:
            RuntimeError: on non-2xx HTTP or network error.
        """
        import urllib.request
        import urllib.parse
        import json as _json

        url = f"{_WHOOP_API_BASE}/{path.lstrip('/')}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return _json.loads(resp.read())
        except Exception as exc:
            raise RuntimeError(f"Whoop API GET {path} failed: {exc}") from exc

    # ── Data fetchers ──────────────────────────────────────────────────────────

    def get_profile(self) -> Dict[str, Any]:
        """Return authenticated user's Whoop profile."""
        return self._get("/user/profile/basic")

    def get_latest_recovery(self) -> Optional[Dict[str, Any]]:
        """Return the most recent recovery record.

        Returns None if no recovery data exists.
        """
        data = self._get("/recovery", {"limit": "1"})
        records: List[Dict] = data.get("records", [])
        return records[0] if records else None

    def get_latest_sleep(self) -> Optional[Dict[str, Any]]:
        """Return the most recent sleep record."""
        data = self._get("/activity/sleep", {"limit": "1"})
        records: List[Dict] = data.get("records", [])
        return records[0] if records else None

    def get_latest_cycle(self) -> Optional[Dict[str, Any]]:
        """Return the most recent physiological cycle (strain)."""
        data = self._get("/cycle", {"limit": "1"})
        records: List[Dict] = data.get("records", [])
        return records[0] if records else None

    def get_latest_workout(self) -> Optional[Dict[str, Any]]:
        """Return the most recent workout record."""
        data = self._get("/activity/workout", {"limit": "1"})
        records: List[Dict] = data.get("records", [])
        return records[0] if records else None

    # ── Metric extraction ──────────────────────────────────────────────────────

    def build_biometric_payload(self, user_id: str) -> Dict[str, Any]:
        """Pull all available Whoop metrics and map to BiometricPayload schema.

        Fields populated (when available):
          - hrv_sdnn: HRV from recovery record (ms)
          - resting_heart_rate: resting HR from recovery record (bpm)
          - recovery_score: 0-100 Whoop recovery score
          - sleep_total_hours: total sleep duration
          - sleep_performance_pct: Whoop sleep performance percentage (0-100)
          - strain_score: 0-21 Whoop day strain
          - skin_temperature_deviation: deviation from personal baseline (°C)

        Returns:
            BiometricPayload dict with all available fields set.
        """
        payload = _empty_whoop_payload(user_id)

        # Recovery → HRV, resting HR, recovery score
        try:
            recovery = self.get_latest_recovery()
            if recovery:
                score = recovery.get("score", {}) or {}
                hrv = score.get("hrv_rmssd_milli")
                if hrv is not None and hrv > 0:
                    payload["hrv_rmssd"] = float(hrv)
                rhr = score.get("resting_heart_rate")
                if rhr is not None and rhr > 0:
                    payload["resting_heart_rate"] = float(rhr)
                rec_score = score.get("recovery_score")
                if rec_score is not None:
                    payload["recovery_score"] = float(rec_score)
                skin_temp = score.get("skin_temp_celsius")
                if skin_temp is not None:
                    # Whoop provides deviation from personal baseline
                    payload["skin_temperature_deviation"] = float(skin_temp)
        except Exception as exc:
            log.warning("Whoop recovery fetch failed: %s", exc)

        # Sleep → duration, performance
        try:
            sleep = self.get_latest_sleep()
            if sleep:
                score = sleep.get("score", {}) or {}
                perf = score.get("sleep_performance_percentage")
                if perf is not None:
                    payload["sleep_performance_pct"] = float(perf)
                stage_summary = sleep.get("score", {}).get("stage_summary", {}) or {}
                total_ms = stage_summary.get("total_in_bed_time_milli", 0)
                if total_ms and total_ms > 0:
                    payload["sleep_total_hours"] = round(total_ms / 3_600_000, 2)
                awake_ms = stage_summary.get("total_awake_time_milli", 0)
                if total_ms and awake_ms is not None:
                    asleep_ms = total_ms - awake_ms
                    if asleep_ms > 0:
                        payload["sleep_efficiency"] = round(
                            min(100.0, (asleep_ms / total_ms) * 100), 1
                        )
                rem_ms = stage_summary.get("total_rem_sleep_time_milli", 0)
                if rem_ms:
                    payload["sleep_rem_hours"] = round(rem_ms / 3_600_000, 2)
                slow_wave_ms = stage_summary.get("total_slow_wave_sleep_time_milli", 0)
                if slow_wave_ms:
                    payload["sleep_deep_hours"] = round(slow_wave_ms / 3_600_000, 2)
        except Exception as exc:
            log.warning("Whoop sleep fetch failed: %s", exc)

        # Cycle → strain score
        try:
            cycle = self.get_latest_cycle()
            if cycle:
                score = cycle.get("score", {}) or {}
                strain = score.get("strain")
                if strain is not None:
                    payload["strain_score"] = round(float(strain), 2)
                kilojoules = score.get("kilojoule")
                if kilojoules is not None:
                    # Convert kJ to kcal (1 kcal ≈ 4.184 kJ)
                    payload["active_energy_kcal"] = round(float(kilojoules) / 4.184, 1)
        except Exception as exc:
            log.warning("Whoop cycle fetch failed: %s", exc)

        payload["fetched_at"] = time.time()
        return payload


# ── Convenience parse functions (for data already fetched by client) ───────────

def parse_recovery(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalized recovery metrics from a raw Whoop recovery record.

    Args:
        record: Raw recovery record from GET /recovery.

    Returns:
        Dict with: recovery_score, hrv_rmssd, resting_heart_rate,
                   skin_temperature_deviation (all float, or None if missing).
    """
    score = record.get("score", {}) or {}
    return {
        "recovery_score": _safe_float(score.get("recovery_score")),
        "hrv_rmssd": _safe_float(score.get("hrv_rmssd_milli")),
        "resting_heart_rate": _safe_float(score.get("resting_heart_rate")),
        "skin_temperature_deviation": _safe_float(score.get("skin_temp_celsius")),
        "user_calibrating": bool(record.get("user_calibrating", False)),
    }


def parse_sleep(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalized sleep metrics from a raw Whoop sleep record.

    Args:
        record: Raw sleep record from GET /activity/sleep.

    Returns:
        Dict with: sleep_performance_pct, sleep_total_hours, sleep_efficiency,
                   sleep_rem_hours, sleep_deep_hours (all float or None).
    """
    score = record.get("score", {}) or {}
    stage_summary = score.get("stage_summary", {}) or {}

    total_ms = stage_summary.get("total_in_bed_time_milli") or 0
    awake_ms = stage_summary.get("total_awake_time_milli") or 0
    rem_ms = stage_summary.get("total_rem_sleep_time_milli") or 0
    slow_wave_ms = stage_summary.get("total_slow_wave_sleep_time_milli") or 0

    efficiency: Optional[float] = None
    if total_ms > 0 and awake_ms is not None:
        asleep_ms = total_ms - awake_ms
        efficiency = round(min(100.0, (asleep_ms / total_ms) * 100), 1)

    return {
        "sleep_performance_pct": _safe_float(score.get("sleep_performance_percentage")),
        "sleep_total_hours": round(total_ms / 3_600_000, 2) if total_ms else None,
        "sleep_efficiency": efficiency,
        "sleep_rem_hours": round(rem_ms / 3_600_000, 2) if rem_ms else None,
        "sleep_deep_hours": round(slow_wave_ms / 3_600_000, 2) if slow_wave_ms else None,
    }


def parse_cycle(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalized cycle (strain) metrics from a raw Whoop cycle record.

    Args:
        record: Raw cycle record from GET /cycle.

    Returns:
        Dict with: strain_score (0-21), active_energy_kcal (float or None).
    """
    score = record.get("score", {}) or {}
    kilojoules = score.get("kilojoule")
    kcal = round(float(kilojoules) / 4.184, 1) if kilojoules else None
    return {
        "strain_score": _safe_float(score.get("strain")),
        "active_energy_kcal": kcal,
    }


# ── Utilities ──────────────────────────────────────────────────────────────────

def _safe_float(value: Any) -> Optional[float]:
    """Convert to float, return None on failure or if value is falsy."""
    if value is None:
        return None
    try:
        f = float(value)
        return f if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None
