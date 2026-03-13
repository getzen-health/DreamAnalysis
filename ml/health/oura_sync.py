"""Oura Ring API v2 integration — OAuth2 + health metric sync.

Pulls readiness score, sleep score, activity score, HRV, and body
temperature from the Oura v2 API and maps them to the BiometricPayload
schema used by the ML backend.

OAuth2 flow:
  1. Direct user to authorization_url() to get an authorization code.
  2. Exchange code for tokens via exchange_code().
  3. Use OuraClient(access_token) for data requests.
  4. Refresh tokens automatically via refresh_tokens().

API base: https://api.ouraring.com/v2
Docs: https://cloud.ouraring.com/v2/docs

Environment variables:
  OURA_CLIENT_ID     — OAuth2 client ID
  OURA_CLIENT_SECRET — OAuth2 client secret
  OURA_REDIRECT_URI  — Redirect URI registered with Oura
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

log = logging.getLogger(__name__)

_OURA_AUTH_URL = "https://cloud.ouraring.com/oauth/authorize"
_OURA_TOKEN_URL = "https://api.ouraring.com/oauth/token"
_OURA_API_BASE = "https://api.ouraring.com/v2"

# Scopes required for the metrics we pull
_REQUIRED_SCOPES = [
    "daily",
    "heartrate",
    "personal",
    "session",
    "sleep",
    "workout",
]


# ── Internal health metric format ─────────────────────────────────────────────

def _empty_oura_payload(user_id: str) -> Dict[str, Any]:
    """Return skeleton BiometricPayload with oura provenance."""
    return {
        "user_id": user_id,
        "source": "oura",
        # All metric fields are optional — only set when data is available
    }


# ── OAuth2 helpers ─────────────────────────────────────────────────────────────

def authorization_url(
    client_id: Optional[str] = None,
    redirect_uri: Optional[str] = None,
    state: Optional[str] = None,
) -> str:
    """Build the Oura OAuth2 authorization URL.

    Args:
        client_id: Oura client ID (falls back to OURA_CLIENT_ID env var).
        redirect_uri: Redirect URI (falls back to OURA_REDIRECT_URI env var).
        state: CSRF state token (optional but recommended).

    Returns:
        Full authorization URL to redirect the user to.

    Raises:
        ValueError: if client_id or redirect_uri cannot be resolved.
    """
    cid = client_id or os.environ.get("OURA_CLIENT_ID")
    ruri = redirect_uri or os.environ.get("OURA_REDIRECT_URI")
    if not cid:
        raise ValueError("OURA_CLIENT_ID is not set")
    if not ruri:
        raise ValueError("OURA_REDIRECT_URI is not set")

    params: Dict[str, str] = {
        "client_id": cid,
        "redirect_uri": ruri,
        "response_type": "code",
        "scope": " ".join(_REQUIRED_SCOPES),
    }
    if state:
        params["state"] = state

    return f"{_OURA_AUTH_URL}?{urlencode(params)}"


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
        Token response dict: {access_token, refresh_token, expires_in, token_type}

    Raises:
        RuntimeError: if the token request fails.
    """
    import urllib.request
    import json as _json

    cid = client_id or os.environ.get("OURA_CLIENT_ID")
    csecret = client_secret or os.environ.get("OURA_CLIENT_SECRET")
    ruri = redirect_uri or os.environ.get("OURA_REDIRECT_URI")

    if not cid or not csecret:
        raise ValueError("OURA_CLIENT_ID and OURA_CLIENT_SECRET must be set")
    if not ruri:
        raise ValueError("OURA_REDIRECT_URI must be set")

    body = urlencode({
        "grant_type": "authorization_code",
        "client_id": cid,
        "client_secret": csecret,
        "redirect_uri": ruri,
        "code": code,
    }).encode()

    req = urllib.request.Request(
        _OURA_TOKEN_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return _json.loads(resp.read())
    except Exception as exc:
        raise RuntimeError(f"Oura token exchange failed: {exc}") from exc


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

    cid = client_id or os.environ.get("OURA_CLIENT_ID")
    csecret = client_secret or os.environ.get("OURA_CLIENT_SECRET")

    if not cid or not csecret:
        raise ValueError("OURA_CLIENT_ID and OURA_CLIENT_SECRET must be set")

    body = urlencode({
        "grant_type": "refresh_token",
        "client_id": cid,
        "client_secret": csecret,
        "refresh_token": refresh_token,
    }).encode()

    req = urllib.request.Request(
        _OURA_TOKEN_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return _json.loads(resp.read())
    except Exception as exc:
        raise RuntimeError(f"Oura token refresh failed: {exc}") from exc


# ── API client ─────────────────────────────────────────────────────────────────

class OuraClient:
    """Authenticated client for Oura Ring API v2.

    Usage:
        client = OuraClient(access_token="...")
        readiness = client.get_daily_readiness()
        payload = client.build_biometric_payload(user_id="u1")
    """

    def __init__(self, access_token: str) -> None:
        if not access_token:
            raise ValueError("access_token must not be empty")
        self._token = access_token

    # ── Low-level HTTP helper ──────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[Dict[str, str]] = None) -> Any:
        """GET {_OURA_API_BASE}/{path} with Bearer auth.

        Raises:
            RuntimeError: on non-2xx HTTP or network error.
        """
        import urllib.request
        import urllib.parse
        import json as _json

        url = f"{_OURA_API_BASE}/{path.lstrip('/')}"
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
            raise RuntimeError(f"Oura API GET {path} failed: {exc}") from exc

    # ── Date helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _today_str() -> str:
        import datetime
        return datetime.date.today().isoformat()

    @staticmethod
    def _n_days_ago_str(n: int) -> str:
        import datetime
        return (datetime.date.today() - datetime.timedelta(days=n)).isoformat()

    # ── Data fetchers ──────────────────────────────────────────────────────────

    def get_personal_info(self) -> Dict[str, Any]:
        """Return authenticated user's Oura personal info."""
        return self._get("/usercollection/personal_info")

    def get_daily_readiness(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return daily readiness records.

        Args:
            start_date: ISO date string (default: 7 days ago).
            end_date: ISO date string (default: today).

        Returns:
            List of daily readiness record dicts.
        """
        params = {
            "start_date": start_date or self._n_days_ago_str(7),
            "end_date": end_date or self._today_str(),
        }
        data = self._get("/usercollection/daily_readiness", params)
        return data.get("data", [])

    def get_daily_sleep(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return daily sleep score records.

        Args:
            start_date: ISO date string (default: 7 days ago).
            end_date: ISO date string (default: today).

        Returns:
            List of daily sleep score record dicts.
        """
        params = {
            "start_date": start_date or self._n_days_ago_str(7),
            "end_date": end_date or self._today_str(),
        }
        data = self._get("/usercollection/daily_sleep", params)
        return data.get("data", [])

    def get_daily_activity(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return daily activity score records.

        Args:
            start_date: ISO date string (default: 7 days ago).
            end_date: ISO date string (default: today).

        Returns:
            List of daily activity record dicts.
        """
        params = {
            "start_date": start_date or self._n_days_ago_str(7),
            "end_date": end_date or self._today_str(),
        }
        data = self._get("/usercollection/daily_activity", params)
        return data.get("data", [])

    def get_sleep_sessions(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return detailed sleep session records (individual sleep sessions).

        Contains HRV, body temperature, respiratory rate per session.
        """
        params = {
            "start_date": start_date or self._n_days_ago_str(3),
            "end_date": end_date or self._today_str(),
        }
        data = self._get("/usercollection/sleep", params)
        return data.get("data", [])

    def get_heartrate(
        self, start_datetime: Optional[str] = None, end_datetime: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return heart rate time series data.

        Args:
            start_datetime: ISO datetime (default: 24 hours ago).
            end_datetime: ISO datetime (default: now).

        Returns:
            List of {bpm, source, timestamp} dicts.
        """
        import datetime
        now = datetime.datetime.utcnow()
        default_start = (now - datetime.timedelta(hours=24)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        params = {
            "start_datetime": start_datetime or default_start,
            "end_datetime": end_datetime or now.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        data = self._get("/usercollection/heartrate", params)
        return data.get("data", [])

    # ── Metric extraction ──────────────────────────────────────────────────────

    def build_biometric_payload(self, user_id: str) -> Dict[str, Any]:
        """Pull all available Oura metrics and map to BiometricPayload schema.

        Fields populated (when available):
          - readiness_score: 0-100 Oura readiness score
          - sleep_score: 0-100 Oura sleep score
          - activity_score: 0-100 Oura activity score
          - hrv_rmssd: average HRV from last sleep session (ms)
          - resting_heart_rate: lowest HR during sleep (bpm)
          - skin_temperature_deviation: deviation from personal baseline (°C)
          - respiratory_rate: average breathing rate during sleep (brpm)
          - sleep_total_hours: total sleep time
          - sleep_rem_hours: REM sleep time
          - sleep_deep_hours: deep sleep time
          - sleep_efficiency: sleep efficiency percentage (0-100)
          - steps_today: daily step count
          - active_energy_kcal: active calories burned today

        Returns:
            BiometricPayload dict with all available fields set.
        """
        payload = _empty_oura_payload(user_id)
        today = self._today_str()

        # Daily readiness → readiness score, HRV, resting HR, temperature
        try:
            readiness_records = self.get_daily_readiness(today, today)
            if readiness_records:
                rec = readiness_records[-1]
                contributors = rec.get("contributors", {}) or {}
                score = rec.get("score")
                if score is not None:
                    payload["readiness_score"] = int(score)
                hrv = rec.get("hrv_balance")
                if hrv is not None:
                    # Oura reports "hrv_balance" as a 0-100 index,
                    # but raw HRV RMSSD is in the sleep session
                    payload["hrv_balance_index"] = float(hrv)
                temp_dev = rec.get("temperature_deviation")
                if temp_dev is not None:
                    payload["skin_temperature_deviation"] = round(float(temp_dev), 2)
        except Exception as exc:
            log.warning("Oura readiness fetch failed: %s", exc)

        # Daily sleep score
        try:
            sleep_score_records = self.get_daily_sleep(today, today)
            if sleep_score_records:
                rec = sleep_score_records[-1]
                score = rec.get("score")
                if score is not None:
                    payload["sleep_score"] = int(score)
        except Exception as exc:
            log.warning("Oura daily sleep score fetch failed: %s", exc)

        # Sleep sessions → detailed HRV, RHR, temperature, respiratory rate
        try:
            sleep_sessions = self.get_sleep_sessions(
                self._n_days_ago_str(1), today
            )
            if sleep_sessions:
                # Use the most recent long sleep (type=long_sleep or highest duration)
                main_sleep = _pick_main_sleep(sleep_sessions)
                if main_sleep:
                    # HRV RMSSD average (ms)
                    hrv_avg = _extract_hrv_avg(main_sleep)
                    if hrv_avg is not None:
                        payload["hrv_rmssd"] = round(hrv_avg, 1)
                    # Resting HR (lowest average HR during sleep)
                    rhr = main_sleep.get("lowest_heart_rate")
                    if rhr is not None:
                        payload["resting_heart_rate"] = float(rhr)
                    # Skin temperature deviation
                    temp_dev = main_sleep.get("readiness", {}).get(
                        "temperature_deviation"
                    ) if main_sleep.get("readiness") else None
                    if temp_dev is not None and "skin_temperature_deviation" not in payload:
                        payload["skin_temperature_deviation"] = round(float(temp_dev), 2)
                    # Respiratory rate
                    resp_rate = main_sleep.get("average_breath")
                    if resp_rate is not None:
                        payload["respiratory_rate"] = round(float(resp_rate), 1)
                    # Sleep timing
                    duration_s = main_sleep.get("total_sleep_duration")
                    if duration_s:
                        payload["sleep_total_hours"] = round(duration_s / 3600, 2)
                    rem_s = main_sleep.get("rem_sleep_duration")
                    if rem_s:
                        payload["sleep_rem_hours"] = round(rem_s / 3600, 2)
                    deep_s = main_sleep.get("deep_sleep_duration")
                    if deep_s:
                        payload["sleep_deep_hours"] = round(deep_s / 3600, 2)
                    efficiency = main_sleep.get("efficiency")
                    if efficiency is not None:
                        payload["sleep_efficiency"] = float(efficiency)
        except Exception as exc:
            log.warning("Oura sleep session fetch failed: %s", exc)

        # Daily activity → score, steps, calories
        try:
            activity_records = self.get_daily_activity(today, today)
            if activity_records:
                rec = activity_records[-1]
                score = rec.get("score")
                if score is not None:
                    payload["activity_score"] = int(score)
                steps = rec.get("steps")
                if steps is not None:
                    payload["steps_today"] = int(steps)
                cal = rec.get("active_calories")
                if cal is not None:
                    payload["active_energy_kcal"] = float(cal)
        except Exception as exc:
            log.warning("Oura activity fetch failed: %s", exc)

        payload["fetched_at"] = time.time()
        return payload


# ── Parse helpers (for data already fetched by client) ────────────────────────

def parse_readiness(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalized readiness metrics from a raw Oura daily_readiness record.

    Args:
        record: Raw record from GET /usercollection/daily_readiness.

    Returns:
        Dict with: readiness_score, hrv_balance_index,
                   skin_temperature_deviation, contributors (all typed or None).
    """
    return {
        "readiness_score": _safe_int(record.get("score")),
        "hrv_balance_index": _safe_float(record.get("hrv_balance")),
        "skin_temperature_deviation": _safe_float(record.get("temperature_deviation")),
        "contributors": record.get("contributors", {}),
        "day": record.get("day"),
    }


def parse_sleep_score(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalized sleep score from a raw Oura daily_sleep record.

    Args:
        record: Raw record from GET /usercollection/daily_sleep.

    Returns:
        Dict with: sleep_score, contributors, day.
    """
    return {
        "sleep_score": _safe_int(record.get("score")),
        "contributors": record.get("contributors", {}),
        "day": record.get("day"),
    }


def parse_sleep_session(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalized sleep metrics from a raw Oura sleep session record.

    Args:
        record: Raw record from GET /usercollection/sleep.

    Returns:
        Dict with: hrv_rmssd_avg, resting_heart_rate, respiratory_rate,
                   sleep_total_hours, sleep_rem_hours, sleep_deep_hours,
                   sleep_efficiency, sleep_type, skin_temperature_deviation.
    """
    hrv_avg = _extract_hrv_avg(record)
    duration_s = record.get("total_sleep_duration")
    rem_s = record.get("rem_sleep_duration")
    deep_s = record.get("deep_sleep_duration")
    readiness = record.get("readiness", {}) or {}

    return {
        "hrv_rmssd_avg": round(hrv_avg, 1) if hrv_avg is not None else None,
        "resting_heart_rate": _safe_float(record.get("lowest_heart_rate")),
        "respiratory_rate": _safe_float(record.get("average_breath")),
        "sleep_total_hours": round(duration_s / 3600, 2) if duration_s else None,
        "sleep_rem_hours": round(rem_s / 3600, 2) if rem_s else None,
        "sleep_deep_hours": round(deep_s / 3600, 2) if deep_s else None,
        "sleep_efficiency": _safe_float(record.get("efficiency")),
        "sleep_type": record.get("type"),
        "skin_temperature_deviation": _safe_float(
            readiness.get("temperature_deviation")
        ),
        "day": record.get("day"),
    }


def parse_activity(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalized activity metrics from a raw Oura daily_activity record.

    Args:
        record: Raw record from GET /usercollection/daily_activity.

    Returns:
        Dict with: activity_score, steps_today, active_energy_kcal, day.
    """
    return {
        "activity_score": _safe_int(record.get("score")),
        "steps_today": _safe_int(record.get("steps")),
        "active_energy_kcal": _safe_float(record.get("active_calories")),
        "day": record.get("day"),
    }


# ── Internal utilities ─────────────────────────────────────────────────────────

def _pick_main_sleep(sessions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the longest sleep session (most likely the main overnight sleep)."""
    if not sessions:
        return None
    # Prefer sessions with type == "long_sleep"
    long_sleeps = [s for s in sessions if s.get("type") == "long_sleep"]
    candidates = long_sleeps if long_sleeps else sessions
    return max(candidates, key=lambda s: s.get("total_sleep_duration") or 0)


def _extract_hrv_avg(session: Dict[str, Any]) -> Optional[float]:
    """Extract average HRV RMSSD from a sleep session's hrv time series."""
    hrv_data = session.get("hrv", {}) or {}
    # Oura v2: hrv.items = [{interval: int, items: [float, ...], timestamp: str}]
    items_container = hrv_data.get("items") if isinstance(hrv_data, dict) else None
    if not items_container:
        return None
    # items_container could be a flat list of values or a list of dicts
    if isinstance(items_container, list):
        if isinstance(items_container[0], dict):
            # Each item is {interval, items, timestamp}
            values: List[float] = []
            for block in items_container:
                block_vals = block.get("items", []) or []
                values.extend(v for v in block_vals if v is not None)
        else:
            # Flat list of floats
            values = [v for v in items_container if v is not None]
        if values:
            return sum(values) / len(values)
    return None


def _safe_float(value: Any) -> Optional[float]:
    """Convert to float, return None on failure."""
    if value is None:
        return None
    try:
        f = float(value)
        return f if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    """Convert to int, return None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
