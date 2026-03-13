"""Garmin Connect API integration — OAuth 1.0a + health metric sync.

Pulls body battery, stress, sleep, heart rate, HRV, and respiration from
the Garmin Health API and maps them to the BiometricPayload schema used by
the ML backend.

Garmin uses OAuth 1.0a (not OAuth 2.0). The three-step OAuth 1.0a flow:
  1. get_request_token() — fetch a temporary request token.
  2. get_authorization_url(request_token) — redirect the user to Garmin.
  3. exchange_verifier(oauth_token, oauth_verifier) — swap for access token.

Garmin also pushes data via webhooks (push model), rather than purely pull.
parse_webhook_payload() handles inbound Garmin webhook POST bodies.

API base: https://apis.garmin.com/wellness-api/rest
Docs: https://developer.garmin.com/gc-developer-program/overview/

Environment variables:
  GARMIN_CONSUMER_KEY    — OAuth 1.0a consumer key
  GARMIN_CONSUMER_SECRET — OAuth 1.0a consumer secret
  GARMIN_CALLBACK_URL    — OAuth 1.0a callback URL (used during request token step)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_GARMIN_REQUEST_TOKEN_URL = "https://connectapi.garmin.com/oauth-service/oauth/request_token"
_GARMIN_AUTHORIZE_URL = "https://connect.garmin.com/oauthConfirm"
_GARMIN_ACCESS_TOKEN_URL = "https://connectapi.garmin.com/oauth-service/oauth/access_token"
_GARMIN_API_BASE = "https://apis.garmin.com/wellness-api/rest"


# ── Internal health metric format ──────────────────────────────────────────────

def _empty_garmin_payload(user_id: str) -> Dict[str, Any]:
    """Return skeleton BiometricPayload with garmin provenance."""
    return {
        "user_id": user_id,
        "source": "garmin",
        # All metric fields are optional — only set when data is available
    }


# ── OAuth 1.0a helpers ─────────────────────────────────────────────────────────

def _get_oauth1_session(
    consumer_key: Optional[str] = None,
    consumer_secret: Optional[str] = None,
    resource_owner_key: Optional[str] = None,
    resource_owner_secret: Optional[str] = None,
    callback_uri: Optional[str] = None,
    verifier: Optional[str] = None,
):
    """Build a requests_oauthlib OAuth1Session.

    Args:
        consumer_key: Garmin consumer key (falls back to GARMIN_CONSUMER_KEY).
        consumer_secret: Garmin consumer secret (falls back to GARMIN_CONSUMER_SECRET).
        resource_owner_key: OAuth token (for access token / signed API calls).
        resource_owner_secret: OAuth token secret.
        callback_uri: Callback URL (for request token step).
        verifier: OAuth verifier (for exchange step).

    Returns:
        requests_oauthlib.OAuth1Session instance.

    Raises:
        ValueError: if consumer key or secret cannot be resolved.
        ImportError: if requests_oauthlib is not installed.
    """
    try:
        from requests_oauthlib import OAuth1Session  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "requests_oauthlib is required for Garmin OAuth 1.0a. "
            "Install with: pip install requests_oauthlib"
        ) from exc

    ck = consumer_key or os.environ.get("GARMIN_CONSUMER_KEY")
    cs = consumer_secret or os.environ.get("GARMIN_CONSUMER_SECRET")
    if not ck:
        raise ValueError("GARMIN_CONSUMER_KEY is not set")
    if not cs:
        raise ValueError("GARMIN_CONSUMER_SECRET is not set")

    kwargs: Dict[str, Any] = {}
    if resource_owner_key:
        kwargs["resource_owner_key"] = resource_owner_key
    if resource_owner_secret:
        kwargs["resource_owner_secret"] = resource_owner_secret
    if callback_uri:
        kwargs["callback_uri"] = callback_uri
    if verifier:
        kwargs["verifier"] = verifier

    return OAuth1Session(ck, client_secret=cs, **kwargs)


def get_request_token(
    consumer_key: Optional[str] = None,
    consumer_secret: Optional[str] = None,
    callback_url: Optional[str] = None,
) -> Dict[str, str]:
    """Fetch a temporary OAuth 1.0a request token from Garmin.

    Args:
        consumer_key: Garmin consumer key (falls back to GARMIN_CONSUMER_KEY env var).
        consumer_secret: Garmin consumer secret (falls back to GARMIN_CONSUMER_SECRET).
        callback_url: OAuth callback URL (falls back to GARMIN_CALLBACK_URL env var).

    Returns:
        Dict with: oauth_token, oauth_token_secret, oauth_callback_confirmed.

    Raises:
        ValueError: if credentials cannot be resolved.
        RuntimeError: if the request token fetch fails.
    """
    cb = callback_url or os.environ.get("GARMIN_CALLBACK_URL", "oob")
    try:
        session = _get_oauth1_session(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            callback_uri=cb,
        )
        resp = session.fetch_request_token(_GARMIN_REQUEST_TOKEN_URL)
        return dict(resp)
    except (ValueError, ImportError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Garmin request token fetch failed: {exc}") from exc


def get_authorization_url(
    oauth_token: str,
    consumer_key: Optional[str] = None,
    consumer_secret: Optional[str] = None,
) -> str:
    """Build the Garmin OAuth 1.0a authorization URL.

    Args:
        oauth_token: Temporary request token from get_request_token().
        consumer_key: Garmin consumer key.
        consumer_secret: Garmin consumer secret.

    Returns:
        Full authorization URL to redirect the user to.

    Raises:
        ValueError: if credentials cannot be resolved.
    """
    try:
        session = _get_oauth1_session(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            resource_owner_key=oauth_token,
        )
        return session.authorization_url(_GARMIN_AUTHORIZE_URL)
    except (ValueError, ImportError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to build Garmin authorization URL: {exc}") from exc


def exchange_verifier(
    oauth_token: str,
    oauth_verifier: str,
    oauth_token_secret: str,
    consumer_key: Optional[str] = None,
    consumer_secret: Optional[str] = None,
) -> Dict[str, str]:
    """Exchange an OAuth verifier for a permanent access token.

    Args:
        oauth_token: Temporary request token from get_request_token().
        oauth_verifier: Verifier code provided by Garmin after user authorizes.
        oauth_token_secret: Token secret from get_request_token().
        consumer_key: Garmin consumer key.
        consumer_secret: Garmin consumer secret.

    Returns:
        Dict with: oauth_token (access), oauth_token_secret (access).

    Raises:
        ValueError: if credentials cannot be resolved.
        RuntimeError: if the token exchange fails.
    """
    try:
        session = _get_oauth1_session(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            resource_owner_key=oauth_token,
            resource_owner_secret=oauth_token_secret,
            verifier=oauth_verifier,
        )
        tokens = session.fetch_access_token(_GARMIN_ACCESS_TOKEN_URL)
        return dict(tokens)
    except (ValueError, ImportError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Garmin verifier exchange failed: {exc}") from exc


# ── Webhook payload parsing ─────────────────────────────────────────────────────

def parse_webhook_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a Garmin webhook push payload into normalized metric dicts.

    Garmin pushes data to registered webhook URLs rather than waiting to be
    pulled. This function normalizes the nested webhook body into a flat dict
    of metrics keyed by Garmin data type.

    Supported payload types: dailies, epochs, sleeps, bodyComps, hrReadings,
    stressDetails, userMetrics.

    Args:
        payload: Raw JSON dict from a Garmin webhook POST body.

    Returns:
        Dict with keys matching Garmin data types, each mapping to a list of
        normalized records. Unknown keys are passed through unchanged.

    Example return structure:
        {
            "dailies": [{"summary_id": ..., "body_battery_charged": 80, ...}],
            "sleeps": [{"summary_id": ..., "sleep_duration_seconds": 25200, ...}],
        }
    """
    normalized: Dict[str, Any] = {}

    # Dailies — body battery, stress, steps, heart rate summary
    if "dailies" in payload:
        normalized["dailies"] = [
            _parse_daily_summary(d) for d in (payload["dailies"] or [])
        ]

    # Epochs — intra-day activity summaries
    if "epochs" in payload:
        normalized["epochs"] = payload["epochs"]

    # Sleep summaries
    if "sleeps" in payload:
        normalized["sleeps"] = [
            _parse_sleep_summary(s) for s in (payload["sleeps"] or [])
        ]

    # Body composition
    if "bodyComps" in payload:
        normalized["body_comps"] = payload["bodyComps"]

    # HR readings (continuous)
    if "hrReadings" in payload:
        normalized["hr_readings"] = payload["hrReadings"]

    # Stress details (intra-day)
    if "stressDetails" in payload:
        normalized["stress_details"] = payload["stressDetails"]

    # User metrics (HRV status, SpO2, respiration)
    if "userMetrics" in payload:
        normalized["user_metrics"] = [
            _parse_user_metrics(m) for m in (payload["userMetrics"] or [])
        ]

    # Pass through any unrecognized top-level keys
    known = {"dailies", "epochs", "sleeps", "bodyComps", "hrReadings",
              "stressDetails", "userMetrics"}
    for key in payload:
        if key not in known:
            normalized[key] = payload[key]

    return normalized


def _parse_daily_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a Garmin daily summary record."""
    return {
        "summary_id": record.get("summaryId"),
        "calendar_date": record.get("calendarDate"),
        "body_battery_charged": _safe_int(record.get("bodyBatteryChargedValue")),
        "body_battery_drained": _safe_int(record.get("bodyBatteryDrainedValue")),
        "body_battery_highest": _safe_int(record.get("highestBodyBattery")),
        "body_battery_lowest": _safe_int(record.get("lowestBodyBattery")),
        "average_stress": _safe_int(record.get("averageStressLevel")),
        "max_stress": _safe_int(record.get("maxStressLevel")),
        "rest_stress_duration_seconds": _safe_int(record.get("restStressDuration")),
        "activity_stress_duration_seconds": _safe_int(record.get("activityStressDuration")),
        "resting_heart_rate": _safe_int(record.get("restingHeartRateInBeatsPerMinute")),
        "average_heart_rate": _safe_int(record.get("averageHeartRateInBeatsPerMinute")),
        "max_heart_rate": _safe_int(record.get("maxHeartRateInBeatsPerMinute")),
        "steps": _safe_int(record.get("steps")),
        "active_time_seconds": _safe_int(record.get("activeTimeInSeconds")),
        "active_kilocalories": _safe_float(record.get("activeKilocalories")),
        "average_spo2": _safe_float(record.get("averageSpo2")),
        "average_respiration": _safe_float(record.get("averageRespirationValue")),
        "user_id": record.get("userId"),
    }


def _parse_sleep_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a Garmin sleep summary record."""
    return {
        "summary_id": record.get("summaryId"),
        "calendar_date": record.get("calendarDate"),
        "duration_seconds": _safe_int(record.get("durationInSeconds")),
        "sleep_start_timestamp_gmt": record.get("startTimeInSeconds"),
        "unmeasurable_sleep_seconds": _safe_int(record.get("unmeasurableSleepInSeconds")),
        "deep_sleep_seconds": _safe_int(record.get("deepSleepDurationInSeconds")),
        "light_sleep_seconds": _safe_int(record.get("lightSleepDurationInSeconds")),
        "rem_sleep_seconds": _safe_int(record.get("remSleepInSeconds")),
        "awake_duration_seconds": _safe_int(record.get("awakeDurationInSeconds")),
        "average_spo2": _safe_float(record.get("averageSpo2Value")),
        "lowest_spo2": _safe_float(record.get("lowestSpo2Value")),
        "average_respiration": _safe_float(record.get("averageRespirationValue")),
        "average_stress": _safe_int(record.get("averageStressLevel")),
        "sleep_score": _safe_int(record.get("sleepScores", {}).get("overall") if isinstance(record.get("sleepScores"), dict) else None),
        "user_id": record.get("userId"),
    }


def _parse_user_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a Garmin user metrics record (HRV, SpO2, respiration)."""
    return {
        "calendar_date": record.get("calendarDate"),
        "hrv_weekly_average": _safe_int(record.get("hrvWeeklyAverage")),
        "hrv_last_night": _safe_int(record.get("lastNight")),
        "hrv_5_min_high": _safe_int(record.get("lastNight5MinHigh")),
        "hrv_status": record.get("hrvStatus"),  # "BALANCED", "LOW", "UNBALANCED"
        "hrv_feedback": record.get("hrvFeedback"),
        "average_spo2": _safe_float(record.get("averageSpo2")),
        "lowest_spo2": _safe_float(record.get("lowestSpo2")),
        "average_respiration": _safe_float(record.get("averageRespiration")),
        "lowest_respiration": _safe_float(record.get("lowestRespiration")),
        "highest_respiration": _safe_float(record.get("highestRespiration")),
        "user_id": record.get("userId"),
    }


# ── API client ──────────────────────────────────────────────────────────────────

class GarminClient:
    """Authenticated client for Garmin Health API (Wellness API).

    Uses OAuth 1.0a signed requests for all API calls.

    Usage:
        client = GarminClient(
            access_token="...",
            access_token_secret="...",
        )
        summary = client.get_daily_summary()
        payload = client.build_biometric_payload(user_id="u1")
    """

    def __init__(
        self,
        access_token: str,
        access_token_secret: str,
        consumer_key: Optional[str] = None,
        consumer_secret: Optional[str] = None,
    ) -> None:
        if not access_token:
            raise ValueError("access_token must not be empty")
        if not access_token_secret:
            raise ValueError("access_token_secret must not be empty")
        self._access_token = access_token
        self._access_token_secret = access_token_secret
        self._consumer_key = consumer_key
        self._consumer_secret = consumer_secret

    # ── Low-level HTTP helper ──────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[Dict[str, str]] = None) -> Any:
        """GET {_GARMIN_API_BASE}/{path} with OAuth 1.0a signing.

        Raises:
            RuntimeError: on non-2xx HTTP or network error.
            ImportError: if requests_oauthlib is not installed.
        """
        import json as _json
        import urllib.parse

        session = _get_oauth1_session(
            consumer_key=self._consumer_key,
            consumer_secret=self._consumer_secret,
            resource_owner_key=self._access_token,
            resource_owner_secret=self._access_token_secret,
        )

        url = f"{_GARMIN_API_BASE}/{path.lstrip('/')}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            raise RuntimeError(f"Garmin API GET {path} failed: {exc}") from exc

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

    def get_daily_summary(
        self, upload_start_time: Optional[int] = None, upload_end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return daily summary records including body battery and stress.

        Args:
            upload_start_time: Unix epoch seconds (default: 24 hours ago).
            upload_end_time: Unix epoch seconds (default: now).

        Returns:
            List of daily summary dicts.
        """
        now = int(time.time())
        params = {
            "uploadStartTimeInSeconds": str(upload_start_time or (now - 86_400)),
            "uploadEndTimeInSeconds": str(upload_end_time or now),
        }
        data = self._get("/dailies", params)
        return data if isinstance(data, list) else data.get("dailies", [])

    def get_sleep_data(
        self, upload_start_time: Optional[int] = None, upload_end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return sleep summary records including sleep stages and SpO2.

        Args:
            upload_start_time: Unix epoch seconds (default: 48 hours ago).
            upload_end_time: Unix epoch seconds (default: now).

        Returns:
            List of sleep summary dicts.
        """
        now = int(time.time())
        params = {
            "uploadStartTimeInSeconds": str(upload_start_time or (now - 172_800)),
            "uploadEndTimeInSeconds": str(upload_end_time or now),
        }
        data = self._get("/sleeps", params)
        return data if isinstance(data, list) else data.get("sleeps", [])

    def get_heart_rate_data(
        self, upload_start_time: Optional[int] = None, upload_end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return heart rate readings (continuous monitoring).

        Args:
            upload_start_time: Unix epoch seconds (default: 24 hours ago).
            upload_end_time: Unix epoch seconds (default: now).

        Returns:
            List of heart rate reading dicts.
        """
        now = int(time.time())
        params = {
            "uploadStartTimeInSeconds": str(upload_start_time or (now - 86_400)),
            "uploadEndTimeInSeconds": str(upload_end_time or now),
        }
        data = self._get("/heartRateVariability", params)
        return data if isinstance(data, list) else data.get("hrReadings", [])

    def get_stress_data(
        self, upload_start_time: Optional[int] = None, upload_end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return stress detail records (intra-day stress levels 0-100).

        Args:
            upload_start_time: Unix epoch seconds (default: 24 hours ago).
            upload_end_time: Unix epoch seconds (default: now).

        Returns:
            List of stress detail dicts.
        """
        now = int(time.time())
        params = {
            "uploadStartTimeInSeconds": str(upload_start_time or (now - 86_400)),
            "uploadEndTimeInSeconds": str(upload_end_time or now),
        }
        data = self._get("/stressDetails", params)
        return data if isinstance(data, list) else data.get("stressDetails", [])

    def get_user_metrics(
        self, upload_start_time: Optional[int] = None, upload_end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return user metrics records (HRV status, SpO2, respiration).

        Args:
            upload_start_time: Unix epoch seconds (default: 48 hours ago).
            upload_end_time: Unix epoch seconds (default: now).

        Returns:
            List of user metric dicts.
        """
        now = int(time.time())
        params = {
            "uploadStartTimeInSeconds": str(upload_start_time or (now - 172_800)),
            "uploadEndTimeInSeconds": str(upload_end_time or now),
        }
        data = self._get("/userMetrics", params)
        return data if isinstance(data, list) else data.get("userMetrics", [])

    def get_body_battery(self, date_str: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return body battery reading for a specific date.

        Body battery is included in the daily summary — this is a convenience
        helper that extracts the body battery fields from the most recent daily.

        Args:
            date_str: ISO date string (default: today).

        Returns:
            Dict with body battery fields, or None if no data.
        """
        now = int(time.time())
        try:
            dailies = self.get_daily_summary(
                upload_start_time=now - 86_400,
                upload_end_time=now,
            )
            if not dailies:
                return None
            latest = dailies[-1]
            return {
                "calendar_date": latest.get("calendarDate"),
                "body_battery_charged": _safe_int(latest.get("bodyBatteryChargedValue")),
                "body_battery_drained": _safe_int(latest.get("bodyBatteryDrainedValue")),
                "body_battery_highest": _safe_int(latest.get("highestBodyBattery")),
                "body_battery_lowest": _safe_int(latest.get("lowestBodyBattery")),
            }
        except Exception as exc:
            log.warning("Garmin body battery fetch failed: %s", exc)
            return None

    # ── Metric extraction ──────────────────────────────────────────────────────

    def build_biometric_payload(self, user_id: str) -> Dict[str, Any]:
        """Pull all available Garmin metrics and map to BiometricPayload schema.

        Fields populated (when available):
          - body_battery_highest: peak body battery for the day (0-100)
          - body_battery_lowest: lowest body battery for the day (0-100)
          - average_stress: average stress level for the day (0-100)
          - max_stress: peak stress level for the day (0-100)
          - resting_heart_rate: resting HR from daily summary (bpm)
          - hrv_last_night: last night's average HRV (ms)
          - hrv_weekly_average: 7-day average HRV (ms)
          - hrv_status: "BALANCED" | "LOW" | "UNBALANCED"
          - sleep_total_hours: total sleep duration
          - sleep_deep_hours: deep sleep duration
          - sleep_rem_hours: REM sleep duration
          - sleep_score: Garmin sleep score (0-100)
          - average_spo2: average SpO2 during sleep (%)
          - average_respiration: average respiration rate (brpm)
          - steps_today: daily step count
          - active_energy_kcal: active calories burned

        Returns:
            BiometricPayload dict with all available fields set.
        """
        payload = _empty_garmin_payload(user_id)
        now = int(time.time())

        # Daily summary → body battery, stress, HR, steps, calories
        try:
            dailies = self.get_daily_summary(
                upload_start_time=now - 86_400,
                upload_end_time=now,
            )
            if dailies:
                latest = dailies[-1]
                bb_high = _safe_int(latest.get("highestBodyBattery"))
                if bb_high is not None:
                    payload["body_battery_highest"] = bb_high
                bb_low = _safe_int(latest.get("lowestBodyBattery"))
                if bb_low is not None:
                    payload["body_battery_lowest"] = bb_low
                avg_stress = _safe_int(latest.get("averageStressLevel"))
                if avg_stress is not None and avg_stress >= 0:
                    payload["average_stress"] = avg_stress
                max_stress = _safe_int(latest.get("maxStressLevel"))
                if max_stress is not None and max_stress >= 0:
                    payload["max_stress"] = max_stress
                rhr = _safe_int(latest.get("restingHeartRateInBeatsPerMinute"))
                if rhr is not None and rhr > 0:
                    payload["resting_heart_rate"] = float(rhr)
                steps = _safe_int(latest.get("steps"))
                if steps is not None:
                    payload["steps_today"] = steps
                kcal = _safe_float(latest.get("activeKilocalories"))
                if kcal is not None:
                    payload["active_energy_kcal"] = kcal
                avg_spo2 = _safe_float(latest.get("averageSpo2"))
                if avg_spo2 is not None:
                    payload["average_spo2"] = avg_spo2
                avg_resp = _safe_float(latest.get("averageRespirationValue"))
                if avg_resp is not None:
                    payload["average_respiration"] = avg_resp
        except Exception as exc:
            log.warning("Garmin daily summary fetch failed: %s", exc)

        # Sleep → duration, stages, SpO2, score
        try:
            sleeps = self.get_sleep_data(
                upload_start_time=now - 172_800,
                upload_end_time=now,
            )
            if sleeps:
                latest_sleep = _pick_latest(sleeps, "startTimeInSeconds")
                if latest_sleep:
                    total_s = _safe_int(latest_sleep.get("durationInSeconds"))
                    if total_s and total_s > 0:
                        payload["sleep_total_hours"] = round(total_s / 3600, 2)
                    deep_s = _safe_int(latest_sleep.get("deepSleepDurationInSeconds"))
                    if deep_s:
                        payload["sleep_deep_hours"] = round(deep_s / 3600, 2)
                    light_s = _safe_int(latest_sleep.get("lightSleepDurationInSeconds"))
                    if light_s:
                        payload["sleep_light_hours"] = round(light_s / 3600, 2)
                    rem_s = _safe_int(latest_sleep.get("remSleepInSeconds"))
                    if rem_s:
                        payload["sleep_rem_hours"] = round(rem_s / 3600, 2)
                    spo2 = _safe_float(latest_sleep.get("averageSpo2Value"))
                    if spo2 is not None and "average_spo2" not in payload:
                        payload["average_spo2"] = spo2
                    resp = _safe_float(latest_sleep.get("averageRespirationValue"))
                    if resp is not None and "average_respiration" not in payload:
                        payload["average_respiration"] = resp
                    sleep_scores = latest_sleep.get("sleepScores")
                    if isinstance(sleep_scores, dict):
                        overall = _safe_int(sleep_scores.get("overall"))
                        if overall is not None:
                            payload["sleep_score"] = overall
        except Exception as exc:
            log.warning("Garmin sleep fetch failed: %s", exc)

        # User metrics → HRV status, SpO2, respiration
        try:
            user_metrics = self.get_user_metrics(
                upload_start_time=now - 172_800,
                upload_end_time=now,
            )
            if user_metrics:
                latest_metrics = _pick_latest(user_metrics, "calendarDate")
                if latest_metrics:
                    hrv_last = _safe_int(latest_metrics.get("lastNight"))
                    if hrv_last is not None and hrv_last > 0:
                        payload["hrv_last_night"] = hrv_last
                    hrv_weekly = _safe_int(latest_metrics.get("hrvWeeklyAverage"))
                    if hrv_weekly is not None and hrv_weekly > 0:
                        payload["hrv_weekly_average"] = hrv_weekly
                    hrv_status = latest_metrics.get("hrvStatus")
                    if hrv_status:
                        payload["hrv_status"] = str(hrv_status)
                    spo2 = _safe_float(latest_metrics.get("averageSpo2"))
                    if spo2 is not None and "average_spo2" not in payload:
                        payload["average_spo2"] = spo2
                    avg_resp = _safe_float(latest_metrics.get("averageRespiration"))
                    if avg_resp is not None and "average_respiration" not in payload:
                        payload["average_respiration"] = avg_resp
        except Exception as exc:
            log.warning("Garmin user metrics fetch failed: %s", exc)

        payload["fetched_at"] = time.time()
        return payload


# ── Convenience parse functions (for data already fetched / webhook data) ───────

def parse_daily_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalized daily metrics from a raw Garmin daily summary record.

    Args:
        record: Raw daily summary record from GET /dailies or webhook payload.

    Returns:
        Dict with body battery, stress, resting HR, steps, calories (typed or None).
    """
    return {
        "calendar_date": record.get("calendarDate"),
        "body_battery_highest": _safe_int(record.get("highestBodyBattery")),
        "body_battery_lowest": _safe_int(record.get("lowestBodyBattery")),
        "body_battery_charged": _safe_int(record.get("bodyBatteryChargedValue")),
        "body_battery_drained": _safe_int(record.get("bodyBatteryDrainedValue")),
        "average_stress": _safe_int(record.get("averageStressLevel")),
        "max_stress": _safe_int(record.get("maxStressLevel")),
        "resting_heart_rate": _safe_int(record.get("restingHeartRateInBeatsPerMinute")),
        "average_heart_rate": _safe_int(record.get("averageHeartRateInBeatsPerMinute")),
        "steps": _safe_int(record.get("steps")),
        "active_kilocalories": _safe_float(record.get("activeKilocalories")),
        "average_spo2": _safe_float(record.get("averageSpo2")),
        "average_respiration": _safe_float(record.get("averageRespirationValue")),
        "user_id": record.get("userId"),
    }


def parse_sleep(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalized sleep metrics from a raw Garmin sleep summary record.

    Args:
        record: Raw sleep record from GET /sleeps or webhook payload.

    Returns:
        Dict with sleep duration hours, stage breakdown, SpO2, score (typed or None).
    """
    duration_s = _safe_int(record.get("durationInSeconds"))
    deep_s = _safe_int(record.get("deepSleepDurationInSeconds"))
    light_s = _safe_int(record.get("lightSleepDurationInSeconds"))
    rem_s = _safe_int(record.get("remSleepInSeconds"))
    awake_s = _safe_int(record.get("awakeDurationInSeconds"))

    sleep_scores = record.get("sleepScores")
    overall_score: Optional[int] = None
    if isinstance(sleep_scores, dict):
        overall_score = _safe_int(sleep_scores.get("overall"))

    return {
        "calendar_date": record.get("calendarDate"),
        "sleep_total_hours": round(duration_s / 3600, 2) if duration_s else None,
        "sleep_deep_hours": round(deep_s / 3600, 2) if deep_s else None,
        "sleep_light_hours": round(light_s / 3600, 2) if light_s else None,
        "sleep_rem_hours": round(rem_s / 3600, 2) if rem_s else None,
        "sleep_awake_hours": round(awake_s / 3600, 2) if awake_s else None,
        "sleep_score": overall_score,
        "average_spo2": _safe_float(record.get("averageSpo2Value")),
        "lowest_spo2": _safe_float(record.get("lowestSpo2Value")),
        "average_respiration": _safe_float(record.get("averageRespirationValue")),
        "average_stress": _safe_int(record.get("averageStressLevel")),
        "user_id": record.get("userId"),
    }


def parse_user_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalized HRV / SpO2 / respiration from a Garmin user metrics record.

    Args:
        record: Raw user metrics record from GET /userMetrics or webhook payload.

    Returns:
        Dict with HRV status, last-night HRV, weekly HRV average, SpO2, respiration.
    """
    return {
        "calendar_date": record.get("calendarDate"),
        "hrv_last_night": _safe_int(record.get("lastNight")),
        "hrv_weekly_average": _safe_int(record.get("hrvWeeklyAverage")),
        "hrv_5_min_high": _safe_int(record.get("lastNight5MinHigh")),
        "hrv_status": record.get("hrvStatus"),  # BALANCED / LOW / UNBALANCED
        "hrv_feedback": record.get("hrvFeedback"),
        "average_spo2": _safe_float(record.get("averageSpo2")),
        "lowest_spo2": _safe_float(record.get("lowestSpo2")),
        "average_respiration": _safe_float(record.get("averageRespiration")),
        "user_id": record.get("userId"),
    }


# ── Internal utilities ──────────────────────────────────────────────────────────

def _pick_latest(records: List[Dict[str, Any]], sort_key: str) -> Optional[Dict[str, Any]]:
    """Return the record with the highest value for sort_key, or the last item."""
    if not records:
        return None
    try:
        return max(records, key=lambda r: r.get(sort_key) or 0)
    except Exception:
        return records[-1]


def _safe_float(value: Any) -> Optional[float]:
    """Convert to float, return None on failure or if value is falsy."""
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
