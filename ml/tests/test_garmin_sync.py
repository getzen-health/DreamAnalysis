"""Tests for Garmin Connect API sync module.

All network calls are monkeypatched — no real Garmin credentials needed.
Covers:
- OAuth 1.0a flow methods (get_request_token, get_authorization_url, exchange_verifier)
- Webhook payload parsing (dailies, sleeps, user metrics, unknown keys)
- Biometric payload mapping (build_biometric_payload)
- Error handling (expired tokens, invalid data, missing credentials)
- parse_daily_summary / parse_sleep / parse_user_metrics helpers
- _safe_float / _safe_int utilities
"""
from __future__ import annotations

import os
import sys
import time
import types
import unittest.mock as mock
from typing import Any, Dict, List, Optional

import pytest

# Ensure ml/ is on sys.path
_ML_DIR = os.path.join(os.path.dirname(__file__), "..")
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from health.garmin_sync import (
    get_request_token,
    get_authorization_url,
    exchange_verifier,
    parse_webhook_payload,
    parse_daily_summary,
    parse_sleep,
    parse_user_metrics,
    GarminClient,
    _safe_float,
    _safe_int,
    _GARMIN_REQUEST_TOKEN_URL,
    _GARMIN_AUTHORIZE_URL,
    _GARMIN_ACCESS_TOKEN_URL,
    _GARMIN_API_BASE,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

FAKE_DAILY = {
    "summaryId": "daily-001",
    "calendarDate": "2026-03-12",
    "userId": "user-abc",
    "highestBodyBattery": 92,
    "lowestBodyBattery": 18,
    "bodyBatteryChargedValue": 74,
    "bodyBatteryDrainedValue": 56,
    "averageStressLevel": 34,
    "maxStressLevel": 71,
    "restingHeartRateInBeatsPerMinute": 55,
    "averageHeartRateInBeatsPerMinute": 72,
    "steps": 9845,
    "activeKilocalories": 412.5,
    "averageSpo2": 97.2,
    "averageRespirationValue": 14.5,
}

FAKE_SLEEP = {
    "summaryId": "sleep-001",
    "calendarDate": "2026-03-12",
    "userId": "user-abc",
    "startTimeInSeconds": 1741830000,
    "durationInSeconds": 27000,       # 7.5 hours
    "deepSleepDurationInSeconds": 5400,   # 1.5 hours
    "lightSleepDurationInSeconds": 12600, # 3.5 hours
    "remSleepInSeconds": 6300,            # 1.75 hours
    "awakeDurationInSeconds": 2700,       # 0.75 hours
    "averageSpo2Value": 96.8,
    "lowestSpo2Value": 93.0,
    "averageRespirationValue": 15.2,
    "averageStressLevel": 22,
    "sleepScores": {"overall": 83},
}

FAKE_USER_METRICS = {
    "calendarDate": "2026-03-12",
    "userId": "user-abc",
    "lastNight": 48,
    "hrvWeeklyAverage": 52,
    "lastNight5MinHigh": 71,
    "hrvStatus": "BALANCED",
    "hrvFeedback": "Your HRV is in a good range.",
    "averageSpo2": 97.0,
    "lowestSpo2": 94.0,
    "averageRespiration": 14.8,
    "lowestRespiration": 12.1,
    "highestRespiration": 17.3,
}

FAKE_WEBHOOK = {
    "dailies": [FAKE_DAILY],
    "sleeps": [FAKE_SLEEP],
    "userMetrics": [FAKE_USER_METRICS],
}


def _fake_oauth_session(fetch_request_token_return=None, fetch_access_token_return=None,
                        authorization_url_return=None, get_return=None):
    """Build a mock OAuth1Session with configurable return values."""
    session = mock.MagicMock()
    if fetch_request_token_return is not None:
        session.fetch_request_token.return_value = fetch_request_token_return
    if fetch_access_token_return is not None:
        session.fetch_access_token.return_value = fetch_access_token_return
    if authorization_url_return is not None:
        session.authorization_url.return_value = authorization_url_return
    if get_return is not None:
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = get_return
        mock_resp.raise_for_status.return_value = None
        session.get.return_value = mock_resp
    return session


def _mock_garmin_client(daily=None, sleep=None, user_metrics=None, body_battery=None):
    """Build a GarminClient stub with monkeypatched fetchers."""
    client = GarminClient.__new__(GarminClient)
    client._access_token = "fake-token"
    client._access_token_secret = "fake-secret"
    client._consumer_key = None
    client._consumer_secret = None

    client.get_daily_summary = mock.MagicMock(return_value=daily or [])
    client.get_sleep_data = mock.MagicMock(return_value=sleep or [])
    client.get_user_metrics = mock.MagicMock(return_value=user_metrics or [])
    client.get_body_battery = mock.MagicMock(return_value=body_battery)
    return client


# ── OAuth 1.0a: get_request_token ─────────────────────────────────────────────

class TestGetRequestToken:
    def test_returns_token_dict(self, monkeypatch):
        fake_token = {"oauth_token": "req-tok", "oauth_token_secret": "req-sec",
                      "oauth_callback_confirmed": "true"}
        session = _fake_oauth_session(fetch_request_token_return=fake_token)

        with mock.patch(
            "health.garmin_sync._get_oauth1_session", return_value=session
        ):
            result = get_request_token(
                consumer_key="ck", consumer_secret="cs", callback_url="https://cb.example.com"
            )

        assert result["oauth_token"] == "req-tok"
        assert result["oauth_token_secret"] == "req-sec"

    def test_raises_on_missing_consumer_key(self, monkeypatch):
        monkeypatch.delenv("GARMIN_CONSUMER_KEY", raising=False)
        with pytest.raises(ValueError, match="GARMIN_CONSUMER_KEY"):
            get_request_token(consumer_key=None, consumer_secret="cs")

    def test_raises_on_missing_consumer_secret(self, monkeypatch):
        monkeypatch.delenv("GARMIN_CONSUMER_SECRET", raising=False)
        with pytest.raises(ValueError, match="GARMIN_CONSUMER_SECRET"):
            get_request_token(consumer_key="ck", consumer_secret=None)

    def test_wraps_network_errors(self, monkeypatch):
        session = mock.MagicMock()
        session.fetch_request_token.side_effect = Exception("timeout")
        with mock.patch("health.garmin_sync._get_oauth1_session", return_value=session):
            with pytest.raises(RuntimeError, match="failed"):
                get_request_token(consumer_key="ck", consumer_secret="cs")

    def test_falls_back_to_env_vars(self, monkeypatch):
        monkeypatch.setenv("GARMIN_CONSUMER_KEY", "env-ck")
        monkeypatch.setenv("GARMIN_CONSUMER_SECRET", "env-cs")
        fake_token = {"oauth_token": "t", "oauth_token_secret": "s"}
        session = _fake_oauth_session(fetch_request_token_return=fake_token)
        with mock.patch("health.garmin_sync._get_oauth1_session", return_value=session):
            result = get_request_token()
        assert result["oauth_token"] == "t"


# ── OAuth 1.0a: get_authorization_url ─────────────────────────────────────────

class TestGetAuthorizationUrl:
    def test_returns_authorization_url_string(self, monkeypatch):
        fake_url = "https://connect.garmin.com/oauthConfirm?oauth_token=req-tok"
        session = _fake_oauth_session(authorization_url_return=fake_url)
        with mock.patch("health.garmin_sync._get_oauth1_session", return_value=session):
            url = get_authorization_url(
                oauth_token="req-tok", consumer_key="ck", consumer_secret="cs"
            )
        assert url == fake_url

    def test_url_contains_garmin_authorize_base(self, monkeypatch):
        session = _fake_oauth_session(
            authorization_url_return=f"{_GARMIN_AUTHORIZE_URL}?oauth_token=x"
        )
        with mock.patch("health.garmin_sync._get_oauth1_session", return_value=session):
            url = get_authorization_url("x", consumer_key="ck", consumer_secret="cs")
        assert _GARMIN_AUTHORIZE_URL in url

    def test_raises_on_missing_credentials(self, monkeypatch):
        monkeypatch.delenv("GARMIN_CONSUMER_KEY", raising=False)
        monkeypatch.delenv("GARMIN_CONSUMER_SECRET", raising=False)
        with pytest.raises(ValueError):
            get_authorization_url("tok")

    def test_wraps_unexpected_errors(self, monkeypatch):
        session = mock.MagicMock()
        session.authorization_url.side_effect = Exception("broken")
        with mock.patch("health.garmin_sync._get_oauth1_session", return_value=session):
            with pytest.raises(RuntimeError):
                get_authorization_url("tok", consumer_key="ck", consumer_secret="cs")


# ── OAuth 1.0a: exchange_verifier ─────────────────────────────────────────────

class TestExchangeVerifier:
    def test_returns_access_tokens(self, monkeypatch):
        fake_access = {"oauth_token": "acc-tok", "oauth_token_secret": "acc-sec"}
        session = _fake_oauth_session(fetch_access_token_return=fake_access)
        with mock.patch("health.garmin_sync._get_oauth1_session", return_value=session):
            result = exchange_verifier(
                oauth_token="req-tok",
                oauth_verifier="verifier-123",
                oauth_token_secret="req-sec",
                consumer_key="ck",
                consumer_secret="cs",
            )
        assert result["oauth_token"] == "acc-tok"
        assert result["oauth_token_secret"] == "acc-sec"

    def test_raises_on_missing_consumer_key(self, monkeypatch):
        monkeypatch.delenv("GARMIN_CONSUMER_KEY", raising=False)
        with pytest.raises(ValueError, match="GARMIN_CONSUMER_KEY"):
            exchange_verifier("tok", "verifier", "sec")

    def test_wraps_network_errors(self, monkeypatch):
        session = mock.MagicMock()
        session.fetch_access_token.side_effect = Exception("401 Unauthorized")
        with mock.patch("health.garmin_sync._get_oauth1_session", return_value=session):
            with pytest.raises(RuntimeError, match="failed"):
                exchange_verifier(
                    "tok", "ver", "sec", consumer_key="ck", consumer_secret="cs"
                )


# ── Webhook payload parsing ────────────────────────────────────────────────────

class TestParseWebhookPayload:
    def test_parses_dailies(self):
        result = parse_webhook_payload({"dailies": [FAKE_DAILY]})
        assert "dailies" in result
        assert len(result["dailies"]) == 1
        assert result["dailies"][0]["body_battery_highest"] == 92

    def test_parses_sleeps(self):
        result = parse_webhook_payload({"sleeps": [FAKE_SLEEP]})
        assert "sleeps" in result
        assert len(result["sleeps"]) == 1
        # webhook parser returns raw seconds; public parse_sleep() returns hours
        assert result["sleeps"][0]["duration_seconds"] == 27000

    def test_parses_user_metrics(self):
        result = parse_webhook_payload({"userMetrics": [FAKE_USER_METRICS]})
        assert "user_metrics" in result
        assert result["user_metrics"][0]["hrv_last_night"] == 48
        assert result["user_metrics"][0]["hrv_status"] == "BALANCED"

    def test_full_webhook_payload(self):
        result = parse_webhook_payload(FAKE_WEBHOOK)
        assert "dailies" in result
        assert "sleeps" in result
        assert "user_metrics" in result

    def test_passes_through_unknown_keys(self):
        payload = {"dailies": [], "customData": {"foo": "bar"}}
        result = parse_webhook_payload(payload)
        assert result.get("customData") == {"foo": "bar"}

    def test_empty_payload(self):
        result = parse_webhook_payload({})
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_handles_none_lists_in_payload(self):
        result = parse_webhook_payload({"dailies": None})
        assert result["dailies"] == []

    def test_parses_epochs_passthrough(self):
        payload = {"epochs": [{"summaryId": "e1"}]}
        result = parse_webhook_payload(payload)
        assert result["epochs"] == [{"summaryId": "e1"}]

    def test_parses_stress_details_passthrough(self):
        payload = {"stressDetails": [{"stress": 45}]}
        result = parse_webhook_payload(payload)
        assert result["stress_details"] == [{"stress": 45}]


# ── parse_daily_summary ────────────────────────────────────────────────────────

class TestParseDailySummary:
    def test_extracts_body_battery(self):
        result = parse_daily_summary(FAKE_DAILY)
        assert result["body_battery_highest"] == 92
        assert result["body_battery_lowest"] == 18
        assert result["body_battery_charged"] == 74
        assert result["body_battery_drained"] == 56

    def test_extracts_stress(self):
        result = parse_daily_summary(FAKE_DAILY)
        assert result["average_stress"] == 34
        assert result["max_stress"] == 71

    def test_extracts_heart_rate(self):
        result = parse_daily_summary(FAKE_DAILY)
        assert result["resting_heart_rate"] == 55

    def test_extracts_steps_and_calories(self):
        result = parse_daily_summary(FAKE_DAILY)
        assert result["steps"] == 9845
        assert result["active_kilocalories"] == pytest.approx(412.5)

    def test_extracts_spo2_and_respiration(self):
        result = parse_daily_summary(FAKE_DAILY)
        assert result["average_spo2"] == pytest.approx(97.2)
        assert result["average_respiration"] == pytest.approx(14.5)

    def test_handles_empty_record(self):
        result = parse_daily_summary({})
        assert result["body_battery_highest"] is None
        assert result["average_stress"] is None
        assert result["steps"] is None

    def test_preserves_calendar_date(self):
        result = parse_daily_summary(FAKE_DAILY)
        assert result["calendar_date"] == "2026-03-12"


# ── parse_sleep ────────────────────────────────────────────────────────────────

class TestParseSleep:
    def test_extracts_total_hours(self):
        result = parse_sleep(FAKE_SLEEP)
        # 27000 / 3600 = 7.5
        assert result["sleep_total_hours"] == pytest.approx(7.5)

    def test_extracts_deep_hours(self):
        result = parse_sleep(FAKE_SLEEP)
        # 5400 / 3600 = 1.5
        assert result["sleep_deep_hours"] == pytest.approx(1.5)

    def test_extracts_light_hours(self):
        result = parse_sleep(FAKE_SLEEP)
        # 12600 / 3600 = 3.5
        assert result["sleep_light_hours"] == pytest.approx(3.5)

    def test_extracts_rem_hours(self):
        result = parse_sleep(FAKE_SLEEP)
        # 6300 / 3600 = 1.75
        assert result["sleep_rem_hours"] == pytest.approx(1.75)

    def test_extracts_sleep_score(self):
        result = parse_sleep(FAKE_SLEEP)
        assert result["sleep_score"] == 83

    def test_extracts_spo2_and_respiration(self):
        result = parse_sleep(FAKE_SLEEP)
        assert result["average_spo2"] == pytest.approx(96.8)
        assert result["lowest_spo2"] == pytest.approx(93.0)
        assert result["average_respiration"] == pytest.approx(15.2)

    def test_handles_empty_record(self):
        result = parse_sleep({})
        assert result["sleep_total_hours"] is None
        assert result["sleep_deep_hours"] is None
        assert result["sleep_score"] is None

    def test_handles_missing_sleep_scores(self):
        record = {**FAKE_SLEEP}
        del record["sleepScores"]
        result = parse_sleep(record)
        assert result["sleep_score"] is None

    def test_handles_non_dict_sleep_scores(self):
        record = {**FAKE_SLEEP, "sleepScores": None}
        result = parse_sleep(record)
        assert result["sleep_score"] is None


# ── parse_user_metrics ─────────────────────────────────────────────────────────

class TestParseUserMetrics:
    def test_extracts_hrv_last_night(self):
        result = parse_user_metrics(FAKE_USER_METRICS)
        assert result["hrv_last_night"] == 48

    def test_extracts_hrv_weekly_average(self):
        result = parse_user_metrics(FAKE_USER_METRICS)
        assert result["hrv_weekly_average"] == 52

    def test_extracts_hrv_status(self):
        result = parse_user_metrics(FAKE_USER_METRICS)
        assert result["hrv_status"] == "BALANCED"

    def test_extracts_spo2(self):
        result = parse_user_metrics(FAKE_USER_METRICS)
        assert result["average_spo2"] == pytest.approx(97.0)
        assert result["lowest_spo2"] == pytest.approx(94.0)

    def test_extracts_respiration(self):
        result = parse_user_metrics(FAKE_USER_METRICS)
        assert result["average_respiration"] == pytest.approx(14.8)

    def test_handles_empty_record(self):
        result = parse_user_metrics({})
        assert result["hrv_last_night"] is None
        assert result["hrv_status"] is None

    def test_hrv_status_low(self):
        record = {**FAKE_USER_METRICS, "hrvStatus": "LOW"}
        result = parse_user_metrics(record)
        assert result["hrv_status"] == "LOW"

    def test_hrv_status_unbalanced(self):
        record = {**FAKE_USER_METRICS, "hrvStatus": "UNBALANCED"}
        result = parse_user_metrics(record)
        assert result["hrv_status"] == "UNBALANCED"


# ── GarminClient.build_biometric_payload ──────────────────────────────────────

class TestBuildBiometricPayload:
    def test_payload_has_user_id(self):
        client = _mock_garmin_client(daily=[FAKE_DAILY], sleep=[FAKE_SLEEP],
                                     user_metrics=[FAKE_USER_METRICS])
        payload = client.build_biometric_payload("user-123")
        assert payload["user_id"] == "user-123"

    def test_payload_source_is_garmin(self):
        client = _mock_garmin_client(daily=[FAKE_DAILY])
        payload = client.build_biometric_payload("u1")
        assert payload["source"] == "garmin"

    def test_payload_contains_body_battery(self):
        client = _mock_garmin_client(daily=[FAKE_DAILY])
        payload = client.build_biometric_payload("u1")
        assert payload.get("body_battery_highest") == 92
        assert payload.get("body_battery_lowest") == 18

    def test_payload_contains_stress(self):
        client = _mock_garmin_client(daily=[FAKE_DAILY])
        payload = client.build_biometric_payload("u1")
        assert payload.get("average_stress") == 34

    def test_payload_contains_resting_hr(self):
        client = _mock_garmin_client(daily=[FAKE_DAILY])
        payload = client.build_biometric_payload("u1")
        assert payload.get("resting_heart_rate") == 55.0

    def test_payload_contains_steps(self):
        client = _mock_garmin_client(daily=[FAKE_DAILY])
        payload = client.build_biometric_payload("u1")
        assert payload.get("steps_today") == 9845

    def test_payload_contains_sleep_hours(self):
        client = _mock_garmin_client(sleep=[FAKE_SLEEP])
        payload = client.build_biometric_payload("u1")
        assert payload.get("sleep_total_hours") == pytest.approx(7.5)
        assert payload.get("sleep_deep_hours") == pytest.approx(1.5)
        assert payload.get("sleep_rem_hours") == pytest.approx(1.75)

    def test_payload_contains_sleep_score(self):
        client = _mock_garmin_client(sleep=[FAKE_SLEEP])
        payload = client.build_biometric_payload("u1")
        assert payload.get("sleep_score") == 83

    def test_payload_contains_hrv_from_user_metrics(self):
        client = _mock_garmin_client(user_metrics=[FAKE_USER_METRICS])
        payload = client.build_biometric_payload("u1")
        assert payload.get("hrv_last_night") == 48
        assert payload.get("hrv_weekly_average") == 52
        assert payload.get("hrv_status") == "BALANCED"

    def test_payload_has_fetched_at(self):
        client = _mock_garmin_client()
        payload = client.build_biometric_payload("u1")
        assert "fetched_at" in payload
        assert isinstance(payload["fetched_at"], float)
        assert payload["fetched_at"] > 0

    def test_payload_graceful_on_empty_data(self):
        client = _mock_garmin_client()
        payload = client.build_biometric_payload("u1")
        assert payload["user_id"] == "u1"
        assert payload["source"] == "garmin"

    def test_payload_graceful_on_api_error(self):
        """A failing fetcher should not crash the whole payload."""
        client = GarminClient.__new__(GarminClient)
        client._access_token = "tok"
        client._access_token_secret = "sec"
        client._consumer_key = None
        client._consumer_secret = None
        client.get_daily_summary = mock.MagicMock(side_effect=RuntimeError("network error"))
        client.get_sleep_data = mock.MagicMock(return_value=[])
        client.get_user_metrics = mock.MagicMock(return_value=[])
        payload = client.build_biometric_payload("u1")
        assert payload["user_id"] == "u1"
        assert "fetched_at" in payload

    def test_payload_spo2_from_daily(self):
        client = _mock_garmin_client(daily=[FAKE_DAILY])
        payload = client.build_biometric_payload("u1")
        assert payload.get("average_spo2") == pytest.approx(97.2)

    def test_payload_respiration_from_daily(self):
        client = _mock_garmin_client(daily=[FAKE_DAILY])
        payload = client.build_biometric_payload("u1")
        assert payload.get("average_respiration") == pytest.approx(14.5)

    def test_daily_spo2_not_overridden_by_user_metrics(self):
        """daily summary spo2 should take precedence — user_metrics fills only if absent."""
        client = _mock_garmin_client(daily=[FAKE_DAILY], user_metrics=[FAKE_USER_METRICS])
        payload = client.build_biometric_payload("u1")
        # daily sets average_spo2=97.2; user_metrics would be 97.0 but should not override
        assert payload.get("average_spo2") == pytest.approx(97.2)


# ── GarminClient constructor ───────────────────────────────────────────────────

class TestGarminClientInit:
    def test_empty_access_token_raises(self):
        with pytest.raises(ValueError, match="access_token"):
            GarminClient("", "secret")

    def test_empty_secret_raises(self):
        with pytest.raises(ValueError, match="access_token_secret"):
            GarminClient("token", "")

    def test_valid_credentials_stored(self):
        client = GarminClient("tok", "sec")
        assert client._access_token == "tok"
        assert client._access_token_secret == "sec"


# ── _safe_float / _safe_int ────────────────────────────────────────────────────

class TestSafeFloat:
    def test_converts_int(self):
        assert _safe_float(42) == 42.0

    def test_converts_string(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_returns_none_on_none(self):
        assert _safe_float(None) is None

    def test_returns_none_on_empty_string(self):
        assert _safe_float("") is None

    def test_returns_none_on_non_numeric(self):
        assert _safe_float("not-a-number") is None


class TestSafeInt:
    def test_converts_float(self):
        assert _safe_int(3.9) == 3

    def test_converts_string(self):
        assert _safe_int("42") == 42

    def test_returns_none_on_none(self):
        assert _safe_int(None) is None

    def test_returns_none_on_non_numeric_string(self):
        assert _safe_int("abc") is None

    def test_returns_zero_for_zero(self):
        assert _safe_int(0) == 0
