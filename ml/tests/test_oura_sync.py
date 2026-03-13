"""Tests for Oura Ring API v2 sync module.

All network calls are monkeypatched — no real Oura credentials needed.
Covers:
- OAuth2 URL construction
- Token exchange / refresh (mocked HTTP)
- OuraClient metric extraction
- parse_readiness / parse_sleep_score / parse_sleep_session / parse_activity helpers
- BiometricPayload schema completeness
- Internal utilities (_pick_main_sleep, _extract_hrv_avg)
- Error handling (API errors, missing data, env vars not set)
"""
from __future__ import annotations

import os
import sys
import json
import unittest.mock as mock
from typing import Any, Dict, List

import pytest

# Ensure ml/ is on sys.path
_ML_DIR = os.path.join(os.path.dirname(__file__), "..")
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from health.oura_sync import (
    authorization_url,
    exchange_code,
    refresh_tokens,
    parse_readiness,
    parse_sleep_score,
    parse_sleep_session,
    parse_activity,
    OuraClient,
    _pick_main_sleep,
    _extract_hrv_avg,
    _safe_float,
    _safe_int,
    _OURA_AUTH_URL,
    _REQUIRED_SCOPES,
)


# ── Fixtures / helpers ─────────────────────────────────────────────────────────

FAKE_READINESS = {
    "id": "r-001",
    "day": "2026-03-10",
    "score": 82,
    "hrv_balance": 75,
    "temperature_deviation": -0.1,
    "contributors": {
        "activity_balance": 85,
        "body_temperature": 95,
        "hrv_balance": 75,
        "previous_day_activity": 90,
        "previous_night": 80,
        "recovery_index": 85,
        "resting_heart_rate": 88,
        "sleep_balance": 77,
    },
}

FAKE_SLEEP_SCORE = {
    "id": "ss-001",
    "day": "2026-03-10",
    "score": 88,
    "contributors": {
        "deep_sleep": 90,
        "efficiency": 95,
        "latency": 85,
        "rem_sleep": 78,
        "restfulness": 82,
        "timing": 88,
        "total_sleep": 87,
    },
}

FAKE_SLEEP_SESSION = {
    "id": "slp-001",
    "day": "2026-03-10",
    "type": "long_sleep",
    "bedtime_start": "2026-03-10T23:00:00+00:00",
    "bedtime_end": "2026-03-11T07:00:00+00:00",
    "total_sleep_duration": 25200,    # 7 hours in seconds
    "rem_sleep_duration": 5400,       # 1.5 hours
    "deep_sleep_duration": 3600,      # 1 hour
    "efficiency": 92,
    "lowest_heart_rate": 50,
    "average_breath": 14.5,
    "hrv": {
        "items": [45.2, 50.1, 48.3, 52.7, 49.0],
    },
    "readiness": {
        "temperature_deviation": -0.2,
    },
}

FAKE_ACTIVITY = {
    "id": "act-001",
    "day": "2026-03-10",
    "score": 75,
    "steps": 8500,
    "active_calories": 420.5,
}


def _mock_oura_client(
    readiness_records=None,
    sleep_score_records=None,
    sleep_sessions=None,
    activity_records=None,
):
    """Build an OuraClient stub that returns provided fixtures."""
    client = OuraClient.__new__(OuraClient)
    client._token = "fake-token"
    client.get_daily_readiness = mock.MagicMock(
        return_value=readiness_records or []
    )
    client.get_daily_sleep = mock.MagicMock(
        return_value=sleep_score_records or []
    )
    client.get_sleep_sessions = mock.MagicMock(
        return_value=sleep_sessions or []
    )
    client.get_daily_activity = mock.MagicMock(
        return_value=activity_records or []
    )
    return client


# ── OAuth2 URL construction ────────────────────────────────────────────────────

class TestAuthorizationUrl:
    def test_url_starts_with_oura_auth_base(self):
        url = authorization_url(
            client_id="test-id",
            redirect_uri="https://example.com/cb",
        )
        assert url.startswith(_OURA_AUTH_URL)

    def test_url_contains_client_id(self):
        url = authorization_url(
            client_id="my-oura-id",
            redirect_uri="https://example.com/cb",
        )
        assert "my-oura-id" in url

    def test_url_contains_response_type_code(self):
        url = authorization_url(
            client_id="cid",
            redirect_uri="https://example.com/cb",
        )
        assert "response_type=code" in url

    def test_url_contains_state_when_provided(self):
        url = authorization_url(
            client_id="cid",
            redirect_uri="https://example.com/cb",
            state="csrf-xyz",
        )
        assert "csrf-xyz" in url

    def test_url_omits_state_when_not_provided(self):
        url = authorization_url(
            client_id="cid",
            redirect_uri="https://example.com/cb",
        )
        assert "state=" not in url

    def test_raises_on_missing_client_id(self, monkeypatch):
        monkeypatch.delenv("OURA_CLIENT_ID", raising=False)
        with pytest.raises(ValueError, match="OURA_CLIENT_ID"):
            authorization_url(redirect_uri="https://example.com/cb")

    def test_raises_on_missing_redirect_uri(self, monkeypatch):
        monkeypatch.delenv("OURA_REDIRECT_URI", raising=False)
        with pytest.raises(ValueError, match="OURA_REDIRECT_URI"):
            authorization_url(client_id="cid")

    def test_falls_back_to_env_vars(self, monkeypatch):
        monkeypatch.setenv("OURA_CLIENT_ID", "env-oura-id")
        monkeypatch.setenv("OURA_REDIRECT_URI", "https://env.example.com/cb")
        url = authorization_url()
        assert "env-oura-id" in url

    def test_required_scopes_include_sleep(self):
        assert "sleep" in _REQUIRED_SCOPES

    def test_required_scopes_include_daily(self):
        assert "daily" in _REQUIRED_SCOPES


# ── Token exchange (mocked HTTP) ───────────────────────────────────────────────

class TestExchangeCode:
    def _fake_urlopen(self, response_body: Dict):
        body = json.dumps(response_body).encode()

        class FakeResp:
            def read(self): return body
            def __enter__(self): return self
            def __exit__(self, *_): pass

        return FakeResp()

    def test_exchange_returns_access_token(self, monkeypatch):
        fake_tokens = {"access_token": "oura-acc", "token_type": "Bearer"}
        monkeypatch.setenv("OURA_CLIENT_ID", "cid")
        monkeypatch.setenv("OURA_CLIENT_SECRET", "csec")
        monkeypatch.setenv("OURA_REDIRECT_URI", "https://example.com/cb")

        with mock.patch("urllib.request.urlopen", return_value=self._fake_urlopen(fake_tokens)):
            result = exchange_code("auth-code")

        assert result["access_token"] == "oura-acc"

    def test_exchange_raises_on_missing_credentials(self, monkeypatch):
        monkeypatch.delenv("OURA_CLIENT_ID", raising=False)
        monkeypatch.delenv("OURA_CLIENT_SECRET", raising=False)
        with pytest.raises(ValueError):
            exchange_code("code", client_id=None, client_secret=None)

    def test_exchange_wraps_http_errors(self, monkeypatch):
        monkeypatch.setenv("OURA_CLIENT_ID", "cid")
        monkeypatch.setenv("OURA_CLIENT_SECRET", "csec")
        monkeypatch.setenv("OURA_REDIRECT_URI", "https://example.com/cb")

        with mock.patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            with pytest.raises(RuntimeError, match="failed"):
                exchange_code("bad-code")


class TestRefreshTokens:
    def test_refresh_returns_new_access_token(self, monkeypatch):
        fake_tokens = {"access_token": "new-oura-acc", "refresh_token": "new-ref"}
        monkeypatch.setenv("OURA_CLIENT_ID", "cid")
        monkeypatch.setenv("OURA_CLIENT_SECRET", "csec")

        body = json.dumps(fake_tokens).encode()

        class FakeResp:
            def read(self): return body
            def __enter__(self): return self
            def __exit__(self, *_): pass

        with mock.patch("urllib.request.urlopen", return_value=FakeResp()):
            result = refresh_tokens("old-refresh")

        assert result["access_token"] == "new-oura-acc"

    def test_refresh_raises_on_missing_credentials(self, monkeypatch):
        monkeypatch.delenv("OURA_CLIENT_ID", raising=False)
        monkeypatch.delenv("OURA_CLIENT_SECRET", raising=False)
        with pytest.raises(ValueError):
            refresh_tokens("tok")


# ── parse_readiness ────────────────────────────────────────────────────────────

class TestParseReadiness:
    def test_extracts_score(self):
        result = parse_readiness(FAKE_READINESS)
        assert result["readiness_score"] == 82

    def test_extracts_hrv_balance(self):
        result = parse_readiness(FAKE_READINESS)
        assert result["hrv_balance_index"] == 75.0

    def test_extracts_temperature_deviation(self):
        result = parse_readiness(FAKE_READINESS)
        assert result["skin_temperature_deviation"] == pytest.approx(-0.1)

    def test_extracts_day(self):
        result = parse_readiness(FAKE_READINESS)
        assert result["day"] == "2026-03-10"

    def test_handles_empty_record(self):
        result = parse_readiness({})
        assert result["readiness_score"] is None
        assert result["hrv_balance_index"] is None


# ── parse_sleep_score ──────────────────────────────────────────────────────────

class TestParseSleepScore:
    def test_extracts_score(self):
        result = parse_sleep_score(FAKE_SLEEP_SCORE)
        assert result["sleep_score"] == 88

    def test_extracts_day(self):
        result = parse_sleep_score(FAKE_SLEEP_SCORE)
        assert result["day"] == "2026-03-10"

    def test_handles_missing_score(self):
        result = parse_sleep_score({"day": "2026-03-10"})
        assert result["sleep_score"] is None


# ── parse_sleep_session ────────────────────────────────────────────────────────

class TestParseSleepSession:
    def test_extracts_total_hours(self):
        result = parse_sleep_session(FAKE_SLEEP_SESSION)
        assert result["sleep_total_hours"] == pytest.approx(7.0)

    def test_extracts_rem_hours(self):
        result = parse_sleep_session(FAKE_SLEEP_SESSION)
        assert result["sleep_rem_hours"] == pytest.approx(1.5)

    def test_extracts_deep_hours(self):
        result = parse_sleep_session(FAKE_SLEEP_SESSION)
        assert result["sleep_deep_hours"] == pytest.approx(1.0)

    def test_extracts_resting_hr(self):
        result = parse_sleep_session(FAKE_SLEEP_SESSION)
        assert result["resting_heart_rate"] == 50.0

    def test_extracts_respiratory_rate(self):
        result = parse_sleep_session(FAKE_SLEEP_SESSION)
        assert result["respiratory_rate"] == pytest.approx(14.5)

    def test_extracts_efficiency(self):
        result = parse_sleep_session(FAKE_SLEEP_SESSION)
        assert result["sleep_efficiency"] == 92.0

    def test_extracts_hrv_avg(self):
        result = parse_sleep_session(FAKE_SLEEP_SESSION)
        assert result["hrv_rmssd_avg"] is not None
        # avg of [45.2, 50.1, 48.3, 52.7, 49.0] = 49.06
        assert abs(result["hrv_rmssd_avg"] - 49.06) < 0.1

    def test_extracts_sleep_type(self):
        result = parse_sleep_session(FAKE_SLEEP_SESSION)
        assert result["sleep_type"] == "long_sleep"

    def test_extracts_temperature_deviation(self):
        result = parse_sleep_session(FAKE_SLEEP_SESSION)
        assert result["skin_temperature_deviation"] == pytest.approx(-0.2)

    def test_handles_empty_session(self):
        result = parse_sleep_session({})
        assert result["sleep_total_hours"] is None
        assert result["hrv_rmssd_avg"] is None


# ── parse_activity ─────────────────────────────────────────────────────────────

class TestParseActivity:
    def test_extracts_score(self):
        result = parse_activity(FAKE_ACTIVITY)
        assert result["activity_score"] == 75

    def test_extracts_steps(self):
        result = parse_activity(FAKE_ACTIVITY)
        assert result["steps_today"] == 8500

    def test_extracts_calories(self):
        result = parse_activity(FAKE_ACTIVITY)
        assert result["active_energy_kcal"] == pytest.approx(420.5)

    def test_handles_missing_fields(self):
        result = parse_activity({"day": "2026-03-10"})
        assert result["activity_score"] is None
        assert result["steps_today"] is None


# ── _pick_main_sleep ───────────────────────────────────────────────────────────

class TestPickMainSleep:
    def test_returns_none_for_empty_list(self):
        assert _pick_main_sleep([]) is None

    def test_prefers_long_sleep_type(self):
        sessions = [
            {"type": "nap", "total_sleep_duration": 3600},
            {"type": "long_sleep", "total_sleep_duration": 25200},
        ]
        result = _pick_main_sleep(sessions)
        assert result["type"] == "long_sleep"

    def test_returns_longest_when_no_long_sleep(self):
        sessions = [
            {"type": "nap", "total_sleep_duration": 3600},
            {"type": "nap", "total_sleep_duration": 7200},
        ]
        result = _pick_main_sleep(sessions)
        assert result["total_sleep_duration"] == 7200

    def test_single_session_returned(self):
        sessions = [FAKE_SLEEP_SESSION]
        result = _pick_main_sleep(sessions)
        assert result is FAKE_SLEEP_SESSION


# ── _extract_hrv_avg ───────────────────────────────────────────────────────────

class TestExtractHrvAvg:
    def test_flat_list_of_values(self):
        session = {"hrv": {"items": [40.0, 50.0, 60.0]}}
        avg = _extract_hrv_avg(session)
        assert avg == pytest.approx(50.0)

    def test_nested_block_format(self):
        session = {
            "hrv": {
                "items": [
                    {"interval": 300, "items": [40.0, 50.0], "timestamp": "T"},
                    {"interval": 300, "items": [60.0], "timestamp": "T2"},
                ]
            }
        }
        avg = _extract_hrv_avg(session)
        assert avg == pytest.approx(50.0)  # (40+50+60)/3

    def test_returns_none_when_no_hrv(self):
        assert _extract_hrv_avg({}) is None

    def test_filters_none_values(self):
        session = {"hrv": {"items": [40.0, None, 60.0]}}
        avg = _extract_hrv_avg(session)
        assert avg == pytest.approx(50.0)


# ── OuraClient.build_biometric_payload ────────────────────────────────────────

class TestBuildBiometricPayload:
    def test_payload_has_user_id(self):
        client = _mock_oura_client(
            readiness_records=[FAKE_READINESS],
            sleep_score_records=[FAKE_SLEEP_SCORE],
            sleep_sessions=[FAKE_SLEEP_SESSION],
            activity_records=[FAKE_ACTIVITY],
        )
        payload = client.build_biometric_payload("user-abc")
        assert payload["user_id"] == "user-abc"

    def test_payload_source_is_oura(self):
        client = _mock_oura_client()
        payload = client.build_biometric_payload("u1")
        assert payload["source"] == "oura"

    def test_payload_contains_readiness_score(self):
        client = _mock_oura_client(readiness_records=[FAKE_READINESS])
        payload = client.build_biometric_payload("u1")
        assert payload.get("readiness_score") == 82

    def test_payload_contains_sleep_score(self):
        client = _mock_oura_client(sleep_score_records=[FAKE_SLEEP_SCORE])
        payload = client.build_biometric_payload("u1")
        assert payload.get("sleep_score") == 88

    def test_payload_contains_hrv_from_sleep_session(self):
        client = _mock_oura_client(sleep_sessions=[FAKE_SLEEP_SESSION])
        payload = client.build_biometric_payload("u1")
        assert "hrv_rmssd" in payload
        assert payload["hrv_rmssd"] is not None

    def test_payload_contains_resting_hr(self):
        client = _mock_oura_client(sleep_sessions=[FAKE_SLEEP_SESSION])
        payload = client.build_biometric_payload("u1")
        assert payload.get("resting_heart_rate") == 50.0

    def test_payload_contains_sleep_hours(self):
        client = _mock_oura_client(sleep_sessions=[FAKE_SLEEP_SESSION])
        payload = client.build_biometric_payload("u1")
        assert payload.get("sleep_total_hours") == pytest.approx(7.0)

    def test_payload_contains_activity_score(self):
        client = _mock_oura_client(activity_records=[FAKE_ACTIVITY])
        payload = client.build_biometric_payload("u1")
        assert payload.get("activity_score") == 75

    def test_payload_contains_steps(self):
        client = _mock_oura_client(activity_records=[FAKE_ACTIVITY])
        payload = client.build_biometric_payload("u1")
        assert payload.get("steps_today") == 8500

    def test_payload_has_fetched_at(self):
        client = _mock_oura_client()
        payload = client.build_biometric_payload("u1")
        assert isinstance(payload.get("fetched_at"), float)

    def test_payload_graceful_on_all_empty(self):
        client = _mock_oura_client(
            readiness_records=[],
            sleep_score_records=[],
            sleep_sessions=[],
            activity_records=[],
        )
        payload = client.build_biometric_payload("u1")
        assert payload["user_id"] == "u1"

    def test_payload_graceful_on_api_error(self):
        client = OuraClient.__new__(OuraClient)
        client._token = "fake"
        client.get_daily_readiness = mock.MagicMock(
            side_effect=RuntimeError("api error")
        )
        client.get_daily_sleep = mock.MagicMock(return_value=[])
        client.get_sleep_sessions = mock.MagicMock(return_value=[])
        client.get_daily_activity = mock.MagicMock(return_value=[])
        payload = client.build_biometric_payload("u1")
        assert "user_id" in payload


# ── OuraClient constructor ─────────────────────────────────────────────────────

class TestOuraClientInit:
    def test_empty_token_raises(self):
        with pytest.raises(ValueError, match="access_token"):
            OuraClient("")

    def test_none_token_raises(self):
        with pytest.raises(Exception):
            OuraClient(None)  # type: ignore

    def test_valid_token_accepted(self):
        client = OuraClient("valid-oura-token")
        assert client._token == "valid-oura-token"


# ── Utility functions ──────────────────────────────────────────────────────────

class TestUtilities:
    def test_safe_float_int(self):
        assert _safe_float(42) == 42.0

    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_string(self):
        assert _safe_float("1.5") == pytest.approx(1.5)

    def test_safe_float_bad_string(self):
        assert _safe_float("bad") is None

    def test_safe_int_float(self):
        assert _safe_int(3.9) == 3

    def test_safe_int_none(self):
        assert _safe_int(None) is None

    def test_safe_int_string(self):
        assert _safe_int("7") == 7
