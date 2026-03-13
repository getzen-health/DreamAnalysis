"""Tests for Whoop API v2 sync module.

All network calls are monkeypatched — no real Whoop credentials needed.
Covers:
- OAuth2 URL construction
- Token exchange / refresh (mocked HTTP)
- WhoopClient metric extraction
- parse_recovery / parse_sleep / parse_cycle helpers
- BiometricPayload schema completeness
- Error handling (API errors, missing data, env vars not set)
"""
from __future__ import annotations

import os
import sys
import json
import types
import unittest.mock as mock
from typing import Any, Dict
from urllib.error import HTTPError

import pytest

# Ensure ml/ is on sys.path
_ML_DIR = os.path.join(os.path.dirname(__file__), "..")
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from health.whoop_sync import (
    authorization_url,
    exchange_code,
    refresh_tokens,
    parse_recovery,
    parse_sleep,
    parse_cycle,
    WhoopClient,
    _safe_float,
    _WHOOP_AUTH_URL,
    _WHOOP_TOKEN_URL,
    _REQUIRED_SCOPES,
)


# ── Fixtures / helpers ─────────────────────────────────────────────────────────

FAKE_RECOVERY = {
    "id": "r-001",
    "user_calibrating": False,
    "score": {
        "recovery_score": 78,
        "hrv_rmssd_milli": 54.2,
        "resting_heart_rate": 52.0,
        "skin_temp_celsius": 0.3,
    },
}

FAKE_SLEEP = {
    "id": "s-001",
    "score": {
        "sleep_performance_percentage": 85,
        "stage_summary": {
            "total_in_bed_time_milli": 28_800_000,   # 8 hours
            "total_awake_time_milli": 1_800_000,      # 30 min
            "total_rem_sleep_time_milli": 5_400_000,  # 1.5 hours
            "total_slow_wave_sleep_time_milli": 3_600_000,  # 1 hour
        },
    },
}

FAKE_CYCLE = {
    "id": "c-001",
    "score": {
        "strain": 12.5,
        "kilojoule": 2000.0,
    },
}


def _mock_client_with(recovery=None, sleep=None, cycle=None):
    """Build a WhoopClient stub that returns the provided fixtures."""
    client = WhoopClient.__new__(WhoopClient)
    client._token = "fake-token"
    client.get_latest_recovery = mock.MagicMock(return_value=recovery)
    client.get_latest_sleep = mock.MagicMock(return_value=sleep)
    client.get_latest_cycle = mock.MagicMock(return_value=cycle)
    return client


# ── OAuth2 URL construction ────────────────────────────────────────────────────

class TestAuthorizationUrl:
    def test_url_starts_with_whoop_auth_base(self):
        url = authorization_url(
            client_id="test-id",
            redirect_uri="https://example.com/cb",
        )
        assert url.startswith(_WHOOP_AUTH_URL)

    def test_url_contains_client_id(self):
        url = authorization_url(
            client_id="my-client-id",
            redirect_uri="https://example.com/cb",
        )
        assert "my-client-id" in url

    def test_url_contains_response_type_code(self):
        url = authorization_url(
            client_id="cid",
            redirect_uri="https://example.com/cb",
        )
        assert "response_type=code" in url

    def test_url_contains_required_scopes(self):
        url = authorization_url(
            client_id="cid",
            redirect_uri="https://example.com/cb",
        )
        # scope param should be present
        assert "scope=" in url

    def test_url_contains_state_when_provided(self):
        url = authorization_url(
            client_id="cid",
            redirect_uri="https://example.com/cb",
            state="csrf-abc",
        )
        assert "csrf-abc" in url

    def test_url_omits_state_when_not_provided(self):
        url = authorization_url(
            client_id="cid",
            redirect_uri="https://example.com/cb",
        )
        assert "state=" not in url

    def test_raises_on_missing_client_id(self, monkeypatch):
        monkeypatch.delenv("WHOOP_CLIENT_ID", raising=False)
        with pytest.raises(ValueError, match="WHOOP_CLIENT_ID"):
            authorization_url(redirect_uri="https://example.com/cb")

    def test_raises_on_missing_redirect_uri(self, monkeypatch):
        monkeypatch.delenv("WHOOP_REDIRECT_URI", raising=False)
        with pytest.raises(ValueError, match="WHOOP_REDIRECT_URI"):
            authorization_url(client_id="cid")

    def test_falls_back_to_env_var(self, monkeypatch):
        monkeypatch.setenv("WHOOP_CLIENT_ID", "env-cid")
        monkeypatch.setenv("WHOOP_REDIRECT_URI", "https://env.example.com/cb")
        url = authorization_url()
        assert "env-cid" in url

    def test_required_scopes_include_offline(self):
        assert "offline" in _REQUIRED_SCOPES

    def test_required_scopes_include_recovery(self):
        assert "read:recovery" in _REQUIRED_SCOPES


# ── Token exchange (mocked HTTP) ───────────────────────────────────────────────

class TestExchangeCode:
    def _fake_urlopen(self, response_body: Dict):
        """Return a context manager that yields a fake HTTP response."""
        body = json.dumps(response_body).encode()

        class FakeResponse:
            def read(self):
                return body
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass

        return FakeResponse()

    def test_exchange_returns_tokens(self, monkeypatch):
        fake_tokens = {
            "access_token": "acc-xyz",
            "refresh_token": "ref-xyz",
            "expires_in": 3600,
        }
        monkeypatch.setenv("WHOOP_CLIENT_ID", "cid")
        monkeypatch.setenv("WHOOP_CLIENT_SECRET", "csecret")
        monkeypatch.setenv("WHOOP_REDIRECT_URI", "https://example.com/cb")

        with mock.patch("urllib.request.urlopen", return_value=self._fake_urlopen(fake_tokens)):
            result = exchange_code("auth-code-123")

        assert result["access_token"] == "acc-xyz"
        assert result["refresh_token"] == "ref-xyz"

    def test_exchange_raises_on_missing_credentials(self, monkeypatch):
        monkeypatch.delenv("WHOOP_CLIENT_ID", raising=False)
        monkeypatch.delenv("WHOOP_CLIENT_SECRET", raising=False)
        with pytest.raises(ValueError):
            exchange_code("code", client_id=None, client_secret=None)

    def test_exchange_wraps_http_errors(self, monkeypatch):
        monkeypatch.setenv("WHOOP_CLIENT_ID", "cid")
        monkeypatch.setenv("WHOOP_CLIENT_SECRET", "csecret")
        monkeypatch.setenv("WHOOP_REDIRECT_URI", "https://example.com/cb")

        with mock.patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            with pytest.raises(RuntimeError, match="failed"):
                exchange_code("bad-code")


class TestRefreshTokens:
    def test_refresh_returns_new_tokens(self, monkeypatch):
        fake_tokens = {
            "access_token": "new-acc",
            "refresh_token": "new-ref",
            "expires_in": 3600,
        }
        monkeypatch.setenv("WHOOP_CLIENT_ID", "cid")
        monkeypatch.setenv("WHOOP_CLIENT_SECRET", "csecret")

        body = json.dumps(fake_tokens).encode()

        class FakeResp:
            def read(self): return body
            def __enter__(self): return self
            def __exit__(self, *_): pass

        with mock.patch("urllib.request.urlopen", return_value=FakeResp()):
            result = refresh_tokens("old-refresh")

        assert result["access_token"] == "new-acc"

    def test_refresh_raises_on_missing_credentials(self, monkeypatch):
        monkeypatch.delenv("WHOOP_CLIENT_ID", raising=False)
        monkeypatch.delenv("WHOOP_CLIENT_SECRET", raising=False)
        with pytest.raises(ValueError):
            refresh_tokens("tok")


# ── parse_recovery ─────────────────────────────────────────────────────────────

class TestParseRecovery:
    def test_extracts_all_fields(self):
        result = parse_recovery(FAKE_RECOVERY)
        assert result["recovery_score"] == 78.0
        assert result["hrv_rmssd"] == 54.2
        assert result["resting_heart_rate"] == 52.0
        assert result["skin_temperature_deviation"] == 0.3

    def test_user_calibrating_flag(self):
        result = parse_recovery(FAKE_RECOVERY)
        assert result["user_calibrating"] is False

    def test_handles_missing_score_block(self):
        result = parse_recovery({"id": "x", "user_calibrating": False})
        assert result["recovery_score"] is None
        assert result["hrv_rmssd"] is None

    def test_handles_partial_score(self):
        record = {"score": {"recovery_score": 60}}
        result = parse_recovery(record)
        assert result["recovery_score"] == 60.0
        assert result["hrv_rmssd"] is None


# ── parse_sleep ────────────────────────────────────────────────────────────────

class TestParseSleep:
    def test_extracts_sleep_performance(self):
        result = parse_sleep(FAKE_SLEEP)
        assert result["sleep_performance_pct"] == 85.0

    def test_extracts_total_hours(self):
        result = parse_sleep(FAKE_SLEEP)
        # 28_800_000 ms = 8 hours
        assert result["sleep_total_hours"] == 8.0

    def test_extracts_rem_hours(self):
        result = parse_sleep(FAKE_SLEEP)
        # 5_400_000 ms = 1.5 hours
        assert result["sleep_rem_hours"] == 1.5

    def test_extracts_deep_hours(self):
        result = parse_sleep(FAKE_SLEEP)
        # 3_600_000 ms = 1.0 hours
        assert result["sleep_deep_hours"] == 1.0

    def test_calculates_efficiency(self):
        result = parse_sleep(FAKE_SLEEP)
        # (28_800_000 - 1_800_000) / 28_800_000 * 100 = 93.75%
        assert result["sleep_efficiency"] is not None
        assert 90 <= result["sleep_efficiency"] <= 95

    def test_handles_empty_record(self):
        result = parse_sleep({"id": "x", "score": {}})
        assert result["sleep_performance_pct"] is None
        assert result["sleep_total_hours"] is None


# ── parse_cycle ────────────────────────────────────────────────────────────────

class TestParseCycle:
    def test_extracts_strain(self):
        result = parse_cycle(FAKE_CYCLE)
        assert result["strain_score"] == 12.5

    def test_converts_kilojoules_to_kcal(self):
        result = parse_cycle(FAKE_CYCLE)
        # 2000 kJ / 4.184 = ~478 kcal
        assert result["active_energy_kcal"] is not None
        assert 470 <= result["active_energy_kcal"] <= 490

    def test_handles_missing_score(self):
        result = parse_cycle({"id": "x"})
        assert result["strain_score"] is None
        assert result["active_energy_kcal"] is None


# ── WhoopClient.build_biometric_payload ───────────────────────────────────────

class TestBuildBiometricPayload:
    def test_payload_has_user_id(self):
        client = _mock_client_with(
            recovery=FAKE_RECOVERY, sleep=FAKE_SLEEP, cycle=FAKE_CYCLE
        )
        payload = client.build_biometric_payload("user-123")
        assert payload["user_id"] == "user-123"

    def test_payload_source_is_whoop(self):
        client = _mock_client_with(recovery=FAKE_RECOVERY)
        payload = client.build_biometric_payload("u1")
        assert payload["source"] == "whoop"

    def test_payload_contains_hrv(self):
        client = _mock_client_with(recovery=FAKE_RECOVERY)
        payload = client.build_biometric_payload("u1")
        assert "hrv_rmssd" in payload
        assert payload["hrv_rmssd"] == 54.2

    def test_payload_contains_resting_hr(self):
        client = _mock_client_with(recovery=FAKE_RECOVERY)
        payload = client.build_biometric_payload("u1")
        assert payload["resting_heart_rate"] == 52.0

    def test_payload_contains_recovery_score(self):
        client = _mock_client_with(recovery=FAKE_RECOVERY)
        payload = client.build_biometric_payload("u1")
        assert payload["recovery_score"] == 78.0

    def test_payload_contains_sleep_hours(self):
        client = _mock_client_with(sleep=FAKE_SLEEP)
        payload = client.build_biometric_payload("u1")
        assert "sleep_total_hours" in payload
        assert payload["sleep_total_hours"] == 8.0

    def test_payload_contains_strain(self):
        client = _mock_client_with(cycle=FAKE_CYCLE)
        payload = client.build_biometric_payload("u1")
        assert "strain_score" in payload
        assert payload["strain_score"] == 12.5

    def test_payload_has_fetched_at_timestamp(self):
        client = _mock_client_with()
        payload = client.build_biometric_payload("u1")
        assert "fetched_at" in payload
        assert isinstance(payload["fetched_at"], float)
        assert payload["fetched_at"] > 0

    def test_payload_graceful_on_all_none(self):
        """No exception when all data sources return None."""
        client = _mock_client_with(recovery=None, sleep=None, cycle=None)
        payload = client.build_biometric_payload("u1")
        assert payload["user_id"] == "u1"
        assert payload["source"] == "whoop"

    def test_payload_graceful_on_api_error(self):
        """API errors in individual fetchers should not crash the whole payload."""
        client = WhoopClient.__new__(WhoopClient)
        client._token = "fake"
        client.get_latest_recovery = mock.MagicMock(
            side_effect=RuntimeError("network error")
        )
        client.get_latest_sleep = mock.MagicMock(return_value=None)
        client.get_latest_cycle = mock.MagicMock(return_value=None)
        payload = client.build_biometric_payload("u1")
        assert "user_id" in payload


# ── _safe_float ────────────────────────────────────────────────────────────────

class TestSafeFloat:
    def test_converts_int(self):
        assert _safe_float(42) == 42.0

    def test_converts_string(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_returns_none_on_none(self):
        assert _safe_float(None) is None

    def test_returns_none_on_empty_string(self):
        # empty string can't be converted
        assert _safe_float("") is None

    def test_returns_none_on_non_numeric(self):
        assert _safe_float("abc") is None


# ── WhoopClient constructor ────────────────────────────────────────────────────

class TestWhoopClientInit:
    def test_empty_token_raises(self):
        with pytest.raises(ValueError, match="access_token"):
            WhoopClient("")

    def test_none_token_raises(self):
        with pytest.raises(Exception):
            WhoopClient(None)  # type: ignore

    def test_valid_token_sets_attribute(self):
        client = WhoopClient("valid-token")
        assert client._token == "valid-token"
