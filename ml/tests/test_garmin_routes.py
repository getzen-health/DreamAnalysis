"""Tests for Garmin sync module and FastAPI routes."""

import os
import pytest
from unittest.mock import MagicMock, patch


# ── GarminClient initialization ────────────────────────────────────────────────

class TestGarminClientInit:
    def test_valid_init(self):
        from health.garmin_sync import GarminClient
        client = GarminClient(
            access_token="tok_abc",
            access_token_secret="sec_xyz",
        )
        assert client._access_token == "tok_abc"
        assert client._access_token_secret == "sec_xyz"

    def test_raises_on_empty_access_token(self):
        from health.garmin_sync import GarminClient
        with pytest.raises(ValueError, match="access_token"):
            GarminClient(access_token="", access_token_secret="secret")

    def test_raises_on_empty_access_token_secret(self):
        from health.garmin_sync import GarminClient
        with pytest.raises(ValueError, match="access_token_secret"):
            GarminClient(access_token="token", access_token_secret="")

    def test_optional_consumer_credentials(self):
        from health.garmin_sync import GarminClient
        client = GarminClient(
            access_token="tok",
            access_token_secret="sec",
            consumer_key="ck",
            consumer_secret="cs",
        )
        assert client._consumer_key == "ck"
        assert client._consumer_secret == "cs"

    def test_consumer_credentials_default_none(self):
        from health.garmin_sync import GarminClient
        client = GarminClient(access_token="tok", access_token_secret="sec")
        assert client._consumer_key is None
        assert client._consumer_secret is None


# ── OAuth helpers ───────────────────────────────────────────────────────────────

class TestGetRequestToken:
    def test_raises_value_error_without_consumer_key(self):
        from health.garmin_sync import get_request_token
        with pytest.raises(ValueError, match="GARMIN_CONSUMER_KEY"):
            # No env var set, no explicit key → ValueError
            get_request_token(consumer_key=None, consumer_secret=None)

    def test_raises_value_error_without_consumer_secret(self):
        from health.garmin_sync import get_request_token
        with pytest.raises(ValueError, match="GARMIN_CONSUMER_SECRET"):
            get_request_token(consumer_key="ck", consumer_secret=None)

    def test_uses_env_vars_when_not_explicitly_passed(self, monkeypatch):
        """get_request_token falls back to env vars for credentials."""
        # We can't call Garmin servers in tests, but we can verify env var lookup
        # by confirming ValueError is NOT raised when env vars are set,
        # and that an ImportError or RuntimeError is raised instead
        # (because requests_oauthlib will attempt a real HTTP request).
        monkeypatch.setenv("GARMIN_CONSUMER_KEY", "test_key")
        monkeypatch.setenv("GARMIN_CONSUMER_SECRET", "test_secret")
        from health.garmin_sync import get_request_token
        with pytest.raises((RuntimeError, ImportError, Exception)) as exc_info:
            get_request_token()
        # Must NOT be a ValueError about missing credentials
        assert "GARMIN_CONSUMER_KEY" not in str(exc_info.value)
        assert "GARMIN_CONSUMER_SECRET" not in str(exc_info.value)


class TestExchangeVerifier:
    def test_raises_value_error_without_credentials(self):
        from health.garmin_sync import exchange_verifier
        with pytest.raises(ValueError):
            exchange_verifier(
                oauth_token="tok",
                oauth_verifier="ver",
                oauth_token_secret="sec",
                consumer_key=None,
                consumer_secret=None,
            )

    def test_raises_runtime_error_on_network_failure(self, monkeypatch):
        """exchange_verifier wraps network errors as RuntimeError."""
        monkeypatch.setenv("GARMIN_CONSUMER_KEY", "ck")
        monkeypatch.setenv("GARMIN_CONSUMER_SECRET", "cs")

        # Mock requests_oauthlib to simulate a network failure
        mock_session = MagicMock()
        mock_session.fetch_access_token.side_effect = Exception("connection refused")

        with patch("health.garmin_sync._get_oauth1_session", return_value=mock_session):
            from health.garmin_sync import exchange_verifier
            with pytest.raises(RuntimeError, match="Garmin verifier exchange failed"):
                exchange_verifier(
                    oauth_token="tok",
                    oauth_verifier="ver",
                    oauth_token_secret="sec",
                    consumer_key="ck",
                    consumer_secret="cs",
                )


class TestGetAuthorizationUrl:
    def test_raises_value_error_without_credentials(self):
        from health.garmin_sync import get_authorization_url
        with pytest.raises(ValueError):
            get_authorization_url(oauth_token="tok", consumer_key=None, consumer_secret=None)

    def test_returns_authorization_url(self, monkeypatch):
        monkeypatch.setenv("GARMIN_CONSUMER_KEY", "ck")
        monkeypatch.setenv("GARMIN_CONSUMER_SECRET", "cs")

        mock_session = MagicMock()
        mock_session.authorization_url.return_value = "https://connect.garmin.com/oauthConfirm?oauth_token=tok"

        with patch("health.garmin_sync._get_oauth1_session", return_value=mock_session):
            from health.garmin_sync import get_authorization_url
            url = get_authorization_url(oauth_token="tok", consumer_key="ck", consumer_secret="cs")
        assert "garmin.com" in url
        assert "tok" in url


# ── build_biometric_payload ────────────────────────────────────────────────────

class TestBuildBiometricPayload:
    def _make_client_with_mock_data(self, daily_data=None, sleep_data=None, user_metrics_data=None):
        from health.garmin_sync import GarminClient
        client = GarminClient(access_token="tok", access_token_secret="sec")
        client.get_daily_summary = MagicMock(return_value=daily_data or [])
        client.get_sleep_data = MagicMock(return_value=sleep_data or [])
        client.get_user_metrics = MagicMock(return_value=user_metrics_data or [])
        return client

    def test_returns_dict_with_user_id_and_source(self):
        client = self._make_client_with_mock_data()
        payload = client.build_biometric_payload("u1")
        assert payload["user_id"] == "u1"
        assert payload["source"] == "garmin"

    def test_fetched_at_is_set(self):
        import time
        client = self._make_client_with_mock_data()
        before = time.time()
        payload = client.build_biometric_payload("u1")
        after = time.time()
        assert before <= payload["fetched_at"] <= after

    def test_body_battery_from_daily_summary(self):
        daily = [{
            "highestBodyBattery": 92,
            "lowestBodyBattery": 15,
            "averageStressLevel": 35,
            "maxStressLevel": 67,
            "restingHeartRateInBeatsPerMinute": 58,
            "steps": 9500,
            "activeKilocalories": 420.5,
        }]
        client = self._make_client_with_mock_data(daily_data=daily)
        payload = client.build_biometric_payload("u1")
        assert payload["body_battery_highest"] == 92
        assert payload["body_battery_lowest"] == 15
        assert payload["average_stress"] == 35
        assert payload["max_stress"] == 67
        assert payload["resting_heart_rate"] == 58.0
        assert payload["steps_today"] == 9500
        assert payload["active_energy_kcal"] == pytest.approx(420.5, abs=0.01)

    def test_sleep_data_mapping(self):
        sleep = [{
            "startTimeInSeconds": 1704067200,
            "durationInSeconds": 27000,     # 7.5 hours
            "deepSleepDurationInSeconds": 5400,  # 1.5 hours
            "lightSleepDurationInSeconds": 12600,  # 3.5 hours
            "remSleepInSeconds": 7200,      # 2 hours
            "averageSpo2Value": 97.2,
            "averageRespirationValue": 14.5,
            "sleepScores": {"overall": 82},
        }]
        client = self._make_client_with_mock_data(sleep_data=sleep)
        payload = client.build_biometric_payload("u1")
        assert payload["sleep_total_hours"] == pytest.approx(7.5, abs=0.01)
        assert payload["sleep_deep_hours"] == pytest.approx(1.5, abs=0.01)
        assert payload["sleep_rem_hours"] == pytest.approx(2.0, abs=0.01)
        assert payload["sleep_score"] == 82

    def test_user_metrics_hrv_mapping(self):
        metrics = [{
            "calendarDate": "2024-01-01",
            "lastNight": 52,
            "hrvWeeklyAverage": 48,
            "hrvStatus": "BALANCED",
            "averageSpo2": 96.8,
        }]
        client = self._make_client_with_mock_data(user_metrics_data=metrics)
        payload = client.build_biometric_payload("u1")
        assert payload["hrv_last_night"] == 52
        assert payload["hrv_weekly_average"] == 48
        assert payload["hrv_status"] == "BALANCED"

    def test_empty_api_responses_return_minimal_payload(self):
        client = self._make_client_with_mock_data()
        payload = client.build_biometric_payload("u1")
        # Only guaranteed fields
        assert "user_id" in payload
        assert "source" in payload
        assert "fetched_at" in payload
        # Optional fields should not be present when data is empty
        assert "body_battery_highest" not in payload
        assert "sleep_total_hours" not in payload

    def test_daily_summary_exception_does_not_crash(self):
        from health.garmin_sync import GarminClient
        client = GarminClient(access_token="tok", access_token_secret="sec")
        client.get_daily_summary = MagicMock(side_effect=RuntimeError("Garmin API GET /dailies failed"))
        client.get_sleep_data = MagicMock(return_value=[])
        client.get_user_metrics = MagicMock(return_value=[])
        # Should not raise — errors are caught and logged
        payload = client.build_biometric_payload("u1")
        assert payload["user_id"] == "u1"


# ── parse_webhook_payload ──────────────────────────────────────────────────────

class TestParseWebhookPayload:
    def test_dailies_parsed(self):
        from health.garmin_sync import parse_webhook_payload
        payload = {
            "dailies": [
                {
                    "summaryId": "abc",
                    "calendarDate": "2024-01-01",
                    "bodyBatteryChargedValue": 85,
                    "averageStressLevel": 30,
                }
            ]
        }
        result = parse_webhook_payload(payload)
        assert "dailies" in result
        assert len(result["dailies"]) == 1
        assert result["dailies"][0]["body_battery_charged"] == 85
        assert result["dailies"][0]["average_stress"] == 30

    def test_sleeps_parsed(self):
        from health.garmin_sync import parse_webhook_payload
        payload = {
            "sleeps": [
                {
                    "summaryId": "s1",
                    "calendarDate": "2024-01-01",
                    "durationInSeconds": 28800,
                }
            ]
        }
        result = parse_webhook_payload(payload)
        assert "sleeps" in result
        assert result["sleeps"][0]["duration_seconds"] == 28800

    def test_unknown_keys_pass_through(self):
        from health.garmin_sync import parse_webhook_payload
        payload = {"customField": {"value": 42}}
        result = parse_webhook_payload(payload)
        assert result["customField"] == {"value": 42}

    def test_empty_payload_returns_empty_dict(self):
        from health.garmin_sync import parse_webhook_payload
        result = parse_webhook_payload({})
        assert result == {}

    def test_epochs_pass_through_unchanged(self):
        from health.garmin_sync import parse_webhook_payload
        epochs_data = [{"epochId": "e1", "steps": 200}]
        result = parse_webhook_payload({"epochs": epochs_data})
        assert result["epochs"] == epochs_data

    def test_user_metrics_parsed(self):
        from health.garmin_sync import parse_webhook_payload
        payload = {
            "userMetrics": [
                {
                    "calendarDate": "2024-01-01",
                    "hrvWeeklyAverage": 50,
                    "lastNight": 55,
                    "hrvStatus": "BALANCED",
                }
            ]
        }
        result = parse_webhook_payload(payload)
        assert "user_metrics" in result
        assert result["user_metrics"][0]["hrv_weekly_average"] == 50
        assert result["user_metrics"][0]["hrv_status"] == "BALANCED"


# ── Garmin FastAPI routes (endpoint existence + structure) ────────────────────

class TestGarminRouteEndpoints:
    """Verify the router defines all 8 expected endpoints."""

    def _get_route_paths(self):
        from api.routes.garmin import router
        return {route.path for route in router.routes}

    def test_status_endpoint_exists(self):
        paths = self._get_route_paths()
        assert "/garmin/status" in paths

    def test_auth_url_endpoint_exists(self):
        paths = self._get_route_paths()
        assert "/garmin/auth/url" in paths

    def test_auth_exchange_endpoint_exists(self):
        paths = self._get_route_paths()
        assert "/garmin/auth/exchange" in paths

    def test_webhook_endpoint_exists(self):
        paths = self._get_route_paths()
        assert "/garmin/webhook" in paths

    def test_sync_endpoint_exists(self):
        paths = self._get_route_paths()
        assert "/garmin/sync/{user_id}" in paths

    def test_body_battery_endpoint_exists(self):
        paths = self._get_route_paths()
        assert "/garmin/body-battery/{user_id}" in paths

    def test_stress_endpoint_exists(self):
        paths = self._get_route_paths()
        assert "/garmin/stress/{user_id}" in paths

    def test_sleep_endpoint_exists(self):
        paths = self._get_route_paths()
        assert "/garmin/sleep/{user_id}" in paths

    def test_router_prefix(self):
        from api.routes.garmin import router
        assert router.prefix == "/garmin"

    def test_router_tags(self):
        from api.routes.garmin import router
        assert "garmin" in router.tags

    def test_total_route_count_at_least_8(self):
        from api.routes.garmin import router
        assert len(router.routes) >= 8


# ── Garmin status route (unit test with TestClient) ────────────────────────────

class TestGarminStatusRoute:
    def _get_test_client(self):
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
            from api.routes.garmin import router
            app = FastAPI()
            app.include_router(router)
            return TestClient(app)
        except ImportError:
            pytest.skip("fastapi or httpx not installed")

    def test_status_unconfigured(self, monkeypatch):
        monkeypatch.delenv("GARMIN_CONSUMER_KEY", raising=False)
        monkeypatch.delenv("GARMIN_CONSUMER_SECRET", raising=False)
        monkeypatch.delenv("GARMIN_CALLBACK_URL", raising=False)
        client = self._get_test_client()
        response = client.get("/garmin/status")
        assert response.status_code == 200
        data = response.json()
        assert data["configured"] is False
        assert data["consumer_key_set"] is False
        assert data["consumer_secret_set"] is False
        assert data["oauth_version"] == "1.0a"

    def test_status_configured(self, monkeypatch):
        monkeypatch.setenv("GARMIN_CONSUMER_KEY", "test_key")
        monkeypatch.setenv("GARMIN_CONSUMER_SECRET", "test_secret")
        client = self._get_test_client()
        response = client.get("/garmin/status")
        assert response.status_code == 200
        data = response.json()
        assert data["configured"] is True
        assert data["consumer_key_set"] is True
        assert data["consumer_secret_set"] is True


# ── Helper utilities ───────────────────────────────────────────────────────────

class TestSafeConversions:
    def test_safe_int_converts_string(self):
        from health.garmin_sync import _safe_int
        assert _safe_int("42") == 42

    def test_safe_int_returns_none_for_none(self):
        from health.garmin_sync import _safe_int
        assert _safe_int(None) is None

    def test_safe_int_returns_none_for_invalid(self):
        from health.garmin_sync import _safe_int
        assert _safe_int("not_a_number") is None

    def test_safe_float_converts_string(self):
        from health.garmin_sync import _safe_float
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_safe_float_returns_none_for_none(self):
        from health.garmin_sync import _safe_float
        assert _safe_float(None) is None

    def test_safe_float_returns_none_for_invalid(self):
        from health.garmin_sync import _safe_float
        assert _safe_float("abc") is None

    def test_pick_latest_returns_max_by_key(self):
        from health.garmin_sync import _pick_latest
        records = [
            {"startTimeInSeconds": 100, "name": "old"},
            {"startTimeInSeconds": 500, "name": "new"},
            {"startTimeInSeconds": 300, "name": "mid"},
        ]
        result = _pick_latest(records, "startTimeInSeconds")
        assert result["name"] == "new"

    def test_pick_latest_returns_none_for_empty(self):
        from health.garmin_sync import _pick_latest
        assert _pick_latest([], "key") is None
