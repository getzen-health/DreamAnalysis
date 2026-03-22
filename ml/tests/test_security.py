"""Tests for ML backend security: API key auth, CORS, rate limiting, path sanitization.

Covers Issue #477 (zero-auth) and Issue #511 (path traversal hardening).
"""

import os
import time
import pytest
from unittest.mock import patch


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def app_with_auth():
    """Create a FastAPI app with ML_API_KEY set for auth testing."""
    with patch.dict(os.environ, {"ML_API_KEY": "test-secret-key-12345"}):
        # Must reimport to pick up the env var
        import importlib
        import api.auth as auth_mod
        importlib.reload(auth_mod)

        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.auth import APIKeyMiddleware

        app = FastAPI()
        app.add_middleware(APIKeyMiddleware)

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        @app.get("/status")
        async def status():
            return {"status": "ok"}

        @app.get("/api/models/status")
        async def models_status():
            return {"models": "loaded"}

        @app.post("/api/analyze-eeg")
        async def analyze():
            return {"result": "ok"}

        yield TestClient(app)


@pytest.fixture(scope="module")
def app_no_auth():
    """Create a FastAPI app without ML_API_KEY (dev mode)."""
    env = os.environ.copy()
    env.pop("ML_API_KEY", None)
    with patch.dict(os.environ, env, clear=True):
        import importlib
        import api.auth as auth_mod
        importlib.reload(auth_mod)

        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.auth import APIKeyMiddleware

        app = FastAPI()
        app.add_middleware(APIKeyMiddleware)

        @app.get("/health")
        async def health_noauth():
            return {"status": "healthy"}

        @app.get("/api/test")
        async def test_endpoint():
            return {"ok": True}

        yield TestClient(app)


# ── API Key Authentication Tests ─────────────────────────────────────────────


class TestAPIKeyMiddleware:
    """Test that API key middleware blocks unauthenticated requests."""

    def test_health_exempt_no_key_needed(self, app_with_auth):
        """Health endpoint should be accessible without API key."""
        r = app_with_auth.get("/health")
        assert r.status_code == 200

    def test_status_exempt_no_key_needed(self, app_with_auth):
        """Status endpoint should be accessible without API key."""
        r = app_with_auth.get("/status")
        assert r.status_code == 200

    def test_protected_endpoint_rejects_no_key(self, app_with_auth):
        """Protected endpoints should return 401 without API key."""
        r = app_with_auth.get("/api/models/status")
        assert r.status_code == 401
        assert "API key" in r.json()["detail"]

    def test_protected_endpoint_rejects_wrong_key(self, app_with_auth):
        """Protected endpoints should return 401 with wrong API key."""
        r = app_with_auth.get(
            "/api/models/status",
            headers={"X-API-Key": "wrong-key"},
        )
        assert r.status_code == 401

    def test_protected_endpoint_accepts_correct_key(self, app_with_auth):
        """Protected endpoints should work with correct API key."""
        r = app_with_auth.get(
            "/api/models/status",
            headers={"X-API-Key": "test-secret-key-12345"},
        )
        assert r.status_code == 200

    def test_post_endpoint_requires_key(self, app_with_auth):
        """POST endpoints also require API key."""
        r = app_with_auth.post("/api/analyze-eeg", json={})
        assert r.status_code == 401

    def test_post_endpoint_works_with_key(self, app_with_auth):
        """POST endpoints work with correct API key."""
        r = app_with_auth.post(
            "/api/analyze-eeg",
            json={},
            headers={"X-API-Key": "test-secret-key-12345"},
        )
        assert r.status_code == 200


class TestDevMode:
    """When ML_API_KEY is not set, all requests should pass (dev mode)."""

    def test_dev_mode_allows_all(self, app_no_auth):
        """Without ML_API_KEY, no auth required."""
        r = app_no_auth.get("/api/test")
        assert r.status_code == 200


# ── Rate Limiting Tests ──────────────────────────────────────────────────────


class TestRateLimiting:
    """Test that rate limiting blocks excessive requests."""

    def test_rate_limiter_import(self):
        """Rate limiter module should be importable."""
        from api.rate_limit import RateLimiter
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        assert limiter is not None

    def test_rate_limiter_allows_under_limit(self):
        """Requests under the limit should be allowed."""
        from api.rate_limit import RateLimiter
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed("127.0.0.1") is True

    def test_rate_limiter_blocks_over_limit(self):
        """Requests over the limit should be blocked."""
        from api.rate_limit import RateLimiter
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            limiter.is_allowed("10.0.0.1")
        assert limiter.is_allowed("10.0.0.1") is False

    def test_rate_limiter_per_ip_isolation(self):
        """Different IPs should have independent limits."""
        from api.rate_limit import RateLimiter
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.is_allowed("10.0.0.1")
        limiter.is_allowed("10.0.0.1")
        # IP 1 is at limit
        assert limiter.is_allowed("10.0.0.1") is False
        # IP 2 should still be allowed
        assert limiter.is_allowed("10.0.0.2") is True

    def test_rate_limiter_window_expiry(self):
        """Requests should be allowed after the window expires."""
        from api.rate_limit import RateLimiter
        limiter = RateLimiter(max_requests=1, window_seconds=1)
        assert limiter.is_allowed("10.0.0.3") is True
        assert limiter.is_allowed("10.0.0.3") is False
        # Wait for window to expire
        time.sleep(1.1)
        assert limiter.is_allowed("10.0.0.3") is True


# ── CORS Tests ───────────────────────────────────────────────────────────────


class TestCORSConfiguration:
    """Test that CORS is properly configured with an allowlist."""

    def test_cors_allows_localhost(self):
        """localhost origins should be in the allowlist."""
        from api.cors import get_allowed_origins
        origins = get_allowed_origins()
        assert "http://localhost:4000" in origins
        assert "http://localhost:5173" in origins

    def test_cors_allows_vercel(self):
        """Vercel app origins should be in the allowlist."""
        from api.cors import get_allowed_origins
        origins = get_allowed_origins()
        # At least one vercel domain should be present
        has_vercel = any(".vercel.app" in o for o in origins)
        assert has_vercel

    def test_cors_no_wildcard_default(self):
        """Default CORS should not be wildcard *."""
        from api.cors import get_allowed_origins
        origins = get_allowed_origins()
        assert "*" not in origins

    def test_cors_env_override(self):
        """CORS_ORIGINS env var should override defaults."""
        with patch.dict(os.environ, {"CORS_ORIGINS": "https://custom.app,https://other.app"}):
            from api.cors import get_allowed_origins
            origins = get_allowed_origins()
            assert "https://custom.app" in origins
            assert "https://other.app" in origins


# ── Path Sanitization Tests ──────────────────────────────────────────────────


class TestPathSanitization:
    """Test that user_id is sanitized in all file path constructions."""

    def test_sanitize_id_rejects_traversal(self):
        """Path traversal attempts should be rejected."""
        from api.routes._shared import sanitize_id
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            sanitize_id("../../etc/passwd")

    def test_sanitize_id_rejects_slashes(self):
        from api.routes._shared import sanitize_id
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            sanitize_id("user/../../root")

    def test_sanitize_id_rejects_null_bytes(self):
        from api.routes._shared import sanitize_id
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            sanitize_id("user\x00evil")

    def test_sanitize_id_accepts_normal_ids(self):
        from api.routes._shared import sanitize_id
        assert sanitize_id("user-123_abc") == "user-123_abc"
        assert sanitize_id("a" * 128) == "a" * 128

    def test_sanitize_id_rejects_empty(self):
        from api.routes._shared import sanitize_id
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            sanitize_id("")

    def test_sanitize_id_rejects_dots_only(self):
        from api.routes._shared import sanitize_id
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            sanitize_id("..")
        with pytest.raises(HTTPException):
            sanitize_id(".")
