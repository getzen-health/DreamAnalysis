"""Tests for API key authentication middleware.

These tests build a minimal FastAPI app with the middleware to avoid
triggering the 60-90s deferred ML model loading in main.py.
"""
import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Ensure ml/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_app(api_key: str | None) -> FastAPI:
    """Create a minimal FastAPI app with APIKeyMiddleware and two routes.

    Args:
        api_key: Value to set for ML_API_KEY env var, or None to unset it.

    Returns:
        FastAPI app with middleware applied.
    """
    # Set/unset the env var BEFORE importing the middleware so the
    # middleware __init__ picks up the right value.
    if api_key is not None:
        os.environ["ML_API_KEY"] = api_key
    else:
        os.environ.pop("ML_API_KEY", None)

    # Force reimport so APIKeyMiddleware re-reads the env var.
    import importlib
    import api.auth
    importlib.reload(api.auth)
    from api.auth import APIKeyMiddleware

    app = FastAPI()
    app.add_middleware(APIKeyMiddleware)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/models/status")
    async def models_status():
        return {"models": "loaded"}

    return app


class TestAPIKeyAuth:
    def test_health_endpoint_no_auth_required(self):
        """Health check should always be accessible even when key is set."""
        app = _make_app("test-secret-key")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_protected_endpoint_rejected_without_key(self):
        """Protected endpoints must return 401 when no API key is provided."""
        app = _make_app("test-secret-key")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/models/status")
        assert resp.status_code == 401

    def test_protected_endpoint_rejected_with_wrong_key(self):
        """Protected endpoints must return 401 when the wrong key is provided."""
        app = _make_app("test-secret-key")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/models/status", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_protected_endpoint_allowed_with_correct_key(self):
        """Protected endpoints must be accessible with the correct API key."""
        app = _make_app("test-secret-key")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/models/status", headers={"X-API-Key": "test-secret-key"})
        assert resp.status_code == 200

    def test_no_auth_when_key_not_configured(self):
        """When ML_API_KEY is unset, all requests pass (dev mode)."""
        app = _make_app(None)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/models/status")
        assert resp.status_code != 401
        assert resp.status_code == 200

    def test_health_still_accessible_without_key_header(self):
        """Health endpoint is public even when key is configured."""
        app = _make_app("super-secret")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_error_response_includes_detail(self):
        """401 response body should explain what header to set."""
        app = _make_app("test-secret-key")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/models/status")
        assert resp.status_code == 401
        body = resp.json()
        assert "detail" in body
        assert "X-API-Key" in body["detail"]
