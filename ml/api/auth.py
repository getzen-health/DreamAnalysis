"""API key authentication middleware for the ML backend.

Validates requests using a shared API key passed via the X-API-Key header.
The key is loaded from the ML_API_KEY environment variable.

When ML_API_KEY is not set, authentication is disabled (development mode).
This allows local development without configuring a key.
"""
import os
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

log = logging.getLogger(__name__)

# Endpoints that don't require authentication
_PUBLIC_PATHS = frozenset({
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
})


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware that validates X-API-Key header against ML_API_KEY env var.

    If ML_API_KEY is not set, all requests are allowed (dev mode).
    """

    def __init__(self, app):
        super().__init__(app)
        self.api_key = os.environ.get("ML_API_KEY")
        if self.api_key:
            log.info("[auth] API key authentication enabled")
        else:
            log.warning(
                "[auth] ML_API_KEY not set — authentication disabled (dev mode)"
            )

    async def dispatch(self, request: Request, call_next):
        # Skip auth if no key configured (dev mode)
        if not self.api_key:
            return await call_next(request)

        # Skip auth for public endpoints
        path = request.url.path.rstrip("/") or "/"
        if path in _PUBLIC_PATHS:
            return await call_next(request)

        # Skip auth for WebSocket upgrades (handled separately)
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        # Validate API key
        provided_key = request.headers.get("x-api-key")
        if not provided_key or provided_key != self.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. Set X-API-Key header."},
            )

        return await call_next(request)
