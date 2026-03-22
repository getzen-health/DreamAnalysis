"""CORS origin allowlist for the ML backend.

Centralizes allowed origins so both main.py and tests can access the same list.
Override with CORS_ORIGINS environment variable (comma-separated).
"""

import os
from typing import List

# Default allowed origins — explicitly listed, no wildcard
_DEFAULT_ORIGINS = [
    # Local development
    "http://localhost:4000",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:3030",
    "http://localhost:8080",
    # Capacitor (mobile)
    "capacitor://localhost",
    "http://localhost",
    # Production — Vercel
    "https://dream-analysis.vercel.app",
    "https://neural-dream-workshop.vercel.app",
]

# Regex for dynamic subdomains (ngrok, Vercel previews)
ORIGIN_REGEX = (
    r"https://.*\.ngrok(-free)?\.app"
    r"|https://.*\.ngrok\.io"
    r"|https://.*\.ngrok-free\.dev"
    r"|https://.*\.vercel\.app"
    r"|capacitor://.*"
    r"|ionic://.*"
)


def get_allowed_origins() -> List[str]:
    """Return the list of allowed CORS origins.

    If CORS_ORIGINS env var is set (comma-separated), use that instead of defaults.
    The special value '*' is rejected — use explicit origins only.
    """
    env_origins = os.environ.get("CORS_ORIGINS", "").strip()

    if env_origins and env_origins != "*":
        return [o.strip() for o in env_origins.split(",") if o.strip()]

    return list(_DEFAULT_ORIGINS)
