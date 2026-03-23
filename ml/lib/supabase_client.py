"""Supabase client singleton for the ML backend.

Returns None when SUPABASE_URL / SUPABASE_SERVICE_KEY are not set,
allowing the rest of the app to degrade gracefully.
"""

import logging
import os

logger = logging.getLogger(__name__)

_client = None


def get_supabase():
    """Return a Supabase client (cached) or None if not configured."""
    global _client
    if _client is not None:
        return _client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

    if not url or not key:
        logger.debug("Supabase not configured (SUPABASE_URL or key missing)")
        return None

    try:
        from supabase import create_client

        _client = create_client(url, key)
        logger.info("Supabase client initialized for ML backend")
        return _client
    except ImportError:
        logger.warning("supabase-py not installed — pip install supabase")
        return None
    except Exception as exc:
        logger.error("Failed to create Supabase client: %s", exc)
        return None


def reset_client():
    """Reset the cached client (useful for testing)."""
    global _client
    _client = None
