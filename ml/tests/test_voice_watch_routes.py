"""Tests for upgraded voice-watch endpoints."""
import base64
import io
import struct
import time
import wave

import pytest


def _make_silent_wav_b64(seconds: int = 3, sr: int = 22050) -> str:
    """Create a silent WAV file encoded as base64."""
    buf = io.BytesIO()
    n = sr * seconds
    data = struct.pack(f"<{n}h", *([0] * n))
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data)
    return base64.b64encode(buf.getvalue()).decode()


def test_cache_store_and_retrieve():
    """Cache endpoint stores result; latest endpoint retrieves it."""
    from api.routes.voice_watch import _VOICE_CACHE
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from api.routes.voice_watch import cache_voice_result, get_latest_voice, CacheRequest

    _VOICE_CACHE.clear()

    payload = {
        "emotion": "happy",
        "valence": 0.5,
        "arousal": 0.6,
        "confidence": 0.85,
        "probabilities": {},
        "model_type": "voice_emotion2vec",
    }
    req = CacheRequest(user_id="test_user_x", emotion_result=payload)
    r = cache_voice_result(req)
    assert r["status"] == "cached"

    result = get_latest_voice("test_user_x")
    assert result is not None
    assert result["emotion"] == "happy"
    _VOICE_CACHE.clear()


def test_latest_returns_none_for_unknown_user():
    from api.routes.voice_watch import get_latest_voice, _VOICE_CACHE
    _VOICE_CACHE.clear()
    assert get_latest_voice("nobody_x") is None


def test_latest_returns_none_for_stale_cache():
    from api.routes.voice_watch import _VOICE_CACHE, get_latest_voice
    _VOICE_CACHE["stale_user"] = {
        "result": {"emotion": "sad", "valence": -0.5, "arousal": 0.3,
                   "confidence": 0.7, "probabilities": {}, "model_type": "test"},
        "ts": time.time() - 400,  # > 300s ago
    }
    assert get_latest_voice("stale_user") is None


def test_cache_ttl_constant_is_300():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import importlib
    vw = importlib.import_module("api.routes.voice_watch")
    assert vw._VOICE_CACHE_TTL == 300
