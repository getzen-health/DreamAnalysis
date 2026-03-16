"""Tests for upgraded voice-watch endpoints."""
import base64
import io
import os
import struct
import sys
import time
import types
import wave

import pytest
import numpy as np


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
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import importlib
    vw = importlib.import_module("api.routes.voice_watch")
    assert vw._VOICE_CACHE_TTL == 300


def test_voice_watch_analyze_auto_logs_supplement_state(monkeypatch):
    from api.routes.supplement_tracker import get_tracker
    from api.routes.voice_watch import VoiceWatchRequest, voice_watch_analyze

    tracker = get_tracker()
    tracker.reset("voice_supp_test")

    fake_soundfile = types.SimpleNamespace(
        read=lambda *_args, **_kwargs: (np.zeros(22050, dtype=np.float32), 22050),
    )

    class FakeVoiceModel:
        def predict_with_biomarkers(self, _audio, sample_rate=22050, real_time=False):
            return {
                "emotion": "happy",
                "probabilities": {"happy": 0.8, "neutral": 0.2},
                "valence": 0.55,
                "arousal": 0.48,
                "confidence": 0.88,
                "model_type": "stub",
                "biomarkers": {
                    "hnr_db": 16.0,
                    "jitter_local": 0.004,
                    "speech_rate": 4.7,
                },
                "mental_health": {
                    "stress": 0.22,
                },
            }

    fake_vm_module = types.SimpleNamespace(get_voice_model=lambda: FakeVoiceModel())

    # Make voice_ensemble import fail so the code falls through to the legacy path
    fake_ensemble = types.SimpleNamespace(
        get_voice_ensemble=lambda: (_ for _ in ()).throw(RuntimeError("no ensemble")),
        VoiceEnsembleRequest=None,
    )

    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setitem(sys.modules, "models.voice_emotion_model", fake_vm_module)
    monkeypatch.setitem(sys.modules, "models.voice_ensemble", fake_ensemble)

    result = voice_watch_analyze(VoiceWatchRequest(
        audio_b64=_make_silent_wav_b64(3),
        user_id="voice_supp_test",
    ))

    assert result["emotion"] == "happy"
    assert len(tracker._brain_states["voice_supp_test"]) == 1
    snapshot = tracker._brain_states["voice_supp_test"][0]
    assert snapshot.source == "voice"
    assert snapshot.speech_rate == 4.7


def test_voice_watch_status_reports_large_tier(monkeypatch):
    from api.routes.voice_watch import voice_watch_status

    class FakeSenseVoice:
        available = False

    class FakeVoiceModel:
        _sensevoice = FakeSenseVoice()

        def _load_e2v_large(self):
            return True

        def _load_e2v(self):
            return False

    fake_vm_module = types.SimpleNamespace(get_voice_model=lambda: FakeVoiceModel())
    monkeypatch.setitem(sys.modules, "models.voice_emotion_model", fake_vm_module)

    status = voice_watch_status()
    assert status["emotion2vec_large_available"] is True
    # ensemble prefix added when voice_ensemble module is importable
    assert "emotion2vec_large" in status["preferred_model_tier"]
    assert status["ready"] is True
