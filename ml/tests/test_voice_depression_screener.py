"""Tests for voice depression/anxiety screener model and API route (#402)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


FS = 16000


def _synthetic_speech(seconds: float = 5.0, fs: int = FS) -> np.ndarray:
    """Generate synthetic speech-like audio with voiced + silence segments."""
    rng = np.random.default_rng(42)
    n = int(seconds * fs)
    t = np.linspace(0, seconds, n)
    # Voiced segments: 150 Hz fundamental + harmonics + noise
    voiced = (
        0.3 * np.sin(2 * np.pi * 150 * t)
        + 0.15 * np.sin(2 * np.pi * 300 * t)
        + 0.05 * rng.normal(0, 1, n)
    ).astype(np.float32)
    # Insert silence gap in the middle
    gap_start = int(n * 0.4)
    gap_end = int(n * 0.5)
    voiced[gap_start:gap_end] = 0.0
    return voiced


@pytest.fixture
def client():
    from api.routes.voice_depression import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# -- model unit tests ---------------------------------------------------------

def test_extract_vocal_biomarkers():
    from models.voice_depression_screener import extract_vocal_biomarkers
    audio = _synthetic_speech(5.0)
    result = extract_vocal_biomarkers(audio, sr=FS)
    assert "error" not in result
    assert "f0_mean" in result
    assert "pitch_variability" in result
    assert "jitter_local" in result
    assert "shimmer_local" in result


def test_extract_short_audio_returns_error():
    from models.voice_depression_screener import extract_vocal_biomarkers
    short = np.zeros(100, dtype=np.float32)
    result = extract_vocal_biomarkers(short, sr=FS)
    assert "error" in result


def test_depression_score_range():
    from models.voice_depression_screener import (
        extract_vocal_biomarkers, score_depression_risk,
    )
    audio = _synthetic_speech(5.0)
    bio = extract_vocal_biomarkers(audio, sr=FS)
    dep = score_depression_risk(bio)
    assert 0.0 <= dep["phq9_score"] <= 27.0


def test_depression_severity_labels():
    from models.voice_depression_screener import score_depression_risk
    valid = {"minimal", "mild", "moderate", "moderately_severe", "severe", "unknown"}
    # Test with empty biomarkers (error case)
    dep = score_depression_risk({"error": "test"})
    assert dep["severity"] in valid
    # Test with real biomarkers
    dep2 = score_depression_risk({"pitch_variability": 0.05, "silence_ratio": 0.5,
                                   "speaking_rate": 1.5, "shimmer_local": 0.2,
                                   "energy_range": 0.01})
    assert dep2["severity"] in valid


def test_anxiety_score_range():
    from models.voice_depression_screener import (
        extract_vocal_biomarkers, score_anxiety_risk,
    )
    audio = _synthetic_speech(5.0)
    bio = extract_vocal_biomarkers(audio, sr=FS)
    anx = score_anxiety_risk(bio)
    assert 0.0 <= anx["gad7_score"] <= 21.0


def test_anxiety_severity_labels():
    from models.voice_depression_screener import score_anxiety_risk
    valid = {"minimal", "mild", "moderate", "severe", "unknown"}
    anx = score_anxiety_risk({"jitter_local": 0.003, "speaking_rate_variability": 0.4,
                               "formant_stability": 0.4, "f0_mean": 230,
                               "energy_std": 0.1, "energy_mean": 0.05})
    assert anx["severity"] in valid


def test_track_vocal_trend_insufficient():
    from models.voice_depression_screener import track_vocal_trend
    result = track_vocal_trend([])
    assert result["depression_trend"] == "insufficient_data"


def test_track_vocal_trend_with_data():
    from models.voice_depression_screener import track_vocal_trend
    # Simulate worsening depression trend
    history = [
        {"phq9_score": 5.0, "gad7_score": 3.0},
        {"phq9_score": 7.0, "gad7_score": 4.0},
        {"phq9_score": 9.0, "gad7_score": 5.0},
        {"phq9_score": 12.0, "gad7_score": 6.0},
    ]
    result = track_vocal_trend(history)
    assert result["depression_trend"] in ("worsening", "stable", "improving")
    assert result["entries_analyzed"] == 4


def test_compute_screening_profile():
    from models.voice_depression_screener import compute_screening_profile
    audio = _synthetic_speech(5.0)
    profile = compute_screening_profile(audio, sr=FS)
    assert "biomarkers" in profile
    assert "depression" in profile
    assert "anxiety" in profile
    assert "disclaimer" in profile


def test_profile_to_dict():
    from models.voice_depression_screener import (
        compute_screening_profile, profile_to_dict,
    )
    audio = _synthetic_speech(5.0)
    profile = compute_screening_profile(audio, sr=FS)
    d = profile_to_dict(profile)
    # Should be JSON-serializable (all native types)
    import json
    json.dumps(d)  # Should not raise


def test_get_screener():
    from models.voice_depression_screener import get_screener
    s = get_screener()
    assert "extract_vocal_biomarkers" in s
    assert "score_depression_risk" in s
    assert "score_anxiety_risk" in s
    assert "track_vocal_trend" in s


def test_pause_metrics_present():
    from models.voice_depression_screener import extract_vocal_biomarkers
    audio = _synthetic_speech(5.0)
    bio = extract_vocal_biomarkers(audio, sr=FS)
    assert "silence_ratio" in bio
    assert "pause_count" in bio
    assert "mean_pause_duration" in bio


def test_formant_stability_present():
    from models.voice_depression_screener import extract_vocal_biomarkers
    audio = _synthetic_speech(5.0)
    bio = extract_vocal_biomarkers(audio, sr=FS)
    assert "formant_stability" in bio
    assert 0.0 <= bio["formant_stability"] <= 1.0


# -- API route tests ----------------------------------------------------------

def test_api_screen(client):
    audio = _synthetic_speech(3.0)
    payload = {
        "audio_samples": audio.tolist(),
        "sample_rate": FS,
        "user_id": "test_user",
    }
    resp = client.post("/voice-screening/screen", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "test_user"
    assert "depression" in data
    assert "anxiety" in data


def test_api_screen_short_audio(client):
    short = np.zeros(100, dtype=np.float32).tolist()
    payload = {"audio_samples": short, "sample_rate": FS, "user_id": "u1"}
    resp = client.post("/voice-screening/screen", json=payload)
    assert resp.status_code == 422


def test_api_screen_with_biomarkers(client):
    audio = _synthetic_speech(3.0)
    payload = {
        "audio_samples": audio.tolist(),
        "sample_rate": FS,
        "user_id": "u2",
        "include_biomarkers": True,
    }
    resp = client.post("/voice-screening/screen", json=payload)
    assert resp.status_code == 200
    assert resp.json()["biomarkers"] is not None


def test_api_status(client):
    resp = client.get("/voice-screening/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ready"] is True
    assert "depression_voice_risk" in data["screening_models"]
    assert "anxiety_voice_risk" in data["screening_models"]
    assert data["not_validated"] is True
    assert "wellness_notice" in data


def test_api_disclaimer_present(client):
    audio = _synthetic_speech(3.0)
    payload = {"audio_samples": audio.tolist(), "sample_rate": FS, "user_id": "u3"}
    resp = client.post("/voice-screening/screen", json=payload)
    assert "disclaimer" in resp.json()
