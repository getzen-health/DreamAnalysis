"""Tests for FastAPI endpoints."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def client():
    from main import app
    from fastapi.testclient import TestClient
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"


class TestModelStatus:
    def test_models_status(self, client):
        r = client.get("/api/models/status")
        assert r.status_code == 200
        data = r.json()
        # Should have model keys
        assert "sleep_staging" in data or "available_states" in data


class TestEEGSimulation:
    def test_simulate_eeg(self, client):
        r = client.post("/api/simulate-eeg", json={
            "state": "rest",
            "duration": 2.0,
            "sample_rate": 256,
            "channels": 1,
        })
        assert r.status_code == 200
        data = r.json()
        assert "signals" in data

    def test_simulate_different_states(self, client):
        for state in ["rest", "focus", "meditation"]:
            r = client.post("/api/simulate-eeg", json={
                "state": state,
                "duration": 1.0,
                "sample_rate": 256,
                "channels": 1,
            })
            assert r.status_code == 200


class TestEEGAnalysis:
    def test_analyze_eeg(self, client):
        import numpy as np
        signal = (np.random.randn(1024) * 20).tolist()
        r = client.post("/api/analyze-eeg", json={
            "signals": [signal],
            "sample_rate": 256,
        })
        assert r.status_code == 200
        data = r.json()
        assert "sleep_staging" in data or "emotions" in data


class TestHealthIntegration:
    def test_supported_metrics(self, client):
        r = client.get("/api/health/supported-metrics")
        assert r.status_code == 200
        data = r.json()
        assert "apple_health" in data or "apple_health_types" in data

    def test_ingest_apple_health(self, client):
        r = client.post("/api/health/ingest", json={
            "source": "apple_health",
            "user_id": "pytest_user",
            "data": {
                "heart_rate": [
                    {"timestamp": 1700000000, "value": 72, "source": "Apple Watch"}
                ]
            },
        })
        assert r.status_code == 200

    def test_daily_summary(self, client):
        r = client.get("/api/health/daily-summary/pytest_user")
        assert r.status_code == 200
        data = r.json()
        assert "brain" in data
        assert "health" in data

    def test_brain_session(self, client):
        r = client.post("/api/health/brain-session", json={
            "user_id": "pytest_user",
            "start_time": 1700000000.0,
            "end_time": 1700003600.0,
            "duration_seconds": 3600.0,
            "analysis": {
                "flow_state": {"state": "flow", "flow_score": 0.7},
                "creativity": {"state": "creative", "creativity_score": 0.6},
                "memory_encoding": {"state": "active_encoding", "will_remember_probability": 0.5},
                "emotions": {"emotion": "focused", "valence": 0.3, "arousal": 0.4},
                "sleep_stage": {"stage": "Wake"},
                "dream_detection": {"is_dreaming": False},
            },
        })
        assert r.status_code == 200


class TestAccuracyPipelineAPI:
    def test_analyze_eeg_accurate(self, client):
        signal = (np.random.randn(1024) * 20).tolist()
        r = client.post("/api/analyze-eeg-accurate", json={
            "signals": [signal],
            "sample_rate": 256,
            "user_id": "test_user",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "quality" in data
        assert "analysis" in data
        assert "smoothed_states" in data
        assert "confidence_summary" in data
        assert "coherence" in data

    def test_signal_quality_endpoint(self, client):
        signal = (np.random.randn(1024) * 20).tolist()
        r = client.post("/api/signal-quality", json={
            "signals": [signal],
            "sample_rate": 256,
        })
        assert r.status_code == 200
        data = r.json()
        assert "quality_score" in data
        assert "is_usable" in data

    def test_confidence_reliability(self, client):
        r = client.get("/api/confidence/reliability")
        assert r.status_code == 200
        data = r.json()
        assert "sleep_staging" in data
        assert "reliability_tier" in data["sleep_staging"]

    def test_state_engine_coherence(self, client):
        r = client.get("/api/state-engine/coherence")
        assert r.status_code == 200
        data = r.json()
        assert "is_coherent" in data
