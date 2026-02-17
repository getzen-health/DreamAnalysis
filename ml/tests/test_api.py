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


class TestSessionAnalytics:
    def test_session_trends_empty(self, client):
        r = client.get("/api/sessions/trends?user_id=nonexistent_user")
        assert r.status_code == 200
        data = r.json()
        assert "session_count" in data

    def test_weekly_report(self, client):
        r = client.get("/api/sessions/weekly-report")
        assert r.status_code == 200
        data = r.json()
        assert "this_week_sessions" in data
        assert "highlights" in data

    def test_compare_missing_sessions(self, client):
        r = client.get("/api/sessions/compare/fake_a/fake_b")
        assert r.status_code == 404

    def test_session_analytics_functions(self):
        from storage.session_analytics import (
            compare_sessions, get_session_trends, get_weekly_report,
            _aggregate_timeline, _is_improvement, _build_narrative,
        )

        # Test aggregation
        timeline = [
            {"band_powers": {"alpha": 10, "beta": 5}, "flow_score": 0.6},
            {"band_powers": {"alpha": 12, "beta": 7}, "flow_score": 0.8},
        ]
        metrics = _aggregate_timeline(timeline)
        assert "band_powers.alpha" in metrics
        assert metrics["band_powers.alpha"] == 11.0
        assert metrics["flow_score"] == 0.7

        # Test improvement detection
        assert _is_improvement("flow_score", 0.1) is True
        assert _is_improvement("flow_score", -0.1) is False
        assert _is_improvement("stress_index", -5) is True
        assert _is_improvement("unknown_metric", 0.1) is None

        # Test narrative
        comparison = {
            "flow_score": {"pct_change": 25.0, "improved": True},
            "stress_index": {"pct_change": -10.0, "improved": True},
        }
        narrative = _build_narrative(comparison)
        assert len(narrative) > 0
