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


class TestSpiritualEnergy:
    """Tests for spiritual energy / self-awareness BCI endpoints."""

    def test_chakra_info(self, client):
        r = client.get("/api/spiritual/chakras/info")
        assert r.status_code == 200
        data = r.json()
        assert "chakras" in data
        assert "root" in data["chakras"]
        assert "crown" in data["chakras"]
        assert data["chakras"]["root"]["sanskrit"] == "Muladhara"
        assert data["chakras"]["crown"]["sanskrit"] == "Sahasrara"
        assert "consciousness_levels" in data

    def test_chakra_analysis(self, client):
        eeg = np.random.randn(1, 1024).tolist()
        r = client.post("/api/spiritual/chakras", json={"signals": eeg, "fs": 256})
        assert r.status_code == 200
        data = r.json()
        assert "chakras" in data
        assert "balance" in data
        for chakra in ["root", "sacral", "solar_plexus", "heart", "throat", "third_eye", "crown"]:
            assert chakra in data["chakras"]
            assert "activation" in data["chakras"][chakra]
            assert 0 <= data["chakras"][chakra]["activation"] <= 100
        assert "harmony_score" in data["balance"]
        assert "dominant_chakra" in data["balance"]

    def test_meditation_depth(self, client):
        eeg = np.random.randn(1, 1024).tolist()
        r = client.post("/api/spiritual/meditation-depth", json={"signals": eeg, "fs": 256})
        assert r.status_code == 200
        data = r.json()
        assert "depth_score" in data
        assert 0 <= data["depth_score"] <= 10
        assert "stage" in data
        assert "guidance" in data

    def test_aura_energy(self, client):
        eeg = np.random.randn(1, 1024).tolist()
        r = client.post("/api/spiritual/aura", json={"signals": eeg, "fs": 256})
        assert r.status_code == 200
        data = r.json()
        assert "dominant_color" in data
        assert "blended_aura_color" in data
        assert "layers" in data
        assert len(data["layers"]) == 3  # inner, middle, outer

    def test_kundalini_flow(self, client):
        eeg = np.random.randn(1, 1024).tolist()
        r = client.post("/api/spiritual/kundalini", json={"signals": eeg, "fs": 256})
        assert r.status_code == 200
        data = r.json()
        assert "highest_chakra_reached" in data
        assert "awakening_status" in data
        assert "chakra_progression" in data
        assert len(data["chakra_progression"]) == 7

    def test_consciousness_level(self, client):
        eeg = np.random.randn(1, 1024).tolist()
        r = client.post("/api/spiritual/consciousness", json={"signals": eeg, "fs": 256})
        assert r.status_code == 200
        data = r.json()
        assert "score" in data
        assert 0 <= data["score"] <= 1000
        assert "level" in data

    def test_third_eye(self, client):
        eeg = np.random.randn(1, 1024).tolist()
        r = client.post("/api/spiritual/third-eye", json={"signals": eeg, "fs": 256})
        assert r.status_code == 200
        data = r.json()
        assert "activation_pct" in data
        assert "status" in data
        assert "insight" in data

    def test_prana_balance(self, client):
        eeg = np.random.randn(2, 1024).tolist()
        r = client.post("/api/spiritual/prana-balance", json={"signals": eeg, "fs": 256})
        assert r.status_code == 200
        data = r.json()
        assert "dominant_nadi" in data
        assert data["dominant_nadi"] in ["ida", "pingala", "sushumna"]
        assert "balance_quality" in data
        assert "guidance" in data

    def test_full_spiritual_analysis(self, client):
        eeg = np.random.randn(2, 1024).tolist()
        r = client.post("/api/spiritual/full-analysis", json={"signals": eeg, "fs": 256})
        assert r.status_code == 200
        data = r.json()
        assert "chakras" in data
        assert "meditation_depth" in data
        assert "aura" in data
        assert "kundalini" in data
        assert "consciousness" in data
        assert "third_eye" in data
        assert "prana_balance" in data  # 2 channels provided
        assert "insight" in data
        assert "summary" in data["insight"]
        assert "recommended_practices" in data["insight"]

    def test_spiritual_module_functions(self):
        """Test spiritual energy functions directly."""
        from processing.spiritual_energy import (
            compute_chakra_activations,
            compute_chakra_balance,
            compute_meditation_depth,
            compute_aura_energy,
            compute_kundalini_flow,
            compute_prana_balance,
            compute_consciousness_level,
            compute_third_eye_activation,
            full_spiritual_analysis,
        )

        np.random.seed(42)
        eeg = np.random.randn(2048)

        # Test chakras
        chakras = compute_chakra_activations(eeg, 256)
        assert len(chakras) == 7
        for name, data in chakras.items():
            assert "activation" in data
            assert "sanskrit" in data

        # Test balance
        balance = compute_chakra_balance(chakras)
        assert 0 <= balance["harmony_score"] <= 100
        assert balance["energy_flow"] in ["ascending", "descending", "balanced"]

        # Test meditation
        med = compute_meditation_depth(eeg, 256)
        assert 0 <= med["depth_score"] <= 10

        # Test aura
        aura = compute_aura_energy(eeg, 256)
        assert aura["blended_aura_color"].startswith("#")

        # Test kundalini
        kund = compute_kundalini_flow(eeg, 256)
        assert len(kund["chakra_progression"]) == 7

        # Test prana
        left = np.random.randn(2048)
        right = np.random.randn(2048)
        prana = compute_prana_balance(left, right, 256)
        assert prana["dominant_nadi"] in ["ida", "pingala", "sushumna"]

        # Test consciousness
        consc = compute_consciousness_level(eeg, 256)
        assert 0 <= consc["score"] <= 1000

        # Test third eye
        te = compute_third_eye_activation(eeg, 256)
        assert 0 <= te["activation_pct"] <= 100

        # Test full analysis
        full = full_spiritual_analysis(eeg, 256, left, right)
        assert "prana_balance" in full
        assert "insight" in full
