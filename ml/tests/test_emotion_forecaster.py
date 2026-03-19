"""Tests for emotion forecaster model and API route (#406)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


def _make_daily_summaries(n_days: int = 14) -> list:
    """Generate synthetic daily emotional summaries."""
    rng = np.random.default_rng(77)
    summaries = []
    for i in range(n_days):
        summaries.append({
            "valence": round(float(np.clip(0.3 + 0.1 * np.sin(i * 0.9) + rng.normal(0, 0.1), -1, 1)), 3),
            "arousal": round(float(np.clip(0.5 + 0.1 * np.cos(i * 0.7) + rng.normal(0, 0.05), 0, 1)), 3),
            "stress": round(float(np.clip(0.4 + 0.05 * i / n_days + rng.normal(0, 0.05), 0, 1)), 3),
            "sleep_quality": round(float(np.clip(0.6 + rng.normal(0, 0.1), 0, 1)), 3),
            "activity_level": round(float(np.clip(0.5 + rng.normal(0, 0.1), 0, 1)), 3),
        })
    return summaries


@pytest.fixture
def client():
    from api.routes.emotion_forecaster import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# -- model unit tests ---------------------------------------------------------

def test_prepare_forecast_input_valid():
    from models.emotion_forecaster import prepare_forecast_input
    summaries = _make_daily_summaries(14)
    result = prepare_forecast_input(summaries)
    assert "error" not in result
    assert result["n_days"] == 14
    assert result["matrix"].shape == (14, 5)


def test_prepare_forecast_input_empty():
    from models.emotion_forecaster import prepare_forecast_input
    result = prepare_forecast_input([])
    assert "error" in result


def test_prepare_forecast_input_too_short():
    from models.emotion_forecaster import prepare_forecast_input
    result = prepare_forecast_input(_make_daily_summaries(2))
    assert "error" in result
    assert result["error"] == "insufficient_data"


def test_prepare_forecast_input_missing_values():
    from models.emotion_forecaster import prepare_forecast_input
    summaries = [
        {"valence": 0.5},  # missing other features
        {"valence": 0.6, "arousal": 0.4},
        {"valence": 0.7, "arousal": 0.5, "stress": 0.3},
        {"valence": 0.6, "arousal": 0.4, "stress": 0.4, "sleep_quality": 0.7, "activity_level": 0.5},
    ]
    result = prepare_forecast_input(summaries)
    assert "matrix" in result
    # NaN should have been imputed
    assert not np.any(np.isnan(result["matrix"]))


def test_forecast_emotion_1day():
    from models.emotion_forecaster import prepare_forecast_input, forecast_emotion
    summaries = _make_daily_summaries(14)
    prepared = prepare_forecast_input(summaries)
    result = forecast_emotion(prepared, horizon=1)
    assert "error" not in result
    assert len(result["forecasts"]) == 1
    fc = result["forecasts"][0]
    assert fc["day"] == 1
    assert -1.0 <= fc["valence"] <= 1.0
    assert 0.0 <= fc["arousal"] <= 1.0


def test_forecast_emotion_7day():
    from models.emotion_forecaster import prepare_forecast_input, forecast_emotion
    summaries = _make_daily_summaries(14)
    prepared = prepare_forecast_input(summaries)
    result = forecast_emotion(prepared, horizon=7)
    assert len(result["forecasts"]) == 7
    assert result["forecasts"][6]["day"] == 7


def test_forecast_has_trend_slopes():
    from models.emotion_forecaster import prepare_forecast_input, forecast_emotion
    summaries = _make_daily_summaries(14)
    prepared = prepare_forecast_input(summaries)
    result = forecast_emotion(prepared, horizon=3)
    assert "trend_slopes" in result
    assert "valence" in result["trend_slopes"]


def test_compute_forecast_confidence():
    from models.emotion_forecaster import (
        prepare_forecast_input, compute_forecast_confidence,
    )
    summaries = _make_daily_summaries(14)
    prepared = prepare_forecast_input(summaries)
    result = compute_forecast_confidence(prepared, horizon=3)
    assert "intervals" in result
    assert len(result["intervals"]) == 3
    assert 0.0 < result["overall_confidence"] <= 1.0


def test_detect_weekly_pattern():
    from models.emotion_forecaster import (
        prepare_forecast_input, detect_weekly_pattern,
    )
    summaries = _make_daily_summaries(21)
    prepared = prepare_forecast_input(summaries)
    result = detect_weekly_pattern(prepared)
    assert "pattern_detected" in result
    assert "pattern_strength" in result


def test_detect_weekly_pattern_too_short():
    from models.emotion_forecaster import (
        prepare_forecast_input, detect_weekly_pattern,
    )
    summaries = _make_daily_summaries(5)
    prepared = prepare_forecast_input(summaries)
    result = detect_weekly_pattern(prepared)
    assert result["pattern_detected"] is False


def test_compute_feature_importance():
    from models.emotion_forecaster import (
        prepare_forecast_input, compute_feature_importance,
    )
    summaries = _make_daily_summaries(14)
    prepared = prepare_forecast_input(summaries)
    result = compute_feature_importance(prepared)
    assert "importances" in result
    assert "ranking" in result
    assert len(result["ranking"]) == 5
    # Importances should sum to ~1.0
    total = sum(result["importances"].values())
    assert abs(total - 1.0) < 0.01


def test_forecast_to_dict():
    from models.emotion_forecaster import (
        prepare_forecast_input, forecast_emotion, forecast_to_dict,
    )
    summaries = _make_daily_summaries(14)
    prepared = prepare_forecast_input(summaries)
    result = forecast_emotion(prepared, horizon=1)
    d = forecast_to_dict(result)
    import json
    json.dumps(d)  # Should not raise


def test_get_forecaster():
    from models.emotion_forecaster import get_forecaster
    f = get_forecaster()
    assert "prepare_forecast_input" in f
    assert "forecast_emotion" in f
    assert "compute_forecast_confidence" in f


# -- API route tests ----------------------------------------------------------

def test_api_predict_1day(client):
    summaries = _make_daily_summaries(14)
    payload = {"daily_summaries": summaries, "horizon": 1}
    resp = client.post("/emotion-forecast/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["horizon_days"] == 1
    assert len(data["forecasts"]) == 1


def test_api_predict_7day(client):
    summaries = _make_daily_summaries(14)
    payload = {"daily_summaries": summaries, "horizon": 7}
    resp = client.post("/emotion-forecast/predict", json=payload)
    assert resp.status_code == 200
    assert len(resp.json()["forecasts"]) == 7


def test_api_predict_with_confidence(client):
    summaries = _make_daily_summaries(14)
    payload = {"daily_summaries": summaries, "horizon": 3, "include_confidence": True}
    resp = client.post("/emotion-forecast/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["confidence"] is not None


def test_api_predict_with_pattern(client):
    summaries = _make_daily_summaries(21)
    payload = {
        "daily_summaries": summaries,
        "horizon": 1,
        "include_weekly_pattern": True,
    }
    resp = client.post("/emotion-forecast/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["weekly_pattern"] is not None


def test_api_predict_with_importance(client):
    summaries = _make_daily_summaries(14)
    payload = {
        "daily_summaries": summaries,
        "horizon": 1,
        "include_feature_importance": True,
    }
    resp = client.post("/emotion-forecast/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["feature_importance"] is not None


def test_api_predict_too_short(client):
    summaries = _make_daily_summaries(2)
    payload = {"daily_summaries": summaries, "horizon": 1}
    resp = client.post("/emotion-forecast/predict", json=payload)
    assert resp.status_code == 422


def test_api_status(client):
    resp = client.get("/emotion-forecast/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ready"] is True
    assert data["min_days"] == 3
    assert "valence" in data["features"]


def test_api_predict_processed_at(client):
    summaries = _make_daily_summaries(7)
    payload = {"daily_summaries": summaries, "horizon": 1}
    resp = client.post("/emotion-forecast/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["processed_at"] > 0
