"""Tests for EEG device adapters model and API route (#404)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


def _make_signal(n_ch: int = 2, n_samples: int = 512, fs: float = 256.0):
    """Generate random EEG-like signals."""
    rng = np.random.default_rng(99)
    t = np.linspace(0, n_samples / fs, n_samples)
    signals = np.zeros((n_ch, n_samples))
    for ch in range(n_ch):
        signals[ch] = (
            0.3 * np.sin(2 * np.pi * 10 * t)  # alpha
            + 0.1 * np.sin(2 * np.pi * 20 * t)  # beta
            + 0.05 * rng.normal(0, 1, n_samples)
        )
    return signals


@pytest.fixture
def client():
    from api.routes.device_adapters import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# -- model unit tests ---------------------------------------------------------

def test_get_device_profile_emotiv():
    from models.device_adapters import get_device_profile
    p = get_device_profile("emotiv_insight")
    assert p["channel_count"] == 5
    assert p["native_sr"] == 128
    assert "AF3" in p["channels"]


def test_get_device_profile_naox():
    from models.device_adapters import get_device_profile
    p = get_device_profile("naox_earbuds")
    assert p["channel_count"] == 2
    assert "TP9" in p["channels"]


def test_get_device_profile_md():
    from models.device_adapters import get_device_profile
    p = get_device_profile("md_neuro")
    assert p["channel_count"] == 2
    assert "FP1" in p["channels"]


def test_get_device_profile_unknown():
    from models.device_adapters import get_device_profile
    p = get_device_profile("nonexistent")
    assert "error" in p
    assert "supported_devices" in p


def test_map_channels_emotiv():
    from models.device_adapters import map_channels
    data = _make_signal(n_ch=5)
    result = map_channels("emotiv_insight", data)
    assert "AF3" in result["mapped_channels"]
    assert "AF4" in result["mapped_channels"]
    assert result["channel_count"] == 5


def test_map_channels_1d():
    from models.device_adapters import map_channels
    data = np.random.randn(512)
    result = map_channels("naox_earbuds", data)
    assert result["channel_count"] == 1


def test_normalize_sampling_rate_same():
    from models.device_adapters import normalize_sampling_rate
    data = np.random.randn(256)
    out = normalize_sampling_rate(data, 256, 256)
    assert len(out) == 256


def test_normalize_sampling_rate_upsample():
    from models.device_adapters import normalize_sampling_rate
    data = np.random.randn(128)  # 128 Hz, 1 second
    out = normalize_sampling_rate(data, 128, 256)
    assert len(out) == 256


def test_normalize_sampling_rate_2d():
    from models.device_adapters import normalize_sampling_rate
    data = np.random.randn(2, 250)  # 250 Hz, 1 second
    out = normalize_sampling_rate(data, 250, 256)
    assert out.shape[0] == 2
    # Should be close to 256 samples
    assert abs(out.shape[1] - 256) < 5


def test_compute_compatible_features_emotiv():
    from models.device_adapters import compute_compatible_features
    data = _make_signal(n_ch=5, n_samples=512)
    result = compute_compatible_features("emotiv_insight", data, fs=256.0)
    assert "avg_alpha_power" in result
    assert "frontal_asymmetry" in result["available"]


def test_compute_compatible_features_naox_no_frontal():
    from models.device_adapters import compute_compatible_features
    data = _make_signal(n_ch=2, n_samples=512)
    result = compute_compatible_features("naox_earbuds", data, fs=256.0)
    assert "frontal_asymmetry" in result["unavailable"]
    assert "temporal_asymmetry" in result["available"]


def test_compute_compatible_features_md_no_temporal():
    from models.device_adapters import compute_compatible_features
    data = _make_signal(n_ch=2, n_samples=512)
    result = compute_compatible_features("md_neuro", data, fs=256.0)
    assert "temporal_asymmetry" in result["unavailable"]
    assert "frontal_asymmetry" in result["available"]


def test_get_capability_matrix():
    from models.device_adapters import get_capability_matrix
    matrix = get_capability_matrix()
    assert "emotiv_insight" in matrix
    assert "naox_earbuds" in matrix
    assert "md_neuro" in matrix
    # Emotiv should support emotion classifier (has frontal pair)
    emotiv = matrix["emotiv_insight"]
    emotion_compat = [c for c in emotiv if c["model"] == "emotion_classifier"]
    assert len(emotion_compat) == 1
    assert emotion_compat[0]["compatible"] is True


def test_naox_no_emotion_classifier():
    from models.device_adapters import get_capability_matrix
    matrix = get_capability_matrix()
    naox = matrix["naox_earbuds"]
    emotion = [c for c in naox if c["model"] == "emotion_classifier"]
    assert len(emotion) == 1
    assert emotion[0]["compatible"] is False  # no frontal pair


def test_device_profile_to_dict():
    from models.device_adapters import device_profile_to_dict
    info = device_profile_to_dict("emotiv_insight")
    assert "compatible_models" in info
    assert "incompatible_models" in info
    assert "capability_details" in info


def test_device_profile_to_dict_unknown():
    from models.device_adapters import device_profile_to_dict
    info = device_profile_to_dict("fake_device")
    assert "error" in info


def test_get_adapter():
    from models.device_adapters import get_adapter
    a = get_adapter()
    assert "get_device_profile" in a
    assert "normalize_sampling_rate" in a


# -- API route tests ----------------------------------------------------------

def test_api_adapt_emotiv(client):
    signals = _make_signal(n_ch=5, n_samples=256).tolist()
    payload = {"device_id": "emotiv_insight", "signals": signals, "source_sr": 128}
    resp = client.post("/device-adapters/adapt", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["device_id"] == "emotiv_insight"
    assert data["target_sr"] == 256
    assert data["resampled_samples"] > 0


def test_api_adapt_unknown_device(client):
    signals = _make_signal(n_ch=2).tolist()
    payload = {"device_id": "fake", "signals": signals}
    resp = client.post("/device-adapters/adapt", json=payload)
    assert resp.status_code == 400


def test_api_devices(client):
    resp = client.get("/device-adapters/devices")
    assert resp.status_code == 200
    data = resp.json()
    assert data["device_count"] == 3
    ids = [d["device_id"] for d in data["devices"]]
    assert "emotiv_insight" in ids


def test_api_status(client):
    resp = client.get("/device-adapters/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ready"] is True
    assert "emotiv_insight" in data["supported_devices"]


def test_api_adapt_with_features(client):
    signals = _make_signal(n_ch=2, n_samples=512).tolist()
    payload = {"device_id": "md_neuro", "signals": signals, "compute_features": True}
    resp = client.post("/device-adapters/adapt", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["features"] is not None


def test_api_adapt_no_features(client):
    signals = _make_signal(n_ch=2, n_samples=512).tolist()
    payload = {"device_id": "md_neuro", "signals": signals, "compute_features": False}
    resp = client.post("/device-adapters/adapt", json=payload)
    assert resp.status_code == 200
    assert resp.json()["features"] is None
