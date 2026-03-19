"""Tests for DFF-Net domain adaptation model and API route (#400)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


@pytest.fixture
def client():
    from api.routes.dff_net import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Model unit tests
# ---------------------------------------------------------------------------


def test_import_model():
    from models.dff_net import (
        DFFNetConfig,
        DomainAdaptationResult,
        FewShotResult,
        create_dff_net_config,
        compute_mmd,
        compute_domain_discrepancy,
        setup_few_shot_adaptation,
        evaluate_cross_subject,
        config_to_dict,
    )
    assert DFFNetConfig is not None
    assert create_dff_net_config is not None


def test_create_config_defaults():
    from models.dff_net import create_dff_net_config, config_to_dict
    config = create_dff_net_config()
    d = config_to_dict(config)
    assert d["n_channels"] == 4
    assert d["n_classes"] == 6
    assert d["adaptation_method"] == "mmd"
    assert d["k_shot"] == 5


def test_create_config_custom():
    from models.dff_net import create_dff_net_config
    config = create_dff_net_config(n_channels=2, n_classes=3, k_shot=10)
    assert config.n_channels == 2
    assert config.n_classes == 3
    assert config.k_shot == 10


def test_compute_mmd_identical():
    from models.dff_net import compute_mmd
    rng = np.random.default_rng(42)
    features = rng.standard_normal((20, 10))
    mmd = compute_mmd(features, features)
    # MMD of identical distributions should be very close to 0
    assert mmd < 0.1


def test_compute_mmd_different():
    from models.dff_net import compute_mmd
    rng = np.random.default_rng(42)
    source = rng.standard_normal((20, 10))
    target = rng.standard_normal((20, 10)) + 3.0  # shifted distribution
    mmd = compute_mmd(source, target)
    assert mmd > 0.0


def test_compute_mmd_linear_kernel():
    from models.dff_net import compute_mmd
    rng = np.random.default_rng(42)
    source = rng.standard_normal((15, 5))
    target = rng.standard_normal((15, 5)) + 2.0
    mmd = compute_mmd(source, target, kernel="linear")
    assert mmd > 0.0


def test_compute_mmd_small_input():
    from models.dff_net import compute_mmd
    source = np.array([[1.0, 2.0]])  # single sample
    target = np.array([[3.0, 4.0]])
    mmd = compute_mmd(source, target)
    assert mmd == 0.0  # insufficient samples


def test_domain_discrepancy_result_fields():
    from models.dff_net import compute_domain_discrepancy
    rng = np.random.default_rng(42)
    source = rng.standard_normal((15, 10))
    target = rng.standard_normal((15, 10))
    result = compute_domain_discrepancy(source, target)
    assert hasattr(result, "mmd_score")
    assert hasattr(result, "domain_accuracy")
    assert hasattr(result, "alignment_score")
    assert result.n_source_samples == 15
    assert result.n_target_samples == 15
    assert result.feature_dim == 10
    assert result.evaluated_at > 0


def test_domain_discrepancy_alignment_range():
    from models.dff_net import compute_domain_discrepancy
    rng = np.random.default_rng(42)
    source = rng.standard_normal((20, 10))
    target = rng.standard_normal((20, 10))
    result = compute_domain_discrepancy(source, target)
    assert 0.0 <= result.alignment_score <= 1.0


def test_few_shot_adaptation():
    from models.dff_net import setup_few_shot_adaptation
    rng = np.random.default_rng(42)
    n_source, n_target, d = 50, 10, 20

    source_features = rng.standard_normal((n_source, d))
    source_labels = rng.integers(0, 3, size=n_source)

    # Target features are shifted (cross-subject gap)
    target_features = rng.standard_normal((n_target, d)) + 0.5
    target_labels = rng.integers(0, 3, size=n_target)

    result = setup_few_shot_adaptation(
        source_features, source_labels, target_features, target_labels
    )
    assert result.adapted is True
    assert len(result.classes_seen) > 0
    assert result.adapted_at > 0
    assert 0.0 <= result.pre_adaptation_accuracy <= 1.0
    assert 0.0 <= result.post_adaptation_accuracy <= 1.0


def test_evaluate_cross_subject():
    from models.dff_net import evaluate_cross_subject
    rng = np.random.default_rng(42)
    source = rng.standard_normal((30, 10))
    s_labels = rng.integers(0, 3, size=30)
    target = rng.standard_normal((10, 10))
    t_labels = rng.integers(0, 3, size=10)

    result = evaluate_cross_subject(source, s_labels, target, t_labels)
    assert "domain_adaptation" in result
    assert "few_shot" in result
    assert "summary" in result
    assert "mmd_score" in result["summary"]
    assert "improvement" in result["summary"]


def test_config_to_dict():
    from models.dff_net import DFFNetConfig, config_to_dict
    config = DFFNetConfig()
    d = config_to_dict(config)
    assert isinstance(d, dict)
    assert "n_channels" in d
    assert "adaptation_method" in d
    assert "k_shot" in d


# ---------------------------------------------------------------------------
# API route tests
# ---------------------------------------------------------------------------


def test_api_config_default(client):
    resp = client.post("/dff-net/config", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["config"]["n_channels"] == 4
    assert data["config"]["adaptation_method"] == "mmd"


def test_api_config_custom(client):
    resp = client.post("/dff-net/config", json={
        "n_channels": 2,
        "n_classes": 4,
        "k_shot": 10,
        "adaptation_method": "adversarial",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["config"]["n_channels"] == 2
    assert data["config"]["n_classes"] == 4
    assert data["config"]["adaptation_method"] == "adversarial"


def test_api_evaluate_no_labels(client):
    rng = np.random.default_rng(42)
    source = rng.standard_normal((10, 5)).tolist()
    target = rng.standard_normal((10, 5)).tolist()
    resp = client.post("/dff-net/evaluate", json={
        "source_features": source,
        "target_features": target,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "domain_adaptation" in data
    assert data["few_shot"] is None
    assert "mmd_score" in data["summary"]


def test_api_evaluate_with_labels(client):
    rng = np.random.default_rng(42)
    source = rng.standard_normal((15, 5)).tolist()
    target = rng.standard_normal((8, 5)).tolist()
    s_labels = rng.integers(0, 3, size=15).tolist()
    t_labels = rng.integers(0, 3, size=8).tolist()
    resp = client.post("/dff-net/evaluate", json={
        "source_features": source,
        "target_features": target,
        "source_labels": s_labels,
        "target_labels": t_labels,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["few_shot"] is not None
    assert "improvement" in data["summary"]


def test_api_evaluate_dimension_mismatch(client):
    resp = client.post("/dff-net/evaluate", json={
        "source_features": [[1.0, 2.0]],
        "target_features": [[1.0, 2.0, 3.0]],
    })
    assert resp.status_code == 422


def test_api_status_empty(client):
    resp = client.get("/dff-net/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "configured" in data


def test_api_status_after_config(client):
    client.post("/dff-net/config", json={"n_classes": 4})
    resp = client.get("/dff-net/status")
    data = resp.json()
    assert data["configured"] is True
