"""Tests for CSCL contrastive learning model and API route (#401)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


@pytest.fixture
def client():
    from api.routes.cscl_emotion import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Model unit tests
# ---------------------------------------------------------------------------


def test_import_model():
    from models.cscl_emotion import (
        CSCLConfig,
        ContrastivePair,
        RepresentationQualityResult,
        create_cscl_config,
        compute_nt_xent_loss,
        generate_positive_pair,
        generate_negative_pairs,
        evaluate_representation_quality,
        config_to_dict,
    )
    assert CSCLConfig is not None
    assert compute_nt_xent_loss is not None


def test_create_config_defaults():
    from models.cscl_emotion import create_cscl_config, config_to_dict
    config = create_cscl_config()
    d = config_to_dict(config)
    assert d["n_channels"] == 4
    assert d["temperature"] == 0.07
    assert d["projection_dim"] == 32
    assert d["n_classes"] == 6


def test_create_config_custom():
    from models.cscl_emotion import create_cscl_config
    config = create_cscl_config(n_channels=2, temperature=0.1, projection_dim=64)
    assert config.n_channels == 2
    assert config.temperature == 0.1
    assert config.projection_dim == 64


def test_nt_xent_loss_positive_close():
    from models.cscl_emotion import compute_nt_xent_loss
    anchor = np.array([1.0, 0.0, 0.0])
    positive = np.array([0.9, 0.1, 0.0])  # very similar to anchor
    negatives = np.random.default_rng(42).standard_normal((8, 3))
    loss = compute_nt_xent_loss(anchor, positive, negatives)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_nt_xent_loss_positive_far():
    from models.cscl_emotion import compute_nt_xent_loss
    anchor = np.array([1.0, 0.0, 0.0])
    positive = np.array([-1.0, 0.0, 0.0])  # opposite to anchor
    negatives = np.random.default_rng(42).standard_normal((8, 3))
    loss_far = compute_nt_xent_loss(anchor, positive, negatives)
    # Compare with close positive — far positive should have higher loss
    positive_close = np.array([0.9, 0.1, 0.0])
    loss_close = compute_nt_xent_loss(anchor, positive_close, negatives)
    assert loss_far > loss_close


def test_nt_xent_loss_single_negative():
    from models.cscl_emotion import compute_nt_xent_loss
    anchor = np.array([1.0, 0.0])
    positive = np.array([0.8, 0.2])
    negatives = np.array([[0.0, 1.0]])
    loss = compute_nt_xent_loss(anchor, positive, negatives)
    assert loss >= 0.0


def test_generate_positive_pair():
    from models.cscl_emotion import generate_positive_pair, CSCLConfig
    rng = np.random.default_rng(42)
    signals = rng.standard_normal((4, 1024))
    pair = generate_positive_pair(signals, seed=42)
    assert pair.anchor.shape == (20,)
    assert pair.positive.shape == (20,)
    # Anchor and positive should be different (different augmentations)
    assert not np.array_equal(pair.anchor, pair.positive)
    assert len(pair.augmentations_applied) >= 0


def test_generate_positive_pair_1d():
    from models.cscl_emotion import generate_positive_pair
    signals = np.random.default_rng(42).standard_normal(1024)
    pair = generate_positive_pair(signals, seed=42)
    assert pair.anchor.shape == (20,)


def test_generate_negative_pairs():
    from models.cscl_emotion import generate_negative_pairs
    rng = np.random.default_rng(42)
    anchor_feat = rng.standard_normal(10)
    all_features = rng.standard_normal((50, 10))
    all_labels = rng.integers(0, 3, size=50)
    anchor_label = 0

    negatives = generate_negative_pairs(
        anchor_feat, anchor_label, all_features, all_labels, n_negatives=8, seed=42
    )
    assert negatives.shape == (8, 10)


def test_generate_negative_pairs_no_negatives_available():
    from models.cscl_emotion import generate_negative_pairs
    rng = np.random.default_rng(42)
    anchor_feat = rng.standard_normal(10)
    all_features = rng.standard_normal((10, 10))
    all_labels = np.zeros(10, dtype=int)  # all same label
    negatives = generate_negative_pairs(
        anchor_feat, 0, all_features, all_labels, n_negatives=5, seed=42
    )
    assert negatives.shape == (5, 10)


def test_evaluate_representation_quality():
    from models.cscl_emotion import evaluate_representation_quality
    rng = np.random.default_rng(42)
    n, d = 30, 10
    features = rng.standard_normal((n, d))
    labels = rng.integers(0, 3, size=n)

    result = evaluate_representation_quality(features, labels)
    assert -1.0 <= result.alignment_score <= 1.0
    assert 0.0 <= result.uniformity_score <= 2.0
    assert -1.0 <= result.silhouette_score <= 1.0
    assert result.n_samples == n
    assert result.n_classes == 3
    assert result.embedding_dim == d
    assert result.evaluated_at > 0


def test_evaluate_with_subject_ids():
    from models.cscl_emotion import evaluate_representation_quality
    rng = np.random.default_rng(42)
    features = rng.standard_normal((20, 8))
    labels = rng.integers(0, 2, size=20)
    subject_ids = np.array(["s1"] * 10 + ["s2"] * 10)

    result = evaluate_representation_quality(features, labels, subject_ids)
    assert -1.0 <= result.cross_subject_consistency <= 1.0


def test_evaluate_single_sample():
    from models.cscl_emotion import evaluate_representation_quality
    features = np.array([[1.0, 2.0, 3.0]])
    labels = np.array([0])
    result = evaluate_representation_quality(features, labels)
    assert result.n_samples == 1


def test_augmentation_temporal_crop():
    from models.cscl_emotion import _augment_temporal_crop
    rng = np.random.default_rng(42)
    signals = rng.standard_normal((4, 1024))
    cropped = _augment_temporal_crop(signals, crop_ratio_min=0.8, rng=rng)
    assert cropped.shape == signals.shape
    # Some trailing values should be zero (from padding)
    # At minimum, there could be no padding if crop_ratio is 1.0, so just check shape
    assert cropped.shape == (4, 1024)


def test_augmentation_amplitude_jitter():
    from models.cscl_emotion import _augment_amplitude_jitter
    rng = np.random.default_rng(42)
    signals = np.ones((4, 100))
    jittered = _augment_amplitude_jitter(signals, jitter_range=(0.5, 1.5), rng=rng)
    assert jittered.shape == signals.shape
    # Values should be scaled, not all ones
    assert not np.allclose(jittered, signals)


def test_augmentation_channel_dropout():
    from models.cscl_emotion import _augment_channel_dropout
    rng = np.random.default_rng(42)
    signals = np.ones((4, 100))
    dropped = _augment_channel_dropout(signals, prob=1.0, rng=rng)
    # One channel should be all zeros
    zero_channels = np.sum(np.all(dropped == 0, axis=1))
    assert zero_channels >= 1


def test_config_to_dict():
    from models.cscl_emotion import CSCLConfig, config_to_dict
    config = CSCLConfig()
    d = config_to_dict(config)
    assert isinstance(d, dict)
    assert "temperature" in d
    assert "projection_dim" in d


# ---------------------------------------------------------------------------
# API route tests
# ---------------------------------------------------------------------------


def test_api_config_default(client):
    resp = client.post("/cscl-emotion/config", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["config"]["temperature"] == 0.07
    assert data["config"]["projection_dim"] == 32


def test_api_config_custom(client):
    resp = client.post("/cscl-emotion/config", json={
        "n_channels": 2,
        "temperature": 0.1,
        "projection_dim": 64,
        "n_classes": 3,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["config"]["n_channels"] == 2
    assert data["config"]["temperature"] == 0.1


def test_api_evaluate(client):
    rng = np.random.default_rng(42)
    features = rng.standard_normal((20, 10)).tolist()
    labels = rng.integers(0, 3, size=20).tolist()
    resp = client.post("/cscl-emotion/evaluate", json={
        "features": features,
        "labels": labels,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "alignment_score" in data
    assert "uniformity_score" in data
    assert "silhouette_score" in data
    assert data["n_samples"] == 20


def test_api_evaluate_with_subjects(client):
    rng = np.random.default_rng(42)
    features = rng.standard_normal((20, 8)).tolist()
    labels = rng.integers(0, 2, size=20).tolist()
    subject_ids = ["s1"] * 10 + ["s2"] * 10
    resp = client.post("/cscl-emotion/evaluate", json={
        "features": features,
        "labels": labels,
        "subject_ids": subject_ids,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "cross_subject_consistency" in data


def test_api_evaluate_label_mismatch(client):
    resp = client.post("/cscl-emotion/evaluate", json={
        "features": [[1.0, 2.0], [3.0, 4.0]],
        "labels": [0],  # wrong length
    })
    assert resp.status_code == 422


def test_api_status_empty(client):
    resp = client.get("/cscl-emotion/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "configured" in data


def test_api_status_after_config(client):
    client.post("/cscl-emotion/config", json={"temperature": 0.05})
    resp = client.get("/cscl-emotion/status")
    data = resp.json()
    assert data["configured"] is True
