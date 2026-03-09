"""Tests for GraphEmotionClassifier — GNN Spatial-Temporal Graph.

Covers:
- Initialisation and attribute existence
- predict() with 4-channel (4, 256) input
- predict() with single-channel (256,) input (graceful degradation)
- Node feature extraction shape (4, 8)
- Adjacency matrix shape (4, 4) and value range
- Output keys present
- Probabilities sum to 1.0
- Valence in [-1, 1]
- Arousal in [0, 1]
- model_type in output
- graph_edge_weights in output
- Singleton getter returns same instance
- Short EEG input (256 samples = 1 second)
- Longer EEG input (1024 samples = 4 seconds)
- All-zeros input does not crash
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Ensure ml/ is on the path so imports work from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.graph_emotion_classifier import (
    GraphEmotionClassifier,
    get_graph_emotion_classifier,
    EMOTION_LABELS,
    N_NODES,
    NODE_FEAT_DIM,
    N_CLASSES,
    _build_static_adj,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────

RNG = np.random.default_rng(0)
VALID_4CH_256 = RNG.standard_normal((4, 256)).astype(np.float32) * 20.0
VALID_4CH_1024 = RNG.standard_normal((4, 1024)).astype(np.float32) * 20.0
SINGLE_CH_256 = RNG.standard_normal(256).astype(np.float32) * 15.0
ALL_ZEROS_4CH = np.zeros((4, 256), dtype=np.float32)


# ── 1. Initialisation ─────────────────────────────────────────────────────────

class TestInit:
    def test_default_n_classes(self):
        clf = GraphEmotionClassifier()
        assert clf.n_classes == 6

    def test_default_hidden_dim(self):
        clf = GraphEmotionClassifier()
        assert clf.hidden_dim == 16

    def test_adjacency_initialised(self):
        clf = GraphEmotionClassifier()
        assert clf._adj is not None
        assert clf._adj.shape == (N_NODES, N_NODES)

    def test_w1_shape(self):
        clf = GraphEmotionClassifier()
        assert clf._W1.shape == (NODE_FEAT_DIM, NODE_FEAT_DIM)

    def test_w2_shape(self):
        clf = GraphEmotionClassifier()
        assert clf._W2.shape == (NODE_FEAT_DIM, NODE_FEAT_DIM)


# ── 2. predict() with 4-channel input ────────────────────────────────────────

class TestPredict4Channel:
    def setup_method(self):
        self.clf = GraphEmotionClassifier()
        self.result = self.clf.predict(VALID_4CH_256, fs=256)

    def test_returns_dict(self):
        assert isinstance(self.result, dict)

    def test_has_emotion_key(self):
        assert "emotion" in self.result

    def test_emotion_is_valid_label(self):
        assert self.result["emotion"] in EMOTION_LABELS

    def test_has_probabilities(self):
        assert "probabilities" in self.result

    def test_probabilities_has_6_classes(self):
        probs = self.result["probabilities"]
        assert len(probs) == 6
        for label in EMOTION_LABELS:
            assert label in probs

    def test_probabilities_sum_to_one(self):
        total = sum(self.result["probabilities"].values())
        assert abs(total - 1.0) < 1e-3, f"Probabilities sum to {total}"

    def test_probabilities_non_negative(self):
        for label, p in self.result["probabilities"].items():
            assert p >= 0.0, f"Negative probability for {label}: {p}"

    def test_valence_in_range(self):
        v = self.result["valence"]
        assert -1.0 <= v <= 1.0, f"Valence {v} out of [-1, 1]"

    def test_arousal_in_range(self):
        a = self.result["arousal"]
        assert 0.0 <= a <= 1.0, f"Arousal {a} out of [0, 1]"

    def test_has_stress_index(self):
        assert "stress_index" in self.result

    def test_has_focus_index(self):
        assert "focus_index" in self.result

    def test_has_relaxation_index(self):
        assert "relaxation_index" in self.result

    def test_has_model_type(self):
        assert "model_type" in self.result

    def test_model_type_value(self):
        assert self.result["model_type"] in ("graph-attention", "feature-based")

    def test_has_graph_edge_weights(self):
        assert "graph_edge_weights" in self.result

    def test_graph_edge_weights_length(self):
        weights = self.result["graph_edge_weights"]
        assert len(weights) == N_NODES * N_NODES, f"Expected {N_NODES**2}, got {len(weights)}"


# ── 3. predict() with single-channel input ───────────────────────────────────

class TestPredictSingleChannel:
    def setup_method(self):
        self.clf = GraphEmotionClassifier()
        self.result = self.clf.predict(SINGLE_CH_256, fs=256)

    def test_returns_dict(self):
        assert isinstance(self.result, dict)

    def test_has_emotion_key(self):
        assert "emotion" in self.result

    def test_emotion_is_valid(self):
        assert self.result["emotion"] in EMOTION_LABELS

    def test_probabilities_sum_to_one(self):
        total = sum(self.result["probabilities"].values())
        assert abs(total - 1.0) < 1e-3

    def test_valence_in_range(self):
        v = self.result["valence"]
        assert -1.0 <= v <= 1.0

    def test_arousal_in_range(self):
        a = self.result["arousal"]
        assert 0.0 <= a <= 1.0


# ── 4. Node feature extraction ────────────────────────────────────────────────

class TestNodeFeatureExtraction:
    def test_shape_4_nodes_8_features(self):
        clf = GraphEmotionClassifier()
        eeg = np.random.randn(4, 256).astype(np.float32) * 20.0
        features = clf._extract_node_features(eeg, fs=256)
        assert features.shape == (N_NODES, NODE_FEAT_DIM), (
            f"Expected ({N_NODES}, {NODE_FEAT_DIM}), got {features.shape}"
        )

    def test_returns_finite_values(self):
        clf = GraphEmotionClassifier()
        eeg = np.random.randn(4, 256).astype(np.float32) * 20.0
        features = clf._extract_node_features(eeg, fs=256)
        assert np.all(np.isfinite(features)), "Node features contain non-finite values"


# ── 5. Adjacency matrix ───────────────────────────────────────────────────────

class TestAdjacencyMatrix:
    def test_shape_4x4(self):
        adj = _build_static_adj()
        assert adj.shape == (N_NODES, N_NODES)

    def test_values_in_0_1(self):
        adj = _build_static_adj()
        assert np.all(adj >= 0.0), "Adjacency has negative values"
        assert np.all(adj <= 1.0), "Adjacency has values > 1"

    def test_symmetric(self):
        adj = _build_static_adj()
        assert np.allclose(adj, adj.T), "Static adjacency is not symmetric"

    def test_self_loops_set_to_one(self):
        adj = _build_static_adj()
        for i in range(N_NODES):
            assert adj[i, i] == 1.0, f"Self-loop at [{i},{i}] not 1.0"

    def test_classifier_adjacency_shape(self):
        clf = GraphEmotionClassifier()
        assert clf._adj.shape == (N_NODES, N_NODES)


# ── 6. Singleton getter ───────────────────────────────────────────────────────

class TestSingleton:
    def test_same_user_returns_same_instance(self):
        clf1 = get_graph_emotion_classifier("user_gnn_test_a")
        clf2 = get_graph_emotion_classifier("user_gnn_test_a")
        assert clf1 is clf2

    def test_different_users_return_different_instances(self):
        clf_a = get_graph_emotion_classifier("user_gnn_alpha")
        clf_b = get_graph_emotion_classifier("user_gnn_beta")
        assert clf_a is not clf_b

    def test_default_user_returns_classifier(self):
        clf = get_graph_emotion_classifier()
        assert isinstance(clf, GraphEmotionClassifier)


# ── 7. Short EEG (256 samples = 1 second) ────────────────────────────────────

class TestShortEEG:
    def test_predict_256_samples_no_crash(self):
        clf = GraphEmotionClassifier()
        eeg = np.random.randn(4, 256).astype(np.float32) * 15.0
        result = clf.predict(eeg, fs=256)
        assert isinstance(result, dict)

    def test_predict_256_samples_valid_emotion(self):
        clf = GraphEmotionClassifier()
        eeg = np.random.randn(4, 256).astype(np.float32) * 15.0
        result = clf.predict(eeg, fs=256)
        assert result["emotion"] in EMOTION_LABELS

    def test_predict_256_samples_probs_sum_one(self):
        clf = GraphEmotionClassifier()
        eeg = np.random.randn(4, 256).astype(np.float32) * 15.0
        result = clf.predict(eeg, fs=256)
        assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-3


# ── 8. Longer EEG (1024 samples = 4 seconds) ─────────────────────────────────

class TestLongerEEG:
    def test_predict_1024_samples_no_crash(self):
        clf = GraphEmotionClassifier()
        result = clf.predict(VALID_4CH_1024, fs=256)
        assert isinstance(result, dict)

    def test_predict_1024_samples_valid_emotion(self):
        clf = GraphEmotionClassifier()
        result = clf.predict(VALID_4CH_1024, fs=256)
        assert result["emotion"] in EMOTION_LABELS

    def test_predict_1024_samples_valence_in_range(self):
        clf = GraphEmotionClassifier()
        result = clf.predict(VALID_4CH_1024, fs=256)
        assert -1.0 <= result["valence"] <= 1.0

    def test_predict_1024_samples_graph_weights_count(self):
        clf = GraphEmotionClassifier()
        result = clf.predict(VALID_4CH_1024, fs=256)
        assert len(result["graph_edge_weights"]) == N_NODES * N_NODES


# ── 9. All-zeros input ────────────────────────────────────────────────────────

class TestAllZerosInput:
    def test_no_crash(self):
        clf = GraphEmotionClassifier()
        result = clf.predict(ALL_ZEROS_4CH, fs=256)
        assert isinstance(result, dict)

    def test_has_emotion_key(self):
        clf = GraphEmotionClassifier()
        result = clf.predict(ALL_ZEROS_4CH, fs=256)
        assert "emotion" in result
        assert result["emotion"] in EMOTION_LABELS

    def test_probabilities_sum_to_one(self):
        clf = GraphEmotionClassifier()
        result = clf.predict(ALL_ZEROS_4CH, fs=256)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-3


# ── 10. get_model_info ────────────────────────────────────────────────────────

class TestModelInfo:
    def test_returns_dict(self):
        clf = GraphEmotionClassifier()
        info = clf.get_model_info()
        assert isinstance(info, dict)

    def test_has_n_nodes(self):
        clf = GraphEmotionClassifier()
        info = clf.get_model_info()
        assert info["n_nodes"] == N_NODES

    def test_has_edge_list(self):
        clf = GraphEmotionClassifier()
        info = clf.get_model_info()
        assert "edge_list" in info
        assert len(info["edge_list"]) == 6

    def test_has_node_feature_dim(self):
        clf = GraphEmotionClassifier()
        info = clf.get_model_info()
        assert info["node_feature_dim"] == NODE_FEAT_DIM

    def test_requires_torch_false(self):
        clf = GraphEmotionClassifier()
        info = clf.get_model_info()
        assert info["requires_torch"] is False

    def test_requires_torch_geometric_false(self):
        clf = GraphEmotionClassifier()
        info = clf.get_model_info()
        assert info["requires_torch_geometric"] is False
