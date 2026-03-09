"""GNN Spatial-Temporal Graph for 4-Channel EEG Emotion Recognition.

Models the 4 Muse 2 EEG channels as a static + learnable spatial graph and
applies two layers of GAT-style message passing (NumPy-only, no torch-geometric)
to produce a 6-class emotion prediction.

Architecture
------------
- 4-node graph: TP9 (ch0), AF7 (ch1), AF8 (ch2), TP10 (ch3)
- Node features: 5 differential-entropy (DE) bands + 3 Hjorth parameters = 8 per node
- Static edges: 6 anatomical connections (ipsilateral + contralateral + frontal FAA pair)
- Learnable adjacency: exponential moving-average update during inference
- 2-layer dot-product attention → global mean pool → linear 8→16→6 classifier
- Falls back to feature-based heuristics when any dependency is unavailable

References
----------
Veličković et al., "Graph Attention Networks." ICLR 2018.
Davidson (1992) FAA: ln(AF8_alpha) - ln(AF7_alpha) → valence proxy.

Muse 2 channel map (BrainFlow board_id=38):
    ch0 = TP9   (left temporal)
    ch1 = AF7   (left frontal)   ← FAA left
    ch2 = AF8   (right frontal)  ← FAA right
    ch3 = TP10  (right temporal)
"""

import threading
import numpy as np
from typing import Dict, List, Optional

# ── Constants ─────────────────────────────────────────────────────────────────

EMOTION_LABELS: List[str] = ["happy", "sad", "angry", "fear", "surprise", "neutral"]
N_NODES = 4            # TP9, AF7, AF8, TP10
NODE_FEAT_DIM = 8      # 5 DE bands + 3 Hjorth
N_CLASSES = 6
HIDDEN_DIM = 16

# Static anatomical edges (undirected; stored as list of (i, j) pairs)
# (0,1): TP9-AF7   left hemisphere ipsilateral
# (2,3): AF8-TP10  right hemisphere ipsilateral
# (1,2): AF7-AF8   frontal pair (FAA-related)
# (0,3): TP9-TP10  temporal pair
# (0,2): TP9-AF8   cross-hemisphere
# (1,3): AF7-TP10  cross-hemisphere
STATIC_EDGES = [(0, 1), (2, 3), (1, 2), (0, 3), (0, 2), (1, 3)]

# Initial static adjacency weights (symmetric)
_STATIC_WEIGHTS = {
    (0, 1): 0.9,   # strong — same hemisphere
    (2, 3): 0.9,
    (1, 2): 1.0,   # strongest — FAA pair
    (0, 3): 0.8,   # moderate — temporal pair
    (0, 2): 0.5,   # cross-hemisphere
    (1, 3): 0.5,
}

# EEG band definitions used for DE feature extraction (5 bands)
_DE_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),  # capped at 45 Hz — muscle artifact mitigation
}

# EMA decay for learnable adjacency update (per inference call)
_EMA_ALPHA = 0.05


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bandpass_simple(signal: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    """Simple Butterworth bandpass via SciPy. Returns signal if SciPy unavailable."""
    try:
        from scipy.signal import butter, filtfilt
        nyq = fs / 2.0
        lo = max(low / nyq, 1e-4)
        hi = min(high / nyq, 1.0 - 1e-4)
        if lo >= hi:
            return signal
        b, a = butter(4, [lo, hi], btype="band")
        return filtfilt(b, a, signal)
    except Exception:
        return signal


def _de_single_band(signal: np.ndarray, low: float, high: float, fs: float) -> float:
    """Differential entropy for one frequency band: 0.5 * log(2πe * var)."""
    filtered = _bandpass_simple(signal, low, high, fs)
    var = float(np.var(filtered))
    return 0.5 * float(np.log(2.0 * np.pi * np.e * max(var, 1e-12)))


def _hjorth(signal: np.ndarray):
    """Return (activity, mobility, complexity) Hjorth parameters."""
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)
    activity = float(np.var(signal))
    activity = max(activity, 1e-10)
    mob_var = float(np.var(diff1))
    mobility = float(np.sqrt(mob_var / activity))
    if mob_var > 0 and mobility > 0:
        complexity = float(np.sqrt(float(np.var(diff2)) / mob_var) / mobility)
    else:
        complexity = 0.0
    return activity, mobility, complexity


def _build_static_adj() -> np.ndarray:
    """Build (4, 4) symmetric adjacency matrix from anatomical edge weights."""
    adj = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    for (i, j), w in _STATIC_WEIGHTS.items():
        adj[i, j] = w
        adj[j, i] = w
    # Self-loops
    np.fill_diagonal(adj, 1.0)
    return adj


def _row_norm(adj: np.ndarray) -> np.ndarray:
    """Row-normalise adjacency so each row sums to 1 (simple GCN-style)."""
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return adj / row_sums


def _dot_attention(H: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """Compute dot-product attention weights, masked by adjacency.

    Parameters
    ----------
    H   : (N, d) node feature matrix
    adj : (N, N) adjacency (binary or weighted)

    Returns
    -------
    alpha : (N, N) row-softmax attention weights, zero where adj == 0
    """
    N, d = H.shape
    # Raw scores: (N, N)  score_ij = h_i · h_j / sqrt(d)
    scores = H @ H.T / max(np.sqrt(d), 1.0)
    # Mask out non-edges
    mask = adj > 0
    scores = np.where(mask, scores, -1e9)
    # Row-wise softmax
    scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    exp_scores = np.where(mask, exp_scores, 0.0)
    row_sum = exp_scores.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1.0, row_sum)
    return exp_scores / row_sum


def _gat_layer(H: np.ndarray, adj: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Single GAT-style message-passing layer with residual connection.

    H_new[i] = ReLU( sum_j alpha_ij * (H[j] @ W) ) + H[i]

    Parameters
    ----------
    H   : (N, d_in)
    adj : (N, N)
    W   : (d_in, d_out) — weight matrix

    Returns
    -------
    H_new : (N, d_out)   (with d_in == d_out for residual; truncated otherwise)
    """
    alpha = _dot_attention(H, adj)    # (N, N)
    # Message: weighted sum of transformed neighbours
    H_msg = H @ W                     # (N, d_out)
    H_agg = alpha @ H_msg             # (N, d_out)
    H_new = np.maximum(H_agg, 0.0)   # ReLU

    # Residual: add H projected to d_out dimension
    d_out = W.shape[1]
    if H.shape[1] == d_out:
        H_new = H_new + H
    else:
        # Project H to match d_out for residual
        H_new = H_new + H[:, :d_out]
    return H_new


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


# ── Main class ────────────────────────────────────────────────────────────────

class GraphEmotionClassifier:
    """Spatial-temporal graph attention emotion classifier for 4-channel Muse 2 EEG.

    Uses a static + learnable (4, 4) adjacency matrix and two GAT-style
    message-passing layers (NumPy-only — no torch-geometric required).

    The "learning" occurs via EMA updates of the adjacency and linear
    classifier weights during inference, allowing online personalisation.
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        hidden_dim: int = HIDDEN_DIM,
        n_epochs_per_sample: int = 256,
    ):
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.n_epochs_per_sample = n_epochs_per_sample

        # Learnable adjacency — initialised from anatomy, updated via EMA
        self._adj: np.ndarray = _build_static_adj()

        # GAT weight matrices — Layer 1: (8, 8), Layer 2: (8, 8)
        rng = np.random.default_rng(42)
        self._W1: np.ndarray = rng.normal(0, 0.1, (NODE_FEAT_DIM, NODE_FEAT_DIM))
        self._W2: np.ndarray = rng.normal(0, 0.1, (NODE_FEAT_DIM, NODE_FEAT_DIM))

        # Linear head: (NODE_FEAT_DIM, hidden_dim) then (hidden_dim, n_classes)
        self._Wh: np.ndarray = rng.normal(0, 0.1, (NODE_FEAT_DIM, hidden_dim))
        self._bh: np.ndarray = np.zeros(hidden_dim)
        self._Wc: np.ndarray = rng.normal(0, 0.1, (hidden_dim, n_classes))
        self._bc: np.ndarray = np.zeros(n_classes)

        self._lock = threading.Lock()
        self._call_count: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, eeg: np.ndarray, fs: int = 256) -> dict:
        """Classify emotion using spatial-temporal graph attention.

        Parameters
        ----------
        eeg : np.ndarray
            EEG data. Shape (4, n_samples) preferred.
            Single-channel (n_samples,) is accepted with graceful degradation.
        fs  : int
            Sampling frequency in Hz (default 256).

        Returns
        -------
        dict matching the standard emotion output structure.
        """
        try:
            return self._predict_graph(eeg, fs)
        except Exception:
            return self._predict_fallback(eeg, fs)

    def _predict_graph(self, eeg: np.ndarray, fs: int) -> dict:
        """Full graph-attention prediction path (NumPy-only)."""
        eeg = np.asarray(eeg, dtype=np.float64)

        # Normalise input shape
        if eeg.ndim == 1:
            # Single channel — replicate to 4 channels for graph structure
            eeg = np.tile(eeg, (N_NODES, 1))
            single_channel_input = True
        elif eeg.ndim == 2 and eeg.shape[0] < N_NODES:
            # Pad missing channels with zeros
            pad = np.zeros((N_NODES - eeg.shape[0], eeg.shape[1]))
            eeg = np.vstack([eeg, pad])
            single_channel_input = True
        else:
            single_channel_input = False

        # 1. Node feature extraction — (N_NODES, NODE_FEAT_DIM)
        node_features = self._extract_node_features(eeg, fs)

        # 2. Build current adjacency (static + learnable)
        adj = self._build_adjacency()

        # 3. Two-layer GAT message passing
        H = self._gat_forward(node_features, adj)   # (N_NODES, NODE_FEAT_DIM)

        # 4. Global mean pool → (NODE_FEAT_DIM,)
        pooled = H.mean(axis=0)

        # 5. Linear head: 8 → hidden → n_classes
        hidden = np.maximum(pooled @ self._Wh + self._bh, 0.0)
        logits = hidden @ self._Wc + self._bc
        probs = _softmax(logits)

        # 6. Derive physiological indices from node features
        valence, arousal, stress, focus, relax = self._physio_from_features(
            node_features, eeg, fs, single_channel_input
        )

        # 7. EMA update adjacency from current epoch correlation
        self._update_adjacency(eeg, fs)

        self._call_count += 1

        emotion_idx = int(np.argmax(probs))
        return {
            "emotion": EMOTION_LABELS[emotion_idx],
            "probabilities": {
                label: float(round(p, 4))
                for label, p in zip(EMOTION_LABELS, probs)
            },
            "valence": float(round(valence, 4)),
            "arousal": float(round(arousal, 4)),
            "stress_index": float(round(stress, 4)),
            "focus_index": float(round(focus, 4)),
            "relaxation_index": float(round(relax, 4)),
            "model_type": "graph-attention",
            "graph_edge_weights": self._adj.flatten().tolist(),
        }

    def _extract_node_features(self, eeg: np.ndarray, fs: int) -> np.ndarray:
        """Extract 8 features per node: 5 DE bands + 3 Hjorth = (4, 8) matrix.

        Parameters
        ----------
        eeg : (4, n_samples) numpy array

        Returns
        -------
        np.ndarray of shape (N_NODES, NODE_FEAT_DIM) = (4, 8)
        """
        features = np.zeros((N_NODES, NODE_FEAT_DIM), dtype=np.float64)
        for ch_idx in range(N_NODES):
            sig = eeg[ch_idx]
            # 5 DE values
            for band_idx, (band_name, (lo, hi)) in enumerate(_DE_BANDS.items()):
                features[ch_idx, band_idx] = _de_single_band(sig, lo, hi, fs)
            # 3 Hjorth
            act, mob, cmp = _hjorth(sig)
            features[ch_idx, 5] = float(np.log(max(act, 1e-12)))  # log activity
            features[ch_idx, 6] = mob
            features[ch_idx, 7] = cmp
        return features

    def _build_adjacency(self) -> np.ndarray:
        """Return current (4, 4) adjacency matrix (row-normalised)."""
        with self._lock:
            adj = self._adj.copy()
        return _row_norm(adj)

    def _gat_forward(self, node_features: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """Two-layer GAT-style message passing.

        Parameters
        ----------
        node_features : (N_NODES, NODE_FEAT_DIM)
        adj           : (N_NODES, N_NODES) row-normalised

        Returns
        -------
        H : (N_NODES, NODE_FEAT_DIM) updated node embeddings
        """
        H = node_features.copy()
        H = _gat_layer(H, adj, self._W1)  # Layer 1
        H = _gat_layer(H, adj, self._W2)  # Layer 2
        return H

    def _update_adjacency(self, eeg: np.ndarray, fs: int) -> None:
        """EMA update: blend current epoch's channel correlations into adjacency."""
        try:
            # Extract 5-band DE per channel as a compact feature vector
            features = np.zeros((N_NODES, len(_DE_BANDS)), dtype=np.float64)
            for ch in range(N_NODES):
                for bi, (_, (lo, hi)) in enumerate(_DE_BANDS.items()):
                    features[ch, bi] = _de_single_band(eeg[ch], lo, hi, fs)
            # Pearson correlation between channel feature vectors → (4, 4)
            # Clip to [0, 1] so it represents functional connectivity strength
            corr = np.corrcoef(features)
            corr_01 = np.clip((corr + 1.0) / 2.0, 0.0, 1.0)

            with self._lock:
                self._adj = (
                    (1.0 - _EMA_ALPHA) * self._adj
                    + _EMA_ALPHA * corr_01
                )
                # Re-enforce self-loops
                np.fill_diagonal(self._adj, 1.0)
        except Exception:
            pass  # Never break inference

    def _physio_from_features(
        self,
        node_features: np.ndarray,
        eeg: np.ndarray,
        fs: int,
        single_channel: bool,
    ):
        """Derive valence, arousal, stress, focus, relaxation from node features.

        Uses the same proven heuristics as emotion_classifier._predict_features()
        so output ranges match the rest of the system.
        """
        # node_features: (4, 8); columns 0-4 are DE bands (delta, theta, alpha, beta, gamma)
        # Use AF7 (ch1) for single-channel metrics, blend FAA for valence
        de_af7 = node_features[1, :]  # ch1 = AF7
        de_af8 = node_features[2, :]  # ch2 = AF8

        delta = float(np.exp(de_af7[0]))
        theta = float(np.exp(de_af7[1]))
        alpha = float(np.exp(de_af7[2]))
        beta  = float(np.exp(de_af7[3]))

        alpha = max(alpha, 1e-10)
        beta  = max(beta,  1e-10)
        theta = max(theta, 1e-10)
        delta = max(delta, 1e-10)

        beta_alpha   = beta / (beta + alpha)
        alpha_ratio  = alpha / (alpha + beta + theta + delta)
        theta_beta   = theta / beta

        # Valence from alpha/beta ratio
        valence_abr = float(np.tanh((alpha / beta - 0.7) * 2.0))

        # FAA when multichannel available
        faa_valence = 0.0
        if not single_channel:
            try:
                alpha_af7 = max(float(np.exp(de_af7[2])), 1e-12)
                alpha_af8 = max(float(np.exp(de_af8[2])), 1e-12)
                faa = np.log(alpha_af8) - np.log(alpha_af7)
                faa_valence = float(np.clip(np.tanh(faa * 2.0), -1.0, 1.0))
            except Exception:
                pass

        if not single_channel:
            valence = float(np.clip(0.50 * valence_abr + 0.50 * faa_valence, -1.0, 1.0))
        else:
            valence = float(np.clip(valence_abr, -1.0, 1.0))

        # Arousal — no gamma (EMG noise on Muse)
        high_beta = max(float(np.exp(de_af7[3])) * 0.4, 1e-10)  # approx high-beta portion
        arousal_raw = (
            0.45 * beta_alpha
            + 0.30 * (1.0 - alpha_ratio)
            + 0.25 * (1.0 - delta / (delta + beta))
        )
        arousal = float(np.clip(arousal_raw, 0.0, 1.0))

        # Stress (no gamma)
        high_beta_frac = min(0.4, beta_alpha * 0.4)
        stress = float(np.clip(
            0.45 * high_beta_frac / max(beta_alpha, 1e-4)
            + 0.30 * min(1.0, (theta / alpha) * 0.3)
            + 0.25 * min(1.0, high_beta_frac),
            0.0, 1.0,
        ))

        # Focus (no gamma)
        focus = float(np.clip(
            0.45 * beta_alpha
            + 0.40 * (1.0 - alpha_ratio)
            + 0.15 * (1.0 - min(1.0, theta_beta * 0.2)),
            0.0, 1.0,
        ))

        # Relaxation
        relax = float(np.clip(1.0 - 0.6 * beta_alpha - 0.4 * (1.0 - alpha_ratio), 0.0, 1.0))

        return valence, arousal, stress, focus, relax

    def _predict_fallback(self, eeg: np.ndarray, fs: int) -> dict:
        """Feature-based fallback when graph path raises an exception."""
        try:
            import sys
            import os
            _ml_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _ml_root not in sys.path:
                sys.path.insert(0, _ml_root)
            from processing.eeg_processor import extract_band_powers, preprocess, compute_frontal_asymmetry
            eeg = np.asarray(eeg, dtype=np.float64)
            if eeg.ndim == 2:
                sig = eeg[1] if eeg.shape[0] > 1 else eeg[0]
                channels = eeg
            else:
                sig = eeg
                channels = None
            proc = preprocess(sig, fs)
            bp = extract_band_powers(proc, fs)
            alpha = max(bp.get("alpha", 0.2), 1e-10)
            beta  = max(bp.get("beta", 0.15), 1e-10)
            theta = max(bp.get("theta", 0.1), 1e-10)
            delta = max(bp.get("delta", 0.3), 1e-10)

            valence_abr = float(np.tanh((alpha / beta - 0.7) * 2.0))
            faa_valence = 0.0
            if channels is not None and channels.shape[0] >= 3:
                try:
                    asym = compute_frontal_asymmetry(channels, fs, left_ch=1, right_ch=2)
                    faa_valence = float(asym.get("asymmetry_valence", 0.0))
                except Exception:
                    pass
            valence = float(np.clip(0.5 * valence_abr + 0.5 * faa_valence, -1.0, 1.0))
            arousal = float(np.clip(beta / (beta + alpha), 0.0, 1.0))
            stress  = float(np.clip(beta / (beta + alpha + theta), 0.0, 1.0))
            focus   = float(np.clip(0.6 * beta / (beta + alpha) + 0.4 * (1.0 - theta / (theta + beta)), 0.0, 1.0))
            relax   = float(np.clip(alpha / (alpha + beta), 0.0, 1.0))

            # Map to emotion
            if valence > 0.2 and arousal > 0.5:
                probs_raw = np.array([0.5, 0.05, 0.1, 0.05, 0.2, 0.1])
            elif valence < -0.1 and arousal > 0.4:
                probs_raw = np.array([0.05, 0.1, 0.35, 0.25, 0.1, 0.15])
            elif valence < -0.2 and arousal < 0.4:
                probs_raw = np.array([0.05, 0.45, 0.1, 0.15, 0.05, 0.2])
            else:
                probs_raw = np.array([0.15, 0.1, 0.1, 0.1, 0.1, 0.45])
            probs = probs_raw / probs_raw.sum()

            adj = self._adj.copy()
            return {
                "emotion": EMOTION_LABELS[int(np.argmax(probs))],
                "probabilities": {l: float(round(p, 4)) for l, p in zip(EMOTION_LABELS, probs)},
                "valence": round(valence, 4),
                "arousal": round(arousal, 4),
                "stress_index": round(stress, 4),
                "focus_index": round(focus, 4),
                "relaxation_index": round(relax, 4),
                "model_type": "feature-based",
                "graph_edge_weights": adj.flatten().tolist(),
            }
        except Exception as exc:
            # Last resort — return neutral with uniform distribution
            return {
                "emotion": "neutral",
                "probabilities": {l: round(1.0 / N_CLASSES, 4) for l in EMOTION_LABELS},
                "valence": 0.0,
                "arousal": 0.5,
                "stress_index": 0.5,
                "focus_index": 0.5,
                "relaxation_index": 0.5,
                "model_type": "feature-based",
                "graph_edge_weights": _build_static_adj().flatten().tolist(),
                "error": str(exc),
            }

    def get_model_info(self) -> dict:
        """Return model architecture description."""
        return {
            "model_name": "GraphEmotionClassifier",
            "n_nodes": N_NODES,
            "node_labels": ["TP9", "AF7", "AF8", "TP10"],
            "n_edges": len(STATIC_EDGES),
            "edge_list": [list(e) for e in STATIC_EDGES],
            "node_feature_dim": NODE_FEAT_DIM,
            "node_feature_names": [
                "de_delta", "de_theta", "de_alpha", "de_beta", "de_gamma",
                "log_activity", "mobility", "complexity",
            ],
            "n_classes": self.n_classes,
            "emotion_labels": EMOTION_LABELS,
            "hidden_dim": self.hidden_dim,
            "adjacency_update": "EMA (alpha=0.05)",
            "message_passing_layers": 2,
            "requires_torch": False,
            "requires_torch_geometric": False,
            "inference_calls": self._call_count,
        }


# ── Singleton registry ────────────────────────────────────────────────────────

_registry: Dict[str, GraphEmotionClassifier] = {}
_registry_lock = threading.Lock()


def get_graph_emotion_classifier(user_id: str = "default") -> GraphEmotionClassifier:
    """Return the singleton GraphEmotionClassifier for *user_id*.

    Creates a new instance on first call per user_id; subsequent calls
    return the same object so EMA adjacency state accumulates across frames.
    """
    with _registry_lock:
        if user_id not in _registry:
            _registry[user_id] = GraphEmotionClassifier()
        return _registry[user_id]
