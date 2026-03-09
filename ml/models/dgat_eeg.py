"""Dynamic Graph Attention Network (DGAT) for EEG channel relationships.

Models 4 EEG channels as a dynamic graph where edges represent functional
connectivity (computed from band power correlations per epoch).

Architecture:
- Dynamic adjacency: pairwise Pearson correlation between channel band-power vectors
- Graph attention: 2-layer GAT with 4 attention heads
- Temporal: 1D-CNN for within-channel temporal features
- Classification: MLP head → 6 emotion classes

Without PyTorch: falls back to correlation-weighted feature averaging
(a simpler version of the same concept using numpy).

Reference:
    Veličković et al., "Graph Attention Networks." ICLR 2018.
    https://arxiv.org/abs/1710.10903

Muse 2 channel map (BrainFlow board_id=38):
    ch0 = TP9   (left temporal)
    ch1 = AF7   (left frontal)
    ch2 = AF8   (right frontal)
    ch3 = TP10  (right temporal)
"""

import threading
import numpy as np
from typing import Dict, List, Optional, Tuple


# ── PyTorch model (optional) ──────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _TemporalCNN(nn.Module):
        """Per-channel 1D temporal feature extractor.

        Produces a 128-dim embedding for a single EEG channel time series.

        Args:
            in_channels:  Number of input signal channels (1 per EEG channel).
            n_filters:    Number of conv filters (16 by default).
            kernel_size:  Temporal convolution kernel length.
            embed_dim:    Output embedding dimension.
        """

        def __init__(
            self,
            in_channels: int = 1,
            n_filters: int = 16,
            kernel_size: int = 32,
            embed_dim: int = 128,
        ):
            super().__init__()
            self.conv = nn.Conv1d(in_channels, n_filters, kernel_size=kernel_size, padding=kernel_size // 2)
            self.bn = nn.BatchNorm1d(n_filters)
            self.pool = nn.AdaptiveAvgPool1d(8)
            self.fc = nn.Linear(n_filters * 8, embed_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, 1, T)
            h = F.relu(self.bn(self.conv(x)))   # (batch, n_filters, T')
            h = self.pool(h)                    # (batch, n_filters, 8)
            h = h.view(h.size(0), -1)           # (batch, n_filters*8)
            return self.fc(h)                   # (batch, embed_dim)

    class _GATLayer(nn.Module):
        """Single Graph Attention layer with multi-head attention.

        Implements the original GAT attention mechanism from Veličković et al.:
            α_ij = softmax( LeakyReLU( a^T [W h_i || W h_j] ) )
            h'_i = concat_k { σ( Σ_j α^k_ij  W^k h_j ) }

        Args:
            in_dim:     Input feature dimension per node.
            out_dim:    Output dimension per attention head.
            n_heads:    Number of attention heads.
            alpha:      LeakyReLU negative slope.
            concat:     If True, concat heads; else average them.
        """

        def __init__(
            self,
            in_dim: int,
            out_dim: int,
            n_heads: int = 4,
            alpha: float = 0.2,
            concat: bool = True,
        ):
            super().__init__()
            self.n_heads = n_heads
            self.out_dim = out_dim
            self.concat = concat

            # Linear projection per head
            self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)
            # Attention coefficients per head: a^T is a (2*out_dim,) vector per head
            self.a = nn.Parameter(torch.empty(n_heads, 2 * out_dim))
            nn.init.xavier_uniform_(self.a.unsqueeze(0))

            self.leakyrelu = nn.LeakyReLU(alpha)

        def forward(
            self,
            x: "torch.Tensor",
            adj: "torch.Tensor",
        ) -> "torch.Tensor":
            """
            Args:
                x:   (batch, n_nodes, in_dim)
                adj: (batch, n_nodes, n_nodes) — normalized adjacency weights [0,1]

            Returns:
                (batch, n_nodes, out_dim*n_heads) if concat else (batch, n_nodes, out_dim)
            """
            B, N, _ = x.shape
            # Project: (B, N, n_heads*out_dim)
            Wh = self.W(x).view(B, N, self.n_heads, self.out_dim)  # (B, N, H, D)

            # Compute attention scores for all pairs (i,j):
            # e_ij = LeakyReLU( a^T [Wh_i || Wh_j] )
            Wh_i = Wh.unsqueeze(3).expand(B, N, self.n_heads, N, self.out_dim)
            Wh_j = Wh.unsqueeze(2).expand(B, N, self.n_heads, N, self.out_dim)
            # Transpose so dims are (B, H, N, N, D)
            Wh_i = Wh_i.permute(0, 2, 1, 3, 4)
            Wh_j = Wh_j.permute(0, 2, 1, 3, 4)

            e = torch.cat([Wh_i, Wh_j], dim=-1)          # (B, H, N, N, 2D)
            # a: (H, 2D) → (1, H, 1, 1, 2D) for broadcasting
            a_exp = self.a.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            e = self.leakyrelu((e * a_exp).sum(dim=-1))   # (B, H, N, N)

            # Mask with adjacency: zero out non-edges before softmax
            adj_exp = adj.unsqueeze(1).expand_as(e)        # (B, H, N, N)
            e = e.masked_fill(adj_exp < 1e-9, float("-inf"))
            alpha = F.softmax(e, dim=-1)                   # (B, H, N, N)
            # Replace nan from rows of all -inf (isolated nodes)
            alpha = torch.nan_to_num(alpha, nan=0.0)

            # Aggregate: (B, H, N, D)
            Wh_val = Wh.permute(0, 2, 1, 3)               # (B, H, N, D)
            out = torch.matmul(alpha, Wh_val)              # (B, H, N, D)

            if self.concat:
                out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.n_heads * self.out_dim)
            else:
                out = out.mean(dim=1)                      # (B, N, D)

            return F.elu(out)

    class _DGATNet(nn.Module):
        """Full DGAT network.

        Computes dynamic adjacency per batch from pre-extracted band-power features,
        then runs 2-layer GAT over the channel graph, followed by an MLP classifier.

        Args:
            n_channels:  Number of EEG channels (4 for Muse 2).
            n_classes:   Number of output classes.
            embed_dim:   Temporal CNN output dimension (128).
            gat_dim:     Per-head GAT feature dimension (64).
            n_heads:     Number of GAT attention heads (4).
        """

        def __init__(
            self,
            n_channels: int = 4,
            n_classes: int = 6,
            embed_dim: int = 128,
            gat_dim: int = 64,
            n_heads: int = 4,
        ):
            super().__init__()
            self.n_channels = n_channels

            # Per-channel temporal CNN
            self.temporal_cnn = _TemporalCNN(in_channels=1, embed_dim=embed_dim)

            # Layer 1: embed_dim → gat_dim * n_heads (= 256)
            self.gat1 = _GATLayer(embed_dim, gat_dim, n_heads=n_heads, concat=True)
            gat1_out = gat_dim * n_heads  # 256

            # Layer 2: gat1_out → gat_dim (= 64, averaged across 2 heads)
            self.gat2 = _GATLayer(gat1_out, gat_dim * 2, n_heads=2, concat=False)
            gat2_out = gat_dim * 2  # 128

            # Global mean pool across channels, then MLP
            self.fc1 = nn.Linear(gat2_out, 64)
            self.dropout = nn.Dropout(0.4)
            self.fc2 = nn.Linear(64, n_classes)

        def _build_adj(self, node_features: "torch.Tensor") -> "torch.Tensor":
            """Build dynamic adjacency matrix from node features via Pearson correlation.

            Args:
                node_features: (batch, n_nodes, feature_dim)

            Returns:
                adj: (batch, n_nodes, n_nodes), values in [0, 1], softmax-normalised per row
            """
            # Normalize each node feature vector
            mean = node_features.mean(dim=-1, keepdim=True)
            std = node_features.std(dim=-1, keepdim=True).clamp(min=1e-8)
            norm = (node_features - mean) / std                  # (B, N, D)

            # Pearson corr matrix via batch matmul
            corr = torch.bmm(norm, norm.transpose(1, 2)) / norm.size(-1)  # (B, N, N)

            # Clip to [0, 1]: only positive correlations form edges
            adj = corr.clamp(min=0.0, max=1.0)

            # Softmax-normalise each row so weights sum to 1 (stable aggregation)
            adj = F.softmax(adj, dim=-1)
            return adj

        def forward(
            self,
            eeg: "torch.Tensor",
            band_feats: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """
            Args:
                eeg:        (batch, n_channels, T) raw EEG
                band_feats: (batch, n_channels, n_band_features) per-channel band powers

            Returns:
                logits: (batch, n_classes)
                adj:    (batch, n_channels, n_channels) dynamic adjacency
            """
            B, C, T = eeg.shape

            # Per-channel temporal embeddings
            ch_embeds = []
            for c in range(C):
                ch_in = eeg[:, c:c+1, :]                     # (B, 1, T)
                ch_embeds.append(self.temporal_cnn(ch_in))   # (B, embed_dim)
            node_h = torch.stack(ch_embeds, dim=1)            # (B, C, embed_dim)

            # Dynamic adjacency from band-power feature correlations
            adj = self._build_adj(band_feats)                 # (B, C, C)

            # 2-layer GAT
            h1 = self.gat1(node_h, adj)                      # (B, C, 256)
            h2 = self.gat2(h1, adj)                          # (B, C, 128)

            # Global mean pooling over channels
            g = h2.mean(dim=1)                               # (B, 128)

            out = self.dropout(F.relu(self.fc1(g)))          # (B, 64)
            logits = self.fc2(out)                           # (B, n_classes)
            return logits, adj

    _TORCH_AVAILABLE = True

except ImportError:
    _TORCH_AVAILABLE = False
    _DGATNet = None


# ── Classifier wrapper ────────────────────────────────────────────────────────

EMOTION_LABELS = ["happy", "sad", "angry", "fear", "surprise", "neutral"]
_BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]


def _extract_band_power_vector(signal: np.ndarray, fs: float) -> np.ndarray:
    """Return a 5-element band-power vector (delta, theta, alpha, beta, gamma)
    for a single-channel 1-D signal using Welch PSD.

    Falls back to a uniform vector if scipy is unavailable.
    """
    try:
        from scipy.signal import welch
        nperseg = min(len(signal), int(fs * 2))
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

        def _band(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return float(np.mean(psd[mask])) if mask.any() else 1e-10

        return np.array([
            _band(0.5, 4),    # delta
            _band(4, 8),      # theta
            _band(8, 12),     # alpha
            _band(12, 30),    # beta
            _band(30, 50),    # gamma
        ], dtype=np.float64)
    except Exception:
        return np.ones(5, dtype=np.float64) / 5.0


def _compute_dynamic_adjacency(features_matrix: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation adjacency matrix from per-channel features.

    Args:
        features_matrix: (n_channels, n_features) — each row is one channel's feature vector

    Returns:
        adj: (n_channels, n_channels) in [0, 1], softmax-normalised per row
    """
    # np.corrcoef returns (n_channels, n_channels) correlation matrix
    if features_matrix.shape[0] < 2:
        return np.ones((features_matrix.shape[0], features_matrix.shape[0]))

    corr = np.corrcoef(features_matrix)                   # (C, C)
    corr = np.nan_to_num(corr, nan=0.0)

    # Clip: only positive correlations form edges (similarity graph)
    adj = np.clip(corr, 0.0, 1.0)

    # Softmax per row
    def _softmax_row(row: np.ndarray) -> np.ndarray:
        e = np.exp(row - row.max())
        return e / (e.sum() + 1e-12)

    adj = np.array([_softmax_row(adj[i]) for i in range(adj.shape[0])])
    return adj


def _numpy_predict_6class(
    features_matrix: np.ndarray,
    adj: np.ndarray,
    fs: float,
) -> Dict:
    """NumPy fallback: graph-weighted feature aggregation → 6-class emotion.

    Steps:
    1. Weighted average of channel band-power features using adjacency weights.
    2. Derive valence/arousal from aggregated features.
    3. Map (valence, arousal) → 6-class probabilities.
    """
    # Graph-weighted average: each channel's features weighted by its row in adj
    agg = adj @ features_matrix                             # (C, n_features)
    mean_agg = agg.mean(axis=0)                            # (n_features,)

    # Band indices: [delta=0, theta=1, alpha=2, beta=3, gamma=4]
    delta, theta, alpha, beta, gamma = (
        max(mean_agg[i], 1e-10) for i in range(5)
    )

    alpha_beta = alpha / (beta + 1e-10)
    theta_beta = theta / (beta + 1e-10)
    high_beta_frac = min(beta / (alpha + beta + 1e-10), 1.0)

    # FAA proxy from per-channel alpha (ch1=AF7 left, ch2=AF8 right if available)
    faa_valence = 0.0
    n_ch = features_matrix.shape[0]
    if n_ch >= 3:
        left_alpha = features_matrix[1, 2]    # ch1 (AF7) alpha
        right_alpha = features_matrix[2, 2]   # ch2 (AF8) alpha
        faa_raw = np.log(right_alpha + 1e-10) - np.log(left_alpha + 1e-10)
        faa_valence = float(np.clip(np.tanh(faa_raw * 2.0), -1.0, 1.0))

    # Valence: blend alpha/beta ratio + FAA
    valence_abr = float(np.clip(np.tanh((alpha_beta - 0.7) * 2.0), -1.0, 1.0))
    valence = float(np.clip(0.50 * valence_abr + 0.50 * faa_valence, -1.0, 1.0))

    # Arousal: beta dominance + inverse delta
    arousal = float(np.clip(
        0.45 * beta / (beta + alpha + 1e-10)
        + 0.30 * (1 - alpha / (alpha + beta + theta + 1e-10))
        + 0.25 * (1 - delta / (delta + beta + 1e-10)),
        0.0, 1.0,
    ))

    # Stress: high-beta fraction
    stress = float(np.clip(0.45 * high_beta_frac + 0.30 * (theta_beta * 0.3) + 0.25 * high_beta_frac, 0.0, 1.0))

    # Map to 6-class probabilities
    probs = np.zeros(6, dtype=np.float64)
    # happy:    positive valence + high arousal
    probs[0] = max(0.0, valence) * (0.5 + 0.5 * arousal)
    # sad:      negative valence + low arousal
    probs[1] = max(0.0, -valence - 0.1) * (1.0 - arousal)
    # angry:    negative valence + high arousal + stress
    probs[2] = max(0.0, (-valence) * 0.5 + arousal * 0.3 + stress * 0.2)
    # fear:     negative valence + moderate-high arousal
    probs[3] = max(0.0, (-valence) * 0.4 + arousal * 0.4 + stress * 0.2)
    # surprise: high arousal + near-neutral valence
    probs[4] = arousal * max(0.0, 1.0 - abs(valence) * 2)
    # neutral:  baseline
    probs[5] = max(0.05, 1.0 - sum(probs[:5]) * 0.8)

    # Normalise
    total = probs.sum()
    if total > 0:
        probs /= total
    else:
        probs = np.ones(6) / 6.0

    pred_idx = int(np.argmax(probs))
    return {
        "emotion": EMOTION_LABELS[pred_idx],
        "probabilities": {EMOTION_LABELS[i]: round(float(probs[i]), 4) for i in range(6)},
        "valence": round(valence, 4),
        "arousal": round(arousal, 4),
        "stress_index": round(stress, 4),
        "model_type": "dgat_numpy_fallback",
    }


class DGATEmotionClassifier:
    """Dynamic Graph Attention Network for EEG emotion classification.

    Models the 4 Muse 2 EEG channels as a dynamic graph. Edges are recomputed
    each epoch from pairwise band-power correlations (functional connectivity).
    Graph attention aggregates spatially-aware channel features.

    PyTorch path (when available):
        - Temporal 1D-CNN per channel → 128-dim embedding
        - 2-layer GAT (4 heads → 2 heads) over dynamic channel graph
        - MLP: 128 → 64 → 6 classes

    NumPy fallback (no PyTorch):
        - 5-band Welch PSD per channel
        - Dynamic adjacency from Pearson correlation of band vectors
        - Graph-weighted feature average → heuristic 6-class mapping

    Args:
        n_channels:  Number of EEG channels (default 4 for Muse 2).
        n_classes:   Number of emotion classes (6: happy/sad/angry/fear/surprise/neutral).
        fs:          Default sampling rate in Hz.
        model_path:  Optional path to saved PyTorch weights (.pt file).
    """

    def __init__(
        self,
        n_channels: int = 4,
        n_classes: int = 6,
        fs: float = 256.0,
        model_path: Optional[str] = None,
    ):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.fs = fs

        # Last computed adjacency stats (updated on every predict() call)
        self._last_adj: Optional[np.ndarray] = None

        # PyTorch model (None if unavailable or weights not found)
        self._net: Optional["_DGATNet"] = None
        self._model_type: str = "dgat_numpy_fallback"

        if _TORCH_AVAILABLE and model_path:
            self._load(model_path)

    def _load(self, model_path: str) -> None:
        """Load saved PyTorch weights."""
        try:
            import torch
            state = torch.load(model_path, map_location="cpu")
            self._net = _DGATNet(n_channels=self.n_channels, n_classes=self.n_classes)
            self._net.load_state_dict(state.get("model_state_dict", state))
            self._net.eval()
            self._model_type = "dgat_pytorch"
        except Exception as exc:
            import logging
            logging.getLogger("dgat_eeg").warning(f"[DGAT] Failed to load weights from {model_path}: {exc}")
            self._net = None
            self._model_type = "dgat_numpy_fallback"

    def _prepare_input(self, eeg: np.ndarray) -> np.ndarray:
        """Normalise EEG to shape (n_channels, n_samples).

        Accepts:
          - (n_samples,)               → single channel; replicated to n_channels
          - (n_channels, n_samples)    → used as-is (possibly truncated/padded)
          - (n_channels,)              → treated as n_channels points, replicated
        """
        eeg = np.asarray(eeg, dtype=np.float64)
        if eeg.ndim == 1:
            # Replicate single channel across all channels
            eeg = np.tile(eeg, (self.n_channels, 1))
        elif eeg.ndim == 2:
            # Accept any number of channels; trim or pad to n_channels
            if eeg.shape[0] < self.n_channels:
                pad = np.zeros((self.n_channels - eeg.shape[0], eeg.shape[1]))
                eeg = np.vstack([eeg, pad])
            elif eeg.shape[0] > self.n_channels:
                eeg = eeg[:self.n_channels, :]
        else:
            raise ValueError(f"Expected 1-D or 2-D EEG array, got shape {eeg.shape}")
        return eeg

    def _build_band_features(self, eeg: np.ndarray, fs: float) -> np.ndarray:
        """Compute (n_channels, 5) band-power feature matrix."""
        return np.array([
            _extract_band_power_vector(eeg[c], fs) for c in range(eeg.shape[0])
        ], dtype=np.float64)

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Classify emotion using the dynamic graph attention model.

        Args:
            eeg: (n_channels, n_samples) or (n_samples,) numpy array in µV.
            fs:  Sampling rate in Hz. Defaults to the value set at construction.

        Returns:
            dict with keys:
                emotion           — predicted label (str)
                probabilities     — 6-class probability dict
                valence           — float in [-1, 1]
                arousal           — float in [0, 1]
                graph_connectivity — mean adjacency weight (float in [0, 1])
                model_type        — "dgat_pytorch" or "dgat_numpy_fallback"
        """
        if fs == 256.0 and self.fs != 256.0:
            fs = self.fs

        eeg = self._prepare_input(eeg)
        band_feats = self._build_band_features(eeg, fs)     # (C, 5)
        adj = _compute_dynamic_adjacency(band_feats)         # (C, C)
        self._last_adj = adj.copy()

        graph_connectivity = float(np.mean(adj))

        if self._net is not None and _TORCH_AVAILABLE:
            result = self._predict_pytorch(eeg, band_feats, fs)
        else:
            result = _numpy_predict_6class(band_feats, adj, fs)

        result["graph_connectivity"] = round(graph_connectivity, 4)
        return result

    def _predict_pytorch(
        self,
        eeg: np.ndarray,
        band_feats: np.ndarray,
        fs: float,
    ) -> Dict:
        """Forward pass through the full DGAT network."""
        import torch

        # Ensure minimum length for adaptive pool
        min_len = 64
        if eeg.shape[1] < min_len:
            pad = np.zeros((eeg.shape[0], min_len - eeg.shape[1]))
            eeg = np.concatenate([eeg, pad], axis=1)

        eeg_t = torch.FloatTensor(eeg).unsqueeze(0)               # (1, C, T)
        bf_t = torch.FloatTensor(band_feats).unsqueeze(0)         # (1, C, 5)

        with torch.no_grad():
            logits, adj_t = self._net(eeg_t, bf_t)
            probs_t = torch.softmax(logits, dim=1)[0]
            adj_np = adj_t[0].cpu().numpy()

        self._last_adj = adj_np
        probs = probs_t.cpu().numpy()
        pred_idx = int(np.argmax(probs))

        # Derive valence/arousal from class probabilities
        # positive emotions: happy(0), surprise(4)  → valence ↑
        # negative emotions: sad(1), angry(2), fear(3) → valence ↓
        valence = float(probs[0] + 0.5 * probs[4] - probs[1] - probs[2] - probs[3])
        valence = float(np.clip(valence, -1.0, 1.0))
        # arousal: high for happy, angry, fear, surprise; low for sad, neutral
        arousal = float(probs[0] + probs[2] + probs[3] + 0.7 * probs[4])
        arousal = float(np.clip(arousal, 0.0, 1.0))

        return {
            "emotion": EMOTION_LABELS[pred_idx],
            "probabilities": {EMOTION_LABELS[i]: round(float(probs[i]), 4) for i in range(6)},
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "stress_index": round(float(probs[2] + probs[3]), 4),
            "model_type": "dgat_pytorch",
        }

    def get_graph_stats(self) -> Dict:
        """Return statistics about the most recently computed dynamic adjacency matrix.

        Returns:
            dict with keys: mean, max, sparsity (fraction of edges with weight < 0.1)
        """
        if self._last_adj is None:
            return {"mean": 0.0, "max": 0.0, "sparsity": 1.0}

        adj = self._last_adj
        mean_w = float(np.mean(adj))
        max_w = float(np.max(adj))
        # Sparsity: fraction of off-diagonal entries below threshold 0.1
        n = adj.shape[0]
        if n < 2:
            sparsity = 0.0
        else:
            off_diag = adj[~np.eye(n, dtype=bool)]
            sparsity = float(np.mean(off_diag < 0.1))

        return {
            "mean": round(mean_w, 4),
            "max": round(max_w, 4),
            "sparsity": round(sparsity, 4),
        }


# ── Singleton registry ────────────────────────────────────────────────────────

_dgat_classifiers: Dict[str, DGATEmotionClassifier] = {}
_dgat_lock = threading.Lock()


def get_dgat_classifier(user_id: str = "default") -> DGATEmotionClassifier:
    """Return a per-user DGATEmotionClassifier singleton.

    Classifiers are created on first access and reused for subsequent calls.
    Thread-safe via a module-level lock.

    Args:
        user_id: Identifier for per-user isolation (matches other model getters).

    Returns:
        DGATEmotionClassifier instance.
    """
    with _dgat_lock:
        if user_id not in _dgat_classifiers:
            _dgat_classifiers[user_id] = DGATEmotionClassifier()
        return _dgat_classifiers[user_id]
