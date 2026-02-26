"""TSception — Temporal-Spatial CNN for EEG emotion recognition.

Architecture designed for asymmetric multi-scale temporal + spatial EEG features.
Especially suited for Muse 2's AF7/AF8 (left/right frontal) asymmetry.

Reference:
    Liu et al., "TSception: Capturing Temporal Dynamics and Spatial Asymmetry
    from EEG for Emotion Recognition." IEEE TAFFC 2022.
    DOI: 10.1109/TAFFC.2022.3169001

Key design choices for Muse 2:
    - 3 temporal filter scales: ~1 min, ~2 min, ~4 min (maps to β/α/θ rhythms)
    - 4-channel spatial conv captures AF7/AF8 asymmetry and TP9/TP10 temporal activity
    - Batch norm + dropout for small consumer-EEG datasets
    - Input shape: (batch, 1, n_channels=4, n_time_points)
"""

import numpy as np
from typing import Dict, Optional, Tuple


# ── PyTorch model ─────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _TSceptionNet(nn.Module):
        """TSception network.

        Args:
            n_classes:      Number of output classes.
            input_size:     (n_channels, n_time_points) — e.g. (4, 1024) for 4s @ 256 Hz.
            sampling_rate:  EEG sampling rate in Hz.
            num_T:          Number of temporal (multi-scale) filters.
            num_S:          Number of spatial filters.
            hidden:         Size of FC hidden layer.
            dropout_rate:   Dropout probability.
        """

        def __init__(
            self,
            n_classes: int = 3,
            input_size: Tuple[int, int] = (4, 1024),
            sampling_rate: float = 256.0,
            num_T: int = 15,
            num_S: int = 15,
            hidden: int = 32,
            dropout_rate: float = 0.5,
        ):
            super().__init__()
            n_channels, n_time = input_size
            self.n_classes = n_classes

            # ── Multi-scale temporal convolutions ────────────────────────────
            # Three kernel sizes spanning different rhythm windows:
            #   ~1/2 s, ~1 s, ~2 s → captures β, α, θ oscillations
            half = int(sampling_rate * 0.5) + 1
            one  = int(sampling_rate) + 1
            two  = int(sampling_rate * 2.0) + 1

            # Each temporal conv: (num_T, 1, 1, kernel_size)
            self.temporal_conv1 = nn.Conv2d(1, num_T, kernel_size=(1, half), padding=(0, half // 2))
            self.temporal_conv2 = nn.Conv2d(1, num_T, kernel_size=(1, one),  padding=(0, one // 2))
            self.temporal_conv3 = nn.Conv2d(1, num_T, kernel_size=(1, two),  padding=(0, two // 2))

            self.bn_t1 = nn.BatchNorm2d(num_T)
            self.bn_t2 = nn.BatchNorm2d(num_T)
            self.bn_t3 = nn.BatchNorm2d(num_T)

            self.avg_pool_t = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))

            # ── Spatial convolution (asymmetric — full channel dimension) ────
            # Applied to concatenated temporal features: (3*num_T, n_channels, t')
            self.spatial_conv = nn.Conv2d(3 * num_T, num_S, kernel_size=(n_channels, 1))
            self.bn_s = nn.BatchNorm2d(num_S)
            self.avg_pool_s = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))

            # ── Classifier ───────────────────────────────────────────────────
            # Compute flattened size after convolutions
            t_out = n_time // 8
            s_out = t_out // 8
            self.fc_in = num_S * 1 * max(s_out, 1)

            self.fc1 = nn.Linear(self.fc_in, hidden)
            self.bn_fc = nn.BatchNorm1d(hidden)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(hidden, n_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, 1, n_channels, n_time)

            # Multi-scale temporal features
            t1 = self.avg_pool_t(F.elu(self.bn_t1(self.temporal_conv1(x))))
            t2 = self.avg_pool_t(F.elu(self.bn_t2(self.temporal_conv2(x))))
            t3 = self.avg_pool_t(F.elu(self.bn_t3(self.temporal_conv3(x))))

            # Concatenate along channel (filter) dimension: (batch, 3*num_T, n_channels, t')
            t_cat = torch.cat([t1, t2, t3], dim=1)

            # Spatial conv: collapse channel dimension
            s = self.avg_pool_s(F.elu(self.bn_s(self.spatial_conv(t_cat))))
            # s: (batch, num_S, 1, t'')

            flat = s.view(s.size(0), -1)
            out = self.dropout(F.elu(self.bn_fc(self.fc1(flat))))
            return self.fc2(out)

    _TORCH_AVAILABLE = True

except ImportError:
    _TORCH_AVAILABLE = False
    _TSceptionNet = None


# ── Wrapper class (mirrors sklearn-style .predict()) ─────────────────────────

class TSceptionClassifier:
    """Wraps _TSceptionNet with the same predict() interface as other model classes.

    Falls back to feature-based heuristics if PyTorch is not available.
    """

    EMOTION_LABELS = ["negative", "neutral", "positive"]

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_classes: int = 3,
        sampling_rate: float = 256.0,
        n_channels: int = 4,
        epoch_sec: float = 4.0,
    ):
        self.n_classes = n_classes
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.epoch_sec = epoch_sec
        self.n_time = int(sampling_rate * epoch_sec)

        self.net: Optional["_TSceptionNet"] = None
        self.device = "cpu"
        self.model_type = "feature-based"

        if model_path and _TORCH_AVAILABLE:
            self._load(model_path)

    def _load(self, model_path: str) -> None:
        """Load saved PyTorch weights."""
        try:
            import torch
            state = torch.load(model_path, map_location="cpu")
            self.net = _TSceptionNet(
                n_classes=self.n_classes,
                input_size=(self.n_channels, self.n_time),
                sampling_rate=self.sampling_rate,
            )
            self.net.load_state_dict(state["model_state_dict"])
            self.net.eval()
            self.model_type = "tsception"
        except Exception as e:
            print(f"[TSception] Failed to load {model_path}: {e}")

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Classify emotion from EEG.

        Args:
            eeg: (n_channels, n_samples) or (n_samples,) numpy array.
            fs:  Sampling rate in Hz.

        Returns:
            Dict with 'emotion', 'probabilities', 'valence', 'confidence'.
        """
        if self.net is not None and _TORCH_AVAILABLE:
            return self._predict_net(eeg, fs)
        return self._predict_heuristic(eeg, fs)

    def _predict_net(self, eeg: np.ndarray, fs: float) -> Dict:
        """Forward pass through TSception network."""
        import torch

        # Ensure shape (n_channels, n_time)
        if eeg.ndim == 1:
            eeg = np.tile(eeg, (self.n_channels, 1))
        if eeg.shape[1] < self.n_time:
            # Pad with zeros
            pad = np.zeros((eeg.shape[0], self.n_time - eeg.shape[1]))
            eeg = np.concatenate([eeg, pad], axis=1)
        else:
            eeg = eeg[:, :self.n_time]

        x = torch.FloatTensor(eeg).unsqueeze(0).unsqueeze(0)  # (1, 1, C, T)
        with torch.no_grad():
            logits = self.net(x)                               # (1, n_classes)
            probs = torch.softmax(logits, dim=1)[0].numpy()

        pred_idx = int(np.argmax(probs))
        label = self.EMOTION_LABELS[min(pred_idx, len(self.EMOTION_LABELS) - 1)]
        valence = float(probs[2] - probs[0])   # positive prob − negative prob

        return {
            "emotion": label,
            "emotion_index": pred_idx,
            "probabilities": {
                self.EMOTION_LABELS[i]: round(float(probs[i]), 4)
                for i in range(len(probs))
            },
            "valence": round(valence, 4),
            "confidence": round(float(probs[pred_idx]), 4),
            "model_type": "tsception",
        }

    def _predict_heuristic(self, eeg: np.ndarray, fs: float) -> Dict:
        """Feature-based fallback (no PyTorch required)."""
        from processing.eeg_processor import extract_features, preprocess

        signal = eeg[0] if eeg.ndim == 2 else eeg
        processed = preprocess(signal, fs)
        feat = extract_features(processed, fs)

        alpha = feat.get("alpha", 0.2)
        beta  = feat.get("beta", 0.2)
        theta = feat.get("theta", 0.1)

        valence_raw = float(np.tanh((alpha / (beta + 1e-8) - 1.0) * 1.5))
        arousal_raw = float(np.tanh((beta  / (alpha + 1e-8) - 0.5) * 2.0))

        if valence_raw > 0.2 and arousal_raw > 0.0:
            pred_idx = 2   # positive
        elif valence_raw < -0.2:
            pred_idx = 0   # negative
        else:
            pred_idx = 1   # neutral

        probs = np.zeros(3)
        probs[pred_idx] = 0.6
        probs[(pred_idx + 1) % 3] = 0.25
        probs[(pred_idx + 2) % 3] = 0.15

        label = self.EMOTION_LABELS[pred_idx]
        return {
            "emotion": label,
            "emotion_index": pred_idx,
            "probabilities": {
                self.EMOTION_LABELS[i]: round(float(probs[i]), 4)
                for i in range(3)
            },
            "valence": round(valence_raw, 4),
            "confidence": 0.6,
            "model_type": "feature-based",
        }
