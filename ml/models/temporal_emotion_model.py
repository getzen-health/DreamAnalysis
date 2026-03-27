"""
Temporal Emotion Model -- LSTM over sequential EEG epochs.

Instead of classifying a single 4-second epoch, this model takes a sequence
of N consecutive epoch feature vectors and uses an LSTM to capture temporal
dynamics (e.g., gradual stress buildup, emotion transitions).

Architecture: Feature extraction (existing) -> LSTM(hidden=64) -> FC(6 emotions)
Input: sequence of feature vectors from consecutive epochs
Output: 6-class emotion probabilities + valence + arousal
"""

import logging
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class TemporalEEGNet(nn.Module if TORCH_AVAILABLE else object):
    """LSTM-based temporal emotion classifier.

    Takes a sequence of epoch feature vectors and predicts emotion from the
    temporal pattern, not just a single snapshot.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each epoch's feature vector (default 41 for
        multichannel features with DASM/RASM).
    hidden_dim : int
        LSTM hidden state size.
    num_classes : int
        Number of discrete emotion classes.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout between LSTM layers (only active when num_layers > 1).
    """

    def __init__(
        self,
        input_dim: int = 41,
        hidden_dim: int = 64,
        num_classes: int = 6,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TemporalEEGNet")
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_emotion = nn.Linear(hidden_dim, num_classes)
        self.fc_valence = nn.Linear(hidden_dim, 1)
        self.fc_arousal = nn.Linear(hidden_dim, 1)

    def forward(self, x: "torch.Tensor") -> tuple:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, input_dim)

        Returns
        -------
        emotion_logits : (batch, num_classes)
        valence : (batch, 1) in [-1, 1]
        arousal : (batch, 1) in [0, 1]
        """
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]  # use last hidden state
        emotion_logits = self.fc_emotion(last)
        valence = torch.tanh(self.fc_valence(last))
        arousal = torch.sigmoid(self.fc_arousal(last))
        return emotion_logits, valence, arousal


class TemporalEmotionClassifier:
    """Wrapper that accumulates epochs and runs temporal prediction.

    Maintains a sliding buffer of feature vectors from consecutive EEG epochs.
    Once `seq_length` epochs are buffered, `predict()` returns either a
    trained-LSTM prediction or a heuristic temporal analysis.

    Parameters
    ----------
    model_path : str or None
        Path to a saved TemporalEEGNet state dict (.pt file).
        If None or file missing, falls back to heuristic analysis.
    seq_length : int
        Number of consecutive epochs required before prediction.
    input_dim : int
        Feature vector dimensionality per epoch.
    """

    EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral"]

    def __init__(
        self,
        model_path: Optional[str] = None,
        seq_length: int = 10,
        input_dim: int = 41,
        hidden_dim: int = 64,
        num_classes: int = 6,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.buffer: List[np.ndarray] = []
        self.model: Optional[object] = None
        self.model_loaded = False

        if model_path and TORCH_AVAILABLE and Path(model_path).exists():
            try:
                self.model = TemporalEEGNet(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_classes=num_classes,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                self.model.load_state_dict(
                    torch.load(model_path, map_location="cpu", weights_only=True)
                )
                self.model.eval()
                self.model_loaded = True
                logger.info("Temporal LSTM model loaded from %s", model_path)
            except Exception as exc:
                logger.warning("Failed to load temporal model: %s", exc)
                self.model = None
                self.model_loaded = False

    def add_epoch(self, features: np.ndarray) -> None:
        """Add a feature vector from one epoch to the sliding buffer.

        If features have more dimensions than ``input_dim``, only the first
        ``input_dim`` values are kept.  The buffer is capped at
        ``seq_length`` entries (oldest are dropped).
        """
        if features.ndim != 1:
            features = features.flatten()
        if features.shape[0] > self.input_dim:
            features = features[: self.input_dim]
        elif features.shape[0] < self.input_dim:
            # Pad with zeros if the feature vector is shorter than expected
            padded = np.zeros(self.input_dim, dtype=features.dtype)
            padded[: features.shape[0]] = features
            features = padded
        self.buffer.append(features.copy())
        if len(self.buffer) > self.seq_length:
            self.buffer.pop(0)

    def is_ready(self) -> bool:
        """True when buffer has enough epochs for prediction."""
        return len(self.buffer) >= self.seq_length

    def predict(self) -> Optional[Dict]:
        """Run temporal prediction on buffered epochs.

        Returns None if the buffer is not yet full.  If no trained model
        is loaded, returns a heuristic temporal analysis instead.
        """
        if not self.is_ready():
            return None

        if not self.model_loaded or not TORCH_AVAILABLE:
            return self._predict_heuristic()

        # Model inference
        seq = np.stack(self.buffer[-self.seq_length :])
        x = torch.FloatTensor(seq).unsqueeze(0)  # (1, seq_len, features)

        with torch.no_grad():
            logits, valence, arousal = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()

        emotion_idx = int(probs.argmax())
        return {
            "emotion": self.EMOTIONS[emotion_idx],
            "probabilities": {
                e: float(p) for e, p in zip(self.EMOTIONS, probs)
            },
            "valence": float(valence.squeeze()),
            "arousal": float(arousal.squeeze()),
            "model_type": "temporal-lstm",
            "epochs_used": self.seq_length,
            "temporal_features": self._compute_temporal_features(),
        }

    def _predict_heuristic(self) -> Dict:
        """Heuristic temporal analysis without a trained model.

        Computes basic temporal statistics (trend, stability, variance)
        from the buffered epoch features.  Returns uniform emotion
        probabilities since no model is available to discriminate.
        """
        features = np.stack(self.buffer[-self.seq_length :])

        temporal_feats = self._compute_temporal_features()

        return {
            "emotion": "neutral",
            "probabilities": {e: 1.0 / len(self.EMOTIONS) for e in self.EMOTIONS},
            "valence": 0.0,
            "arousal": 0.5,
            "model_type": "temporal-heuristic",
            "epochs_used": self.seq_length,
            "temporal_features": temporal_feats,
        }

    def _compute_temporal_features(self) -> Dict:
        """Extract temporal dynamics from the buffer.

        Returns
        -------
        dict with:
            mean_stability : float
                1 minus mean feature standard deviation across epochs.
                Higher = more stable signal.
            trend_direction : float
                Mean difference between last and first epoch features.
                Positive = features increasing over time.
            variance : float
                Mean feature variance across epochs.
            epochs_buffered : int
                Current buffer length.
        """
        if len(self.buffer) < 2:
            return {}
        features = np.stack(self.buffer)
        return {
            "mean_stability": float(1.0 - np.mean(np.std(features, axis=0))),
            "trend_direction": float(np.mean(features[-1] - features[0])),
            "variance": float(np.mean(np.var(features, axis=0))),
            "epochs_buffered": len(self.buffer),
        }

    def reset(self) -> None:
        """Clear the epoch buffer."""
        self.buffer.clear()
