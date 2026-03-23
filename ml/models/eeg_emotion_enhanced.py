"""Enhanced emotion classifier combining SSL learned representations + hand-crafted features.

Combines three feature sources:
    1. Pretrained SSL encoder: 64-dim learned representation (from eeg_ssl.py)
    2. Hand-crafted features: 53-dim DE/asymmetry/Hjorth (from emotion_features_enhanced.py)
    3. Temporal delta features: 53-dim (optional, from emotion_features_enhanced.py)

Total input to classifier: 64 + 53 + 53 = 170 features (or 64 + 53 = 117 without temporal)

Architecture:
    Linear(170, 128) -> BN -> ReLU -> Dropout(0.3)
    Linear(128, 64)  -> ReLU -> Dropout(0.2)
    Linear(64, n_classes)

References:
    - CNN-KAN-F2CA (PLOS ONE 2025): hybrid CNN + hand-crafted features
    - MDPI Mathematics (2025): SSL pretraining + fine-tuning on consumer EEG
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.eeg_ssl import EEGContrastiveEncoder
from processing.emotion_features_enhanced import ENHANCED_FEATURE_DIM

log = logging.getLogger(__name__)

# Default emotion classes (matches existing emotion_classifier.py)
EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]

if TORCH_AVAILABLE:

    class EnhancedEmotionClassifier(nn.Module):
        """Hybrid classifier combining SSL encoder + hand-crafted EEG features.

        The SSL encoder backbone is frozen by default (features only).
        The classifier head is trainable and combines learned + engineered features.

        Args:
            n_classes: Number of emotion classes.
            pretrained_encoder: Optional pretrained EEGContrastiveEncoder.
                If None, a fresh encoder is created (untrained).
            include_temporal: Whether to include temporal delta features (53 extra dims).
            freeze_encoder: Whether to freeze the SSL encoder weights.
        """

        def __init__(
            self,
            n_classes: int = 6,
            pretrained_encoder: Optional["EEGContrastiveEncoder"] = None,
            include_temporal: bool = True,
            freeze_encoder: bool = True,
        ):
            super().__init__()

            self.n_classes = n_classes
            self.include_temporal = include_temporal
            self.freeze_encoder = freeze_encoder

            # SSL encoder
            if pretrained_encoder is not None:
                self.encoder = pretrained_encoder
            else:
                self.encoder = EEGContrastiveEncoder(
                    n_channels=4, n_samples=1024, embed_dim=64, proj_dim=32
                )

            # Freeze encoder if requested
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False

            # Feature dimensions
            ssl_dim = self.encoder.embed_dim  # 64
            handcrafted_dim = ENHANCED_FEATURE_DIM  # 53
            temporal_dim = ENHANCED_FEATURE_DIM if include_temporal else 0  # 53 or 0
            total_dim = ssl_dim + handcrafted_dim + temporal_dim  # 170 or 117

            self._total_dim = total_dim

            # Classifier head
            self.classifier = nn.Sequential(
                nn.Linear(total_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, n_classes),
            )

            self._init_classifier_weights()

        def _init_classifier_weights(self) -> None:
            """Initialize only the classifier head weights."""
            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(
            self,
            raw_eeg: "torch.Tensor",
            handcrafted_features: "torch.Tensor",
        ) -> "torch.Tensor":
            """Forward pass combining SSL representation and hand-crafted features.

            Args:
                raw_eeg: (batch, n_channels, n_samples) raw EEG tensor.
                handcrafted_features: (batch, 53) or (batch, 106) hand-crafted feature tensor.
                    If include_temporal=True, expects 106 dims (53 instant + 53 delta).
                    If include_temporal=False, expects 53 dims.

            Returns:
                (batch, n_classes) logits.
            """
            # SSL encoding
            if self.freeze_encoder:
                with torch.no_grad():
                    learned = self.encoder.encode(raw_eeg)
            else:
                learned = self.encoder.encode(raw_eeg)

            # Combine features
            combined = torch.cat([learned, handcrafted_features], dim=1)

            return self.classifier(combined)

        def predict_proba(
            self,
            raw_eeg: "torch.Tensor",
            handcrafted_features: "torch.Tensor",
        ) -> np.ndarray:
            """Return softmax probabilities as numpy array."""
            self.eval()
            with torch.no_grad():
                logits = self(raw_eeg, handcrafted_features)
                return F.softmax(logits, dim=-1).cpu().numpy()

        def predict_emotion(
            self,
            raw_eeg: "torch.Tensor",
            handcrafted_features: "torch.Tensor",
        ) -> dict:
            """Return full emotion prediction dict (matches emotion_classifier.py output format).

            Returns:
                Dict with 'emotion', 'probabilities', 'model_type'.
            """
            probs = self.predict_proba(raw_eeg, handcrafted_features)
            if probs.ndim == 2:
                probs = probs[0]  # single sample

            emotion_idx = int(np.argmax(probs))
            return {
                "emotion": EMOTIONS[emotion_idx],
                "probabilities": {
                    EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))
                },
                "model_type": "enhanced_ssl_handcrafted",
            }

        # --- Serialization ---

        def save(self, path: str | Path) -> None:
            """Save full model (encoder + classifier) and config."""
            torch.save(
                {
                    "state_dict": self.state_dict(),
                    "config": {
                        "n_classes": self.n_classes,
                        "include_temporal": self.include_temporal,
                        "freeze_encoder": self.freeze_encoder,
                        "encoder_config": {
                            "n_channels": self.encoder.n_channels,
                            "n_samples": self.encoder.n_samples,
                            "embed_dim": self.encoder.embed_dim,
                            "proj_dim": self.encoder.proj_dim,
                        },
                    },
                },
                path,
            )
            log.info("EnhancedEmotionClassifier saved -> %s", path)

        @classmethod
        def load(cls, path: str | Path) -> "EnhancedEmotionClassifier":
            """Load model from checkpoint."""
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            config = ckpt["config"]

            encoder = EEGContrastiveEncoder(**config["encoder_config"])

            model = cls(
                n_classes=config["n_classes"],
                pretrained_encoder=encoder,
                include_temporal=config["include_temporal"],
                freeze_encoder=config["freeze_encoder"],
            )
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            log.info("EnhancedEmotionClassifier loaded <- %s", path)
            return model

        def export_onnx(self, path: str | Path) -> None:
            """Export to ONNX for on-device inference.

            The ONNX model takes two inputs:
                - raw_eeg: (batch, 4, 1024)
                - handcrafted_features: (batch, 106) or (batch, 53) depending on include_temporal
            """
            self.eval()
            batch_size = 1
            dummy_eeg = torch.zeros(batch_size, self.encoder.n_channels, self.encoder.n_samples)
            expected_feat_dim = ENHANCED_FEATURE_DIM * 2 if self.include_temporal else ENHANCED_FEATURE_DIM
            dummy_features = torch.zeros(batch_size, expected_feat_dim)

            torch.onnx.export(
                self,
                (dummy_eeg, dummy_features),
                str(path),
                input_names=["raw_eeg", "handcrafted_features"],
                output_names=["logits"],
                dynamic_axes={
                    "raw_eeg": {0: "batch"},
                    "handcrafted_features": {0: "batch"},
                    "logits": {0: "batch"},
                },
                opset_version=17,
            )
            log.info("EnhancedEmotionClassifier ONNX exported -> %s", path)

else:
    # Stub
    class EnhancedEmotionClassifier:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            pass

        def predict_emotion(self, *args, **kwargs):
            raise RuntimeError("PyTorch required for EnhancedEmotionClassifier")
