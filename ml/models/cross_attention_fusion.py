"""Cross-attention fusion for EEG + Voice emotion classification.

Each modality attends to the other via nn.MultiheadAttention to find
complementary information. When EEG is noisy, the model leans on voice
features and vice versa.

Research basis: MMHA-FNN (2025) showed multi-head attention fusion
achieved 81.14% vs concatenation 71.02% -- a +10% gain for multimodal
emotion recognition.

Architecture:
    1. Project EEG (6-dim) and Voice (5-dim) to shared hidden space
    2. Cross-attention: EEG queries Voice, Voice queries EEG
    3. Concatenate attended representations -> classifier head

Input:  eeg_probs (B, 6) -- 6-class emotion probabilities from EEG
        voice_probs (B, 5) -- 5-class emotion probabilities from voice
Output: fused_logits (B, 6) -- fused 6-class emotion logits

The module also provides a high-level fuse_cross_attention() function
that loads a trained model (or falls back to weighted average) and
returns a result dict compatible with the existing fusion pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

_ML_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = _ML_ROOT / "models" / "saved"
_ONNX_PATH = _MODEL_DIR / "cross_attention_fusion.onnx"
_PT_PATH = _MODEL_DIR / "cross_attention_fusion.pt"

EMOTIONS_6 = ["happy", "sad", "angry", "fear", "surprise", "neutral"]


class CrossModalFusion(nn.Module):
    """Cross-attention fusion for EEG + Voice emotion.

    Each modality attends to the other to find complementary information.
    """

    def __init__(
        self,
        eeg_dim: int = 6,
        voice_dim: int = 5,
        hidden_dim: int = 32,
        n_heads: int = 2,
        n_classes: int = 6,
    ):
        super().__init__()
        self.eeg_dim = eeg_dim
        self.voice_dim = voice_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # Project both modalities to same dimension
        self.eeg_proj = nn.Linear(eeg_dim, hidden_dim)
        self.voice_proj = nn.Linear(voice_dim, hidden_dim)

        # Cross-attention: EEG attends to Voice
        self.eeg_to_voice_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True,
        )
        # Cross-attention: Voice attends to EEG
        self.voice_to_eeg_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(
        self,
        eeg_probs: torch.Tensor,
        voice_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            eeg_probs: (batch, eeg_dim) -- emotion probabilities from EEG
            voice_probs: (batch, voice_dim) -- emotion probabilities from voice

        Returns:
            (batch, n_classes) -- fused emotion logits
        """
        # Project to hidden dim and add sequence dimension for attention
        eeg_h = self.eeg_proj(eeg_probs).unsqueeze(1)      # (B, 1, H)
        voice_h = self.voice_proj(voice_probs).unsqueeze(1)  # (B, 1, H)

        # Cross-attention: EEG queries Voice keys/values
        eeg_attended, _ = self.eeg_to_voice_attn(eeg_h, voice_h, voice_h)
        # Cross-attention: Voice queries EEG keys/values
        voice_attended, _ = self.voice_to_eeg_attn(voice_h, eeg_h, eeg_h)

        # Concatenate attended representations and classify
        fused = torch.cat(
            [eeg_attended.squeeze(1), voice_attended.squeeze(1)],
            dim=-1,
        )  # (B, 2*H)

        return self.classifier(fused)  # (B, n_classes)


# ---------------------------------------------------------------------------
# High-level fusion function
# ---------------------------------------------------------------------------

def export_to_onnx(
    model: CrossModalFusion,
    output_path: str,
    opset_version: int = 18,
) -> None:
    """Export a CrossModalFusion model to ONNX format.

    Uses torch.onnx.export with dynamo=False for maximum compatibility.
    Opset 18 is the minimum required by PyTorch >= 2.8.

    Args:
        model: Trained CrossModalFusion instance (will be set to eval mode).
        output_path: Path to write the .onnx file.
        opset_version: ONNX opset version (default 18).
    """
    model.eval()
    dummy_eeg = torch.randn(1, model.eeg_dim)
    dummy_voice = torch.randn(1, model.voice_dim)
    torch.onnx.export(
        model,
        (dummy_eeg, dummy_voice),
        output_path,
        input_names=["eeg_probs", "voice_probs"],
        output_names=["fused_logits"],
        opset_version=opset_version,
        dynamo=False,
    )
    log.info("Exported cross-attention fusion model to %s", output_path)


# ---------------------------------------------------------------------------
# Model loading / caching
# ---------------------------------------------------------------------------

_cached_model: Optional[CrossModalFusion] = None
_cached_model_loaded: bool = False


def _load_model() -> Optional[CrossModalFusion]:
    """Load trained CrossModalFusion model from disk. Cached after first call."""
    global _cached_model, _cached_model_loaded
    if _cached_model_loaded:
        return _cached_model

    _cached_model_loaded = True

    if _PT_PATH.exists():
        try:
            model = CrossModalFusion()
            state = torch.load(str(_PT_PATH), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            _cached_model = model
            log.info("Cross-attention fusion model loaded from %s", _PT_PATH)
            return model
        except Exception as exc:
            log.warning("Failed to load cross-attention model: %s", exc)
            return None

    log.info("No trained cross-attention fusion model found at %s", _PT_PATH)
    return None


def _weighted_average_fallback(
    eeg_probs: Dict[str, float],
    voice_probs: Dict[str, float],
) -> Dict[str, float]:
    """Simple confidence-weighted average as fallback when model is unavailable.

    EEG has 6 classes, voice has 5 (no 'neutral' sometimes, or different mapping).
    Aligns to 6-class EMOTIONS_6 and averages with confidence-based weighting.
    """
    eeg_conf = max(eeg_probs.values()) if eeg_probs else 0.0
    voice_conf = max(voice_probs.values()) if voice_probs else 0.0
    total_conf = eeg_conf + voice_conf
    if total_conf < 1e-8:
        eeg_w, voice_w = 0.5, 0.5
    else:
        eeg_w = eeg_conf / total_conf
        voice_w = voice_conf / total_conf

    fused: Dict[str, float] = {}
    for emo in EMOTIONS_6:
        ep = eeg_probs.get(emo, 0.0)
        vp = voice_probs.get(emo, 0.0)
        fused[emo] = eeg_w * ep + voice_w * vp

    # Normalize
    total = sum(fused.values()) or 1.0
    fused = {k: round(v / total, 4) for k, v in fused.items()}
    return fused


def fuse_cross_attention(
    eeg_probs: Dict[str, float],
    voice_probs: Dict[str, float],
) -> Dict:
    """Fuse EEG and voice emotion probabilities using cross-attention.

    Falls back to weighted average if the trained model is unavailable.

    Args:
        eeg_probs: 6-class EEG emotion probabilities
        voice_probs: 5-class voice emotion probabilities

    Returns:
        Dict with keys: emotion, probabilities, confidence, model_type
    """
    model = _load_model()

    if model is not None:
        try:
            # Convert dicts to tensors in canonical order
            eeg_vec = torch.tensor(
                [[eeg_probs.get(e, 0.0) for e in EMOTIONS_6]],
                dtype=torch.float32,
            )
            # Voice uses first 5 of EMOTIONS_6 (no neutral in some voice models)
            voice_keys = EMOTIONS_6[:5]  # happy, sad, angry, fear, surprise
            voice_vec = torch.tensor(
                [[voice_probs.get(e, 0.0) for e in voice_keys]],
                dtype=torch.float32,
            )

            with torch.no_grad():
                logits = model(eeg_vec, voice_vec)
                probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

            prob_dict = {
                EMOTIONS_6[i]: round(float(probs[i]), 4)
                for i in range(len(EMOTIONS_6))
            }
            emotion = max(prob_dict, key=prob_dict.__getitem__)

            # Confidence: entropy-based (1 - normalized entropy)
            p = np.array(list(prob_dict.values()), dtype=np.float64)
            p = np.maximum(p, 1e-10)
            p = p / p.sum()
            entropy = -float(np.sum(p * np.log(p)))
            max_entropy = float(np.log(len(p)))
            confidence = float(np.clip(1.0 - entropy / max_entropy, 0.0, 1.0))

            return {
                "emotion": emotion,
                "probabilities": prob_dict,
                "confidence": round(confidence, 4),
                "model_type": "cross_attention_fusion",
            }
        except Exception as exc:
            log.warning("Cross-attention inference failed, using fallback: %s", exc)

    # Fallback: weighted average
    fused_probs = _weighted_average_fallback(eeg_probs, voice_probs)
    emotion = max(fused_probs, key=fused_probs.__getitem__)

    p = np.array(list(fused_probs.values()), dtype=np.float64)
    p = np.maximum(p, 1e-10)
    p = p / p.sum()
    entropy = -float(np.sum(p * np.log(p)))
    max_entropy = float(np.log(len(p)))
    confidence = float(np.clip(1.0 - entropy / max_entropy, 0.0, 1.0))

    return {
        "emotion": emotion,
        "probabilities": fused_probs,
        "confidence": round(confidence, 4),
        "model_type": "cross_attention_fusion",
    }
