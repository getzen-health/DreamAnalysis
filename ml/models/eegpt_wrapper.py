"""Wrapper for EEGPT foundation model for EEG classification.

EEGPT (NeurIPS 2024) is a 10M-parameter transformer pre-trained on a
large mixed EEG corpus. It achieves state-of-the-art on many EEG
benchmarks through linear probing or fine-tuning on downstream tasks.

This wrapper documents the integration path and provides stub functions
for emotion classification from 4-channel Muse 2 input.

Key challenges for Muse 2 integration:
    1. EEGPT is pre-trained on research-grade EEG (64+ channels, gel electrodes).
       Muse 2 has only 4 dry electrodes at non-standard positions.
    2. Fine-tuning on Muse-specific data is REQUIRED before useful predictions.
    3. The model expects fixed-length input windows (typically 4 seconds).
    4. Channel count mismatch requires either:
       a) Zero-padding unused channels (simple but suboptimal)
       b) Training a channel adapter layer (better)
       c) Re-training with variable channel support

Model details:
    - Parameters: ~10M (transformer encoder)
    - Pre-training: Self-supervised on large EEG corpus
    - Fine-tuning: Linear probe on downstream task
    - Input: (n_channels, n_samples) EEG array
    - Output: Task-dependent (emotion, sleep stage, attention, etc.)

Integration steps:
    1. Clone: git clone https://github.com/BINE022/EEGPT
    2. Download pre-trained weights from the repository
    3. Fine-tune on Muse 2 labeled data (minimum ~500 labeled samples)
    4. Export fine-tuned model to models/saved/eegpt_muse4ch.pt

Requirements:
    - PyTorch >= 2.0
    - EEGPT repository cloned and in PYTHONPATH
    - Pre-trained weights downloaded
    - Fine-tuned weights for Muse 2 (see fine-tuning docs below)

Reference:
    BINE022/EEGPT: https://github.com/BINE022/EEGPT
    NeurIPS 2024 — "EEGPT: Unleashing the Potential of EEG Generalist
    Foundation Model by Autoregressive Pre-training"
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# Expected input parameters
DEFAULT_FS = 256            # Muse 2 sampling rate
DEFAULT_EPOCH_SEC = 4.0     # 4-second input windows
MUSE_CHANNELS = 4           # TP9, AF7, AF8, TP10
EEGPT_EXPECTED_CHANNELS = 64  # What EEGPT was pre-trained on

# 6-class emotion labels matching the rest of the pipeline
EMOTIONS_6: List[str] = ["happy", "sad", "angry", "fear", "surprise", "neutral"]

# Muse 2 channel names in BrainFlow order
MUSE_CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]


class EEGPTWrapper:
    """Wrapper for EEGPT foundation model with Muse 2 adaptation.

    Uses lazy loading: model is not loaded until predict() is first called.
    Returns None when model is unavailable (not fine-tuned for Muse 2).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_classes: int = 6,
    ):
        """Initialize EEGPT wrapper.

        Args:
            model_path: Path to fine-tuned EEGPT weights for Muse 2.
                        If None, looks in models/saved/eegpt_muse4ch.pt.
            n_classes: Number of output classes (default 6 for emotions).
        """
        if model_path is None:
            self._model_path = (
                Path(__file__).resolve().parent / "saved" / "eegpt_muse4ch.pt"
            )
        else:
            self._model_path = Path(model_path)

        self._n_classes = n_classes
        self._model = None
        self._available: Optional[bool] = None
        self._load_attempted = False

    def _try_load(self):
        """Attempt to load fine-tuned EEGPT model. Lazy loading."""
        if self._load_attempted:
            return
        self._load_attempted = True

        if not self._model_path.exists():
            log.info(
                "EEGPT fine-tuned model not found at %s. "
                "Fine-tuning on Muse 2 data required before use. "
                "See docstring for integration steps.",
                self._model_path,
            )
            self._available = False
            return

        try:
            import torch  # type: ignore

            log.info("Loading EEGPT model from %s", self._model_path)
            self._model = torch.load(
                self._model_path,
                map_location="cpu",
                weights_only=False,
            )
            self._model.eval()
            self._available = True
            log.info("EEGPT model loaded successfully")
        except ImportError:
            log.info("PyTorch not installed — EEGPT unavailable")
            self._available = False
        except Exception as e:
            log.warning("EEGPT model failed to load: %s", e)
            self._available = False

    @property
    def available(self) -> bool:
        """Whether the model is loaded and ready for inference."""
        if self._available is None:
            self._try_load()
        return self._available or False

    def predict(
        self,
        signals: np.ndarray,
        fs: int = DEFAULT_FS,
    ) -> Optional[Dict]:
        """Predict emotion from 4-channel Muse 2 EEG.

        Args:
            signals: (4, n_samples) EEG array.
                     Channel order: [TP9, AF7, AF8, TP10].
            fs: Sampling rate (default 256 Hz for Muse 2).

        Returns:
            Dict with emotion, probabilities, valence, arousal, confidence,
            model_type. Returns None if model unavailable or input invalid.
        """
        if not self.available:
            return None

        # Validate input
        if signals is None:
            return None
        if signals.ndim == 1:
            log.warning("EEGPT requires multichannel input, got 1D")
            return None
        if signals.shape[0] < MUSE_CHANNELS:
            log.warning(
                "EEGPT expects %d channels, got %d",
                MUSE_CHANNELS, signals.shape[0],
            )
            return None

        min_samples = int(fs * 2)  # Minimum 2 seconds
        if signals.shape[1] < min_samples:
            log.warning(
                "EEGPT needs at least 2s of data (%d samples), got %d",
                min_samples, signals.shape[1],
            )
            return None

        try:
            return self._run_inference(signals[:MUSE_CHANNELS], fs)
        except Exception as e:
            log.warning("EEGPT inference failed: %s", e)
            return None

    def _run_inference(self, signals: np.ndarray, fs: int) -> Dict:
        """Run actual model inference.

        This method is only called when the model is loaded and input
        is validated. Override or extend for custom inference logic.

        Args:
            signals: (4, n_samples) validated EEG array.
            fs: Sampling rate.

        Returns:
            Standardized emotion dict.
        """
        import torch  # type: ignore

        # Prepare input tensor
        # EEGPT expects (batch, channels, samples) — pad to expected channels
        x = torch.tensor(signals, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()

        # Build 6-class probability dict
        probs_dict = {}
        for i, emotion in enumerate(EMOTIONS_6[:self._n_classes]):
            probs_dict[emotion] = float(probs[i]) if i < len(probs) else 0.0

        # Fill missing emotions with zero
        for e in EMOTIONS_6:
            if e not in probs_dict:
                probs_dict[e] = 0.0

        # Normalize
        total = sum(probs_dict.values())
        if total > 0:
            probs_dict = {k: v / total for k, v in probs_dict.items()}

        emotion = max(probs_dict, key=probs_dict.get)  # type: ignore
        confidence = probs_dict[emotion]

        # Valence and arousal from probability-weighted mapping
        valence_map = {
            "happy": 0.7, "sad": -0.6, "angry": -0.5,
            "fear": -0.4, "surprise": 0.2, "neutral": 0.0,
        }
        arousal_map = {
            "happy": 0.6, "sad": 0.2, "angry": 0.8,
            "fear": 0.7, "surprise": 0.7, "neutral": 0.3,
        }
        valence = sum(probs_dict[e] * valence_map[e] for e in EMOTIONS_6)
        arousal = sum(probs_dict[e] * arousal_map[e] for e in EMOTIONS_6)

        return {
            "emotion": emotion,
            "probabilities": probs_dict,
            "valence": float(np.clip(valence, -1.0, 1.0)),
            "arousal": float(np.clip(arousal, 0.0, 1.0)),
            "confidence": float(confidence),
            "model_type": "eegpt_foundation",
        }

    @staticmethod
    def get_model_info() -> Dict:
        """Return model metadata for status endpoints."""
        return {
            "name": "EEGPT",
            "source": "github.com/BINE022/EEGPT",
            "paper": "NeurIPS 2024",
            "parameters": "~10M",
            "input_channels": MUSE_CHANNELS,
            "pretrained_channels": EEGPT_EXPECTED_CHANNELS,
            "requires_finetuning": True,
            "finetuning_note": (
                "Pre-trained on 64-channel research EEG. Must be fine-tuned "
                "on Muse 2 (4-channel) labeled data before use. Minimum "
                "~500 labeled samples recommended."
            ),
            "requires": ["torch"],
        }

    @staticmethod
    def get_finetuning_guide() -> str:
        """Return step-by-step fine-tuning instructions."""
        return """
EEGPT Fine-tuning Guide for Muse 2
===================================

Prerequisites:
    - PyTorch >= 2.0
    - EEGPT repository: git clone https://github.com/BINE022/EEGPT
    - Pre-trained weights from the EEGPT repository
    - Labeled Muse 2 EEG data (minimum ~500 samples)

Steps:
    1. Clone EEGPT repository and download pre-trained weights
    2. Collect labeled EEG data using the NeuralDreamWorkshop app:
       - Use POST /collect-training-data endpoint during sessions
       - Aim for ~100 samples per emotion class
    3. Create channel adapter:
       - EEGPT expects 64 channels; Muse 2 has 4
       - Option A: Zero-pad to 64 channels (quick, lower accuracy)
       - Option B: Train a linear adapter (4 -> 64) jointly with probe
    4. Fine-tune with linear probing:
       - Freeze EEGPT encoder weights
       - Train only the classification head + channel adapter
       - Use Adam optimizer, lr=1e-4, 50 epochs
    5. Evaluate cross-validation accuracy:
       - Target: >60% cross-subject 6-class accuracy
       - If below 60%, collect more data or try full fine-tuning
    6. Export model:
       torch.save(model, 'ml/models/saved/eegpt_muse4ch.pt')

Expected accuracy:
    - Linear probe only: 55-65% (limited by channel count)
    - Full fine-tune: 60-70% (risk of overfitting with small data)
    - With personalization: 70-80%
"""
