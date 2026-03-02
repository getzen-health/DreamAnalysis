"""REVE Foundation Model — brain-bzh/reve-base via braindecode (NeurIPS 2025).

Architecture: 69.7M transformer, 4D positional encoding, patch_size=200 (1 sec at 200Hz).
Pretrained on 60,000+ hours of EEG from 92 datasets / 25,000 subjects.

States:
    PRETRAINED   — full pretrained weights from brain-bzh/reve-base (best accuracy)
    RANDOM_INIT  — architecture only (random weights), ready to train from scratch
    NO_BRAINDECODE — braindecode not installed

Access to pretrained weights:
    1. Visit https://huggingface.co/brain-bzh/reve-base
    2. Click "Request access" and fill the form
    3. Set HF_TOKEN env var with a read token that has approved access
    4. Restart the backend — PRETRAINED state activates automatically

Without pretrained weights:
    The wrapper uses RANDOM_INIT state.  Run train_reve_braindecode.py to fine-tune
    the 69.7M architecture on FACED + DREAMER data.  Saved to:
        models/saved/reve_braindecode_4ch.pt

Muse 2 channels (all confirmed in reve-positions bank):
    ch0=TP9  [-0.0856, -0.0465, -0.0457]
    ch1=AF7  [-0.0548,  0.0686, -0.0106]
    ch2=AF8  [ 0.0557,  0.0697, -0.0108]
    ch3=TP10 [ 0.0862, -0.0470, -0.0459]

Resampling: Muse 2 → 256Hz, REVE expects → 200Hz. scipy.signal.resample applied.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent          # ml/models/
_ML_ROOT = _HERE.parent                           # ml/
_WEIGHTS_PATH = _ML_ROOT / "models" / "saved" / "reve_braindecode_4ch.pt"
_HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Muse 2 channel order from BrainFlow (board 38): TP9, AF7, AF8, TP10
_MUSE2_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

# Confirmed 3D positions from brain-bzh/reve-positions (543-position bank)
_MUSE2_POSITIONS = [
    [-0.08562, -0.04651, -0.04571],   # TP9
    [-0.05484,  0.06857, -0.01059],   # AF7
    [ 0.05574,  0.06966, -0.01075],   # AF8
    [ 0.08616, -0.04704, -0.04587],   # TP10
]

_REVE_FS = 200        # REVE expects 200 Hz
_MUSE_FS = 256        # Muse 2 delivers 256 Hz
_REVE_SECS = 30       # 30-second input window
_N_EMOTIONS = 6
_EMOTION_LABELS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]


class REVEFoundationWrapper:
    """Wraps braindecode REVE for Muse 2 emotion classification.

    Tries to load pretrained weights from brain-bzh/reve-base (gated HF).
    Falls back to fine-tuned weights saved locally.
    Falls back to random initialization (trainable from scratch).
    """

    def __init__(self) -> None:
        self._state = "uninitialized"
        self._model = None             # braindecode REVE instance
        self._pos_tensor = None        # (1, 4, 3) Muse 2 positions

        self._load()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True when model is ready for inference (any init state except NO_BRAINDECODE)."""
        return self._model is not None and self._state != "NO_BRAINDECODE"

    def is_pretrained(self) -> bool:
        """True only when brain-bzh/reve-base weights are loaded."""
        return self._state == "PRETRAINED"

    def status(self) -> str:
        """Current state: PRETRAINED | FINETUNED | RANDOM_INIT | NO_BRAINDECODE | ACCESS_DENIED."""
        return self._state

    def predict(self, eeg: np.ndarray, fs: int = 256) -> Dict:
        """Run REVE inference on 4-channel Muse 2 EEG.

        Args:
            eeg: (4, n_samples) float array, µV, channels=[TP9, AF7, AF8, TP10]
            fs:  sampling rate (256 for Muse 2)

        Returns:
            Standard emotion dict with keys matching the other classifiers.
        """
        if not self.is_available():
            raise RuntimeError(f"REVE foundation not available (state={self._state})")

        import torch
        from scipy.signal import resample as sp_resample

        eeg = np.asarray(eeg, dtype=np.float32)

        # Resample 256 Hz → 200 Hz
        if fs != _REVE_FS:
            n_out = int(eeg.shape[1] * _REVE_FS / fs)
            eeg = sp_resample(eeg, n_out, axis=1).astype(np.float32)

        # Pad or trim to exactly 30 seconds (6000 samples at 200Hz)
        n_target = _REVE_FS * _REVE_SECS
        if eeg.shape[1] < n_target:
            pad = np.zeros((4, n_target - eeg.shape[1]), dtype=np.float32)
            eeg = np.concatenate([eeg, pad], axis=1)
        elif eeg.shape[1] > n_target:
            eeg = eeg[:, -n_target:]   # keep the most recent 30 seconds

        eeg_t = torch.from_numpy(eeg).unsqueeze(0)   # (1, 4, 6000)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(eeg_t, pos=self._pos_tensor)   # (1, 6)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

        return self._build_result(eeg, probs)

    def embed(self, eeg: np.ndarray, fs: int = 256) -> np.ndarray:
        """Return transformer embeddings (for downstream research/fine-tuning).

        Returns:
            List of layer-wise tensors (return_output=True mode).
        """
        if not self.is_available():
            raise RuntimeError(f"REVE not available (state={self._state})")

        import torch
        from scipy.signal import resample as sp_resample

        eeg = np.asarray(eeg, dtype=np.float32)
        if fs != _REVE_FS:
            n_out = int(eeg.shape[1] * _REVE_FS / fs)
            eeg = sp_resample(eeg, n_out, axis=1).astype(np.float32)

        n_target = _REVE_FS * _REVE_SECS
        if eeg.shape[1] < n_target:
            eeg = np.pad(eeg, ((0, 0), (0, n_target - eeg.shape[1])))
        else:
            eeg = eeg[:, -n_target:]

        eeg_t = torch.from_numpy(eeg).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            layers = self._model(eeg_t, pos=self._pos_tensor, return_output=True)

        # Return last-layer mean pooled embedding
        last = layers[-1]   # (1, n_patches, embed_dim)
        return last.mean(dim=1).squeeze(0).cpu().numpy()

    def request_access_instructions(self) -> str:
        return (
            "REVE pretrained weights require HuggingFace approval.\n"
            "\n"
            "Steps:\n"
            "  1. Log in at https://huggingface.co\n"
            "  2. Visit https://huggingface.co/brain-bzh/reve-base\n"
            "  3. Click 'Request access' — fill name/institution/research purpose\n"
            "  4. Once approved (typically 1-3 days), set in .env:\n"
            "        HF_TOKEN=hf_your_token_here\n"
            "  5. Restart the ML backend — PRETRAINED state activates automatically\n"
            "\n"
            f"Current state: {self._state}\n"
            "Fine-tuned weights: "
            + ("exist at " + str(_WEIGHTS_PATH) if _WEIGHTS_PATH.exists()
               else "not yet trained (run train_reve_braindecode.py)")
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            from braindecode.models import REVE
        except ImportError:
            self._state = "NO_BRAINDECODE"
            log.warning("[reve_foundation] braindecode not installed — run: pip install braindecode")
            return

        import torch

        # Store Muse 2 electrode positions as a cached tensor
        self._pos_tensor = torch.tensor(
            [_MUSE2_POSITIONS], dtype=torch.float32
        )  # (1, 4, 3)

        # Attempt 1: load pretrained weights from brain-bzh/reve-base
        if _HF_TOKEN:
            try:
                self._model = REVE.from_pretrained(
                    "brain-bzh/reve-base",
                    token=_HF_TOKEN,
                    n_chans=4,
                    sfreq=_REVE_FS,
                    input_window_seconds=_REVE_SECS,
                    n_outputs=_N_EMOTIONS,
                )
                self._model.eval()
                self._state = "PRETRAINED"
                n_params = sum(p.numel() for p in self._model.parameters())
                log.info(
                    "[reve_foundation] brain-bzh/reve-base loaded! state=PRETRAINED params=%dM",
                    n_params // 1_000_000,
                )
                return
            except Exception as exc:
                err = str(exc)
                if "403" in err or "GatedRepo" in type(exc).__name__:
                    log.info(
                        "[reve_foundation] reve-base access not yet approved "
                        "(visit huggingface.co/brain-bzh/reve-base to request)"
                    )
                    self._state = "ACCESS_DENIED"
                else:
                    log.warning("[reve_foundation] unexpected load error: %s", exc)
                    self._state = "LOAD_ERROR"

        # Attempt 2: load our fine-tuned weights (trained on FACED data)
        model = REVE(
            n_chans=4,
            sfreq=_REVE_FS,
            input_window_seconds=_REVE_SECS,
            n_outputs=_N_EMOTIONS,
        )

        if _WEIGHTS_PATH.exists():
            try:
                state = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
                model.eval()
                self._model = model
                self._state = "FINETUNED"
                log.info("[reve_foundation] fine-tuned REVE loaded from %s", _WEIGHTS_PATH)
                return
            except Exception as exc:
                log.warning("[reve_foundation] fine-tuned weights failed to load: %s", exc)

        # Attempt 3: random initialization (model is trainable, just not useful yet)
        model.eval()
        self._model = model
        self._state = "RANDOM_INIT"
        n_params = sum(p.numel() for p in model.parameters())
        log.info(
            "[reve_foundation] REVE initialized with random weights (%dM params). "
            "Run train_reve_braindecode.py to train. "
            "Or request HF access to load pretrained weights.",
            n_params // 1_000_000,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Result formatting
    # ─────────────────────────────────────────────────────────────────────────

    def _build_result(self, eeg_200hz: np.ndarray, probs: np.ndarray) -> Dict:
        """Build the standard emotion output dict."""
        from processing.eeg_processor import extract_band_powers

        prob_dict = {label: float(probs[i]) for i, label in enumerate(_EMOTION_LABELS)}
        emotion = _EMOTION_LABELS[int(np.argmax(probs))]
        confidence = float(np.max(probs))

        try:
            # BrainFlow Muse 2 channel order: ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10
            # Resample back to 256 Hz for band-power computation (eeg_processor expects 256Hz)
            from scipy.signal import resample as sp_resample
            eeg_256 = sp_resample(eeg_200hz, int(eeg_200hz.shape[1] * _MUSE_FS / _REVE_FS), axis=1)

            bp_af7 = extract_band_powers(eeg_256[1], _MUSE_FS)  # AF7 = ch1
            bp_af8 = extract_band_powers(eeg_256[2], _MUSE_FS)  # AF8 = ch2

            alpha_l = bp_af7.get("alpha", 0.1)
            alpha_r = bp_af8.get("alpha", 0.1)
            beta_l  = bp_af7.get("beta",  0.1)
            beta_r  = bp_af8.get("beta",  0.1)
            theta   = bp_af7.get("theta", 0.1)

            faa = float(np.log(max(alpha_r, 1e-10)) - np.log(max(alpha_l, 1e-10)))
            alpha = (alpha_l + alpha_r) / 2
            beta  = (beta_l  + beta_r)  / 2

            valence_abr = float(np.tanh((alpha / max(beta, 1e-10) - 0.7) * 2.0))
            faa_val     = float(np.clip(np.tanh(faa * 2.0), -1, 1))
            valence     = float(np.clip(0.5 * valence_abr + 0.5 * faa_val, -1, 1))
            arousal     = float(np.clip(
                0.45 * beta / max(alpha + beta, 1e-10)
                + 0.30 * (1 - alpha / max(alpha + beta + theta, 1e-10))
                + 0.25 * confidence,
                0, 1
            ))

        except Exception:
            valence = 0.0
            arousal = 0.5
            faa = 0.0

        stress      = float(np.clip(arousal * 0.5 + max(0, -valence) * 0.5, 0, 1))
        focus       = float(np.clip(prob_dict.get("focused", 0) + arousal * 0.3, 0, 1))
        relaxation  = float(np.clip(prob_dict.get("relaxed", 0) + max(0, valence) * 0.3, 0, 1))

        return {
            "emotion":            emotion,
            "confidence":         round(confidence, 3),
            "probabilities":      {k: round(v, 4) for k, v in prob_dict.items()},
            "valence":            round(valence, 3),
            "arousal":            round(arousal, 3),
            "stress_index":       round(stress, 3),
            "focus_index":        round(focus, 3),
            "relaxation_index":   round(relaxation, 3),
            "anger_index":        round(prob_dict.get("angry", 0), 3),
            "fear_index":         round(prob_dict.get("fearful", 0), 3),
            "frontal_asymmetry":  round(faa, 4),
            "model_type":         f"reve-{self._state.lower()}",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Fine-tuning helpers
    # ─────────────────────────────────────────────────────────────────────────

    def fine_tune(
        self,
        eeg_list: list,
        label_list: list,
        fs: int = 256,
        epochs: int = 20,
        lr: float = 1e-4,
        freeze_backbone: bool = True,
    ) -> float:
        """Fine-tune REVE on labeled Muse 2 data.

        Args:
            eeg_list:        list of (4, n_samples) arrays
            label_list:      list of int emotion indices (0-5)
            fs:              sampling rate
            epochs:          training epochs
            lr:              learning rate
            freeze_backbone: if True, only trains the final classification head

        Returns:
            Final training accuracy (0-1).
        """
        if not self.is_available():
            raise RuntimeError("REVE must be loaded before fine-tuning")

        import torch
        import torch.nn as nn
        from scipy.signal import resample as sp_resample

        log.info("[reve_foundation] fine-tuning on %d samples, freeze_backbone=%s",
                 len(eeg_list), freeze_backbone)

        # Freeze/unfreeze backbone
        if freeze_backbone:
            # Only train the final classification layer
            for name, param in self._model.named_parameters():
                param.requires_grad = "fc" in name or "head" in name or "classifier" in name
        else:
            for param in self._model.parameters():
                param.requires_grad = True

        optimizer = torch.optim.AdamW(
            [p for p in self._model.parameters() if p.requires_grad],
            lr=lr, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        # Preprocess all samples
        eegs, labels = [], []
        n_target = _REVE_FS * _REVE_SECS
        for eeg, lbl in zip(eeg_list, label_list):
            eeg = np.asarray(eeg, dtype=np.float32)
            if fs != _REVE_FS:
                n_out = int(eeg.shape[1] * _REVE_FS / fs)
                eeg = sp_resample(eeg, n_out, axis=1).astype(np.float32)
            if eeg.shape[1] < n_target:
                eeg = np.pad(eeg, ((0, 0), (0, n_target - eeg.shape[1])))
            else:
                eeg = eeg[:, -n_target:]
            eegs.append(eeg)
            labels.append(int(lbl))

        X = torch.from_numpy(np.stack(eegs)).float()     # (N, 4, 6000)
        y = torch.tensor(labels, dtype=torch.long)
        pos = self._pos_tensor.expand(len(eegs), -1, -1)  # (N, 4, 3)

        self._model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self._model(X, pos=pos)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        self._model.eval()

        with torch.no_grad():
            logits = self._model(X, pos=pos)
            acc = float((logits.argmax(1) == y).float().mean())

        # Save weights
        _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), _WEIGHTS_PATH)
        prev_state = self._state
        self._state = "FINETUNED"
        log.info("[reve_foundation] fine-tuned weights saved to %s (acc=%.1f%%, was %s)",
                 _WEIGHTS_PATH, acc * 100, prev_state)
        return acc


# ─── Module-level singleton ──────────────────────────────────────────────────

_instance: Optional[REVEFoundationWrapper] = None


def get_reve_foundation() -> REVEFoundationWrapper:
    """Return the module-level singleton, loading it on first call."""
    global _instance
    if _instance is None:
        _instance = REVEFoundationWrapper()
    return _instance
