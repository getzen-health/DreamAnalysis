"""Personalized EEG Emotion Model — per-user fine-tuning on top of the central model.

Architecture: Two-tier system
──────────────────────────────────────────────────────────────────────────────
TIER 1 — Central model (EEGNet / mega LGBM):
  Trained on all available datasets (DEAP, SEED, Emognition, pilot data).
  Works for everyone immediately. Cross-person accuracy: 60–74%.
  Never modified by user-specific data.

TIER 2 — Personal model (per-user adapter):
  A lightweight adapter on top of the central model that learns THIS person's
  EEG patterns over time. Saved at:
    ml/models/saved/personal/{user_id}/adapter.pt
    ml/models/saved/personal/{user_id}/baseline.npz
    ml/models/saved/personal/{user_id}/meta.json

  Starts from central model weights. After each labeled session, fine-tunes
  only the classifier head (last layer) — fast, doesn't need much data.

  Accuracy progression (from literature):
    Session 1  (no personal data)  → central model accuracy: 60–74%
    Session 3  (~180 labeled epochs) → +8–12%  → ~72–82%
    Session 10 (~600 labeled epochs) → +15–20% → ~80–90%
    Session 30 (~1800 epochs)        → +20–26% → ~85–92%

Why only fine-tune the classifier head (not full network):
  - Prevents catastrophic forgetting of the central model's learned features
  - Needs far less data — converges in 50–200 samples
  - Preserves generalisation from the full training corpus
  - Fast: fine-tune in <5 seconds on CPU after each session

Data that feeds the personal model (everything the user has):
  - EEG sessions with self-reported labels (thumb up/down after a session)
  - Apple HealthKit context (HRV, sleep) stored alongside EEG epochs
  - Food log timestamps matched with EEG state
  - Biofeedback session outcomes (stress before/after)

How labels are collected without burdening the user:
  1. Implicit: if user opens biofeedback → system infers stress was high at that time
  2. Implicit: if user logs food after EEG → label that epoch as "hunger" or "post-meal"
  3. Explicit: single 3-button prompt after session: "How were you feeling? 😊 😐 😣"
  4. Continuous: user can correct any emotion label shown on the dashboard (1 tap)
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

log = logging.getLogger(__name__)

EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]
N_CLASSES = len(EMOTIONS)
_PERSONAL_DIR = Path(__file__).parent / "saved" / "personal"
_PERSONALIZATION_START_SESSIONS = 5
_GLOBAL_BASELINE_ACCURACY_PCT = 71.0
_PERSONAL_BLEND_WEIGHT = 0.70


# ── Baseline calibrator (session-level normalisation) ─────────────────────

class PersonalBaseline:
    """Stores per-user EEG feature baseline for z-score normalisation.

    Each user has very different absolute EEG amplitudes due to skull thickness,
    hair, electrode contact quality, and individual neurophysiology. Without
    normalising to the person's own resting baseline, cross-person models are
    essentially comparing apples and oranges.

    Usage:
        baseline = PersonalBaseline.load(user_id)   # or new if first session
        baseline.add_resting_frame(features)         # during 2-min eyes-closed
        features_normalised = baseline.normalise(features)   # during task
        baseline.save(user_id)
    """

    _MIN_FRAMES = 30    # ~30 seconds of resting data at 1 Hz feature extraction

    def __init__(self) -> None:
        self._frames: List[np.ndarray] = []
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_ready: bool = False
        self.n_frames_collected: int = 0

    def add_resting_frame(self, features: np.ndarray) -> None:
        """Add one feature vector from resting-state baseline recording."""
        self._frames.append(np.asarray(features, dtype=np.float32))
        self.n_frames_collected += 1
        if len(self._frames) >= self._MIN_FRAMES:
            self._fit()

    def _fit(self) -> None:
        X = np.stack(self._frames)
        self.mean = X.mean(axis=0)
        self.std  = np.where(X.std(axis=0) > 1e-6, X.std(axis=0), 1.0)
        self.is_ready = True

    def normalise(self, features: np.ndarray) -> np.ndarray:
        """Z-score features against personal resting baseline.
        Falls back to identity if baseline not ready."""
        if not self.is_ready:
            return features
        return (np.asarray(features, dtype=np.float32) - self.mean) / self.std

    def save(self, user_id: str) -> None:
        path = _PERSONAL_DIR / user_id
        path.mkdir(parents=True, exist_ok=True)
        if self.mean is not None:
            np.savez(path / "baseline.npz", mean=self.mean, std=self.std,
                     n_frames=np.array([self.n_frames_collected]))
            log.debug("Baseline saved for user %s (%d frames)", user_id, self.n_frames_collected)

    @classmethod
    def load(cls, user_id: str) -> "PersonalBaseline":
        b = cls()
        path = _PERSONAL_DIR / user_id / "baseline.npz"
        if path.exists():
            data = np.load(path)
            b.mean = data["mean"]
            b.std  = data["std"]
            b.n_frames_collected = int(data["n_frames"][0])
            b.is_ready = True
            log.debug("Baseline loaded for user %s", user_id)
        return b


# ── Classifier head adapter ───────────────────────────────────────────────

class _PersonalHead(nn.Module):
    """Lightweight adapter: 2-layer MLP that sits on top of EEGNet features.

    The central EEGNet's feature extractor is frozen. Only this head is
    trained on personal data. It takes the 512-dim feature vector from
    EEGNet's flatten layer and outputs per-class logits.

    Parameters:
        feat_dim: dimension of EEGNet's flattened feature vector (default 512)
        n_classes: number of emotion classes (6)
        hidden: hidden layer size (default 64 — small, avoids overfitting)
    """

    def __init__(self, feat_dim: int = 512, n_classes: int = N_CLASSES, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── PersonalModel — the full per-user system ──────────────────────────────

class PersonalModel:
    """Per-user fine-tuned emotion model.

    Wraps the central EEGNet + a personal classifier head.
    The EEGNet backbone is frozen; only the head is trained on user data.

    Usage::

        pm = PersonalModel.load_or_create(user_id="sravya", n_channels=4)

        # During a session — get prediction (uses personal model if ready,
        # central model otherwise):
        result = pm.predict(raw_eeg, fs=256)

        # After session — user corrects label or system infers label:
        pm.add_labeled_epoch(raw_eeg, label=4)   # 4 = "relaxed"

        # After session — fine-tune on accumulated personal data:
        pm.fine_tune()

        # Persist:
        pm.save()
    """

    _EMA_ALPHA = 0.35
    _MIN_SAMPLES_PER_CLASS_TO_ACTIVATE = 10   # legacy constant retained for compatibility
    _MIN_TOTAL_SAMPLES = _PERSONALIZATION_START_SESSIONS
    _FINE_TUNE_EPOCHS = 50
    _FINE_TUNE_LR = 5e-4
    _MAX_BUFFER = 2000                         # rolling window of most recent labeled epochs

    def __init__(self, user_id: str, n_channels: int = 4, feat_dim: int = 512):
        self.user_id = user_id
        self.n_channels = n_channels
        self.feat_dim = feat_dim

        self.baseline = PersonalBaseline.load(user_id)
        self.head = _PersonalHead(feat_dim=feat_dim)
        self.head_accuracy: float = 0.0        # last fine-tune val accuracy
        self.total_sessions: int = 0
        self.total_labeled_epochs: int = 0
        self.feature_priors: Dict[str, float] = {
            "alpha_mean": 0.0,
            "beta_mean": 0.0,
            "theta_mean": 0.0,
        }
        self._prior_count: int = 0

        # Rolling buffer of (raw_eeg, label) for continuous learning
        self._buffer_X: List[np.ndarray] = []  # list of (n_channels, n_samples) arrays
        self._buffer_y: List[int] = []

        # EMA for output smoothing
        self._ema_probs: Optional[np.ndarray] = None

        # Reference to frozen central EEGNet (loaded on first predict call)
        self._backbone: Optional[Any] = None   # EEGNet instance, features only

    # ── Public interface ──────────────────────────────────────────────────

    @classmethod
    def load_or_create(cls, user_id: str, n_channels: int = 4) -> "PersonalModel":
        """Load existing personal model or create a fresh one."""
        pm = cls(user_id=user_id, n_channels=n_channels)
        pm._load_head()
        pm._load_meta()
        return pm

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict[str, Any]:
        """Predict emotion using personal model (if ready) or central model.

        Returns same dict format as EmotionClassifier.predict().
        """
        self._ensure_backbone()

        if self._backbone is None:
            return {"emotion": "neutral", "model_type": "fallback_no_backbone",
                    "stress_index": 0.5, "valence": 0.0, "arousal": 0.5,
                    "focus_index": 0.5, "relaxation_index": 0.5}

        # Extract features from frozen backbone
        features = self._extract_features(eeg, fs)

        # Use personal head if it has enough data.
        # Central head is a stub that returns uniform distribution — signal fallback
        # so predict_emotion() routes to EmotionClassifier (mega LGBM) instead.
        if self._personal_head_ready():
            probs = self._personal_predict(features)
            model_tag = f"personal_{self.n_channels}ch"
        else:
            return {"emotion": "neutral", "model_type": "fallback_no_backbone",
                    "stress_index": 0.5, "valence": 0.0, "arousal": 0.5,
                    "focus_index": 0.5, "relaxation_index": 0.5}

        # EMA smoothing
        if self._ema_probs is None:
            self._ema_probs = probs
        else:
            self._ema_probs = self._EMA_ALPHA * probs + (1 - self._EMA_ALPHA) * self._ema_probs
        smoothed = self._ema_probs / (self._ema_probs.sum() + 1e-10)

        return self._probs_to_dict(smoothed, model_tag)

    def add_labeled_epoch(self, eeg: np.ndarray, label: int) -> None:
        """Add a labeled EEG epoch to the personal training buffer.

        Parameters
        ----------
        eeg   : (n_channels, n_samples) raw EEG
        label : integer 0–5 (index into EMOTIONS)
        """
        if label < 0 or label >= N_CLASSES:
            log.warning("Invalid label %d — must be 0–%d", label, N_CLASSES - 1)
            return
        self._buffer_X.append(eeg.astype(np.float32))
        self._buffer_y.append(int(label))
        self.total_labeled_epochs += 1
        self._update_feature_priors(eeg)

        # Trim buffer to rolling window
        if len(self._buffer_X) > self._MAX_BUFFER:
            self._buffer_X = self._buffer_X[-self._MAX_BUFFER:]
            self._buffer_y = self._buffer_y[-self._MAX_BUFFER:]

        log.debug("Personal buffer: %d labeled epochs for user %s", len(self._buffer_y), self.user_id)

    def fine_tune(self) -> float:
        """Fine-tune personal head on buffered labeled data. Returns val accuracy."""
        if len(self._buffer_y) < self._MIN_TOTAL_SAMPLES:
            log.info(
                "Not enough data for fine-tuning user %s (%d/%d epochs). "
                "Keep using app — model will improve automatically.",
                self.user_id, len(self._buffer_y), self._MIN_TOTAL_SAMPLES,
            )
            return 0.0

        self._ensure_backbone()
        if self._backbone is None:
            return 0.0

        # Extract features for all buffered epochs
        log.info("Fine-tuning personal model for %s (%d epochs)...", self.user_id, len(self._buffer_y))
        features_list = []
        for eeg in self._buffer_X:
            features_list.append(self._extract_features(eeg))
        X = np.stack(features_list)
        y = np.array(self._buffer_y, dtype=np.int64)

        # Split train/val
        n_val = max(1, int(len(y) * 0.2))
        idx = np.random.permutation(len(y))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_train = torch.from_numpy(X[train_idx])
        y_train = torch.from_numpy(y[train_idx])
        X_val   = torch.from_numpy(X[val_idx])
        y_val   = torch.from_numpy(y[val_idx])

        # Class-weighted loss
        counts = np.bincount(y, minlength=N_CLASSES).astype(np.float32)
        counts = np.where(counts == 0, 1, counts)
        weights = torch.tensor(1.0 / counts)

        optimizer = AdamW(self.head.parameters(), lr=self._FINE_TUNE_LR, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss(weight=weights)

        best_acc = 0.0
        best_state = None

        self.head.train()
        for epoch in range(self._FINE_TUNE_EPOCHS):
            # Mini-batch (entire buffer is small, use full batch)
            optimizer.zero_grad()
            loss = criterion(self.head(X_train), y_train)
            loss.backward()
            optimizer.step()

            # Validate
            self.head.eval()
            with torch.no_grad():
                preds = self.head(X_val).argmax(dim=1)
                acc = (preds == y_val).float().mean().item()
            self.head.train()

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in self.head.state_dict().items()}

        if best_state:
            self.head.load_state_dict(best_state)
        self.head.eval()
        self.head_accuracy = best_acc
        log.info("Fine-tune complete for %s — val_acc=%.2f%%", self.user_id, best_acc * 100)
        return best_acc

    def mark_session_complete(self) -> None:
        """Call at end of each session — fine-tunes if enough new data."""
        self.total_sessions += 1
        if self.total_sessions >= _PERSONALIZATION_START_SESSIONS:
            self.fine_tune()
        self.save()

    def save(self) -> None:
        path = _PERSONAL_DIR / self.user_id
        path.mkdir(parents=True, exist_ok=True)

        torch.save({
            "state_dict": self.head.state_dict(),
            "feat_dim":   self.feat_dim,
        }, path / "adapter.pt")

        # Save buffer as NPZ for next session
        if self._buffer_X:
            np.savez(
                path / "buffer.npz",
                X=np.stack(self._buffer_X),
                y=np.array(self._buffer_y),
            )

        meta = {
            "user_id":             self.user_id,
            "n_channels":          self.n_channels,
            "total_sessions":      self.total_sessions,
            "total_labeled_epochs": self.total_labeled_epochs,
            "head_accuracy":       self.head_accuracy,
            "personal_ready":      self._personal_head_ready(),
            "buffer_size":         len(self._buffer_y),
            "feature_priors":      self.feature_priors,
            "prior_count":         self._prior_count,
            "last_updated":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))
        self.baseline.save(self.user_id)
        log.debug("PersonalModel saved for %s", self.user_id)

    # ── Internals ─────────────────────────────────────────────────────────

    def _personal_head_ready(self) -> bool:
        """True when personal head has been fine-tuned with enough data."""
        return (
            self.total_sessions >= _PERSONALIZATION_START_SESSIONS
            and len(self._buffer_y) >= _PERSONALIZATION_START_SESSIONS
            and self.head_accuracy > 0.0
        )

    def record_feedback_session(self) -> None:
        """Count one user-corrected session toward personalization."""
        self.total_sessions += 1

    def blend_with_global(self, eeg: np.ndarray, global_result: Dict[str, Any], fs: float = 256.0) -> Dict[str, Any]:
        """Blend personal and global emotion probabilities with a 70/30 weighting."""
        if not self._personal_head_ready():
            result = dict(global_result)
            result["personal_model_active"] = False
            result["personal_blend_weight"] = 0.0
            return result

        personal_result = self.predict(eeg, fs=fs)
        personal_probs = personal_result.get("probabilities", {})
        global_probs = global_result.get("probabilities", {})
        if not personal_probs or not global_probs:
            result = dict(global_result)
            result["personal_model_active"] = False
            result["personal_blend_weight"] = 0.0
            return result

        blended_vec = np.array([
            _PERSONAL_BLEND_WEIGHT * float(personal_probs.get(emotion, 0.0))
            + (1.0 - _PERSONAL_BLEND_WEIGHT) * float(global_probs.get(emotion, 0.0))
            for emotion in EMOTIONS
        ], dtype=np.float32)
        blended_vec /= (blended_vec.sum() + 1e-10)

        blended = self._probs_to_dict(blended_vec, f"personal_blend_{self.n_channels}ch")
        blended["personal_model_active"] = True
        blended["personal_blend_weight"] = _PERSONAL_BLEND_WEIGHT
        blended["global_blend_weight"] = round(1.0 - _PERSONAL_BLEND_WEIGHT, 2)
        blended["head_accuracy_pct"] = round(self.head_accuracy * 100, 1)
        return blended

    def _update_feature_priors(self, eeg: np.ndarray) -> None:
        """Maintain running means for alpha/beta/theta features per user."""
        try:
            from processing.eeg_processor import extract_band_powers

            bands = extract_band_powers(eeg, fs=256.0)
            alpha = float(bands.get("alpha", 0.0))
            beta = float(bands.get("beta", 0.0))
            theta = float(bands.get("theta", 0.0))
        except Exception:
            return

        self._prior_count += 1
        weight = 1.0 / self._prior_count
        self.feature_priors["alpha_mean"] += (alpha - self.feature_priors["alpha_mean"]) * weight
        self.feature_priors["beta_mean"] += (beta - self.feature_priors["beta_mean"]) * weight
        self.feature_priors["theta_mean"] += (theta - self.feature_priors["theta_mean"]) * weight

    def _ensure_backbone(self) -> None:
        """Lazy-load the central EEGNet backbone (frozen)."""
        if self._backbone is not None:
            return
        try:
            from models.eegnet import EEGNet
            pt_path = Path(__file__).parent / "saved" / f"eegnet_emotion_{self.n_channels}ch.pt"
            if pt_path.exists():
                net = EEGNet.load(pt_path)
                net.eval()
                # Remove classifier head — we only use the feature extractor
                net.classifier = nn.Identity()
                self._backbone = net
                self.feat_dim = net._forward_features(
                    torch.zeros(1, 1, self.n_channels, net.n_samples)
                ).shape[1]
                # Rebuild head with correct feat_dim
                self.head = _PersonalHead(feat_dim=self.feat_dim)
                self._load_head()
                log.info("EEGNet backbone loaded for personal model (user=%s, %dch)", self.user_id, self.n_channels)
        except Exception as exc:
            log.warning("Could not load EEGNet backbone: %s", exc)

    def _extract_features(self, eeg: np.ndarray, fs: float = 256.0) -> np.ndarray:
        """Run EEGNet backbone to get feature vector."""
        if self._backbone is None:
            return np.zeros(self.feat_dim, dtype=np.float32)

        # Normalise
        eeg_norm = (eeg - eeg.mean(axis=1, keepdims=True)) / (eeg.std(axis=1, keepdims=True) + 1e-7)

        # Pad/trim to backbone's expected n_samples
        target = self._backbone.n_samples
        n_samples = eeg_norm.shape[1]
        if n_samples < target:
            eeg_norm = np.pad(eeg_norm, ((0, 0), (0, target - n_samples)), mode="edge")
        elif n_samples > target:
            eeg_norm = eeg_norm[:, :target]

        x = torch.from_numpy(eeg_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,ch,t)
        with torch.no_grad():
            feats = self._backbone._forward_features(x)
        return feats.squeeze(0).numpy()

    def _personal_predict(self, features: np.ndarray) -> np.ndarray:
        """Use personal head for prediction."""
        x = torch.from_numpy(features).unsqueeze(0)
        with torch.no_grad():
            logits = self.head(x)
        return F.softmax(logits, dim=-1).squeeze(0).numpy()

    def _central_predict(self, features: np.ndarray) -> np.ndarray:
        """Use central EEGNet's original classifier head for prediction."""
        if self._backbone is None:
            return np.ones(N_CLASSES, dtype=np.float32) / N_CLASSES
        # The backbone classifier was replaced with Identity — load original if needed
        # Fallback: uniform distribution (central model inference happens via EmotionClassifier)
        return np.ones(N_CLASSES, dtype=np.float32) / N_CLASSES

    def _load_head(self) -> None:
        pt_path = _PERSONAL_DIR / self.user_id / "adapter.pt"
        if pt_path.exists():
            try:
                ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
                feat_dim = ckpt.get("feat_dim", self.feat_dim)
                self.head = _PersonalHead(feat_dim=feat_dim)
                self.head.load_state_dict(ckpt["state_dict"])
                self.head.eval()
                log.info("Personal head loaded for user %s", self.user_id)
            except Exception as exc:
                log.warning("Could not load personal head for %s: %s", self.user_id, exc)

    def _load_meta(self) -> None:
        meta_path = _PERSONAL_DIR / self.user_id / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                self.total_sessions      = meta.get("total_sessions", 0)
                self.total_labeled_epochs = meta.get("total_labeled_epochs", 0)
                self.head_accuracy       = meta.get("head_accuracy", 0.0)
                self.feature_priors      = meta.get("feature_priors", self.feature_priors)
                self._prior_count        = meta.get("prior_count", 0)
            except Exception:
                pass

        # Reload buffer
        buf_path = _PERSONAL_DIR / self.user_id / "buffer.npz"
        if buf_path.exists():
            try:
                data = np.load(buf_path)
                self._buffer_X = list(data["X"])
                self._buffer_y = list(data["y"].astype(int))
                log.debug("Buffer reloaded for %s: %d epochs", self.user_id, len(self._buffer_y))
            except Exception:
                pass

    @staticmethod
    def _probs_to_dict(probs: np.ndarray, model_tag: str) -> Dict[str, Any]:
        emotion_idx = int(np.argmax(probs))
        valence = (
            probs[0] * 0.9 + probs[4] * 0.4 + probs[5] * 0.2
            - probs[1] * 0.9 - probs[2] * 0.7 - probs[3] * 0.8
        )
        arousal = float(np.clip(
            probs[0]*0.7 + probs[2]*0.9 + probs[3]*0.8 + probs[5]*0.7
            - probs[1]*0.5 - probs[4]*0.8 + 0.5, 0.0, 1.0
        ))
        return {
            "emotion":          EMOTIONS[emotion_idx],
            "probabilities":    {e: round(float(p), 4) for e, p in zip(EMOTIONS, probs)},
            "valence":          round(float(np.clip(valence, -1.0, 1.0)), 3),
            "arousal":          round(arousal, 3),
            "stress_index":     round(float(np.clip(probs[2]*0.85 + probs[3]*0.8 - probs[4]*0.7, 0, 1)), 3),
            "focus_index":      round(float(np.clip(probs[5]*0.9 + probs[0]*0.3, 0, 1)), 3),
            "relaxation_index": round(float(np.clip(probs[4]*0.9 + probs[1]*0.2, 0, 1)), 3),
            "model_type":       model_tag,
        }

    def status(self) -> Dict[str, Any]:
        """Return human-readable status for the dashboard."""
        counts = np.bincount(self._buffer_y, minlength=N_CLASSES) if self._buffer_y else np.zeros(N_CLASSES, int)
        progress_pct = min(100, round((self.total_sessions / _PERSONALIZATION_START_SESSIONS) * 100))
        head_accuracy_pct = round(self.head_accuracy * 100, 1)
        accuracy_improvement_pct = round(max(0.0, head_accuracy_pct - _GLOBAL_BASELINE_ACCURACY_PCT), 1)
        return {
            "user_id":              self.user_id,
            "personal_model_active": self._personal_head_ready(),
            "total_sessions":        self.total_sessions,
            "total_labeled_epochs":  self.total_labeled_epochs,
            "buffer_size":           len(self._buffer_y),
            "head_accuracy_pct":     head_accuracy_pct,
            "estimated_global_accuracy_pct": _GLOBAL_BASELINE_ACCURACY_PCT,
            "accuracy_improvement_pct": accuracy_improvement_pct,
            "personalization_progress_pct": progress_pct,
            "activation_threshold_sessions": _PERSONALIZATION_START_SESSIONS,
            "personal_blend_weight_pct": int(_PERSONAL_BLEND_WEIGHT * 100),
            "baseline_ready":        self.baseline.is_ready,
            "baseline_frames":       self.baseline.n_frames_collected,
            "feature_priors":        {k: round(v, 4) for k, v in self.feature_priors.items()},
            "class_counts":          {e: int(c) for e, c in zip(EMOTIONS, counts)},
            "next_milestone":        _next_milestone(self.total_sessions),
            "message":               _progress_message(self.total_sessions, self._personal_head_ready()),
        }


# ── Progress messaging ────────────────────────────────────────────────────

def _next_milestone(n: int) -> int:
    for m in [5, 10, 20, 30, 50, 100]:
        if n < m:
            return m
    return n + 10


def _progress_message(n: int, active: bool) -> str:
    if active:
        return f"Personal model active — blended 70/30 with the global model. {n} corrected sessions recorded."
    if n == 0:
        return "Correct 5 sessions to activate your personal model."
    return f"{n}/5 corrected sessions collected. Keep correcting labels to activate personalization."


# ── Global per-user registry ──────────────────────────────────────────────

_registry: Dict[str, PersonalModel] = {}


def get_personal_model(user_id: str, n_channels: int = 4) -> PersonalModel:
    """Get or create the PersonalModel for a given user. Thread-safe for single-process."""
    key = f"{user_id}_{n_channels}"
    if key not in _registry:
        _registry[key] = PersonalModel.load_or_create(user_id, n_channels)
    return _registry[key]
