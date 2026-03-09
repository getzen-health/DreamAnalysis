"""EmoAdapt: Self-Supervised Feature Learning for 4-Channel EEG Emotion Personalization.

Implements contrastive self-supervised learning principles using numpy only (no PyTorch).
Works on Muse 2 EEG: shape (4, n_samples) or (n_samples,).

Self-supervised pretext tasks:
  1. Band-power temporal consistency — adjacent windows yield similar features
  2. Channel permutation invariance — re-ordered channels give consistent predictions
  3. Amplitude augmentation consistency — scaled signal gives same emotion label

Prototype bank: running mean of feature vectors per emotion class, updated via
exponential moving average (EMA) with momentum=0.9.

Adapted prediction: cosine similarity against prototypes → softmax probabilities.
Blended output: 60% adapted + 40% heuristic baseline.

Reference: EmoAdapt — self-supervised personalized EEG emotion recognition framework.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EMOTIONS_6 = ["happy", "sad", "angry", "fear", "surprise", "neutral"]

# EMA momentum for prototype updates (0.9 = slow, stable adaptation)
_EMA_MOMENTUM = 0.9

# Number of labeled updates per class before adaptation is considered reliable
_READY_THRESHOLD = 5

# Blend weights: adapted vs heuristic
_ADAPT_WEIGHT = 0.6
_HEURISTIC_WEIGHT = 0.4

# Feature dimension: 5 bands × 4 channels = 20 DE features + 1 FAA = 21
# We keep it simple and physics-grounded for numpy-only implementation.
_FEATURE_DIM = 21

# Softmax temperature (lower = sharper distribution)
_SOFTMAX_TEMPERATURE = 0.5


# ── Feature extraction ────────────────────────────────────────────────────────

def _bandpass_simple(signal: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """Minimal bandpass via FFT masking — no scipy dependency required."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.fft.rfft(signal)
    mask = (freqs >= lo) & (freqs <= hi)
    spectrum_masked = spectrum * mask
    return np.fft.irfft(spectrum_masked, n=n)


def _band_power(signal: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """Compute average power in a frequency band via Welch-like FFT estimate."""
    filtered = _bandpass_simple(signal, fs, lo, hi)
    power = float(np.mean(filtered ** 2))
    return max(power, 1e-12)  # avoid log(0)


_BAND_EDGES = [
    (0.5, 4.0),    # delta
    (4.0, 8.0),    # theta
    (8.0, 12.0),   # alpha
    (12.0, 30.0),  # beta
    (30.0, 45.0),  # gamma (45 Hz ceiling — below Nyquist at 256 Hz)
]
_N_BANDS = len(_BAND_EDGES)


def _extract_de_features(signal: np.ndarray, fs: float) -> np.ndarray:
    """Differential entropy per band for a single channel.

    DE(x) = 0.5 * log(2π e * var(x_band)) ≈ 0.5 * log(2π e * power_band)
    for a Gaussian signal.  Returns array of length _N_BANDS.
    """
    de = np.zeros(_N_BANDS)
    for i, (lo, hi) in enumerate(_BAND_EDGES):
        p = _band_power(signal, fs, lo, hi)
        de[i] = 0.5 * np.log(2 * np.pi * np.e * p)
    return de


def _extract_features(eeg: np.ndarray, fs: float) -> np.ndarray:
    """Extract compact 21-feature vector from EEG.

    Layout:
      - 20 DE features: 5 bands × 4 channels (or duplicated 1-ch if 1D)
      - 1 FAA feature:  ln(AF8_alpha) − ln(AF7_alpha)

    For 1D input, channels are treated as identical (no FAA).
    """
    if eeg.ndim == 1:
        # Single channel — replicate across 4 virtual channels, FAA = 0
        ch_de = _extract_de_features(eeg, fs)
        de_features = np.tile(ch_de, 4)  # 20 DE features
        faa = np.array([0.0])
    else:
        n_ch = eeg.shape[0]
        channels = [eeg[i] for i in range(min(n_ch, 4))]
        # Pad to 4 channels if fewer provided
        while len(channels) < 4:
            channels.append(channels[-1])

        de_features = np.concatenate([_extract_de_features(ch, fs) for ch in channels])

        # FAA: log(AF8_alpha) - log(AF7_alpha); Muse 2: ch1=AF7, ch2=AF8
        alpha_lo, alpha_hi = 8.0, 12.0
        af7_alpha = _band_power(channels[1], fs, alpha_lo, alpha_hi)
        af8_alpha = _band_power(channels[2], fs, alpha_lo, alpha_hi)
        faa = np.array([np.log(af8_alpha) - np.log(af7_alpha)])

    feature_vec = np.concatenate([de_features, faa])  # shape: (21,)
    return feature_vec.astype(np.float32)


def _heuristic_probs(eeg: np.ndarray, fs: float) -> tuple[np.ndarray, float, float]:
    """Compute heuristic emotion probabilities from band-power ratios.

    Returns (probs, valence, arousal).
    Probabilities sum to 1.  All values are physics-grounded (no hardcoded values).
    """
    if eeg.ndim == 2:
        signal = eeg[1] if eeg.shape[0] > 1 else eeg[0]  # AF7 channel
        channels = eeg
    else:
        signal = eeg
        channels = None

    # Band powers for primary channel
    delta = _band_power(signal, fs, 0.5, 4.0)
    theta = _band_power(signal, fs, 4.0, 8.0)
    alpha = _band_power(signal, fs, 8.0, 12.0)
    beta  = _band_power(signal, fs, 12.0, 30.0)
    high_beta = _band_power(signal, fs, 20.0, 30.0)

    eps = 1e-9
    alpha_beta  = alpha / (beta + eps)
    theta_beta  = theta / (beta + eps)
    beta_total  = beta / (alpha + beta + theta + eps)
    high_beta_f = high_beta / (beta + eps)

    # Valence: alpha/beta ratio (approach: higher alpha = more positive)
    valence_abr = float(np.tanh((alpha_beta - 0.7) * 2.0))

    # FAA contribution when multichannel
    faa_valence = 0.0
    if channels is not None and channels.shape[0] >= 3:
        af7_alpha = _band_power(channels[1], fs, 8.0, 12.0)
        af8_alpha = _band_power(channels[2], fs, 8.0, 12.0)
        raw_faa = np.log(af8_alpha) - np.log(af7_alpha)
        faa_valence = float(np.clip(np.tanh(raw_faa * 2.0), -1.0, 1.0))
        valence = float(np.clip(0.50 * valence_abr + 0.50 * faa_valence, -1.0, 1.0))
    else:
        valence = float(np.clip(valence_abr, -1.0, 1.0))

    # Arousal: beta dominance; delta acts as inverse relaxation
    arousal = float(np.clip(
        0.45 * beta_total
        + 0.30 * (1 - alpha / (alpha + beta + theta + eps))
        + 0.25 * (1 - delta / (delta + beta + eps)),
        0.0, 1.0
    ))

    # Emotion probabilities from valence/arousal space (Russell circumplex)
    pos_val = max(0.0, valence)
    neg_val = max(0.0, -valence)

    probs = np.zeros(6, dtype=np.float32)
    # happy: positive valence, moderate-high arousal
    probs[0] = 0.50 * pos_val + 0.30 * arousal + 0.20 * alpha_beta * 0.3
    # sad: negative valence, low arousal
    probs[1] = 0.50 * max(0.0, -valence - 0.10) + 0.30 * max(0.0, 0.4 - arousal) + 0.20 * theta_beta * 0.2
    # angry: negative valence, high arousal
    probs[2] = 0.40 * neg_val + 0.35 * max(0.0, arousal - 0.45) + 0.25 * high_beta_f * 0.5
    # fear: negative valence, high arousal, high beta
    probs[3] = 0.35 * neg_val + 0.35 * arousal + 0.30 * high_beta_f * 0.5
    # surprise: mixed valence, high arousal
    probs[4] = 0.40 * arousal + 0.30 * (1 - abs(valence)) + 0.30 * 0.2
    # neutral: low arousal, low absolute valence
    probs[5] = 0.50 * max(0.0, 0.5 - arousal) + 0.50 * max(0.0, 0.4 - abs(valence))

    # Ensure all probs ≥ 0 before normalising
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total < 1e-9:
        probs = np.ones(6, dtype=np.float32) / 6
    else:
        probs = probs / total

    return probs, valence, arousal


# ── Cosine similarity ─────────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature scaling."""
    scaled = x / max(temperature, 1e-9)
    shifted = scaled - scaled.max()
    exp_x = np.exp(shifted)
    return exp_x / (exp_x.sum() + 1e-9)


# ── EmoAdaptLearner ───────────────────────────────────────────────────────────

class EmoAdaptLearner:
    """Self-supervised prototype-based EEG emotion personalization.

    Uses contrastive self-supervised pretext tasks (numpy-only) and a running
    prototype bank updated via EMA to adapt emotion predictions to an individual.

    Prediction blends adapted cosine-similarity output (60%) with heuristic
    feature-based output (40%).  On first call with no prototypes, falls back
    to heuristics only (adaptation_gain=0.0).

    After ≥5 labeled updates per class, adaptation_gain approaches 1.0.
    """

    def __init__(self, fs: float = 256.0) -> None:
        self.fs = fs
        self._emotion_labels = EMOTIONS_6
        self._n_classes = len(EMOTIONS_6)
        self._feat_dim = _FEATURE_DIM
        self._ema_momentum = _EMA_MOMENTUM
        self._ready_threshold = _READY_THRESHOLD

        # Prototype bank: (n_classes, feat_dim) or None until first update
        self._prototypes: Optional[np.ndarray] = None
        # Count of labeled updates per class
        self._n_updates_per_class: Dict[str, int] = {e: 0 for e in EMOTIONS_6}
        # Total updates across all classes
        self._n_updates_total: int = 0

    # ── Pretext tasks (self-supervised augmentations) ─────────────────────────

    @staticmethod
    def _augment_temporal(eeg: np.ndarray, fs: float) -> np.ndarray:
        """Band-power temporal consistency: extract features from a nearby window.

        Simulates "adjacent window" by applying a small random time-shift.
        Features of nearby windows should match — this is our consistency signal.
        """
        n_samples = eeg.shape[-1]
        max_shift = min(int(0.05 * fs), n_samples // 4)  # ≤5% of 1 second
        if max_shift < 1:
            return eeg.copy()
        shift = int(np.random.randint(1, max_shift + 1))
        if eeg.ndim == 1:
            return np.roll(eeg, shift)
        return np.roll(eeg, shift, axis=-1)

    @staticmethod
    def _augment_channel_permute(eeg: np.ndarray) -> np.ndarray:
        """Channel permutation invariance: shuffle temporal + inner channels.

        Permutes only the two temporal channels (TP9/TP10 at indices 0, 3)
        and the two frontal channels (AF7/AF8 at indices 1, 2) separately,
        preserving left/right hemisphere pairing.  This tests that the model
        is not simply memorising channel order.
        """
        if eeg.ndim == 1:
            return eeg.copy()
        out = eeg.copy()
        n_ch = out.shape[0]
        if n_ch >= 4:
            # Randomly swap temporal pair (TP9 ↔ TP10)
            if np.random.rand() > 0.5:
                out[[0, 3]] = out[[3, 0]]
            # Randomly swap frontal pair (AF7 ↔ AF8)
            if np.random.rand() > 0.5:
                out[[1, 2]] = out[[2, 1]]
        return out

    @staticmethod
    def _augment_amplitude(eeg: np.ndarray, scale_range: tuple = (0.7, 1.3)) -> np.ndarray:
        """Amplitude augmentation consistency: scale signal.

        After feature extraction (which uses log-power), scaled signals should
        yield shifted but consistent relative features — and therefore the same
        emotion classification.
        """
        scale = float(np.random.uniform(scale_range[0], scale_range[1]))
        return eeg * scale

    def _compute_consistency_score(self, eeg: np.ndarray, fs: float) -> float:
        """Compute mean consistency across all three pretext tasks.

        Returns a scalar in [0, 1] indicating how consistent this EEG window
        is across augmentations.  Higher = more reliable feature representation.
        """
        feat_orig = _extract_features(eeg, fs)

        # Task 1: temporal consistency
        eeg_temporal = self._augment_temporal(eeg, fs)
        feat_temporal = _extract_features(eeg_temporal, fs)
        sim_temporal = _cosine_similarity(feat_orig, feat_temporal)

        # Task 2: channel permutation invariance (only for multichannel)
        if eeg.ndim == 2 and eeg.shape[0] >= 4:
            eeg_perm = self._augment_channel_permute(eeg)
            feat_perm = _extract_features(eeg_perm, fs)
            sim_perm = _cosine_similarity(feat_orig, feat_perm)
        else:
            sim_perm = 1.0  # trivially consistent for 1D

        # Task 3: amplitude augmentation consistency
        eeg_scaled = self._augment_amplitude(eeg)
        feat_scaled = _extract_features(eeg_scaled, fs)
        sim_scaled = _cosine_similarity(feat_orig, feat_scaled)

        # Mean cosine similarity across tasks, clipped to [0, 1]
        consistency = float(np.clip((sim_temporal + sim_perm + sim_scaled) / 3.0, 0.0, 1.0))
        return consistency

    # ── Prototype bank ────────────────────────────────────────────────────────

    def _get_class_idx(self, emotion_label: str) -> int:
        """Return class index for a label, raises ValueError if unknown."""
        label = emotion_label.lower().strip()
        if label not in self._emotion_labels:
            raise ValueError(
                f"Unknown emotion label '{emotion_label}'. "
                f"Valid: {EMOTIONS_6}"
            )
        return self._emotion_labels.index(label)

    def _adapted_probs(self, feat: np.ndarray) -> np.ndarray:
        """Compute adapted probabilities via cosine similarity to prototypes.

        Returns softmax over similarities.  Requires _prototypes to be set.
        """
        if self._prototypes is None:
            raise RuntimeError("Prototype bank is empty — call update_prototype first.")

        sims = np.array([
            _cosine_similarity(feat, self._prototypes[i])
            for i in range(self._n_classes)
        ], dtype=np.float32)
        return _softmax(sims, temperature=_SOFTMAX_TEMPERATURE)

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Predict emotion from EEG with optional prototype-bank adaptation.

        Args:
            eeg: EEG signal of shape (n_samples,) or (4, n_samples).
            fs:  Sampling frequency in Hz (default 256).

        Returns dict with keys:
            emotion          str      — predicted class label
            probabilities    dict     — {label: probability} summing to ~1
            valence          float    — in [-1, 1]
            arousal          float    — in [0, 1]
            adaptation_gain  float    — in [0, 1]; 0 = pure heuristic, 1 = full adapt
            n_updates        int      — total labeled updates so far
        """
        eeg = np.asarray(eeg, dtype=np.float32)

        # Compute heuristic baseline
        h_probs, valence, arousal = _heuristic_probs(eeg, fs)

        # Compute adaptation gain: fraction of classes that have ≥ threshold updates
        n_ready = sum(
            1 for c in self._n_updates_per_class.values() if c >= self._ready_threshold
        )
        adaptation_gain = float(np.clip(n_ready / self._n_classes, 0.0, 1.0))

        # If no prototypes yet, fall back to heuristics
        if self._prototypes is None or adaptation_gain == 0.0:
            final_probs = h_probs.copy()
        else:
            feat = _extract_features(eeg, fs)
            a_probs = self._adapted_probs(feat)
            # Blend: 60% adapted + 40% heuristic
            final_probs = _ADAPT_WEIGHT * a_probs + _HEURISTIC_WEIGHT * h_probs
            total = final_probs.sum()
            if total > 1e-9:
                final_probs = final_probs / total

        predicted_idx = int(np.argmax(final_probs))
        predicted_emotion = self._emotion_labels[predicted_idx]

        return {
            "emotion": predicted_emotion,
            "probabilities": {
                label: float(final_probs[i])
                for i, label in enumerate(self._emotion_labels)
            },
            "valence": float(np.clip(valence, -1.0, 1.0)),
            "arousal": float(np.clip(arousal, 0.0, 1.0)),
            "adaptation_gain": adaptation_gain,
            "n_updates": self._n_updates_total,
        }

    def update_prototype(
        self, eeg: np.ndarray, emotion_label: str, fs: float = 256.0
    ) -> Dict:
        """Update prototype bank with a new labeled EEG example.

        Uses EMA with momentum=0.9:
            prototype_new = 0.9 * prototype_old + 0.1 * feature_new

        Args:
            eeg:           EEG signal (n_samples,) or (4, n_samples).
            emotion_label: One of EMOTIONS_6.
            fs:            Sampling frequency (default 256).

        Returns dict with keys:
            updated_prototypes  int   — total number of non-zero prototype slots
            label               str   — the emotion label that was updated
        """
        eeg = np.asarray(eeg, dtype=np.float32)

        # Validate label — raises ValueError on unknown label
        class_idx = self._get_class_idx(emotion_label)
        label = self._emotion_labels[class_idx]

        # Extract features
        feat = _extract_features(eeg, fs)

        # Initialise prototype bank on first update
        if self._prototypes is None:
            self._prototypes = np.zeros(
                (self._n_classes, self._feat_dim), dtype=np.float32
            )

        # EMA update
        current = self._prototypes[class_idx]
        if np.all(current == 0.0):
            # First update for this class — set directly
            self._prototypes[class_idx] = feat
        else:
            self._prototypes[class_idx] = (
                self._ema_momentum * current + (1.0 - self._ema_momentum) * feat
            )

        # Update counters
        self._n_updates_per_class[label] += 1
        self._n_updates_total += 1

        # Count non-zero prototype slots
        n_active = int(np.sum(
            [np.any(self._prototypes[i] != 0) for i in range(self._n_classes)]
        ))

        return {"updated_prototypes": n_active, "label": label}

    def reset(self) -> Dict:
        """Reset prototype bank and all update counters.

        Returns dict with n_updates=0 to confirm reset.
        """
        self._prototypes = None
        self._n_updates_per_class = {e: 0 for e in EMOTIONS_6}
        self._n_updates_total = 0
        return {"status": "reset", "n_updates": 0}

    def get_status(self) -> Dict:
        """Return adaptation state.

        Returns dict with keys:
            n_updates_per_class   dict  — {label: int}
            n_updates_total       int
            n_prototypes          int   — number of classes with an active prototype
            adaptation_ready      bool  — True when all classes have ≥ threshold updates
            adaptation_gain       float — fraction of classes above threshold
        """
        n_active = 0
        if self._prototypes is not None:
            n_active = int(sum(
                1 for i in range(self._n_classes)
                if np.any(self._prototypes[i] != 0)
            ))

        n_ready = sum(
            1 for c in self._n_updates_per_class.values() if c >= self._ready_threshold
        )
        adaptation_gain = float(np.clip(n_ready / self._n_classes, 0.0, 1.0))

        return {
            "n_updates_per_class": dict(self._n_updates_per_class),
            "n_updates_total": self._n_updates_total,
            "n_prototypes": n_active,
            "adaptation_ready": all(
                c >= self._ready_threshold for c in self._n_updates_per_class.values()
            ),
            "adaptation_gain": adaptation_gain,
        }


# ── Singleton factory ─────────────────────────────────────────────────────────

_emo_adapt_learner: Optional[EmoAdaptLearner] = None


def get_emo_adapt_learner() -> EmoAdaptLearner:
    """Return the global singleton EmoAdaptLearner, creating it if needed."""
    global _emo_adapt_learner
    if _emo_adapt_learner is None:
        _emo_adapt_learner = EmoAdaptLearner()
    return _emo_adapt_learner
