"""CNN-LSTM Emotion Classifier from EEG signals.

Classifies EEG into 6 emotions: happy, sad, angry, fearful, relaxed, focused.
Also outputs valence (-1 to 1) and arousal (0 to 1) scores.

Uses neuroscience-backed feature-based classification with exponential
smoothing to prevent rapid state flickering. ONNX/sklearn models are
only used when their benchmark accuracy exceeds 60%.
"""

import logging
import numpy as np
from typing import Dict, Optional
from collections import deque
from processing.eeg_processor import (
    extract_band_powers, differential_entropy, extract_features, preprocess,
    compute_frontal_asymmetry, compute_dasm_rasm,
    compute_frontal_midline_theta, compute_coherence,
    compute_pairwise_plv,
    compute_band_hjorth_mobility, compute_hjorth_mobility_ratio,
    EuclideanAligner,
)
from processing.covariate_shift_detector import CovariateShiftDetector
from processing.channel_maps import get_channel_map
from models.emotion_granularity import map_vad_to_granular_emotions

# Pre-initialize PyTorch thread pool before any joblib/sklearn loading.
# torch.nn.TransformerEncoder hangs when torch.nn is first imported AFTER
# joblib.load() of a large pkl file (e.g. sleep_staging_model.pkl, 42 MB).
# Importing torch here — before SleepStagingModel's joblib call — fixes the hang.
try:
    import torch as _torch  # noqa: F401  (side-effect: initializes thread pool)
    import torch.nn as _torch_nn  # noqa: F401
except ImportError:
    pass

# Amplitude threshold above which an epoch is flagged as artifact-contaminated.
# From Krigolson (2021): 75 µV catches most blinks (100-200 µV) and EMG bursts.
_ARTIFACT_THRESHOLD_UV = 75.0

EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]

# Minimum benchmark accuracy to trust a trained model over feature-based.
# 60% threshold: the DEAP model at ~51% has severe class-imbalance bias toward "sad"
# (11 165 sad samples vs 1 769 focused) — below this threshold we use feature-based.
_MIN_MODEL_ACCURACY = 0.60

# ── Device-aware gamma masking ────────────────────────────────────────────────
# Gamma (30-50 Hz) is dominated by EMG (muscle artifact) on consumer dry-electrode
# EEG devices.  When one of these devices is active, gamma features are zeroed
# before the LGBM prediction so the model relies on alpha/beta/theta instead.
# Research-grade EEG (gel electrodes, controlled lab) uses gamma normally.

# Gamma feature indices in the 85-feature vector
# band-major layout: delta(0-15), theta(16-31), alpha(32-47), beta(48-63), gamma(64-79)
# DASM: delta(80) theta(81) alpha(82) beta(83) gamma(84)
_GAMMA_FEAT_IDX: list[int] = list(range(64, 80)) + [84]  # 17 features

# Consumer dry-electrode devices — gamma is EMG, not neural signal
_CONSUMER_EEG_DEVICES: frozenset[str] = frozenset({
    "muse_2", "muse_2_bled", "muse_s", "muse_s_bled",
    "muse_2016", "muse_2016_bled", "muse",
})

# ── 85-feature human-readable names ────────────────────────────────────────
# Layout matches _extract_muse_live_features() and train_cross_dataset_lgbm.py:
#   80 band-power stats: 5 bands x 4 channels x 4 stats (mean, std, median, IQR)
#    5 DASM features:    mean(AF8_band) - mean(AF7_band) per band
# Channel order: TP9=0, AF7=1, AF8=2, TP10=3
_BANDS_5 = ["delta", "theta", "alpha", "beta", "gamma"]
_CHANNELS_4 = ["TP9", "AF7", "AF8", "TP10"]
_STATS_4 = ["mean", "std", "median", "iqr"]

_FEATURE_NAMES_85: list[str] = []
for _band in _BANDS_5:
    for _ch in _CHANNELS_4:
        for _stat in _STATS_4:
            _FEATURE_NAMES_85.append(f"{_band}_{_ch}_{_stat}")
for _band in _BANDS_5:
    _FEATURE_NAMES_85.append(f"DASM_{_band}")
assert len(_FEATURE_NAMES_85) == 85, f"Expected 85 feature names, got {len(_FEATURE_NAMES_85)}"

def _compute_dominance(beta_alpha: float, theta_beta_ratio: float) -> float:
    """Estimate dominance (sense of control / agency) from EEG band ratios.

    High beta + low theta at frontal = high dominance (in control).
    High theta + low beta = low dominance (overwhelmed).
    Returns value clipped to [0, 1].
    """
    return float(np.clip(
        0.50 * np.tanh((beta_alpha - 0.5) * 2.0)
        + 0.50 * np.tanh((1 - theta_beta_ratio) * 2.0),
        0, 1))


# ── Russell Circumplex emotion centers (valence, arousal) ─────────────────────
# Coordinates calibrated to the Muse 2 EEG output range:
#   valence in [-1, +1], arousal in [0, 1]
#
# Sources:
#   Russell (1980) — original 2D circumplex model
#   Davidson (1992) — FAA valence anchors
#   Posner, Russell & Peterson (2005) — updated center estimates
#
# Emotion mapping to EMOTIONS list indices:
#   0=happy, 1=sad, 2=angry, 3=fearful, 4=relaxed, 5=focused
#
# "relaxed" maps to low-arousal positive quadrant; "focused" maps to moderate
# arousal with near-neutral valence (task engagement without strong affect).
# "fearful" (freeze/flight) occupies high-arousal negative quadrant distinct
# from "angry" (approach/fight) which is also high-arousal negative but with
# stronger beta dominance — the FAA/arousal split differentiates them.
_CIRCUMPLEX_CENTERS: dict = {
    # emotion: (valence_center, arousal_center) — Russell's Circumplex
    # Uses the standard 6 emotions: happy, sad, angry, fear, surprise, neutral
    "happy":    ( 0.60,  0.65),  # high positive valence, moderately high arousal
    "sad":      (-0.55,  0.22),  # negative valence, clearly low arousal
    "angry":    (-0.40,  0.75),  # negative valence, high arousal (approach drive)
    "fear":     (-0.50,  0.70),  # negative valence, high arousal (freeze/flight)
    "surprise": ( 0.15,  0.85),  # slightly positive valence, very high arousal
    "neutral":  ( 0.00,  0.40),  # near-zero valence, moderate arousal
}

# Elliptical half-axes for each emotion in the circumplex space.
# Wider valence axis for happy/sad (gradual onset); narrower for angry/fear
# (requires stronger signals before those labels fire).
# Format: (sigma_valence, sigma_arousal)
_CIRCUMPLEX_SIGMA: dict = {
    "happy":    (0.45, 0.40),
    "sad":      (0.45, 0.35),
    "angry":    (0.35, 0.30),
    "fear":     (0.35, 0.30),
    "surprise": (0.35, 0.30),
    "neutral":  (0.50, 0.45),  # wider — neutral is the "default" region
}

# Softmax temperature for converting inverse-distances to probabilities.
# Higher value → sharper peaks (more decisive); lower → more uniform.
_CIRCUMPLEX_TEMPERATURE: float = 4.0


def _expand_3class_to_6(
    p_pos: float,
    p_neu: float,
    p_neg: float,
    valence: float,
    arousal: float,
) -> np.ndarray:
    """Expand 3-class LGBM output (positive/neutral/negative) to 6-class
    probabilities using soft distance-based blending in valence-arousal space.

    Algorithm
    ---------
    1. Compute a per-emotion weight from the elliptical Gaussian distance of
       (valence, arousal) from each emotion's circumplex center.
    2. Apply a positivity mask: positive-class probability is gated to
       positive-valence emotions (happy, relaxed); neutral to near-neutral
       emotions (focused, relaxed); negative to negative-valence emotions
       (sad, angry, fearful).  The mask is *soft* — cross-quadrant leakage is
       allowed but suppressed, avoiding hard cutoffs.
    3. Multiply each masked weight by the corresponding class probability
       (p_pos, p_neu, p_neg) and softmax-normalize.

    This replaces the previous hard if/elif threshold routing which assigned
    100% of each class probability to a single bucket, making the output
    extremely sensitive to threshold crossings.

    Parameters
    ----------
    p_pos, p_neu, p_neg : float
        LGBM probabilities for positive / neutral / negative class.
        Must sum to ≈ 1.0.
    valence : float
        Blended valence estimate in [-1, +1] (FAA + alpha/beta ratio).
    arousal : float
        Arousal estimate in [0, 1] (beta/alpha + delta).

    Returns
    -------
    np.ndarray, shape (6,)
        Probability vector over EMOTIONS = [happy, sad, angry, fear, surprise, neutral].
        Sums to 1.0.
    """
    emotions_ordered = ["happy", "sad", "angry", "fear", "surprise", "neutral"]

    # Step 1: Gaussian distance weight for each emotion center
    weights = np.zeros(6, dtype=np.float64)
    for i, emo in enumerate(emotions_ordered):
        cv, ca = _CIRCUMPLEX_CENTERS[emo]
        sv, sa = _CIRCUMPLEX_SIGMA[emo]
        dv = (valence - cv) / sv
        da = (arousal - ca) / sa
        weights[i] = np.exp(-0.5 * (dv * dv + da * da))

    # Step 2: Soft class-membership masks
    # Gate function: sigmoid(k * (x - threshold))
    def _sigmoid(x: float, k: float = 4.0) -> float:
        return float(1.0 / (1.0 + np.exp(-k * x)))

    pos_gate = _sigmoid(valence, k=4.0)        # 1 when valence >> 0
    neg_gate = _sigmoid(-valence, k=4.0)       # 1 when valence << 0
    neu_gate = 1.0 - abs(valence) * 0.7        # peaks at valence=0

    hi_arousal_gate = _sigmoid(arousal - 0.50, k=6.0)
    lo_arousal_gate = _sigmoid(0.50 - arousal, k=6.0)
    very_hi_arousal = _sigmoid(arousal - 0.70, k=6.0)

    # Per-emotion class affinity:
    # happy:    positive valence, moderate-high arousal
    # sad:      negative valence, low arousal
    # angry:    negative valence, high arousal (splits with fear)
    # fear:     negative valence, high arousal (splits with angry)
    # surprise: any valence, very high arousal
    # neutral:  near-zero valence, moderate arousal
    class_affinity = np.array([
        p_pos * pos_gate,                                    # happy
        p_neg * neg_gate * lo_arousal_gate,                  # sad
        p_neg * neg_gate * hi_arousal_gate * 0.5,            # angry
        p_neg * neg_gate * hi_arousal_gate * 0.5,            # fear
        (p_pos * 0.3 + p_neu * 0.3 + p_neg * 0.1) * very_hi_arousal,  # surprise (any valence, very high arousal)
        p_neu * neu_gate * (1.0 - very_hi_arousal),          # neutral (not high arousal)
    ], dtype=np.float64)

    # Step 3: Element-wise product: circumplex weight × class affinity
    combined = weights * class_affinity

    # Softmax with temperature to keep output smooth
    combined_temp = combined * _CIRCUMPLEX_TEMPERATURE
    combined_temp -= combined_temp.max()   # numerical stability
    exp_vals = np.exp(combined_temp)
    total = exp_vals.sum()
    if total < 1e-10:
        # Degenerate: fall back to uniform
        return np.ones(6, dtype=np.float32) / 6.0
    return (exp_vals / total).astype(np.float32)


# ── PredictionStabilityTracker ─────────────────────────────────────────────────
#
# Tracks how *stable* the classifier's predictions are over time using cosine
# similarity between consecutive probability vectors.  Rapid flipping
# (happy->angry->sad in 10 seconds) means the signal is noisy or the person
# is near an emotion boundary — either way, confidence should be penalized.
#
# The tracker maintains an EMA of pairwise cosine similarities.  High stability
# (~1.0) means predictions are consistent.  Low stability (<0.5) means rapid
# flipping, and confidence is multiplied by a penalty in [0.5, 1.0].


class PredictionStabilityTracker:
    """Cosine-similarity-based emotion prediction stability tracker.

    Computes cosine similarity between the current and previous probability
    vectors.  An EMA over these similarities produces a smooth ``stability``
    score in [0, 1].  When stability is low, ``adjust_confidence()`` applies
    a penalty to reduce reported confidence.

    The penalty ramp is:
        stability >= 0.8  -> no penalty  (multiplier = 1.0)
        stability <= 0.3  -> max penalty (multiplier = 0.5)
        in between        -> linear interpolation
    """

    _EMA_ALPHA = 0.3  # Smoothing factor for stability EMA

    def __init__(self) -> None:
        self._prev_probs: Optional[np.ndarray] = None
        self._stability: float = 1.0  # Default: fully stable

    @property
    def stability(self) -> float:
        return self._stability

    def update(self, probs: np.ndarray) -> float:
        """Feed a new probability vector and return the updated stability score.

        Args:
            probs: 1-D array of class probabilities (e.g. 6 emotions).

        Returns:
            Updated stability score in [0, 1].
        """
        probs = np.asarray(probs, dtype=float)

        if self._prev_probs is None:
            self._prev_probs = probs.copy()
            return self._stability  # 1.0 on first call

        # Cosine similarity: dot(a,b) / (|a| * |b|)
        norm_curr = np.linalg.norm(probs)
        norm_prev = np.linalg.norm(self._prev_probs)

        if norm_curr < 1e-12 or norm_prev < 1e-12:
            cos_sim = 0.0  # Degenerate (all-zero) -> treat as unstable
        else:
            cos_sim = float(np.dot(probs, self._prev_probs) / (norm_curr * norm_prev))
            cos_sim = max(0.0, min(1.0, cos_sim))  # Clamp to [0, 1]

        # EMA update
        self._stability = (
            self._EMA_ALPHA * cos_sim + (1 - self._EMA_ALPHA) * self._stability
        )
        self._prev_probs = probs.copy()

        return self._stability

    def adjust_confidence(self, raw_confidence: float) -> float:
        """Apply stability-based penalty to raw confidence.

        When predictions are flipping rapidly, confidence should be lower
        because the model is effectively guessing.

        Penalty ramp:
            stability >= 0.8  -> multiplier = 1.0 (no penalty)
            stability <= 0.3  -> multiplier = 0.5 (halved confidence)
            between           -> linear interpolation
        """
        if self._stability >= 0.8:
            multiplier = 1.0
        elif self._stability <= 0.3:
            multiplier = 0.5
        else:
            # Linear ramp from 0.5 at stability=0.3 to 1.0 at stability=0.8
            multiplier = 0.5 + 0.5 * (self._stability - 0.3) / 0.5
        return float(raw_confidence * multiplier)


# Singleton RunningNormalizer for session drift correction
_running_normalizer: Optional[object] = None


def _get_running_normalizer() -> Optional[object]:
    global _running_normalizer
    if _running_normalizer is None:
        try:
            from processing.eeg_processor import RunningNormalizer
            _running_normalizer = RunningNormalizer()
        except Exception:
            pass
    return _running_normalizer


class EmotionClassifier:
    """EEG-based emotion classifier with ONNX/sklearn/feature-based inference.

    Priority: multichannel DEAP model → ONNX → sklearn .pkl → feature-based.
    Trained models are only used if their benchmark accuracy >= 60%.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.onnx_session = None
        self.sklearn_model = None
        self.feature_names = None
        self.scaler = None
        self.multichannel_model = False  # True when loaded pkl was trained with multichannel features
        self.model_type = "feature-based"
        self._benchmark_accuracy = 0.0

        # Exponential moving average for smoothing (prevents flickering)
        self._ema_probs = None  # smoothed probabilities
        self._ema_alpha = 0.35  # smoothing factor — 0.35 gives slightly faster response
        #                         than 0.30 while still suppressing rapid noise bursts
        self._history = deque(maxlen=10)  # recent band power snapshots

        # EMA smoothing for continuous output indices (valence, arousal, mental-state indices).
        # With 2-second epoch hops, alpha=0.4 gives effective time constant
        # tau = -2/ln(0.6) ≈ 3.9 seconds — within the recommended 3-5 sec range.
        # Reduces frame-to-frame jitter on dashboard readings while preserving
        # responsiveness to genuine emotional state changes.
        self._ema_indices_alpha = 0.4
        self._ema_valence: Optional[float] = None
        self._ema_arousal: Optional[float] = None
        self._ema_stress: Optional[float] = None
        self._ema_focus: Optional[float] = None
        self._ema_relaxation: Optional[float] = None
        self._ema_anger: Optional[float] = None
        self._ema_fear: Optional[float] = None

        # Valence trajectory — last 5 EMA-smoothed valence values for trend detection.
        # Linear regression slope → "improving" / "stable" / "declining".
        # 5 samples at 2-sec hops = 10-sec lookback window.
        self._valence_history: deque = deque(maxlen=5)

        # Prediction stability — cosine similarity tracker for confidence penalty
        self._stability_tracker = PredictionStabilityTracker()

        # Covariate shift detector — monitors whether live EEG feature
        # distributions have drifted from the baseline/training distribution.
        # Feeds 85-dim feature vectors from _extract_muse_live_features().
        # During baseline calibration, features go to add_reference().
        # During live inference, features go to add_live().
        # When shift is detected, confidence is penalized in the output.
        self._shift_detector = CovariateShiftDetector(
            n_features=85,
            reference_window=60,    # ~2 min at 2-sec hops
            live_window=15,         # ~30 sec at 2-sec hops
            alpha=0.05,
            min_reference=30,       # ~1 min before activation
            min_live=10,            # ~20 sec of live data
        )
        self._shift_baseline_complete = False

        # SHAP explainers — lazily initialised on first use (one per LGBM model)
        self._shap_explainer_mega: Optional[object] = None
        self._shap_explainer_muse: Optional[object] = None

        # Device-aware gamma masking — set via set_device_type() on connect/disconnect
        self.device_type: Optional[str] = None

        # Euclidean Alignment (He & Wu, IEEE TBME 2020) — online session alignment.
        # Accumulates per-session covariance and whitens incoming epochs to remove
        # subject-specific spatial patterns. Activates after 10 epochs (~20 sec at
        # 2-sec hop). Provides +4.33% cross-subject accuracy (arXiv:2401.10746).
        self._euclidean_aligner = EuclideanAligner(n_channels=4, min_epochs=10)

        # EEGNet — device-agnostic CNN, highest priority when trained model exists.
        # Works with 4-ch Muse 2, 8-ch OpenBCI Cyton, 16-ch Cyton+Daisy without retraining.
        # Model file: models/saved/eegnet_emotion_{n_channels}ch.pt
        try:
            from models.eegnet import EEGNetEmotionClassifier
            self._eegnet = EEGNetEmotionClassifier()
        except Exception:
            self._eegnet = None

        # TSception — asymmetry-aware spatial CNN (69% CV), fallback after LGBM models.
        # Specifically designed for left/right hemisphere asymmetry (AF7/AF8 on Muse 2).
        # Requires >= 4-second epoch (1024 samples @ 256 Hz). Weights: models/saved/tsception_emotion.pt
        self._tsception = None
        self._try_load_tsception()

        # REVE Foundation (brain-bzh/reve-base, NeurIPS 2025) — highest priority when available.
        # Pre-trained on 60K+ hours of EEG from 92 datasets / 25,000 subjects.
        # Gated on HuggingFace: request access at huggingface.co/brain-bzh/reve-base
        # Falls through silently when access not yet granted (state=ACCESS_DENIED).
        self._reve_foundation = None
        self._try_load_reve_foundation()

        # REVE-inspired DETransformer (FACED dataset, 30-sec epochs, temporal DE features)
        # Second-highest priority — activates only when >= 30 sec of data available.
        # Model file: models/saved/reve_emotion_4ch.pt
        self._reve = None
        self._try_load_reve()

        # Mega cross-dataset LGBM with global PCA (DEAP+DREAMER+GAMEEMO+DENS, highest priority)
        self.mega_lgbm_model  = None
        self.mega_lgbm_scaler = None  # StandardScaler (85→85)
        self.mega_lgbm_pca    = None  # PCA (85→80)
        self._mega_lgbm_benchmark = 0.0
        self._try_load_mega_lgbm()
        # Muse-native LightGBM (85 raw features, no PCA)
        self.lgbm_muse_model = None
        self.lgbm_muse_scaler = None
        self._lgbm_muse_benchmark = 0.0
        self._try_load_lgbm_muse_model()
        # Try loading the DEAP-trained multichannel model first
        self._try_load_deap_model()

        if model_path and self.model_type == "feature-based":
            self._load_model(model_path)

    # ── EMA helpers for continuous indices ──────────────────────────────────────

    def _smooth_index(self, attr: str, raw: float) -> float:
        """Apply EMA smoothing to a continuous output index.

        On first call (stored value is None), seeds with the raw value.
        On subsequent calls, blends raw into the running average using
        ``_ema_indices_alpha``.
        """
        prev = getattr(self, attr)
        if prev is None:
            smoothed = raw
        else:
            a = self._ema_indices_alpha
            smoothed = a * raw + (1.0 - a) * prev
        setattr(self, attr, smoothed)
        return smoothed

    # ── Valence trajectory (mood trend) ──────────────────────────────────────

    def _compute_trajectory(self) -> Dict:
        """Compute emotional trajectory from recent valence history.

        Performs simple linear regression on the last N EMA-smoothed valence
        values.  Returns direction label, magnitude, and confidence.

        Returns:
            Dict with:
            - trajectory: "improving" | "stable" | "declining"
            - trajectory_magnitude: absolute slope (float >= 0)
            - trajectory_confidence: R^2 * sample_weight (float 0-1)
        """
        n = len(self._valence_history)
        if n < 2:
            return {
                "trajectory": "stable",
                "trajectory_magnitude": 0.0,
                "trajectory_confidence": 0.0,
            }

        y = np.array(self._valence_history)
        x = np.arange(n, dtype=float)

        # Simple linear regression: slope = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        ss_xx = float(np.sum((x - x_mean) ** 2))
        ss_xy = float(np.sum((x - x_mean) * (y - y_mean)))

        if ss_xx < 1e-10:
            slope = 0.0
        else:
            slope = ss_xy / ss_xx

        # R^2 — coefficient of determination
        ss_yy = float(np.sum((y - y_mean) ** 2))
        if ss_yy < 1e-10:
            r_squared = 1.0 if abs(slope) < 1e-6 else 0.0
        else:
            y_pred = x_mean + slope * (x - x_mean) + y_mean - slope * x_mean
            # Simpler: y_pred = slope * x + (y_mean - slope * x_mean)
            ss_res = float(np.sum((y - (slope * x + (y_mean - slope * x_mean))) ** 2))
            r_squared = float(np.clip(1.0 - ss_res / ss_yy, 0.0, 1.0))

        # Sample weight: scale confidence by how many of 5 samples we have
        sample_weight = min(n, 5) / 5.0
        confidence = float(np.clip(r_squared * sample_weight, 0.0, 1.0))

        # Direction: slope threshold at 0.02 per step (~0.01 valence per second
        # with 2-sec hops). Below threshold = stable.
        abs_slope = abs(slope)
        if abs_slope < 0.02:
            direction = "stable"
        elif slope > 0:
            direction = "improving"
        else:
            direction = "declining"

        return {
            "trajectory": direction,
            "trajectory_magnitude": round(abs_slope, 4),
            "trajectory_confidence": round(confidence, 4),
        }

    # ── Device-aware gamma masking ─────────────────────────────────────────────

    def set_device_type(self, device_type: Optional[str]) -> None:
        """Notify the classifier which EEG device is connected.

        Call this whenever a device connects or disconnects.  When a consumer
        dry-electrode device (Muse 2, Muse S, …) is active, gamma features are
        zeroed before LGBM inference because gamma is EMG on those devices.
        """
        self.device_type = device_type

    @property
    def _is_consumer_device(self) -> bool:
        """True when the active device records EMG-contaminated gamma."""
        if self.device_type is None:
            return False
        return self.device_type.lower() in _CONSUMER_EEG_DEVICES

    # ── Fast epoch-level signal quality for adaptive ensemble ──────────────────
    #
    # Lightweight quality check that runs inside the ensemble prediction path.
    # Uses three fast, FFT-free metrics:
    #   1. Amplitude range — flatline or saturated channels
    #   2. High-frequency power ratio — EMG / broadband noise
    #   3. Channel variance consistency — disconnected electrode detection
    #
    # NOT a replacement for SignalQualityChecker (which is thorough but too slow
    # to run on every prediction frame).  This is a 0.1 ms check that drives
    # the adaptive ensemble weights.

    # Quality thresholds for the adaptive ensemble weight ramp.
    # Above _QUALITY_GOOD: full EEGNet weight (default 0.6).
    # Below _QUALITY_POOR: zero EEGNet weight (heuristic-only).
    # Between: linear interpolation.
    _QUALITY_GOOD = 0.7
    _QUALITY_POOR = 0.35

    def _compute_epoch_quality(self, eeg: np.ndarray, fs: float = 256.0) -> float:
        """Fast signal quality estimate for a single epoch.

        Computes a 0-1 quality score from three lightweight metrics averaged
        across channels.  Designed to be fast enough to run on every prediction
        frame (~0.1 ms for 4ch x 1024 samples).

        Args:
            eeg: 1D (n_samples,) or 2D (n_channels, n_samples) EEG array.
            fs:  Sampling frequency in Hz.

        Returns:
            Quality score in [0, 1].  1.0 = clean lab-grade signal,
            0.0 = flatline or fully contaminated.
        """
        if eeg.ndim == 1:
            eeg = eeg.reshape(1, -1)

        n_channels, n_samples = eeg.shape
        if n_samples < 4:
            return 0.0

        scores = []

        for ch in range(n_channels):
            signal = eeg[ch]
            ch_score = self._channel_quality(signal, fs)
            scores.append(ch_score)

        if not scores:
            return 0.0

        # Per-channel quality averaged, then penalize if channels are
        # inconsistent (one good, one dead = unreliable spatial features).
        mean_quality = float(np.mean(scores))

        if n_channels >= 2:
            variance_penalty = float(np.std(scores))
            # Penalize up to 0.2 for high inter-channel variance
            mean_quality -= min(0.2, variance_penalty * 0.5)

        return float(np.clip(mean_quality, 0.0, 1.0))

    @staticmethod
    def _channel_quality(signal: np.ndarray, fs: float) -> float:
        """Quality score for a single channel. Returns float in [0, 1]."""
        rms = float(np.sqrt(np.mean(signal ** 2)))
        peak = float(np.max(np.abs(signal)))

        # 1. Amplitude score:
        #    Flatline (RMS < 1 uV) -> hard 0 (no neural info regardless of other metrics)
        #    Clean EEG (5-30 uV RMS) = high
        #    High RMS (> 40 uV) = noisy / near-artifact
        #    Approaching artifact threshold (peak > 75 uV) = low
        if rms < 1.0:
            # Flatline / disconnected: no neural information at all.
            # Return immediately — no point checking HF or stationarity.
            return 0.05
        elif rms > 60.0:
            amp_score = 0.15  # near-artifact / saturated
        elif peak > _ARTIFACT_THRESHOLD_UV:
            amp_score = 0.25  # individual peaks hit artifact threshold
        elif rms > 40.0:
            amp_score = 0.40  # elevated — noisy but not saturated
        else:
            # Good range: 1-40 uV RMS.  Peak quality around 10-25 uV.
            amp_score = float(np.clip(1.0 - abs(rms - 17.0) / 30.0, 0.4, 1.0))

        # 2. High-frequency power ratio (EMG contamination proxy).
        #    Compute power above 30 Hz vs total power using simple time-domain
        #    second-order difference filter (no FFT needed — fast).
        if len(signal) >= 3:
            hf_proxy = np.diff(signal, n=2)  # approximates high-pass
            hf_power = float(np.mean(hf_proxy ** 2))
            total_power = float(np.mean(signal ** 2)) + 1e-10
            hf_ratio = hf_power / total_power
            # Clean EEG: hf_ratio ~0.5-1.5 (dominated by low-freq oscillations).
            # Heavy EMG / line noise: hf_ratio > 3.
            # Map [0, 4] -> [1.0, 0.1]
            hf_score = float(np.clip(1.0 - hf_ratio / 4.0, 0.1, 1.0))
        else:
            hf_score = 0.5

        # 3. Stationarity (variance stability across epoch quarters).
        #    Non-stationary signal = motion artifact or electrode shift.
        quarter = max(1, len(signal) // 4)
        quarters = [signal[i * quarter:(i + 1) * quarter] for i in range(4)]
        quarter_vars = [float(np.var(q)) for q in quarters if len(q) > 0]
        if len(quarter_vars) >= 2 and max(quarter_vars) > 0:
            cv = float(np.std(quarter_vars)) / (float(np.mean(quarter_vars)) + 1e-10)
            # Coefficient of variation: < 0.5 = stationary, > 2.0 = non-stationary
            stat_score = float(np.clip(1.0 - cv / 2.0, 0.2, 1.0))
        else:
            stat_score = 0.5

        # Weighted combination: amplitude most important, then HF, then stationarity
        return 0.40 * amp_score + 0.35 * hf_score + 0.25 * stat_score

    def _adaptive_ensemble_weights(
        self, epoch_quality: float
    ) -> tuple:
        """Compute quality-adaptive ensemble weights for EEGNet vs heuristic.

        When signal quality is high, use the default weights (0.6 EEGNet / 0.4
        heuristic).  When quality drops below the poor threshold, use heuristic
        only (0.0 / 1.0).  Linear ramp between thresholds.

        Returns:
            (w_eegnet, w_heuristic) tuple that sums to 1.0.
        """
        max_w_eegnet = self._ENSEMBLE_W_EEGNET  # 0.6

        if epoch_quality >= self._QUALITY_GOOD:
            w_e = max_w_eegnet
        elif epoch_quality <= self._QUALITY_POOR:
            w_e = 0.0
        else:
            # Linear ramp from 0 at _QUALITY_POOR to max_w_eegnet at _QUALITY_GOOD
            t = (epoch_quality - self._QUALITY_POOR) / (self._QUALITY_GOOD - self._QUALITY_POOR)
            w_e = max_w_eegnet * t

        w_h = 1.0 - w_e
        return (round(w_e, 4), round(w_h, 4))

    # ── SHAP Explainability ────────────────────────────────────────────────────

    def _get_shap_explainer(self, model, cache_attr: str):
        """Lazily create and cache a SHAP TreeExplainer for a LightGBM model.

        Returns None if shap is not installed or explainer creation fails.
        """
        cached = getattr(self, cache_attr, None)
        if cached is not None:
            return cached
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            setattr(self, cache_attr, explainer)
            return explainer
        except Exception as exc:
            logging.getLogger(__name__).debug("SHAP explainer init failed: %s", exc)
            return None

    def _compute_shap_explanation_lgbm(
        self,
        model,
        feat_input: np.ndarray,
        predicted_class_idx: int,
        cache_attr: str,
        feature_names: list,
        pca=None,
        feat_pre_pca: Optional[np.ndarray] = None,
    ) -> list:
        """Compute SHAP explanation for a LightGBM prediction.

        For PCA models: projects SHAP values back to the original 85-feature
        space using pca.components_ so explanations reference real EEG features
        (e.g. "alpha_AF7_mean") rather than opaque PCA component indices.

        Returns a list of top-3 contributing features, each:
            {"feature": str, "impact": float, "direction": "increases"/"decreases"}

        Returns [] on any failure — SHAP must never break prediction.
        """
        try:
            explainer = self._get_shap_explainer(model, cache_attr)
            if explainer is None:
                return []

            # shap_values format varies by shap version:
            #   Old (list): list of arrays, one per class, each (n_samples, n_features)
            #   New (3D ndarray): shape (n_samples, n_features, n_classes)
            shap_values = explainer.shap_values(feat_input)

            # Extract SHAP values for the predicted class
            if isinstance(shap_values, list):
                # Old format: list[class_idx] -> (n_samples, n_features)
                sv_row = shap_values[predicted_class_idx][0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # New format: (n_samples, n_features, n_classes)
                sv_row = shap_values[0, :, predicted_class_idx]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                # Binary or single output: (n_samples, n_features)
                sv_row = shap_values[0]
            else:
                sv_row = np.asarray(shap_values).ravel()

            # Project back from PCA space to original 85-feature space
            if pca is not None and feat_pre_pca is not None:
                # pca.components_ shape: (n_components, n_original_features)
                # sv_row shape: (n_components,)
                # original_importance = components_.T @ sv_row → (n_original_features,)
                original_importance = pca.components_.T @ sv_row
                names = feature_names
            else:
                original_importance = sv_row
                names = feature_names

            # Top 3 by absolute impact
            top_k = min(3, len(original_importance))
            top_indices = np.argsort(np.abs(original_importance))[-top_k:][::-1]

            explanation = []
            for i in top_indices:
                impact = float(original_importance[i])
                explanation.append({
                    "feature": names[i] if i < len(names) else f"feature_{i}",
                    "impact": round(impact, 6),
                    "direction": "increases" if impact > 0 else "decreases",
                })
            return explanation

        except Exception as exc:
            logging.getLogger(__name__).debug("SHAP explanation failed: %s", exc)
            return []

    @staticmethod
    def _compute_heuristic_explanation(contributions: list) -> list:
        """Build top-3 explanation from explicit heuristic formula contributions.

        Each entry in *contributions* is a (name, value) tuple where *value*
        is the signed contribution of that feature to the predicted emotion.
        """
        try:
            sorted_contribs = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
            explanation = []
            for name, value in sorted_contribs[:3]:
                explanation.append({
                    "feature": name,
                    "impact": round(float(value), 6),
                    "direction": "increases" if value > 0 else "decreases",
                })
            return explanation
        except Exception:
            return []

    def _ensure_explanation(self, result: Dict) -> Dict:
        """Guarantee the result dict contains explanation, dominance, and granular_emotions.

        External model paths (EEGNet, REVE, TSception) do not compute SHAP
        explanations or granularity fields. This method injects defaults so
        the API contract is always satisfied.

        Also applies EMA smoothing to continuous indices — without this,
        external model paths return raw per-frame values that jitter on every
        2-second epoch hop, while only the feature-based path was smoothed.
        """
        if "explanation" not in result:
            result["explanation"] = []
        # EMA smooth continuous indices for external model paths (EEGNet, REVE, TSception).
        # _smooth_index is a no-op on first call (seeds EMA), then blends on subsequent calls.
        if "valence" in result:
            result["valence"] = self._smooth_index("_ema_valence", result["valence"])
        if "arousal" in result:
            result["arousal"] = self._smooth_index("_ema_arousal", result["arousal"])
        if "stress_index" in result:
            result["stress_index"] = self._smooth_index("_ema_stress", result["stress_index"])
        if "focus_index" in result:
            result["focus_index"] = self._smooth_index("_ema_focus", result["focus_index"])
        if "relaxation_index" in result:
            result["relaxation_index"] = self._smooth_index("_ema_relaxation", result["relaxation_index"])
        if "anger_index" in result:
            result["anger_index"] = self._smooth_index("_ema_anger", result["anger_index"])
        if "fear_index" in result:
            result["fear_index"] = self._smooth_index("_ema_fear", result["fear_index"])
        # Ensure dominance + granular_emotions for external model paths
        if "dominance" not in result:
            # Default dominance when band power ratios unavailable from external models
            result["dominance"] = 0.5
        if "granular_emotions" not in result:
            result["granular_emotions"] = map_vad_to_granular_emotions(
                result.get("valence", 0.0),
                result.get("arousal", 0.5),
                result.get("dominance", 0.5),
            )
        return result

    # ── Per-user fine-tuned model inference ──────────────────────────────────

    # Cache: user_id -> (model, scaler, meta) or None (checked and missing)
    _user_model_cache: Dict[str, Optional[tuple]] = {}

    def _try_predict_user_model(
        self, eeg: np.ndarray, fs: float, user_id: str, device_type: str
    ) -> Optional[Dict]:
        """Try inference with a per-user fine-tuned EEG model.

        Loads from user_models/{user_id}/eeg_classifier.pkl if it exists.
        Returns None if no user model is available (falls through to generic).
        """
        # Check cache first
        if user_id not in self._user_model_cache:
            try:
                from training.retrain_from_user_data import UserModelRetrainer
                retrainer = UserModelRetrainer(user_id)
                loaded = retrainer.load_eeg_model()
                self._user_model_cache[user_id] = loaded
            except Exception:
                self._user_model_cache[user_id] = None

        cached = self._user_model_cache.get(user_id)
        if cached is None:
            return None

        model, scaler, meta = cached
        try:
            # Extract the same 85-dim features used for training
            feat = self._extract_muse_live_features(eeg, fs)  # 85-dim
            if self._is_consumer_device:
                feat[_GAMMA_FEAT_IDX] = 0.0

            # Pad to 170-dim (training concatenated raw + scaled features)
            feat_170 = np.zeros(170, dtype=np.float32)
            feat_170[:85] = feat
            feat_170[85:170] = feat  # duplicate as stand-in for scaled version

            if scaler is not None:
                feat_input = scaler.transform(feat_170.reshape(1, -1))
            else:
                feat_input = feat_170.reshape(1, -1)

            proba = model.predict_proba(feat_input)[0]
            pred_idx = int(np.argmax(proba))
            classes = meta.get("classes", meta.get("label_encoder_classes", EMOTIONS))

            # Map prediction back to emotion label
            if pred_idx < len(classes):
                emotion = classes[pred_idx]
            else:
                emotion = "neutral"

            # Build a 6-class probability dict matching EMOTIONS order
            probs6 = np.zeros(6, dtype=np.float32)
            for i, cls in enumerate(classes):
                if i < len(proba) and cls in EMOTIONS:
                    emo_idx = EMOTIONS.index(cls)
                    probs6[emo_idx] = proba[i]
            total = probs6.sum()
            if total > 0:
                probs6 /= total
            else:
                probs6 = np.ones(6, dtype=np.float32) / 6.0

            # Build result via standard path for consistent output format
            emotion_idx = int(np.argmax(probs6))
            result = self._build_muse_result(
                emotion_idx, probs6, eeg, fs,
                artifact_detected=False, device_type=device_type,
            )
            result["model_type"] = "user-finetuned"
            result["user_model_accuracy"] = meta.get("train_accuracy")
            result["explanation"] = []
            return result

        except Exception as exc:
            logging.getLogger(__name__).debug(
                "User model inference failed for %s: %s", user_id, exc
            )
            return None

    def invalidate_user_model_cache(self, user_id: str) -> None:
        """Clear cached user model so it gets reloaded on next predict call."""
        self._user_model_cache.pop(user_id, None)

    def _try_load_deap_model(self):
        """Try loading the DEAP-trained Muse 2 model (best accuracy)."""
        from pathlib import Path
        pkl_path = Path("models/saved/emotion_deap_muse.pkl")
        if not pkl_path.exists():
            return

        bench = self._read_deap_benchmark()
        if bench < _MIN_MODEL_ACCURACY:
            return

        try:
            import joblib
            data = joblib.load(pkl_path)
            self.sklearn_model = data["model"]
            self.feature_names = data["feature_names"]
            self.scaler = data.get("scaler")
            self.multichannel_model = bool(data.get("multichannel", False))
            self.model_type = "sklearn-deap"
            self._benchmark_accuracy = bench
        except Exception:
            pass

    def _try_load_mega_lgbm(self) -> None:
        """Load the mega cross-dataset LGBM (global PCA 85→80, DEAP+DREAMER+GAMEEMO+DENS)."""
        import json
        from pathlib import Path
        bench_path = Path("benchmarks/emotion_mega_lgbm_benchmark.json")
        model_path = Path("models/saved/emotion_mega_lgbm.pkl")
        if not bench_path.exists() or not model_path.exists():
            return
        try:
            bench_data = json.loads(bench_path.read_text())
            acc = float(bench_data.get("accuracy", 0.0))
            if acc < _MIN_MODEL_ACCURACY:
                return
            import pickle
            with open(model_path, "rb") as fh:
                payload = pickle.load(fh)
            self.mega_lgbm_model  = payload["model"]
            self.mega_lgbm_scaler = payload["scaler"]
            self.mega_lgbm_pca    = payload["pca"]
            self._mega_lgbm_benchmark = acc
            print(f"[EmotionClassifier] Loaded mega cross-dataset LGBM "
                  f"(CV acc={acc:.4f}, datasets={payload.get('datasets')})")
        except Exception:
            pass

    def _try_load_reve_foundation(self) -> None:
        """Load the real REVE foundation model (brain-bzh/reve-base, NeurIPS 2025).

        Only activates when weights are PRETRAINED or FINETUNED (not RANDOM_INIT).
        Requires HuggingFace approval: huggingface.co/brain-bzh/reve-base
        Set HF_TOKEN env var with a token that has approved access.
        Falls through silently when access not yet granted or only random weights.
        """
        try:
            from models.reve_foundation import REVEFoundationWrapper
            wrapper = REVEFoundationWrapper()
            state = wrapper.status()
            if state in ("PRETRAINED", "FINETUNED"):
                self._reve_foundation = wrapper
                print(f"[EmotionClassifier] REVE Foundation loaded (state={state})")
            elif state == "ACCESS_DENIED":
                print("[EmotionClassifier] REVE Foundation: access not yet approved "
                      "— request at huggingface.co/brain-bzh/reve-base")
            elif state == "RANDOM_INIT":
                print("[EmotionClassifier] REVE Foundation: random init (not trained) — "
                      "run train_reve_braindecode.py to fine-tune")
        except Exception:
            pass

    def _try_load_reve(self) -> None:
        """Load the REVE-inspired DETransformer (FACED DE, 30-sec temporal sequences)."""
        try:
            from models.reve_emotion_classifier import REVEEmotionClassifier
            reve = REVEEmotionClassifier()
            if reve.is_available():
                self._reve = reve
        except Exception:
            pass

    def _try_load_tsception(self) -> None:
        """Load TSception weights from models/saved/tsception_emotion.pt.

        TSception is an asymmetry-aware spatial CNN that captures left/right hemispheric
        differences — well-suited for Muse 2's AF7/AF8 frontal pair.
        Falls through silently if weights not found or PyTorch is unavailable.
        If weights file is missing, self._tsception remains None and TSception is
        skipped in the fallback chain; train via ml/training/train_tsception.py.
        """
        import logging
        from pathlib import Path
        pt_path = Path("models/saved/tsception_emotion.pt")
        if not pt_path.exists():
            return
        try:
            from models.tsception import TSceptionClassifier
            self._tsception = TSceptionClassifier(
                model_path=str(pt_path),
                n_classes=3,
                sampling_rate=256.0,
                n_channels=4,
                epoch_sec=4.0,
            )
        except Exception as exc:
            logging.getLogger(__name__).debug(f"TSception load failed: {exc}")
            self._tsception = None

    def _try_load_lgbm_muse_model(self) -> None:
        """Load the Muse-native LightGBM model (80 raw features, no PCA)."""
        import json
        import joblib
        from pathlib import Path
        bench_path = Path("benchmarks/emotion_lgbm_muse_live_benchmark.json")
        model_path = Path("models/saved/emotion_lgbm_muse_live.pkl")
        if not bench_path.exists() or not model_path.exists():
            return
        try:
            bench_data = json.loads(bench_path.read_text())
            acc = float(bench_data.get("accuracy", 0.0))
            if acc < _MIN_MODEL_ACCURACY:
                return
            payload = joblib.load(model_path)
            self.lgbm_muse_model = payload["model"]
            self.lgbm_muse_scaler = payload["scaler"]
            self._lgbm_muse_benchmark = acc
            print(f"[EmotionClassifier] Loaded Muse-native LightGBM (CV acc={acc:.4f})")
        except Exception:
            pass

    def _load_model(self, model_path: str):
        """Load model from file (ONNX or sklearn pkl)."""
        self._benchmark_accuracy = self._read_benchmark()

        if self._benchmark_accuracy < _MIN_MODEL_ACCURACY:
            return

        if model_path.endswith(".onnx"):
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(model_path)
                self.model_type = "onnx"
            except Exception:
                pass
        elif model_path.endswith(".pkl"):
            try:
                import joblib
                data = joblib.load(model_path)
                self.sklearn_model = data["model"]
                self.feature_names = data["feature_names"]
                self.scaler = data.get("scaler")
                self.multichannel_model = bool(data.get("multichannel", False))
                self.model_type = "sklearn"
            except Exception:
                pass

    @staticmethod
    def _read_benchmark() -> float:
        """Read the latest benchmark accuracy from disk."""
        import json
        from pathlib import Path
        bench_path = Path("benchmarks/emotion_classifier_benchmark.json")
        if bench_path.exists():
            try:
                data = json.loads(bench_path.read_text())
                return float(data.get("accuracy", 0))
            except Exception:
                pass
        return 0.0

    @staticmethod
    def _read_deap_benchmark() -> float:
        """Read the DEAP model benchmark accuracy (separate from the GBM classifier benchmark)."""
        import json
        from pathlib import Path
        bench_path = Path("benchmarks/emotion_deap_benchmark.json")
        if bench_path.exists():
            try:
                data = json.loads(bench_path.read_text())
                return float(data.get("accuracy", 0))
            except Exception:
                pass
        return 0.0

    def _extract_muse_live_features(self, eeg: np.ndarray, fs: float) -> np.ndarray:
        """Extract 85-feature vector for the Muse-native LightGBM model.

        Feature layout:
            [0:80]  80 band-power stats: 5 bands × 4 channels × 4 stats (mean,std,med,iqr)
                    Band-major order: Delta_TP9, Delta_AF7, Delta_AF8, Delta_TP10, Theta_TP9, ...
            [80:85] 5 DASM features: mean(AF8_band) - mean(AF7_band) for each of 5 bands
                    AF7=ch1 (left-frontal), AF8=ch2 (right-frontal)
        Matches train_cross_dataset_lgbm.extract_features() exactly.
        """
        WIN_SW = 128   # 0.5-sec sub-window at 256 Hz
        HOP    = 64    # 50 % overlap → ~30 sub-windows per 4-sec epoch

        n_samples = eeg.shape[1]
        powers: list = []

        for start in range(0, n_samples - WIN_SW + 1, HOP):
            end = start + WIN_SW
            sub_bands = []
            for ch in range(4):  # TP9=0, AF7=1, AF8=2, TP10=3
                try:
                    proc = preprocess(eeg[ch, start:end], fs)
                    bp   = extract_band_powers(proc, fs)
                    sub_bands.append([
                        bp.get("delta", 0.0),
                        bp.get("theta", 0.0),
                        bp.get("alpha", 0.0),
                        bp.get("beta",  0.0),
                        bp.get("gamma", 0.0),
                    ])
                except Exception:
                    sub_bands.append([0.0] * 5)
            # sub_bands: (4 channels, 5 bands) → transpose to (5 bands, 4 channels)
            powers.append(np.array(sub_bands, dtype=np.float32).T)

        if len(powers) == 0:
            return np.zeros(85, dtype=np.float32)

        arr = np.array(powers, dtype=np.float32)  # (n_subwins, 5, 4)

        feat: list = []
        # 80 band-power stats
        for band_idx in range(5):
            for ch_idx in range(4):
                vals = arr[:, band_idx, ch_idx]
                vals = vals[np.isfinite(vals)]
                if len(vals) < 2:
                    feat.extend([0.0, 0.0, 0.0, 0.0])
                else:
                    feat.extend([
                        float(np.mean(vals)),
                        float(np.std(vals)),
                        float(np.median(vals)),
                        float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                    ])

        # 5 DASM features: mean(AF8_band) - mean(AF7_band) per band
        # AF7=ch1 (left frontal), AF8=ch2 (right frontal)
        for band_idx in range(5):
            af8 = arr[:, band_idx, 2]
            af8 = af8[np.isfinite(af8)]
            af7 = arr[:, band_idx, 1]
            af7 = af7[np.isfinite(af7)]
            if len(af8) > 0 and len(af7) > 0:
                feat.append(float(np.mean(af8)) - float(np.mean(af7)))
            else:
                feat.append(0.0)

        return np.array(feat, dtype=np.float32)  # 85 features

    def _predict_lgbm_muse(self, eeg: np.ndarray, fs: float, device_type: str = "muse_2") -> Dict:
        """Run Muse-native LightGBM inference (85 features: 80 band-power + 5 DASM, no PCA).

        Expands the 3-class LGBM output (positive/neutral/negative)
        to the 6-class EMOTIONS list using ancillary EEG features.
        """
        # Artifact rejection — freeze EMA on bad epoch, but ONLY if we have prior history.
        # On the very first epoch we must run the model regardless, otherwise _ema_probs
        # gets stuck at uniform 1/6 forever whenever the first epoch contains a blink/spike.
        _artifact_now = bool(np.any(np.abs(eeg) > _ARTIFACT_THRESHOLD_UV))
        if _artifact_now and self._ema_probs is not None:
            return self._build_muse_result(
                int(np.argmax(self._ema_probs)), self._ema_probs,
                eeg, fs, artifact_detected=True, device_type=device_type
            )

        # Euclidean Alignment: accumulate covariance + align when ready.
        # Feeds the aligner with each incoming 4-channel epoch and, once
        # min_epochs (10) are accumulated, whitens the epoch to remove
        # subject-specific covariance structure before feature extraction.
        if eeg.ndim == 2 and eeg.shape[0] == 4:
            eeg_for_features = self._euclidean_aligner.update_and_align(eeg)
        else:
            eeg_for_features = eeg

        # Feature extraction + running normalization + scale
        feat = self._extract_muse_live_features(eeg_for_features, fs)
        # Zero gamma features for consumer dry-electrode devices (gamma = EMG)
        if self._is_consumer_device:
            feat[_GAMMA_FEAT_IDX] = 0.0

        # Feed features to covariate shift detector
        self._feed_shift_detector(feat)

        # Apply running normalization for session drift correction
        rn = _get_running_normalizer()
        if rn is not None and device_type:
            feat = rn.normalize(feat, device_type)
        feat_scaled = self.lgbm_muse_scaler.transform(feat.reshape(1, -1))

        # LGBM predict → (positive=0, neutral=1, negative=2) probabilities
        proba3 = self.lgbm_muse_model.predict_proba(feat_scaled)[0]
        p_pos, p_neu, p_neg = float(proba3[0]), float(proba3[1]), float(proba3[2])

        # SHAP explanation — direct on 85 features (no PCA in this path)
        predicted_3class_idx = int(np.argmax(proba3))
        shap_explanation = self._compute_shap_explanation_lgbm(
            model=self.lgbm_muse_model,
            feat_input=feat_scaled,
            predicted_class_idx=predicted_3class_idx,
            cache_attr="_shap_explainer_muse",
            feature_names=_FEATURE_NAMES_85,
        )

        # Ancillary features for 3→6 expansion
        try:
            proc_af7 = preprocess(eeg[1], fs)
            proc_af8 = preprocess(eeg[2], fs)
            bp7  = extract_band_powers(proc_af7, fs)
            bp8  = extract_band_powers(proc_af8, fs)
            a7   = max(bp7.get("alpha", 1e-6), 1e-6)
            a8   = max(bp8.get("alpha", 1e-6), 1e-6)
            alpha = (a7 + a8) / 2
            beta  = (max(bp7.get("beta",  1e-6), 1e-6)
                     + max(bp8.get("beta",  1e-6), 1e-6)) / 2
            theta = (max(bp7.get("theta", 1e-6), 1e-6)
                     + max(bp8.get("theta", 1e-6), 1e-6)) / 2
            delta = (max(bp7.get("delta", 1e-6), 1e-6)
                     + max(bp8.get("delta", 1e-6), 1e-6)) / 2
            faa   = float(np.log(a8) - np.log(a7))
        except Exception:
            alpha = beta = theta = delta = 0.1
            faa = 0.0

        beta_alpha = beta / max(alpha, 1e-6)
        arousal = float(np.clip(
            0.45 * beta  / max(beta + alpha, 1e-6)
            + 0.30 * (1 - alpha / max(alpha + beta + theta, 1e-6))
            + 0.25 * (1 - delta / max(delta + beta, 1e-6)),
            0, 1))
        faa_val = float(np.clip(np.tanh(faa * 2.0), -1, 1))
        valence_for_expand = float(np.clip(
            0.65 * np.tanh((alpha / max(beta, 1e-6) - 0.7) * 2.0)
            + 0.35 * np.tanh((alpha - 0.15) * 4),
            -1, 1,
        ))
        # Blend FAA into valence for expansion (same weights as _build_muse_result)
        valence_for_expand = float(np.clip(
            0.50 * valence_for_expand + 0.50 * faa_val, -1, 1,
        ))

        # 3→6 class expansion — soft distance-based blending via Russell circumplex
        probs6 = _expand_3class_to_6(p_pos, p_neu, p_neg, valence_for_expand, arousal)

        # EMA smoothing
        if self._ema_probs is None:
            self._ema_probs = probs6.copy()
        else:
            self._ema_probs = (self._ema_alpha * probs6
                               + (1 - self._ema_alpha) * self._ema_probs)

        smoothed = self._ema_probs
        emotion_idx = int(np.argmax(smoothed))
        return self._build_muse_result(emotion_idx, smoothed, eeg, fs,
                                       artifact_detected=_artifact_now,
                                       device_type=device_type,
                                       explanation=shap_explanation)

    def _build_muse_result(self, emotion_idx: int, smoothed: np.ndarray,
                           eeg: np.ndarray, fs: float,
                           artifact_detected: bool,
                           device_type: str = "muse_2",
                           explanation: Optional[list] = None) -> Dict:
        """Build the standard return dict for the Muse-native LGBM path."""
        n_ch = eeg.shape[0] if eeg.ndim == 2 else 1
        cmap = get_channel_map(device_type, n_ch)
        lf   = cmap["left_frontal"]   # left-frontal channel index
        rf   = cmap["right_frontal"]  # right-frontal channel index

        try:
            proc  = preprocess(eeg[lf], fs)  # left-frontal channel
            bp    = extract_band_powers(proc, fs)
            de    = differential_entropy(proc, fs)
            alpha     = max(bp.get("alpha",    1e-6), 1e-6)
            beta      = max(bp.get("beta",     1e-6), 1e-6)
            theta     = max(bp.get("theta",    1e-6), 1e-6)
            delta     = max(bp.get("delta",    1e-6), 1e-6)
            high_beta = max(bp.get("high_beta",1e-6), 1e-6)
            low_beta  = max(bp.get("low_beta", 1e-6), 1e-6)
            gamma     = max(bp.get("gamma",    1e-6), 1e-6)
            bands = {k: float(v) for k, v in bp.items()
                     if k in ("delta","theta","alpha","beta","low_beta","high_beta","gamma")}
        except Exception:
            alpha = beta = theta = delta = high_beta = low_beta = gamma = 0.1
            bands = {}
            de = {}

        try:
            dasm_rasm = compute_dasm_rasm(eeg, fs, left_ch=lf, right_ch=rf)
        except Exception:
            dasm_rasm = {}
        try:
            fmt = compute_frontal_midline_theta(eeg[lf], fs)
        except Exception:
            fmt = {}
        try:
            asym = compute_frontal_asymmetry(eeg, fs, left_ch=lf, right_ch=rf)
            faa_valence = float(asym.get("asymmetry_valence", 0.0))
        except Exception:
            faa_valence = 0.0
        # Frontal alpha coherence (AF7-AF8)
        try:
            frontal_pair = eeg[np.array([lf, rf])]
            _frontal_coh = float(np.clip(compute_coherence(frontal_pair, fs, "alpha"), 0.0, 1.0))
        except Exception:
            _frontal_coh = 0.0

        beta_alpha = beta / max(alpha, 1e-6)
        dasm_alpha_val = float(dasm_rasm.get("dasm_alpha", 0.0)) * 0.5
        valence_abr = float(np.clip(
            0.65 * np.tanh((alpha / beta - 0.7) * 2.0)
            + 0.35 * np.tanh((alpha - 0.15) * 4), -1, 1))
        valence = float(np.clip(
            0.40 * valence_abr + 0.35 * faa_valence + 0.25 * dasm_alpha_val, -1, 1))
        arousal = float(np.clip(
            0.45 * beta  / max(beta + alpha, 1e-6)
            + 0.30 * (1 - alpha / max(alpha + beta + theta, 1e-6))
            + 0.25 * (1 - delta / max(delta + beta, 1e-6)),
            0, 1))

        dasm_beta_stress = float(dasm_rasm.get("dasm_beta", 0.0)) * 0.5
        theta_beta_ratio = theta / max(beta, 1e-6)
        dominance = _compute_dominance(beta_alpha, theta_beta_ratio)
        stress_index = float(np.clip(
            0.40 * min(1, beta_alpha * 0.3)
            + 0.25 * max(0, 1 - alpha * 2.5)
            + 0.25 * min(1, high_beta * 4)
            + 0.10 * dasm_beta_stress, 0, 1))
        focus_index = float(np.clip(
            0.45 * min(1, beta * 3.5)
            + 0.40 * max(0, 1 - theta_beta_ratio * 0.40)
            + 0.15 * min(1, low_beta * 5), 0, 1))
        relaxation_index = float(np.clip(
            0.50 * min(1, alpha * 2.5)
            + 0.30 * max(0, 1 - beta_alpha * 0.3)
            + 0.20 * min(1, theta * 1.5), 0, 1))
        anger_index = float(np.clip(
            0.40 * min(1, max(0, beta_alpha - 1.0) * 0.5)
            + 0.30 * max(0, 1 - alpha * 5)
            + 0.20 * min(1, high_beta * 3)
            + 0.10 * min(1, gamma * 3), 0, 1))
        fear_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha - 1.5) * 0.6)
            + 0.30 * min(1, high_beta * 5)
            + 0.25 * max(0, 1 - alpha * 5)
            + 0.10 * max(0, arousal - 0.45), 0, 1))

        # EMA smooth continuous indices to reduce frame-to-frame jitter.
        # Same _smooth_index() used in _predict_features() — ensures all model
        # paths (LGBM, EEGNet via _build_muse_result) produce temporally stable
        # dashboard readings.  Without this, valence/arousal/stress/focus jump
        # on every 2-second epoch hop.
        # SKIP smoothing on artifact epochs — the band powers are garbage
        # (from a blink/EMG burst), so blending them into the EMA would
        # contaminate the running average.  Instead, return frozen values.
        if not artifact_detected:
            valence          = self._smooth_index("_ema_valence", valence)
            arousal          = self._smooth_index("_ema_arousal", arousal)
            stress_index     = self._smooth_index("_ema_stress", stress_index)
            focus_index      = self._smooth_index("_ema_focus", focus_index)
            relaxation_index = self._smooth_index("_ema_relaxation", relaxation_index)
            anger_index      = self._smooth_index("_ema_anger", anger_index)
            fear_index       = self._smooth_index("_ema_fear", fear_index)
        else:
            # Return frozen EMA values when available, else keep raw
            valence          = self._ema_valence if self._ema_valence is not None else valence
            arousal          = self._ema_arousal if self._ema_arousal is not None else arousal
            stress_index     = self._ema_stress if self._ema_stress is not None else stress_index
            focus_index      = self._ema_focus if self._ema_focus is not None else focus_index
            relaxation_index = self._ema_relaxation if self._ema_relaxation is not None else relaxation_index
            anger_index      = self._ema_anger if self._ema_anger is not None else anger_index
            fear_index       = self._ema_fear if self._ema_fear is not None else fear_index

        granular_emotions = map_vad_to_granular_emotions(valence, arousal, dominance)

        # Band Hjorth mobility ratio (frontal/temporal)
        try:
            _lt = cmap.get("left_temporal", 0)
            _rt = cmap.get("right_temporal", n_ch - 1)
            _band_hjorth = compute_hjorth_mobility_ratio(
                eeg, fs, frontal_chs=[lf, rf], temporal_chs=[_lt, _rt],
                bands=["beta"],
            )
        except Exception:
            _band_hjorth = {}

        top_conf    = float(np.max(smoothed))
        emotion_lbl = EMOTIONS[emotion_idx]
        return {
            "emotion":               emotion_lbl,
            "emotion_index":         emotion_idx,
            "confidence":            top_conf,
            "probabilities":         {EMOTIONS[i]: float(p) for i, p in enumerate(smoothed)},
            "valence":               valence,
            "arousal":               arousal,
            "dominance":             dominance,
            "granular_emotions":     granular_emotions,
            "stress_index":          stress_index,
            "focus_index":           focus_index,
            "relaxation_index":      relaxation_index,
            "anger_index":           anger_index,
            "fear_index":            fear_index,
            "band_powers":           bands,
            "differential_entropy":  de,
            "dasm_rasm":             dasm_rasm,
            "frontal_midline_theta": fmt,
            "frontal_alpha_coherence": _frontal_coh,
            "band_hjorth":           _band_hjorth,
            "artifact_detected":     artifact_detected,
            "model_type":            "lgbm-muse",
            "explanation":           explanation if explanation else [],
        }

    def _predict_mega_lgbm(self, eeg: np.ndarray, fs: float, device_type: str = "muse_2",
                            user_id: str = "default") -> Dict:
        """Run mega cross-dataset LGBM inference (85 raw → PCA 80 → 3-class → 6-class).

        Reuses _extract_muse_live_features() and _build_muse_result() from the
        Muse-native LGBM path for identical output format.
        """
        # Same logic as _predict_lgbm_muse: only freeze EMA if we have prior history.
        _artifact_now = bool(np.any(np.abs(eeg) > _ARTIFACT_THRESHOLD_UV))
        if _artifact_now and self._ema_probs is not None:
            return self._build_muse_result(
                int(np.argmax(self._ema_probs)), self._ema_probs,
                eeg, fs, artifact_detected=True, device_type=device_type
            )

        feat = self._extract_muse_live_features(eeg, fs)   # 85-dim
        if self._is_consumer_device:
            feat[_GAMMA_FEAT_IDX] = 0.0

        # Feed features to covariate shift detector.
        # During first ~1 min (before baseline is collected), features build
        # the reference distribution. After that, features go to the live window.
        self._feed_shift_detector(feat)

        # Apply running normalization for session drift correction
        rn = _get_running_normalizer()
        if rn is not None and user_id:
            feat = rn.normalize(feat, user_id)

        # Global scaler → PCA → LGBM
        feat_sc  = self.mega_lgbm_scaler.transform(feat.reshape(1, -1))
        feat_pca = self.mega_lgbm_pca.transform(feat_sc)
        proba3   = self.mega_lgbm_model.predict_proba(feat_pca)[0]
        p_pos, p_neu, p_neg = float(proba3[0]), float(proba3[1]), float(proba3[2])

        # SHAP explanation — projects PCA-space values back to original 85 features
        predicted_3class_idx = int(np.argmax(proba3))
        shap_explanation = self._compute_shap_explanation_lgbm(
            model=self.mega_lgbm_model,
            feat_input=feat_pca,
            predicted_class_idx=predicted_3class_idx,
            cache_attr="_shap_explainer_mega",
            feature_names=_FEATURE_NAMES_85,
            pca=self.mega_lgbm_pca,
            feat_pre_pca=feat_sc,
        )

        # Ancillary features for 3→6 expansion (identical to _predict_lgbm_muse)
        cmap = get_channel_map(device_type, eeg.shape[0])
        lf   = cmap["left_frontal"]   # left-frontal electrode index
        rf   = cmap["right_frontal"]  # right-frontal electrode index
        try:
            proc_af7 = preprocess(eeg[lf], fs)
            proc_af8 = preprocess(eeg[rf], fs)
            bp7  = extract_band_powers(proc_af7, fs)
            bp8  = extract_band_powers(proc_af8, fs)
            a7   = max(bp7.get("alpha", 1e-6), 1e-6)
            a8   = max(bp8.get("alpha", 1e-6), 1e-6)
            alpha = (a7 + a8) / 2
            beta  = (max(bp7.get("beta",  1e-6), 1e-6) + max(bp8.get("beta",  1e-6), 1e-6)) / 2
            theta = (max(bp7.get("theta", 1e-6), 1e-6) + max(bp8.get("theta", 1e-6), 1e-6)) / 2
            delta = (max(bp7.get("delta", 1e-6), 1e-6) + max(bp8.get("delta", 1e-6), 1e-6)) / 2
            faa   = float(np.log(a8) - np.log(a7))
        except Exception:
            alpha = beta = theta = delta = 0.1
            faa = 0.0

        beta_alpha = beta / max(alpha, 1e-6)
        arousal    = float(np.clip(
            0.45 * beta / max(beta + alpha, 1e-6)
            + 0.30 * (1 - alpha / max(alpha + beta + theta, 1e-6))
            + 0.25 * (1 - delta / max(delta + beta, 1e-6)), 0, 1))
        faa_val = float(np.clip(np.tanh(faa * 2.0), -1, 1))
        valence_for_expand = float(np.clip(
            0.65 * np.tanh((alpha / max(beta, 1e-6) - 0.7) * 2.0)
            + 0.35 * np.tanh((alpha - 0.15) * 4),
            -1, 1,
        ))
        # Blend FAA into valence for expansion (same weights as _build_muse_result)
        valence_for_expand = float(np.clip(
            0.50 * valence_for_expand + 0.50 * faa_val, -1, 1,
        ))

        # 3→6 class expansion — soft distance-based blending via Russell circumplex
        probs6 = _expand_3class_to_6(p_pos, p_neu, p_neg, valence_for_expand, arousal)

        if self._ema_probs is None:
            self._ema_probs = probs6.copy()
        else:
            self._ema_probs = (self._ema_alpha * probs6
                               + (1 - self._ema_alpha) * self._ema_probs)
        smoothed    = self._ema_probs
        emotion_idx = int(np.argmax(smoothed))
        result = self._build_muse_result(emotion_idx, smoothed, eeg, fs,
                                         artifact_detected=_artifact_now,
                                         device_type=device_type,
                                         explanation=shap_explanation)
        result["model_type"] = "mega-lgbm-pca"
        return result

    def predict(self, eeg: np.ndarray, fs: float = 256.0, device_type: str = "muse_2",
                user_id: str = "default") -> Dict:
        """Classify emotion from EEG signal.

        Args:
            eeg:         1D (single channel) or 2D (n_channels, n_samples) array.
            fs:          Sampling frequency.
            device_type: Device name used to select correct left/right frontal
                         channel indices for FAA/DASM computation (see
                         processing/channel_maps.py for the full registry).
            user_id:     Per-user identifier for rolling z-score normalization.
                         Used by RunningNormalizer to maintain separate buffers
                         per user and correct within-session EEG drift.

        Returns:
            Dict with emotion prediction fields.  Always includes:
            - ``prediction_stability`` (float, 0-1): cosine similarity stability
              of recent probability vectors.  1.0 = very stable, <0.5 = rapid flipping.
            - ``stability_adjusted_confidence`` (float): confidence penalized when
              predictions are unstable (rapid emotion flipping).
        """
        result = self._predict_core(eeg, fs, device_type, user_id)
        return self._apply_stability_tracking(result)

    def _feed_shift_detector(self, feat: np.ndarray) -> None:
        """Feed an 85-dim feature vector to the covariate shift detector.

        During the first ~1 min of a session (before baseline is complete),
        features build the reference distribution. After that, features go
        to the live sliding window for comparison.
        """
        if not self._shift_baseline_complete:
            self._shift_detector.add_reference(feat)
            if self._shift_detector.is_ready:
                self._shift_baseline_complete = True
        else:
            self._shift_detector.add_live(feat)

    def get_shift_status(self) -> Dict:
        """Get current covariate shift detection status.

        Returns dict with shift_detected, confidence_penalty, fraction_shifted,
        recommendation, and buffer counts.
        """
        status = self._shift_detector.get_status()
        if self._shift_detector.is_ready and self._shift_detector.live_count >= self._shift_detector.min_live:
            detection = self._shift_detector.detect()
            status.update(detection)
        else:
            status["shift_detected"] = False
            status["confidence_penalty"] = 1.0
            status["status"] = "collecting"
        status["baseline_complete"] = self._shift_baseline_complete
        return status

    def _apply_stability_tracking(self, result: Dict) -> Dict:
        """Feed probability vector to stability tracker and augment result dict.

        Adds ``prediction_stability`` and ``stability_adjusted_confidence``
        fields to the result.  Operates on every output path via ``predict()``.
        On artifact-frozen epochs the tracker is not updated (frozen probs are
        identical to previous, so stability would trivially be 1.0 — no info).
        """
        probs_dict = result.get("probabilities", {})
        if probs_dict:
            # Build an aligned probability array in EMOTIONS order
            probs_arr = np.array(
                [float(probs_dict.get(e, 0.0)) for e in EMOTIONS],
                dtype=float,
            )
            # Skip stability update on artifact-frozen epochs
            if not result.get("artifact_detected", False):
                self._stability_tracker.update(probs_arr)

        stability = self._stability_tracker.stability
        raw_conf = float(result.get("confidence", 0.0))
        adjusted_conf = self._stability_tracker.adjust_confidence(raw_conf)

        result["prediction_stability"] = round(stability, 4)
        result["stability_adjusted_confidence"] = round(adjusted_conf, 4)

        # ── Covariate shift status ────────────────────────────────────────────
        # When shift is detected, include a warning and penalty in the output
        # so downstream consumers (dashboard, API) can surface a recalibration
        # prompt to the user.
        if self._shift_baseline_complete and self._shift_detector.live_count >= self._shift_detector.min_live:
            shift_result = self._shift_detector.detect()
            if shift_result["shift_detected"]:
                result["covariate_shift"] = {
                    "shift_detected": True,
                    "confidence_penalty": shift_result["confidence_penalty"],
                    "fraction_shifted": shift_result["fraction_shifted"],
                    "mean_ks_statistic": shift_result["mean_ks_statistic"],
                    "recommendation": shift_result.get("recommendation"),
                }
                # Apply shift penalty to the stability-adjusted confidence
                result["stability_adjusted_confidence"] = round(
                    adjusted_conf * shift_result["confidence_penalty"], 4
                )

        # ── Valence trajectory (mood trend) ──────────────────────────────────
        # Append EMA-smoothed valence to history (skip artifact-frozen epochs
        # to avoid contaminating the trend with stale repeated values).
        if not result.get("artifact_detected", False):
            valence = result.get("valence")
            if valence is not None:
                self._valence_history.append(float(valence))
        result.update(self._compute_trajectory())

        return result

    def _predict_core(self, eeg: np.ndarray, fs: float = 256.0,
                      device_type: str = "muse_2",
                      user_id: str = "default") -> Dict:
        """Internal prediction dispatch — selects the best available model.

        All model paths return through here; ``predict()`` wraps the output
        with stability tracking.
        """
        # Centralized artifact check — runs before every model path so no path bypasses it.
        # TSception and EEGNet return early without updating _ema_probs, so we must check
        # here (after TSception seeds EMA on its first run) rather than relying on each
        # model to have its own check. On the very first epoch we must run the model even
        # with an artifact so that _ema_probs gets seeded for future freeze decisions.
        _artifact_now = bool(np.any(np.abs(eeg) > _ARTIFACT_THRESHOLD_UV))
        if _artifact_now and self._ema_probs is not None:
            # Freeze EMA — return prior state with artifact flag.
            smoothed = self._ema_probs / (self._ema_probs.sum() + 1e-10)
            n = len(smoothed)
            # Pad/truncate to 6-class if EMA comes from a 3-class model (TSception).
            if n < 6:
                smoothed6 = np.zeros(6)
                smoothed6[:n] = smoothed
                smoothed6 /= smoothed6.sum() + 1e-10
            else:
                smoothed6 = smoothed[:6]
            emotion_idx = int(np.argmax(smoothed6))
            result = self._build_muse_result(
                emotion_idx, smoothed6, eeg, fs,
                artifact_detected=True, device_type=device_type,
            )
            result["explanation"] = []
            return result

        # Per-user fine-tuned model — highest priority after artifact check.
        # Loaded from user_models/{user_id}/ if a retrained model exists.
        # Falls through to generic models on any error or missing model.
        if user_id and user_id != "default" and eeg.ndim == 2 and eeg.shape[0] >= 4:
            user_result = self._try_predict_user_model(eeg, fs, user_id, device_type)
            if user_result is not None:
                return user_result

        # REVE Foundation (brain-bzh/reve-base) — highest priority when HF access is granted.
        # Pre-trained on 60K+ hours, handles arbitrary channel layouts via 4D positional encoding.
        # Requires >= 4 sec of EEG (minimum useful input for embedding extraction).
        if (self._reve_foundation is not None
                and eeg.ndim == 2
                and eeg.shape[0] >= 4
                and eeg.shape[1] >= int(fs * 4)):
            try:
                return self._ensure_explanation(self._reve_foundation.predict(eeg, fs))
            except Exception:
                pass  # fall through on any inference error

        # REVE DETransformer — temporal DE features over 30-sec epochs.
        # Preferred over EEGNet for long sessions: REVE models temporal dynamics
        # across the whole 30-second sequence; EEGNet treats it as a static epoch.
        # Falls through to EEGNet / mega-LGBM for shorter windows.
        if (self._reve is not None
                and eeg.ndim == 2
                and eeg.shape[0] >= 4
                and eeg.shape[1] >= int(fs * 30)):
            try:
                return self._ensure_explanation(self._reve.predict(eeg, fs, device_type))
            except Exception:
                pass  # fall through on any inference error

        # EEGNet — device-agnostic CNN on raw EEG. Best for short-to-medium epochs.
        # Works on any channel count (4ch Muse, 8ch OpenBCI, 16ch Cyton+Daisy).
        # Only activates if ml/models/saved/eegnet_emotion_{n_ch}ch.pt exists
        # AND its benchmark accuracy >= 60%.
        #
        # ENSEMBLE: When EEGNet is available, run BOTH EEGNet and feature-based
        # heuristics and average their probability outputs. Diverse classifiers
        # (CNN on raw waveforms + neuroscience heuristics on band power ratios)
        # typically beat either alone by 3-7% — a classic ensemble win.
        # Weights: 0.6 EEGNet (higher CV accuracy) + 0.4 heuristic.
        if (self._eegnet is not None and eeg.ndim == 2
                and self._eegnet.is_available(eeg.shape[0], min_accuracy=_MIN_MODEL_ACCURACY)):
            return self._predict_ensemble_eegnet_heuristic(eeg, fs, device_type)

        # Mega cross-dataset LGBM with global PCA — third priority
        if (self.mega_lgbm_model is not None and eeg.ndim == 2 and eeg.shape[0] >= 4
                and self._mega_lgbm_benchmark >= _MIN_MODEL_ACCURACY):
            return self._predict_mega_lgbm(eeg, fs, device_type, user_id=user_id)
        # Muse-native LightGBM (85 raw features, no PCA)
        if (self.lgbm_muse_model is not None and eeg.ndim == 2 and eeg.shape[0] >= 4
                and self._lgbm_muse_benchmark >= _MIN_MODEL_ACCURACY):
            return self._predict_lgbm_muse(eeg, fs, device_type)
        # Multichannel DEAP model (requires exactly 4 Muse 2 channels: AF7, AF8, TP9, TP10)
        # Gated on accuracy — DEAP model at ~51% has severe class-imbalance bias toward "sad"
        # (11,165 sad samples vs 1,769 focused), so skip it unless it meets the 60% threshold.
        if (self.model_type == "sklearn-deap" and eeg.ndim == 2 and eeg.shape[0] >= 4
                and self._benchmark_accuracy >= _MIN_MODEL_ACCURACY):
            return self._predict_multichannel(eeg, fs, device_type)
        if self.onnx_session is not None and self._benchmark_accuracy >= _MIN_MODEL_ACCURACY:
            return self._predict_onnx(eeg if eeg.ndim == 1 else eeg[0], fs)
        if self.sklearn_model is not None and self._benchmark_accuracy >= _MIN_MODEL_ACCURACY:
            # Multichannel sklearn models (trained with DASM/RASM/FAA features) need the
            # full (n_channels, n_samples) array; single-channel models need eeg[0].
            if self.multichannel_model and eeg.ndim == 2:
                return self._predict_sklearn(eeg, fs)
            return self._predict_sklearn(eeg if eeg.ndim == 1 else eeg[0], fs)
        # TSception fallback (69% CV) — asymmetry-aware spatial CNN.
        # Requires >= 4-second epoch (1024 samples @ 256 Hz) for meaningful temporal convolution.
        # Output is 3-class (negative/neutral/positive); passes through for downstream EMA smoothing.
        if (self._tsception is not None
                and eeg.ndim == 2
                and eeg.shape[1] >= 1024):
            try:
                result = self._tsception.predict(eeg, fs=fs)
                result["model_type"] = "tsception"
                # Seed _ema_probs so the centralized artifact check can freeze on
                # subsequent high-amplitude epochs (TSception has no own artifact gate).
                probs_dict = result.get("probabilities", {})
                if probs_dict:
                    probs_arr = np.array(list(probs_dict.values()), dtype=float)
                    if self._ema_probs is None or len(self._ema_probs) != len(probs_arr):
                        self._ema_probs = probs_arr.copy()
                    else:
                        self._ema_probs = (self._ema_alpha * probs_arr
                                           + (1 - self._ema_alpha) * self._ema_probs)
                return self._ensure_explanation(result)
            except Exception as exc:  # noqa: BLE001
                logging.warning("TSception inference failed, falling through: %s", exc)

        return self._predict_features(eeg, fs, device_type)  # pass full multichannel array for FAA

    # ────────────────────────────────────────────────────────────────
    # Multichannel DEAP-trained model (primary for Muse 2)
    # ────────────────────────────────────────────────────────────────

    def _predict_multichannel(self, channels: np.ndarray, fs: float, device_type: str = "muse_2") -> Dict:
        """Predict using the DEAP-trained multichannel model."""
        from training.train_deap_muse import extract_multichannel_features

        features = extract_multichannel_features(channels, fs)
        feat_vec = np.array([features.get(k, 0.0) for k in self.feature_names]).reshape(1, -1)

        if self.scaler is not None:
            feat_vec = self.scaler.transform(feat_vec)

        feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)

        probs = self.sklearn_model.predict_proba(feat_vec)[0]
        emotion_idx = int(np.argmax(probs))

        # EMA smoothing
        if self._ema_probs is None:
            self._ema_probs = probs
        else:
            self._ema_probs = self._ema_alpha * probs + (1 - self._ema_alpha) * self._ema_probs

        smoothed = self._ema_probs / (self._ema_probs.sum() + 1e-10)
        emotion_idx = int(np.argmax(smoothed))

        # Extract band powers from first channel for extra metrics
        processed = preprocess(channels[0], fs)
        bands = extract_band_powers(processed, fs)
        de = differential_entropy(processed, fs)

        alpha_raw     = bands.get("alpha",     0)
        beta_raw      = bands.get("beta",      0)
        theta_raw     = bands.get("theta",     0)
        gamma_raw     = bands.get("gamma",     0)
        delta_raw     = bands.get("delta",     0)
        high_beta_raw = bands.get("high_beta", 0)
        low_beta_raw  = bands.get("low_beta",  0)

        total_power = alpha_raw + beta_raw + theta_raw + gamma_raw + delta_raw
        if total_power < 1e-6:
            total_power = 1.0
        alpha     = alpha_raw     / total_power
        beta      = beta_raw      / total_power
        theta     = theta_raw     / total_power
        gamma     = gamma_raw     / total_power
        high_beta = high_beta_raw / total_power
        low_beta  = low_beta_raw  / total_power
        delta     = delta_raw     / total_power

        alpha_beta_ratio = alpha / max(beta, 1e-10)
        beta_alpha_ratio = beta / max(alpha, 1e-10)
        theta_beta_ratio = theta / max(beta, 1e-10)

        # Frontal alpha asymmetry (Davidson 1992): richer valence signal
        _cmap = get_channel_map(device_type, channels.shape[0])
        asymmetry = compute_frontal_asymmetry(
            channels, fs,
            left_ch=_cmap["left_frontal"], right_ch=_cmap["right_frontal"]
        )
        asym_valence = asymmetry.get("asymmetry_valence", 0.0)  # -1 to +1

        # Blend alpha-beta ratio valence (60%) with frontal asymmetry (40%)
        # Asymmetry is a stronger valence signal when 4 channels are available.
        # Neutral baseline 0.8 (eyes-open resting: beta slightly > alpha for Muse).
        # Gamma removed from valence — it's typically < 0.05 on consumer EEG and
        # would constantly subtract from valence, falsely pushing toward negative.
        valence_abr = float(np.tanh((alpha_beta_ratio - 0.8) * 1.5) * 0.6)
        valence = float(np.clip(0.60 * valence_abr + 0.40 * asym_valence, -1, 1))
        # Arousal without gamma: frontal gamma at AF7/AF8 is frontalis EMG, not neural.
        arousal = float(np.clip(
            0.45 * beta / max(beta + alpha, 1e-10)
            + 0.30 * (1.0 - alpha / max(alpha + beta + theta, 1e-10))
            + 0.25 * (1.0 - delta / max(delta + beta, 1e-10)),
            0, 1
        ))
        dominance = _compute_dominance(beta_alpha_ratio, theta_beta_ratio)

        stress_index = float(np.clip(
            0.40 * min(1, beta_alpha_ratio * 0.3) + 0.25 * max(0, 1 - alpha * 2.5)
            + 0.20 * min(1, high_beta * 4) + 0.15 * min(1, gamma * 2), 0, 1))
        focus_index = float(np.clip(
            0.40 * min(1, beta * 3.5) + 0.35 * max(0, 1 - theta_beta_ratio * 0.40)
            + 0.15 * min(1, low_beta * 5) + 0.10 * min(1, gamma * 2), 0, 1))
        relaxation_index = float(np.clip(
            0.50 * min(1, alpha * 2.5) + 0.30 * max(0, 1 - beta_alpha_ratio * 0.3)
            + 0.20 * min(1, theta * 1.5), 0, 1))
        anger_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.0) * 0.5) + 0.25 * min(1, gamma * 3)
            + 0.25 * max(0, 1 - alpha * 5) + 0.15 * min(1, high_beta * 3), 0, 1))
        fear_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.5) * 0.6) + 0.30 * min(1, high_beta * 5)
            + 0.25 * max(0, 1 - alpha * 5) + 0.10 * max(0, arousal - 0.45), 0, 1))

        # EMA smooth continuous indices (same as _build_muse_result and _predict_features)
        valence          = self._smooth_index("_ema_valence", valence)
        arousal          = self._smooth_index("_ema_arousal", arousal)
        stress_index     = self._smooth_index("_ema_stress", stress_index)
        focus_index      = self._smooth_index("_ema_focus", focus_index)
        relaxation_index = self._smooth_index("_ema_relaxation", relaxation_index)
        anger_index      = self._smooth_index("_ema_anger", anger_index)
        fear_index       = self._smooth_index("_ema_fear", fear_index)

        granular_emotions = map_vad_to_granular_emotions(valence, arousal, dominance)

        return {
            "emotion": EMOTIONS[emotion_idx],
            "emotion_index": emotion_idx,
            "confidence": float(smoothed[emotion_idx]),
            "probabilities": {EMOTIONS[i]: float(p) for i, p in enumerate(smoothed)},
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "granular_emotions": granular_emotions,
            "stress_index": stress_index,
            "focus_index": focus_index,
            "relaxation_index": relaxation_index,
            "anger_index": anger_index,
            "fear_index": fear_index,
            "band_powers": bands,
            "differential_entropy": de,
            "explanation": [],
        }

    # ────────────────────────────────────────────────────────────────
    # Ensemble: EEGNet + feature-based heuristics
    # ────────────────────────────────────────────────────────────────

    # Ensemble weights: EEGNet gets higher weight (higher CV accuracy: 85%)
    # vs feature-based heuristics (estimated 55-65% real-world accuracy).
    _ENSEMBLE_W_EEGNET = 0.6
    _ENSEMBLE_W_HEURISTIC = 0.4

    # Number of TTA augmented views for EEGNet inference.
    # 3 is a good trade-off: +1-3% accuracy for ~3x inference cost.
    # Set to 0 to disable TTA and use only the original input.
    _TTA_N_AUGMENTATIONS = 3

    def _predict_ensemble_eegnet_heuristic(
        self, eeg: np.ndarray, fs: float, device_type: str = "muse_2"
    ) -> Dict:
        """Ensemble of EEGNet CNN + feature-based heuristics with TTA.

        Runs both classifiers on the same epoch, averages their 6-class
        probability vectors (weighted 0.6 EEGNet + 0.4 heuristic), then
        returns the feature-based result dict (which has all the detailed
        fields: band_powers, DE, DASM, etc.) with the ensemble-averaged
        probabilities and emotion label.

        Test-Time Augmentation (TTA): Before running EEGNet, creates N
        augmented copies of the input (noise, time shift, amplitude scale)
        and runs each through the model. The EEGNet probabilities are the
        average across the original + all augmented views. This reduces
        variance from electrode noise and epoch boundary jitter by 1-3%
        accuracy with no retraining required.

        Also logs per-prediction model contributions:
        - ``eegnet_contribution``: how much the final prediction relied on EEGNet
        - ``heuristic_contribution``: how much it relied on feature heuristics
        These are based on which sub-model's top class matches the ensemble winner.

        Why this works: EEGNet learns temporal and spatial patterns from
        raw waveforms. Feature-based heuristics use domain-specific band
        power ratios (FAA, DASM, ABR). These are fundamentally different
        representations, so their errors are weakly correlated -- the ideal
        condition for ensemble gains.
        """
        from models.eegnet import _tta_augment_eeg

        # 1. Get EEGNet prediction with MC Dropout uncertainty + TTA
        # Run on original input first (this also updates EEGNet's internal EMA)
        eegnet_result = self._eegnet.predict_with_uncertainty(eeg, fs, n_forward=5)
        eegnet_probs_dict = eegnet_result.get("probabilities", {})

        # TTA: average EEGNet probabilities across augmented views
        if self._TTA_N_AUGMENTATIONS > 0:
            augmented_views = _tta_augment_eeg(
                eeg, n_augmentations=self._TTA_N_AUGMENTATIONS
            )
            tta_probs_list = [
                np.array([float(eegnet_probs_dict.get(e, 0.0)) for e in EMOTIONS],
                         dtype=float)
            ]
            for aug_eeg in augmented_views:
                try:
                    aug_result = self._eegnet.predict_with_uncertainty(
                        aug_eeg, fs, n_forward=3  # fewer MC passes for speed
                    )
                    aug_probs = np.array(
                        [float(aug_result.get("probabilities", {}).get(e, 0.0))
                         for e in EMOTIONS],
                        dtype=float,
                    )
                    tta_probs_list.append(aug_probs)
                except Exception:
                    pass  # skip failed augmentation, use remaining
            # Average across all views (original + augmented)
            if len(tta_probs_list) > 1:
                tta_avg = np.mean(tta_probs_list, axis=0)
                tta_sum = tta_avg.sum()
                if tta_sum > 1e-10:
                    tta_avg = tta_avg / tta_sum
                # Override eegnet_probs_dict with TTA-averaged probabilities
                eegnet_probs_dict = {
                    EMOTIONS[i]: float(tta_avg[i]) for i in range(len(EMOTIONS))
                }

        # 2. Get feature-based prediction.
        # Save and restore _ema_probs to prevent double-EMA: _predict_features
        # applies its own EMA internally, but we want to apply EMA once on the
        # ensemble-averaged probabilities, not on the heuristic probs alone.
        saved_ema_probs = self._ema_probs
        heuristic_result = self._predict_features(eeg, fs, device_type)
        self._ema_probs = saved_ema_probs  # restore pre-heuristic state
        heuristic_probs_dict = heuristic_result.get("probabilities", {})

        # 3. Convert both to aligned numpy arrays (6 classes, same order)
        eegnet_probs = np.array(
            [float(eegnet_probs_dict.get(e, 0.0)) for e in EMOTIONS],
            dtype=float,
        )
        heuristic_probs = np.array(
            [float(heuristic_probs_dict.get(e, 0.0)) for e in EMOTIONS],
            dtype=float,
        )

        # 4. Adaptive weighted average — signal quality drives EEGNet weight.
        # EEGNet (CNN on raw waveforms) degrades faster than heuristics (band-power
        # ratios) on noisy signals.  When quality is poor, shift weight to heuristic.
        _epoch_quality = self._compute_epoch_quality(eeg, fs)
        w_e, w_h = self._adaptive_ensemble_weights(_epoch_quality)
        ensemble_probs = w_e * eegnet_probs + w_h * heuristic_probs

        # Re-normalize to sum to 1
        prob_sum = ensemble_probs.sum()
        if prob_sum > 1e-10:
            ensemble_probs = ensemble_probs / prob_sum

        # 5. Update _ema_probs with the ensemble probabilities
        # (overrides what _predict_features set, so EMA tracks the ensemble)
        if self._ema_probs is None:
            self._ema_probs = ensemble_probs.copy()
        else:
            self._ema_probs = (
                self._ema_alpha * ensemble_probs
                + (1 - self._ema_alpha) * self._ema_probs
            )
        smoothed = self._ema_probs / (self._ema_probs.sum() + 1e-10)
        emotion_idx = int(np.argmax(smoothed))

        # 6. Build result from the heuristic base (preserves band_powers, DE, etc.)
        # but override probabilities, emotion, confidence with ensemble output.
        result = heuristic_result.copy()
        result["probabilities"] = {
            EMOTIONS[i]: round(float(p), 4) for i, p in enumerate(smoothed)
        }

        _CONFIDENCE_THRESHOLD = 0.25
        top_conf = float(smoothed[emotion_idx])
        result["emotion"] = EMOTIONS[emotion_idx] if top_conf >= _CONFIDENCE_THRESHOLD else "neutral"
        result["emotion_index"] = emotion_idx
        result["model_type"] = "ensemble-eegnet-heuristic"

        # 6b. MC Dropout uncertainty — more principled than max(softmax).
        # Use MC confidence when available; fall back to top_conf (max prob) if not.
        mc_confidence = eegnet_result.get("mc_confidence")
        if mc_confidence is not None:
            result["confidence"] = round(float(mc_confidence), 4)
            result["mc_uncertainty"] = eegnet_result.get("mc_uncertainty", 0.0)
            result["mc_confidence"] = round(float(mc_confidence), 4)
            result["mc_pred_std"] = eegnet_result.get("mc_pred_std", {})
            result["mc_predictive_entropy"] = eegnet_result.get("mc_predictive_entropy", 0.0)
            result["confidence_label"] = eegnet_result.get("confidence_label", "medium")
        else:
            result["confidence"] = top_conf

        # 7. Blend valence/arousal: weighted average of EEGNet and heuristic
        # (heuristic values already went through _smooth_index in _predict_features,
        # EEGNet values are from its own EMA)
        eegnet_valence = float(eegnet_result.get("valence", 0.0))
        eegnet_arousal = float(eegnet_result.get("arousal", 0.5))
        result["valence"] = float(np.clip(
            w_e * eegnet_valence + w_h * result["valence"], -1, 1
        ))
        result["arousal"] = float(np.clip(
            w_e * eegnet_arousal + w_h * result["arousal"], 0, 1
        ))

        # Re-apply EMA smoothing on the blended valence/arousal
        result["valence"] = self._smooth_index("_ema_valence", result["valence"])
        result["arousal"] = self._smooth_index("_ema_arousal", result["arousal"])

        # 8. Model contribution logging — measure how much each sub-model
        # influenced the final prediction.  Uses KL-divergence of the ensemble
        # probability vector from each sub-model's probability vector.  Lower KL
        # from a sub-model means the ensemble tracked that sub-model more closely.
        # Normalized to sum to 1.0.
        eps = 1e-10
        ensemble_arr = np.array(
            [float(result["probabilities"].get(e, 0.0)) for e in EMOTIONS],
            dtype=float,
        )
        ensemble_arr = ensemble_arr / (ensemble_arr.sum() + eps)

        eegnet_arr = np.array(
            [float(eegnet_probs_dict.get(e, 0.0)) for e in EMOTIONS], dtype=float
        )
        eegnet_arr = eegnet_arr / (eegnet_arr.sum() + eps)

        heuristic_arr = np.array(
            [float(heuristic_probs_dict.get(e, 0.0)) for e in EMOTIONS], dtype=float
        )
        heuristic_arr = heuristic_arr / (heuristic_arr.sum() + eps)

        # KL(ensemble || sub_model) — lower = more similar
        kl_eegnet = float(np.sum(ensemble_arr * np.log((ensemble_arr + eps) / (eegnet_arr + eps))))
        kl_heuristic = float(np.sum(ensemble_arr * np.log((ensemble_arr + eps) / (heuristic_arr + eps))))

        # Convert KL to "closeness" (inverse), then normalize to sum to 1
        closeness_eegnet = 1.0 / (kl_eegnet + eps)
        closeness_heuristic = 1.0 / (kl_heuristic + eps)
        total_closeness = closeness_eegnet + closeness_heuristic

        result["eegnet_contribution"] = round(closeness_eegnet / total_closeness, 4)
        result["heuristic_contribution"] = round(closeness_heuristic / total_closeness, 4)

        # 9. Signal quality adaptive weighting — expose for debugging and UI
        result["epoch_quality"] = round(_epoch_quality, 4)
        result["adaptive_eegnet_weight"] = w_e
        result["adaptive_heuristic_weight"] = w_h

        return result

    # ────────────────────────────────────────────────────────────────
    # Feature-based classifier (primary path for Muse 2)
    # ────────────────────────────────────────────────────────────────

    def _predict_features(self, eeg: np.ndarray, fs: float, device_type: str = "muse_2") -> Dict:
        """Neuroscience-backed emotion classification from band powers.

        Accepts 1D (single channel) or 2D (n_channels, n_samples) input.
        When multichannel Muse 2 data is provided (AF7=ch0, AF8=ch1), computes
        Frontal Alpha Asymmetry (FAA) per Davidson (1992) to improve valence accuracy.

        EEG-emotion correlates used:
        - Alpha (8-12 Hz): inversely related to cortical arousal; dominant in relaxation
        - Low-beta (12-20 Hz): active cognition, focus, motor planning
        - High-beta (20-30 Hz): anxiety, stress, fear — strongest differentiator for fearful
        - Beta (12-30 Hz): active thinking, concentration, anxiety when excessive
        - Theta (4-8 Hz): drowsiness, meditation, creative states
        - Gamma (30+ Hz): EXCLUDED from Muse 2 arousal/stress/valence — dominated by
          frontalis EMG artifact at AF7/AF8 sites (not neural). Only used as minimal
          discriminator in angry vs fearful distinction.
        - Delta (0.5-4 Hz): deep sleep, regeneration
        """
        # ── Artifact rejection ───────────────────────────────────
        # Reject epochs with extreme amplitude (eye blinks: 100-200 µV, EMG bursts).
        # ONLY freeze EMA if we have prior history — on the first epoch, run the model
        # regardless, otherwise _ema_probs stays at uniform 1/6 forever.
        max_amp = float(np.max(np.abs(eeg)))
        _artifact_now = max_amp > _ARTIFACT_THRESHOLD_UV
        if _artifact_now and self._ema_probs is not None:
            smoothed = self._ema_probs / (self._ema_probs.sum() + 1e-10)
            emotion_idx = int(np.argmax(smoothed))
            top_conf = float(smoothed[emotion_idx])
            emotion_label = EMOTIONS[emotion_idx] if top_conf >= 0.25 else "neutral"
            result = {e: float(smoothed[i]) for i, e in enumerate(EMOTIONS)}
            # Return frozen EMA state — do not update with artifact epoch.
            # Use last smoothed continuous indices if available, else neutral defaults.
            _fv = self._ema_valence if self._ema_valence is not None else 0.0
            _fa = self._ema_arousal if self._ema_arousal is not None else 0.5
            _frozen_dominance = 0.5
            _frozen_granular = map_vad_to_granular_emotions(_fv, _fa, _frozen_dominance)
            return {
                "emotion": emotion_label, "emotion_index": emotion_idx,
                "confidence": top_conf, "probabilities": result,
                "valence": _fv,
                "arousal": _fa,
                "dominance": _frozen_dominance,
                "granular_emotions": _frozen_granular,
                "stress_index": self._ema_stress if self._ema_stress is not None else 0.5,
                "focus_index": self._ema_focus if self._ema_focus is not None else 0.5,
                "relaxation_index": self._ema_relaxation if self._ema_relaxation is not None else 0.5,
                "anger_index": self._ema_anger if self._ema_anger is not None else 0.0,
                "fear_index": self._ema_fear if self._ema_fear is not None else 0.0,
                "band_powers": {}, "differential_entropy": {},
                "dasm_rasm": {}, "frontal_midline_theta": {},
                "frontal_alpha_coherence": 0.0,
                "plv_connectivity": {
                    "plv_frontal_alpha": 0.0, "plv_frontal_beta": 0.0,
                    "plv_fronto_temporal_alpha": 0.0,
                    "plv_mean_alpha": 0.0, "plv_mean_theta": 0.0, "plv_mean_beta": 0.0,
                },
                "band_hjorth": {},
                "artifact_detected": True,
                "explanation": [],
            }

        # ── Multichannel support ─────────────────────────────────
        # Keep original channels array for FAA; use channel 0 (AF7) for band powers.
        channels = eeg if eeg.ndim == 2 else None
        signal = eeg[0] if eeg.ndim == 2 else eeg

        processed = preprocess(signal, fs)
        bands = extract_band_powers(processed, fs)
        de = differential_entropy(processed, fs)

        alpha_raw      = bands.get("alpha",      0)
        beta_raw       = bands.get("beta",       0)
        theta_raw      = bands.get("theta",      0)
        gamma_raw      = bands.get("gamma",      0)
        delta_raw      = bands.get("delta",      0)
        high_beta_raw  = bands.get("high_beta",  0)
        low_beta_raw   = bands.get("low_beta",   0)
        low_alpha_raw  = bands.get("low_alpha",  0)
        high_alpha_raw = bands.get("high_alpha", 0)

        # Normalize to relative fractions that sum to 1.
        # Without this the absolute power values from BrainFlow (sum often 0.4-0.8)
        # make every formula threshold fire at different levels, producing flat probs.
        total_power = alpha_raw + beta_raw + theta_raw + gamma_raw + delta_raw
        if total_power < 1e-6:
            total_power = 1.0
        alpha      = alpha_raw      / total_power
        beta       = beta_raw       / total_power
        theta      = theta_raw      / total_power
        gamma      = gamma_raw      / total_power
        delta      = delta_raw      / total_power
        high_beta  = high_beta_raw  / total_power
        low_beta   = low_beta_raw   / total_power
        low_alpha  = low_alpha_raw  / total_power
        high_alpha = high_alpha_raw / total_power

        # high_beta_frac: fraction of beta that is 20-30 Hz (fear/anxiety marker).
        # Fearful/anxious states skew beta distribution toward high-beta.
        high_beta_frac = high_beta_raw / max(beta_raw, 1e-10)

        # high_alpha_frac: fraction of alpha that is 10-12 Hz (emotion-specific).
        # High-alpha is more emotion-relevant than low-alpha (Bazanova & Vernon 2014).
        # High fraction = emotional engagement; low fraction = general alertness only.
        high_alpha_frac = high_alpha_raw / max(alpha_raw, 1e-10)

        # Store snapshot for temporal analysis
        self._history.append(bands)

        # ── Frontal Alpha Asymmetry (FAA) ────────────────────────
        # FAA = ln(R_alpha) - ln(L_alpha) → positive = approach motivation / positive affect.
        # Davidson (1992). Primary valence biomarker when Muse 2 multichannel data available.
        # Note: FAA reflects approach motivation (left-lateralized), not pure positive valence —
        # anger (approach + negative valence) also shows left FAA. Blend with ABR for robustness.
        faa_valence = 0.0
        if channels is not None and channels.shape[0] >= 2:
            _cmap = get_channel_map(device_type, channels.shape[0])
            _lf, _rf = _cmap["left_frontal"], _cmap["right_frontal"]
            asym = compute_frontal_asymmetry(channels, fs, left_ch=_lf, right_ch=_rf)
            faa_valence = asym.get("asymmetry_valence", 0.0)

        # ── High-Alpha Asymmetry (HAA) ────────────────────────
        # HAA = ln(R_high_alpha) - ln(L_high_alpha) in the 10-12 Hz sub-band.
        # High-alpha is MORE emotion-specific than full-band alpha (8-12 Hz):
        #   - Low-alpha (8-10 Hz) indexes general alertness/arousal (Klimesch 1999)
        #   - High-alpha (10-12 Hz) indexes task-specific processing, emotional engagement
        # HAA provides a more precise valence signal than FAA for active emotional states.
        # Bazanova & Vernon (2014): high-alpha specificity for emotional states.
        haa_valence = 0.0
        if channels is not None and channels.shape[0] >= 2:
            _cmap = get_channel_map(device_type, channels.shape[0])
            _lf, _rf = _cmap["left_frontal"], _cmap["right_frontal"]
            # Compute HAA directly from high-alpha band power
            from processing.eeg_processor import bandpass_filter as _bp_filter
            _left_ha = _bp_filter(preprocess(channels[_lf], fs), 10.0, 12.0, fs)
            _right_ha = _bp_filter(preprocess(channels[_rf], fs), 10.0, 12.0, fs)
            _lp = float(np.var(_left_ha))
            _rp = float(np.var(_right_ha))
            _haa_raw = float(np.log(max(_rp, 1e-12)) - np.log(max(_lp, 1e-12)))
            haa_valence = float(np.clip(np.tanh(_haa_raw * 2.0), -1, 1))

        # ── DASM Alpha (DE-based asymmetry complement to FAA) ────
        # DASM_alpha = DE(AF8_alpha) - DE(AF7_alpha). Same directional meaning as FAA
        # but computed from differential entropy rather than raw log-power. Adds an
        # independent estimate of hemispheric alpha asymmetry for valence blending.
        # DASM_beta used to refine stress: right-dominant beta elevation = withdrawal stress.
        dasm_rasm: Dict = {}
        dasm_alpha_valence = 0.0
        dasm_beta_stress = 0.0
        if channels is not None and channels.shape[0] >= 3:
            _cmap = get_channel_map(device_type, channels.shape[0])
            _lf, _rf = _cmap["left_frontal"], _cmap["right_frontal"]
            dasm_rasm = compute_dasm_rasm(channels, fs, left_ch=_lf, right_ch=_rf)
            dasm_alpha_valence = float(
                np.clip(np.tanh(dasm_rasm.get("dasm_alpha", 0.0)), -1, 1)
            )
            # Positive dasm_beta = right-dominant beta = slight withdrawal/stress signal
            dasm_beta_stress = float(
                np.clip(np.tanh(dasm_rasm.get("dasm_beta", 0.0) * 0.5), 0, 1)
            )

        # ── Frontal Midline Theta (FMT) ──────────────────────────
        # FMT = ACC/mPFC theta (4-8 Hz). More reference-robust than FAA because
        # theta at Fpz is less sensitive to the Fpz reference electrode problem.
        # Left hemisphere FMT asymmetry → pleasant valence.
        # Used as a supplemental output in the emotion dict.
        fmt: Dict = {}
        if channels is not None and channels.shape[0] >= 2:
            try:
                fmt = compute_frontal_midline_theta(channels[1], fs)  # AF7 channel
            except Exception:
                fmt = {}

        # ── Frontal Alpha Coherence ──────────────────────────────
        # Inter-channel coherence in the alpha band between AF7-AF8.
        # High frontal alpha coherence → integrated emotional processing,
        # deep relaxation, coherent brain state → emotion reading is more
        # trustworthy. Low coherence → dissociated/fragmented states.
        # Research: Tass et al. (1998), Fingelkurts & Fingelkurts (2006).
        frontal_alpha_coh = 0.0
        if channels is not None and channels.shape[0] >= 3:
            try:
                _cmap = get_channel_map(device_type, channels.shape[0])
                _lf, _rf = _cmap["left_frontal"], _cmap["right_frontal"]
                # Compute coherence on just the frontal pair
                frontal_pair = channels[np.array([_lf, _rf])]
                frontal_alpha_coh = float(compute_coherence(frontal_pair, fs, "alpha"))
                # Clamp to [0, 1] for safety
                frontal_alpha_coh = float(np.clip(frontal_alpha_coh, 0.0, 1.0))
            except Exception:
                frontal_alpha_coh = 0.0

        # ── PLV Functional Connectivity ──────────────────────────
        # Per-pair Phase Locking Value across theta/alpha/beta bands.
        # Wang et al. (2024): PLV + microstates + PSD fusion → 7%+ accuracy
        # improvement over single-domain features. High-frequency PLV in
        # prefrontal and temporal regions enhances emotional differentiation.
        #
        # plv_frontal_alpha: AF7-AF8 phase synchrony in alpha band.
        #   High frontal PLV + positive FAA → stronger positive valence signal.
        #   Low frontal PLV → dissociated hemispheres → less trust in FAA.
        # plv_fronto_temporal_alpha: AF7-TP9 / AF8-TP10 mean PLV.
        #   High fronto-temporal coupling → integrated emotional processing → arousal.
        # plv_frontal_beta: AF7-AF8 PLV in beta band.
        #   High frontal beta synchrony → cognitive engagement / focus.
        plv_data: Dict = {}
        plv_frontal_alpha = 0.0
        plv_frontal_beta = 0.0
        plv_ft_alpha = 0.0
        if channels is not None and channels.shape[0] >= 4:
            try:
                plv_data = compute_pairwise_plv(channels, fs)
                plv_frontal_alpha = plv_data.get("plv_frontal_alpha", 0.0)
                plv_frontal_beta = plv_data.get("plv_frontal_beta", 0.0)
                plv_ft_alpha = plv_data.get("plv_fronto_temporal_alpha", 0.0)
            except Exception:
                plv_data = {}

        # ── Band-specific Hjorth Mobility ──────────────────────
        # Hjorth mobility computed on band-filtered signals captures the
        # spectral centroid within each band — a richer feature than raw
        # band power.  Beta mobility achieved 83.33% accuracy (AUC 0.904)
        # on SEED dataset as the single best feature among 18 examined
        # (PubMed, SVM leave-one-subject-out).
        #
        # The frontal/temporal mobility ratio indicates WHERE beta
        # processing is concentrated.  High frontal beta mobility
        # (ratio > 1) suggests focused cognitive engagement; temporal
        # dominance may indicate emotional processing or mind-wandering.
        band_hjorth: Dict = {}
        beta_mobility_ratio_ft = 1.0  # neutral default
        if channels is not None and channels.shape[0] >= 4:
            try:
                _cmap = get_channel_map(device_type, channels.shape[0])
                _lf, _rf = _cmap["left_frontal"], _cmap["right_frontal"]
                _lt, _rt = _cmap.get("left_temporal", 0), _cmap.get("right_temporal", channels.shape[0] - 1)
                band_hjorth = compute_hjorth_mobility_ratio(
                    channels, fs,
                    frontal_chs=[_lf, _rf],
                    temporal_chs=[_lt, _rt],
                    bands=["beta"],
                )
                beta_mobility_ratio_ft = band_hjorth.get("beta_mobility_ratio_ft", 1.0)
            except Exception:
                band_hjorth = {}
        elif channels is not None and channels.shape[0] >= 2:
            try:
                # With 2-3 channels, compute single-channel beta mobility
                single_mob = compute_band_hjorth_mobility(processed, fs, bands=["beta"])
                band_hjorth = single_mob
            except Exception:
                band_hjorth = {}

        # ── Valence (pleasantness) ──────────────────────────────
        # Alpha/beta ratio (ABR) as secondary valence signal.
        # Use high_alpha (10-12 Hz) instead of full alpha for the ratio term:
        # high-alpha is more emotion-specific (Bazanova & Vernon 2014), while
        # low-alpha tracks general arousal. This makes ABR more responsive to
        # emotional state changes rather than general alertness fluctuations.
        alpha_beta_ratio = alpha / max(beta, 1e-10)
        high_alpha_beta_ratio = high_alpha / max(beta, 1e-10)
        valence_abr = (
            0.45 * np.tanh((alpha_beta_ratio - 0.7) * 2.0)           # full alpha/beta (general)
            + 0.20 * np.tanh((high_alpha_beta_ratio - 0.3) * 3.0)    # high-alpha/beta (emotion-specific)
            + 0.35 * np.tanh((alpha - 0.15) * 4)                     # absolute alpha level
        )
        if channels is not None and channels.shape[0] >= 3 and dasm_rasm:
            # 5-signal blend: ABR + FAA + HAA + DASM_alpha + PLV-weighted FAA boost.
            # HAA (high-alpha asymmetry) is more emotion-specific than FAA.
            # Splitting FAA weight: 20% full-band FAA + 15% HAA.
            #
            # PLV confidence modulation (Wang et al. 2024): when AF7-AF8 phase
            # locking in alpha is high (>0.5), the FAA signal is more reliable
            # because the hemispheres are coherently processing — amplify the
            # asymmetry signal by up to 10%.  When PLV is low, asymmetry signals
            # (FAA, HAA, DASM) may be noise-driven — reduce their weight.
            # plv_faa_boost: 0 when plv_frontal_alpha=0, up to 0.10 when plv=1.
            plv_faa_boost = 0.10 * plv_frontal_alpha * abs(faa_valence)
            plv_faa_sign = 1.0 if faa_valence >= 0 else -1.0
            valence = float(np.clip(
                0.33 * valence_abr
                + 0.20 * faa_valence
                + 0.12 * haa_valence
                + 0.25 * dasm_alpha_valence
                + 0.10 * plv_faa_sign * plv_faa_boost,
                -1, 1
            ))
        elif channels is not None and channels.shape[0] >= 2:
            # 3-signal blend: ABR + FAA + HAA
            valence = float(np.clip(
                0.40 * valence_abr + 0.30 * faa_valence + 0.30 * haa_valence,
                -1, 1
            ))
        else:
            valence = float(np.clip(valence_abr, -1, 1))

        # ── Arousal (activation level) ──────────────────────────
        # Beta indicates cortical activation; alpha + delta indicate deactivation.
        # Gamma EXCLUDED: at Muse 2 frontal sites (AF7/AF8), 30-100 Hz is dominated
        # by frontalis EMG artifact, not neural gamma. Including it artificially
        # inflates arousal whenever the user tenses their forehead or jaw.
        # PLV fronto-temporal arousal component: high alpha coupling between
        # frontal and temporal regions indicates integrated emotional processing
        # (Pessoa 2008, "The cognitive-emotional brain").  This reflects
        # emotional engagement — a signal complementary to spectral arousal.
        # Weight is small (0.08) to avoid overwhelming the proven spectral features.
        arousal_raw = (
            0.42 * beta / max(beta + alpha, 1e-10)                     # beta proportion (primary)
            + 0.28 * (1.0 - alpha / max(alpha + beta + theta, 1e-10))  # inverse alpha
            + 0.22 * (1.0 - delta / max(delta + beta, 1e-10))          # inverse delta
            + 0.08 * plv_ft_alpha                                       # fronto-temporal PLV coupling
        )
        arousal = float(np.clip(arousal_raw, 0, 1))

        # ── Emotion probability estimation ──────────────────────
        # Based on the circumplex model with EEG-specific weightings
        probs = np.zeros(6)

        theta_beta_ratio = theta / max(beta, 1e-10)
        beta_alpha_ratio = beta / max(alpha, 1e-10)

        # Happy: positive valence, moderate-high arousal, good alpha.
        # Gamma removed — predominantly frontalis EMG at Muse 2 AF7/AF8 sites.
        probs[0] = (
            0.45 * max(0, valence)                     # primary: positive valence (includes FAA)
            + 0.30 * max(0, arousal - 0.3)             # moderate-high arousal
            + 0.25 * min(1, alpha_beta_ratio * 0.8)    # alpha > beta (calm-positive)
        )

        # Sad: negative valence + low arousal.
        # Threshold lowered from -0.25 → -0.10: with FAA now blended into valence,
        # negative states register earlier. Resting FAA ≈ 0 keeps neutral at ~0;
        # genuinely sad states show negative FAA pushing valence clearly below -0.10.
        probs[1] = (
            0.50 * max(0, -valence - 0.10)                  # needs negative valence
            + 0.30 * max(0, 0.35 - arousal)                 # must be truly low arousal
            + 0.20 * max(0, theta_beta_ratio - 2.0) * 0.4  # only VERY high theta/beta
        )

        # Angry: negative valence + elevated arousal + beta dominance.
        # Key differentiator from fearful: anger = fight/approach motivation.
        # Gamma weight drastically reduced — EMG at Muse 2 frontal sites creates
        # false anger spikes whenever the user clenches jaw or tenses forehead.
        probs[2] = (
            0.40 * max(0, -valence - 0.1)                       # negative valence (primary)
            + 0.30 * max(0, arousal - 0.45)                     # elevated arousal
            + 0.25 * min(1, max(0, beta_alpha_ratio - 1.3))     # beta dominance
            + 0.05 * min(1, gamma * 2.5)                        # minimal gamma (noisy)
        )

        # Fearful/Anxious: negative valence + elevated arousal + beta dominance.
        # Key differentiator from angry: fear = freeze/avoidance (high-beta dominant,
        # LESS gamma than angry). Substantially lowered thresholds to capture moderate fear.
        # high_beta (20-30 Hz) is the strongest EEG marker for anxiety/fear states.
        probs[3] = (
            0.30 * max(0, -valence - 0.15)                       # negative valence (lowered: -0.3→-0.15)
            + 0.25 * max(0, arousal - 0.50)                      # elevated arousal (lowered: 0.6→0.50)
            + 0.30 * min(1, max(0, beta_alpha_ratio - 1.8))      # beta dominance (lowered: 2.5→1.8)
            + 0.15 * min(1, high_beta * 5.0)                     # high-beta power: primary fear marker
        )

        # Relaxed: low arousal + dominant alpha OR high theta (meditative/drowsy).
        # Theta at 30-55% of power = meditative/drowsy waking state → relaxed, not fearful.
        probs[4] = (
            0.10 * max(0, valence * 0.5)                  # positive valence is a soft bonus
            + 0.20 * max(0, 0.6 - arousal)                # low arousal
            + 0.35 * min(1, alpha * 2.5)                   # dominant alpha (primary relaxation signal)
            + 0.20 * min(1, theta * 1.5)                   # high theta = meditative/drowsy → relaxed
            + 0.15 * max(0, 1 - beta_alpha_ratio * 0.5)   # low beta relative to alpha
        )

        # Focused: moderate-high arousal, strong beta (especially low-beta), low theta/beta.
        # Low-beta (12-20 Hz) is the working-memory and attentional band; stronger weight.
        # Slightly lowered arousal threshold — focused states can be calm but engaged.
        probs[5] = (
            0.10 * max(0, 0.5 + valence * 0.5)               # soft neutral-to-positive valence bonus
            + 0.25 * min(1, max(0, arousal - 0.30) * 2.0)    # moderate arousal (lowered: 0.35→0.30)
            + 0.40 * min(1, beta * 3.5)                       # strong beta (increased: 0.35→0.40)
            + 0.25 * max(0, 1 - theta_beta_ratio * 0.40)      # low theta/beta (adjusted multiplier)
        )

        # If all raw scores are near zero (degenerate case — e.g. synthetic board
        # data with features in the neutral zone), don't emit uniform 17% for each
        # emotion. Instead return empty probabilities so the UI shows "—" rather
        # than six bars all frozen at 17%.
        if probs.max() < 1e-4:
            # Degenerate: initialize EMA to uniform so future artifact calls can freeze
            if self._ema_probs is None:
                self._ema_probs = np.ones(len(probs)) / len(probs)
            _degen_dominance = _compute_dominance(beta_alpha_ratio, theta_beta_ratio)
            _degen_granular = map_vad_to_granular_emotions(valence, arousal, _degen_dominance)
            return {
                "emotion": "neutral",
                "emotion_index": 5,
                "confidence": 0.0,
                "probabilities": {},
                "classification_confidence": "low",
                "valence": valence,
                "arousal": arousal,
                "dominance": _degen_dominance,
                "granular_emotions": _degen_granular,
                "stress_index": float(np.clip(
                    0.40 * min(1, beta_alpha_ratio * 0.3)
                    + 0.25 * max(0, 1 - alpha * 2.5)
                    + 0.25 * min(1, high_beta * 4)
                    + 0.10 * dasm_beta_stress,
                    0, 1
                )),
                "focus_index": float(np.clip(
                    0.45 * min(1, beta * 3.5)
                    + 0.40 * max(0, 1 - theta_beta_ratio * 0.40)
                    + 0.15 * min(1, low_beta * 5),
                    0, 1
                )),
                "relaxation_index": float(np.clip(alpha * 2.0, 0, 1)),
                "anger_index": 0.0,
                "fear_index": 0.0,
                "band_powers": bands,
                "differential_entropy": de,
                "dasm_rasm": dasm_rasm,
                "frontal_midline_theta": fmt,
                "frontal_alpha_coherence": frontal_alpha_coh,
                "plv_connectivity": {
                    "plv_frontal_alpha": float(plv_frontal_alpha),
                    "plv_frontal_beta": float(plv_frontal_beta),
                    "plv_fronto_temporal_alpha": float(plv_ft_alpha),
                    "plv_mean_alpha": float(plv_data.get("plv_mean_alpha", 0.0)),
                    "plv_mean_theta": float(plv_data.get("plv_mean_theta", 0.0)),
                    "plv_mean_beta": float(plv_data.get("plv_mean_beta", 0.0)),
                },
                "band_hjorth": band_hjorth,
                "artifact_detected": _artifact_now,
                "model_type": "feature-based",
                "explanation": [],
            }

        # Softmax-like normalization (temperature=2.5 for clear sharpness).
        # Subtract max before exp to prevent overflow (numerically stable softmax).
        temp = 2.5
        scaled = probs * temp
        scaled -= scaled.max()  # shift so max is 0 → exp values in (0, 1]
        probs_exp = np.exp(scaled)
        probs = probs_exp / (probs_exp.sum() + 1e-10)

        # Exponential moving average smoothing
        if self._ema_probs is None:
            self._ema_probs = probs
        else:
            self._ema_probs = self._ema_alpha * probs + (1 - self._ema_alpha) * self._ema_probs

        smoothed = self._ema_probs / (self._ema_probs.sum() + 1e-10)
        emotion_idx = int(np.argmax(smoothed))

        # ── Dominance (sense of control / agency) ─────────────
        dominance = _compute_dominance(beta_alpha_ratio, theta_beta_ratio)

        # ── Mental state indices (0-1 scale) ────────────────────
        # Stress: high beta/alpha ratio + low alpha + high-beta + optional DASM beta.
        # DASM beta (10% weight): right-dominant beta correlates with withdrawal stress.
        # Gamma excluded — EMG noise at AF7/AF8 inflates stress artificially (jaw clenching).
        stress_index = float(np.clip(
            0.40 * min(1, beta_alpha_ratio * 0.3)
            + 0.25 * max(0, 1 - alpha * 2.5)
            + 0.25 * min(1, high_beta * 4)         # high-beta is the real stress marker
            + 0.10 * dasm_beta_stress,              # right-dominant beta = withdrawal stress
            0, 1
        ))

        # Focus: strong beta (especially low-beta), low theta/beta, moderate arousal.
        # Gamma excluded — EMG noise at AF7/AF8 produces false focus spikes.
        # PLV frontal beta (5% weight): high AF7-AF8 beta synchrony indicates
        # coordinated cognitive engagement across hemispheres (Sauseng et al. 2005).
        # Beta mobility ratio (5% weight): frontal-dominant beta mobility
        # (ratio > 1) indicates focused cognitive engagement in prefrontal cortex.
        # PubMed: Hjorth mobility in beta = best single EEG feature for emotion
        # recognition (83.33% acc, AUC 0.904 on SEED, SVM LOSO).
        # Normalize: ratio 1.0 → 0.5, ratio 2.0 → 1.0, ratio 0.5 → 0.25.
        _beta_mob_focus = float(np.clip(beta_mobility_ratio_ft * 0.5, 0, 1))
        focus_index = float(np.clip(
            0.40 * min(1, beta * 3.5)
            + 0.35 * max(0, 1 - theta_beta_ratio * 0.40)
            + 0.15 * min(1, low_beta * 5)           # low-beta is the primary attentional band
            + 0.05 * plv_frontal_beta                # frontal beta synchrony = cognitive engagement
            + 0.05 * _beta_mob_focus,                # frontal beta mobility ratio
            0, 1
        ))

        # Relaxation: high alpha (especially high-alpha), low beta, theta welcome.
        # High-alpha (10-12 Hz) is more specific to relaxed-engaged states than
        # low-alpha (general drowsiness/alertness). Split alpha contribution:
        # 32% full-band alpha + 15% high-alpha bonus for emotional relaxation.
        # Frontal alpha coherence (5% weight): high coherence between AF7-AF8
        # indicates integrated, synchronized brain state — validated marker of
        # deep relaxation and meditation (Tass 1998, Fingelkurts 2006).
        relaxation_index = float(np.clip(
            0.32 * min(1, alpha * 2.5)
            + 0.15 * min(1, high_alpha * 5.0)            # high-alpha bonus for emotional relaxation
            + 0.28 * max(0, 1 - beta_alpha_ratio * 0.3)
            + 0.20 * min(1, theta * 1.5)
            + 0.05 * frontal_alpha_coh,
            0, 1
        ))

        # Anger: high beta/alpha + suppressed alpha + high-beta elevation.
        # Gamma weight minimized — EMG at Muse 2 frontal sites creates false anger.
        anger_index = float(np.clip(
            0.40 * min(1, max(0, beta_alpha_ratio - 1.0) * 0.5)
            + 0.30 * max(0, 1 - alpha * 5)
            + 0.20 * min(1, high_beta * 3)
            + 0.10 * min(1, gamma * 3),            # minimal weight (EMG noise)
            0, 1
        ))

        # Fear: high-beta dominance + extreme beta/alpha + suppressed alpha.
        # Distinguishes anxious/fearful from stressed-but-focused (which lacks the
        # beta/alpha elevation) and from angry (which has more gamma, less high-beta frac).
        fear_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.5) * 0.6)
            + 0.30 * min(1, high_beta * 5)        # high-beta is the strongest fear marker
            + 0.25 * max(0, 1 - alpha * 5)        # alpha suppression in fear
            + 0.10 * max(0, arousal - 0.45),      # elevated arousal (not pure freeze)
            0, 1
        ))

        # If max confidence is below threshold the EEG doesn't clearly match any
        # of the 6 trained emotions — label as "neutral" instead of forcing a wrong label.
        # With 6 classes, random baseline = 0.167. Threshold 0.25 means the top emotion
        # is at least 50% above chance before we commit to a label — prevents borderline
        # states (e.g. slightly negative valence) from being labeled as "sad".
        _CONFIDENCE_THRESHOLD = 0.25
        top_conf = float(smoothed[emotion_idx])
        emotion_label = EMOTIONS[emotion_idx] if top_conf >= _CONFIDENCE_THRESHOLD else "neutral"

        # ── EMA smoothing on continuous indices ─────────────────
        # Reduces frame-to-frame jitter on dashboard readings.
        # tau ≈ 3.9 sec with alpha=0.4 and 2-sec hops — within the
        # recommended 3-5 sec decay range from EEG literature.
        # Raw values are still used in the probability formulas above
        # (probabilities have their own EMA via _ema_probs).
        valence          = self._smooth_index("_ema_valence", valence)
        arousal          = self._smooth_index("_ema_arousal", arousal)
        stress_index     = self._smooth_index("_ema_stress", stress_index)
        focus_index      = self._smooth_index("_ema_focus", focus_index)
        relaxation_index = self._smooth_index("_ema_relaxation", relaxation_index)
        anger_index      = self._smooth_index("_ema_anger", anger_index)
        fear_index       = self._smooth_index("_ema_fear", fear_index)

        # ── Heuristic explanation ─────────────────────────────────
        # Collect the key feature contributions that drove valence, arousal,
        # and the predicted emotion. Each contribution is the weighted term
        # value from the formulas above — already computed, just surfaced.
        _contributions = [
            ("alpha_beta_ratio", float(valence_abr)),
            ("frontal_asymmetry", float(faa_valence)),
            ("high_alpha_asymmetry", float(haa_valence)),
            ("dasm_alpha", float(dasm_alpha_valence)),
            ("beta_activation", float(arousal)),
            ("high_beta_stress", float(stress_index)),
            ("theta_relaxation", float(relaxation_index)),
            ("beta_alpha_ratio", float(beta_alpha_ratio)),
            ("high_alpha_frac", float(high_alpha_frac)),
            ("dasm_beta_stress", float(dasm_beta_stress)),
            ("frontal_alpha_coherence", float(frontal_alpha_coh)),
            ("plv_frontal_alpha", float(plv_frontal_alpha)),
            ("plv_frontal_beta", float(plv_frontal_beta)),
            ("plv_fronto_temporal_alpha", float(plv_ft_alpha)),
            ("beta_mobility_ratio_ft", float(beta_mobility_ratio_ft)),
        ]
        heuristic_explanation = self._compute_heuristic_explanation(_contributions)

        granular_emotions = map_vad_to_granular_emotions(valence, arousal, dominance)

        return {
            "emotion": emotion_label,
            "emotion_index": emotion_idx,
            "confidence": top_conf,
            "probabilities": {EMOTIONS[i]: float(p) for i, p in enumerate(smoothed)},
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "granular_emotions": granular_emotions,
            "stress_index": stress_index,
            "focus_index": focus_index,
            "relaxation_index": relaxation_index,
            "anger_index": anger_index,
            "fear_index": fear_index,
            "band_powers": bands,
            "differential_entropy": de,
            "dasm_rasm": dasm_rasm,
            "frontal_midline_theta": fmt,
            "frontal_alpha_coherence": frontal_alpha_coh,
            "high_alpha_asymmetry": float(haa_valence),
            "high_alpha_fraction": float(high_alpha_frac),
            "plv_connectivity": {
                "plv_frontal_alpha": float(plv_frontal_alpha),
                "plv_frontal_beta": float(plv_frontal_beta),
                "plv_fronto_temporal_alpha": float(plv_ft_alpha),
                "plv_mean_alpha": float(plv_data.get("plv_mean_alpha", 0.0)),
                "plv_mean_theta": float(plv_data.get("plv_mean_theta", 0.0)),
                "plv_mean_beta": float(plv_data.get("plv_mean_beta", 0.0)),
            },
            "band_hjorth": band_hjorth,
            "artifact_detected": _artifact_now,
            "model_type": "feature-based",
            "explanation": heuristic_explanation,
        }

    # ────────────────────────────────────────────────────────────────
    # ONNX inference (only used if benchmark >= 60%)
    # ────────────────────────────────────────────────────────────────

    def _predict_onnx(self, eeg: np.ndarray, fs: float) -> Dict:
        """ONNX model inference."""
        processed = preprocess(eeg, fs)
        bands = extract_band_powers(processed, fs)
        features_dict = extract_features(processed, fs)
        de = differential_entropy(processed, fs)
        features = np.array(list(features_dict.values()), dtype=np.float32).reshape(1, -1)

        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: features})
        emotion_idx = int(outputs[0][0])
        prob_map = outputs[1][0]
        n_classes = len(EMOTIONS)
        probs = [float(prob_map.get(i, 0.0)) for i in range(n_classes)]

        alpha_raw     = bands.get("alpha",     0)
        beta_raw      = bands.get("beta",      0)
        theta_raw     = bands.get("theta",     0)
        gamma_raw     = bands.get("gamma",     0)
        delta_raw     = bands.get("delta",     0)
        high_beta_raw = bands.get("high_beta", 0)
        low_beta_raw  = bands.get("low_beta",  0)

        total_power = alpha_raw + beta_raw + theta_raw + gamma_raw + delta_raw
        if total_power < 1e-6:
            total_power = 1.0
        alpha     = alpha_raw     / total_power
        beta      = beta_raw      / total_power
        theta     = theta_raw     / total_power
        gamma     = gamma_raw     / total_power
        high_beta = high_beta_raw / total_power
        low_beta  = low_beta_raw  / total_power

        delta     = delta_raw     / total_power
        beta_alpha_ratio = beta / max(alpha, 1e-10)
        theta_beta_ratio = theta / max(beta, 1e-10)

        valence = float(np.clip(
            0.65 * np.tanh((alpha / max(beta, 1e-10) - 0.7) * 2.0)
            + 0.35 * np.tanh((alpha - 0.15) * 4),
            -1, 1
        ))
        arousal = float(np.clip(
            0.45 * beta / max(beta + alpha, 1e-10)
            + 0.30 * (1.0 - alpha / max(alpha + beta + theta, 1e-10))
            + 0.25 * (1.0 - delta / max(delta + beta, 1e-10)),
            0, 1
        ))
        dominance = _compute_dominance(beta_alpha_ratio, theta_beta_ratio)

        stress_index = float(np.clip(
            0.40 * min(1, beta_alpha_ratio * 0.3) + 0.25 * max(0, 1 - alpha * 2.5)
            + 0.20 * min(1, high_beta * 4) + 0.15 * min(1, gamma * 2), 0, 1))
        focus_index = float(np.clip(
            0.40 * min(1, beta * 3.5) + 0.35 * max(0, 1 - theta_beta_ratio * 0.40)
            + 0.15 * min(1, low_beta * 5) + 0.10 * min(1, gamma * 2), 0, 1))
        relaxation_index = float(np.clip(
            0.50 * min(1, alpha * 2.5) + 0.30 * max(0, 1 - beta_alpha_ratio * 0.3)
            + 0.20 * min(1, theta * 1.5), 0, 1))
        anger_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.0) * 0.5) + 0.25 * min(1, gamma * 3)
            + 0.25 * max(0, 1 - alpha * 5) + 0.15 * min(1, high_beta * 3), 0, 1))
        fear_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.5) * 0.6) + 0.30 * min(1, high_beta * 5)
            + 0.25 * max(0, 1 - alpha * 5) + 0.10 * max(0, arousal - 0.45), 0, 1))

        # EMA smooth continuous indices (same as all other paths)
        valence          = self._smooth_index("_ema_valence", valence)
        arousal          = self._smooth_index("_ema_arousal", arousal)
        stress_index     = self._smooth_index("_ema_stress", stress_index)
        focus_index      = self._smooth_index("_ema_focus", focus_index)
        relaxation_index = self._smooth_index("_ema_relaxation", relaxation_index)
        anger_index      = self._smooth_index("_ema_anger", anger_index)
        fear_index       = self._smooth_index("_ema_fear", fear_index)

        granular_emotions = map_vad_to_granular_emotions(valence, arousal, dominance)

        return {
            "emotion": EMOTIONS[emotion_idx] if emotion_idx < n_classes else "unknown",
            "emotion_index": emotion_idx,
            "confidence": float(probs[emotion_idx]) if emotion_idx < n_classes else 0.0,
            "probabilities": {EMOTIONS[i]: probs[i] for i in range(n_classes)},
            "valence": float(np.clip(valence, -1, 1)),
            "arousal": float(np.clip(arousal, 0, 1)),
            "dominance": dominance,
            "granular_emotions": granular_emotions,
            "stress_index": stress_index,
            "focus_index": focus_index,
            "relaxation_index": relaxation_index,
            "anger_index": anger_index,
            "fear_index": fear_index,
            "band_powers": bands,
            "differential_entropy": de,
            "explanation": [],
        }

    # ────────────────────────────────────────────────────────────────
    # Sklearn inference (only used if benchmark >= 60%)
    # ────────────────────────────────────────────────────────────────

    def _predict_sklearn(self, eeg: np.ndarray, fs: float) -> Dict:
        """Sklearn model inference using extracted features.

        Supports both:
          - Legacy single-channel 17-feature models (self.multichannel_model = False)
          - Multichannel 41-feature models trained with DASM/RASM/FAA features (multichannel = True)
        """
        from processing.eeg_processor import extract_features_multichannel as _mc_feats

        # ── Feature extraction ────────────────────────────────────
        if self.multichannel_model and eeg.ndim == 2:
            # Multichannel model: extract full 41-feature vector, apply stored scaler
            mc_dict = _mc_feats(eeg, fs)
            feature_vector = np.array(list(mc_dict.values()), dtype=np.float32).reshape(1, -1)
            if self.scaler is not None:
                feature_vector = self.scaler.transform(feature_vector)
            # Band powers from first channel (AF7) for downstream indices
            signal_for_bands = eeg[1] if eeg.shape[0] > 1 else eeg[0]
        else:
            signal_for_bands = eeg if eeg.ndim == 1 else eeg
            processed = preprocess(signal_for_bands, fs)
            features = extract_features(processed, fs)
            try:
                feature_vector = np.array([features.get(k, 0.0) for k in self.feature_names]).reshape(1, -1)
            except KeyError:
                return self._predict_features(eeg, fs)
            if self.scaler is not None:
                feature_vector = self.scaler.transform(feature_vector)
            signal_for_bands = processed

        processed = preprocess(signal_for_bands if signal_for_bands.ndim == 1
                                else signal_for_bands[0], fs)
        bands = extract_band_powers(processed, fs)
        de = differential_entropy(processed, fs)
        probs = self.sklearn_model.predict_proba(feature_vector)[0]
        emotion_idx = int(np.argmax(probs))

        alpha_raw     = bands.get("alpha",     0)
        beta_raw      = bands.get("beta",      0)
        theta_raw     = bands.get("theta",     0)
        gamma_raw     = bands.get("gamma",     0)
        delta_raw     = bands.get("delta",     0)
        high_beta_raw = bands.get("high_beta", 0)
        low_beta_raw  = bands.get("low_beta",  0)

        total_power = alpha_raw + beta_raw + theta_raw + gamma_raw + delta_raw
        if total_power < 1e-6:
            total_power = 1.0
        alpha     = alpha_raw     / total_power
        beta      = beta_raw      / total_power
        theta     = theta_raw     / total_power
        gamma     = gamma_raw     / total_power
        high_beta = high_beta_raw / total_power
        low_beta  = low_beta_raw  / total_power

        delta     = delta_raw     / total_power
        beta_alpha_ratio = beta / max(alpha, 1e-10)
        theta_beta_ratio = theta / max(beta, 1e-10)

        valence = float(np.clip(
            0.65 * np.tanh((alpha / max(beta, 1e-10) - 0.7) * 2.0)
            + 0.35 * np.tanh((alpha - 0.15) * 4),
            -1, 1
        ))
        arousal = float(np.clip(
            0.45 * beta / max(beta + alpha, 1e-10)
            + 0.30 * (1.0 - alpha / max(alpha + beta + theta, 1e-10))
            + 0.25 * (1.0 - delta / max(delta + beta, 1e-10)),
            0, 1
        ))
        dominance = _compute_dominance(beta_alpha_ratio, theta_beta_ratio)

        stress_index = float(np.clip(
            0.40 * min(1, beta_alpha_ratio * 0.3) + 0.25 * max(0, 1 - alpha * 2.5)
            + 0.20 * min(1, high_beta * 4) + 0.15 * min(1, gamma * 2), 0, 1))
        focus_index = float(np.clip(
            0.40 * min(1, beta * 3.5) + 0.35 * max(0, 1 - theta_beta_ratio * 0.40)
            + 0.15 * min(1, low_beta * 5) + 0.10 * min(1, gamma * 2), 0, 1))
        relaxation_index = float(np.clip(
            0.50 * min(1, alpha * 2.5) + 0.30 * max(0, 1 - beta_alpha_ratio * 0.3)
            + 0.20 * min(1, theta * 1.5), 0, 1))
        anger_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.0) * 0.5) + 0.25 * min(1, gamma * 3)
            + 0.25 * max(0, 1 - alpha * 5) + 0.15 * min(1, high_beta * 3), 0, 1))
        fear_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.5) * 0.6) + 0.30 * min(1, high_beta * 5)
            + 0.25 * max(0, 1 - alpha * 5) + 0.10 * max(0, arousal - 0.45), 0, 1))

        # EMA smooth continuous indices (same as all other paths)
        valence          = self._smooth_index("_ema_valence", valence)
        arousal          = self._smooth_index("_ema_arousal", arousal)
        stress_index     = self._smooth_index("_ema_stress", stress_index)
        focus_index      = self._smooth_index("_ema_focus", focus_index)
        relaxation_index = self._smooth_index("_ema_relaxation", relaxation_index)
        anger_index      = self._smooth_index("_ema_anger", anger_index)
        fear_index       = self._smooth_index("_ema_fear", fear_index)

        granular_emotions = map_vad_to_granular_emotions(valence, arousal, dominance)

        return {
            "emotion": EMOTIONS[emotion_idx],
            "emotion_index": emotion_idx,
            "confidence": float(probs[emotion_idx]),
            "probabilities": {EMOTIONS[i]: float(p) for i, p in enumerate(probs)},
            "valence": float(np.clip(valence, -1, 1)),
            "arousal": float(np.clip(arousal, 0, 1)),
            "dominance": dominance,
            "granular_emotions": granular_emotions,
            "stress_index": stress_index,
            "focus_index": focus_index,
            "relaxation_index": relaxation_index,
            "anger_index": anger_index,
            "fear_index": fear_index,
            "band_powers": bands,
            "differential_entropy": de,
            "explanation": [],
        }
