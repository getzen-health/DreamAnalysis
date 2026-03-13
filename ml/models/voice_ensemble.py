"""Voice emotion ensemble — stacking model for improved accuracy (issue #334).

Combines emotion2vec+ probabilities with acoustic features via a weighted
meta-learner and applies temporal smoothing to stabilise predictions.

Architecture:
    Layer 1 — Base models:
        * emotion2vec+ (via VoiceEmotionModel)
        * Acoustic biomarkers: pitch, energy, spectral centroid, MFCC-13

    Layer 2 — Meta-learner:
        * Weighted average: 70% emotion2vec+ + 30% acoustic adjustment
        * High pitch + high energy  → boost excited/angry
        * Low pitch + low energy    → boost sad/calm

    Layer 3 — Temporal smoothing:
        * Sliding window of last 5 predictions
        * Exponential moving average (decay=0.7)
        * Rapid-flip guard: ignores transitions within 10 seconds when the
          emotion flips more than once (e.g. angry→happy→angry)

Extensions (issue #366):
    SimpleSpeakerDiarizer  — energy-based speaker segmentation + pitch clustering
    LanguageDetector       — pitch statistics + phonetic density → ISO 639-1 code

Output:
    Same schema as VoiceEmotionModel + extra keys:
        ensemble_active: bool
        acoustic_features: dict
        smoothed: bool
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_6CLASS: List[str] = ["happy", "sad", "angry", "fear", "surprise", "neutral"]

# ── Layer 2 constants ──────────────────────────────────────────────────────────
_E2V_WEIGHT = 0.70       # weight for emotion2vec+ probabilities
_ACOUSTIC_WEIGHT = 0.30  # weight for acoustic-based adjustment

# ── Layer 3 constants ──────────────────────────────────────────────────────────
_WINDOW_SIZE = 5          # number of recent predictions to keep
_EMA_DECAY = 0.7          # EMA decay for temporal smoothing (higher = more weight on recent)
_FLIP_GUARD_SECS = 10.0   # guard window in seconds for rapid-flip detection
_MIN_SAMPLES_FOR_SMOOTH = 2  # need at least this many predictions before smoothing

# ── Acoustic feature thresholds (heuristics calibrated to typical voice) ───────
_PITCH_HIGH = 220.0       # Hz — above this is "high" pitch (excited/angry)
_PITCH_LOW = 130.0        # Hz — below this is "low" pitch (sad/calm)
_ENERGY_HIGH = 0.05       # RMS — above this is "high" energy
_ENERGY_LOW = 0.01        # RMS — below this is "low" energy


# ── Acoustic feature extraction ────────────────────────────────────────────────

def extract_acoustic_features(audio: np.ndarray, sr: int = 22050) -> Dict[str, float]:
    """Extract low-level acoustic features from a raw audio numpy array.

    All features are computed with numpy/scipy only — no heavy models needed.

    Returns:
        dict with keys:
            pitch_mean, pitch_std, pitch_range  — F0 statistics in Hz
            energy_mean, energy_std             — RMS energy statistics
            speaking_rate_proxy                 — normalised ZCR (syllable rate proxy)
            spectral_centroid_mean              — spectral centroid in Hz
            mfcc_{1..13}                        — first 13 MFCC coefficients (means)
    """
    result: Dict[str, float] = {
        "pitch_mean": 0.0,
        "pitch_std": 0.0,
        "pitch_range": 0.0,
        "energy_mean": 0.0,
        "energy_std": 0.0,
        "speaking_rate_proxy": 0.0,
        "spectral_centroid_mean": 0.0,
    }
    for i in range(1, 14):
        result[f"mfcc_{i}"] = 0.0

    if audio is None or len(audio) < sr // 10:  # < 0.1s
        return result

    y = audio.astype(np.float32)
    # Normalise to -1..1 safely
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak

    # ── Energy (RMS) ────────────────────────────────────────────────────────
    frame_len = int(sr * 0.025)   # 25 ms frames
    hop_len = int(sr * 0.010)     # 10 ms hop
    n_frames = max(1, (len(y) - frame_len) // hop_len + 1)
    rms_frames: List[float] = []
    for i in range(n_frames):
        start = i * hop_len
        frame = y[start: start + frame_len]
        if len(frame) < frame_len:
            break
        rms_frames.append(float(np.sqrt(np.mean(frame ** 2))))

    if rms_frames:
        result["energy_mean"] = float(np.mean(rms_frames))
        result["energy_std"] = float(np.std(rms_frames))

    # ── Pitch (F0) via autocorrelation ─────────────────────────────────────
    f0_values = _estimate_f0_autocorr(y, sr, frame_len, hop_len)
    if f0_values:
        arr = np.array(f0_values, dtype=np.float32)
        result["pitch_mean"] = float(arr.mean())
        result["pitch_std"] = float(arr.std())
        result["pitch_range"] = float(arr.max() - arr.min())

    # ── Speaking rate proxy (ZCR) ───────────────────────────────────────────
    zcr_frames: List[float] = []
    for i in range(n_frames):
        start = i * hop_len
        frame = y[start: start + frame_len]
        if len(frame) < frame_len:
            break
        crossings = float(np.sum(np.diff(np.sign(frame)) != 0)) / frame_len
        zcr_frames.append(crossings)
    if zcr_frames:
        # Normalise to ~0-1 range (typical ZCR * frame_len is 0-50ish)
        result["speaking_rate_proxy"] = float(min(1.0, np.mean(zcr_frames) * 20))

    # ── Spectral centroid ───────────────────────────────────────────────────
    try:
        fft_size = 512
        sc_values: List[float] = []
        for i in range(n_frames):
            start = i * hop_len
            frame = y[start: start + frame_len]
            if len(frame) < frame_len:
                break
            padded = np.zeros(fft_size, dtype=np.float32)
            padded[: len(frame)] = frame
            magnitude = np.abs(np.fft.rfft(padded))
            freqs = np.fft.rfftfreq(fft_size, d=1.0 / sr)
            denom = magnitude.sum()
            if denom > 1e-10:
                sc_values.append(float(np.sum(freqs * magnitude) / denom))
        if sc_values:
            result["spectral_centroid_mean"] = float(np.mean(sc_values))
    except Exception as exc:
        log.debug("Spectral centroid extraction failed: %s", exc)

    # ── MFCC-13 ─────────────────────────────────────────────────────────────
    try:
        mfcc_vec = _compute_mfcc_numpy(y, sr, n_mfcc=13)
        for i, val in enumerate(mfcc_vec[:13], start=1):
            result[f"mfcc_{i}"] = float(val)
    except Exception as exc:
        log.debug("MFCC extraction failed: %s", exc)

    return result


def _estimate_f0_autocorr(
    y: np.ndarray, sr: int, frame_len: int, hop_len: int
) -> List[float]:
    """Estimate F0 per frame via normalised autocorrelation.

    Returns only voiced frames (those where a clear pitch peak exists).
    F0 range searched: 60–500 Hz.
    """
    f0_min, f0_max = 60, 500
    lag_max = int(sr / f0_min)
    lag_min = int(sr / f0_max)
    f0_values: List[float] = []

    n_frames = max(1, (len(y) - frame_len) // hop_len + 1)
    for i in range(n_frames):
        start = i * hop_len
        frame = y[start: start + frame_len]
        if len(frame) < frame_len:
            break
        # Apply Hann window
        win = frame * np.hanning(len(frame))
        # Normalised autocorrelation
        ac = np.correlate(win, win, mode="full")
        ac = ac[len(ac) // 2:]  # keep positive lags
        ac_norm_denom = ac[0]
        if ac_norm_denom < 1e-10:
            continue
        ac_norm = ac / ac_norm_denom

        # Find peak in valid lag range
        search = ac_norm[lag_min: lag_max + 1]
        if len(search) == 0:
            continue
        peak_idx = int(np.argmax(search))
        peak_val = float(search[peak_idx])

        # Only count as voiced if autocorrelation peak is strong enough
        if peak_val >= 0.3:
            lag = lag_min + peak_idx
            if lag > 0:
                f0_values.append(sr / lag)

    return f0_values


def _compute_mfcc_numpy(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """Compute mean MFCC coefficients using numpy only (no librosa required).

    Uses a simplified mel filterbank and DCT.
    """
    # Short-time Fourier transform (power spectrum)
    frame_len = min(512, len(y))
    hop = frame_len // 2
    n_fft = frame_len
    n_frames = max(1, (len(y) - frame_len) // hop + 1)

    power_frames = np.zeros((n_frames, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = y[start: start + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        windowed = frame * np.hanning(frame_len)
        spectrum = np.fft.rfft(windowed, n=n_fft)
        power_frames[i] = np.abs(spectrum) ** 2

    # Mel filterbank (40 filters)
    n_filters = 40
    f_min, f_max = 80.0, float(sr) / 2.0
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft // 2)

    filterbank = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_filters + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]
        for k in range(f_left, f_center):
            if f_center > f_left:
                filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right > f_center:
                filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)

    # Apply filterbank → log energy → DCT to get MFCCs
    mel_energies = power_frames @ filterbank.T
    mel_energies = np.where(mel_energies > 1e-10, mel_energies, 1e-10)
    log_mel = np.log(mel_energies)

    # Type II DCT
    n_f = n_filters
    dct_matrix = np.cos(
        np.pi * np.arange(n_mfcc)[:, None] * (2 * np.arange(n_f)[None, :] + 1) / (2 * n_f)
    ).astype(np.float32)
    mfccs = log_mel @ dct_matrix.T  # shape: (n_frames, n_mfcc)

    return mfccs.mean(axis=0)


# ── Layer 2: acoustic-feature adjustment ──────────────────────────────────────

def _acoustic_adjustment(
    acoustic: Dict[str, float],
) -> Dict[str, float]:
    """Return an additive delta to apply to emotion2vec+ probabilities.

    Rules:
        high pitch + high energy → boost angry (0.10) and happy (0.08)
        high pitch + low energy  → boost fear (0.08) and surprise (0.06)
        low pitch + low energy   → boost sad (0.12) and neutral (0.06)
        low pitch + high energy  → boost angry (0.06)
        (all other emotions receive corresponding small reductions)

    Returns a delta dict (values can be negative) that sums to 0.
    """
    pitch = acoustic.get("pitch_mean", 0.0)
    energy = acoustic.get("energy_mean", 0.0)
    spectral = acoustic.get("spectral_centroid_mean", 0.0)

    high_pitch = pitch > _PITCH_HIGH
    low_pitch = (pitch > 0) and (pitch < _PITCH_LOW)
    high_energy = energy > _ENERGY_HIGH
    low_energy = (energy > 0) and (energy < _ENERGY_LOW)

    delta: Dict[str, float] = {e: 0.0 for e in _6CLASS}

    if high_pitch and high_energy:
        # Excited / angry / happy pattern
        delta["angry"] += 0.10
        delta["happy"] += 0.08
        delta["fear"] += 0.02
        # Spread reduction
        delta["sad"] -= 0.08
        delta["neutral"] -= 0.08
        delta["surprise"] -= 0.04

    elif high_pitch and low_energy:
        # Fearful / surprised / pleading pattern
        delta["fear"] += 0.08
        delta["surprise"] += 0.06
        delta["happy"] += 0.02
        delta["sad"] -= 0.06
        delta["angry"] -= 0.06
        delta["neutral"] -= 0.04

    elif low_pitch and low_energy:
        # Sad / depressed / exhausted pattern
        delta["sad"] += 0.12
        delta["neutral"] += 0.06
        delta["happy"] -= 0.08
        delta["angry"] -= 0.05
        delta["fear"] -= 0.03
        delta["surprise"] -= 0.02

    elif low_pitch and high_energy:
        # Low angry / dominant pattern
        delta["angry"] += 0.06
        delta["neutral"] += 0.03
        delta["happy"] -= 0.04
        delta["fear"] -= 0.03
        delta["surprise"] -= 0.02

    # Additional: bright timbre (high spectral centroid) boosts happy/surprise
    if spectral > 3000:
        brightness_boost = min(0.05, (spectral - 3000) / 5000)
        delta["happy"] += brightness_boost
        delta["surprise"] += brightness_boost * 0.5
        delta["sad"] -= brightness_boost * 0.8
        delta["neutral"] -= brightness_boost * 0.7

    return delta


def _blend_e2v_acoustic(
    e2v_probs: Dict[str, float],
    acoustic: Dict[str, float],
) -> Dict[str, float]:
    """Blend emotion2vec+ probs (70%) with acoustic adjustment (30%).

    The acoustic component is an additive delta scaled by _ACOUSTIC_WEIGHT,
    then the combined probabilities are renormalised to sum to 1.
    """
    delta = _acoustic_adjustment(acoustic)
    blended: Dict[str, float] = {}
    for e in _6CLASS:
        raw = (
            _E2V_WEIGHT * e2v_probs.get(e, 1.0 / 6)
            + _ACOUSTIC_WEIGHT * delta.get(e, 0.0)
        )
        blended[e] = max(0.0, raw)

    total = sum(blended.values()) or 1.0
    return {k: round(v / total, 4) for k, v in blended.items()}


# ── Layer 3: temporal smoothing ────────────────────────────────────────────────

class TemporalSmoother:
    """Sliding-window EMA smoother with rapid-flip guard.

    Maintains a rolling window of the last _WINDOW_SIZE predictions.
    Applies exponential moving average (decay=_EMA_DECAY) across the window.
    Detects and suppresses rapid emotion flips (e.g. angry→happy→angry within
    _FLIP_GUARD_SECS seconds).
    """

    def __init__(
        self,
        window_size: int = _WINDOW_SIZE,
        ema_decay: float = _EMA_DECAY,
        flip_guard_secs: float = _FLIP_GUARD_SECS,
    ) -> None:
        self._window: Deque[Tuple[float, Dict[str, float], str]] = deque(
            maxlen=window_size
        )
        self._decay = ema_decay
        self._flip_guard = flip_guard_secs
        self._last_committed_emotion: Optional[str] = None
        self._last_flip_ts: float = 0.0

    def reset(self) -> None:
        """Clear all history (call between unrelated audio sessions)."""
        self._window.clear()
        self._last_committed_emotion = None
        self._last_flip_ts = 0.0

    def update(
        self, probs: Dict[str, float], emotion: str, ts: Optional[float] = None
    ) -> Dict[str, float]:
        """Add a new prediction and return smoothed probabilities.

        Args:
            probs:   Raw blended probability dict (6-class, sums to 1).
            emotion: The argmax emotion for the current prediction.
            ts:      Unix timestamp for this prediction (default: now).

        Returns:
            Smoothed probability dict (6-class, sums to 1).
        """
        if ts is None:
            ts = time.time()

        self._window.append((ts, probs, emotion))

        if len(self._window) < _MIN_SAMPLES_FOR_SMOOTH:
            return probs

        # ── EMA across the window ──────────────────────────────────────────
        # Most recent prediction has highest weight.
        weights = [self._decay ** i for i in range(len(self._window) - 1, -1, -1)]
        w_total = sum(weights)

        smoothed: Dict[str, float] = {e: 0.0 for e in _6CLASS}
        for w, (_, p, _e) in zip(weights, self._window):
            for e in _6CLASS:
                smoothed[e] += w * p.get(e, 0.0)

        smoothed = {k: round(v / w_total, 4) for k, v in smoothed.items()}

        # ── Rapid-flip guard ───────────────────────────────────────────────
        smoothed_emotion = max(smoothed, key=smoothed.__getitem__)
        if (
            self._last_committed_emotion is not None
            and smoothed_emotion != self._last_committed_emotion
        ):
            # Check if we flipped back to a recent emotion within the guard window
            recent_emotions = [_e for (_, _, _e) in self._window]
            if (
                len(recent_emotions) >= 3
                and recent_emotions[-1] != recent_emotions[-2]
                and recent_emotions[-1] == recent_emotions[-3]
                and (ts - self._last_flip_ts) < self._flip_guard
            ):
                # Suppress the flip — return probs with previous emotion boosted
                log.debug(
                    "Rapid-flip guard: suppressing %s→%s within %.1fs",
                    self._last_committed_emotion,
                    smoothed_emotion,
                    ts - self._last_flip_ts,
                )
                # Give previous emotion the top spot
                prev = self._last_committed_emotion
                smoothed[prev] = max(smoothed[prev], smoothed[smoothed_emotion] + 0.01)
                total = sum(smoothed.values()) or 1.0
                smoothed = {k: round(v / total, 4) for k, v in smoothed.items()}
                return smoothed

            self._last_flip_ts = ts

        self._last_committed_emotion = smoothed_emotion
        return smoothed


# ── Public ensemble class ──────────────────────────────────────────────────────

class VoiceEnsemble:
    """Stacking ensemble for voice emotion detection.

    Layers:
        1. emotion2vec+ probabilities (via VoiceEmotionModel)
        2. Acoustic feature extraction + weighted blending
        3. Temporal EMA smoothing + rapid-flip guard

    Usage:
        ensemble = VoiceEnsemble()
        result = ensemble.predict(audio, sample_rate=22050)
        # result contains 'ensemble_active', 'acoustic_features', 'smoothed' keys
    """

    def __init__(self) -> None:
        self._smoother = TemporalSmoother()
        self._voice_model = None

    def _get_voice_model(self):
        if self._voice_model is None:
            try:
                from models.voice_emotion_model import get_voice_model  # type: ignore
                self._voice_model = get_voice_model()
            except Exception as exc:
                log.warning("VoiceEmotionModel unavailable in ensemble: %s", exc)
        return self._voice_model

    def reset_temporal_state(self) -> None:
        """Reset smoother between unrelated audio sessions."""
        self._smoother.reset()

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int = 22050,
        ts: Optional[float] = None,
        apply_temporal_smoothing: bool = True,
        **kwargs,
    ) -> Optional[Dict]:
        """Run the 3-layer ensemble.

        Args:
            audio:                   Raw 1D float32 audio array.
            sample_rate:             Sampling rate in Hz.
            ts:                      Timestamp for this prediction (default: now).
            apply_temporal_smoothing: Set False for single-shot predictions.

        Returns:
            Emotion dict with all standard keys plus:
                ensemble_active:   True (always, if we reach this point)
                acoustic_features: dict of extracted acoustic features
                smoothed:          True if temporal smoothing was applied
            Returns None if audio is too short or all models fail.
        """
        if ts is None:
            ts = time.time()

        if audio is None or len(audio) < sample_rate // 2:
            return None

        # ── Layer 1: emotion2vec+ base prediction ──────────────────────────
        vm = self._get_voice_model()
        base_result: Optional[Dict] = None
        if vm is not None:
            try:
                base_result = vm.predict(audio, sample_rate=sample_rate, **kwargs)
            except Exception as exc:
                log.warning("Base model prediction failed in ensemble: %s", exc)

        if base_result is None:
            log.debug("Base model returned None — ensemble cannot proceed")
            return None

        e2v_probs: Dict[str, float] = {
            e: float(base_result.get("probabilities", {}).get(e, 1.0 / 6))
            for e in _6CLASS
        }

        # ── Acoustic feature extraction ────────────────────────────────────
        acoustic: Dict[str, float] = {}
        try:
            acoustic = extract_acoustic_features(audio, sr=sample_rate)
        except Exception as exc:
            log.warning("Acoustic feature extraction failed: %s", exc)

        # ── Layer 2: meta-learner blend ────────────────────────────────────
        try:
            blended_probs = _blend_e2v_acoustic(e2v_probs, acoustic)
        except Exception as exc:
            log.warning("Acoustic blend failed — using raw e2v probs: %s", exc)
            blended_probs = e2v_probs

        emotion = max(blended_probs, key=blended_probs.__getitem__)

        # ── Layer 3: temporal smoothing ────────────────────────────────────
        smoothed = False
        final_probs = blended_probs
        if apply_temporal_smoothing:
            try:
                final_probs = self._smoother.update(blended_probs, emotion, ts=ts)
                emotion = max(final_probs, key=final_probs.__getitem__)
                smoothed = True
            except Exception as exc:
                log.warning("Temporal smoothing failed — using blended probs: %s", exc)

        # Re-derive valence/arousal/confidence from final probs
        from models.voice_emotion_model import _valence_arousal  # type: ignore
        valence, arousal = _valence_arousal(final_probs)
        confidence = final_probs[emotion]

        result: Dict = {
            "emotion": emotion,
            "probabilities": final_probs,
            "valence": round(float(valence), 4),
            "arousal": round(float(arousal), 4),
            "confidence": round(float(confidence), 4),
            "model_type": base_result.get("model_type", "voice_feature_heuristic") + "_ensemble",
            "ensemble_active": True,
            "acoustic_features": acoustic,
            "smoothed": smoothed,
        }

        # Forward biomarkers if the base model provided them
        if "biomarkers" in base_result:
            result["biomarkers"] = base_result["biomarkers"]
        if "mental_health" in base_result:
            result["mental_health"] = base_result["mental_health"]

        return result


# ── Module-level singleton ─────────────────────────────────────────────────────

_ensemble_instance: Optional[VoiceEnsemble] = None


def get_voice_ensemble() -> VoiceEnsemble:
    """Return (or create) the module-level VoiceEnsemble singleton."""
    global _ensemble_instance
    if _ensemble_instance is None:
        _ensemble_instance = VoiceEnsemble()
    return _ensemble_instance


# ── Per-user voice baseline calibration ───────────────────────────────────────

# Keys in the acoustic feature dict that are used for calibration normalization.
# These map directly to the keys returned by extract_acoustic_features().
_CALIBRATION_KEYS: List[str] = [
    "pitch_mean",
    "pitch_std",
    "energy_mean",
    "energy_std",
    "speaking_rate_proxy",
    "spectral_centroid_mean",
]

_MIN_CALIBRATION_FRAMES = 10  # minimum frames before calibration is considered ready


class VoiceBaselineCalibrator:
    """Per-user voice baseline calibration for acoustic feature normalization.

    Collects neutral-speech acoustic features over ~30s (multiple frames) and
    computes a personal baseline (mean + std per feature).  All subsequent
    feature dicts can then be z-score normalized against that baseline before
    emotion classification, correcting for individual voice variation such as
    naturally high/low pitch, energy range, and speaking rate.

    Mirrors the ``BaselineCalibrator`` pattern in
    ``ml/processing/eeg_processor.py``.

    Usage::

        cal = VoiceBaselineCalibrator()
        # During neutral-speech calibration phase (send 10+ frames of ~1s audio):
        for audio_features in neutral_frames:
            cal.add_frame(audio_features)
        # During live analysis:
        if cal.is_ready:
            norm_features = cal.normalize(raw_acoustic_features)

    Calibration features (subset of acoustic feature dict):
        pitch_mean, pitch_std, energy_mean, energy_std,
        speaking_rate_proxy, spectral_centroid_mean
    """

    _MIN_FRAMES: int = _MIN_CALIBRATION_FRAMES

    def __init__(self) -> None:
        self._frames: List[Dict[str, float]] = []
        self._mean: Dict[str, float] = {}
        self._std: Dict[str, float] = {}
        self._ready: bool = False

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True once enough calibration frames have been collected."""
        return self._ready

    @property
    def n_frames(self) -> int:
        """Number of calibration frames accumulated so far."""
        return len(self._frames)

    # ── Frame accumulation ──────────────────────────────────────────────────

    def add_frame(self, audio_features: Dict[str, float]) -> bool:
        """Accumulate one neutral-speech acoustic feature frame.

        Automatically triggers statistics computation when ``_MIN_FRAMES``
        frames have been collected.

        Args:
            audio_features: dict returned by ``extract_acoustic_features()``.
                Only the keys listed in ``_CALIBRATION_KEYS`` are used.

        Returns:
            True if calibration just became ready (crossed the threshold on
            this call), False otherwise.
        """
        # Keep only the relevant calibration keys; fill missing keys with 0
        frame: Dict[str, float] = {
            k: float(audio_features.get(k, 0.0)) for k in _CALIBRATION_KEYS
        }
        self._frames.append(frame)

        if len(self._frames) >= self._MIN_FRAMES and not self._ready:
            self._compute_statistics()
            return True
        return False

    def _compute_statistics(self) -> None:
        """Compute per-feature mean and std across all accumulated frames."""
        for k in _CALIBRATION_KEYS:
            vals = [f[k] for f in self._frames]
            self._mean[k] = float(np.mean(vals))
            self._std[k] = float(np.std(vals))
        self._ready = True
        log.debug(
            "VoiceBaselineCalibrator ready after %d frames — mean pitch=%.1f Hz",
            len(self._frames),
            self._mean.get("pitch_mean", 0.0),
        )

    # ── Normalization ───────────────────────────────────────────────────────

    def normalize(self, features: Dict[str, float]) -> Dict[str, float]:
        """Z-score normalize a feature dict against the personal baseline.

        Only calibration keys are normalized; all other keys pass through
        unchanged.  Features with near-zero baseline variance are set to 0.0
        to avoid division-by-zero issues (constant-at-rest features carry no
        discriminative information anyway).

        Returns the features dict unchanged if calibration is not yet ready.
        """
        if not self._ready:
            return features

        out = dict(features)
        for k in _CALIBRATION_KEYS:
            if k not in features:
                continue
            std = self._std.get(k, 0.0)
            mean = self._mean.get(k, features[k])
            out[k] = float((features[k] - mean) / std) if std > 1e-8 else 0.0
        return out

    # ── Status & lifecycle ──────────────────────────────────────────────────

    def get_status(self) -> Dict[str, object]:
        """Return calibration progress information.

        Returns:
            dict with keys:
                is_ready      — bool: calibration complete
                n_frames      — int: frames collected so far
                frames_needed — int: minimum frames required
                progress_pct  — float: 0-100 completion percentage
                baseline_mean — dict[str, float]: per-feature means (empty if not ready)
        """
        return {
            "is_ready": self._ready,
            "n_frames": len(self._frames),
            "frames_needed": self._MIN_FRAMES,
            "progress_pct": round(
                min(100.0, len(self._frames) / self._MIN_FRAMES * 100), 1
            ),
            "baseline_mean": dict(self._mean) if self._ready else {},
        }

    def get_optimal_threshold(self) -> float:
        """Return a calibrated confidence threshold based on voice quality.

        The threshold is derived from the mean Harmonics-to-Noise Ratio (HNR)
        observed across the calibration frames.  HNR is a proxy for vocal
        clarity: high HNR → clean, tonal voice → the model is more likely to
        produce reliable high-confidence predictions → use a stricter threshold.
        Low HNR → breathy, nasal, or noisy voice → the model may produce lower
        peak probabilities for correct predictions → use a more lenient threshold.

        Mapping:
            HNR ≥ 20 dB  (clear voice):  threshold = 0.80
            HNR ≥ 15 dB  (good voice):   threshold ≈ 0.70
            HNR ≥ 10 dB  (average):      threshold ≈ 0.60
            HNR <  10 dB  (breathy/noisy):threshold ≈ 0.50
            HNR ≤ 5 dB   (very noisy):   threshold = 0.45

        When calibration is not yet ready (``is_ready == False``), or when no
        HNR data is available in the accumulated frames, the global default of
        0.60 is returned unchanged.

        Returns:
            float in [0.45, 0.80] — the recommended per-user confidence
            threshold for ``VoiceEmotionModel.predict(user_threshold=...)``.
        """
        _GLOBAL_DEFAULT = 0.60
        if not self._ready:
            return _GLOBAL_DEFAULT

        # HNR may not be present in all acoustic feature frames (it is an
        # optional key that arrives from voice_biomarkers extraction).
        hnr_values = [
            f["hnr"] for f in self._frames if "hnr" in f and f["hnr"] > 0
        ]

        if not hnr_values:
            # HNR not available in calibration data — return global default
            return _GLOBAL_DEFAULT

        mean_hnr = float(np.mean(hnr_values))

        # Linear interpolation between anchor points:
        #   HNR ≤ 5 dB  → threshold = 0.45
        #   HNR = 10 dB → threshold = 0.55
        #   HNR = 15 dB → threshold = 0.65
        #   HNR ≥ 20 dB → threshold = 0.80
        if mean_hnr >= 20.0:
            threshold = 0.80
        elif mean_hnr >= 15.0:
            # Linear: 15→0.65, 20→0.80
            threshold = 0.65 + (mean_hnr - 15.0) / 5.0 * (0.80 - 0.65)
        elif mean_hnr >= 10.0:
            # Linear: 10→0.55, 15→0.65
            threshold = 0.55 + (mean_hnr - 10.0) / 5.0 * (0.65 - 0.55)
        elif mean_hnr >= 5.0:
            # Linear: 5→0.45, 10→0.55
            threshold = 0.45 + (mean_hnr - 5.0) / 5.0 * (0.55 - 0.45)
        else:
            threshold = 0.45

        threshold = round(float(np.clip(threshold, 0.45, 0.80)), 4)
        log.debug(
            "VoiceBaselineCalibrator.get_optimal_threshold: HNR=%.1f dB → "
            "threshold=%.2f",
            mean_hnr,
            threshold,
        )
        return threshold

    def reset(self) -> None:
        """Clear all calibration data (call to restart the calibration phase)."""
        self._frames.clear()
        self._mean.clear()
        self._std.clear()
        self._ready = False
        log.debug("VoiceBaselineCalibrator reset")


# ── Per-user calibrator registry ───────────────────────────────────────────────

_calibrator_registry: Dict[str, VoiceBaselineCalibrator] = {}


def get_voice_calibrator(user_id: str) -> VoiceBaselineCalibrator:
    """Return (or create) a per-user VoiceBaselineCalibrator instance."""
    if user_id not in _calibrator_registry:
        _calibrator_registry[user_id] = VoiceBaselineCalibrator()
    return _calibrator_registry[user_id]


# ── Issue #366: Speaker diarization + language detection ───────────────────────

@dataclass
class SpeakerSegment:
    """A contiguous audio segment attributed to one speaker.

    Attributes:
        speaker_id:    Label string, e.g. "Speaker_A", "Speaker_B".
        start_time:    Start of segment in seconds.
        end_time:      End of segment in seconds.
        audio_segment: Raw float32 audio samples for this segment.
    """
    speaker_id: str
    start_time: float
    end_time: float
    audio_segment: np.ndarray = field(repr=False)


# Silence threshold: gaps longer than this (in seconds) mark a new segment
_SILENCE_GAP_SEC = 0.500
# RMS level below which a frame is considered silent
_SILENCE_RMS_THRESHOLD = 0.01
# Frame length for energy analysis (25 ms)
_DIARIZE_FRAME_SEC = 0.025

# Pitch ranges for speaker clustering (Hz) — 3 non-overlapping buckets
# We assign each segment to a bucket based on its mean pitch, then merge
# adjacent segments in the same bucket into a single labelled speaker.
_PITCH_BUCKETS: List[Tuple[float, float]] = [
    (60.0, 160.0),   # typically male voice range
    (160.0, 260.0),  # mixed / higher male or low female
    (260.0, 500.0),  # typically female voice range
]


class SimpleSpeakerDiarizer:
    """Lightweight energy-based speaker segmentation with pitch clustering.

    Algorithm:
        1. Split audio into frames (25 ms).
        2. Mark silent frames (RMS < threshold).
        3. Merge consecutive non-silent frames separated by <500 ms of silence
           into raw activity segments.
        4. Estimate mean pitch (F0 via autocorrelation) for each segment.
        5. Assign each segment to a pitch bucket → speaker label (Speaker_A/B/C).

    No neural models or external libraries are required.
    """

    def diarize(self, audio: np.ndarray, sr: int) -> List[SpeakerSegment]:
        """Segment audio by speaker using energy gaps and pitch clustering.

        Args:
            audio: 1-D float32 audio array.
            sr:    Sample rate in Hz.

        Returns:
            List of SpeakerSegment, in chronological order.
            Returns an empty list if audio is too short (< 0.5 s).
        """
        if audio is None or len(audio) < sr // 2:
            return []

        y = audio.astype(np.float32)
        peak = np.abs(y).max()
        if peak > 1e-9:
            y = y / peak

        frame_len = max(1, int(sr * _DIARIZE_FRAME_SEC))
        hop_len = frame_len  # non-overlapping for simplicity
        n_frames = len(y) // frame_len

        # ── Step 1-2: compute per-frame RMS and flag silence ───────────────
        is_silent = np.zeros(n_frames, dtype=bool)
        for i in range(n_frames):
            frame = y[i * frame_len: (i + 1) * frame_len]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            is_silent[i] = rms < _SILENCE_RMS_THRESHOLD

        # ── Step 3: merge activity into raw segments ───────────────────────
        silence_frames_threshold = max(
            1, int(_SILENCE_GAP_SEC / _DIARIZE_FRAME_SEC)
        )
        raw_segments: List[Tuple[int, int]] = []  # (start_frame, end_frame)
        seg_start: Optional[int] = None
        silent_run = 0

        for i in range(n_frames):
            if not is_silent[i]:
                if seg_start is None:
                    seg_start = i
                silent_run = 0
            else:
                silent_run += 1
                if seg_start is not None and silent_run >= silence_frames_threshold:
                    raw_segments.append((seg_start, i - silent_run))
                    seg_start = None
                    silent_run = 0

        # Close any open segment at end of audio
        if seg_start is not None:
            raw_segments.append((seg_start, n_frames - 1))

        if not raw_segments:
            # No speech detected — return whole audio as one segment
            return [
                SpeakerSegment(
                    speaker_id="Speaker_A",
                    start_time=0.0,
                    end_time=len(y) / sr,
                    audio_segment=y,
                )
            ]

        # ── Step 4: estimate mean pitch per segment ────────────────────────
        pitches: List[float] = []
        for start_f, end_f in raw_segments:
            seg_audio = y[start_f * frame_len: (end_f + 1) * frame_len]
            f0_vals = _estimate_f0_autocorr(seg_audio, sr, frame_len, frame_len)
            pitches.append(float(np.mean(f0_vals)) if f0_vals else 0.0)

        # ── Step 5: assign pitch bucket → speaker label ────────────────────
        bucket_to_label: Dict[int, str] = {}
        label_counter = 0
        labels = []
        for p in pitches:
            bucket = self._pitch_bucket(p)
            if bucket not in bucket_to_label:
                bucket_to_label[bucket] = f"Speaker_{chr(65 + label_counter)}"
                label_counter += 1
            labels.append(bucket_to_label[bucket])

        # ── Build SpeakerSegment list ──────────────────────────────────────
        segments: List[SpeakerSegment] = []
        for (start_f, end_f), label in zip(raw_segments, labels):
            start_sample = start_f * frame_len
            end_sample = min((end_f + 1) * frame_len, len(y))
            seg = SpeakerSegment(
                speaker_id=label,
                start_time=start_sample / sr,
                end_time=end_sample / sr,
                audio_segment=y[start_sample:end_sample],
            )
            segments.append(seg)

        return segments

    @staticmethod
    def _pitch_bucket(f0: float) -> int:
        """Return bucket index for a given F0 value (0, 1, or 2)."""
        for idx, (lo, hi) in enumerate(_PITCH_BUCKETS):
            if lo <= f0 < hi:
                return idx
        if f0 < _PITCH_BUCKETS[0][0]:
            return 0   # below min → bucket 0
        return len(_PITCH_BUCKETS) - 1  # above max → last bucket


# ── Language detection (issue #366) ───────────────────────────────────────────

# Language-specific pitch thresholds (mean F0 in Hz) for voice baseline calibration.
# These are population-average speaking-pitch midpoints from literature.
# Used to adjust pitch-based heuristics when language shifts expected register.
LANGUAGE_PITCH_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "en": {"pitch_low": 130.0, "pitch_high": 220.0},
    "es": {"pitch_low": 135.0, "pitch_high": 225.0},
    "fr": {"pitch_low": 145.0, "pitch_high": 235.0},
    "de": {"pitch_low": 120.0, "pitch_high": 210.0},
    "zh": {"pitch_low": 155.0, "pitch_high": 260.0},
    "ja": {"pitch_low": 140.0, "pitch_high": 245.0},
    "ko": {"pitch_low": 145.0, "pitch_high": 240.0},
    "hi": {"pitch_low": 150.0, "pitch_high": 250.0},
    "ar": {"pitch_low": 125.0, "pitch_high": 215.0},
}

# Per-language zero-crossing rate (ZCR) density — phonetically dense languages
# have higher average ZCR due to consonant clusters and tonal variation.
# Values are normalised proxies (0-1 scale from speaking_rate_proxy units).
_LANGUAGE_ZCR_SIGNATURES: Dict[str, float] = {
    "zh": 0.75,  # tonal, dense
    "ja": 0.70,  # mora-timed, many short syllables
    "ko": 0.68,
    "ar": 0.60,
    "hi": 0.62,
    "es": 0.55,
    "fr": 0.52,
    "en": 0.48,
    "de": 0.45,  # stress-timed, fewer syllables per second
}

# Spectral centroid ranges differ across languages due to phoneme inventory.
# Tonal/dense languages → higher centroid variance; stress-timed → lower.
_LANGUAGE_CENTROID_SIGNATURES: Dict[str, float] = {
    "zh": 2800.0,
    "ja": 2700.0,
    "ko": 2750.0,
    "hi": 2600.0,
    "ar": 2500.0,
    "es": 2400.0,
    "fr": 2450.0,
    "en": 2300.0,
    "de": 2200.0,
}


class LanguageDetector:
    """Lightweight language detector using pitch statistics and phonetic density.

    Does NOT require a speech recognition model or any external dependency.
    Uses three acoustic proxies:
        1. Mean F0 (pitch) — varies by language phonology and register.
        2. ZCR density (speaking_rate_proxy) — tonal/syllable-timed languages
           have systematically higher zero-crossing rates.
        3. Spectral centroid — correlated with average place of articulation
           and consonant-vowel density.

    Each proxy is scored against per-language signatures using L1 distance,
    the three distances are weighted (pitch 50%, ZCR 30%, centroid 20%) and
    summed, then the language with the lowest combined distance wins.

    Supported ISO 639-1 codes: en, es, fr, de, zh, ja, ko, hi, ar.
    Returns "en" as the default when detection confidence is low.
    """

    # Minimum audio length to attempt detection (seconds)
    _MIN_DURATION_SEC: float = 1.0

    def detect_language(self, audio: np.ndarray, sr: int) -> str:
        """Detect spoken language from acoustic features.

        Args:
            audio: 1-D float32 audio array (at least 1 second recommended).
            sr:    Sample rate in Hz.

        Returns:
            ISO 639-1 language code string (e.g. "en", "zh").
            Falls back to "en" if audio is too short or features are zero.
        """
        if audio is None or len(audio) < int(sr * self._MIN_DURATION_SEC):
            return "en"

        feats = extract_acoustic_features(audio, sr=sr)
        pitch_mean = feats.get("pitch_mean", 0.0)
        zcr = feats.get("speaking_rate_proxy", 0.0)
        centroid = feats.get("spectral_centroid_mean", 0.0)

        # If extraction produced all zeros (very short/silent audio) → default
        if pitch_mean == 0.0 and zcr == 0.0:
            return "en"

        # Normalisation ranges for L1 distance scoring
        PITCH_RANGE = 150.0    # typical spread across languages (Hz)
        ZCR_RANGE = 0.40       # typical ZCR spread (normalised units)
        CENTROID_RANGE = 800.0  # typical centroid spread (Hz)

        best_lang = "en"
        best_score = float("inf")

        for lang in _LANGUAGE_ZCR_SIGNATURES:
            pitch_ref = (
                LANGUAGE_PITCH_THRESHOLDS[lang]["pitch_low"]
                + LANGUAGE_PITCH_THRESHOLDS[lang]["pitch_high"]
            ) / 2.0
            zcr_ref = _LANGUAGE_ZCR_SIGNATURES[lang]
            centroid_ref = _LANGUAGE_CENTROID_SIGNATURES[lang]

            d_pitch = abs(pitch_mean - pitch_ref) / PITCH_RANGE if pitch_mean > 0 else 0.5
            d_zcr = abs(zcr - zcr_ref) / ZCR_RANGE if zcr > 0 else 0.5
            d_centroid = abs(centroid - centroid_ref) / CENTROID_RANGE if centroid > 0 else 0.5

            score = 0.50 * d_pitch + 0.30 * d_zcr + 0.20 * d_centroid

            if score < best_score:
                best_score = score
                best_lang = lang

        log.debug(
            "LanguageDetector: detected=%s score=%.3f pitch=%.1f zcr=%.3f centroid=%.0f",
            best_lang,
            best_score,
            pitch_mean,
            zcr,
            centroid,
        )
        return best_lang
