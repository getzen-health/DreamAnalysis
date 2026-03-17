"""Voice emotion detection: emotion2vec+ (primary) with LightGBM fallback.

Output format matches EEG EmotionClassifier (6-class):
    emotion: "happy"|"sad"|"angry"|"fear"|"surprise"|"neutral"
    probabilities: {emotion: 0.0-1.0, ...}
    valence: -1.0 to 1.0
    arousal: 0.0 to 1.0
    confidence: 0.0 to 1.0
    model_type: "voice_emotion2vec" | "voice_distilhubert" | "voice_lgbm_fallback"

Compound emotion detection (issue #375):
    After base 6-class classification, compound emotions are detected by
    examining the raw probability distribution against acoustic cues.
    Compound labels (12-class extension):
        "nostalgic"   — sad + happy both > 0.2 (bittersweet blend)
        "content"     — happy dominant + low arousal (pitch variance < baseline)
        "bored"       — neutral dominant + low energy + slow speaking rate
        "anticipation"— happy + fear both present + high arousal
        "frustrated"  — angry moderate (0.3–0.6) + sad > 0.15
        "guilt"       — sad + fear both > 0.2 + low pitch

Fallback chain (highest to lowest priority):
    1. emotion2vec_plus_large (funasr, 300M) — 9-class, ACL 2024, highest accuracy
    2. emotion2vec_plus_base (funasr) — 9-class, faster, lower memory
    3. DistilHuBERT SUPERB-ER (transformers) — 4-class, 23 MB, 34 ms, no funasr needed
    4. LightGBM MFCC (pkl) — 3-class, requires audio_emotion_lgbm.pkl
    5. Feature-based heuristics — no model file needed
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

_ML_ROOT = Path(__file__).resolve().parent.parent
_LGBM_PATH = _ML_ROOT / "models" / "saved" / "audio_emotion_lgbm.pkl"

# emotion2vec+ 9-class labels and mapping to 6-class
_E2V_LABELS: List[str] = [
    "angry", "disgusted", "fearful", "happy", "neutral",
    "other", "sad", "surprised", "unknown",
]
_E2V_MAP: Dict[str, str] = {
    "angry":     "angry",
    "disgusted": "angry",    # merge
    "fearful":   "fear",
    "happy":     "happy",
    "neutral":   "neutral",
    "other":     "neutral",  # merge
    "sad":       "sad",
    "surprised": "surprise",
    "unknown":   "neutral",  # merge
}
_6CLASS: List[str] = ["happy", "sad", "angry", "fear", "surprise", "neutral"]

_MIN_SAMPLES = 8000  # ~0.5s at 16 kHz minimum

# ── 12-class compound emotion labels (issue #375) ──────────────────────────────
# The base 6 classes are augmented with 6 compound emotions detected from the
# probability distribution + acoustic cues.  Compound labels are only assigned
# when a clear pattern is found; otherwise the base label is preserved.
_COMPOUND_EMOTIONS: List[str] = [
    "nostalgic",    # sad + happy both > 0.2 (bittersweet)
    "content",      # happy dominant + low arousal (low pitch variance)
    "bored",        # neutral dominant + low energy + slow speaking rate
    "anticipation", # happy + fear both present + high arousal
    "frustrated",   # angry moderate (0.3-0.6) + sad > 0.15
    "guilt",        # sad + fear both > 0.2 + low pitch
]
_ALL_12_EMOTIONS: List[str] = _6CLASS + _COMPOUND_EMOTIONS

# Acoustic thresholds used by compound detection
_COMPOUND_AROUSAL_HIGH = 0.55   # above this → "high arousal"
_COMPOUND_AROUSAL_LOW = 0.35    # below this → "low arousal"
_COMPOUND_PITCH_VARIANCE_LOW = 30.0  # Hz — below this = flat/monotone delivery

# ── Quality thresholds ─────────────────────────────────────────────────────────
_CONFIDENCE_THRESHOLD = 0.60  # global fallback — per-user threshold takes priority
_WINDOW_DURATION_S = 3.0      # each analysis window in seconds
_WINDOW_HOP_S = 1.5           # hop between windows (50% overlap)
_MIN_WINDOWS = 1              # minimum windows needed to aggregate


def detect_compound_emotion(
    probs: Dict[str, float],
    arousal: float,
    pitch_std: float = 0.0,
    energy_mean: float = 0.0,
    speaking_rate: float = 0.5,
    pitch_mean: float = 150.0,
) -> Optional[str]:
    """Detect compound emotion from base probabilities + acoustic cues.

    Runs AFTER base 6-class classification.  Returns a compound label string
    when the probability pattern and acoustic features match one of the six
    compound patterns.  Returns None if no compound pattern is matched, meaning
    the base emotion label should be used unchanged.

    Args:
        probs:         6-class probability dict from the base classifier.
        arousal:       Arousal score (0.0–1.0) derived from the prediction.
        pitch_std:     F0 standard deviation in Hz (measure of pitch variance).
        energy_mean:   Mean RMS energy of the audio segment.
        speaking_rate: Speaking rate proxy (0–1, higher = faster).
        pitch_mean:    Mean F0 in Hz (used to detect low-pitch delivery).

    Returns:
        Compound emotion label string, or None.

    Compound patterns (ordered by priority — first match wins):

        1. "anticipation" — happy AND fear signals both present + high arousal
           Rationale: excitement mixed with nervousness, e.g. before a big event.
           Conditions: happy > 0.20, fear > 0.20, arousal > _COMPOUND_AROUSAL_HIGH

        2. "nostalgic" — sad AND happy signals both present (bittersweet blend)
           Conditions: sad > 0.20, happy > 0.20
           (does not require arousal signal — can be low or medium)

        3. "frustrated" — moderate anger + underlying sadness
           Conditions: 0.30 ≤ angry ≤ 0.60, sad > 0.15

        4. "guilt" — sad + fear both present + low pitch delivery
           Conditions: sad > 0.20, fear > 0.20, pitch_mean < 150 Hz
           (low pitch = quiet, subdued delivery typical of guilt)

        5. "content" — happy dominant but low arousal + low pitch variance
           Conditions: happy is argmax, arousal < _COMPOUND_AROUSAL_LOW,
                       pitch_std < _COMPOUND_PITCH_VARIANCE_LOW

        6. "bored" — neutral dominant + low energy + slow speaking rate
           Conditions: neutral is argmax, energy_mean < 0.02,
                       speaking_rate < 0.30
    """
    happy    = probs.get("happy",    0.0)
    sad      = probs.get("sad",      0.0)
    angry    = probs.get("angry",    0.0)
    fear     = probs.get("fear",     0.0)
    neutral  = probs.get("neutral",  0.0)
    surprise = probs.get("surprise", 0.0)  # noqa: F841 — not used in compound rules

    dominant = max(probs, key=probs.__getitem__)

    # 1. anticipation: happy + fear both present + high arousal
    if (
        happy > 0.20
        and fear > 0.20
        and arousal > _COMPOUND_AROUSAL_HIGH
    ):
        return "anticipation"

    # 2. nostalgic: sad + happy bittersweet blend
    if sad > 0.20 and happy > 0.20:
        return "nostalgic"

    # 3. frustrated: moderate anger + underlying sadness
    if 0.30 <= angry <= 0.60 and sad > 0.15:
        return "frustrated"

    # 4. guilt: sad + fear + low-pitch subdued delivery
    if sad > 0.20 and fear > 0.20 and pitch_mean < 150.0:
        return "guilt"

    # 5. content: happy dominant, low arousal, monotone delivery
    if (
        dominant == "happy"
        and arousal < _COMPOUND_AROUSAL_LOW
        and pitch_std < _COMPOUND_PITCH_VARIANCE_LOW
    ):
        return "content"

    # 6. bored: neutral dominant, low energy, slow rate
    if (
        dominant == "neutral"
        and energy_mean < 0.02
        and speaking_rate < 0.30
    ):
        return "bored"

    return None


class SenseVoiceEmotionDetector:
    """Fast voice emotion using SenseVoice-Small (Alibaba, 2024).

    SenseVoice performs ASR + emotion detection simultaneously.
    ~70ms latency for 10 seconds of audio (15x faster than Whisper-Large).
    Used as fast path during WebSocket streaming.

    Falls back gracefully if funasr not installed.
    """

    _SENSE_LABELS = {
        "<|HAPPY|>": "happy",
        "<|SAD|>": "sad",
        "<|ANGRY|>": "angry",
        "<|FEARFUL|>": "fear",
        "<|DISGUSTED|>": "angry",
        "<|SURPRISED|>": "surprise",
        "<|NEUTRAL|>": "neutral",
        "happy": "happy", "sad": "sad", "angry": "angry",
        "fear": "fear", "fearful": "fear",
        "surprise": "surprise", "surprised": "surprise",
        "neutral": "neutral", "disgusted": "angry",
    }

    def __init__(self):
        self._model = None
        self._available = False
        self._try_load()

    def _try_load(self):
        try:
            from funasr import AutoModel
            self._model = AutoModel(
                model="FunAudioLLM/SenseVoiceSmall",
                trust_remote_code=True,
                disable_update=True,
            )
            self._available = True
            log.info("SenseVoice loaded successfully")
        except Exception as e:
            log.info(
                "SenseVoice not available (funasr not installed or model unavailable): %s", e
            )
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def predict(self, audio: np.ndarray, fs: int = 16000) -> Optional[Dict]:
        """Run SenseVoice emotion + ASR.

        Returns dict with emotion, probabilities, transcription, latency_ms.
        Returns None if SenseVoice unavailable.
        """
        if not self._available or self._model is None:
            return None

        import time
        start = time.time()

        try:
            # SenseVoice expects 16kHz mono float32
            if fs != 16000:
                # Simple integer ratio resampling
                ratio = 16000 / fs
                n_out = int(len(audio) * ratio)
                indices = np.round(np.linspace(0, len(audio) - 1, n_out)).astype(int)
                audio = audio[indices]

            audio_f32 = audio.astype(np.float32)
            if audio_f32.max() > 1.0:
                audio_f32 = audio_f32 / 32768.0  # normalize int16 range

            result = self._model.generate(
                input=audio_f32,
                cache={},
                language="auto",
                use_itn=False,
            )

            elapsed_ms = (time.time() - start) * 1000

            if not result or not result[0]:
                return None

            text = result[0].get("text", "") or ""

            # Extract emotion tag from SenseVoice rich text format
            detected_emotion = self._parse_emotion(text)

            # Build 6-class probability vector
            probs = {e: 0.02 for e in _6CLASS}  # low base
            if detected_emotion in probs:
                probs[detected_emotion] = 0.80
                # Spread remaining 0.10 across other classes
                others = [e for e in _6CLASS if e != detected_emotion]
                for e in others:
                    probs[e] = 0.10 / len(others)

            # Valence/arousal from emotion
            valence_map = {
                "happy": 0.7, "surprise": 0.2, "neutral": 0.0,
                "sad": -0.6, "fear": -0.5, "angry": -0.5,
            }
            arousal_map = {
                "happy": 0.7, "surprise": 0.8, "angry": 0.8,
                "fear": 0.7, "neutral": 0.3, "sad": 0.2,
            }

            return {
                "emotion": detected_emotion,
                "probabilities": probs,
                "valence": valence_map.get(detected_emotion, 0.0),
                "arousal": arousal_map.get(detected_emotion, 0.5),
                "confidence": 0.75,
                "transcription": self._clean_text(text),
                "model_type": "sensevoice",
                "latency_ms": round(elapsed_ms, 1),
            }
        except Exception as e:
            log.debug("SenseVoice prediction failed: %s", e)
            return None

    def _parse_emotion(self, rich_text: str) -> str:
        """Extract emotion tag from SenseVoice output like '<|HAPPY|>text<|/HAPPY|>'."""
        import re
        # Look for emotion tags in angle brackets
        match = re.search(r"<\|([A-Z]+)\|>", rich_text)
        if match:
            tag = f"<|{match.group(1)}|>"
            return self._SENSE_LABELS.get(tag, "neutral")
        # Fallback: check keyword mapping
        lower = rich_text.lower()
        for key, val in self._SENSE_LABELS.items():
            if key.lower() in lower and not key.startswith("<"):
                return val
        return "neutral"

    def _clean_text(self, rich_text: str) -> str:
        """Remove emotion tags to get clean transcription."""
        import re
        return re.sub(r"<\|[^|]+\|>", "", rich_text).strip()


# DistilHuBERT SUPERB-ER label schema → 6-class mapping
# SUPERB uses 4 classes: hap, sad, ang, neu (no fear/surprise)
_DISTILHUBERT_MODEL = "superb/distilhubert-base-superb-er"
_SUPERB_MAP: Dict[str, str] = {
    "hap": "happy",
    "sad": "sad",
    "ang": "angry",
    "fea": "fear",
    "sur": "surprise",
    "neu": "neutral",
    # Defensive: map any unexpected labels to neutral
}
_DISTILHUBERT_SR = 16000  # model expects 16 kHz audio

# MFCC feature extraction constants (mirrors train_audio_emotion.py)
_SR = 22050
_N_MFCC = 40
_N_FEATS = 92


def _extract_prosodic_features(y: np.ndarray, sr: int) -> dict:
    """Extract jitter, shimmer, F0, pause, and GFCC features from audio.

    Delegates to VoiceBiomarkerExtractor (numpy/librosa only — no
    parselmouth or gammatone needed).  Falls back to zero-value dict
    if the extractor is unavailable or audio is too short.
    """
    try:
        from models.voice_biomarkers import get_biomarker_extractor

        extractor = get_biomarker_extractor()
        bio = extractor.extract(y, sr=sr)
        if bio.get("error"):
            return _empty_prosodic()

        # Map to the legacy key names expected by _predict_features()
        return {
            "jitter_local": bio.get("jitter_local", 0.0),
            "shimmer_local": bio.get("shimmer_local", 0.0),
            "f0_mean": bio.get("f0_mean", 0.0),
            "f0_std": bio.get("f0_std", 0.0),
            "voiced_fraction": 0.0,  # not directly in new extractor
            "pause_rate": bio.get("silence_ratio", 0.0),
            "gfcc_mean": bio.get("gfcc_mean", [0.0] * 13),
            "gfcc_std": bio.get("gfcc_std", [0.0] * 13),
            # New fields available for enrichment
            "jitter_rap": bio.get("jitter_rap", 0.0),
            "jitter_ppq5": bio.get("jitter_ppq5", 0.0),
            "shimmer_apq3": bio.get("shimmer_apq3", 0.0),
            "hnr": bio.get("hnr", 0.0),
            "f0_range": bio.get("f0_range", 0.0),
            "pause_count": bio.get("pause_count", 0),
            "mean_pause_duration": bio.get("mean_pause_duration", 0.0),
            "max_pause_duration": bio.get("max_pause_duration", 0.0),
            "speech_rate": bio.get("speech_rate", 0.0),
            "articulation_rate": bio.get("articulation_rate", 0.0),
            "energy_mean": bio.get("energy_mean", 0.0),
            "energy_std": bio.get("energy_std", 0.0),
        }
    except Exception as exc:
        log.warning("Prosodic feature extraction failed: %s", exc)
        return _empty_prosodic()


def _empty_prosodic() -> dict:
    """Return zeroed prosodic feature dict."""
    return {
        "jitter_local": 0.0,
        "shimmer_local": 0.0,
        "f0_mean": 0.0,
        "f0_std": 0.0,
        "voiced_fraction": 0.0,
        "pause_rate": 0.0,
        "gfcc_mean": [0.0] * 13,
        "gfcc_std": [0.0] * 13,
    }


def _valence_arousal(probs: Dict[str, float]) -> tuple[float, float]:
    """Design-doc formulas for valence + arousal from 6-class probabilities."""
    happy    = probs.get("happy",    0.0)
    sad      = probs.get("sad",      0.0)
    angry    = probs.get("angry",    0.0)
    fear     = probs.get("fear",     0.0)
    surprise = probs.get("surprise", 0.0)
    valence = float(np.clip(
        (happy + surprise) * 0.5 - (sad + angry + fear) * 0.5,
        -1.0, 1.0,
    ))
    arousal = float(np.clip(
        (angry + fear + surprise) * 0.6 + happy * 0.3,
        0.0, 1.0,
    ))
    return valence, arousal


def _extract_mfcc_features(y: np.ndarray, sr: int = _SR) -> np.ndarray:
    """Extract 92-dim MFCC feature vector — same pipeline as train_audio_emotion.py.

    92-dim: 40 MFCCs×2 (mean+std)=80, spectral_centroid×2=2,
            bandwidth×2=2, rolloff×2=2, ZCR×2=2, RMS×2=2, flatness×2=2
    """
    import librosa  # type: ignore

    if len(y) < sr // 4:  # < 0.25 sec
        return np.zeros(_N_FEATS, dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=_N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    sc  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sb  = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    sr2 = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    sf  = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    def ms(x: np.ndarray) -> List[float]:
        return [float(x.mean()), float(x.std())]

    feat = np.concatenate([
        mfcc_mean, mfcc_std,
        ms(sc), ms(sb), ms(sr2),
        ms(zcr), ms(rms),
        ms(sf),
    ]).astype(np.float32)

    return np.where(np.isfinite(feat), feat, 0.0)


class VoiceEmotionModel:
    """Wraps emotion2vec+ (funasr) with LightGBM MFCC fallback."""

    def __init__(self) -> None:
        # SenseVoice fast path (emotion2vec-compatible, <100ms)
        self._sensevoice = SenseVoiceEmotionDetector()
        # emotion2vec+ large (funasr, 300M params) — highest accuracy, ACL 2024
        self._e2v_large_model = None
        self._e2v_large_tried = False
        # emotion2vec+ base (funasr) — faster, lower memory
        self._e2v_model = None
        self._e2v_tried = False
        # DistilHuBERT SUPERB-ER (transformers pipeline)
        self._distilhubert_pipe = None
        self._distilhubert_tried = False
        # LightGBM fallback — stores entire pkl dict
        self._lgbm_bundle: Optional[Dict] = None
        self._lgbm_tried = False

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _load_e2v_large(self) -> bool:
        """Load emotion2vec_plus_large (300M params, ACL 2024). Lazy, cached."""
        if self._e2v_large_tried:
            return self._e2v_large_model is not None
        self._e2v_large_tried = True
        try:
            from funasr import AutoModel  # type: ignore
            self._e2v_large_model = AutoModel(
                model="iic/emotion2vec_plus_large",
                disable_update=True,
            )
            log.info("emotion2vec_plus_large loaded successfully (300M params)")
            return True
        except Exception as exc:
            log.warning(
                "emotion2vec_plus_large load failed (%s) — falling back to base", exc
            )
            return False

    def _load_e2v(self) -> bool:
        if self._e2v_tried:
            return self._e2v_model is not None
        self._e2v_tried = True
        try:
            from funasr import AutoModel  # type: ignore
            self._e2v_model = AutoModel(
                model="iic/emotion2vec_plus_base",
                disable_update=True,
            )
            log.info("emotion2vec_plus_base loaded successfully")
            return True
        except Exception as exc:
            log.warning(
                "emotion2vec+ load failed (%s) — using LightGBM fallback", exc
            )
            return False

    def _load_distilhubert(self) -> bool:
        if self._distilhubert_tried:
            return self._distilhubert_pipe is not None
        self._distilhubert_tried = True
        try:
            from transformers import pipeline  # type: ignore
            self._distilhubert_pipe = pipeline(
                "audio-classification",
                model=_DISTILHUBERT_MODEL,
            )
            log.info("DistilHuBERT SUPERB-ER loaded successfully")
            return True
        except ImportError:
            log.warning(
                "transformers not installed — DistilHuBERT unavailable"
            )
            return False
        except Exception as exc:
            log.warning(
                "DistilHuBERT load failed (%s) — falling through to LightGBM", exc
            )
            return False

    def _load_lgbm(self) -> bool:
        if self._lgbm_tried:
            return self._lgbm_bundle is not None
        self._lgbm_tried = True
        if not _LGBM_PATH.exists():
            log.warning("audio_emotion_lgbm.pkl not found at %s", _LGBM_PATH)
            return False
        try:
            with open(_LGBM_PATH, "rb") as f:
                self._lgbm_bundle = pickle.load(f)
            log.info("audio_emotion_lgbm.pkl loaded as voice fallback")
            return True
        except Exception as exc:
            log.warning("LightGBM voice fallback load failed: %s", exc)
            return False

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int = 22050,
        user_threshold: Optional[float] = None,
        **kwargs,
    ) -> Optional[Dict]:
        """Predict emotion from audio array (6-class base + optional compound).

        Uses multi-window analysis: splits audio into overlapping windows,
        runs inference on each, then aggregates by probability averaging.
        After aggregation, compound emotion detection (issue #375) is applied
        to the probability distribution + acoustic cues to resolve one of six
        compound labels (nostalgic, content, bored, anticipation, frustrated,
        guilt) when the pattern is unambiguous.

        Confidence gating:
            Uses ``user_threshold`` when provided (per-user calibrated value
            from ``VoiceBaselineCalibrator.get_optimal_threshold()``).
            Falls back to the global ``_CONFIDENCE_THRESHOLD`` (0.60).

        Args:
            audio:          1D float32 audio array.
            sample_rate:    Sampling rate in Hz.
            user_threshold: Per-user calibrated confidence threshold (0.45–0.80).
                            When None, the global default 0.60 is used.
            real_time:      If True, prefer SenseVoice fast path (<100 ms).
            apply_confidence_gate:
                            If False, skip confidence threshold check.
                            Used internally for window-level predictions.

        Returns None if audio is too short, all models fail, or aggregated
        confidence is below the effective threshold.
        """
        if audio is None or len(audio) < _MIN_SAMPLES:
            return None

        # Resolve effective confidence threshold
        effective_threshold = (
            float(np.clip(user_threshold, 0.45, 0.80))
            if user_threshold is not None
            else _CONFIDENCE_THRESHOLD
        )

        # Fast path: SenseVoice (<100ms) — use when real_time=True
        # SenseVoice is designed for streaming; skip multi-window for it
        if kwargs.get("real_time", False) and self._sensevoice.available:
            result = self._sensevoice.predict(audio, fs=sample_rate)
            if result is not None:
                return result

        # Multi-window analysis for longer audio clips
        apply_gate = kwargs.get("apply_confidence_gate", True)
        result = self._predict_multi_window(audio, sample_rate)
        if result is None:
            return None

        if apply_gate and result["confidence"] < effective_threshold:
            log.debug(
                "Voice emotion suppressed — confidence %.3f below threshold %.2f",
                result["confidence"],
                effective_threshold,
            )
            return None

        # ── Compound emotion detection (issue #375) ───────────────────────────
        # Extract acoustic cues needed for compound rules.  Use lightweight
        # feature extraction — these fields may already be present if the
        # heuristic path was used, otherwise fall back to 0 defaults.
        prosodic = result.get("prosodic_features", {})
        pitch_std = float(prosodic.get("f0_std", 0.0))
        pitch_mean = float(prosodic.get("f0_mean", 150.0))
        energy_mean = float(prosodic.get("energy_mean", 0.0))
        speaking_rate = float(prosodic.get("speech_rate", 0.5))

        compound = detect_compound_emotion(
            probs=result["probabilities"],
            arousal=result.get("arousal", 0.5),
            pitch_std=pitch_std,
            energy_mean=energy_mean,
            speaking_rate=speaking_rate,
            pitch_mean=pitch_mean,
        )
        if compound is not None:
            result["compound_emotion"] = compound
        else:
            result["compound_emotion"] = None

        return result

    def _predict_single(
        self, audio: np.ndarray, sample_rate: int
    ) -> Optional[Dict]:
        """Run the model fallback chain on a single audio segment."""
        # Tier 0: emotion2vec+ large (300M, ACL 2024) — highest accuracy
        if self._load_e2v_large():
            result = self._predict_e2v(audio, sample_rate, model=self._e2v_large_model)
            if result is not None:
                result["model_type"] = "voice_emotion2vec_large"
                return result

        # Tier 1: emotion2vec+ base — faster fallback
        if self._load_e2v():
            result = self._predict_e2v(audio, sample_rate)
            if result is not None:
                return result

        # Try DistilHuBERT (SUPERB-ER) as secondary tier
        if self._load_distilhubert():
            result = self._predict_distilhubert(audio, sample_rate)
            if result is not None:
                return result

        # Fall back to LightGBM
        result = self._predict_lgbm(audio, sample_rate)
        if result is not None:
            return result

        # Last resort: feature-based heuristics (no saved model needed)
        return self._predict_features(audio, sample_rate)

    def _predict_multi_window(
        self, audio: np.ndarray, sample_rate: int
    ) -> Optional[Dict]:
        """Split audio into overlapping windows, aggregate results by averaging.

        Window size: _WINDOW_DURATION_S seconds.
        Hop: _WINDOW_HOP_S seconds (50% overlap by default).
        If audio is shorter than one window, falls back to single-segment predict.

        Returns aggregated result dict or None if all windows fail.
        """
        window_samples = int(_WINDOW_DURATION_S * sample_rate)
        hop_samples = int(_WINDOW_HOP_S * sample_rate)

        # For short clips or when windowing gains nothing, use whole segment
        if len(audio) <= window_samples:
            return self._predict_single(audio, sample_rate)

        # Build windows
        windows: List[np.ndarray] = []
        start = 0
        while start + window_samples <= len(audio):
            windows.append(audio[start: start + window_samples])
            start += hop_samples
        # Include trailing segment if meaningful (>= half a window)
        remainder = audio[start:]
        if len(remainder) >= window_samples // 2 and len(remainder) >= _MIN_SAMPLES:
            windows.append(remainder)

        if not windows:
            return self._predict_single(audio, sample_rate)

        # Run inference per window; collect probabilities
        window_results: List[Dict] = []
        for win in windows:
            r = self._predict_single(win, sample_rate)
            if r is not None:
                window_results.append(r)

        if not window_results:
            return None

        # Aggregate: confidence-weighted average across windows
        weights = np.array([r.get("confidence", 0.5) for r in window_results])
        weights = np.maximum(weights, 0.1)  # floor to avoid zero weight
        weights /= weights.sum()

        all_probs = np.array([
            [r.get("probabilities", {}).get(c, 0.0) for c in _6CLASS]
            for r in window_results
        ])
        weighted_probs = np.average(all_probs, axis=0, weights=weights)

        n = len(window_results)
        agg_probs: Dict[str, float] = dict(zip(_6CLASS, weighted_probs.tolist()))

        # Re-normalize to handle floating-point drift
        total = sum(agg_probs.values()) or 1.0
        agg_probs = {k: round(v / total, 4) for k, v in agg_probs.items()}

        emotion = max(agg_probs, key=agg_probs.__getitem__)
        confidence = agg_probs[emotion]
        valence, arousal = _valence_arousal(agg_probs)

        # Use model_type from last window (all windows use same model chain)
        model_type = window_results[-1].get("model_type", "voice_feature_heuristic")

        return {
            "emotion": emotion,
            "probabilities": agg_probs,
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "confidence": round(confidence, 4),
            "model_type": model_type,
            "windows_analyzed": n,
        }

    def predict_with_biomarkers(
        self, audio: np.ndarray, sample_rate: int = 22050, **kwargs
    ) -> Optional[Dict]:
        """Predict emotion AND extract mental-health biomarkers.

        Returns the standard emotion dict enriched with:
          - biomarkers: raw jitter/shimmer/HNR/pause/GFCC values
          - mental_health: depression, anxiety, and stress screening scores

        Applies the 60% confidence gate: returns None if confidence is too low,
        even if a raw emotion prediction was available.

        This is an opt-in enrichment — callers who only need emotion
        should use ``predict()`` for lower latency.
        """
        result = self.predict(audio, sample_rate, **kwargs)
        if result is None:
            return None

        try:
            from models.voice_biomarkers import get_biomarker_extractor

            extractor = get_biomarker_extractor()
            biomarkers = extractor.extract(audio, sr=sample_rate)
            if not biomarkers.get("error"):
                result["biomarkers"] = biomarkers
                result["mental_health"] = {
                    "depression": extractor.screen_depression(biomarkers),
                    "anxiety": extractor.screen_anxiety(biomarkers),
                    "stress": extractor.screen_stress(biomarkers),
                }
        except Exception as exc:
            log.warning("Biomarker enrichment failed: %s", exc)

        return result

    # ── Internal inference ────────────────────────────────────────────────────

    def _predict_e2v(
        self, audio: np.ndarray, sample_rate: int, model=None
    ) -> Optional[Dict]:
        """Run emotion2vec inference. Pass model= to use a specific instance (e.g. large)."""
        e2v = model if model is not None else self._e2v_model
        try:
            import tempfile
            from pathlib import Path as _Path

            # Create temp file safely (no race condition)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name

            try:
                # Write audio
                try:
                    import soundfile as sf  # type: ignore
                    sf.write(tmp_path, audio, sample_rate)
                except ImportError:
                    import wave
                    import struct

                    n = len(audio)
                    pcm = struct.pack(
                        f"<{n}h",
                        *np.clip(audio * 32767, -32768, 32767).astype(np.int16),
                    )
                    with wave.open(tmp_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sample_rate)
                        wf.writeframes(pcm)

                # Run inference
                res = e2v.generate(
                    input=tmp_path,
                    granularity="utterance",
                    extract_embedding=False,
                )
            finally:
                # Always clean up temp file, even if generate() raises
                _Path(tmp_path).unlink(missing_ok=True)

            item = res[0] if isinstance(res, list) and res else res
            labels = item.get("labels", _E2V_LABELS)
            scores = item.get("scores", [])
            if not scores:
                return None

            # Map 9-class → 6-class
            raw = dict(zip(labels, scores))
            probs_6: Dict[str, float] = {c: 0.0 for c in _6CLASS}
            for label, prob in raw.items():
                target = _E2V_MAP.get(label, "neutral")
                probs_6[target] = probs_6[target] + float(prob)

            total = sum(probs_6.values()) or 1.0
            probs_6 = {k: v / total for k, v in probs_6.items()}

            emotion = max(probs_6, key=probs_6.__getitem__)
            confidence = probs_6[emotion]
            valence, arousal = _valence_arousal(probs_6)
            return {
                "emotion": emotion,
                "probabilities": {k: round(v, 4) for k, v in probs_6.items()},
                "valence": round(valence, 4),
                "arousal": round(arousal, 4),
                "confidence": round(confidence, 4),
                "model_type": "voice_emotion2vec",
            }
        except Exception as exc:
            log.warning("emotion2vec predict failed: %s", exc)
            return None

    def _predict_distilhubert(
        self, audio: np.ndarray, sample_rate: int
    ) -> Optional[Dict]:
        """Run DistilHuBERT SUPERB-ER inference and map to 6-class schema."""
        try:
            # Resample to 16 kHz if necessary (model hard-requires 16 kHz)
            y = audio.astype(np.float32)
            sr = sample_rate
            if sr != _DISTILHUBERT_SR:
                try:
                    import librosa  # type: ignore
                    y = librosa.resample(y, orig_sr=sr, target_sr=_DISTILHUBERT_SR)
                    sr = _DISTILHUBERT_SR
                except Exception as exc:
                    log.warning("DistilHuBERT resample failed: %s", exc)
                    return None

            # transformers pipeline accepts a numpy array + sampling_rate dict
            raw_results = self._distilhubert_pipe(
                {"array": y, "sampling_rate": sr},
                top_k=None,
            )

            if not raw_results:
                return None

            # Map SUPERB labels → 6-class and accumulate scores
            probs_6: Dict[str, float] = {c: 0.0 for c in _6CLASS}
            for item in raw_results:
                label_raw: str = item.get("label", "").lower()
                score: float = float(item.get("score", 0.0))
                target = _SUPERB_MAP.get(label_raw, "neutral")
                probs_6[target] = probs_6[target] + score

            total = sum(probs_6.values()) or 1.0
            probs_6 = {k: v / total for k, v in probs_6.items()}

            emotion = max(probs_6, key=probs_6.__getitem__)
            confidence = probs_6[emotion]
            valence, arousal = _valence_arousal(probs_6)
            return {
                "emotion": emotion,
                "probabilities": {k: round(v, 4) for k, v in probs_6.items()},
                "valence": round(valence, 4),
                "arousal": round(arousal, 4),
                "confidence": round(confidence, 4),
                "model_type": "voice_distilhubert",
            }
        except Exception as exc:
            log.warning("DistilHuBERT predict failed: %s", exc)
            return None

    def _predict_lgbm(
        self, audio: np.ndarray, sample_rate: int
    ) -> Optional[Dict]:
        if not self._load_lgbm():
            return None
        bundle = self._lgbm_bundle
        if bundle is None:
            return None
        try:
            lgbm_model = bundle["model"]
            scaler = bundle["scaler"]
            pca = bundle["pca"]
            class_names: List[str] = bundle["class_names"]  # ['positive','neutral','negative']
            bundle_sr: int = bundle.get("sr", _SR)

            # Resample to training sr if needed
            y = audio
            if sample_rate != bundle_sr:
                try:
                    import librosa  # type: ignore
                    y = librosa.resample(
                        audio, orig_sr=sample_rate, target_sr=bundle_sr
                    )
                except Exception:
                    pass

            # Extract 92-dim MFCC features
            feat = _extract_mfcc_features(y, sr=bundle_sr)

            # Apply scaler + PCA (same pipeline used at training time)
            feat_scaled = scaler.transform(feat.reshape(1, -1))
            feat_pca = pca.transform(feat_scaled)

            # Get 3-class probabilities [positive, neutral, negative]
            proba_3 = lgbm_model.predict_proba(feat_pca)[0]

            # Map class_names to probabilities
            name_to_prob: Dict[str, float] = {
                name: float(proba_3[i])
                for i, name in enumerate(class_names)
            }
            pos = name_to_prob.get("positive", 0.0)
            neu = name_to_prob.get("neutral",  0.0)
            neg = name_to_prob.get("negative", 0.0)

            # 3-class → approximate 6-class
            probs_6: Dict[str, float] = {
                "happy":    round(pos * 0.70, 4),
                "surprise": round(pos * 0.30, 4),
                "neutral":  round(neu,        4),
                "sad":      round(neg * 0.50, 4),
                "angry":    round(neg * 0.30, 4),
                "fear":     round(neg * 0.20, 4),
            }

            total = sum(probs_6.values()) or 1.0
            probs_6 = {k: v / total for k, v in probs_6.items()}

            emotion = max(probs_6, key=probs_6.__getitem__)
            confidence = probs_6[emotion]
            valence, arousal = _valence_arousal(probs_6)
            return {
                "emotion": emotion,
                "probabilities": probs_6,
                "valence": round(valence, 4),
                "arousal": round(arousal, 4),
                "confidence": round(confidence, 4),
                "model_type": "voice_lgbm_fallback",
            }
        except Exception as exc:
            log.warning("LightGBM voice fallback failed: %s", exc)
            return None

    def _predict_features(
        self, audio: np.ndarray, sample_rate: int
    ) -> Optional[Dict]:
        """Feature-based voice emotion heuristics — no saved model needed.

        Uses pitch, energy, speech rate, and spectral features to estimate
        valence and arousal. Accuracy is lower than ML models (~55-60%) but
        works without any model file.
        """
        try:
            import librosa  # type: ignore

            y = audio
            if sample_rate != _SR:
                y = librosa.resample(audio, orig_sr=sample_rate, target_sr=_SR)

            # Energy (RMS)
            rms = librosa.feature.rms(y=y)[0]
            energy_mean = float(rms.mean())
            energy_std = float(rms.std())

            # Pitch (F0) via pyin
            f0, voiced, _ = librosa.pyin(
                y, fmin=60, fmax=500, sr=_SR
            )
            f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            pitch_mean = float(f0_valid.mean()) if len(f0_valid) > 0 else 150.0
            pitch_std = float(f0_valid.std()) if len(f0_valid) > 1 else 0.0
            voiced_frac = float(voiced.mean()) if voiced is not None else 0.5

            # Speech rate proxy (zero-crossing rate)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = float(zcr.mean())

            # Spectral centroid (brightness)
            sc = librosa.feature.spectral_centroid(y=y, sr=_SR)[0]
            brightness = float(sc.mean()) / _SR  # normalize to 0-0.5

            # ── Heuristic rules ──
            # High pitch + high energy + high variability → excited/happy/angry
            # Low pitch + low energy → sad/neutral
            # High pitch + low energy → fear/surprise

            # Arousal: energy + pitch variation + speech rate
            arousal = float(np.clip(
                0.35 * min(1.0, energy_mean * 10) +
                0.25 * min(1.0, pitch_std / 80) +
                0.20 * min(1.0, zcr_mean * 5) +
                0.20 * min(1.0, energy_std * 15),
                0.0, 1.0
            ))

            # Valence: higher pitch mean + brightness → positive
            # lower pitch + less voiced → negative
            pitch_norm = float(np.clip((pitch_mean - 120) / 200, -1, 1))
            valence = float(np.clip(
                0.40 * pitch_norm +
                0.30 * (brightness - 0.15) * 5 +
                0.30 * (voiced_frac - 0.5) * 2,
                -1.0, 1.0
            ))

            # Map to 6-class probabilities
            pos_w = max(0.0, valence)
            neg_w = max(0.0, -valence)
            neu_w = max(0.0, 1.0 - abs(valence) * 2)
            total = pos_w + neg_w + neu_w + 0.001

            probs_6: Dict[str, float] = {
                "happy":    round((pos_w * 0.65 + arousal * 0.1) / total, 4),
                "surprise": round((pos_w * 0.25 + arousal * 0.15) / total, 4),
                "neutral":  round((neu_w * 0.8 + (1 - arousal) * 0.1) / total, 4),
                "sad":      round((neg_w * 0.50 + (1 - arousal) * 0.1) / total, 4),
                "angry":    round((neg_w * 0.30 + arousal * 0.15) / total, 4),
                "fear":     round((neg_w * 0.20 + arousal * 0.1) / total, 4),
            }
            # Re-normalize
            t2 = sum(probs_6.values()) or 1.0
            probs_6 = {k: round(v / t2, 4) for k, v in probs_6.items()}

            emotion = max(probs_6, key=probs_6.__getitem__)
            confidence = probs_6[emotion]

            # ── Prosodic features: jitter/shimmer/GFCC/pause ──────────────
            prosodic = _extract_prosodic_features(y, _SR)

            # High jitter + high pause_rate → stress signal → boost fear/angry
            stress_boost = min(
                0.15,
                prosodic["jitter_local"] * 5 + prosodic["pause_rate"] * 0.1,
            )
            if stress_boost > 0.05:
                probs_6["fear"] = min(1.0, probs_6.get("fear", 0) + stress_boost * 0.5)
                probs_6["angry"] = min(1.0, probs_6.get("angry", 0) + stress_boost * 0.3)
                # Re-normalize
                total_p = sum(probs_6.values())
                probs_6 = {k: v / total_p for k, v in probs_6.items()}
                # Re-round
                probs_6 = {k: round(v, 4) for k, v in probs_6.items()}
                emotion = max(probs_6, key=probs_6.__getitem__)
                confidence = probs_6[emotion]

            return {
                "emotion": emotion,
                "probabilities": probs_6,
                "valence": round(valence, 4),
                "arousal": round(arousal, 4),
                "confidence": round(confidence, 4),
                "model_type": "voice_feature_heuristic",
                "prosodic_features": prosodic,
                "stress_signal": round(stress_boost, 4),
            }
        except Exception as exc:
            log.warning("Feature-based voice emotion failed: %s", exc)
            return None


# ── Module-level singleton ─────────────────────────────────────────────────────

_instance: Optional[VoiceEmotionModel] = None


def get_voice_model() -> VoiceEmotionModel:
    """Return (or create) the module-level VoiceEmotionModel singleton."""
    global _instance
    if _instance is None:
        _instance = VoiceEmotionModel()
    return _instance
