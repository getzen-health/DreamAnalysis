"""Voice emotion detection: emotion2vec+ (primary) with LightGBM fallback.

Output format matches EEG EmotionClassifier (6-class):
    emotion: "happy"|"sad"|"angry"|"fear"|"surprise"|"neutral"
    probabilities: {emotion: 0.0-1.0, ...}
    valence: -1.0 to 1.0
    arousal: 0.0 to 1.0
    confidence: 0.0 to 1.0
    model_type: "voice_emotion2vec" | "voice_lgbm_fallback"
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

# MFCC feature extraction constants (mirrors train_audio_emotion.py)
_SR = 22050
_N_MFCC = 40
_N_FEATS = 92


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
        # emotion2vec+ funasr model
        self._e2v_model = None
        self._e2v_tried = False
        # LightGBM fallback — stores entire pkl dict
        self._lgbm_bundle: Optional[Dict] = None
        self._lgbm_tried = False

    # ── Lazy loaders ──────────────────────────────────────────────────────────

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
        self, audio: np.ndarray, sample_rate: int = 22050
    ) -> Optional[Dict]:
        """Predict 6-class emotion from audio array.

        Returns None if audio is too short or all models fail.
        """
        if audio is None or len(audio) < _MIN_SAMPLES:
            return None

        # Try emotion2vec+ first
        if self._load_e2v():
            result = self._predict_e2v(audio, sample_rate)
            if result is not None:
                return result

        # Fall back to LightGBM
        return self._predict_lgbm(audio, sample_rate)

    # ── Internal inference ────────────────────────────────────────────────────

    def _predict_e2v(
        self, audio: np.ndarray, sample_rate: int
    ) -> Optional[Dict]:
        try:
            import tempfile
            from pathlib import Path as _Path

            try:
                import soundfile as sf  # type: ignore
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_path = f.name
                sf.write(tmp_path, audio, sample_rate)
            except ImportError:
                import wave
                import struct

                tmp_path = tempfile.mktemp(suffix=".wav")
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

            res = self._e2v_model.generate(
                input=tmp_path,
                granularity="utterance",
                extract_embedding=False,
            )
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


# ── Module-level singleton ─────────────────────────────────────────────────────

_instance: Optional[VoiceEmotionModel] = None


def get_voice_model() -> VoiceEmotionModel:
    """Return (or create) the module-level VoiceEmotionModel singleton."""
    global _instance
    if _instance is None:
        _instance = VoiceEmotionModel()
    return _instance
