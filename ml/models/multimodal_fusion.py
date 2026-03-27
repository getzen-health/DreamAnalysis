"""Multimodal Emotion Fusion Model.

Combines EEG, audio, and video emotion predictions at inference time
using weighted late fusion:
  fused = EEG×0.50 + Audio×0.25 + Video×0.25

Each modality is optional — weights are renormalized if a modality
is missing or fails. The EEG model is always required.

Classes (3): positive(0), neutral(1), negative(2)
Output emotion names (6-class expansion):
  positive → happy / excited   (based on arousal from EEG)
  neutral  → neutral / focused (based on EEG focus_index)
  negative → sad / angry / fearful (based on EEG features)
"""

from __future__ import annotations

import base64
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_ML_ROOT   = Path(__file__).resolve().parent.parent
_MODEL_DIR = _ML_ROOT / "models" / "saved"

# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_audio_features(audio_samples: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract 92-dim features from a raw audio waveform (numpy array, mono)."""
    try:
        import librosa
        y = audio_samples.astype(np.float32)
        if y.max() > 1.0:
            y = y / 32768.0  # normalize int16 → float32

        n_mfcc = 40
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        def ms(x: np.ndarray):
            return [float(x.mean()), float(x.std())]

        feat = np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            ms(librosa.feature.spectral_centroid(y=y, sr=sr)[0]),
            ms(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]),
            ms(librosa.feature.spectral_rolloff(y=y, sr=sr)[0]),
            ms(librosa.feature.zero_crossing_rate(y)[0]),
            ms(librosa.feature.rms(y=y)[0]),
            ms(librosa.feature.spectral_flatness(y=y)[0]),
        ]).astype(np.float32)
        return np.where(np.isfinite(feat), feat, 0.0)
    except Exception as exc:
        log.debug("Audio feature extraction failed: %s", exc)
        return np.zeros(92, np.float32)


def _extract_video_features(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Extract 72-dim features from a single BGR video frame.

    Returns None if no face detected.
    """
    try:
        import cv2
        FACE_SIZE = 48
        GRID      = 6
        BLOCK     = FACE_SIZE // GRID

        cascade_path = (cv2.data.haarcascades +
                        "haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))

        if len(faces) == 0:
            return None

        x, y, w, h = sorted(faces, key=lambda f: -f[2])[0]
        face = cv2.resize(gray[y:y+h, x:x+w], (FACE_SIZE, FACE_SIZE))
        face_f = face.astype(np.float32) / 255.0

        gx  = cv2.Sobel(face_f, cv2.CV_32F, 1, 0, ksize=3)
        gy  = cv2.Sobel(face_f, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)

        intensity = face_f.reshape(GRID, BLOCK, GRID, BLOCK).mean(axis=(1, 3))
        gradient  = mag.reshape(GRID, BLOCK, GRID, BLOCK).mean(axis=(1, 3))

        return np.concatenate([intensity.flatten(), gradient.flatten()]).astype(np.float32)
    except Exception as exc:
        log.debug("Video feature extraction failed: %s", exc)
        return None


# ── Model loader ─────────────────────────────────────────────────────────────

def _load_pkl(path: Path) -> Optional[dict]:
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        log.warning("Could not load %s: %s", path, exc)
        return None


def _predict_3class(payload: dict, feat: np.ndarray) -> np.ndarray:
    """Run scaler → PCA → LGBM on a feature vector, return (3,) probabilities."""
    sc  = payload["scaler"]
    pca = payload["pca"]
    clf = payload["model"]
    x   = sc.transform(feat.reshape(1, -1))
    x   = pca.transform(x)
    return clf.predict_proba(x)[0].astype(np.float32)   # (3,)


def _predict_combined(payload: dict, feat: np.ndarray) -> np.ndarray:
    """Run the combined multimodal LGBM (no scaler/PCA) on a 251-dim vector.

    NaN values in feat are handled natively by LightGBM.
    Returns (3,) probabilities.
    """
    clf = payload["model"]
    return clf.predict_proba(feat.reshape(1, -1))[0].astype(np.float32)


# ── Main fusion class ─────────────────────────────────────────────────────────

class MultimodalEmotionFusion:
    """Fuses EEG + audio + video emotion predictions.

    Weights (renormalized if a modality is absent):
        EEG   = 0.50
        Audio = 0.25
        Video = 0.25

    Usage:
        fusion = MultimodalEmotionFusion()
        result = fusion.predict(
            eeg_array=np.array(...),   # (4, n_samples)  required
            audio_samples=np.array(...),  # (n_samples,) mono, optional
            video_frame=np.array(...),    # (H, W, 3) BGR, optional
            fs_eeg=256.0,
            sr_audio=22050,
        )
    """

    WEIGHTS = {"eeg": 0.50, "audio": 0.25, "video": 0.25}
    LABELS  = ["positive", "neutral", "negative"]
    EMOTION_MAP = {
        # expanded to 6 discrete emotions based on 3-class + EEG context
        "positive_high_arousal": "happy",
        "positive_low_arousal":  "relaxed",
        "neutral_high_focus":    "focused",
        "neutral_low_focus":     "neutral",
        "negative_high_arousal": "angry",
        "negative_low_arousal":  "sad",
    }

    # Feature dimensions for the combined model
    N_EEG   = 85
    N_AUDIO = 92
    N_VIDEO = 72
    N_TOTAL = 251   # 85+92+72+2

    def __init__(self) -> None:
        self._eeg_payload      = _load_pkl(_MODEL_DIR / "emotion_mega_lgbm.pkl")
        self._audio_payload    = _load_pkl(_MODEL_DIR / "audio_emotion_lgbm.pkl")
        self._video_payload    = _load_pkl(_MODEL_DIR / "video_emotion_lgbm.pkl")
        self._combined_payload = _load_pkl(_MODEL_DIR / "multimodal_mega_lgbm.pkl")

        ok_eeg      = self._eeg_payload      is not None
        ok_audio    = self._audio_payload    is not None
        ok_video    = self._video_payload    is not None
        ok_combined = self._combined_payload is not None
        log.info(
            "MultimodalFusion loaded — EEG:%s  Audio:%s  Video:%s  Combined:%s",
            "✓" if ok_eeg else "✗",
            "✓" if ok_audio else "✗",
            "✓" if ok_video else "✗",
            "✓" if ok_combined else "✗ (train multimodal_mega_lgbm to enable)",
        )

    # ── Raw EEG feature extraction (for combined model) ───────────────────────

    def _extract_eeg_feat(self, eeg: np.ndarray, fs: float) -> Optional[np.ndarray]:
        """Return 85-dim raw EEG feature vector (no scaler/PCA)."""
        try:
            import sys as _sys
            _sys.path.insert(0, str(_ML_ROOT))
            from training.train_mega_lgbm_unified import (
                extract_features as _ext,
                _sliding_windows as _sw,
            )
            if eeg.ndim == 2 and eeg.shape[0] == 4:
                rows = _sw(eeg, fs, win_s=4.0, hop_s=2.0)
                if len(rows) > 0:
                    return rows.mean(axis=0).astype(np.float32)
            return _ext(eeg if eeg.ndim == 2 else eeg.reshape(1, -1), fs)
        except Exception as exc:
            log.debug("EEG feature extraction failed: %s", exc)
            return None

    # ── Combined multimodal predict ───────────────────────────────────────────

    def _predict_combined_model(
        self,
        eeg: np.ndarray,
        fs: float,
        audio_samples: Optional[np.ndarray],
        sr: int,
        frame_bgr: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Use the trained 251-dim combined LGBM when available.

        Returns (3,) probabilities or None if the model is not loaded.
        """
        if self._combined_payload is None:
            return None
        try:
            eeg_feat   = self._extract_eeg_feat(eeg, fs)
            if eeg_feat is None:
                return None

            audio_feat = (
                _extract_audio_features(audio_samples, sr)
                if audio_samples is not None else None
            )
            video_feat = (
                _extract_video_features(frame_bgr)
                if frame_bgr is not None else None
            )

            # Build 251-dim vector with NaN for absent modalities
            row = np.full(self.N_TOTAL, np.nan, dtype=np.float32)
            row[:self.N_EEG] = eeg_feat

            if audio_feat is not None:
                row[self.N_EEG: self.N_EEG + self.N_AUDIO]       = audio_feat
                row[self.N_EEG + self.N_AUDIO + self.N_VIDEO]     = 1.0
            else:
                row[self.N_EEG + self.N_AUDIO + self.N_VIDEO]     = 0.0

            if video_feat is not None:
                row[self.N_EEG + self.N_AUDIO: self.N_EEG + self.N_AUDIO + self.N_VIDEO] = video_feat
                row[self.N_EEG + self.N_AUDIO + self.N_VIDEO + 1] = 1.0
            else:
                row[self.N_EEG + self.N_AUDIO + self.N_VIDEO + 1] = 0.0

            return _predict_combined(self._combined_payload, row)
        except Exception as exc:
            log.debug("Combined model predict failed: %s", exc)
            return None

    # ── Internal EEG predict ──────────────────────────────────────────────────

    def _predict_eeg(self, eeg: np.ndarray, fs: float) -> Tuple[np.ndarray, dict]:
        """Return (3-class probs, eeg_extras) using the mega LGBM model."""
        if self._eeg_payload is None:
            return np.array([1/3, 1/3, 1/3], np.float32), {}

        try:
            import sys as _sys
            _sys.path.insert(0, str(_ML_ROOT))
            from training.train_mega_lgbm_unified import (
                extract_features as _ext,
                _sliding_windows as _sw,
            )
            if eeg.ndim == 2 and eeg.shape[0] == 4:
                rows = _sw(eeg, fs, win_s=4.0, hop_s=2.0)
            else:
                rows = None

            if rows is not None and len(rows) > 0:
                feat = rows.mean(axis=0)
            else:
                feat = _ext(eeg if eeg.ndim == 2 else eeg.reshape(1, -1), fs)

            probs = _predict_3class(self._eeg_payload, feat)

            # Extract EEG context for 6-class expansion
            from processing.eeg_processor import extract_features, preprocess
            sig = eeg[1] if eeg.ndim == 2 else eeg
            eeg_feats = extract_features(preprocess(sig, fs), fs)
            extras = {
                "arousal":     float(eeg_feats.get("beta_ratio", 0.5)),
                "focus_index": float(eeg_feats.get("theta_beta_ratio", 0.5)),
                "stress":      float(eeg_feats.get("high_beta_frac", 0.3)),
            }
            return probs, extras
        except Exception as exc:
            log.warning("EEG predict failed: %s", exc)
            return np.array([1/3, 1/3, 1/3], np.float32), {}

    # ── Internal audio predict ────────────────────────────────────────────────

    def _predict_audio(
        self, audio_samples: np.ndarray, sr: int = 22050
    ) -> Optional[np.ndarray]:
        if self._audio_payload is None or audio_samples is None:
            return None
        try:
            feat  = _extract_audio_features(audio_samples, sr)
            probs = _predict_3class(self._audio_payload, feat)
            return probs
        except Exception as exc:
            log.warning("Audio predict failed: %s", exc)
            return None

    # ── Internal video predict ────────────────────────────────────────────────

    def _predict_video(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self._video_payload is None or frame_bgr is None:
            return None
        try:
            feat = _extract_video_features(frame_bgr)
            if feat is None:
                return None
            probs = _predict_3class(self._video_payload, feat)
            return probs
        except Exception as exc:
            log.warning("Video predict failed: %s", exc)
            return None

    # ── 6-class expansion ─────────────────────────────────────────────────────

    def _expand_to_6class(
        self, probs3: np.ndarray, extras: dict
    ) -> Tuple[str, Dict[str, float]]:
        """Convert 3-class probs + EEG context → 6-class emotion + discrete label."""
        pos, neu, neg = float(probs3[0]), float(probs3[1]), float(probs3[2])
        arousal    = extras.get("arousal", 0.5)
        focus      = extras.get("focus_index", 0.5)
        stress     = extras.get("stress", 0.3)

        # Split positive into happy / relaxed based on arousal
        happy   = pos * min(1.0, arousal * 1.5)
        relaxed = pos * max(0.0, 1.0 - arousal * 1.5)

        # Split neutral into focused / neutral based on focus
        focused  = neu * min(1.0, (1.0 - focus) * 2.0)
        neutral  = neu * max(0.0, 1.0 - (1.0 - focus) * 2.0)

        # Split negative into angry / sad based on arousal + stress
        angry = neg * min(1.0, (arousal + stress) * 0.8)
        sad   = neg * max(0.0, 1.0 - (arousal + stress) * 0.8)

        probs6 = np.array([happy, sad, angry, 0.0, relaxed, focused + neutral],
                          dtype=np.float32)
        # Renorm
        total = probs6.sum()
        if total > 0:
            probs6 /= total

        # Discrete label (highest probability)
        names = ["happy", "sad", "angry", "fearful", "relaxed", "neutral"]
        label = names[int(probs6.argmax())]

        return label, {n: round(float(p), 4) for n, p in zip(names, probs6)}

    # ── Main predict ──────────────────────────────────────────────────────────

    def predict(
        self,
        eeg_array:     np.ndarray,
        fs_eeg:        float = 256.0,
        audio_samples: Optional[np.ndarray] = None,
        sr_audio:      int = 22050,
        video_frame:   Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Fuse EEG + audio + video → multimodal emotion dict.

        Args:
            eeg_array:     (4, n_samples) EEG from Muse 2
            fs_eeg:        EEG sampling rate (default 256 Hz)
            audio_samples: (n_samples,) mono audio waveform, or None
            sr_audio:      Audio sample rate (default 22050 Hz)
            video_frame:   (H, W, 3) BGR frame from webcam, or None

        Returns:
            Dict with keys: emotion, probabilities, valence, arousal,
                            eeg_result, audio_result, video_result,
                            fusion_weights, modalities_used
        """
        # ── Per-modality predictions ──────────────────────────────────────────
        eeg_probs, extras = self._predict_eeg(eeg_array, fs_eeg)
        aud_probs         = self._predict_audio(audio_samples, sr_audio)
        vid_probs         = self._predict_video(video_frame)

        # ── Combined feature-level model (preferred when trained) ─────────────
        combined_probs = self._predict_combined_model(
            eeg_array, fs_eeg, audio_samples, sr_audio, video_frame
        )

        if combined_probs is not None:
            # TODO: Replace fixed 70/30 blend with AttentionFusion.fuse()
            # from processing.multimodal_fusion (issue #543). The new attention-
            # based fusion computes dynamic weights from signal quality,
            # confidence, and cross-modality agreement instead of fixed ratios.
            # Feature-level fusion: blend 70% combined model + 30% EEG-only
            # (gives the combined model primacy while keeping EEG stability)
            fused   = 0.70 * combined_probs + 0.30 * eeg_probs
            weights = {"combined": 0.70, "eeg_fallback": 0.30}
        else:
            # Fallback: weighted late fusion of per-modality models
            weights = {"eeg": self.WEIGHTS["eeg"]}
            if aud_probs is not None:
                weights["audio"] = self.WEIGHTS["audio"]
            if vid_probs is not None:
                weights["video"] = self.WEIGHTS["video"]

            # Renormalize weights to sum to 1
            total_w = sum(weights.values())
            weights = {k: v / total_w for k, v in weights.items()}

            fused = eeg_probs * weights["eeg"]
            if aud_probs is not None:
                fused += aud_probs * weights.get("audio", 0.0)
            if vid_probs is not None:
                fused += vid_probs * weights.get("video", 0.0)

        # ── 3-class result ────────────────────────────────────────────────────
        probs3 = {lbl: round(float(fused[i]), 4) for i, lbl in enumerate(self.LABELS)}
        valence = float(fused[0] - fused[2])   # pos - neg  → −1..+1
        arousal = extras.get("arousal", 0.5)

        # ── Expand to 6-class discrete emotion ───────────────────────────────
        emotion_label, probs6 = self._expand_to_6class(fused, extras)

        # ── Per-modality breakdowns ───────────────────────────────────────────
        def _to_3dict(p):
            if p is None:
                return None
            return {lbl: round(float(p[i]), 4) for i, lbl in enumerate(self.LABELS)}

        return {
            "emotion":       emotion_label,
            "probabilities": probs6,
            "valence":       round(valence, 4),
            "arousal":       round(float(arousal), 4),
            "stress_index":  round(float(extras.get("stress", 0.3)), 4),
            # 3-class fused
            "fused_3class":  probs3,
            # per-modality breakdowns
            "eeg_result":   _to_3dict(eeg_probs),
            "audio_result": _to_3dict(aud_probs),
            "video_result": _to_3dict(vid_probs),
            # meta
            "fusion_weights":   {k: round(v, 3) for k, v in weights.items()},
            "modalities_used":  list(weights.keys()),
            "model_type": (
                "multimodal_combined_lgbm" if combined_probs is not None
                else "multimodal_late_fusion"
            ),
        }
