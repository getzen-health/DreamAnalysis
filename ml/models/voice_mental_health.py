"""Multi-condition mental health screening from speech.

Based on: JMIR 2024 study (865 adults) — Whisper encoder embeddings
+ max pooling + binary classifiers.

Conditions screened:
  - Depression (AUC 0.76-0.78)
  - Anxiety (AUC 0.77)
  - Insomnia (AUC 0.73)
  - Fatigue (AUC 0.68)

Requires minimum 30 seconds of speech audio.
No trained model needed — uses feature-based heuristics as fallback.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

MODEL_DIR = pathlib.Path(__file__).parent / "saved"

# Risk level thresholds
RISK_LEVELS = {
    (0.0, 0.30): "minimal",
    (0.30, 0.50): "mild",
    (0.50, 0.70): "moderate",
    (0.70, 1.00): "elevated",
}

CONDITIONS = ["depression", "anxiety", "insomnia", "fatigue"]


def _risk_level(score: float) -> str:
    for (lo, hi), label in RISK_LEVELS.items():
        if lo <= score < hi:
            return label
    return "elevated"


class VoiceMentalHealthScreener:
    """Multi-condition mental health screening from speech.

    Inference chain:
    1. Whisper encoder embeddings + LightGBM (if model files exist)
    2. Prosodic heuristics fallback (always available)
    """

    def __init__(self, whisper_size: str = "small"):
        self._whisper = None
        self._classifiers: Dict = {}
        self._whisper_size = whisper_size
        self._try_load()

    def _try_load(self):
        """Lazy-load Whisper and saved classifiers."""
        # Try Whisper
        try:
            import whisper as openai_whisper
            self._whisper = openai_whisper.load_model(self._whisper_size)
            log.info("Whisper %s loaded for mental health screening", self._whisper_size)
        except Exception as e:
            log.info(
                "Whisper unavailable for mental health screening: %s"
                " — using prosodic heuristics",
                e,
            )

        # Try saved LightGBM classifiers (one per condition)
        for condition in CONDITIONS:
            path = MODEL_DIR / f"mh_{condition}_lgbm.pkl"
            if path.exists():
                try:
                    import pickle
                    with open(path, "rb") as f:
                        self._classifiers[condition] = pickle.load(f)
                    log.info("Loaded mental health classifier for %s", condition)
                except Exception as e:
                    log.warning("Failed to load %s classifier: %s", condition, e)

    def screen(self, audio: np.ndarray, fs: int = 16000) -> Dict:
        """Screen for depression, anxiety, insomnia, fatigue.

        Args:
            audio: 1D float32 audio array (min 30 seconds recommended)
            fs: sampling rate (resampled to 16kHz internally)

        Returns:
            dict with per-condition risk scores, levels, and recommendations
        """
        if len(audio) < fs * 5:  # minimum 5 seconds
            return self._empty_result("Audio too short — minimum 5 seconds required")

        # Resample to 16kHz if needed
        if fs != 16000:
            n_out = int(len(audio) * 16000 / fs)
            indices = np.round(np.linspace(0, len(audio) - 1, n_out)).astype(int)
            audio = audio[indices]

        audio_f32 = audio.astype(np.float32)
        if np.abs(audio_f32).max() > 1.0:
            audio_f32 = audio_f32 / 32768.0

        # Try Whisper embeddings path
        if self._whisper is not None:
            try:
                return self._screen_whisper(audio_f32)
            except Exception as e:
                log.debug(
                    "Whisper screening failed: %s — using prosodic fallback", e
                )

        return self._screen_prosodic(audio_f32)

    def _screen_whisper(self, audio: np.ndarray) -> Dict:
        """Whisper encoder + max pooling + classifiers."""
        import torch
        import whisper as openai_whisper

        mel = openai_whisper.log_mel_spectrogram(audio)
        if mel.shape[-1] > 3000:
            mel = mel[..., :3000]  # cap to 30s

        device = next(self._whisper.parameters()).device
        mel = mel.unsqueeze(0).to(device)

        with torch.no_grad():
            enc = self._whisper.encoder(mel)  # (1, T, D)

        # Max pooling over time
        pooled = enc.max(dim=1).values.squeeze(0).cpu().numpy()  # (D,)

        results = {}
        for condition in CONDITIONS:
            if condition in self._classifiers:
                clf = self._classifiers[condition]
                score = float(clf.predict_proba(pooled.reshape(1, -1))[0, 1])
            else:
                # No classifier — use energy/spectral proxies
                score = self._heuristic_from_embedding(pooled, condition)

            results[condition] = {
                "risk_score": round(score, 3),
                "risk_level": _risk_level(score),
                "confidence": "model" if condition in self._classifiers else "heuristic",
            }

        return self._format_result(results, method="whisper_encoder")

    def _screen_prosodic(self, audio: np.ndarray) -> Dict:
        """Prosodic feature heuristics when Whisper unavailable.

        Uses energy, zero-crossing rate, and spectral centroid as proxies.
        """
        fs = 16000
        frame_len = int(0.025 * fs)
        hop = int(0.010 * fs)

        frames = []
        for i in range(0, len(audio) - frame_len, hop):
            frames.append(audio[i:i + frame_len])

        if not frames:
            return self._empty_result("Audio too short for analysis")

        frames_arr = np.array(frames)

        # Energy features
        rms = np.sqrt((frames_arr ** 2).mean(axis=1))
        mean_energy = float(rms.mean())
        energy_var = float(rms.var())
        low_energy_frac = float((rms < rms.mean() * 0.5).mean())

        # Zero-crossing rate
        zcr = (np.diff(np.sign(frames_arr), axis=1) != 0).sum(axis=1) / frame_len
        mean_zcr = float(zcr.mean())

        # Spectral centroid (proxy for vocal brightness)
        fft_mag = np.abs(np.fft.rfft(frames_arr, axis=1))
        freqs = np.fft.rfftfreq(frame_len, d=1.0 / fs)
        centroid = (fft_mag * freqs).sum(axis=1) / (fft_mag.sum(axis=1) + 1e-10)
        mean_centroid = float(centroid.mean())

        # Heuristics:
        # Depression: low energy, low ZCR, low spectral centroid, high low-energy fraction
        # Anxiety: high ZCR, high energy variance, high centroid
        # Insomnia: reduced energy variance (flat affect), low ZCR
        # Fatigue: very low energy, very low centroid

        dep_score = float(np.clip(
            0.4 * (1 - mean_energy * 5)
            + 0.3 * low_energy_frac
            + 0.3 * (1 - min(mean_centroid / 2000, 1)),
            0.1, 0.85,
        ))
        anx_score = float(np.clip(
            0.5 * min(mean_zcr * 5, 1) + 0.5 * min(energy_var * 10, 1),
            0.1, 0.85,
        ))
        ins_score = float(np.clip(
            0.6 * (1 - min(energy_var * 8, 1)) + 0.4 * (1 - mean_zcr * 3),
            0.1, 0.85,
        ))
        fat_score = float(np.clip(
            0.5 * (1 - mean_energy * 5)
            + 0.5 * (1 - min(mean_centroid / 1500, 1)),
            0.1, 0.85,
        ))

        results = {}
        for condition, score in zip(
            CONDITIONS, [dep_score, anx_score, ins_score, fat_score]
        ):
            results[condition] = {
                "risk_score": round(float(score), 3),
                "risk_level": _risk_level(float(score)),
                "confidence": "heuristic",
            }

        return self._format_result(results, method="prosodic_heuristic")

    def _heuristic_from_embedding(self, embedding: np.ndarray, condition: str) -> float:
        """Rough score from embedding statistics when no classifier trained."""
        # Use L2 norm and variance as rough proxies
        norm = float(np.linalg.norm(embedding))
        var = float(embedding.var())
        # Map to [0.2, 0.6] range — explicitly uncertain without real labels
        return float(np.clip(0.3 + 0.1 * (var / (norm + 1e-3)), 0.2, 0.6))

    def _format_result(self, conditions: Dict, method: str) -> Dict:
        """Add recommendations and overall risk summary."""
        recommendations = []
        high_risk = [
            c for c, d in conditions.items()
            if d["risk_level"] in ("moderate", "elevated")
        ]
        if high_risk:
            recommendations.append(
                f"Voice patterns suggest elevated {', '.join(high_risk)} markers."
                " Consider consulting a mental health professional."
            )
        recommendations.append(
            "This screening is indicative only and not a clinical diagnosis."
        )

        return {
            "conditions": conditions,
            "overall_risk": (
                "elevated" if len(high_risk) >= 2
                else ("moderate" if high_risk else "minimal")
            ),
            "recommendations": recommendations,
            "method": method,
            "disclaimer": (
                "Not a clinical diagnosis. Consult a healthcare professional"
                " for assessment."
            ),
        }

    def _empty_result(self, reason: str) -> Dict:
        return {
            "conditions": {
                c: {"risk_score": 0.0, "risk_level": "unknown", "confidence": "n/a"}
                for c in CONDITIONS
            },
            "overall_risk": "unknown",
            "recommendations": [reason],
            "method": "none",
            "disclaimer": "Not a clinical diagnosis.",
        }


# ── Module-level singleton ─────────────────────────────────────────────────────

_screener_instance: Optional[VoiceMentalHealthScreener] = None


def get_mh_screener() -> VoiceMentalHealthScreener:
    """Return (or create) the module-level VoiceMentalHealthScreener singleton."""
    global _screener_instance
    if _screener_instance is None:
        _screener_instance = VoiceMentalHealthScreener()
    return _screener_instance
