"""Voice-based cognitive load estimation from prosodic features.

Based on Frontiers in Psychiatry (Sep 2025): pitch variation + intensity
variation + voice activity ratio predict cognitive strain. Language-independent.

Complements EEG cognitive load (theta/beta ratio) with voice biomarkers.
"""

import logging
import numpy as np
from typing import Dict, Optional

log = logging.getLogger(__name__)


class VoiceCognitiveLoadEstimator:
    """Estimate cognitive load from prosodic voice features."""

    def predict(self, audio: np.ndarray, sr: int = 16000) -> Dict:
        """Estimate cognitive load from voice audio.

        Features (Frontiers in Psychiatry 2025):
        - F0 variation: higher SD = more engagement = lower overload
        - Intensity variation: reduced variation = higher mental strain
        - Voice activity ratio: less speech = higher cognitive load

        Args:
            audio: 1D audio waveform
            sr: sample rate (default 16kHz)

        Returns:
            Dict with voice_load_index (0-1), components, level
        """
        try:
            import librosa
        except ImportError:
            log.warning("librosa not available for voice cognitive load")
            return self._empty_result()

        if len(audio) < sr:  # need at least 1 second
            return self._empty_result()

        # F0 (pitch) extraction
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=60, fmax=400, sr=sr
            )
            f0_valid = f0[~np.isnan(f0)]
            f0_variation = float(np.std(f0_valid)) if len(f0_valid) > 5 else 0.0
            # Normalize: typical F0 SD range is 10-80 Hz
            f0_norm = float(np.clip(f0_variation / 80.0, 0, 1))
        except Exception:
            f0_norm = 0.5
            voiced_flag = np.ones(1, dtype=bool)

        # Intensity (RMS energy) variation
        try:
            rms = librosa.feature.rms(y=audio, hop_length=512)[0]
            intensity_variation = float(np.std(rms))
            # Normalize: typical RMS SD range
            intensity_norm = float(np.clip(intensity_variation / 0.05, 0, 1))
        except Exception:
            intensity_norm = 0.5

        # Voice activity ratio
        try:
            voice_activity = float(np.sum(voiced_flag) / max(len(voiced_flag), 1))
        except Exception:
            voice_activity = 0.5

        # Cognitive load estimation:
        # Lower F0 variation + lower intensity variation + lower voice activity = higher load
        voice_load_index = float(np.clip(
            0.35 * (1.0 - f0_norm) +          # less pitch variation = more load
            0.35 * (1.0 - intensity_norm) +    # less intensity variation = more load
            0.30 * (1.0 - voice_activity),     # less speech = more load
            0, 1
        ))

        # Classify level
        if voice_load_index >= 0.6:
            level = "high"
            level_index = 2
        elif voice_load_index >= 0.3:
            level = "moderate"
            level_index = 1
        else:
            level = "low"
            level_index = 0

        return {
            "level": level,
            "level_index": level_index,
            "voice_load_index": round(voice_load_index, 3),
            "components": {
                "f0_variation_norm": round(f0_norm, 3),
                "intensity_variation_norm": round(intensity_norm, 3),
                "voice_activity_ratio": round(voice_activity, 3),
            },
            "model_type": "voice_prosodic",
        }

    def _empty_result(self) -> Dict:
        return {
            "level": "unknown",
            "level_index": -1,
            "voice_load_index": 0.0,
            "components": {},
            "model_type": "voice_prosodic",
        }
