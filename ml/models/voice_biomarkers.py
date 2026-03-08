"""Voice biomarkers for mental health screening.

Extracts clinical-grade voice features for depression, anxiety, and stress
detection. All computations use numpy/librosa/scipy only -- no parselmouth
or gammatone dependencies required.

Based on:
- Kintsugi Health (2025): 25s speech -> >70% depression screening
- Brain Sciences (2025): jitter has NEGATIVE relationship with anxiety
  (lower jitter = tighter vocal control = higher anxiety)
- Stress & Health (2025): F0 alone is NOT a reliable stress biomarker
  after publication bias correction
- JMIR Mental Health (2025): pause patterns are the strongest single
  predictor of depression
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


class VoiceBiomarkerExtractor:
    """Extract clinical voice biomarkers for mental health screening.

    All methods require librosa (already a project dependency).
    No parselmouth, gammatone, or spafe needed.
    """

    # ---- public API -----------------------------------------------------------

    def extract(self, audio: np.ndarray, sr: int = 16000) -> Dict:
        """Extract all biomarkers from an audio waveform.

        Minimum audio length: ~2 seconds for any estimate, ~10 seconds for
        reliable jitter/shimmer/pause metrics.

        Args:
            audio: 1-D float32 waveform (mono).
            sr:    Sample rate in Hz.

        Returns:
            Dict of biomarker values, or ``{"error": "..."}`` on failure.
        """
        try:
            import librosa  # type: ignore
        except ImportError:
            return {"error": "librosa_not_available"}

        if audio is None or len(audio) < sr * 2:
            return {"error": "insufficient_audio"}

        result: Dict = {}

        # -- F0 (pitch) features -----------------------------------------------
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=60, fmax=400, sr=sr
        )
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        result["f0_mean"] = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0
        result["f0_std"] = float(np.std(f0_valid)) if len(f0_valid) > 0 else 0.0
        result["f0_range"] = float(np.ptp(f0_valid)) if len(f0_valid) > 0 else 0.0

        # -- Jitter (cycle-to-cycle pitch perturbation) -------------------------
        result.update(self._compute_jitter(f0_valid))

        # -- Shimmer (amplitude perturbation) -----------------------------------
        rms = librosa.feature.rms(y=audio, hop_length=512)[0]
        result.update(self._compute_shimmer(rms))

        # -- HNR (harmonics-to-noise ratio via autocorrelation) -----------------
        result["hnr"] = self._estimate_hnr(audio, sr)

        # -- Pause metrics (strongest depression predictor) ---------------------
        result.update(self._extract_pause_metrics(audio, sr, rms))

        # -- Speech rate --------------------------------------------------------
        voiced_bool = (
            voiced_flag.astype(bool) if voiced_flag is not None
            else np.ones(1, dtype=bool)
        )
        result.update(self._extract_speech_rate(audio, sr, voiced_bool))

        # -- Energy features ----------------------------------------------------
        result["energy_mean"] = float(np.mean(rms))
        result["energy_std"] = float(np.std(rms))

        # -- GFCC (gammatone-approximated via mel filterbank) -------------------
        result.update(self._extract_gfcc(audio, sr))

        return result

    # ---- screening functions -------------------------------------------------

    def screen_depression(self, biomarkers: Dict) -> Dict:
        """Depression risk score from voice biomarkers.

        Key indicators (literature):
          - More / longer pauses (strongest predictor, JMIR 2025)
          - Lower F0 mean and range (monotone speech)
          - Lower energy variability (flat affect)
          - Slower speech rate

        Returns:
            Dict with risk_score (0-1), severity label, indicator list.
        """
        if not biomarkers or biomarkers.get("error"):
            return {"risk_score": 0.0, "severity": "unknown", "indicators": []}

        indicators: List[str] = []
        score = 0.0

        # Pause metrics (weight 35%)
        sr = biomarkers.get("silence_ratio", 0.0)
        if sr > 0.4:
            score += 0.35
            indicators.append("excessive_pauses")
        elif sr > 0.25:
            score += 0.35 * (sr - 0.25) / 0.15

        # F0 range (weight 25%) -- narrow range = monotone = depression
        f0_range = biomarkers.get("f0_range", 100.0)
        if f0_range < 30:
            score += 0.25
            indicators.append("monotone_speech")
        elif f0_range < 60:
            score += 0.25 * (60.0 - f0_range) / 30.0

        # Energy variability (weight 20%)
        energy_std = biomarkers.get("energy_std", 0.02)
        if energy_std < 0.005:
            score += 0.20
            indicators.append("flat_energy")
        elif energy_std < 0.015:
            score += 0.20 * (0.015 - energy_std) / 0.010

        # Speech rate (weight 20%)
        speech_rate = biomarkers.get("speech_rate", 4.0)
        if speech_rate < 2.0:
            score += 0.20
            indicators.append("slow_speech")
        elif speech_rate < 3.0:
            score += 0.20 * (3.0 - speech_rate) / 1.0

        score = float(np.clip(score, 0.0, 1.0))
        if score >= 0.6:
            severity = "moderate_to_severe"
        elif score >= 0.3:
            severity = "mild"
        else:
            severity = "minimal"

        return {
            "risk_score": round(score, 3),
            "severity": severity,
            "indicators": indicators,
        }

    def screen_anxiety(self, biomarkers: Dict) -> Dict:
        """Anxiety indicators from voice.

        Anxiety pattern (Brain Sciences 2025):
          - Faster speech rate
          - *Lower* jitter (tighter vocal-fold control under stress)
          - Higher F0 (elevated pitch)
          - Higher energy variability (erratic loudness)

        Returns:
            Dict with risk_score (0-1), severity label, indicator list.
        """
        if not biomarkers or biomarkers.get("error"):
            return {"risk_score": 0.0, "severity": "unknown", "indicators": []}

        indicators: List[str] = []
        score = 0.0

        # Low jitter (weight 30%) -- tighter vocal control under anxiety
        jitter = biomarkers.get("jitter_local", 0.01)
        if jitter < 0.005:
            score += 0.30
            indicators.append("low_jitter_tight_control")
        elif jitter < 0.01:
            score += 0.30 * (0.01 - jitter) / 0.005

        # High speech rate (weight 25%)
        speech_rate = biomarkers.get("speech_rate", 4.0)
        if speech_rate > 6.0:
            score += 0.25
            indicators.append("rapid_speech")
        elif speech_rate > 5.0:
            score += 0.25 * (speech_rate - 5.0) / 1.0

        # High F0 (weight 25%)
        f0_mean = biomarkers.get("f0_mean", 150.0)
        if f0_mean > 250:
            score += 0.25
            indicators.append("elevated_pitch")
        elif f0_mean > 200:
            score += 0.25 * (f0_mean - 200.0) / 50.0

        # High energy variability (weight 20%)
        energy_std = biomarkers.get("energy_std", 0.02)
        if energy_std > 0.04:
            score += 0.20
            indicators.append("variable_energy")
        elif energy_std > 0.025:
            score += 0.20 * (energy_std - 0.025) / 0.015

        score = float(np.clip(score, 0.0, 1.0))
        if score >= 0.6:
            severity = "high"
        elif score >= 0.3:
            severity = "moderate"
        else:
            severity = "low"

        return {
            "risk_score": round(score, 3),
            "severity": severity,
            "indicators": indicators,
        }

    def screen_stress(self, biomarkers: Dict) -> Dict:
        """Stress indicators from voice.

        Per Stress & Health (2025): F0 alone is NOT reliable after
        publication-bias correction.  We use a multi-feature approach:
          - High-beta shimmer (voice breaks under tension)
          - Reduced HNR (breathy / strained voice)
          - Elevated speech rate
          - Elevated F0 standard deviation (erratic pitch)

        Returns:
            Dict with risk_score (0-1), severity label, indicator list.
        """
        if not biomarkers or biomarkers.get("error"):
            return {"risk_score": 0.0, "severity": "unknown", "indicators": []}

        indicators: List[str] = []
        score = 0.0

        # High shimmer (weight 30%) -- voice quality degrades under stress
        shimmer = biomarkers.get("shimmer_local", 0.0)
        if shimmer > 0.15:
            score += 0.30
            indicators.append("high_shimmer")
        elif shimmer > 0.08:
            score += 0.30 * (shimmer - 0.08) / 0.07

        # Low HNR (weight 25%) -- breathy/strained voice
        hnr = biomarkers.get("hnr", 15.0)
        if hnr < 5.0:
            score += 0.25
            indicators.append("low_hnr_strained_voice")
        elif hnr < 10.0:
            score += 0.25 * (10.0 - hnr) / 5.0

        # High F0 variability (weight 25%) -- erratic pitch, NOT mean F0
        f0_std = biomarkers.get("f0_std", 30.0)
        if f0_std > 60:
            score += 0.25
            indicators.append("erratic_pitch")
        elif f0_std > 40:
            score += 0.25 * (f0_std - 40.0) / 20.0

        # Elevated speech rate (weight 20%)
        speech_rate = biomarkers.get("speech_rate", 4.0)
        if speech_rate > 6.0:
            score += 0.20
            indicators.append("rapid_speech")
        elif speech_rate > 5.0:
            score += 0.20 * (speech_rate - 5.0) / 1.0

        score = float(np.clip(score, 0.0, 1.0))
        if score >= 0.6:
            severity = "high"
        elif score >= 0.3:
            severity = "moderate"
        else:
            severity = "low"

        return {
            "risk_score": round(score, 3),
            "severity": severity,
            "indicators": indicators,
        }

    # ---- internal helpers ----------------------------------------------------

    def _compute_jitter(self, f0_valid: np.ndarray) -> Dict:
        """Jitter (pitch perturbation quotient) from valid F0 values.

        Jitter_local = mean |T_i - T_{i+1}| / mean(T)
        Jitter_RAP   = 3-point running-average perturbation.
        """
        if len(f0_valid) < 3:
            return {"jitter_local": 0.0, "jitter_rap": 0.0, "jitter_ppq5": 0.0}

        periods = 1.0 / (f0_valid + 1e-10)
        mean_period = np.mean(periods)

        # Local jitter
        diffs = np.abs(np.diff(periods))
        jitter_local = float(np.mean(diffs) / (mean_period + 1e-10))

        # RAP -- 3-point running average perturbation
        jitter_rap = 0.0
        if len(periods) > 3:
            smoothed_3 = np.convolve(periods, np.ones(3) / 3, mode="valid")
            rap_diffs = np.abs(periods[1:-1] - smoothed_3)
            jitter_rap = float(np.mean(rap_diffs) / (mean_period + 1e-10))

        # PPQ5 -- 5-point perturbation quotient
        jitter_ppq5 = 0.0
        if len(periods) > 5:
            smoothed_5 = np.convolve(periods, np.ones(5) / 5, mode="valid")
            offset = (len(periods) - len(smoothed_5)) // 2
            ppq5_diffs = np.abs(
                periods[offset : offset + len(smoothed_5)] - smoothed_5
            )
            jitter_ppq5 = float(np.mean(ppq5_diffs) / (mean_period + 1e-10))

        return {
            "jitter_local": round(jitter_local, 6),
            "jitter_rap": round(jitter_rap, 6),
            "jitter_ppq5": round(jitter_ppq5, 6),
        }

    def _compute_shimmer(self, rms: np.ndarray) -> Dict:
        """Shimmer (amplitude perturbation) from RMS energy contour.

        Shimmer_local = mean |A_i - A_{i+1}| / mean(A)
        Shimmer_apq3  = 3-point running-average amplitude perturbation.
        """
        if len(rms) < 3:
            return {"shimmer_local": 0.0, "shimmer_apq3": 0.0}

        mean_amp = np.mean(rms)
        diffs = np.abs(np.diff(rms))
        shimmer_local = float(np.mean(diffs) / (mean_amp + 1e-10))

        # APQ3 -- 3-point amplitude perturbation
        shimmer_apq3 = 0.0
        if len(rms) > 3:
            smoothed = np.convolve(rms, np.ones(3) / 3, mode="valid")
            apq_diffs = np.abs(rms[1:-1] - smoothed)
            shimmer_apq3 = float(np.mean(apq_diffs) / (mean_amp + 1e-10))

        return {
            "shimmer_local": round(shimmer_local, 6),
            "shimmer_apq3": round(shimmer_apq3, 6),
        }

    def _estimate_hnr(self, audio: np.ndarray, sr: int) -> float:
        """Estimate Harmonics-to-Noise Ratio via autocorrelation method.

        For each 25 ms frame: find the autocorrelation peak in the F0
        range (60-400 Hz), then HNR = 10 * log10(peak / (r0 - peak)).

        Returns average HNR in dB across voiced frames.
        """
        frame_length = int(0.025 * sr)  # 25 ms
        hop = int(0.010 * sr)  # 10 ms
        min_lag = int(sr / 400)  # 400 Hz upper bound
        max_lag = int(sr / 60)  # 60 Hz lower bound

        hnr_values: List[float] = []
        for start in range(0, len(audio) - frame_length, hop):
            frame = audio[start : start + frame_length]
            if np.max(np.abs(frame)) < 1e-6:
                continue  # silence
            autocorr = np.correlate(frame, frame, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            r0 = autocorr[0]
            if r0 <= 0 or max_lag >= len(autocorr):
                continue
            search_region = autocorr[min_lag:max_lag]
            if len(search_region) == 0:
                continue
            peak = float(np.max(search_region))
            if peak > 0 and r0 > peak:
                hnr_db = 10.0 * np.log10(peak / (r0 - peak + 1e-10))
                if np.isfinite(hnr_db):
                    hnr_values.append(hnr_db)

        return round(float(np.mean(hnr_values)), 2) if hnr_values else 0.0

    def _extract_pause_metrics(
        self, audio: np.ndarray, sr: int, rms: np.ndarray
    ) -> Dict:
        """Extract silence / pause metrics.

        A pause is defined as consecutive RMS frames below 10% of the
        mean energy, lasting at least 200 ms.
        """
        import librosa  # type: ignore  # noqa: F811

        threshold = np.mean(rms) * 0.1
        is_silent = rms < threshold

        # Detect pause segments (consecutive silent frames >= 200 ms)
        min_pause_frames = max(1, int(0.2 * sr / 512))
        pauses: List[float] = []
        current_pause = 0
        for s in is_silent:
            if s:
                current_pause += 1
            else:
                if current_pause >= min_pause_frames:
                    pauses.append(current_pause * 512 / sr)
                current_pause = 0
        if current_pause >= min_pause_frames:
            pauses.append(current_pause * 512 / sr)

        total_duration = len(audio) / sr
        silence_duration = sum(pauses)

        return {
            "silence_ratio": round(
                float(silence_duration / (total_duration + 1e-10)), 4
            ),
            "pause_count": len(pauses),
            "mean_pause_duration": (
                round(float(np.mean(pauses)), 3) if pauses else 0.0
            ),
            "max_pause_duration": (
                round(float(np.max(pauses)), 3) if pauses else 0.0
            ),
            "total_pause_duration": round(float(silence_duration), 3),
        }

    def _extract_speech_rate(
        self, audio: np.ndarray, sr: int, voiced: np.ndarray
    ) -> Dict:
        """Estimate speech rate from voiced/unvoiced transitions.

        Each voiced-to-unvoiced transition pair approximates one syllable.
        """
        voiced_float = voiced.astype(float)
        transitions = float(np.sum(np.abs(np.diff(voiced_float))))
        syllable_estimate = transitions / 2.0

        total_duration = len(audio) / sr
        if len(voiced) > 0:
            speech_duration = (
                float(np.sum(voiced))
                * (len(audio) / len(voiced))
                / sr
            )
        else:
            speech_duration = total_duration

        return {
            "speech_rate": round(
                float(syllable_estimate / (total_duration + 1e-10)), 3
            ),
            "articulation_rate": round(
                float(syllable_estimate / (speech_duration + 1e-10)), 3
            ),
        }

    def _extract_gfcc(
        self, audio: np.ndarray, sr: int = 16000, num_ceps: int = 13
    ) -> Dict:
        """Gammatone Frequency Cepstral Coefficients.

        Better than MFCC for stress detection (literature).  Approximated
        using a mel-spectrogram with gammatone-like spacing -- avoids the
        ``spafe`` or ``gammatone`` dependency entirely.

        The mel scale is a reasonable approximation of the gammatone
        filterbank for cepstral-coefficient extraction purposes.
        """
        try:
            import librosa  # type: ignore
            from scipy.fft import dct  # type: ignore

            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=64, fmax=8000
            )
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            gfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:num_ceps]

            return {
                "gfcc_mean": [round(float(x), 4) for x in np.mean(gfcc, axis=1)],
                "gfcc_std": [round(float(x), 4) for x in np.std(gfcc, axis=1)],
            }
        except Exception as exc:
            log.warning("GFCC extraction failed: %s", exc)
            return {
                "gfcc_mean": [0.0] * num_ceps,
                "gfcc_std": [0.0] * num_ceps,
            }


# -- module-level singleton ----------------------------------------------------

_instance: Optional[VoiceBiomarkerExtractor] = None


def get_biomarker_extractor() -> VoiceBiomarkerExtractor:
    """Return (or create) the module-level VoiceBiomarkerExtractor singleton."""
    global _instance
    if _instance is None:
        _instance = VoiceBiomarkerExtractor()
    return _instance
