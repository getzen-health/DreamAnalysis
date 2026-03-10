"""Voice Health Analyzer — fatigue, cognitive sharpness, and wellness scoring from voice audio.

Uses only numpy for signal processing (no librosa dependency required).
Provides per-user baseline tracking via exponential moving average.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np


class VoiceHealthAnalyzer:
    """Analyzes physical health indicators from voice audio using acoustic features.

    All computation is pure numpy — no librosa or torch required.
    Per-user baselines are maintained in memory via EMA (alpha=0.1).
    """

    # EMA learning rate for baseline updates
    _EMA_ALPHA: float = 0.1

    # Silence detection energy threshold (fraction of mean energy)
    _SILENCE_THRESHOLD_FACTOR: float = 0.1

    # Minimum plausible F0 in Hz (exclude sub-bass artifacts)
    _F0_MIN: float = 60.0
    # Maximum plausible F0 in Hz (exclude harmonics mistaken for fundamental)
    _F0_MAX: float = 400.0

    def __init__(self) -> None:
        # user_id -> {"mean": {...}, "std": {...}, "n": int}
        self._baselines: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        audio_data: np.ndarray,
        sr: int,
        user_id: str = "default",
    ) -> dict:
        """Analyze voice health from raw audio.

        Parameters
        ----------
        audio_data:
            1-D float array of audio samples (any amplitude scale).
        sr:
            Sample rate in Hz.
        user_id:
            Identifier used for per-user baseline tracking.

        Returns
        -------
        dict with keys: fatigue_index, cognitive_sharpness, voice_wellness_score,
        voice_health_change, wellness_flags, recommendation, raw_features,
        baseline_comparison, n_baseline_samples, model_type.
        """
        audio = np.asarray(audio_data, dtype=np.float64)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        features = self._extract_features(audio, sr)

        fatigue_index = self._compute_fatigue_index(features)
        cognitive_sharpness = self._compute_cognitive_sharpness(features)
        voice_wellness_score = self._compute_wellness_score(fatigue_index, cognitive_sharpness, features)

        baseline_comparison: Optional[dict] = None
        n_baseline = 0
        voice_health_change = 0.0

        bl = self._baselines.get(user_id)
        if bl is not None and bl.get("n", 0) >= 1:
            baseline_comparison = self.compare_to_baseline(user_id, features)
            n_baseline = bl["n"]
            # Positive drift in wellness features → improved; use inverse of overall_drift
            # (overall_drift is a z-score magnitude — negative means worse than baseline)
            voice_health_change = float(
                np.clip(-baseline_comparison["overall_drift"], -1.0, 1.0)
            )

        wellness_flags = self._compute_wellness_flags(features, fatigue_index, cognitive_sharpness)
        recommendation = self._generate_recommendation(wellness_flags, fatigue_index, cognitive_sharpness)

        return {
            "fatigue_index": round(float(fatigue_index), 4),
            "cognitive_sharpness": round(float(cognitive_sharpness), 4),
            "voice_wellness_score": round(float(voice_wellness_score), 4),
            "voice_health_change": round(float(voice_health_change), 4),
            "wellness_flags": wellness_flags,
            "recommendation": recommendation,
            "raw_features": {k: round(float(v), 6) for k, v in features.items()},
            "baseline_comparison": baseline_comparison,
            "n_baseline_samples": n_baseline,
            "model_type": "heuristic",
        }

    def update_baseline(self, user_id: str, features: dict) -> None:
        """Update the per-user baseline using an exponential moving average.

        Parameters
        ----------
        user_id:
            User identifier.
        features:
            Feature dict as returned by _extract_features.
        """
        alpha = self._EMA_ALPHA
        bl = self._baselines.get(user_id)

        if bl is None:
            # First sample: initialise mean = features, std = zeros (will grow)
            self._baselines[user_id] = {
                "mean": {k: float(v) for k, v in features.items()},
                "std": {k: 0.0 for k in features},
                "n": 1,
            }
            return

        old_mean = bl["mean"]
        old_std = bl["std"]
        n = bl["n"]

        new_mean: Dict[str, float] = {}
        new_std: Dict[str, float] = {}

        for k, v in features.items():
            v_f = float(v)
            m = old_mean.get(k, v_f)
            s = old_std.get(k, 0.0)

            # EMA mean
            new_m = (1.0 - alpha) * m + alpha * v_f
            # EMA variance (Welford-style approximation via EMA of squared deviation)
            new_s = (1.0 - alpha) * s + alpha * abs(v_f - new_m)

            new_mean[k] = new_m
            new_std[k] = new_s

        self._baselines[user_id] = {
            "mean": new_mean,
            "std": new_std,
            "n": n + 1,
        }

    def compare_to_baseline(self, user_id: str, features: dict) -> dict:
        """Compute z-score-style deviation from per-user baseline.

        Parameters
        ----------
        user_id:
            User identifier.
        features:
            Feature dict to compare.

        Returns
        -------
        dict with "feature_deviations" (per-feature signed z-score) and
        "overall_drift" (mean signed z-score across all features).
        """
        bl = self._baselines.get(user_id)
        if bl is None or bl["n"] == 0:
            return {"feature_deviations": {}, "overall_drift": 0.0}

        mean_bl = bl["mean"]
        std_bl = bl["std"]

        deviations: Dict[str, float] = {}
        for k, v in features.items():
            m = mean_bl.get(k, float(v))
            s = std_bl.get(k, 0.0)
            if s > 1e-9:
                deviations[k] = round((float(v) - m) / s, 4)
            else:
                deviations[k] = 0.0

        overall_drift = float(np.mean(list(deviations.values()))) if deviations else 0.0

        return {
            "feature_deviations": deviations,
            "overall_drift": round(overall_drift, 4),
        }

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, audio: np.ndarray, sr: int) -> dict:
        """Extract all acoustic features from a 1-D audio array."""
        voice_energy = self._rms_energy(audio)
        pause_ratio = self._pause_ratio(audio, sr, voice_energy)
        speech_rate = self._speech_rate_zcr(audio, sr, pause_ratio)
        f0_mean, f0_variability = self._f0_stats(audio, sr)
        hnr_estimate = self._hnr_estimate(audio, sr)
        articulation_rate = self._articulation_rate(speech_rate, pause_ratio)

        return {
            "speech_rate": speech_rate,
            "pause_ratio": pause_ratio,
            "f0_variability": f0_variability,
            "f0_mean": f0_mean,
            "voice_energy": voice_energy,
            "hnr_estimate": hnr_estimate,
            "articulation_rate": articulation_rate,
        }

    @staticmethod
    def _rms_energy(audio: np.ndarray) -> float:
        """Mean RMS energy across the whole clip."""
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))

    def _pause_ratio(self, audio: np.ndarray, sr: int, rms: float) -> float:
        """Fraction of 20 ms frames classified as silence."""
        frame_len = int(sr * 0.02)
        if frame_len < 1 or len(audio) < frame_len:
            return 0.0
        threshold = max(rms * self._SILENCE_THRESHOLD_FACTOR, 1e-9)
        n_frames = len(audio) // frame_len
        silent = 0
        for i in range(n_frames):
            frame = audio[i * frame_len: (i + 1) * frame_len]
            frame_rms = float(np.sqrt(np.mean(frame ** 2)))
            if frame_rms < threshold:
                silent += 1
        return float(silent / n_frames) if n_frames > 0 else 0.0

    @staticmethod
    def _speech_rate_zcr(audio: np.ndarray, sr: int, pause_ratio: float) -> float:
        """Estimate speech rate (pseudo syllables/sec) via zero-crossing rate.

        ZCR is a reasonable proxy for articulatory rate in the absence of a
        full ASR pipeline.  We normalise against the total duration and scale
        empirically so that typical conversational speech maps to ~3-5 syll/s.
        """
        if len(audio) < 2:
            return 0.0
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0)
        # ZCR is per-sample; multiply by sr to get crossings/sec
        crossings_per_sec = zcr * sr
        # Empirical scale: typical voiced speech ~100-200 crossings/s → ~3-5 syll/s
        rate = crossings_per_sec / 40.0
        # Account for pause time: total_duration = speech_time / (1 - pause_ratio)
        # speech_rate is over the speech segments only; divide by spoken fraction
        spoken_fraction = max(1.0 - pause_ratio, 0.05)
        return float(np.clip(rate * spoken_fraction, 0.0, 20.0))

    def _f0_stats(self, audio: np.ndarray, sr: int) -> tuple[float, float]:
        """Estimate mean and std of F0 using autocorrelation-based pitch tracking.

        Returns (f0_mean_hz, f0_std_hz).  Returns (0.0, 0.0) on unvoiced audio.
        """
        frame_len = int(sr * 0.04)   # 40 ms frames
        hop_len = int(sr * 0.01)     # 10 ms hop
        if len(audio) < frame_len or frame_len < 2:
            return 0.0, 0.0

        min_lag = int(sr / self._F0_MAX)
        max_lag = int(sr / self._F0_MIN)
        if max_lag >= frame_len or min_lag < 1:
            return 0.0, 0.0

        f0_estimates: List[float] = []
        n_frames = max(1, (len(audio) - frame_len) // hop_len + 1)

        for i in range(n_frames):
            start = i * hop_len
            frame = audio[start: start + frame_len].copy()
            # Apply Hanning window
            frame *= np.hanning(len(frame))
            # Normalised autocorrelation
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2:]  # keep positive lags only
            # Normalise
            if corr[0] < 1e-12:
                continue
            corr /= corr[0]
            # Search for peak in [min_lag, max_lag]
            search = corr[min_lag: max_lag + 1]
            if len(search) == 0:
                continue
            peak_idx = int(np.argmax(search))
            peak_val = search[peak_idx]
            # Only accept frames where autocorrelation is strong (voiced)
            if peak_val > 0.3:
                lag = peak_idx + min_lag
                f0 = sr / lag
                if self._F0_MIN <= f0 <= self._F0_MAX:
                    f0_estimates.append(f0)

        if not f0_estimates:
            return 0.0, 0.0

        f0_arr = np.array(f0_estimates)
        return float(np.mean(f0_arr)), float(np.std(f0_arr))

    @staticmethod
    def _hnr_estimate(audio: np.ndarray, sr: int) -> float:
        """Estimate Harmonic-to-Noise Ratio using cepstral method.

        Returns dB value (higher = more periodic = healthier voice).
        Typical voiced speech: 15-25 dB.  Dysphonia / strain: < 10 dB.
        """
        if len(audio) < 2:
            return 0.0

        # Use a 40 ms window from the centre of the signal
        win_len = min(int(sr * 0.04), len(audio))
        start = (len(audio) - win_len) // 2
        frame = audio[start: start + win_len].copy()
        frame -= frame.mean()
        if np.max(np.abs(frame)) < 1e-12:
            return 0.0

        frame *= np.hanning(len(frame))

        # Autocorrelation
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]
        if corr[0] < 1e-12:
            return 0.0
        corr_norm = corr / corr[0]

        # Find the peak in the F0 search range
        f0_min, f0_max = 60.0, 400.0
        min_lag = max(1, int(sr / f0_max))
        max_lag = min(len(corr_norm) - 1, int(sr / f0_min))
        if max_lag <= min_lag:
            return 0.0

        search = corr_norm[min_lag: max_lag + 1]
        peak = float(np.max(search))
        peak = np.clip(peak, 0.0, 0.9999)

        # HNR in dB
        hnr = 10.0 * math.log10(peak / (1.0 - peak + 1e-12))
        return float(np.clip(hnr, -10.0, 40.0))

    @staticmethod
    def _articulation_rate(speech_rate: float, pause_ratio: float) -> float:
        """Speech rate normalised to speaking-only segments (excludes pauses)."""
        spoken_fraction = max(1.0 - pause_ratio, 0.05)
        return float(np.clip(speech_rate / spoken_fraction, 0.0, 25.0))

    # ------------------------------------------------------------------
    # Health indices
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fatigue_index(features: dict) -> float:
        """Fatigue index (0-1): higher = more fatigued.

        Driven by:
        - Low F0 variability (monotone voice)
        - Slow speech rate
        - High pause ratio
        """
        f0_var = features["f0_variability"]
        speech_rate = features["speech_rate"]
        pause_ratio = features["pause_ratio"]

        # Normalised fatigue components (each 0-1, higher = more fatigued)
        # F0 variability: normal ~20-50 Hz; flat <10 Hz is fatigued
        f0_fatigue = float(np.clip(1.0 - f0_var / 40.0, 0.0, 1.0))
        # Speech rate: normal ~3-5 syll/s; slow <2 is fatigued
        rate_fatigue = float(np.clip(1.0 - (speech_rate - 1.0) / 4.0, 0.0, 1.0))
        # Pause ratio: high pausing indicates effort / fatigue
        pause_fatigue = float(np.clip(pause_ratio / 0.5, 0.0, 1.0))

        fatigue = 0.35 * f0_fatigue + 0.35 * rate_fatigue + 0.30 * pause_fatigue
        return float(np.clip(fatigue, 0.0, 1.0))

    @staticmethod
    def _compute_cognitive_sharpness(features: dict) -> float:
        """Cognitive sharpness index (0-1): higher = sharper.

        Driven by:
        - High articulation rate
        - High HNR (clean, periodic voice production)
        """
        art_rate = features["articulation_rate"]
        hnr = features["hnr_estimate"]

        # Articulation rate: sharp = >4 syll/s; sluggish = <2
        rate_score = float(np.clip((art_rate - 1.5) / 4.0, 0.0, 1.0))
        # HNR: clear voice = >15 dB; strained = <8 dB
        hnr_score = float(np.clip((hnr - 5.0) / 20.0, 0.0, 1.0))

        sharpness = 0.55 * rate_score + 0.45 * hnr_score
        return float(np.clip(sharpness, 0.0, 1.0))

    @staticmethod
    def _compute_wellness_score(
        fatigue_index: float,
        cognitive_sharpness: float,
        features: dict,
    ) -> float:
        """Overall voice wellness score (0-1): inverse of weighted fatigue combo."""
        # Add energy contribution: low energy can signal illness / lethargy
        energy = features["voice_energy"]
        # Normalise: typical RMS ~0.01-0.3 depending on recording level; clip at 0.2
        energy_score = float(np.clip(energy / 0.15, 0.0, 1.0))

        wellness = (
            0.40 * (1.0 - fatigue_index)
            + 0.35 * cognitive_sharpness
            + 0.25 * energy_score
        )
        return float(np.clip(wellness, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Flags and recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_wellness_flags(
        features: dict,
        fatigue_index: float,
        cognitive_sharpness: float,
    ) -> List[str]:
        """Return list of wellness flag strings based on thresholds."""
        flags: List[str] = []

        if fatigue_index > 0.65:
            flags.append("possible_fatigue")

        if features["f0_variability"] < 15.0:
            flags.append("reduced_vocal_expression")

        if features["pause_ratio"] > 0.45:
            flags.append("high_pause_ratio")

        if cognitive_sharpness < 0.35:
            flags.append("reduced_cognitive_sharpness")

        if features["hnr_estimate"] < 8.0:
            flags.append("voice_strain")

        return flags

    @staticmethod
    def _generate_recommendation(
        wellness_flags: List[str],
        fatigue_index: float,
        cognitive_sharpness: float,
    ) -> str:
        """Generate a single human-readable recommendation."""
        if not wellness_flags:
            if fatigue_index < 0.25 and cognitive_sharpness > 0.7:
                return "Voice indicators look healthy. Keep up good vocal habits."
            return "No notable concerns detected. Stay hydrated and rest as needed."

        # Priority: fatigue > cognitive sharpness > individual flags
        if "possible_fatigue" in wellness_flags and "high_pause_ratio" in wellness_flags:
            return (
                "Signs of fatigue detected. Consider a short break, hydrate, and "
                "avoid extended speaking sessions until you feel rested."
            )
        if "possible_fatigue" in wellness_flags:
            return (
                "Vocal fatigue indicators present. Try breathing exercises and "
                "reduce speaking load if possible."
            )
        if "voice_strain" in wellness_flags:
            return (
                "Voice strain detected. Rest your voice, avoid whispering "
                "(which strains cords), and drink warm fluids."
            )
        if "reduced_cognitive_sharpness" in wellness_flags:
            return (
                "Articulation rate and voice quality are lower than optimal. "
                "A short rest or a brief mindfulness exercise may help mental clarity."
            )
        if "reduced_vocal_expression" in wellness_flags:
            return (
                "Reduced pitch variation detected, which can accompany low energy "
                "or mood. Gentle vocal warm-ups may help."
            )
        if "high_pause_ratio" in wellness_flags:
            return (
                "Frequent pausing observed. This may indicate word-finding effort "
                "or fatigue. Ensure adequate sleep and hydration."
            )

        return "Monitor the flagged indicators. Rest and hydration are recommended."
