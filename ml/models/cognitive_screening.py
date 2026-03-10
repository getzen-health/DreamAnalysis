"""Voice-based cognitive decline screening and elderly emotional monitoring.

Detects early MCI (Mild Cognitive Impairment) from speech biomarkers and
provides age-adapted emotional monitoring for elderly users.

Key features ranked by discriminative power (2024-2025 literature):
1. Pause patterns — filled pauses, between-utterance pauses >2s
2. Speech rate — syllables per second, articulation rate
3. Prosodic variation — f0 std, energy variability
4. Voice quality — jitter, shimmer, HNR changes

IMPORTANT: This is a SCREENING tool only, NOT diagnostic.
Always recommend professional evaluation for elevated risk scores.

References:
    JMIR Aging (2024) — AUC 0.87 from 29 acoustic features, Framingham Heart Study
    npj Dementia (2025) — AUC 0.945-0.988 MCI detection
    Age and Ageing (2025) — Meta-analysis, 51 studies, 17,340 participants
    Positivity effect meta-analysis — 100 studies, N=7,129
"""

import logging
import time
import threading
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

_DISCLAIMER = (
    "This is a screening tool only and does NOT constitute a medical diagnosis. "
    "Elevated risk scores should be discussed with a healthcare professional. "
    "Many factors (fatigue, medication, environment) can affect voice patterns."
)

# Population norms for cognitive speech markers (approximate, from literature)
_NORMS = {
    "pause_rate_per_min": {"mean": 8.0, "std": 3.0},   # pauses >250ms per minute
    "pause_duration_mean": {"mean": 0.45, "std": 0.15},  # seconds
    "speech_rate_syl_s": {"mean": 3.5, "std": 0.8},     # syllables per second
    "f0_std": {"mean": 30.0, "std": 12.0},               # Hz
    "energy_cv": {"mean": 0.5, "std": 0.15},              # coefficient of variation
    "jitter": {"mean": 0.015, "std": 0.008},              # relative jitter
    "shimmer": {"mean": 0.05, "std": 0.025},              # relative shimmer
    "hnr": {"mean": 15.0, "std": 4.0},                    # dB
}

# Maximum history entries per user
_MAX_HISTORY = 500

_lock = threading.Lock()


def _z_score(value: float, mean: float, std: float) -> float:
    """Z-score with safe division."""
    if std < 1e-10:
        return 0.0
    return (value - mean) / std


class VoiceCognitiveScreener:
    """Detect early MCI from speech biomarkers.

    Uses acoustic features only (no transcript needed) for maximum
    accessibility. Extracts pause patterns, speech rate, prosodic
    variation, and voice quality markers.
    """

    def __init__(self) -> None:
        self._trajectories: Dict[str, List[Dict]] = {}

    def extract_cognitive_features(
        self, audio: np.ndarray, fs: int = 16000
    ) -> Dict:
        """Extract cognitive biomarker vector from audio.

        Args:
            audio: 1D audio waveform (mono).
            fs: Sample rate in Hz.

        Returns:
            Dict of extracted acoustic features.
        """
        try:
            import librosa
            has_librosa = True
        except ImportError:
            has_librosa = False

        audio = np.asarray(audio, dtype=float).ravel()
        duration = len(audio) / fs

        if duration < 1.0:
            return self._empty_features()

        features: Dict = {"duration_seconds": round(duration, 2)}

        # --- Pause detection (energy-based) ---
        hop = int(fs * 0.025)  # 25ms hop
        frame_len = int(fs * 0.05)  # 50ms frame

        if has_librosa:
            rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop)[0]
        else:
            # Manual RMS
            n_frames = max(1, (len(audio) - frame_len) // hop + 1)
            rms = np.array([
                np.sqrt(np.mean(audio[i * hop:i * hop + frame_len] ** 2))
                for i in range(n_frames)
            ])

        rms_threshold = max(np.median(rms) * 0.3, 1e-6)
        is_silence = rms < rms_threshold

        # Find pause durations
        pause_durations = []
        frame_dur = hop / fs
        in_pause = False
        pause_start = 0
        for i, silent in enumerate(is_silence):
            if silent and not in_pause:
                in_pause = True
                pause_start = i
            elif not silent and in_pause:
                in_pause = False
                pdur = (i - pause_start) * frame_dur
                if pdur >= 0.25:  # only count pauses >= 250ms
                    pause_durations.append(pdur)
        if in_pause:
            pdur = (len(is_silence) - pause_start) * frame_dur
            if pdur >= 0.25:
                pause_durations.append(pdur)

        features["pause_count"] = len(pause_durations)
        features["pause_rate_per_min"] = round(
            len(pause_durations) / max(duration / 60, 0.01), 2
        )
        features["pause_duration_mean"] = round(
            float(np.mean(pause_durations)) if pause_durations else 0.0, 3
        )
        features["pause_duration_max"] = round(
            float(np.max(pause_durations)) if pause_durations else 0.0, 3
        )

        # Estimate speech rate (syllable count proxy via energy peaks)
        speech_frames = rms[~is_silence]
        if len(speech_frames) > 0:
            speech_duration = float(np.sum(~is_silence)) * frame_dur
            # Rough syllable estimation: count energy peaks in voiced segments
            if has_librosa:
                onset_frames = librosa.onset.onset_detect(
                    y=audio, sr=fs, hop_length=hop, backtrack=False
                )
                syl_count = max(len(onset_frames), 1)
            else:
                # Simple peak counting
                diff = np.diff(rms)
                peaks = np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0]
                syl_count = max(len(peaks), 1)
            features["speech_rate_syl_s"] = round(
                syl_count / max(speech_duration, 0.1), 2
            )
        else:
            features["speech_rate_syl_s"] = 0.0

        # --- F0 (pitch) statistics ---
        if has_librosa:
            try:
                f0, _, _ = librosa.pyin(audio, fmin=60, fmax=400, sr=fs)
                f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            except Exception:
                f0_valid = np.array([])
        else:
            f0_valid = np.array([])

        if len(f0_valid) > 5:
            features["f0_mean"] = round(float(np.mean(f0_valid)), 2)
            features["f0_std"] = round(float(np.std(f0_valid)), 2)
            features["f0_range"] = round(
                float(np.max(f0_valid) - np.min(f0_valid)), 2
            )
        else:
            features["f0_mean"] = 0.0
            features["f0_std"] = 0.0
            features["f0_range"] = 0.0

        # --- Energy variability ---
        if len(speech_frames) > 5:
            mean_e = float(np.mean(speech_frames))
            features["energy_cv"] = round(
                float(np.std(speech_frames)) / max(mean_e, 1e-10), 3
            )
        else:
            features["energy_cv"] = 0.0

        # --- Voice quality (jitter, shimmer, HNR) ---
        features.update(self._extract_voice_quality(audio, fs, f0_valid))

        return features

    def screen(
        self,
        audio: np.ndarray,
        fs: int = 16000,
        age: Optional[int] = None,
    ) -> Dict:
        """Run cognitive screening from voice audio.

        Args:
            audio: 1D audio waveform.
            fs: Sample rate.
            age: Optional age for age-adjusted scoring.

        Returns:
            Dict with cognitive_risk_score, risk_level, feature_flags,
            features, confidence, and disclaimer.
        """
        features = self.extract_cognitive_features(audio, fs)
        duration = features.get("duration_seconds", 0)

        if duration < 2.0:
            return {
                "cognitive_risk_score": 0.0,
                "risk_level": "insufficient_data",
                "feature_flags": [],
                "features": features,
                "confidence": 0.0,
                "disclaimer": _DISCLAIMER,
            }

        # Confidence based on duration (more speech = more reliable)
        confidence = float(np.clip(duration / 60.0, 0.1, 1.0))

        # Score each domain
        flags = []

        # 1. Pause abnormality (40% weight)
        pause_z = _z_score(
            features["pause_rate_per_min"],
            _NORMS["pause_rate_per_min"]["mean"],
            _NORMS["pause_rate_per_min"]["std"],
        )
        pause_dur_z = _z_score(
            features["pause_duration_mean"],
            _NORMS["pause_duration_mean"]["mean"],
            _NORMS["pause_duration_mean"]["std"],
        )
        pause_score = float(np.clip(
            (max(pause_z, 0) + max(pause_dur_z, 0)) / 4.0, 0, 1
        ))
        if pause_z > 1.5:
            flags.append("elevated_pause_frequency")
        if pause_dur_z > 1.5:
            flags.append("prolonged_pauses")

        # 2. Speech fluency (25% weight)
        rate = features["speech_rate_syl_s"]
        rate_z = _z_score(
            rate,
            _NORMS["speech_rate_syl_s"]["mean"],
            _NORMS["speech_rate_syl_s"]["std"],
        )
        # Low speech rate is concerning (negative z = slower than norm)
        fluency_score = float(np.clip(max(-rate_z, 0) / 3.0, 0, 1))
        if rate_z < -1.5:
            flags.append("reduced_speech_rate")

        # 3. Prosodic score (20% weight)
        f0_std_z = _z_score(
            features["f0_std"],
            _NORMS["f0_std"]["mean"],
            _NORMS["f0_std"]["std"],
        )
        ecv_z = _z_score(
            features["energy_cv"],
            _NORMS["energy_cv"]["mean"],
            _NORMS["energy_cv"]["std"],
        )
        # Low prosodic variation is concerning
        prosodic_score = float(np.clip(
            (max(-f0_std_z, 0) + max(-ecv_z, 0)) / 4.0, 0, 1
        ))
        if f0_std_z < -1.5:
            flags.append("reduced_pitch_variation")
        if ecv_z < -1.5:
            flags.append("reduced_energy_variation")

        # 4. Voice quality (15% weight)
        jitter_z = _z_score(
            features.get("jitter", 0),
            _NORMS["jitter"]["mean"],
            _NORMS["jitter"]["std"],
        )
        shimmer_z = _z_score(
            features.get("shimmer", 0),
            _NORMS["shimmer"]["mean"],
            _NORMS["shimmer"]["std"],
        )
        hnr_z = _z_score(
            features.get("hnr", 0),
            _NORMS["hnr"]["mean"],
            _NORMS["hnr"]["std"],
        )
        # High jitter/shimmer and low HNR are concerning
        vq_score = float(np.clip(
            (max(jitter_z, 0) + max(shimmer_z, 0) + max(-hnr_z, 0)) / 6.0, 0, 1
        ))
        if jitter_z > 1.5:
            flags.append("elevated_jitter")
        if hnr_z < -1.5:
            flags.append("reduced_hnr")

        # Composite score
        risk_score = (
            0.40 * pause_score
            + 0.25 * fluency_score
            + 0.20 * prosodic_score
            + 0.15 * vq_score
        )
        risk_score = float(np.clip(risk_score, 0, 1))

        # Risk level
        if risk_score >= 0.60:
            risk_level = "evaluate"
        elif risk_score >= 0.35:
            risk_level = "monitor"
        else:
            risk_level = "normal"

        return {
            "cognitive_risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "feature_flags": flags,
            "features": features,
            "confidence": round(confidence, 3),
            "disclaimer": _DISCLAIMER,
            "component_scores": {
                "pause_abnormality": round(pause_score, 4),
                "speech_fluency": round(fluency_score, 4),
                "prosodic_variation": round(prosodic_score, 4),
                "voice_quality": round(vq_score, 4),
            },
        }

    def add_longitudinal_point(
        self, user_id: str, screening_result: Dict
    ) -> None:
        """Store a screening result for longitudinal trajectory tracking."""
        with _lock:
            if user_id not in self._trajectories:
                self._trajectories[user_id] = []
            entry = {
                "timestamp": time.time(),
                "risk_score": screening_result.get("cognitive_risk_score", 0),
                "risk_level": screening_result.get("risk_level", "unknown"),
                "flags": screening_result.get("feature_flags", []),
            }
            self._trajectories[user_id].append(entry)
            if len(self._trajectories[user_id]) > _MAX_HISTORY:
                self._trajectories[user_id] = self._trajectories[user_id][
                    -_MAX_HISTORY:
                ]

    def get_trajectory(
        self, user_id: str, last_n: Optional[int] = None
    ) -> Dict:
        """Get cognitive trajectory over time for a user."""
        with _lock:
            history = self._trajectories.get(user_id, [])

        if last_n is not None and last_n > 0:
            history = history[-last_n:]

        if len(history) < 2:
            trend = "insufficient_data"
        else:
            mid = len(history) // 2
            first_half = float(np.mean([h["risk_score"] for h in history[:mid]]))
            second_half = float(np.mean([h["risk_score"] for h in history[mid:]]))
            diff = second_half - first_half
            if diff > 0.05:
                trend = "worsening"
            elif diff < -0.05:
                trend = "improving"
            else:
                trend = "stable"

        return {
            "user_id": user_id,
            "n_assessments": len(history),
            "trajectory": history,
            "trend": trend,
        }

    def _extract_voice_quality(
        self, audio: np.ndarray, fs: int, f0_valid: np.ndarray
    ) -> Dict:
        """Extract jitter, shimmer, HNR from audio."""
        result = {"jitter": 0.0, "shimmer": 0.0, "hnr": 0.0}

        if len(f0_valid) < 5:
            return result

        # Jitter: cycle-to-cycle F0 variation
        periods = 1.0 / f0_valid
        period_diffs = np.abs(np.diff(periods))
        if np.mean(periods) > 0:
            result["jitter"] = round(
                float(np.mean(period_diffs) / np.mean(periods)), 5
            )

        # Shimmer approximation from RMS energy
        hop = int(fs * 0.01)
        frame_len = int(fs * 0.025)
        n_frames = max(1, (len(audio) - frame_len) // hop + 1)
        rms_vals = np.array([
            np.sqrt(np.mean(audio[i * hop:i * hop + frame_len] ** 2))
            for i in range(min(n_frames, len(f0_valid)))
        ])
        if len(rms_vals) > 2 and np.mean(rms_vals) > 0:
            amp_diffs = np.abs(np.diff(rms_vals))
            result["shimmer"] = round(
                float(np.mean(amp_diffs) / np.mean(rms_vals)), 5
            )

        # HNR: harmonics-to-noise ratio (autocorrelation method)
        try:
            period_samples = int(fs / float(np.median(f0_valid)))
            if period_samples > 0 and period_samples < len(audio) // 2:
                frame = audio[:period_samples * 4]
                autocorr = np.correlate(frame, frame, mode="full")
                autocorr = autocorr[len(autocorr) // 2:]
                if len(autocorr) > period_samples:
                    r0 = autocorr[0]
                    r_peak = autocorr[period_samples]
                    if r0 > 0 and r_peak > 0 and r_peak < r0:
                        hnr_linear = r_peak / (r0 - r_peak)
                        result["hnr"] = round(
                            float(10 * np.log10(max(hnr_linear, 1e-10))), 2
                        )
        except Exception:
            pass

        return result

    def _empty_features(self) -> Dict:
        """Return empty feature dict for too-short audio."""
        return {
            "duration_seconds": 0.0,
            "pause_count": 0,
            "pause_rate_per_min": 0.0,
            "pause_duration_mean": 0.0,
            "pause_duration_max": 0.0,
            "speech_rate_syl_s": 0.0,
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "f0_range": 0.0,
            "energy_cv": 0.0,
            "jitter": 0.0,
            "shimmer": 0.0,
            "hnr": 0.0,
        }


class ElderlyEmotionMonitor:
    """Age-adapted emotional monitoring accounting for positivity bias.

    Older adults have a reliable processing bias toward positive information
    (meta-analysis of 100 studies, N=7,129). Sustained negative affect is
    MORE clinically significant in elderly users.
    """

    POSITIVITY_BASELINE = 0.15
    NEGATIVE_AFFECT_WEIGHT = 1.5

    def assess(
        self,
        voice_features: Dict,
        emotion_result: Optional[Dict] = None,
        age: Optional[int] = None,
    ) -> Dict:
        """Age-adapted emotional assessment.

        Args:
            voice_features: Acoustic features from VoiceCognitiveScreener.
            emotion_result: Optional emotion prediction (valence, arousal, etc.).
            age: User's age.

        Returns:
            Dict with adjusted_valence, wellbeing_concern, loneliness_risk,
            positivity_deviation.
        """
        is_elderly = age is not None and age >= 60
        valence = float((emotion_result or {}).get("valence", 0.0))

        # Adjust for positivity bias in elderly
        if is_elderly:
            expected_baseline = self.POSITIVITY_BASELINE
            positivity_deviation = expected_baseline - valence
            # Amplify negative affect significance
            if valence < 0:
                adjusted_valence = valence * self.NEGATIVE_AFFECT_WEIGHT
                adjusted_valence = max(adjusted_valence, -1.0)
            else:
                adjusted_valence = valence
        else:
            expected_baseline = 0.0
            positivity_deviation = -valence
            adjusted_valence = valence

        # Loneliness risk from prosodic markers
        loneliness = self.assess_loneliness_risk(voice_features)

        # Wellbeing concern level
        concern_score = 0.0
        if adjusted_valence < -0.3:
            concern_score += 0.4
        if loneliness["loneliness_risk_score"] > 0.5:
            concern_score += 0.3
        if positivity_deviation > 0.3:
            concern_score += 0.3

        if concern_score >= 0.7:
            concern = "high"
        elif concern_score >= 0.4:
            concern = "moderate"
        elif concern_score >= 0.2:
            concern = "mild"
        else:
            concern = "none"

        return {
            "adjusted_valence": round(adjusted_valence, 4),
            "wellbeing_concern": concern,
            "loneliness_risk": loneliness,
            "positivity_deviation": round(positivity_deviation, 4),
            "is_elderly_adjusted": is_elderly,
        }

    def assess_loneliness_risk(self, voice_features: Dict) -> Dict:
        """Estimate loneliness risk from voice prosodic markers.

        Indicators of social isolation:
        - Low f0 variability (monotone speech)
        - Low energy variability
        - Increased pause frequency
        - Decreased speech rate
        """
        markers = []
        scores = []

        f0_std = voice_features.get("f0_std", 0)
        if f0_std > 0:
            f0_z = _z_score(f0_std, _NORMS["f0_std"]["mean"], _NORMS["f0_std"]["std"])
            if f0_z < -1.0:
                markers.append("monotone_speech")
                scores.append(min(abs(f0_z) / 3.0, 1.0))

        ecv = voice_features.get("energy_cv", 0)
        if ecv > 0:
            ecv_z = _z_score(ecv, _NORMS["energy_cv"]["mean"], _NORMS["energy_cv"]["std"])
            if ecv_z < -1.0:
                markers.append("reduced_energy_variation")
                scores.append(min(abs(ecv_z) / 3.0, 1.0))

        pause_rate = voice_features.get("pause_rate_per_min", 0)
        if pause_rate > 0:
            pause_z = _z_score(
                pause_rate,
                _NORMS["pause_rate_per_min"]["mean"],
                _NORMS["pause_rate_per_min"]["std"],
            )
            if pause_z > 1.0:
                markers.append("increased_pauses")
                scores.append(min(pause_z / 3.0, 1.0))

        rate = voice_features.get("speech_rate_syl_s", 0)
        if rate > 0:
            rate_z = _z_score(
                rate,
                _NORMS["speech_rate_syl_s"]["mean"],
                _NORMS["speech_rate_syl_s"]["std"],
            )
            if rate_z < -1.0:
                markers.append("decreased_speech_rate")
                scores.append(min(abs(rate_z) / 3.0, 1.0))

        risk_score = float(np.mean(scores)) if scores else 0.0

        if risk_score >= 0.6:
            risk_level = "high"
        elif risk_score >= 0.3:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return {
            "loneliness_risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "markers": markers,
        }
