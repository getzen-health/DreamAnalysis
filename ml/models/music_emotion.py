"""Music-Induced Emotion Detector from EEG signals.

Detects emotional responses specifically to music stimuli using temporal
and frontal EEG biomarkers from the Muse 2 headband (4 channels).

Key markers:
  - Temporal alpha asymmetry (TP9 vs TP10) as primary valence marker --
    music processing is predominantly temporal-lobe driven (auditory cortex).
  - Frontal theta for musical chills / frisson detection (ACC activation).
  - Alpha/beta ratio for musical engagement vs passive listening.

4 music emotion quadrants (Russell circumplex adapted for music):
  - energetic_positive: upbeat, joyful, exciting music
  - calm_positive: peaceful, serene, beautiful music
  - energetic_negative: aggressive, tense, anxious music
  - calm_negative: melancholic, sad, somber music

Frisson detection:
  Sudden frontal theta burst + alpha drop = "musical chills" (goosebumps).
  Blood (1999), Craig (2005), Sachs et al. (2016) -- frisson correlates
  with dopaminergic reward circuitry activation.

Scientific basis:
  - Koelsch (2014): Brain correlates of music-evoked emotions
  - Daly et al. (2019): Music-induced emotion from EEG using temporal features
  - Sammler et al. (2007): Temporal alpha asymmetry for music valence
  - Sachs et al. (2016): Frisson and reward from music
"""

import numpy as np
from typing import Dict, List, Optional

from processing.eeg_processor import (
    preprocess,
    extract_band_powers,
    compute_frontal_midline_theta,
)


MUSIC_QUADRANTS = [
    "energetic_positive",
    "calm_positive",
    "energetic_negative",
    "calm_negative",
]

ENGAGEMENT_LEVELS = ["passive", "moderate", "deep"]


class MusicEmotionDetector:
    """Detects music-induced emotional responses from EEG.

    Uses temporal alpha asymmetry (TP9 vs TP10) as primary valence marker
    because music processing is heavily lateralised in temporal cortex.
    Frontal theta bursts detect frisson (musical chills).

    Muse 2 channel order:
        ch0 = TP9  (left temporal)  -- music processing left
        ch1 = AF7  (left frontal)   -- frisson / theta
        ch2 = AF8  (right frontal)  -- frisson / theta
        ch3 = TP10 (right temporal) -- music processing right

    References:
        Koelsch (2014), Daly et al. (2019), Sammler et al. (2007),
        Sachs et al. (2016)
    """

    def __init__(self) -> None:
        self._baseline: Optional[Dict[str, float]] = None
        self._history: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_baseline(self, eeg: np.ndarray, fs: float = 256.0) -> None:
        """Record resting-state EEG baseline for normalisation.

        Call with 30-120 seconds of eyes-closed resting EEG before
        music listening begins.

        Args:
            eeg: 1D (single channel) or 2D (n_channels, n_samples) EEG.
            fs: Sampling frequency in Hz.
        """
        signal = self._pick_primary_signal(eeg)
        processed = preprocess(signal, fs)
        bands = extract_band_powers(processed, fs)

        self._baseline = {
            "alpha": bands.get("alpha", 0.0),
            "beta": bands.get("beta", 0.0),
            "theta": bands.get("theta", 0.0),
            "delta": bands.get("delta", 0.0),
        }

        # Store temporal baseline if multichannel
        if eeg.ndim == 2 and eeg.shape[0] >= 4:
            tp9_proc = preprocess(eeg[0], fs)
            tp10_proc = preprocess(eeg[3], fs)
            tp9_bands = extract_band_powers(tp9_proc, fs)
            tp10_bands = extract_band_powers(tp10_proc, fs)
            self._baseline["tp9_alpha"] = tp9_bands.get("alpha", 0.0)
            self._baseline["tp10_alpha"] = tp10_bands.get("alpha", 0.0)

    def assess(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Assess music-induced emotional response from EEG.

        Args:
            eeg: 1D (single channel) or 2D (n_channels, n_samples) EEG array.
            fs: Sampling frequency in Hz.

        Returns:
            Dict with:
                music_valence: float (-1 to 1), negative to positive
                music_arousal: float (0 to 1), calm to energetic
                music_emotion: str, one of MUSIC_QUADRANTS
                engagement_level: str, one of ENGAGEMENT_LEVELS
                temporal_asymmetry: float, log(TP10_alpha) - log(TP9_alpha)
                frisson_detected: bool
                components: dict of sub-scores
        """
        # Extract per-channel band powers
        signal = self._pick_primary_signal(eeg)
        processed = preprocess(signal, fs)
        bands = extract_band_powers(processed, fs)

        alpha = bands.get("alpha", 0.0)
        beta = bands.get("beta", 0.0)
        theta = bands.get("theta", 0.0)
        delta = bands.get("delta", 0.0)
        high_beta = bands.get("high_beta", 0.0)

        # === Temporal Alpha Asymmetry (primary music valence) ===
        temporal_asym = self._compute_temporal_asymmetry(eeg, fs)

        # === Valence: temporal asymmetry + alpha/beta ratio ===
        # Alpha/beta ratio component: high alpha = relaxed/positive
        abr_valence = float(np.tanh((alpha / max(beta, 1e-10) - 0.7) * 1.5))

        if self._baseline is not None:
            # Baseline-relative alpha/beta
            base_ratio = self._baseline["alpha"] / max(self._baseline["beta"], 1e-10)
            curr_ratio = alpha / max(beta, 1e-10)
            abr_valence = float(np.tanh((curr_ratio - base_ratio) * 2.0))

        # Blend: 60% temporal asymmetry + 40% alpha/beta ratio
        # Temporal asymmetry is primary because music emotion is temporal-lobe driven
        if eeg.ndim == 2 and eeg.shape[0] >= 4:
            asym_valence = float(np.tanh(temporal_asym * 2.0))
            music_valence = float(np.clip(
                0.60 * asym_valence + 0.40 * abr_valence, -1.0, 1.0
            ))
        else:
            # Single-channel fallback: alpha/beta ratio only
            music_valence = float(np.clip(abr_valence, -1.0, 1.0))

        # === Arousal: beta/(alpha+beta) ratio (no gamma -- EMG on Muse 2) ===
        arousal_raw = beta / max(alpha + beta, 1e-10)
        # Add high-beta component for stress/energy discrimination
        hb_component = high_beta / max(beta, 1e-10) if beta > 1e-10 else 0.0

        if self._baseline is not None:
            base_arousal = self._baseline["beta"] / max(
                self._baseline["alpha"] + self._baseline["beta"], 1e-10
            )
            arousal_raw = float(np.clip(
                arousal_raw + 0.3 * (arousal_raw - base_arousal), 0, 1
            ))

        music_arousal = float(np.clip(
            0.70 * arousal_raw + 0.30 * hb_component, 0.0, 1.0
        ))

        # === Music Emotion Quadrant ===
        music_emotion = self._classify_quadrant(music_valence, music_arousal)

        # === Engagement Level ===
        engagement_level = self._classify_engagement(alpha, beta, theta)

        # === Frisson Detection ===
        frisson_result = self.detect_frisson(eeg, fs)
        frisson_detected = frisson_result["frisson_detected"]

        result = {
            "music_valence": round(music_valence, 4),
            "music_arousal": round(music_arousal, 4),
            "music_emotion": music_emotion,
            "engagement_level": engagement_level,
            "temporal_asymmetry": round(temporal_asym, 4),
            "frisson_detected": frisson_detected,
            "components": {
                "abr_valence": round(abr_valence, 4),
                "temporal_valence": round(float(np.tanh(temporal_asym * 2.0)), 4)
                    if eeg.ndim == 2 and eeg.shape[0] >= 4 else 0.0,
                "arousal_raw": round(arousal_raw, 4),
                "hb_component": round(hb_component, 4),
                "frisson_score": round(frisson_result["frisson_score"], 4),
            },
            "band_powers": bands,
        }

        # Store in history
        self._history.append({
            "music_valence": result["music_valence"],
            "music_arousal": result["music_arousal"],
            "music_emotion": result["music_emotion"],
            "engagement_level": result["engagement_level"],
            "temporal_asymmetry": result["temporal_asymmetry"],
            "frisson_detected": result["frisson_detected"],
        })

        return result

    def detect_frisson(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Detect musical frisson (chills / goosebumps) from frontal theta burst.

        Frisson is characterised by a sudden increase in frontal midline theta
        (ACC activation) coupled with alpha suppression. This reflects the
        dopaminergic reward response to music (Sachs et al., 2016).

        Args:
            eeg: 1D or 2D EEG array.
            fs: Sampling frequency.

        Returns:
            Dict with:
                frisson_detected: bool
                frisson_score: float (0-1)
                frontal_theta_ratio: float
        """
        # Use frontal channels (AF7=ch1, AF8=ch2) if available
        if eeg.ndim == 2 and eeg.shape[0] >= 3:
            # Average AF7 and AF8 for frontal signal
            frontal = (eeg[1] + eeg[min(2, eeg.shape[0] - 1)]) / 2.0
        elif eeg.ndim == 2:
            frontal = eeg[0]
        else:
            frontal = eeg

        processed = preprocess(frontal, fs)
        bands = extract_band_powers(processed, fs)

        theta_power = bands.get("theta", 0.0)
        alpha_power = bands.get("alpha", 0.0)
        beta_power = bands.get("beta", 0.0)

        # Frontal theta ratio: theta / (alpha + theta)
        frontal_theta_ratio = theta_power / max(theta_power + alpha_power, 1e-10)

        # Frisson: high theta + low alpha on frontal channels
        # Theta dominance score
        theta_dominance = theta_power / max(alpha_power, 1e-10)

        # Alpha suppression score (lower alpha = more frisson)
        alpha_suppression = float(np.clip(1.0 - alpha_power * 3.0, 0.0, 1.0))

        # Compute FMT for additional evidence
        fmt = compute_frontal_midline_theta(processed, fs)
        fmt_relative = fmt.get("fmt_relative", 0.0)

        # Frisson score: weighted combination
        frisson_score = float(np.clip(
            0.40 * float(np.tanh(theta_dominance - 1.0))
            + 0.30 * alpha_suppression
            + 0.30 * float(np.clip(fmt_relative * 3.0, 0.0, 1.0)),
            0.0, 1.0,
        ))

        # Threshold for frisson detection
        frisson_detected = frisson_score > 0.45

        return {
            "frisson_detected": bool(frisson_detected),
            "frisson_score": round(frisson_score, 4),
            "frontal_theta_ratio": round(frontal_theta_ratio, 4),
            "theta_dominance": round(theta_dominance, 4),
            "alpha_suppression": round(alpha_suppression, 4),
        }

    def get_session_stats(self) -> Dict:
        """Return aggregated statistics for the current listening session.

        Returns:
            Dict with n_assessments, mean_valence, mean_arousal,
            frisson_count, dominant_quadrant, quadrant_distribution.
        """
        if not self._history:
            return {"n_assessments": 0}

        valences = [h["music_valence"] for h in self._history]
        arousals = [h["music_arousal"] for h in self._history]
        frisson_count = sum(1 for h in self._history if h["frisson_detected"])

        # Quadrant distribution
        quadrant_counts: Dict[str, int] = {}
        for h in self._history:
            q = h["music_emotion"]
            quadrant_counts[q] = quadrant_counts.get(q, 0) + 1

        dominant_quadrant = max(quadrant_counts, key=quadrant_counts.get)

        return {
            "n_assessments": len(self._history),
            "mean_valence": round(float(np.mean(valences)), 4),
            "mean_arousal": round(float(np.mean(arousals)), 4),
            "frisson_count": frisson_count,
            "dominant_quadrant": dominant_quadrant,
            "quadrant_distribution": quadrant_counts,
        }

    def get_history(self) -> List[Dict]:
        """Return full assessment history."""
        return list(self._history)

    def reset(self) -> None:
        """Clear all state: history, baseline, frisson tracking."""
        self._baseline = None
        self._history = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_primary_signal(eeg: np.ndarray) -> np.ndarray:
        """Select the primary 1D signal for band power extraction.

        For multichannel: averages frontal channels (AF7, AF8) if available,
        otherwise uses ch0.
        """
        if eeg.ndim == 1:
            return eeg
        if eeg.shape[0] >= 3:
            # Average AF7 (ch1) and AF8 (ch2) for frontal signal
            return (eeg[1] + eeg[2]) / 2.0
        return eeg[0]

    @staticmethod
    def _compute_temporal_asymmetry(eeg: np.ndarray, fs: float) -> float:
        """Compute temporal alpha asymmetry: log(TP10_alpha) - log(TP9_alpha).

        Positive = right temporal dominant = positive music valence.
        Based on Sammler et al. (2007): pleasant music activates left temporal
        more, which means less alpha on left (alpha desynchronisation = activation),
        so TP10 has more alpha -> positive asymmetry -> positive valence.

        Returns 0.0 if fewer than 4 channels.
        """
        if eeg.ndim < 2 or eeg.shape[0] < 4:
            return 0.0

        tp9_proc = preprocess(eeg[0], fs)
        tp10_proc = preprocess(eeg[3], fs)

        tp9_bands = extract_band_powers(tp9_proc, fs)
        tp10_bands = extract_band_powers(tp10_proc, fs)

        tp9_alpha = max(tp9_bands.get("alpha", 1e-12), 1e-12)
        tp10_alpha = max(tp10_bands.get("alpha", 1e-12), 1e-12)

        return float(np.log(tp10_alpha) - np.log(tp9_alpha))

    @staticmethod
    def _classify_quadrant(valence: float, arousal: float) -> str:
        """Map valence and arousal to one of 4 music emotion quadrants."""
        if valence >= 0:
            if arousal >= 0.5:
                return "energetic_positive"
            else:
                return "calm_positive"
        else:
            if arousal >= 0.5:
                return "energetic_negative"
            else:
                return "calm_negative"

    @staticmethod
    def _classify_engagement(alpha: float, beta: float, theta: float) -> str:
        """Classify musical engagement from band power ratios.

        Deep: high beta/alpha ratio (active processing)
        Passive: high alpha/beta ratio (relaxed, not engaged)
        Moderate: in between
        """
        beta_alpha = beta / max(alpha, 1e-10)

        if beta_alpha > 1.5:
            return "deep"
        elif beta_alpha > 0.6:
            return "moderate"
        else:
            return "passive"
