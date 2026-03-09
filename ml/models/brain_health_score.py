"""Brain Health Score Calculator.

Computes a comprehensive brain health score (0-100) from sleep EEG and
waking EEG data across five health domains: sleep, cognition, stress,
mood, and vitality.

Each domain is scored 0-100 based on EEG-derived biomarkers:
- Sleep: band power distribution during sleep (delta/theta dominance,
  sleep efficiency proxy from spectral ratios)
- Cognition: alpha peak frequency + spectral entropy (neural complexity
  and processing speed)
- Stress: alpha/beta ratio + HRV proxy from EEG amplitude variability
  (stress resilience and autonomic balance)
- Mood: frontal alpha asymmetry proxy + valence indicators (emotional
  regulation and affective tone)
- Vitality: beta/theta ratio during waking task + overall signal quality
  (neural efficiency and engagement)

The overall brain health score is a weighted average of the five domains
with letter-grade mapping (A/B/C/D/F).

References:
    Jeste et al. (2015) — Brain health framework for comprehensive
        neuropsychiatric assessment
    Davidson (1992) — Frontal alpha asymmetry and emotional regulation
    Stern (2009) — Cognitive reserve theory and neural efficiency
    Klimesch (1999) — Alpha peak frequency and cognitive performance
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal

# Domain names and their weights in the overall score
DOMAINS = ["sleep", "cognition", "stress", "mood", "vitality"]

DOMAIN_WEIGHTS = {
    "sleep": 0.25,
    "cognition": 0.25,
    "stress": 0.20,
    "mood": 0.15,
    "vitality": 0.15,
}

# Grade thresholds
GRADE_THRESHOLDS = [
    (90, "A"),
    (80, "B"),
    (70, "C"),
    (60, "D"),
    (0, "F"),
]

# EEG frequency band definitions (Hz)
_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "high_beta": (20.0, 30.0),
    "gamma": (30.0, 100.0),
}

# NumPy 2.0 renamed np.trapz -> np.trapezoid
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


class BrainHealthScore:
    """Compute a comprehensive brain health score from EEG data.

    Tracks five health domains (sleep, cognition, stress, mood, vitality),
    each scored 0-100. Combines them into an overall brain health score
    with letter grade and actionable recommendations.

    Usage:
        scorer = BrainHealthScore()
        scorer.add_sleep_data(eeg_array, fs=256, duration_hours=7.5)
        scorer.add_waking_data(eeg_array, fs=256, context="resting")
        result = scorer.compute_score()
        # result["overall_score"] -> 0-100
        # result["grade"] -> "A"/"B"/"C"/"D"/"F"
    """

    def __init__(self) -> None:
        self._sleep_features: List[Dict[str, float]] = []
        self._waking_features: List[Dict[str, float]] = []
        self._history: List[Dict] = []

    def add_sleep_data(
        self,
        eeg: np.ndarray,
        fs: float = 256.0,
        duration_hours: float = 7.0,
    ) -> Dict[str, float]:
        """Add a sleep EEG recording for brain health assessment.

        Args:
            eeg: 1D (n_samples,) or 2D (n_channels, n_samples) EEG array.
            fs: Sampling frequency in Hz.
            duration_hours: Total sleep duration in hours.

        Returns:
            Dict with extracted sleep features.
        """
        eeg = np.asarray(eeg, dtype=float)
        if eeg.ndim == 2:
            eeg = np.mean(eeg, axis=0)

        features = self._extract_sleep_features(eeg, fs, duration_hours)
        self._sleep_features.append(features)
        return features

    def add_waking_data(
        self,
        eeg: np.ndarray,
        fs: float = 256.0,
        context: str = "resting",
    ) -> Dict[str, float]:
        """Add a waking EEG recording for brain health assessment.

        Args:
            eeg: 1D (n_samples,) or 2D (n_channels, n_samples) EEG array.
            fs: Sampling frequency in Hz.
            context: Recording context — "resting", "task", or "meditation".

        Returns:
            Dict with extracted waking features.
        """
        eeg = np.asarray(eeg, dtype=float)

        multichannel = None
        if eeg.ndim == 2:
            multichannel = eeg
            eeg = np.mean(eeg, axis=0)

        features = self._extract_waking_features(
            eeg, fs, context, multichannel
        )
        self._waking_features.append(features)
        return features

    def compute_score(self) -> Dict:
        """Compute the overall brain health score.

        Requires at least one call to add_sleep_data() or add_waking_data()
        before calling this method.

        Returns:
            Dict with:
                - overall_score: float 0-100
                - grade: str "A"/"B"/"C"/"D"/"F"
                - domains: dict of 5 domain scores (0-100 each)
                - top_strength: str (highest domain name)
                - top_weakness: str (lowest domain name)
                - recommendations: list of str
        """
        if not self._sleep_features and not self._waking_features:
            raise ValueError(
                "No data added. Call add_sleep_data() or "
                "add_waking_data() first."
            )

        domains = self._compute_domain_scores()

        # Weighted overall score
        overall = sum(
            DOMAIN_WEIGHTS[d] * domains[d] for d in DOMAINS
        )
        overall = float(np.clip(overall, 0.0, 100.0))

        # Grade
        grade = "F"
        for threshold, letter in GRADE_THRESHOLDS:
            if overall >= threshold:
                grade = letter
                break

        # Strength and weakness
        top_strength = max(domains, key=domains.get)
        top_weakness = min(domains, key=domains.get)

        # Recommendations
        recommendations = self._generate_recommendations(domains)

        result = {
            "overall_score": round(overall, 1),
            "grade": grade,
            "domains": {k: round(v, 1) for k, v in domains.items()},
            "top_strength": top_strength,
            "top_weakness": top_weakness,
            "recommendations": recommendations,
        }

        self._history.append(result)
        if len(self._history) > 365:
            self._history = self._history[-365:]

        return result

    def get_domain_scores(self) -> Dict[str, float]:
        """Get current domain scores without computing the full result.

        Returns:
            Dict of 5 domain scores (0-100 each).
        """
        if not self._sleep_features and not self._waking_features:
            return {d: 0.0 for d in DOMAINS}
        return self._compute_domain_scores()

    def get_trends(self, window: int = 7) -> Dict:
        """Get brain health trends from recent history.

        Args:
            window: Number of recent scores to analyze.

        Returns:
            Dict with trend direction, slope, and mean score.
        """
        if len(self._history) < 2:
            return {
                "trend": "insufficient_data",
                "n_scores": len(self._history),
            }

        recent = self._history[-window:]
        scores = [h["overall_score"] for h in recent]
        slope = float(np.polyfit(range(len(scores)), scores, 1)[0])

        if slope > 1.0:
            trend = "improving"
        elif slope < -1.0:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": round(slope, 2),
            "mean_score": round(float(np.mean(scores)), 1),
            "n_scores": len(recent),
        }

    def get_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """Get scoring history.

        Args:
            last_n: Return only the last N entries. None returns all.

        Returns:
            List of past compute_score() results.
        """
        if last_n is not None:
            return list(self._history[-last_n:])
        return list(self._history)

    def reset(self) -> None:
        """Clear all accumulated data and history."""
        self._sleep_features.clear()
        self._waking_features.clear()
        self._history.clear()

    # ── Domain scoring ───────────────────────────────────────────

    def _compute_domain_scores(self) -> Dict[str, float]:
        """Compute all five domain scores from accumulated features."""
        # Aggregate features by averaging across recordings
        sleep_agg = self._aggregate_features(self._sleep_features)
        waking_agg = self._aggregate_features(self._waking_features)

        return {
            "sleep": self._score_sleep(sleep_agg),
            "cognition": self._score_cognition(waking_agg),
            "stress": self._score_stress(waking_agg),
            "mood": self._score_mood(waking_agg, sleep_agg),
            "vitality": self._score_vitality(waking_agg, sleep_agg),
        }

    def _score_sleep(self, features: Dict[str, float]) -> float:
        """Score sleep quality domain (0-100).

        Based on:
        - Delta power fraction (deep sleep marker)
        - Sleep duration
        - Alpha/delta ratio (lower is better during sleep)
        """
        if not features:
            return 50.0  # neutral if no sleep data

        delta = features.get("delta_power", 0.3)
        duration = features.get("duration_hours", 7.0)
        alpha_delta = features.get("alpha_delta_ratio", 0.5)

        # Delta power: optimal 0.30-0.50 (healthy deep sleep)
        if 0.30 <= delta <= 0.50:
            delta_score = 100.0
        elif delta > 0.50:
            delta_score = max(100.0 - (delta - 0.50) * 200, 50.0)
        else:
            delta_score = float(np.clip(delta / 0.30 * 100, 0, 100))

        # Duration: optimal 7-9 hours
        if 7.0 <= duration <= 9.0:
            duration_score = 100.0
        elif 6.0 <= duration < 7.0 or 9.0 < duration <= 10.0:
            duration_score = 70.0
        elif 5.0 <= duration < 6.0:
            duration_score = 40.0
        elif duration < 5.0:
            duration_score = 20.0
        else:
            duration_score = 60.0

        # Alpha/delta ratio during sleep: lower is better (less arousal)
        # Optimal < 0.3
        alpha_delta_score = float(
            np.clip((1.0 - alpha_delta) * 100, 0, 100)
        )

        return float(np.clip(
            0.40 * delta_score
            + 0.35 * duration_score
            + 0.25 * alpha_delta_score,
            0, 100
        ))

    def _score_cognition(self, features: Dict[str, float]) -> float:
        """Score cognitive vitality domain (0-100).

        Based on:
        - Alpha peak frequency (higher = better processing speed)
        - Spectral entropy (higher = more complex neural processing)
        """
        if not features:
            return 50.0

        apf = features.get("alpha_peak_freq", 10.0)
        se = features.get("spectral_entropy", 0.7)

        # Alpha peak frequency: optimal 10-11.5 Hz (Klimesch 1999)
        apf_score = float(np.clip((apf - 7.0) / 5.0 * 100, 0, 100))

        # Spectral entropy: optimal 0.7-0.9 (complex processing)
        se_score = float(np.clip(se * 120, 0, 100))

        return float(np.clip(
            0.55 * apf_score + 0.45 * se_score,
            0, 100
        ))

    def _score_stress(self, features: Dict[str, float]) -> float:
        """Score stress resilience domain (0-100).

        Higher score = lower stress / better resilience.

        Based on:
        - Alpha/beta ratio (higher = more relaxed)
        - HRV proxy from EEG amplitude variability
        """
        if not features:
            return 50.0

        alpha_beta = features.get("alpha_beta_ratio", 1.0)
        hrv_proxy = features.get("hrv_proxy", 0.5)

        # Alpha/beta ratio: higher = more relaxed
        # Optimal > 1.2 (relaxed state)
        ab_score = float(np.clip(alpha_beta / 1.5 * 100, 0, 100))

        # HRV proxy (coefficient of variation of EEG amplitude)
        # Moderate variability (0.3-0.6) is healthy
        if 0.3 <= hrv_proxy <= 0.6:
            hrv_score = 100.0
        elif hrv_proxy < 0.3:
            hrv_score = float(np.clip(hrv_proxy / 0.3 * 100, 0, 100))
        else:
            hrv_score = float(np.clip((1.0 - hrv_proxy) / 0.4 * 100, 20, 100))

        return float(np.clip(
            0.60 * ab_score + 0.40 * hrv_score,
            0, 100
        ))

    def _score_mood(
        self,
        waking: Dict[str, float],
        sleep: Dict[str, float],
    ) -> float:
        """Score mood / emotional regulation domain (0-100).

        Based on:
        - Frontal alpha asymmetry proxy (positive = approach motivation)
        - Alpha power during rest (higher alpha = calm positive state)
        - Sleep quality influence on mood
        """
        if not waking and not sleep:
            return 50.0

        faa_proxy = 0.0
        alpha_level = 0.3
        if waking:
            faa_proxy = waking.get("faa_proxy", 0.0)
            alpha_level = waking.get("alpha_power", 0.3)

        # FAA proxy: positive = positive mood (Davidson 1992)
        # Range typically -0.5 to +0.5
        faa_score = float(np.clip(
            50.0 + faa_proxy * 100.0,
            0, 100
        ))

        # Alpha level: higher during rest = calm/positive
        alpha_score = float(np.clip(alpha_level / 0.4 * 100, 0, 100))

        # Sleep contribution to mood
        sleep_mood_bonus = 0.0
        if sleep:
            duration = sleep.get("duration_hours", 7.0)
            if duration >= 7.0:
                sleep_mood_bonus = 10.0
            elif duration < 5.0:
                sleep_mood_bonus = -10.0

        mood_raw = 0.45 * faa_score + 0.35 * alpha_score + 0.20 * 50.0
        return float(np.clip(mood_raw + sleep_mood_bonus, 0, 100))

    def _score_vitality(
        self,
        waking: Dict[str, float],
        sleep: Dict[str, float],
    ) -> float:
        """Score vitality / neural efficiency domain (0-100).

        Based on:
        - Beta/theta ratio during task (neural efficiency)
        - Overall signal quality (amplitude in healthy range)
        - Sleep restoration contribution
        """
        if not waking and not sleep:
            return 50.0

        beta_theta = 1.0
        signal_quality = 0.7
        if waking:
            beta_theta = waking.get("beta_theta_ratio", 1.0)
            signal_quality = waking.get("signal_quality", 0.7)

        # Beta/theta ratio: moderate values (1.0-3.0) = engaged
        # Too low = drowsy, too high = anxious
        if 1.0 <= beta_theta <= 3.0:
            bt_score = 100.0
        elif beta_theta < 1.0:
            bt_score = float(np.clip(beta_theta / 1.0 * 100, 20, 100))
        else:
            bt_score = float(np.clip((5.0 - beta_theta) / 2.0 * 100, 20, 100))

        # Signal quality: 0-1 scale
        sq_score = float(np.clip(signal_quality * 100, 0, 100))

        # Sleep restoration bonus
        sleep_bonus = 0.0
        if sleep:
            delta = sleep.get("delta_power", 0.0)
            if delta >= 0.30:
                sleep_bonus = 10.0

        vitality_raw = 0.50 * bt_score + 0.30 * sq_score + 0.20 * 50.0
        return float(np.clip(vitality_raw + sleep_bonus, 0, 100))

    # ── Feature extraction ───────────────────────────────────────

    def _extract_sleep_features(
        self,
        eeg: np.ndarray,
        fs: float,
        duration_hours: float,
    ) -> Dict[str, float]:
        """Extract sleep-relevant features from EEG."""
        powers = self._compute_band_powers(eeg, fs)
        alpha = powers.get("alpha", 0.0)
        delta = powers.get("delta", 0.0)

        return {
            "delta_power": delta,
            "theta_power": powers.get("theta", 0.0),
            "alpha_power": alpha,
            "alpha_delta_ratio": alpha / max(delta, 1e-10),
            "duration_hours": duration_hours,
        }

    def _extract_waking_features(
        self,
        eeg: np.ndarray,
        fs: float,
        context: str,
        multichannel: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Extract waking-state features from EEG."""
        powers = self._compute_band_powers(eeg, fs)
        alpha = powers.get("alpha", 0.0)
        beta = powers.get("beta", 0.0)
        theta = powers.get("theta", 0.0)

        # Alpha peak frequency
        apf = self._compute_alpha_peak_freq(eeg, fs)

        # Spectral entropy
        se = self._compute_spectral_entropy(eeg, fs)

        # HRV proxy: coefficient of variation of EEG amplitude envelope
        hrv_proxy = self._compute_hrv_proxy(eeg, fs)

        # FAA proxy from multichannel data
        faa_proxy = 0.0
        if multichannel is not None and multichannel.shape[0] >= 2:
            faa_proxy = self._compute_faa_proxy(multichannel, fs)

        # Signal quality: fraction of samples within healthy range
        signal_quality = self._compute_signal_quality(eeg)

        return {
            "alpha_power": alpha,
            "beta_power": beta,
            "theta_power": theta,
            "alpha_beta_ratio": alpha / max(beta, 1e-10),
            "beta_theta_ratio": beta / max(theta, 1e-10),
            "alpha_peak_freq": apf,
            "spectral_entropy": se,
            "hrv_proxy": hrv_proxy,
            "faa_proxy": faa_proxy,
            "signal_quality": signal_quality,
            "context": context,
        }

    # ── Low-level computation helpers ────────────────────────────

    def _compute_band_powers(
        self, eeg: np.ndarray, fs: float
    ) -> Dict[str, float]:
        """Compute normalized band powers via Welch PSD."""
        nperseg = min(len(eeg), int(fs * 2))
        if nperseg < 4:
            return {band: 0.0 for band in _BANDS}

        try:
            freqs, psd = scipy_signal.welch(eeg, fs=fs, nperseg=nperseg)
        except Exception:
            return {band: 0.0 for band in _BANDS}

        total = _trapezoid(psd, freqs)
        if total < 1e-10:
            return {band: 0.0 for band in _BANDS}

        powers = {}
        for band_name, (low, high) in _BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                powers[band_name] = float(
                    _trapezoid(psd[mask], freqs[mask]) / total
                )
            else:
                powers[band_name] = 0.0

        return powers

    def _compute_alpha_peak_freq(
        self, eeg: np.ndarray, fs: float
    ) -> float:
        """Find the peak frequency in the alpha band (8-12 Hz)."""
        nperseg = min(len(eeg), int(fs * 2))
        if nperseg < 4:
            return 10.0

        try:
            freqs, psd = scipy_signal.welch(eeg, fs=fs, nperseg=nperseg)
        except Exception:
            return 10.0

        alpha_mask = (freqs >= 8) & (freqs <= 12)
        if not np.any(alpha_mask):
            return 10.0

        alpha_psd = psd[alpha_mask]
        alpha_freqs = freqs[alpha_mask]
        return float(alpha_freqs[np.argmax(alpha_psd)])

    def _compute_spectral_entropy(
        self, eeg: np.ndarray, fs: float
    ) -> float:
        """Compute normalized spectral entropy."""
        nperseg = min(len(eeg), int(fs * 2))
        if nperseg < 4:
            return 0.5

        try:
            _, psd = scipy_signal.welch(eeg, fs=fs, nperseg=nperseg)
        except Exception:
            return 0.5

        psd_norm = psd / (np.sum(psd) + 1e-10)
        psd_positive = psd_norm[psd_norm > 0]
        if len(psd_positive) == 0:
            return 0.0

        se = -np.sum(psd_positive * np.log2(psd_positive + 1e-10))
        max_entropy = np.log2(len(psd_positive))
        if max_entropy < 1e-10:
            return 0.0

        return float(np.clip(se / max_entropy, 0.0, 1.0))

    def _compute_hrv_proxy(self, eeg: np.ndarray, fs: float) -> float:
        """Compute HRV proxy from EEG amplitude envelope variability.

        Uses the coefficient of variation of the analytic signal
        envelope, which correlates with autonomic nervous system
        activity reflected in EEG amplitude modulation.
        """
        if len(eeg) < 10:
            return 0.5

        # Compute amplitude envelope via Hilbert transform
        try:
            analytic = scipy_signal.hilbert(eeg)
            envelope = np.abs(analytic)
        except Exception:
            return 0.5

        mean_env = np.mean(envelope)
        if mean_env < 1e-10:
            return 0.0

        cv = float(np.std(envelope) / mean_env)
        return float(np.clip(cv, 0.0, 1.0))

    def _compute_faa_proxy(
        self, multichannel: np.ndarray, fs: float
    ) -> float:
        """Compute frontal alpha asymmetry proxy.

        FAA = log(right_alpha) - log(left_alpha)
        Positive = approach motivation / positive affect.

        Uses channels 0 (left) and 1 (right) for 2-channel data,
        or channels 1 (AF7) and 2 (AF8) for 4-channel Muse 2 data.
        """
        n_ch = multichannel.shape[0]
        if n_ch >= 4:
            left_ch, right_ch = 1, 2  # AF7, AF8 for Muse 2
        else:
            left_ch, right_ch = 0, 1

        left_alpha = self._get_alpha_power(multichannel[left_ch], fs)
        right_alpha = self._get_alpha_power(multichannel[right_ch], fs)

        # FAA = log(right) - log(left)
        faa = np.log(max(right_alpha, 1e-10)) - np.log(max(left_alpha, 1e-10))
        return float(np.clip(np.tanh(faa), -1.0, 1.0))

    def _get_alpha_power(self, eeg: np.ndarray, fs: float) -> float:
        """Get absolute alpha band power for a single channel."""
        nperseg = min(len(eeg), int(fs * 2))
        if nperseg < 4:
            return 1e-10

        try:
            freqs, psd = scipy_signal.welch(eeg, fs=fs, nperseg=nperseg)
        except Exception:
            return 1e-10

        alpha_mask = (freqs >= 8) & (freqs <= 12)
        if not np.any(alpha_mask):
            return 1e-10

        return float(max(_trapezoid(psd[alpha_mask], freqs[alpha_mask]), 1e-10))

    @staticmethod
    def _compute_signal_quality(eeg: np.ndarray) -> float:
        """Compute signal quality as fraction of samples in healthy range.

        Good EEG amplitude is typically 5-75 uV. Samples outside this
        range indicate artifacts or poor electrode contact.
        """
        if len(eeg) == 0:
            return 0.0

        abs_eeg = np.abs(eeg)
        good_samples = np.sum((abs_eeg >= 0.5) & (abs_eeg <= 100))
        return float(good_samples / len(eeg))

    @staticmethod
    def _aggregate_features(
        feature_list: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """Average features across multiple recordings."""
        if not feature_list:
            return {}

        all_keys = set()
        for f in feature_list:
            all_keys.update(k for k, v in f.items() if isinstance(v, (int, float)))

        aggregated = {}
        for key in all_keys:
            values = [
                f[key] for f in feature_list
                if key in f and isinstance(f[key], (int, float))
            ]
            if values:
                aggregated[key] = float(np.mean(values))

        return aggregated

    # ── Recommendations ──────────────────────────────────────────

    @staticmethod
    def _generate_recommendations(
        domains: Dict[str, float],
    ) -> List[str]:
        """Generate actionable recommendations for low-scoring domains."""
        recommendations = []

        if domains.get("sleep", 100) < 60:
            recommendations.append(
                "sleep: prioritize 7-9 hours of sleep; avoid caffeine "
                "after 2pm and screens 1 hour before bed"
            )
        if domains.get("cognition", 100) < 60:
            recommendations.append(
                "cognition: engage in novel learning, puzzles, or "
                "reading to stimulate neural complexity"
            )
        if domains.get("stress", 100) < 60:
            recommendations.append(
                "stress: practice deep breathing or meditation for "
                "10 minutes daily to improve alpha/beta balance"
            )
        if domains.get("mood", 100) < 60:
            recommendations.append(
                "mood: regular aerobic exercise and social connection "
                "improve frontal alpha asymmetry and emotional regulation"
            )
        if domains.get("vitality", 100) < 60:
            recommendations.append(
                "vitality: ensure adequate sleep and regular physical "
                "activity to maintain neural efficiency"
            )

        if not recommendations:
            recommendations.append(
                "overall: maintain current healthy habits; your brain "
                "health metrics are in good range"
            )

        return recommendations
