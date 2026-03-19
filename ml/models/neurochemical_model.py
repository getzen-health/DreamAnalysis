"""Neurochemical state inference from EEG spectral features.

Estimates relative levels of dopamine, serotonin, cortisol, norepinephrine,
GABA, and endorphins using validated EEG spectral proxies.  Each estimate is
0-1 normalized with a confidence score.

Scientific basis (proxy mapping):
  - Dopamine: beta power in frontal regions (reward/motivation), alpha
    suppression during anticipation (Knyazev, 2007; Schutter & Knyazev, 2012)
  - Serotonin: frontal alpha power (calm contentment), low irritability
    markers, emotional stability (Suhhova et al., 2018)
  - Cortisol: high beta/alpha ratio (stress), right frontal activation,
    HRV depression (Al-Shargie et al., 2016)
  - Norepinephrine: arousal level, overall beta power, alertness markers
    (Barry et al., 2007)
  - GABA: alpha power magnitude, relaxation depth, sleep onset markers
    (Porjesz et al., 2002)
  - Endorphin: post-exercise alpha rebound, pain threshold correlates
    (Fumoto et al., 2010)

Usage:
    estimator = NeurochemicalEstimator()
    result = estimator.estimate_neurochemical_state(eeg, fs=256)
    profile = estimator.compute_balance_profile("user1", eeg, fs=256)
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Maximum trend history entries per user.
_MAX_TREND_ENTRIES = 500

# Minimum readings for trend analysis.
_MIN_READINGS_FOR_TREND = 3


# ── Data classes ────────────────────────────────────────────────────


@dataclass
class EEGSpectralFeatures:
    """Extracted spectral features from EEG used for neurochemical estimation."""

    delta_power: float = 0.0
    theta_power: float = 0.0
    alpha_power: float = 0.0
    low_beta_power: float = 0.0
    high_beta_power: float = 0.0
    beta_power: float = 0.0
    gamma_power: float = 0.0
    frontal_alpha_asymmetry: float = 0.0  # FAA: log(right) - log(left)
    alpha_beta_ratio: float = 0.0
    theta_beta_ratio: float = 0.0
    high_beta_fraction: float = 0.0  # high_beta / total_beta
    total_power: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NeurochemicalEstimate:
    """Estimated level of a single neurochemical."""

    name: str
    level: float  # 0-1 normalized
    confidence: float  # 0-1
    description: str = ""
    contributing_features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NeurochemicalProfile:
    """Full neurochemical profile with balance assessment."""

    user_id: str
    timestamp: float
    estimates: List[NeurochemicalEstimate]
    balance_score: float  # 0-1, how balanced the system is overall
    dominant_system: str  # which neurochemical is most elevated
    depleted_system: str  # which neurochemical is most suppressed
    imbalances: List[Dict[str, Any]]  # list of detected imbalances
    spectral_features: EEGSpectralFeatures
    mood_inference: str  # brief mood state inferred from balance

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ── Helper utilities ────────────────────────────────────────────────


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator < 1e-12:
        return default
    return numerator / denominator


def _sigmoid(x: float, center: float = 0.0, scale: float = 1.0) -> float:
    """Sigmoid mapping to 0-1 range."""
    z = (x - center) * scale
    z = max(-20.0, min(20.0, z))  # prevent overflow
    return 1.0 / (1.0 + math.exp(-z))


# ── Core estimator ──────────────────────────────────────────────────


class NeurochemicalEstimator:
    """Estimate neurochemical balance from EEG spectral features.

    Maintains per-user baseline and trend history for longitudinal
    tracking.  All estimates are proxy-based approximations, not
    direct measurements.

    Args:
        max_trend_entries: Maximum trend history entries per user.
    """

    def __init__(self, max_trend_entries: int = _MAX_TREND_ENTRIES) -> None:
        self._max_trend_entries = max_trend_entries
        # Per-user trend storage: user_id -> list of (timestamp, estimates_dict)
        self._trend_history: Dict[str, List[Dict[str, Any]]] = {}
        # Per-user baseline: user_id -> dict of neurochemical baselines
        self._baselines: Dict[str, Dict[str, float]] = {}

    def _ensure_user(self, user_id: str) -> None:
        if user_id not in self._trend_history:
            self._trend_history[user_id] = []
        if user_id not in self._baselines:
            self._baselines[user_id] = {}

    # ── Feature extraction ──────────────────────────────────────────

    @staticmethod
    def extract_spectral_features(
        eeg: "np.ndarray",
        fs: float = 256.0,
    ) -> EEGSpectralFeatures:
        """Extract spectral features from raw EEG for neurochemical estimation.

        Args:
            eeg: EEG data, shape (n_samples,) or (n_channels, n_samples).
            fs: Sampling frequency in Hz.

        Returns:
            EEGSpectralFeatures with all band powers and derived ratios.
        """
        try:
            from processing.eeg_processor import (
                extract_band_powers,
                preprocess,
                compute_frontal_asymmetry,
            )
        except ImportError:
            # Fallback: compute from raw signal using scipy
            return NeurochemicalEstimator._extract_features_fallback(eeg, fs)

        channels = eeg if eeg.ndim == 2 else None
        signal = eeg[0] if eeg.ndim == 2 else eeg

        processed = preprocess(signal, fs)
        bands = extract_band_powers(processed, fs)

        delta = float(bands.get("delta", 0.0))
        theta = float(bands.get("theta", 0.0))
        alpha = float(bands.get("alpha", 0.0))
        beta = float(bands.get("beta", 0.0))
        gamma = float(bands.get("gamma", 0.0))
        low_beta = float(bands.get("low_beta", beta * 0.6))
        high_beta = float(bands.get("high_beta", beta * 0.4))

        total = delta + theta + alpha + beta + gamma
        if total < 1e-12:
            total = 1e-12

        # FAA
        faa = 0.0
        if channels is not None and channels.shape[0] >= 3:
            try:
                asym = compute_frontal_asymmetry(channels, fs, left_ch=1, right_ch=2)
                faa = float(asym.get("asymmetry_valence", 0.0))
            except Exception:
                pass

        return EEGSpectralFeatures(
            delta_power=round(delta, 6),
            theta_power=round(theta, 6),
            alpha_power=round(alpha, 6),
            low_beta_power=round(low_beta, 6),
            high_beta_power=round(high_beta, 6),
            beta_power=round(beta, 6),
            gamma_power=round(gamma, 6),
            frontal_alpha_asymmetry=round(faa, 6),
            alpha_beta_ratio=round(_safe_ratio(alpha, beta, 1.0), 6),
            theta_beta_ratio=round(_safe_ratio(theta, beta, 0.5), 6),
            high_beta_fraction=round(_safe_ratio(high_beta, beta, 0.4), 6),
            total_power=round(total, 6),
        )

    @staticmethod
    def _extract_features_fallback(
        eeg: "np.ndarray",
        fs: float = 256.0,
    ) -> EEGSpectralFeatures:
        """Fallback feature extraction using scipy only."""
        from scipy.signal import welch

        signal = eeg[0] if eeg.ndim == 2 else eeg
        nperseg = min(len(signal), int(fs * 2))
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

        def _band_power(f_low: float, f_high: float) -> float:
            mask = (freqs >= f_low) & (freqs < f_high)
            return float(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.0

        delta = _band_power(0.5, 4.0)
        theta = _band_power(4.0, 8.0)
        alpha = _band_power(8.0, 12.0)
        low_beta = _band_power(12.0, 20.0)
        high_beta = _band_power(20.0, 30.0)
        beta = low_beta + high_beta
        gamma = _band_power(30.0, 50.0)
        total = delta + theta + alpha + beta + gamma
        if total < 1e-12:
            total = 1e-12

        return EEGSpectralFeatures(
            delta_power=round(delta, 6),
            theta_power=round(theta, 6),
            alpha_power=round(alpha, 6),
            low_beta_power=round(low_beta, 6),
            high_beta_power=round(high_beta, 6),
            beta_power=round(beta, 6),
            gamma_power=round(gamma, 6),
            frontal_alpha_asymmetry=0.0,
            alpha_beta_ratio=round(_safe_ratio(alpha, beta, 1.0), 6),
            theta_beta_ratio=round(_safe_ratio(theta, beta, 0.5), 6),
            high_beta_fraction=round(_safe_ratio(high_beta, beta, 0.4), 6),
            total_power=round(total, 6),
        )

    # ── Individual neurochemical estimators ─────────────────────────

    @staticmethod
    def _estimate_dopamine(sf: EEGSpectralFeatures) -> NeurochemicalEstimate:
        """Dopamine proxy: frontal beta power (reward/motivation) + alpha suppression.

        Higher beta in frontal regions correlates with reward anticipation
        and dopaminergic activity. Alpha suppression during motivated states
        also indicates dopaminergic drive.
        """
        # Beta relative power as reward/motivation signal
        beta_rel = _safe_ratio(sf.beta_power, sf.total_power, 0.2)
        beta_component = _sigmoid(beta_rel, center=0.20, scale=12.0)

        # Alpha suppression: lower alpha relative to baseline = higher dopamine
        alpha_rel = _safe_ratio(sf.alpha_power, sf.total_power, 0.25)
        alpha_suppression = 1.0 - _sigmoid(alpha_rel, center=0.25, scale=10.0)

        # Positive FAA (left-frontal activation) correlates with approach/reward
        faa_component = _sigmoid(sf.frontal_alpha_asymmetry, center=0.0, scale=3.0)

        level = _clamp(0.40 * beta_component + 0.35 * alpha_suppression + 0.25 * faa_component)

        # Confidence based on total power (signal quality)
        confidence = _clamp(_sigmoid(sf.total_power, center=0.05, scale=20.0))

        return NeurochemicalEstimate(
            name="dopamine",
            level=round(level, 4),
            confidence=round(confidence, 4),
            description="Reward, motivation, and pleasure signaling",
            contributing_features=["beta_power", "alpha_suppression", "frontal_asymmetry"],
        )

    @staticmethod
    def _estimate_serotonin(sf: EEGSpectralFeatures) -> NeurochemicalEstimate:
        """Serotonin proxy: frontal alpha (calm contentment), emotional stability.

        Higher alpha power (especially frontal) with low irritability markers
        (low high-beta) indicates serotonergic tone.
        """
        # Alpha power: calm, content state
        alpha_rel = _safe_ratio(sf.alpha_power, sf.total_power, 0.25)
        alpha_component = _sigmoid(alpha_rel, center=0.20, scale=10.0)

        # Low irritability: inverse of high-beta fraction
        irritability = sf.high_beta_fraction
        calm_component = 1.0 - _sigmoid(irritability, center=0.45, scale=8.0)

        # Emotional stability: moderate theta/alpha ratio (not too high, not too low)
        stability = 1.0 - abs(sf.theta_beta_ratio - 0.8) / 1.5
        stability = _clamp(stability)

        level = _clamp(0.45 * alpha_component + 0.30 * calm_component + 0.25 * stability)
        confidence = _clamp(_sigmoid(sf.total_power, center=0.05, scale=20.0))

        return NeurochemicalEstimate(
            name="serotonin",
            level=round(level, 4),
            confidence=round(confidence, 4),
            description="Mood stability, calm contentment, emotional regulation",
            contributing_features=["alpha_power", "low_irritability", "emotional_stability"],
        )

    @staticmethod
    def _estimate_cortisol(sf: EEGSpectralFeatures) -> NeurochemicalEstimate:
        """Cortisol proxy: high beta/alpha ratio (stress), right frontal activation.

        Sustained elevated beta with reduced alpha indicates chronic stress
        and cortisol elevation.
        """
        # High beta/alpha ratio = stress
        ba_ratio = _safe_ratio(sf.beta_power, sf.alpha_power, 1.0)
        stress_component = _sigmoid(ba_ratio, center=1.0, scale=2.5)

        # High-beta fraction: anxiety/fight-or-flight
        hb_component = _sigmoid(sf.high_beta_fraction, center=0.40, scale=8.0)

        # Right frontal activation (negative FAA = right-dominant = withdrawal)
        right_frontal = _sigmoid(-sf.frontal_alpha_asymmetry, center=0.0, scale=3.0)

        level = _clamp(0.40 * stress_component + 0.35 * hb_component + 0.25 * right_frontal)
        confidence = _clamp(_sigmoid(sf.total_power, center=0.05, scale=20.0))

        return NeurochemicalEstimate(
            name="cortisol",
            level=round(level, 4),
            confidence=round(confidence, 4),
            description="Stress response, fight-or-flight activation",
            contributing_features=["beta_alpha_ratio", "high_beta", "right_frontal_activation"],
        )

    @staticmethod
    def _estimate_norepinephrine(sf: EEGSpectralFeatures) -> NeurochemicalEstimate:
        """Norepinephrine proxy: arousal level, overall beta, alertness markers.

        Higher overall beta and reduced theta indicate noradrenergic arousal
        and vigilance.
        """
        # Overall beta: arousal/alertness
        beta_rel = _safe_ratio(sf.beta_power, sf.total_power, 0.2)
        beta_arousal = _sigmoid(beta_rel, center=0.18, scale=12.0)

        # Low theta: inverse drowsiness (theta suppression = alert)
        theta_rel = _safe_ratio(sf.theta_power, sf.total_power, 0.2)
        alertness = 1.0 - _sigmoid(theta_rel, center=0.25, scale=8.0)

        # Low delta: awake, not drowsy
        delta_rel = _safe_ratio(sf.delta_power, sf.total_power, 0.3)
        wakefulness = 1.0 - _sigmoid(delta_rel, center=0.35, scale=6.0)

        level = _clamp(0.40 * beta_arousal + 0.35 * alertness + 0.25 * wakefulness)
        confidence = _clamp(_sigmoid(sf.total_power, center=0.05, scale=20.0))

        return NeurochemicalEstimate(
            name="norepinephrine",
            level=round(level, 4),
            confidence=round(confidence, 4),
            description="Arousal, alertness, and vigilance",
            contributing_features=["beta_power", "theta_suppression", "delta_suppression"],
        )

    @staticmethod
    def _estimate_gaba(sf: EEGSpectralFeatures) -> NeurochemicalEstimate:
        """GABA proxy: alpha power magnitude, relaxation depth, sleep onset markers.

        Strong alpha power with low beta indicates GABAergic inhibition
        and neural quieting.
        """
        # Alpha magnitude: relaxation/inhibition
        alpha_rel = _safe_ratio(sf.alpha_power, sf.total_power, 0.25)
        alpha_component = _sigmoid(alpha_rel, center=0.20, scale=10.0)

        # Alpha/beta ratio: relaxation depth
        relaxation = _sigmoid(sf.alpha_beta_ratio, center=1.0, scale=3.0)

        # Sleep onset: elevated theta + delta with alpha
        sleep_marker = _safe_ratio(
            sf.delta_power + sf.theta_power,
            sf.total_power,
            0.4,
        )
        drowsy_component = _sigmoid(sleep_marker, center=0.50, scale=6.0)

        level = _clamp(0.40 * alpha_component + 0.35 * relaxation + 0.25 * drowsy_component)
        confidence = _clamp(_sigmoid(sf.total_power, center=0.05, scale=20.0))

        return NeurochemicalEstimate(
            name="gaba",
            level=round(level, 4),
            confidence=round(confidence, 4),
            description="Neural inhibition, relaxation, and calming",
            contributing_features=["alpha_power", "alpha_beta_ratio", "sleep_onset_markers"],
        )

    @staticmethod
    def _estimate_endorphin(sf: EEGSpectralFeatures) -> NeurochemicalEstimate:
        """Endorphin proxy: post-exercise alpha rebound, pain threshold correlates.

        Alpha rebound (elevated alpha after exertion) and positive
        frontal asymmetry correlate with endorphin release.
        """
        # Alpha rebound: elevated alpha power (post-exercise or pain relief)
        alpha_rel = _safe_ratio(sf.alpha_power, sf.total_power, 0.25)
        alpha_rebound = _sigmoid(alpha_rel, center=0.25, scale=8.0)

        # Positive valence (approach motivation, positive affect)
        positive_affect = _sigmoid(sf.frontal_alpha_asymmetry, center=0.0, scale=3.0)

        # Low stress markers: low high-beta, good alpha presence
        low_stress = 1.0 - _sigmoid(sf.high_beta_fraction, center=0.45, scale=8.0)

        level = _clamp(0.40 * alpha_rebound + 0.30 * positive_affect + 0.30 * low_stress)
        confidence = _clamp(_sigmoid(sf.total_power, center=0.05, scale=20.0) * 0.8)

        return NeurochemicalEstimate(
            name="endorphin",
            level=round(level, 4),
            confidence=round(confidence, 4),
            description="Pain relief, euphoria, and well-being",
            contributing_features=["alpha_rebound", "positive_affect", "low_stress_markers"],
        )

    # ── Main estimation functions ──────────────────────────────────

    def estimate_neurochemical_state(
        self,
        eeg: "np.ndarray",
        fs: float = 256.0,
    ) -> Dict[str, Any]:
        """Estimate current neurochemical state from EEG.

        Args:
            eeg: EEG data, shape (n_samples,) or (n_channels, n_samples).
            fs: Sampling frequency in Hz.

        Returns:
            Dict with estimates for each neurochemical, spectral features,
            and overall balance assessment.
        """
        sf = self.extract_spectral_features(eeg, fs)

        estimates = [
            self._estimate_dopamine(sf),
            self._estimate_serotonin(sf),
            self._estimate_cortisol(sf),
            self._estimate_norepinephrine(sf),
            self._estimate_gaba(sf),
            self._estimate_endorphin(sf),
        ]

        estimates_dict = {e.name: e.to_dict() for e in estimates}

        levels = {e.name: e.level for e in estimates}
        dominant = max(levels, key=levels.get)  # type: ignore[arg-type]
        depleted = min(levels, key=levels.get)  # type: ignore[arg-type]

        # Balance score: inverse of standard deviation of levels
        vals = list(levels.values())
        mean_level = sum(vals) / len(vals)
        variance = sum((v - mean_level) ** 2 for v in vals) / len(vals)
        std_dev = math.sqrt(variance)
        balance_score = _clamp(1.0 - std_dev * 3.0)

        return {
            "estimates": estimates_dict,
            "spectral_features": sf.to_dict(),
            "dominant_system": dominant,
            "depleted_system": depleted,
            "balance_score": round(balance_score, 4),
            "mean_level": round(mean_level, 4),
            "timestamp": time.time(),
        }

    def compute_balance_profile(
        self,
        user_id: str,
        eeg: "np.ndarray",
        fs: float = 256.0,
    ) -> NeurochemicalProfile:
        """Compute full neurochemical profile with balance assessment.

        Also stores the result in trend history for the user.

        Args:
            user_id: User identifier.
            eeg: EEG data.
            fs: Sampling frequency.

        Returns:
            NeurochemicalProfile with full assessment.
        """
        self._ensure_user(user_id)

        sf = self.extract_spectral_features(eeg, fs)
        now = time.time()

        estimates = [
            self._estimate_dopamine(sf),
            self._estimate_serotonin(sf),
            self._estimate_cortisol(sf),
            self._estimate_norepinephrine(sf),
            self._estimate_gaba(sf),
            self._estimate_endorphin(sf),
        ]

        levels = {e.name: e.level for e in estimates}
        dominant = max(levels, key=levels.get)  # type: ignore[arg-type]
        depleted = min(levels, key=levels.get)  # type: ignore[arg-type]

        vals = list(levels.values())
        mean_level = sum(vals) / len(vals)
        variance = sum((v - mean_level) ** 2 for v in vals) / len(vals)
        std_dev = math.sqrt(variance)
        balance_score = _clamp(1.0 - std_dev * 3.0)

        imbalances = self.detect_imbalance(levels)
        mood = self._infer_mood(levels)

        profile = NeurochemicalProfile(
            user_id=user_id,
            timestamp=now,
            estimates=estimates,
            balance_score=round(balance_score, 4),
            dominant_system=dominant,
            depleted_system=depleted,
            imbalances=imbalances,
            spectral_features=sf,
            mood_inference=mood,
        )

        # Store in trend history
        entry = {
            "timestamp": now,
            "levels": levels,
            "balance_score": round(balance_score, 4),
            "dominant": dominant,
            "depleted": depleted,
        }
        self._trend_history[user_id].append(entry)
        if len(self._trend_history[user_id]) > self._max_trend_entries:
            self._trend_history[user_id] = self._trend_history[user_id][
                -self._max_trend_entries:
            ]

        return profile

    def detect_imbalance(
        self,
        levels: Dict[str, float],
        threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """Detect which neurochemical systems are high or low relative to mean.

        Args:
            levels: Dict of neurochemical name -> level (0-1).
            threshold: Deviation from mean to flag as imbalanced.

        Returns:
            List of imbalance dicts with name, direction, deviation.
        """
        if not levels:
            return []

        vals = list(levels.values())
        mean_level = sum(vals) / len(vals)

        imbalances = []
        for name, level in levels.items():
            deviation = level - mean_level
            if abs(deviation) > threshold:
                imbalances.append({
                    "system": name,
                    "direction": "elevated" if deviation > 0 else "depleted",
                    "level": round(level, 4),
                    "deviation": round(deviation, 4),
                    "severity": round(abs(deviation), 4),
                })

        # Sort by severity descending
        imbalances.sort(key=lambda x: x["severity"], reverse=True)
        return imbalances

    def track_neurochemical_trend(
        self,
        user_id: str,
        last_n: int = 20,
    ) -> Dict[str, Any]:
        """Get neurochemical trend data for a user.

        Args:
            user_id: User identifier.
            last_n: Number of most recent entries.

        Returns:
            Dict with trend data per neurochemical and overall balance trend.
        """
        self._ensure_user(user_id)

        history = self._trend_history[user_id][-last_n:]
        if len(history) < _MIN_READINGS_FOR_TREND:
            return {
                "user_id": user_id,
                "trend_available": False,
                "reason": "insufficient_data",
                "entries_count": len(history),
                "min_required": _MIN_READINGS_FOR_TREND,
            }

        # Collect per-neurochemical trend
        neurochemicals = ["dopamine", "serotonin", "cortisol",
                          "norepinephrine", "gaba", "endorphin"]
        trends: Dict[str, Dict[str, Any]] = {}

        for nc in neurochemicals:
            values = [e["levels"].get(nc, 0.0) for e in history]
            mean_val = sum(values) / len(values)

            # Simple trend direction: compare first half to second half
            mid = len(values) // 2
            first_half_mean = sum(values[:mid]) / max(mid, 1)
            second_half_mean = sum(values[mid:]) / max(len(values) - mid, 1)
            trend_direction = second_half_mean - first_half_mean

            if abs(trend_direction) < 0.05:
                direction_label = "stable"
            elif trend_direction > 0:
                direction_label = "increasing"
            else:
                direction_label = "decreasing"

            trends[nc] = {
                "mean": round(mean_val, 4),
                "latest": round(values[-1], 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "trend_direction": direction_label,
                "trend_magnitude": round(abs(trend_direction), 4),
            }

        # Balance trend
        balance_scores = [e["balance_score"] for e in history]
        mean_balance = sum(balance_scores) / len(balance_scores)

        return {
            "user_id": user_id,
            "trend_available": True,
            "entries_count": len(history),
            "neurochemicals": trends,
            "balance_trend": {
                "mean": round(mean_balance, 4),
                "latest": round(balance_scores[-1], 4),
                "min": round(min(balance_scores), 4),
                "max": round(max(balance_scores), 4),
            },
            "timestamps": {
                "first": history[0]["timestamp"],
                "last": history[-1]["timestamp"],
            },
        }

    def profile_to_dict(
        self,
        user_id: str,
        eeg: "np.ndarray",
        fs: float = 256.0,
    ) -> Dict[str, Any]:
        """Convenience: compute profile and return as dict.

        Args:
            user_id: User identifier.
            eeg: EEG data.
            fs: Sampling frequency.

        Returns:
            Profile as a plain dict.
        """
        profile = self.compute_balance_profile(user_id, eeg, fs)
        return profile.to_dict()

    def set_baseline(
        self,
        user_id: str,
        eeg: "np.ndarray",
        fs: float = 256.0,
    ) -> Dict[str, float]:
        """Record baseline neurochemical levels during resting state.

        Args:
            user_id: User identifier.
            eeg: Resting-state EEG data.
            fs: Sampling frequency.

        Returns:
            Dict of baseline levels per neurochemical.
        """
        self._ensure_user(user_id)

        sf = self.extract_spectral_features(eeg, fs)
        estimates = [
            self._estimate_dopamine(sf),
            self._estimate_serotonin(sf),
            self._estimate_cortisol(sf),
            self._estimate_norepinephrine(sf),
            self._estimate_gaba(sf),
            self._estimate_endorphin(sf),
        ]

        baseline = {e.name: e.level for e in estimates}
        self._baselines[user_id] = baseline
        logger.info("Set neurochemical baseline for user %s", user_id)
        return baseline

    def get_baseline(self, user_id: str) -> Optional[Dict[str, float]]:
        """Get stored baseline for a user. Returns None if no baseline set."""
        self._ensure_user(user_id)
        return self._baselines.get(user_id) or None

    def reset(self, user_id: str) -> None:
        """Clear all data for a user."""
        self._trend_history.pop(user_id, None)
        self._baselines.pop(user_id, None)
        logger.info("Reset neurochemical data for user %s", user_id)

    # ── Private helpers ─────────────────────────────────────────────

    @staticmethod
    def _infer_mood(levels: Dict[str, float]) -> str:
        """Infer a brief mood description from neurochemical levels."""
        dopamine = levels.get("dopamine", 0.5)
        serotonin = levels.get("serotonin", 0.5)
        cortisol = levels.get("cortisol", 0.5)
        norepinephrine = levels.get("norepinephrine", 0.5)
        gaba = levels.get("gaba", 0.5)
        endorphin = levels.get("endorphin", 0.5)

        # Simple rule-based mood inference
        if cortisol > 0.7 and norepinephrine > 0.6:
            return "stressed and hypervigilant"
        if cortisol > 0.7:
            return "stressed"
        if dopamine > 0.7 and serotonin > 0.6:
            return "motivated and content"
        if dopamine > 0.7:
            return "motivated and driven"
        if serotonin > 0.7 and gaba > 0.6:
            return "calm and peaceful"
        if serotonin > 0.7:
            return "content and stable"
        if gaba > 0.7:
            return "deeply relaxed"
        if norepinephrine > 0.7:
            return "alert and focused"
        if endorphin > 0.7:
            return "euphoric"
        if dopamine < 0.3 and serotonin < 0.3:
            return "low energy and flat affect"
        if cortisol > 0.5 and serotonin < 0.3:
            return "anxious"

        return "balanced"
