"""Emotional Intelligence composite index from EEG multi-metric analysis.

Aggregates five EI dimensions into a single trackable EIQ score (0-100).
Each dimension maps onto Bar-On's EI model adapted for EEG measurement:

    1. Self-perception   -- emotional awareness (FAA clarity, alpha coherence)
    2. Self-expression   -- emotional expressivity (FAA variability, beta reactivity)
    3. Interpersonal     -- emotional synchrony (inter-channel PLV, coherence)
    4. Decision-making   -- emotion-regulated decisions (theta/beta ratio, frontal theta)
    5. Stress-management -- stress resilience (alpha/high-beta ratio, approach asymmetry)

When pre-computed component scores from external EI modules are provided (e.g.,
granularity, flexibility, alexithymia), they are mapped onto the five dimensions
and blended with any EEG-derived scores.

References:
    Bar-On (2006) -- The Bar-On model of emotional-social intelligence
    Mayer, Salovey & Caruso (2008) -- four-branch EI model
    Davidson (1992) -- FAA and approach motivation
    Killgore (2019) -- EEG correlates of emotional intelligence

Channel layout (Muse 2, BrainFlow board_id 38):
    ch0 = TP9   (left temporal)
    ch1 = AF7   (left frontal)  -- FAA left
    ch2 = AF8   (right frontal) -- FAA right
    ch3 = TP10  (right temporal)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import butter, coherence, filtfilt, hilbert, welch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "high_beta": (20.0, 30.0),
    "gamma": (30.0, 45.0),
}

_MAX_HISTORY = 500
_MIN_SAMPLES = 64

# Grade thresholds
_GRADE_A = 80.0
_GRADE_B = 65.0
_GRADE_C = 50.0
_GRADE_D = 35.0

# Dimension names for external component mapping
_EXTERNAL_COMPONENTS = {
    "granularity",
    "flexibility",
    "synchrony",
    "interoception",
    "reactivity_regulation",
    "affect_labeling",
    "alexithymia",
    "emotional_memory",
    "mood_stability",
}

# Dimension names
_DIMENSIONS = [
    "self_perception",
    "self_expression",
    "interpersonal",
    "decision_making",
    "stress_management",
]


# ---------------------------------------------------------------------------
# Signal helpers (all scipy, no ml/ imports)
# ---------------------------------------------------------------------------


def _bandpass(signal: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    low_n = max(lo / nyq, 1e-5)
    high_n = min(hi / nyq, 0.9999)
    if low_n >= high_n:
        return signal.copy()
    b, a = butter(order, [low_n, high_n], btype="band")
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        return signal.copy()
    return filtfilt(b, a, signal)


def _band_power(signal: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """Mean PSD power in a frequency band via Welch."""
    nperseg = min(len(signal), int(fs * 2))
    if nperseg < 4:
        return 1e-9
    f, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (f >= lo) & (f <= hi)
    if not mask.any():
        return 1e-9
    return max(float(np.mean(psd[mask])), 1e-9)


def _plv_alpha(signal1: np.ndarray, signal2: np.ndarray, fs: float) -> float:
    """Phase-locking value in alpha band between two signals."""
    if len(signal1) < _MIN_SAMPLES or len(signal2) < _MIN_SAMPLES:
        return 0.0
    lo, hi = _BANDS["alpha"]
    f1 = _bandpass(signal1, fs, lo, hi)
    f2 = _bandpass(signal2, fs, lo, hi)
    a1 = hilbert(f1)
    a2 = hilbert(f2)
    phase_diff = np.angle(a1) - np.angle(a2)
    plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
    if np.isnan(plv):
        return 0.0
    return float(np.clip(plv, 0.0, 1.0))


def _mean_coherence(signal1: np.ndarray, signal2: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """Mean coherence between two signals in a frequency band."""
    if len(signal1) < _MIN_SAMPLES or len(signal2) < _MIN_SAMPLES:
        return 0.0
    nperseg = min(len(signal1), int(fs * 2))
    if nperseg < 4:
        return 0.0
    try:
        freqs, coh = coherence(signal1, signal2, fs=fs, nperseg=nperseg)
    except Exception:
        return 0.0
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return 0.0
    vals = np.nan_to_num(coh[mask], nan=0.0)
    return float(np.clip(np.mean(vals), 0.0, 1.0))


def _sigmoid_map(x: float, center: float = 0.0, scale: float = 1.0) -> float:
    """Map a value to 0-100 via sigmoid."""
    val = 1.0 / (1.0 + np.exp(-scale * (x - center)))
    return float(np.clip(val * 100.0, 0.0, 100.0))


def _ratio_to_score(ratio: float, low: float = 0.0, high: float = 2.0) -> float:
    """Linearly map a ratio to 0-100, clamped."""
    if high <= low:
        return 50.0
    normed = (ratio - low) / (high - low)
    return float(np.clip(normed * 100.0, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class EmotionalIntelligenceIndex:
    """Computes a composite Emotional Intelligence Quotient from EEG signals.

    Aggregates five dimensions into an EIQ score (0-100) with letter grade.
    Supports multi-user tracking, baseline calibration, and external component
    score injection from other EI modules.

    Args:
        fs: Default sampling rate in Hz.
    """

    def __init__(self, fs: float = 256.0) -> None:
        self._fs = fs
        # Per-user storage
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}
        self._components: Dict[str, Dict[str, float]] = {}

    # -- Public API ---------------------------------------------------------

    def compute_eiq(
        self,
        signals: Optional[np.ndarray] = None,
        fs: Optional[float] = None,
        component_scores: Optional[Dict[str, float]] = None,
        user_id: str = "default",
    ) -> Optional[Dict]:
        """Compute the composite EI score.

        Accepts either raw EEG signals (computes basic metrics internally) or
        pre-computed component scores dict, or both. When both are provided,
        EEG-derived dimension scores are blended with component-derived scores.

        Args:
            signals: EEG array, shape (n_channels, n_samples) or (n_samples,).
                     Can be None if component_scores are provided.
            fs: Sampling rate override. Uses default if None.
            component_scores: Dict mapping component names to scores (0-1 each).
                Recognized keys: granularity, flexibility, synchrony,
                interoception, reactivity_regulation, affect_labeling,
                alexithymia, emotional_memory, mood_stability.
            user_id: User identifier for multi-user support.

        Returns:
            Dict with eiq_score, eiq_grade, dimensions, strengths, growth_areas,
            has_baseline. Returns None if no signals and no component_scores.
        """
        sample_rate = fs if fs is not None else self._fs

        if signals is None and component_scores is None:
            # Check if we have stored components
            stored = self._components.get(user_id, {})
            if not stored:
                return None
            component_scores = stored

        # Compute EEG-based dimension scores
        eeg_dims: Optional[Dict[str, float]] = None
        if signals is not None:
            signals = np.asarray(signals, dtype=np.float64)
            # Sanitize NaN/inf
            signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
            if signals.ndim == 1:
                signals = signals[np.newaxis, :]
            eeg_dims = self._compute_dimensions_from_eeg(signals, sample_rate, user_id)

        # Compute component-based dimension scores
        comp_dims: Optional[Dict[str, float]] = None
        # Merge stored components with any passed in
        merged_components = dict(self._components.get(user_id, {}))
        if component_scores is not None:
            for k, v in component_scores.items():
                if k in _EXTERNAL_COMPONENTS:
                    merged_components[k] = float(np.clip(v, 0.0, 1.0))
        if merged_components and any(k in _EXTERNAL_COMPONENTS for k in merged_components):
            comp_dims = self._map_components_to_dimensions(merged_components)

        # Blend EEG and component dimensions
        dimensions = self._blend_dimensions(eeg_dims, comp_dims)

        # Compute EIQ = equal-weighted mean of 5 dimensions
        dim_values = [dimensions[d] for d in _DIMENSIONS]
        eiq_score = float(np.mean(dim_values))
        eiq_score = float(np.clip(eiq_score, 0.0, 100.0))

        # Grade
        eiq_grade = self._grade(eiq_score)

        # Strengths and growth areas
        strengths = [d for d in _DIMENSIONS if dimensions[d] > 70.0]
        growth_areas = [d for d in _DIMENSIONS if dimensions[d] < 40.0]

        has_baseline = user_id in self._baselines

        result = {
            "eiq_score": round(eiq_score, 2),
            "eiq_grade": eiq_grade,
            "dimensions": {d: round(dimensions[d], 2) for d in _DIMENSIONS},
            "strengths": strengths,
            "growth_areas": growth_areas,
            "has_baseline": has_baseline,
        }

        # Append to history
        self._ensure_history(user_id)
        self._history[user_id].append(result)
        if len(self._history[user_id]) > _MAX_HISTORY:
            self._history[user_id] = self._history[user_id][-_MAX_HISTORY:]

        return result

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record resting-state baseline for a user.

        Call during 2-3 min eyes-closed resting state. Baseline values are
        used to normalize EEG-derived dimension scores.

        Args:
            signals: EEG array, shape (n_channels, n_samples) or (n_samples,).
            fs: Sampling rate override.
            user_id: User identifier.

        Returns:
            Dict with baseline_set (bool), n_channels (int), n_samples (int).
        """
        sample_rate = fs if fs is not None else self._fs
        signals = np.asarray(signals, dtype=np.float64)
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

        if signals.ndim == 1:
            signals = signals[np.newaxis, :]

        n_channels, n_samples = signals.shape

        # Compute baseline band powers and metrics
        baseline = {}
        for ch_idx in range(n_channels):
            sig = signals[ch_idx]
            for band_name, (lo, hi) in _BANDS.items():
                key = f"ch{ch_idx}_{band_name}"
                baseline[key] = _band_power(sig, sample_rate, lo, hi)

        # FAA baseline
        if n_channels >= 3:
            left_alpha = _band_power(signals[1], sample_rate, *_BANDS["alpha"])
            right_alpha = _band_power(signals[2], sample_rate, *_BANDS["alpha"])
            baseline["faa"] = float(np.log(right_alpha) - np.log(left_alpha))
        else:
            baseline["faa"] = 0.0

        baseline["n_channels"] = n_channels
        baseline["n_samples"] = n_samples
        baseline["fs"] = sample_rate

        self._baselines[user_id] = baseline

        return {
            "baseline_set": True,
            "n_channels": n_channels,
            "n_samples": n_samples,
        }

    def update_component(
        self,
        component_name: str,
        score: float,
        user_id: str = "default",
    ) -> None:
        """Update a specific component score from an external EI module.

        Args:
            component_name: One of: granularity, flexibility, synchrony,
                interoception, reactivity_regulation, affect_labeling,
                alexithymia, emotional_memory, mood_stability.
            score: Score value in [0, 1].
            user_id: User identifier.
        """
        if component_name not in _EXTERNAL_COMPONENTS:
            logger.warning("Unknown EI component: %s", component_name)
            return
        if user_id not in self._components:
            self._components[user_id] = {}
        self._components[user_id][component_name] = float(np.clip(score, 0.0, 1.0))

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get session-level statistics for a user.

        Returns:
            Dict with n_assessments, mean_eiq (float or None), trend (str or None).
        """
        self._ensure_history(user_id)
        history = self._history[user_id]

        n = len(history)
        if n == 0:
            return {
                "n_assessments": 0,
                "mean_eiq": None,
                "trend": None,
            }

        scores = [h["eiq_score"] for h in history]
        mean_eiq = round(float(np.mean(scores)), 2)

        # Trend: compare last third to first third
        trend = None
        if n >= 3:
            third = max(n // 3, 1)
            first_avg = float(np.mean(scores[:third]))
            last_avg = float(np.mean(scores[-third:]))
            diff = last_avg - first_avg
            if diff > 3.0:
                trend = "improving"
            elif diff < -3.0:
                trend = "declining"
            else:
                trend = "stable"

        return {
            "n_assessments": n,
            "mean_eiq": mean_eiq,
            "trend": trend,
        }

    def get_history(
        self,
        user_id: str = "default",
        last_n: Optional[int] = None,
    ) -> List[Dict]:
        """Get EIQ computation history for a user.

        Args:
            user_id: User identifier.
            last_n: If provided, return only the last N entries.

        Returns:
            List of EIQ result dicts.
        """
        self._ensure_history(user_id)
        history = self._history[user_id]
        if last_n is not None and last_n > 0:
            return list(history[-last_n:])
        return list(history)

    def reset(self, user_id: str = "default") -> None:
        """Clear all data for a user (baseline, history, components)."""
        self._baselines.pop(user_id, None)
        self._history.pop(user_id, None)
        self._components.pop(user_id, None)

    # -- Private: EEG-based dimension computation ----------------------------

    def _compute_dimensions_from_eeg(
        self,
        signals: np.ndarray,
        fs: float,
        user_id: str,
    ) -> Dict[str, float]:
        """Compute the 5 EI dimensions from raw EEG signals."""
        n_channels, n_samples = signals.shape

        # Extract band powers per channel
        ch_powers: List[Dict[str, float]] = []
        for ch_idx in range(n_channels):
            sig = signals[ch_idx]
            powers = {}
            for band_name, (lo, hi) in _BANDS.items():
                powers[band_name] = _band_power(sig, fs, lo, hi)
            ch_powers.append(powers)

        # Average band powers across channels
        avg_powers: Dict[str, float] = {}
        for band_name in _BANDS:
            avg_powers[band_name] = float(np.mean([cp[band_name] for cp in ch_powers]))

        alpha = avg_powers["alpha"]
        beta = avg_powers["beta"]
        theta = avg_powers["theta"]
        high_beta = avg_powers["high_beta"]
        delta = avg_powers["delta"]

        # FAA
        faa = 0.0
        if n_channels >= 3:
            left_alpha = ch_powers[1]["alpha"]
            right_alpha = ch_powers[2]["alpha"]
            faa = float(np.log(right_alpha) - np.log(left_alpha))

        # Baseline normalization if available
        baseline = self._baselines.get(user_id)
        baseline_faa = 0.0
        if baseline is not None:
            baseline_faa = baseline.get("faa", 0.0)

        # ---- 1. Self-perception (0-100) ----
        # FAA clarity: how well-defined the FAA signal is (distance from zero)
        faa_clarity = _sigmoid_map(abs(faa - baseline_faa), center=0.1, scale=8.0)
        # Alpha coherence between frontal-temporal pairs
        alpha_coh = 0.0
        if n_channels >= 4:
            c1 = _mean_coherence(signals[1], signals[0], fs, *_BANDS["alpha"])
            c2 = _mean_coherence(signals[2], signals[3], fs, *_BANDS["alpha"])
            alpha_coh = (c1 + c2) / 2.0
        elif n_channels >= 2:
            alpha_coh = _mean_coherence(signals[0], signals[1], fs, *_BANDS["alpha"])
        alpha_coh_score = float(np.clip(alpha_coh * 100.0, 0.0, 100.0))
        self_perception = 0.60 * faa_clarity + 0.40 * alpha_coh_score

        # ---- 2. Self-expression (0-100) ----
        # FAA variability: split epoch into sub-windows, compute FAA std
        faa_var_score = 50.0  # default
        if n_channels >= 3 and n_samples >= _MIN_SAMPLES * 2:
            n_windows = min(8, n_samples // _MIN_SAMPLES)
            if n_windows >= 2:
                win_len = n_samples // n_windows
                faa_vals = []
                for w in range(n_windows):
                    start = w * win_len
                    end = start + win_len
                    left_a = _band_power(signals[1, start:end], fs, *_BANDS["alpha"])
                    right_a = _band_power(signals[2, start:end], fs, *_BANDS["alpha"])
                    faa_vals.append(float(np.log(right_a) - np.log(left_a)))
                faa_std = float(np.std(faa_vals))
                faa_var_score = _sigmoid_map(faa_std, center=0.15, scale=10.0)

        # Beta amplitude variability at frontal channels
        beta_var_score = 50.0
        if n_channels >= 3 and n_samples >= _MIN_SAMPLES * 2:
            n_windows = min(8, n_samples // _MIN_SAMPLES)
            if n_windows >= 2:
                win_len = n_samples // n_windows
                beta_vals = []
                for w in range(n_windows):
                    start = w * win_len
                    end = start + win_len
                    b1 = _band_power(signals[1, start:end], fs, *_BANDS["beta"])
                    b2 = _band_power(signals[2, start:end], fs, *_BANDS["beta"])
                    beta_vals.append((b1 + b2) / 2.0)
                beta_cv = float(np.std(beta_vals) / (np.mean(beta_vals) + 1e-9))
                beta_var_score = _sigmoid_map(beta_cv, center=0.3, scale=6.0)

        self_expression = 0.50 * faa_var_score + 0.50 * beta_var_score

        # ---- 3. Interpersonal (0-100) ----
        # Mean PLV across all channel pairs in alpha band
        plv_scores = []
        if n_channels >= 2 and n_samples >= _MIN_SAMPLES:
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    plv_scores.append(_plv_alpha(signals[i], signals[j], fs))
        mean_plv = float(np.mean(plv_scores)) if plv_scores else 0.0
        plv_score = float(np.clip(mean_plv * 100.0, 0.0, 100.0))
        # Beta coherence across pairs
        beta_coh_vals = []
        if n_channels >= 2 and n_samples >= _MIN_SAMPLES:
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    beta_coh_vals.append(
                        _mean_coherence(signals[i], signals[j], fs, *_BANDS["beta"])
                    )
        mean_beta_coh = float(np.mean(beta_coh_vals)) if beta_coh_vals else 0.0
        beta_coh_score = float(np.clip(mean_beta_coh * 100.0, 0.0, 100.0))
        interpersonal = 0.60 * plv_score + 0.40 * beta_coh_score

        # ---- 4. Decision-making (0-100) ----
        # Theta/beta ratio: higher = more deliberative processing
        tb_ratio = theta / (beta + 1e-9)
        tb_score = _ratio_to_score(tb_ratio, low=0.2, high=2.0)
        # Frontal theta power (from frontal channels if available)
        if n_channels >= 3:
            frontal_theta = (ch_powers[1]["theta"] + ch_powers[2]["theta"]) / 2.0
        else:
            frontal_theta = ch_powers[0]["theta"]
        ft_score = _sigmoid_map(np.log1p(frontal_theta), center=0.5, scale=3.0)
        decision_making = 0.50 * tb_score + 0.50 * ft_score

        # ---- 5. Stress-management (0-100) ----
        # Alpha / (alpha + high_beta) ratio: higher = more relaxed
        alpha_hb_ratio = alpha / (alpha + high_beta + 1e-9)
        relax_score = _ratio_to_score(alpha_hb_ratio, low=0.3, high=0.9)
        # Alpha asymmetry: left-dominant (positive FAA) = approach = resilient
        faa_adjusted = faa - baseline_faa
        asym_score = _sigmoid_map(faa_adjusted, center=0.0, scale=5.0)
        stress_management = 0.60 * relax_score + 0.40 * asym_score

        return {
            "self_perception": float(np.clip(self_perception, 0.0, 100.0)),
            "self_expression": float(np.clip(self_expression, 0.0, 100.0)),
            "interpersonal": float(np.clip(interpersonal, 0.0, 100.0)),
            "decision_making": float(np.clip(decision_making, 0.0, 100.0)),
            "stress_management": float(np.clip(stress_management, 0.0, 100.0)),
        }

    # -- Private: Component-to-dimension mapping ----------------------------

    def _map_components_to_dimensions(
        self,
        components: Dict[str, float],
    ) -> Dict[str, float]:
        """Map external component scores (0-1) to dimension scores (0-100)."""
        dims: Dict[str, List[float]] = {d: [] for d in _DIMENSIONS}

        # self_perception: granularity, alexithymia (inverted), interoception
        if "granularity" in components:
            dims["self_perception"].append(components["granularity"] * 100.0)
        if "alexithymia" in components:
            dims["self_perception"].append((1.0 - components["alexithymia"]) * 100.0)
        if "interoception" in components:
            dims["self_perception"].append(components["interoception"] * 100.0)

        # self_expression: flexibility, affect_labeling
        if "flexibility" in components:
            dims["self_expression"].append(components["flexibility"] * 100.0)
        if "affect_labeling" in components:
            dims["self_expression"].append(components["affect_labeling"] * 100.0)

        # interpersonal: synchrony
        if "synchrony" in components:
            dims["interpersonal"].append(components["synchrony"] * 100.0)

        # decision_making: emotional_memory
        if "emotional_memory" in components:
            dims["decision_making"].append(components["emotional_memory"] * 100.0)

        # stress_management: reactivity_regulation, mood_stability
        if "reactivity_regulation" in components:
            dims["stress_management"].append(components["reactivity_regulation"] * 100.0)
        if "mood_stability" in components:
            dims["stress_management"].append(components["mood_stability"] * 100.0)

        # Average within each dimension; default to 50 if no components mapped
        result: Dict[str, float] = {}
        for d in _DIMENSIONS:
            if dims[d]:
                result[d] = float(np.clip(np.mean(dims[d]), 0.0, 100.0))
            else:
                result[d] = 50.0

        return result

    # -- Private: Blend EEG and component dimensions -------------------------

    def _blend_dimensions(
        self,
        eeg_dims: Optional[Dict[str, float]],
        comp_dims: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """Blend EEG-derived and component-derived dimension scores.

        If both are available, uses 50/50 blend.
        If only one is available, uses that one.
        """
        if eeg_dims is not None and comp_dims is not None:
            return {
                d: float(np.clip(0.50 * eeg_dims[d] + 0.50 * comp_dims[d], 0.0, 100.0))
                for d in _DIMENSIONS
            }
        if eeg_dims is not None:
            return eeg_dims
        if comp_dims is not None:
            return comp_dims
        # Should not reach here; return neutral
        return {d: 50.0 for d in _DIMENSIONS}

    # -- Private: Grading ---------------------------------------------------

    @staticmethod
    def _grade(score: float) -> str:
        """Map EIQ score to letter grade."""
        if score >= _GRADE_A:
            return "A"
        elif score >= _GRADE_B:
            return "B"
        elif score >= _GRADE_C:
            return "C"
        elif score >= _GRADE_D:
            return "D"
        else:
            return "F"

    # -- Private: Helpers ---------------------------------------------------

    def _ensure_history(self, user_id: str) -> None:
        """Initialize history list for a user if not present."""
        if user_id not in self._history:
            self._history[user_id] = []


# ---------------------------------------------------------------------------
# Module-level singleton (matches project convention)
# ---------------------------------------------------------------------------

_model = EmotionalIntelligenceIndex()


def get_model() -> EmotionalIntelligenceIndex:
    return _model
