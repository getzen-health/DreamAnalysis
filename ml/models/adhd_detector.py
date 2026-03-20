"""ADHD attention profile detector using validated EEG biomarkers.

Screens for ADHD-like attention patterns using theta/beta ratio (TBR),
theta power excess, beta deficit, alpha peak frequency variability,
and frontal beta inhibition proxy. Supports multi-user tracking,
baseline calibration, session statistics, and assessment history.

DISCLAIMER: This is a screening tool only, NOT a clinical diagnostic
instrument. Clinical ADHD diagnosis requires evaluation by a qualified
healthcare professional using standardized assessment instruments
(DSM-5 criteria, behavioral rating scales, neuropsychological testing).

Scientific references:
    Arns et al. (2013) — Meta-analysis of EEG theta/beta ratio in ADHD
    Monastra et al. (2001) — TBR as quantitative EEG marker for ADHD
    Snyder et al. (2015) — FDA-cleared TBR biomarker evaluation
    Clarke et al. (2001) — EEG subtypes in ADHD (inattentive vs combined)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.signal import welch


DISCLAIMER = (
    "This is a screening tool only, not a clinical diagnostic instrument. "
    "ADHD diagnosis requires evaluation by a qualified healthcare professional."
)

# Normative TBR ranges by age group (from Arns et al., 2013; Monastra et al., 2001)
# Values represent population mean TBR at frontal sites
_NORMATIVE_TBR = {
    "child": {"mean": 5.0, "std": 1.5, "range": (4.0, 6.0)},
    "adolescent": {"mean": 3.75, "std": 1.0, "range": (3.0, 4.5)},
    "adult": {"mean": 2.75, "std": 0.75, "range": (2.0, 3.5)},
}

# NumPy 2.0 renamed np.trapz -> np.trapezoid
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


class ADHDDetector:
    """ADHD attention profile screener using EEG biomarkers.

    Primary biomarker: Theta/Beta Ratio (TBR) at frontal sites (AF7/AF8).
    Secondary biomarkers: theta excess, beta deficit, alpha peak variability,
    frontal beta as response inhibition proxy.

    Muse 2 channel order: ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10.
    Frontal channels (AF7+AF8) are used for TBR computation.
    """

    def __init__(
        self,
        age_group: str = "adult",
        fs: float = 256.0,
        history_limit: int = 1000,
    ):
        if age_group not in _NORMATIVE_TBR:
            age_group = "adult"
        self._age_group = age_group
        self._fs = fs
        self._history_limit = history_limit

        # Per-user state
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}

    # ── Public API ────────────────────────────────────────

    def set_baseline(
        self,
        eeg: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record resting-state baseline for TBR normalization.

        Args:
            eeg: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate in Hz.
            user_id: User identifier for multi-user support.

        Returns:
            Dict with baseline_set, baseline_tbr, baseline_theta,
            baseline_beta, and baseline_alpha.
        """
        fs = fs or self._fs
        eeg = np.asarray(eeg, dtype=float)

        frontal = self._get_frontal(eeg)
        bands = self._extract_bands(frontal, fs)

        baseline = {
            "tbr": bands["tbr"],
            "theta": bands["theta"],
            "beta": bands["beta"],
            "alpha": bands["alpha"],
        }
        self._baselines[user_id] = baseline

        return {
            "baseline_set": True,
            "baseline_tbr": round(baseline["tbr"], 4),
            "baseline_theta": round(baseline["theta"], 6),
            "baseline_beta": round(baseline["beta"], 6),
            "baseline_alpha": round(baseline["alpha"], 6),
        }

    def assess(
        self,
        eeg: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Assess ADHD-like attention patterns from EEG.

        Args:
            eeg: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate in Hz.
            user_id: User identifier.

        Returns:
            Dict with tbr_score, tbr_percentile, attention_variability,
            inhibition_index, risk_level, attention_profile,
            component_scores, and disclaimer.
        """
        fs = fs or self._fs
        eeg = np.asarray(eeg, dtype=float)

        frontal = self._get_frontal(eeg)
        bands = self._extract_bands(frontal, fs)

        tbr = bands["tbr"]
        theta = bands["theta"]
        beta = bands["beta"]
        alpha = bands["alpha"]

        # -- TBR percentile against normative data --
        norms = _NORMATIVE_TBR[self._age_group]
        tbr_z = (tbr - norms["mean"]) / max(norms["std"], 1e-10)
        # Convert z-score to percentile (0-100) using sigmoid approximation
        tbr_percentile = float(100.0 / (1.0 + np.exp(-tbr_z)))

        # -- Component scores (all 0-1) --
        # TBR component: how elevated is TBR vs norm
        tbr_component = float(np.clip(tbr_z / 4.0 * 0.5 + 0.5, 0.0, 1.0))

        # Theta excess: relative theta power
        total_power = theta + alpha + beta + 1e-10
        theta_relative = theta / total_power
        # Typical theta relative: ~0.25-0.35 for adults
        theta_excess = float(np.clip((theta_relative - 0.25) / 0.30, 0.0, 1.0))

        # Beta deficit: inverse of relative beta power
        beta_relative = beta / total_power
        # Typical beta relative: ~0.30-0.45 for adults
        beta_deficit = float(np.clip((0.40 - beta_relative) / 0.30, 0.0, 1.0))

        # Alpha peak frequency variability (lower IAF = higher risk)
        iaf = self._find_alpha_peak(frontal, fs)
        # Typical IAF ~10 Hz; lower values (<9 Hz) associated with attention issues
        alpha_variability = float(np.clip((10.0 - iaf) / 4.0, 0.0, 1.0))

        component_scores = {
            "tbr_component": round(tbr_component, 4),
            "theta_excess": round(theta_excess, 4),
            "beta_deficit": round(beta_deficit, 4),
            "alpha_variability": round(alpha_variability, 4),
        }

        # -- Inhibition index (frontal beta as proxy for response inhibition) --
        # Higher frontal beta = better inhibition; lower = worse
        # Normalize: typical frontal beta power ~0.15-0.35
        inhibition_index = float(np.clip(beta_relative / 0.45, 0.0, 1.0))

        # -- Attention variability (how much TBR fluctuates across assessments) --
        history = self._history.get(user_id, [])
        if len(history) >= 2:
            past_tbrs = [h["tbr_score"] for h in history]
            tbr_cv = float(np.std(past_tbrs) / max(np.mean(past_tbrs), 1e-10))
            attention_variability = float(np.clip(tbr_cv / 0.5, 0.0, 1.0))
        else:
            attention_variability = 0.0

        # -- Composite risk score --
        risk_score = (
            0.40 * tbr_component
            + 0.25 * theta_excess
            + 0.20 * beta_deficit
            + 0.15 * alpha_variability
        )
        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        # -- Risk level --
        if risk_score >= 0.70:
            risk_level = "high"
        elif risk_score >= 0.55:
            risk_level = "elevated"
        elif risk_score >= 0.35:
            risk_level = "moderate"
        else:
            risk_level = "low"

        # -- Attention profile classification --
        # Based on Clarke et al. (2001) EEG subtypes
        attention_profile = self._classify_profile(
            tbr, theta_excess, beta_deficit, beta_relative,
        )

        result = {
            "tbr_score": round(tbr, 4),
            "tbr_percentile": round(tbr_percentile, 2),
            "attention_variability": round(attention_variability, 4),
            "inhibition_index": round(inhibition_index, 4),
            "risk_level": risk_level,
            "risk_score": round(risk_score, 4),
            "attention_profile": attention_profile,
            "component_scores": component_scores,
            "theta_power": round(float(theta), 6),
            "beta_power": round(float(beta), 6),
            "alpha_power": round(float(alpha), 6),
            "iaf_hz": round(iaf, 2),
            "has_baseline": user_id in self._baselines,
            "age_group": self._age_group,
            "disclaimer": DISCLAIMER,
            "not_validated": True,
            "scale_context": (
                "EEG pattern scores are research-grade estimates from consumer hardware. "
                "They have not been validated against clinical diagnostic instruments."
            ),
        }

        # Store in history
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > self._history_limit:
            self._history[user_id] = self._history[user_id][-self._history_limit:]

        return result

    def get_attention_profile(self, user_id: str = "default") -> Dict:
        """Get the latest attention profile for a user.

        Returns:
            Dict with profile, risk_level, and summary, or empty dict
            if no assessments exist.
        """
        history = self._history.get(user_id, [])
        if not history:
            return {"status": "no_assessments"}
        latest = history[-1]
        return {
            "attention_profile": latest["attention_profile"],
            "risk_level": latest["risk_level"],
            "tbr_score": latest["tbr_score"],
            "tbr_percentile": latest["tbr_percentile"],
            "inhibition_index": latest["inhibition_index"],
            "disclaimer": DISCLAIMER,
        }

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get aggregate statistics for the current session.

        Returns:
            Dict with n_assessments, mean_tbr, mean_risk_score,
            dominant_profile, and has_baseline.
        """
        history = self._history.get(user_id, [])
        if not history:
            return {
                "n_assessments": 0,
                "has_baseline": user_id in self._baselines,
            }

        tbrs = [h["tbr_score"] for h in history]
        risk_scores = [h["risk_score"] for h in history]
        profiles = [h["attention_profile"] for h in history]

        # Find dominant profile
        profile_counts: Dict[str, int] = {}
        for p in profiles:
            profile_counts[p] = profile_counts.get(p, 0) + 1
        dominant_profile = max(profile_counts, key=profile_counts.get)

        return {
            "n_assessments": len(history),
            "mean_tbr": round(float(np.mean(tbrs)), 4),
            "std_tbr": round(float(np.std(tbrs)), 4),
            "mean_risk_score": round(float(np.mean(risk_scores)), 4),
            "dominant_profile": dominant_profile,
            "has_baseline": user_id in self._baselines,
        }

    def get_history(
        self,
        user_id: str = "default",
        last_n: Optional[int] = None,
    ) -> List[Dict]:
        """Get assessment history for a user.

        Args:
            user_id: User identifier.
            last_n: Return only the last N entries. None = all.

        Returns:
            List of assessment result dicts.
        """
        history = self._history.get(user_id, [])
        if last_n is not None and last_n > 0:
            return history[-last_n:]
        return list(history)

    def reset(self, user_id: Optional[str] = None):
        """Clear session state.

        Args:
            user_id: If provided, clear only that user. If None, clear all.
        """
        if user_id is not None:
            self._baselines.pop(user_id, None)
            self._history.pop(user_id, None)
        else:
            self._baselines.clear()
            self._history.clear()

    # ── Private helpers ───────────────────────────────────

    def _get_frontal(self, eeg: np.ndarray) -> np.ndarray:
        """Extract frontal signal (average of AF7+AF8 if multichannel)."""
        if eeg.ndim == 2 and eeg.shape[0] >= 3:
            # AF7 = ch1, AF8 = ch2 for Muse 2
            return (eeg[1] + eeg[2]) / 2.0
        elif eeg.ndim == 2:
            return eeg[0]
        return eeg

    def _extract_bands(self, signal: np.ndarray, fs: float) -> Dict:
        """Extract theta, alpha, beta band powers and TBR."""
        nperseg = min(len(signal), int(fs * 2))
        if nperseg < 4:
            return {"theta": 0.0, "alpha": 0.0, "beta": 0.0, "tbr": 0.0}

        try:
            freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        except Exception:
            return {"theta": 0.0, "alpha": 0.0, "beta": 0.0, "tbr": 0.0}

        theta = self._band_power(freqs, psd, 4.0, 8.0)
        alpha = self._band_power(freqs, psd, 8.0, 12.0)
        beta = self._band_power(freqs, psd, 12.0, 30.0)

        eps = 1e-10
        tbr = float(theta / (beta + eps))

        return {
            "theta": float(theta),
            "alpha": float(alpha),
            "beta": float(beta),
            "tbr": tbr,
        }

    def _band_power(
        self, freqs: np.ndarray, psd: np.ndarray, low: float, high: float,
    ) -> float:
        """Compute power in a frequency band via trapezoidal integration."""
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 0.0
        return float(_trapezoid(psd[mask], freqs[mask]))

    def _find_alpha_peak(self, signal: np.ndarray, fs: float) -> float:
        """Find individual alpha peak frequency (IAF)."""
        nperseg = min(len(signal), int(fs * 2))
        if nperseg < 4:
            return 10.0

        try:
            freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        except Exception:
            return 10.0

        alpha_mask = (freqs >= 7) & (freqs <= 13)
        if not alpha_mask.any():
            return 10.0

        return float(freqs[alpha_mask][np.argmax(psd[alpha_mask])])

    def _classify_profile(
        self,
        tbr: float,
        theta_excess: float,
        beta_deficit: float,
        beta_relative: float,
    ) -> str:
        """Classify EEG attention pattern.

        Based on Clarke et al. (2001) EEG subtypes:
        - theta_dominant: elevated TBR, high theta, normal/low beta
        - beta_deficit: normal/low TBR, low beta, elevated beta variability
        - mixed_pattern: elevated TBR + low beta
        - typical: normal TBR, balanced powers
        """
        norms = _NORMATIVE_TBR[self._age_group]
        tbr_elevated = tbr > (norms["mean"] + norms["std"])
        high_theta = theta_excess > 0.4
        low_beta = beta_deficit > 0.4

        if tbr_elevated and high_theta and low_beta:
            return "mixed_pattern"
        elif tbr_elevated or high_theta:
            return "theta_dominant"
        elif low_beta and beta_relative < 0.25:
            return "beta_deficit"
        return "typical"
