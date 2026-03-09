"""Autism spectrum EEG screening using validated biomarkers from 4-channel Muse 2.

DISCLAIMER: This is a research screening tool only. It is NOT a clinical
diagnostic instrument. Autism spectrum diagnosis requires comprehensive
evaluation by a qualified healthcare professional using standardized
assessment instruments (ADOS-2, ADI-R, DSM-5 criteria).

Scientific basis (2024-2025 literature):
    - Increased theta/beta ratio with distinct topographic pattern vs ADHD
      (Wang et al., 2023; Dickinson et al., 2018)
    - Reduced mu (8-13 Hz) suppression at temporal sites during social
      observation -- mirror neuron dysfunction (Oberman et al., 2005;
      Bernier et al., 2007)
    - Atypical frontal alpha asymmetry patterns (Burnette et al., 2011)
    - Reduced inter-channel coherence / functional connectivity
      (Murias et al., 2007; Coben et al., 2008)
    - Higher spectral entropy in some frequency bands (Bosl et al., 2011)
    - U-shaped power profile: excess low-frequency AND high-frequency power
      (Wang et al., 2013)

Muse 2 channel order: ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10 (256 Hz).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.signal import welch, coherence as scipy_coherence

# NumPy 2.0 renamed np.trapz -> np.trapezoid
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

MEDICAL_DISCLAIMER = (
    "This is a research screening tool only, not a clinical diagnostic instrument. "
    "Autism spectrum diagnosis requires comprehensive evaluation by a qualified "
    "healthcare professional using standardized assessment instruments "
    "(ADOS-2, ADI-R, DSM-5 criteria)."
)

# Frequency band definitions (Hz)
_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "mu": (8.0, 13.0),
    "beta": (12.0, 30.0),
    "low_beta": (12.0, 20.0),
    "high_beta": (20.0, 30.0),
    "gamma": (30.0, 100.0),
}

# History cap per user
_HISTORY_LIMIT = 500


class AutismScreener:
    """Autism spectrum EEG biomarker screener for 4-channel Muse 2.

    Computes six biomarker dimensions from raw EEG:
      1. Theta/beta ratio (TBR) -- elevated in ASD with frontal-temporal gradient
      2. Mu suppression index -- reduced suppression indicates mirror neuron atypicality
      3. Inter-channel coherence -- reduced functional connectivity in ASD
      4. Alpha asymmetry atypicality -- atypical frontal asymmetry patterns
      5. Spectral entropy -- higher entropy in some bands reflects disorganized cortical activity
      6. U-shaped power profile -- excess delta AND gamma relative to mid-bands

    Output: atypicality_score (0-100) and categorical risk_level.
    """

    def __init__(self, fs: float = 256.0):
        self._fs = fs
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record resting-state baseline for normalization.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array in uV.
            fs: Sampling rate. Falls back to constructor value.
            user_id: User identifier for multi-user support.

        Returns:
            Dict with baseline_set (bool) and baseline_metrics (dict).
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)

        bands = self._extract_all_bands(signals, fs)
        coh = self._mean_coherence(signals, fs)
        entropy_val = self._spectral_entropy(signals, fs)

        baseline = {
            "bands": bands,
            "coherence": coh,
            "entropy": entropy_val,
        }
        self._baselines[user_id] = baseline

        return {
            "baseline_set": True,
            "baseline_metrics": {
                "theta": round(bands["theta"], 6),
                "alpha": round(bands["alpha"], 6),
                "beta": round(bands["beta"], 6),
                "coherence": round(coh, 4),
                "entropy": round(entropy_val, 4),
            },
        }

    def screen(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Screen EEG for autism-associated biomarker patterns.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array in uV.
            fs: Sampling rate. Falls back to constructor value.
            user_id: User identifier.

        Returns:
            Dict with atypicality_score (0-100), risk_level, biomarker indices,
            individual biomarker scores, and medical disclaimer.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        has_baseline = user_id in self._baselines

        # --- Individual biomarkers ---
        tbr = self._compute_tbr(signals, fs)
        mu_suppression = self._compute_mu_suppression(signals, fs, user_id)
        coh_index = self._mean_coherence(signals, fs)
        asym_atyp = self._compute_asymmetry_atypicality(signals, fs)
        entropy_val = self._spectral_entropy(signals, fs)
        u_shape = self._compute_u_shape(signals, fs)

        # --- Biomarker sub-scores (all 0-1, higher = more atypical) ---
        # TBR: elevated TBR > ~3.5 at frontal sites is atypical in ASD
        tbr_score = float(np.clip((tbr - 1.5) / 4.0, 0.0, 1.0))

        # Mu suppression: 0 = no suppression = most atypical, 1 = full suppression = typical
        # Invert: low mu_suppression -> high atypicality
        mu_atyp_score = float(np.clip(1.0 - mu_suppression, 0.0, 1.0))

        # Coherence: low coherence = atypical
        coh_atyp_score = float(np.clip(1.0 - coh_index, 0.0, 1.0))

        # Asymmetry atypicality: already 0-1
        asym_score = float(np.clip(asym_atyp, 0.0, 1.0))

        # Entropy: higher = more atypical (relative to 0.7 baseline)
        entropy_score = float(np.clip((entropy_val - 0.5) / 0.4, 0.0, 1.0))

        # U-shape: higher = stronger U-shaped power profile
        u_shape_score = float(np.clip(u_shape, 0.0, 1.0))

        biomarkers = {
            "tbr_score": round(tbr_score, 4),
            "mu_atypicality_score": round(mu_atyp_score, 4),
            "coherence_atypicality_score": round(coh_atyp_score, 4),
            "asymmetry_score": round(asym_score, 4),
            "entropy_score": round(entropy_score, 4),
            "u_shape_score": round(u_shape_score, 4),
        }

        # --- Composite atypicality score (0-100) ---
        # Weighted sum of individual biomarker sub-scores
        composite = (
            0.20 * tbr_score
            + 0.25 * mu_atyp_score
            + 0.20 * coh_atyp_score
            + 0.15 * asym_score
            + 0.10 * entropy_score
            + 0.10 * u_shape_score
        )
        atypicality_score = float(np.clip(composite * 100.0, 0.0, 100.0))

        # --- Risk level ---
        if atypicality_score >= 70.0:
            risk_level = "significantly_atypical"
        elif atypicality_score >= 45.0:
            risk_level = "moderately_atypical"
        elif atypicality_score >= 25.0:
            risk_level = "mildly_atypical"
        else:
            risk_level = "typical"

        result = {
            "atypicality_score": round(atypicality_score, 2),
            "risk_level": risk_level,
            "mu_suppression_index": round(mu_suppression, 4),
            "theta_beta_ratio": round(tbr, 4),
            "coherence_index": round(coh_index, 4),
            "asymmetry_atypicality": round(asym_atyp, 4),
            "biomarkers": biomarkers,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
            "has_baseline": has_baseline,
        }

        # Store in history
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > _HISTORY_LIMIT:
            self._history[user_id] = self._history[user_id][-_HISTORY_LIMIT:]

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get aggregate session statistics for a user.

        Returns:
            Dict with n_epochs, mean_atypicality, and has_baseline.
        """
        history = self._history.get(user_id, [])
        has_baseline = user_id in self._baselines

        if not history:
            return {
                "n_epochs": 0,
                "mean_atypicality": 0.0,
                "has_baseline": has_baseline,
            }

        scores = [h["atypicality_score"] for h in history]
        return {
            "n_epochs": len(history),
            "mean_atypicality": round(float(np.mean(scores)), 2),
            "has_baseline": has_baseline,
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Get screening history for a user.

        Args:
            user_id: User identifier.
            last_n: Return only the last N entries. None = all.

        Returns:
            List of screening result dicts.
        """
        history = self._history.get(user_id, [])
        if last_n is not None and last_n > 0:
            return history[-last_n:]
        return list(history)

    def reset(self, user_id: str = "default") -> None:
        """Clear all state for a user (baseline + history)."""
        self._baselines.pop(user_id, None)
        self._history.pop(user_id, None)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_frontal(self, signals: np.ndarray) -> np.ndarray:
        """Average of AF7 (ch1) + AF8 (ch2), or first channel / 1D signal."""
        if signals.ndim == 2 and signals.shape[0] >= 3:
            return (signals[1] + signals[2]) / 2.0
        elif signals.ndim == 2:
            return signals[0]
        return signals

    def _get_temporal(self, signals: np.ndarray) -> np.ndarray:
        """Average of TP9 (ch0) + TP10 (ch3), or first channel / 1D signal."""
        if signals.ndim == 2 and signals.shape[0] >= 4:
            return (signals[0] + signals[3]) / 2.0
        elif signals.ndim == 2:
            return signals[0]
        return signals

    def _band_power(
        self, freqs: np.ndarray, psd: np.ndarray, low: float, high: float
    ) -> float:
        """Integrate PSD in a frequency band via trapezoidal rule."""
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 0.0
        return float(_trapezoid(psd[mask], freqs[mask]))

    def _welch_psd(self, signal: np.ndarray, fs: float):
        """Compute Welch PSD with safe nperseg."""
        nperseg = min(len(signal), int(fs * 2))
        if nperseg < 4:
            nperseg = len(signal)
        return welch(signal, fs=fs, nperseg=nperseg)

    def _extract_all_bands(self, signals: np.ndarray, fs: float) -> Dict[str, float]:
        """Extract band powers from frontal channels."""
        frontal = self._get_frontal(signals)
        freqs, psd = self._welch_psd(frontal, fs)

        total = _trapezoid(psd, freqs)
        if total <= 0:
            total = 1e-10

        result = {}
        for name, (lo, hi) in _BANDS.items():
            bp = self._band_power(freqs, psd, lo, hi)
            result[name] = float(bp / total)
        return result

    def _compute_tbr(self, signals: np.ndarray, fs: float) -> float:
        """Compute theta/beta ratio at frontal sites."""
        frontal = self._get_frontal(signals)
        freqs, psd = self._welch_psd(frontal, fs)
        theta = self._band_power(freqs, psd, 4.0, 8.0)
        beta = self._band_power(freqs, psd, 12.0, 30.0)
        return float(theta / (beta + 1e-10))

    def _compute_mu_suppression(
        self, signals: np.ndarray, fs: float, user_id: str
    ) -> float:
        """Compute mu suppression index at temporal sites.

        Mu rhythm (8-13 Hz) suppression at temporal sites (TP9/TP10) is a
        marker for mirror neuron activity. Reduced suppression relative to
        baseline is associated with ASD.

        Returns:
            0-1 where 0 = no suppression (atypical) and 1 = full suppression (typical).
            Without baseline, returns 0.5 (indeterminate).
        """
        temporal = self._get_temporal(signals)
        freqs, psd = self._welch_psd(temporal, fs)
        mu_power = self._band_power(freqs, psd, 8.0, 13.0)

        baseline = self._baselines.get(user_id)
        if baseline is None:
            # Without baseline, use a heuristic based on relative mu power
            total = _trapezoid(psd, freqs)
            if total <= 0:
                return 0.5
            mu_relative = mu_power / total
            # Typical mu relative: ~0.20-0.35; higher = less suppression
            return float(np.clip(1.0 - (mu_relative - 0.15) / 0.30, 0.0, 1.0))

        # With baseline: suppression = how much mu decreased from resting
        baseline_mu = baseline["bands"].get("mu", 0.0)
        if baseline_mu <= 1e-10:
            return 0.5

        # Normalize to total power for fair comparison
        total = _trapezoid(psd, freqs)
        mu_relative = mu_power / max(total, 1e-10)

        # Suppression ratio: lower live mu relative to baseline = more suppression
        suppression = float(np.clip(1.0 - mu_relative / baseline_mu, 0.0, 1.0))
        return suppression

    def _mean_coherence(self, signals: np.ndarray, fs: float) -> float:
        """Compute mean inter-channel coherence across alpha + theta bands.

        Low coherence = reduced functional connectivity = ASD-like pattern.

        Returns:
            0-1 coherence index.
        """
        if signals.ndim < 2 or signals.shape[0] < 2:
            return 0.5  # indeterminate for single channel

        n_channels = signals.shape[0]
        nperseg = min(signals.shape[1], int(fs * 2))
        if nperseg < 4:
            return 0.5

        coh_values = []
        bands_to_check = [("theta", 4.0, 8.0), ("alpha", 8.0, 12.0)]

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                try:
                    freqs, coh = scipy_coherence(
                        signals[i], signals[j], fs=fs, nperseg=nperseg
                    )
                    for _name, lo, hi in bands_to_check:
                        mask = (freqs >= lo) & (freqs <= hi)
                        if mask.any():
                            coh_values.append(float(np.mean(coh[mask])))
                except Exception:
                    continue

        if not coh_values:
            return 0.5
        return float(np.clip(np.mean(coh_values), 0.0, 1.0))

    def _compute_asymmetry_atypicality(
        self, signals: np.ndarray, fs: float
    ) -> float:
        """Compute frontal alpha asymmetry atypicality.

        Atypical asymmetry in ASD: reduced or reversed frontal asymmetry.
        Normal pattern: slightly higher right-frontal alpha (approach motivation).
        ASD pattern: minimal asymmetry or reversed.

        Returns:
            0-1 where 0 = typical asymmetry pattern and 1 = maximally atypical.
        """
        if signals.ndim < 2 or signals.shape[0] < 3:
            return 0.5  # indeterminate for single channel

        # AF7 = ch1, AF8 = ch2
        freqs_l, psd_l = self._welch_psd(signals[1], fs)
        freqs_r, psd_r = self._welch_psd(signals[2], fs)

        alpha_l = self._band_power(freqs_l, psd_l, 8.0, 12.0)
        alpha_r = self._band_power(freqs_r, psd_r, 8.0, 12.0)

        # FAA = ln(right_alpha) - ln(left_alpha)
        # Positive = normal approach motivation
        # Near zero or negative = atypical
        faa = float(np.log(alpha_r + 1e-10) - np.log(alpha_l + 1e-10))

        # Typical FAA is slightly positive (~0.05-0.15)
        # Atypicality increases as FAA drops below 0.05 or goes strongly negative
        # Also atypical if FAA is extremely high (> 0.5)
        if faa < 0.05:
            # Below typical range: more negative = more atypical
            atyp = float(np.clip((0.05 - faa) / 0.4, 0.0, 1.0))
        elif faa > 0.5:
            # Abnormally high asymmetry
            atyp = float(np.clip((faa - 0.5) / 0.5, 0.0, 0.5))
        else:
            atyp = 0.0

        return atyp

    def _spectral_entropy(self, signals: np.ndarray, fs: float) -> float:
        """Compute normalized spectral entropy (frontal channels).

        Higher entropy = more disorganized spectral profile.

        Returns:
            0-1 normalized spectral entropy.
        """
        frontal = self._get_frontal(signals)
        freqs, psd = self._welch_psd(frontal, fs)

        psd_norm = psd / (psd.sum() + 1e-10)
        # Shannon entropy
        psd_pos = psd_norm[psd_norm > 0]
        se = -float(np.sum(psd_pos * np.log(psd_pos)))

        # Normalize by max possible entropy
        max_entropy = np.log(len(psd_norm)) if len(psd_norm) > 0 else 1.0
        if max_entropy <= 0:
            return 0.0
        return float(np.clip(se / max_entropy, 0.0, 1.0))

    def _compute_u_shape(self, signals: np.ndarray, fs: float) -> float:
        """Compute U-shaped power profile index.

        ASD-associated pattern: excess delta + gamma relative to alpha + beta.
        The U-shape metric captures relative elevation of spectral extremes.

        Returns:
            0-1 where 0 = flat/normal profile and 1 = strong U-shape.
        """
        frontal = self._get_frontal(signals)
        freqs, psd = self._welch_psd(frontal, fs)

        delta = self._band_power(freqs, psd, 0.5, 4.0)
        theta = self._band_power(freqs, psd, 4.0, 8.0)
        alpha = self._band_power(freqs, psd, 8.0, 12.0)
        beta = self._band_power(freqs, psd, 12.0, 30.0)
        gamma = self._band_power(freqs, psd, 30.0, 100.0)

        # Extremes vs middle bands
        extremes = delta + gamma
        middle = alpha + beta + theta
        total = extremes + middle + 1e-10

        # U-shape index: ratio of extremes to total
        # Typical brain: alpha/beta dominate => low ratio
        # ASD pattern: delta + gamma elevated => higher ratio
        raw = extremes / total

        # Normalize: typical ratio ~0.15-0.30, atypical > 0.40
        u_score = float(np.clip((raw - 0.15) / 0.35, 0.0, 1.0))
        return u_score
