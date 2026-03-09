"""Mild Cognitive Impairment (MCI) / early Alzheimer's screening from EEG.

Uses 4-channel Muse 2 EEG (TP9, AF7, AF8, TP10 at 256 Hz) to compute
biomarkers associated with cognitive decline:

  - Alpha peak frequency slowing (below 8 Hz suggests MCI)
  - Increased theta/alpha ratio (theta dominance in MCI)
  - Reduced alpha power overall
  - Decreased spectral entropy (less complex signals)
  - Reduced inter-channel coherence (network disconnection)
  - Increased delta power (cortical slowing)
  - 1/f aperiodic slope changes (steeper in neurodegeneration)

Scientific basis:
  - Babiloni et al. (2021): Alpha rhythm slowing is the most replicated MCI EEG marker
  - Cassani et al. (2018): Theta/alpha ratio discriminates MCI from healthy controls
  - Meghdadi et al. (2021): Consumer-grade EEG can detect spectral slowing
  - Donoghue et al. (2020): Aperiodic 1/f slope steepens with neurodegeneration
  - Koenig et al. (2005): Coherence reduction reflects disconnection syndrome in AD

MEDICAL DISCLAIMER: This is a research-grade screening tool only.
It is NOT a medical diagnosis. Never use this as a substitute for
professional neurological evaluation. Consult a healthcare professional
for any cognitive health concerns.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch, coherence as scipy_coherence
from typing import Dict, List, Optional

# NumPy 2.0 renamed np.trapz -> np.trapezoid; 1.x only has np.trapz
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

MEDICAL_DISCLAIMER = (
    "This is a research-grade cognitive screening tool only. "
    "It is NOT a medical diagnosis and should never replace professional "
    "neurological evaluation. Consult a healthcare professional for any "
    "cognitive health concerns."
)

RISK_LEVELS = ["low", "mild", "moderate", "elevated"]

# Frequency band definitions (Hz)
_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
}

# History cap per user
_MAX_HISTORY = 500


class MCIScreener:
    """MCI/early Alzheimer's screening from 4-channel Muse 2 EEG.

    Computes a composite risk score (0-100) from multiple EEG biomarkers
    known to change in mild cognitive impairment: alpha peak frequency,
    theta/alpha ratio, spectral entropy, inter-channel coherence,
    delta dominance, and 1/f aperiodic slope.

    Multi-user support: pass user_id to maintain separate baselines and
    histories per user.
    """

    def __init__(self, fs: float = 256.0):
        self._fs = fs
        # Per-user state: baselines and screening history
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        age: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record a resting-state EEG baseline for a user.

        Should be collected during 2-3 minutes of relaxed, eyes-closed rest.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array in uV.
            fs: Sampling rate override (uses instance default if None).
            age: User's chronological age in years (optional, adjusts norms).
            user_id: Identifier for multi-user support.

        Returns:
            Dict with baseline_set (bool) and baseline_metrics (dict).
        """
        fs = fs or self._fs
        metrics = self._compute_biomarkers(signals, fs)
        self._baselines[user_id] = {
            "metrics": metrics,
            "age": age,
        }
        return {
            "baseline_set": True,
            "baseline_metrics": metrics,
        }

    # ------------------------------------------------------------------
    # Screen
    # ------------------------------------------------------------------

    def screen(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Screen EEG epoch for MCI biomarkers.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array in uV.
            fs: Sampling rate override.
            user_id: Identifier for multi-user support.

        Returns:
            Dict with risk_score, risk_level, individual biomarkers,
            has_baseline flag, and medical_disclaimer.
        """
        fs = fs or self._fs
        metrics = self._compute_biomarkers(signals, fs)
        baseline = self._baselines.get(user_id)
        has_baseline = baseline is not None
        age = baseline["age"] if baseline else None

        # Compute individual biomarker risk scores (each 0-100)
        biomarkers = self._score_biomarkers(metrics, baseline, age)

        # Composite risk score: weighted combination of biomarker scores
        risk_score = self._compute_composite_risk(biomarkers)

        # Classify risk level
        risk_level = self._classify_risk(risk_score)

        # Store in history
        record = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "alpha_peak_freq": metrics["alpha_peak_freq"],
            "theta_alpha_ratio": metrics["theta_alpha_ratio"],
            "spectral_entropy": metrics["spectral_entropy"],
            "coherence_index": metrics["coherence_index"],
            "delta_ratio": metrics["delta_ratio"],
            "aperiodic_slope": metrics["aperiodic_slope"],
            "biomarkers": biomarkers,
            "has_baseline": has_baseline,
        }
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(record)
        if len(self._history[user_id]) > _MAX_HISTORY:
            self._history[user_id] = self._history[user_id][-_MAX_HISTORY:]

        return {
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level,
            "alpha_peak_freq": round(metrics["alpha_peak_freq"], 2),
            "theta_alpha_ratio": round(metrics["theta_alpha_ratio"], 3),
            "spectral_entropy": round(metrics["spectral_entropy"], 3),
            "coherence_index": round(metrics["coherence_index"], 3),
            "delta_ratio": round(metrics["delta_ratio"], 3),
            "aperiodic_slope": round(metrics["aperiodic_slope"], 3),
            "biomarkers": {k: round(v, 1) for k, v in biomarkers.items()},
            "has_baseline": has_baseline,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        }

    # ------------------------------------------------------------------
    # Session stats / history
    # ------------------------------------------------------------------

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Return summary statistics for a user's screening session."""
        history = self._history.get(user_id, [])
        has_baseline = user_id in self._baselines
        if not history:
            return {
                "n_epochs": 0,
                "mean_risk": 0.0,
                "has_baseline": has_baseline,
            }
        scores = [h["risk_score"] for h in history]
        return {
            "n_epochs": len(history),
            "mean_risk": round(float(np.mean(scores)), 1),
            "has_baseline": has_baseline,
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Return screening history for a user.

        Args:
            user_id: User identifier.
            last_n: If set, return only the last N records.

        Returns:
            List of screening result dicts (oldest first).
        """
        history = self._history.get(user_id, [])
        if last_n is not None and last_n > 0:
            return history[-last_n:]
        return list(history)

    def reset(self, user_id: str = "default") -> None:
        """Clear baseline and history for a user."""
        self._baselines.pop(user_id, None)
        self._history.pop(user_id, None)

    # ------------------------------------------------------------------
    # Internal: biomarker extraction
    # ------------------------------------------------------------------

    def _compute_biomarkers(self, signals: np.ndarray, fs: float) -> Dict:
        """Extract all MCI-relevant biomarkers from EEG signals."""
        # Normalize to (n_channels, n_samples)
        if signals.ndim == 1:
            channels = signals.reshape(1, -1)
        else:
            channels = signals

        n_channels, n_samples = channels.shape

        # Per-channel metrics, then average
        alpha_peaks = []
        theta_alpha_ratios = []
        entropies = []
        delta_ratios = []
        aperiodic_slopes = []
        alpha_powers = []

        for ch_idx in range(n_channels):
            ch = channels[ch_idx].astype(np.float64)
            nperseg = min(len(ch), int(fs * 4))
            if nperseg < 8:
                nperseg = len(ch)
            freqs, psd = welch(ch, fs=fs, nperseg=nperseg)

            # Alpha peak frequency
            alpha_peaks.append(self._alpha_peak_freq(freqs, psd))

            # Band powers
            bands = self._band_powers(freqs, psd)
            theta_alpha_ratios.append(
                bands["theta"] / (bands["alpha"] + 1e-10)
            )
            delta_ratios.append(
                bands["delta"]
                / (bands["delta"] + bands["theta"] + bands["alpha"] + bands["beta"] + 1e-10)
            )
            alpha_powers.append(bands["alpha"])

            # Spectral entropy
            entropies.append(self._spectral_entropy(psd))

            # Aperiodic (1/f) slope
            aperiodic_slopes.append(self._aperiodic_slope(freqs, psd))

        # Inter-channel coherence (needs >= 2 channels)
        coherence_index = self._coherence_index(channels, fs)

        return {
            "alpha_peak_freq": float(np.mean(alpha_peaks)),
            "theta_alpha_ratio": float(np.mean(theta_alpha_ratios)),
            "spectral_entropy": float(np.mean(entropies)),
            "coherence_index": coherence_index,
            "delta_ratio": float(np.mean(delta_ratios)),
            "aperiodic_slope": float(np.mean(aperiodic_slopes)),
            "alpha_power": float(np.mean(alpha_powers)),
        }

    def _alpha_peak_freq(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """Find peak frequency within the alpha band (6-14 Hz extended range).

        Uses an extended search range (6-14 Hz) because MCI subjects often
        have alpha peaks that have slowed below the canonical 8 Hz boundary.
        """
        mask = (freqs >= 6.0) & (freqs <= 14.0)
        if not mask.any():
            return 10.0  # default healthy alpha peak
        alpha_psd = psd[mask]
        alpha_freqs = freqs[mask]
        peak_idx = np.argmax(alpha_psd)
        return float(alpha_freqs[peak_idx])

    def _band_powers(self, freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
        """Compute relative band powers from pre-computed PSD."""
        total = _trapezoid(psd, freqs)
        if total <= 0:
            total = 1e-10
        powers = {}
        for name, (low, high) in _BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            if mask.any():
                powers[name] = float(_trapezoid(psd[mask], freqs[mask]) / total)
            else:
                powers[name] = 0.0
        return powers

    def _spectral_entropy(self, psd: np.ndarray) -> float:
        """Normalized spectral entropy (0-1). Lower = less complex = MCI marker."""
        psd_norm = psd / (psd.sum() + 1e-10)
        psd_norm = psd_norm[psd_norm > 0]
        if len(psd_norm) == 0:
            return 0.0
        se = -float(np.sum(psd_norm * np.log(psd_norm)))
        max_se = np.log(len(psd_norm)) if len(psd_norm) > 1 else 1.0
        return float(np.clip(se / max_se, 0.0, 1.0))

    def _aperiodic_slope(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """Estimate 1/f aperiodic exponent via log-log linear regression.

        Returns the negative slope (exponent). Healthy adults: ~1.5-2.5.
        MCI/AD: steeper (higher exponent, >2.5).
        """
        mask = (freqs >= 2.0) & (freqs <= 40.0)
        if mask.sum() < 5:
            return 2.0  # default
        log_f = np.log10(freqs[mask])
        log_p = np.log10(psd[mask] + 1e-30)
        coeffs = np.polyfit(log_f, log_p, 1)
        return float(-coeffs[0])

    def _coherence_index(self, channels: np.ndarray, fs: float) -> float:
        """Mean alpha-band coherence across all channel pairs.

        Reduced coherence indicates cortical disconnection, a hallmark of AD.
        Returns 0-1 (higher = more connected = healthier).
        """
        n_ch = channels.shape[0]
        if n_ch < 2:
            return 1.0  # single channel, no disconnection measurable

        alpha_low, alpha_high = _BANDS["alpha"]
        coh_values = []

        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                nperseg = min(channels.shape[1], int(fs * 2))
                if nperseg < 8:
                    nperseg = channels.shape[1]
                freqs, coh = scipy_coherence(
                    channels[i], channels[j], fs=fs, nperseg=nperseg
                )
                mask = (freqs >= alpha_low) & (freqs <= alpha_high)
                if mask.any():
                    coh_values.append(float(np.mean(coh[mask])))

        if not coh_values:
            return 0.0
        return float(np.clip(np.mean(coh_values), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Internal: biomarker scoring
    # ------------------------------------------------------------------

    def _score_biomarkers(
        self,
        metrics: Dict,
        baseline: Optional[Dict],
        age: Optional[float],
    ) -> Dict[str, float]:
        """Convert raw biomarker values into individual risk scores (0-100).

        Each biomarker is scored independently using population norms from
        the MCI/AD literature. If a baseline is available, scores are
        adjusted relative to the user's personal baseline.
        """
        base_metrics = baseline["metrics"] if baseline else None

        # 1. Alpha Peak Frequency (APF)
        # Healthy: 9.5-11 Hz. MCI: often <8 Hz. AD: <7 Hz.
        apf = metrics["alpha_peak_freq"]
        if base_metrics:
            # Score relative to personal baseline shift
            apf_shift = base_metrics["alpha_peak_freq"] - apf
            apf_score = float(np.clip(apf_shift / 3.0 * 100, 0, 100))
        else:
            # Population norm scoring
            if apf >= 10.0:
                apf_score = 0.0
            elif apf >= 8.0:
                # Linear ramp: 10 Hz -> 0, 8 Hz -> 50
                apf_score = (10.0 - apf) / 2.0 * 50
            else:
                # Below 8 Hz: 50 + ramp to 100 at 6 Hz
                apf_score = 50.0 + (8.0 - apf) / 2.0 * 50
            apf_score = float(np.clip(apf_score, 0, 100))

        # Age adjustment: older adults naturally have lower APF
        if age is not None and age > 60:
            age_adj = min((age - 60) * 0.5, 10.0)  # up to 10 points reduction
            apf_score = max(0.0, apf_score - age_adj)

        # 2. Theta/Alpha Ratio (TAR)
        # Healthy: <1.0. MCI: 1.0-2.0. AD: >2.0.
        tar = metrics["theta_alpha_ratio"]
        if base_metrics:
            tar_shift = tar - base_metrics["theta_alpha_ratio"]
            tar_score = float(np.clip(tar_shift / 1.5 * 100, 0, 100))
        else:
            if tar <= 0.8:
                tar_score = 0.0
            elif tar <= 1.5:
                tar_score = (tar - 0.8) / 0.7 * 50
            else:
                tar_score = 50.0 + min((tar - 1.5) / 1.0 * 50, 50.0)
            tar_score = float(np.clip(tar_score, 0, 100))

        # 3. Spectral Entropy (SE)
        # Healthy: >0.7. MCI: 0.5-0.7. Lower = less complex.
        se = metrics["spectral_entropy"]
        if base_metrics:
            se_shift = base_metrics["spectral_entropy"] - se
            se_score = float(np.clip(se_shift / 0.3 * 100, 0, 100))
        else:
            if se >= 0.75:
                se_score = 0.0
            elif se >= 0.5:
                se_score = (0.75 - se) / 0.25 * 50
            else:
                se_score = 50.0 + (0.5 - se) / 0.25 * 50
            se_score = float(np.clip(se_score, 0, 100))

        # 4. Coherence Index (CI)
        # Healthy: >0.5. MCI/AD: <0.3 (disconnection).
        ci = metrics["coherence_index"]
        if base_metrics:
            ci_shift = base_metrics["coherence_index"] - ci
            ci_score = float(np.clip(ci_shift / 0.3 * 100, 0, 100))
        else:
            if ci >= 0.5:
                ci_score = 0.0
            elif ci >= 0.3:
                ci_score = (0.5 - ci) / 0.2 * 50
            else:
                ci_score = 50.0 + (0.3 - ci) / 0.3 * 50
            ci_score = float(np.clip(ci_score, 0, 100))

        # 5. Delta Ratio (DR)
        # Healthy: <0.3. MCI: 0.3-0.5. AD: >0.5.
        dr = metrics["delta_ratio"]
        if base_metrics:
            dr_shift = dr - base_metrics["delta_ratio"]
            dr_score = float(np.clip(dr_shift / 0.3 * 100, 0, 100))
        else:
            if dr <= 0.25:
                dr_score = 0.0
            elif dr <= 0.45:
                dr_score = (dr - 0.25) / 0.20 * 50
            else:
                dr_score = 50.0 + min((dr - 0.45) / 0.20 * 50, 50.0)
            dr_score = float(np.clip(dr_score, 0, 100))

        # 6. Aperiodic Slope (AS)
        # Healthy: 1.5-2.5. MCI/AD: steeper (>2.5).
        aslope = metrics["aperiodic_slope"]
        if base_metrics:
            as_shift = aslope - base_metrics["aperiodic_slope"]
            as_score = float(np.clip(as_shift / 1.0 * 100, 0, 100))
        else:
            if aslope <= 2.0:
                as_score = 0.0
            elif aslope <= 2.8:
                as_score = (aslope - 2.0) / 0.8 * 50
            else:
                as_score = 50.0 + min((aslope - 2.8) / 0.7 * 50, 50.0)
            as_score = float(np.clip(as_score, 0, 100))

        return {
            "alpha_peak_freq": apf_score,
            "theta_alpha_ratio": tar_score,
            "spectral_entropy": se_score,
            "coherence_index": ci_score,
            "delta_ratio": dr_score,
            "aperiodic_slope": as_score,
        }

    def _compute_composite_risk(self, biomarkers: Dict[str, float]) -> float:
        """Weighted composite risk score (0-100).

        Weights reflect discriminative power from MCI literature:
        - Alpha peak freq: strongest single MCI marker (Babiloni 2021)
        - Theta/alpha ratio: second strongest (Cassani 2018)
        - Coherence: structural disconnection (Koenig 2005)
        - Spectral entropy, delta, aperiodic: supporting markers
        """
        weights = {
            "alpha_peak_freq": 0.25,
            "theta_alpha_ratio": 0.20,
            "coherence_index": 0.18,
            "spectral_entropy": 0.15,
            "delta_ratio": 0.12,
            "aperiodic_slope": 0.10,
        }
        score = sum(biomarkers[k] * weights[k] for k in weights)
        return float(np.clip(score, 0, 100))

    def _classify_risk(self, score: float) -> str:
        """Map composite score to risk level."""
        if score < 25:
            return "low"
        elif score < 50:
            return "mild"
        elif score < 75:
            return "moderate"
        else:
            return "elevated"


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_instance: Optional[MCIScreener] = None


def get_mci_screener() -> MCIScreener:
    """Return module-level MCIScreener singleton."""
    global _instance
    if _instance is None:
        _instance = MCIScreener()
    return _instance
