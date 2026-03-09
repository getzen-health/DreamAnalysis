"""Closed-loop neurostimulation guidance from EEG.

Monitors 4-channel Muse 2 EEG and recommends optimal stimulation parameters.
Does NOT control any stimulation device -- advisory only.

Scientific basis:
- tACS at individual alpha frequency (IAF) enhances alpha power
  (Zaehle et al. 2010, Neuhaus et al. 2017)
- tDCS over left DLPFC (AF7 area) improves mood via positive FAA shift
  (Brunoni et al. 2012, meta-analysis of 6+ RCTs)
- Frontal theta indicates executive control engagement -- theta-burst
  protocols (iTBS) target this (Huang et al. 2005)
- Phase-locked stimulation is most effective -- alpha phase detection
  for optimal timing (Zrenner et al. 2018)

Safety:
- Every recommendation includes a disclaimer requiring clinical supervision
- Contraindication checks: seizure-like patterns (>200 uV), alpha surplus
- This class is purely advisory and never actuates hardware

Muse 2 channel order (BrainFlow board_id 38):
  ch0 = TP9  (left temporal)
  ch1 = AF7  (left frontal)
  ch2 = AF8  (right frontal)
  ch3 = TP10 (right temporal)
"""

import numpy as np
from typing import Dict, List, Optional
from scipy.signal import welch

# np.trapezoid (NumPy 2.0+) with np.trapz fallback
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

_DISCLAIMER = (
    "Advisory only -- neurostimulation requires clinical supervision. "
    "Do not use these parameters without a qualified clinician's approval."
)

_VALID_PROTOCOLS = {"tacs_alpha", "tdcs_left_dlpfc", "tacs_theta", "no_stimulation"}
_VALID_TARGETS = {"focus", "relaxation", "mood", "default"}

_HISTORY_CAP = 500

# Amplitude threshold for seizure-like activity (uV)
_SEIZURE_THRESHOLD_UV = 200.0

# Default band definitions (Hz)
_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
}


class NeurostimGuide:
    """Monitors EEG and recommends neurostimulation parameters.

    Multi-user: each user_id maintains independent baseline, history,
    and session stats.
    """

    def __init__(self, fs: float = 256.0):
        self.fs = fs
        # Per-user state: {user_id: {"baseline": {...}, "history": [...]}}
        self._users: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_user(self, user_id: str) -> dict:
        """Return or create per-user state dict."""
        if user_id not in self._users:
            self._users[user_id] = {
                "baseline": None,  # {iaf, baseline_alpha, baseline_theta}
                "history": [],
            }
        return self._users[user_id]

    @staticmethod
    def _to_single_channel(signals: np.ndarray) -> np.ndarray:
        """Extract a representative single channel from possibly multichannel input.

        For multichannel (2D), averages AF7 (ch1) and AF8 (ch2) if available,
        otherwise uses first channel. For 1D, returns as-is.
        """
        if signals.ndim == 1:
            return signals
        if signals.shape[0] >= 3:
            # Average AF7 + AF8 for frontal representation
            return (signals[1] + signals[2]) / 2.0
        return signals[0]

    @staticmethod
    def _estimate_iaf(signal: np.ndarray, fs: float) -> float:
        """Estimate Individual Alpha Frequency from a 1D signal."""
        nperseg = min(int(fs * 2), len(signal))
        if nperseg < int(fs * 0.5):
            return 10.0
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        mask = (freqs >= 7.0) & (freqs <= 14.0)
        if not np.any(mask):
            return 10.0
        alpha_psd = psd[mask]
        alpha_freqs = freqs[mask]
        return float(alpha_freqs[np.argmax(alpha_psd)])

    @staticmethod
    def _band_power(signal: np.ndarray, fs: float, low: float, high: float) -> float:
        """Compute relative band power in [low, high] Hz."""
        nperseg = min(int(fs * 2), len(signal))
        if nperseg < 4:
            return 0.0
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        total = _trapezoid(psd, freqs)
        if total <= 0:
            return 0.0
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 0.0
        return float(_trapezoid(psd[mask], freqs[mask]) / total)

    @staticmethod
    def _max_amplitude(signals: np.ndarray) -> float:
        """Return the maximum absolute amplitude across all channels (uV)."""
        return float(np.max(np.abs(signals)))

    def _check_contraindications(self, signals: np.ndarray, alpha_power: float,
                                  baseline: Optional[dict]) -> dict:
        """Run safety checks on the current EEG epoch."""
        reasons: List[str] = []

        # Seizure-like: very high amplitude
        max_amp = self._max_amplitude(signals)
        if max_amp > _SEIZURE_THRESHOLD_UV:
            reasons.append(
                f"High amplitude detected ({max_amp:.0f} uV > {_SEIZURE_THRESHOLD_UV:.0f} uV threshold) "
                "-- possible seizure-like activity"
            )

        # Alpha already very high -- no need for enhancement
        if baseline is not None:
            bl_alpha = baseline.get("baseline_alpha", 0.0)
            if bl_alpha > 0 and alpha_power > bl_alpha * 1.5:
                reasons.append(
                    "Alpha power already significantly above baseline -- "
                    "enhancement not recommended"
                )

        return {"safe": len(reasons) == 0, "reasons": reasons}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_baseline(self, signals: np.ndarray, fs: Optional[float] = None,
                     user_id: str = "default") -> dict:
        """Record resting-state baseline for a user.

        Args:
            signals: 1D or 2D (4, n_samples) EEG array.
            fs: Sampling rate override.
            user_id: User identifier for multi-user support.

        Returns:
            Dict with baseline_set, iaf, baseline_alpha, baseline_theta.
        """
        use_fs = fs if fs is not None else self.fs
        ch = self._to_single_channel(signals)
        iaf = self._estimate_iaf(ch, use_fs)
        alpha_power = self._band_power(ch, use_fs, *_BANDS["alpha"])
        theta_power = self._band_power(ch, use_fs, *_BANDS["theta"])

        user = self._get_user(user_id)
        user["baseline"] = {
            "iaf": iaf,
            "baseline_alpha": alpha_power,
            "baseline_theta": theta_power,
        }

        return {
            "baseline_set": True,
            "iaf": iaf,
            "baseline_alpha": alpha_power,
            "baseline_theta": theta_power,
        }

    def recommend(self, signals: np.ndarray, fs: Optional[float] = None,
                  target_state: str = "focus",
                  user_id: str = "default") -> dict:
        """Analyze EEG and recommend stimulation parameters.

        Args:
            signals: 1D or 2D (4, n_samples) EEG array.
            fs: Sampling rate override.
            target_state: One of 'focus', 'relaxation', 'mood', 'default'.
            user_id: User identifier.

        Returns:
            Dict with protocol, target_frequency, power measurements,
            deficit flags, readiness_score, contraindication_check,
            disclaimer, has_baseline.
        """
        use_fs = fs if fs is not None else self.fs
        user = self._get_user(user_id)
        baseline = user["baseline"]
        has_baseline = baseline is not None

        ch = self._to_single_channel(signals)
        alpha_power = self._band_power(ch, use_fs, *_BANDS["alpha"])
        theta_power = self._band_power(ch, use_fs, *_BANDS["theta"])
        beta_power = self._band_power(ch, use_fs, *_BANDS["beta"])

        # IAF: use baseline if available, else estimate live
        iaf = baseline["iaf"] if has_baseline else self._estimate_iaf(ch, use_fs)

        # Deficit/excess detection
        if has_baseline:
            bl_alpha = baseline["baseline_alpha"]
            bl_theta = baseline["baseline_theta"]
            alpha_deficit = alpha_power < bl_alpha * 0.8 if bl_alpha > 0 else False
            theta_excess = theta_power > bl_theta * 1.2 if bl_theta > 0 else False
        else:
            # Without baseline, use population-average heuristics
            alpha_deficit = alpha_power < 0.15
            theta_excess = theta_power > 0.35

        # Safety checks
        contra = self._check_contraindications(signals, alpha_power, baseline)

        # Protocol selection
        if not contra["safe"]:
            protocol = "no_stimulation"
        else:
            protocol = self._select_protocol(
                target_state, alpha_deficit, theta_excess,
                alpha_power, theta_power, beta_power, baseline
            )

        # Readiness score: higher when conditions are well-characterised
        readiness = self._compute_readiness(
            has_baseline, contra["safe"], alpha_power, theta_power, beta_power
        )

        # Target frequency
        target_frequency = self._target_frequency(protocol, iaf)

        result = {
            "protocol": protocol,
            "target_frequency": target_frequency,
            "current_alpha_power": alpha_power,
            "current_theta_power": theta_power,
            "alpha_deficit": alpha_deficit,
            "theta_excess": theta_excess,
            "readiness_score": readiness,
            "contraindication_check": contra,
            "disclaimer": _DISCLAIMER,
            "has_baseline": has_baseline,
        }

        # Append to history (capped)
        history = user["history"]
        history.append(result)
        if len(history) > _HISTORY_CAP:
            user["history"] = history[-_HISTORY_CAP:]

        return result

    def get_session_stats(self, user_id: str = "default") -> dict:
        """Return aggregate stats for a user's session.

        Returns:
            Dict with n_epochs, recommended_protocols distribution,
            mean_readiness.
        """
        user = self._get_user(user_id)
        history = user["history"]

        if not history:
            return {
                "n_epochs": 0,
                "recommended_protocols": {},
                "mean_readiness": 0.0,
            }

        protocol_counts: Dict[str, int] = {}
        total_readiness = 0.0
        for entry in history:
            p = entry["protocol"]
            protocol_counts[p] = protocol_counts.get(p, 0) + 1
            total_readiness += entry["readiness_score"]

        return {
            "n_epochs": len(history),
            "recommended_protocols": protocol_counts,
            "mean_readiness": total_readiness / len(history),
        }

    def get_history(self, user_id: str = "default",
                    last_n: Optional[int] = None) -> list:
        """Return recommendation history for a user.

        Args:
            user_id: User identifier.
            last_n: If set, return only the last N entries.
        """
        user = self._get_user(user_id)
        history = user["history"]
        if last_n is not None and last_n < len(history):
            return list(history[-last_n:])
        return list(history)

    def reset(self, user_id: str = "default"):
        """Clear baseline and history for a user."""
        if user_id in self._users:
            self._users[user_id] = {
                "baseline": None,
                "history": [],
            }

    # ------------------------------------------------------------------
    # Private decision logic
    # ------------------------------------------------------------------

    def _select_protocol(self, target_state: str, alpha_deficit: bool,
                         theta_excess: bool, alpha_power: float,
                         theta_power: float, beta_power: float,
                         baseline: Optional[dict]) -> str:
        """Choose the best stimulation protocol for the target state."""

        if target_state == "relaxation":
            if alpha_deficit:
                return "tacs_alpha"
            # Alpha already adequate -- no stimulation needed
            return "no_stimulation"

        if target_state == "mood":
            return "tdcs_left_dlpfc"

        if target_state == "focus":
            if theta_excess:
                # Excessive drowsiness/mind-wandering -- theta burst to reset
                return "tacs_theta"
            # Boost executive function via left DLPFC
            return "tdcs_left_dlpfc"

        # "default" -- adaptive: pick based on current state
        if alpha_deficit and theta_excess:
            return "tacs_alpha"  # relaxation takes priority when both present
        if alpha_deficit:
            return "tacs_alpha"
        if theta_excess:
            return "tacs_theta"
        return "no_stimulation"

    @staticmethod
    def _target_frequency(protocol: str, iaf: float) -> float:
        """Return the target stimulation frequency for the chosen protocol."""
        if protocol == "tacs_alpha":
            return iaf
        if protocol == "tacs_theta":
            # Theta-burst: typically 5 Hz (within theta band)
            return 5.0
        # tDCS and no_stimulation: DC or N/A
        return 0.0

    @staticmethod
    def _compute_readiness(has_baseline: bool, is_safe: bool,
                           alpha: float, theta: float, beta: float) -> int:
        """Compute a 0-100 readiness score for stimulation suitability.

        Factors:
        - Baseline available (+30 points)
        - Safe contraindication check (+30 points)
        - Signal quality heuristic: band powers are non-zero and reasonable (+40 points)
        """
        score = 0

        # Baseline bonus
        if has_baseline:
            score += 30

        # Safety bonus
        if is_safe:
            score += 30

        # Signal quality: all bands non-zero and within plausible ranges
        bands_ok = all(0.001 < p < 0.95 for p in [alpha, theta, beta])
        if bands_ok:
            score += 40
        elif any(p > 0.001 for p in [alpha, theta, beta]):
            # Partial credit -- some signal present
            score += 20

        return min(score, 100)
