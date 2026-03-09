"""Pre-ictal seizure prediction from 4-channel consumer EEG.

Detects pre-ictal EEG changes that can precede seizures by 10-60 minutes.
Uses sliding window trend analysis on validated pre-ictal biomarkers:

1. Spectral entropy decrease — loss of signal complexity before seizure
2. Increased synchronization — cross-channel correlation rises in pre-ictal state
3. Theta/alpha ratio shift — theta power increases relative to alpha
4. High-frequency oscillation increase — subtle beta/low-gamma elevation
5. Decreased complexity — Lyapunov exponent proxy via signal predictability

IMPORTANT DISCLAIMER: This is NOT a medical device. It is a research/educational
tool and must NEVER be used as a clinical seizure predictor. Pre-ictal prediction
from 4-channel consumer EEG has NOT been clinically validated. Users with epilepsy
must rely on clinician-supervised monitoring systems. False negatives WILL occur.

References:
    Mormann et al. (2007) — Seizure prediction: the long and winding road.
    Gadhoumi et al. (2016) — Seizure prediction for therapeutic devices.
    Iasemidis et al. (2003) — Lyapunov exponent convergence in pre-ictal state.
    Lehnertz & Elger (1998) — Spectral entropy decline before seizures.
"""
from typing import Dict, List, Optional

import numpy as np
_trapezoid = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
from scipy import signal as scipy_signal

MEDICAL_DISCLAIMER = (
    "NOT a medical device. Research/educational use only. "
    "Pre-ictal prediction from consumer EEG is NOT clinically validated. "
    "Users with epilepsy must consult their neurologist and use "
    "clinician-supervised monitoring. False negatives WILL occur."
)

# Risk thresholds
_RISK_NORMAL = 0.3
_RISK_ELEVATED = 0.5
_RISK_WARNING = 0.7

# Maximum history entries retained
_MAX_HISTORY = 2000

# Minimum samples required (0.5 seconds at 256 Hz)
_MIN_SAMPLES_FACTOR = 0.5


class PreictalPredictor:
    """Pre-ictal state predictor from 4-channel EEG.

    Tracks EEG feature trends over time to detect the gradual changes
    that precede seizures (pre-ictal state). Designed for Muse 2 channels:
    TP9 (ch0), AF7 (ch1), AF8 (ch2), TP10 (ch3).

    Unlike SeizureDetector (which detects ongoing seizures), this class
    looks for the slow buildup that can precede a seizure by 10-60 minutes.
    """

    def __init__(
        self,
        fs: float = 256.0,
        trend_window: int = 10,
        alert_threshold: float = 0.7,
        sustained_count: int = 3,
    ):
        """Initialize the pre-ictal predictor.

        Args:
            fs: Sampling rate in Hz.
            trend_window: Number of past assessments to use for trend analysis.
            alert_threshold: Risk level (0-1) that triggers an alert.
            sustained_count: Number of consecutive above-threshold assessments
                before an alert fires.
        """
        self._fs = float(fs)
        self._trend_window = max(2, int(trend_window))
        self._alert_threshold = float(np.clip(alert_threshold, 0.1, 0.99))
        self._sustained_count = max(1, int(sustained_count))

        # Baseline state
        self._baseline: Optional[Dict] = None

        # History tracking
        self._history: List[Dict] = []
        self._consecutive_elevated: int = 0

    def set_baseline(
        self,
        eeg: np.ndarray,
        fs: Optional[float] = None,
    ) -> Dict:
        """Record resting-state baseline for comparison.

        Should be called during a known normal/interictal period.
        Minimum 2 seconds of data recommended.

        Args:
            eeg: (n_channels, n_samples) or (n_samples,) EEG array in uV.
            fs: Sampling rate override.

        Returns:
            Dict with baseline status and feature summary.
        """
        fs = fs or self._fs
        eeg = np.asarray(eeg, dtype=float)
        if eeg.ndim == 1:
            eeg = eeg.reshape(1, -1)

        n_channels, n_samples = eeg.shape
        min_samples = int(fs * _MIN_SAMPLES_FACTOR)
        if n_samples < min_samples:
            return {
                "baseline_set": False,
                "reason": f"Need at least {min_samples} samples ({_MIN_SAMPLES_FACTOR}s)",
                "disclaimer": MEDICAL_DISCLAIMER,
            }

        features = self._extract_features(eeg, fs)
        self._baseline = features
        return {
            "baseline_set": True,
            "n_channels": n_channels,
            "n_samples": n_samples,
            "features": {
                "spectral_entropy": round(features["spectral_entropy"], 4),
                "mean_synchrony": round(features["mean_synchrony"], 4),
                "theta_alpha_ratio": round(features["theta_alpha_ratio"], 4),
                "hfo_power": round(features["hfo_power"], 4),
                "complexity": round(features["complexity"], 4),
            },
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    def assess(
        self,
        eeg: np.ndarray,
        fs: Optional[float] = None,
    ) -> Dict:
        """Assess current pre-ictal risk from an EEG epoch.

        Args:
            eeg: (n_channels, n_samples) or (n_samples,) EEG array in uV.
            fs: Sampling rate override.

        Returns:
            Dict with preictal_risk, risk_level, feature_changes,
            entropy_trend, synchrony_index, alert, alert_message, disclaimer.
        """
        fs = fs or self._fs
        eeg = np.asarray(eeg, dtype=float)
        if eeg.ndim == 1:
            eeg = eeg.reshape(1, -1)

        n_samples = eeg.shape[1]
        min_samples = int(fs * _MIN_SAMPLES_FACTOR)
        if n_samples < min_samples:
            return self._insufficient_data_result()

        features = self._extract_features(eeg, fs)
        feature_changes = self._compute_feature_changes(features)
        risk = self._compute_risk(feature_changes, features)
        risk_level = self._classify_risk(risk)

        # Track trend
        entropy_trend = self._compute_entropy_trend(features["spectral_entropy"])
        synchrony_index = features["mean_synchrony"]

        # Alert logic — sustained elevated risk
        if risk >= self._alert_threshold:
            self._consecutive_elevated += 1
        else:
            self._consecutive_elevated = max(0, self._consecutive_elevated - 1)

        alert = self._consecutive_elevated >= self._sustained_count
        alert_message = None
        if alert:
            alert_message = (
                f"Pre-ictal risk has been {risk_level} for "
                f"{self._consecutive_elevated} consecutive assessments. "
                "This is a RESEARCH indicator only. Consult your neurologist."
            )

        result = {
            "preictal_risk": round(float(risk), 4),
            "risk_level": risk_level,
            "entropy_trend": round(float(entropy_trend), 4),
            "synchrony_index": round(float(synchrony_index), 4),
            "feature_changes": {
                k: round(float(v), 4) for k, v in feature_changes.items()
            },
            "alert": alert,
            "alert_message": alert_message,
            "has_baseline": self._baseline is not None,
            "consecutive_elevated": self._consecutive_elevated,
            "disclaimer": MEDICAL_DISCLAIMER,
        }

        # Append to history
        self._history.append(result)
        if len(self._history) > _MAX_HISTORY:
            self._history = self._history[-_MAX_HISTORY:]

        return result

    def get_risk_timeline(self) -> List[Dict]:
        """Get the full timeline of risk assessments.

        Returns:
            List of dicts with preictal_risk and risk_level per assessment.
        """
        return [
            {
                "preictal_risk": entry["preictal_risk"],
                "risk_level": entry["risk_level"],
                "entropy_trend": entry["entropy_trend"],
                "synchrony_index": entry["synchrony_index"],
            }
            for entry in self._history
        ]

    def get_session_stats(self) -> Dict:
        """Get summary statistics for the current monitoring session.

        Returns:
            Dict with session summary: total assessments, peak risk,
            mean risk, alert count, and time in each risk level.
        """
        if not self._history:
            return {
                "total_assessments": 0,
                "peak_risk": 0.0,
                "mean_risk": 0.0,
                "alerts_fired": 0,
                "time_in_normal": 0,
                "time_in_elevated": 0,
                "time_in_warning": 0,
                "time_in_critical": 0,
                "has_baseline": self._baseline is not None,
                "disclaimer": MEDICAL_DISCLAIMER,
            }

        risks = [entry["preictal_risk"] for entry in self._history]
        levels = [entry["risk_level"] for entry in self._history]
        alerts = sum(1 for entry in self._history if entry["alert"])

        return {
            "total_assessments": len(self._history),
            "peak_risk": round(float(max(risks)), 4),
            "mean_risk": round(float(np.mean(risks)), 4),
            "alerts_fired": alerts,
            "time_in_normal": sum(1 for lv in levels if lv == "normal"),
            "time_in_elevated": sum(1 for lv in levels if lv == "elevated"),
            "time_in_warning": sum(1 for lv in levels if lv == "warning"),
            "time_in_critical": sum(1 for lv in levels if lv == "critical"),
            "has_baseline": self._baseline is not None,
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    def get_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """Get raw assessment history.

        Args:
            last_n: If provided, return only the last N entries.

        Returns:
            List of full assessment result dicts.
        """
        if last_n is not None and last_n > 0:
            return list(self._history[-last_n:])
        return list(self._history)

    def reset(self):
        """Clear all state: history, baseline, and consecutive counters."""
        self._baseline = None
        self._history = []
        self._consecutive_elevated = 0

    # ── Private: Feature Extraction ──────────────────────────────────

    def _extract_features(self, eeg: np.ndarray, fs: float) -> Dict:
        """Extract pre-ictal biomarker features from multichannel EEG.

        Returns dict with: spectral_entropy, mean_synchrony, theta_alpha_ratio,
        hfo_power, complexity, and per-channel band powers.
        """
        n_channels = eeg.shape[0]

        # Per-channel features
        entropies = []
        theta_powers = []
        alpha_powers = []
        hfo_powers = []
        complexities = []

        for ch in range(n_channels):
            sig = eeg[ch]

            powers = self._band_powers(sig, fs)
            theta_powers.append(powers["theta"])
            alpha_powers.append(powers["alpha"])
            hfo_powers.append(powers["high_beta"])

            ent = self._spectral_entropy(sig, fs)
            entropies.append(ent)

            cplx = self._signal_complexity(sig)
            complexities.append(cplx)

        # Cross-channel synchronization
        mean_sync = self._cross_channel_sync(eeg)

        # Aggregate
        total_alpha = float(np.mean(alpha_powers))
        total_theta = float(np.mean(theta_powers))
        theta_alpha = total_theta / (total_alpha + 1e-10)

        return {
            "spectral_entropy": float(np.mean(entropies)),
            "mean_synchrony": float(mean_sync),
            "theta_alpha_ratio": float(theta_alpha),
            "hfo_power": float(np.mean(hfo_powers)),
            "complexity": float(np.mean(complexities)),
            "channel_entropies": [float(e) for e in entropies],
            "channel_theta": [float(t) for t in theta_powers],
            "channel_alpha": [float(a) for a in alpha_powers],
        }

    def _band_powers(self, signal: np.ndarray, fs: float) -> Dict[str, float]:
        """Extract band powers via Welch PSD."""
        nperseg = min(len(signal), int(fs * 2))
        if nperseg < 4:
            return {
                "delta": 0.0, "theta": 0.0, "alpha": 0.0,
                "beta": 0.0, "high_beta": 0.0,
            }

        try:
            freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)
        except Exception:
            return {
                "delta": 0.0, "theta": 0.0, "alpha": 0.0,
                "beta": 0.0, "high_beta": 0.0,
            }

        def _bp(low: float, high: float) -> float:
            mask = (freqs >= low) & (freqs <= high)
            if not np.any(mask):
                return 0.0
            return float(_trapezoid(psd[mask], freqs[mask]))

        return {
            "delta": _bp(0.5, 4),
            "theta": _bp(4, 8),
            "alpha": _bp(8, 12),
            "beta": _bp(12, 30),
            "high_beta": _bp(20, 30),
        }

    def _spectral_entropy(self, signal: np.ndarray, fs: float) -> float:
        """Compute normalized spectral entropy (0 = ordered, 1 = uniform).

        Pre-ictal state: entropy DECREASES as the brain becomes more
        rhythmic/ordered before a seizure (Lehnertz & Elger, 1998).
        """
        nperseg = min(len(signal), int(fs * 2))
        if nperseg < 4:
            return 1.0

        try:
            freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)
        except Exception:
            return 1.0

        # Limit to 1-45 Hz (physiological range)
        mask = (freqs >= 1) & (freqs <= 45)
        psd_band = psd[mask]

        if len(psd_band) == 0 or np.sum(psd_band) < 1e-20:
            return 1.0

        # Normalize to probability distribution
        psd_norm = psd_band / np.sum(psd_band)
        psd_norm = psd_norm[psd_norm > 0]

        if len(psd_norm) <= 1:
            return 0.0

        # Shannon entropy, normalized to [0, 1]
        entropy = -np.sum(psd_norm * np.log2(psd_norm))
        max_entropy = np.log2(len(psd_norm))
        if max_entropy < 1e-10:
            return 1.0

        return float(np.clip(entropy / max_entropy, 0, 1))

    def _signal_complexity(self, signal: np.ndarray) -> float:
        """Estimate signal complexity as a Lyapunov exponent proxy.

        Uses sample entropy approximation: higher values indicate more
        complex/unpredictable signals. Pre-ictal state shows DECREASED
        complexity (Iasemidis et al., 2003).

        Returns value in [0, 1] range (normalized).
        """
        n = len(signal)
        if n < 20:
            return 0.5

        # Hjorth complexity: ratio of mobility of first derivative
        # to mobility of the signal. Fast proxy for Lyapunov exponent.
        diff1 = np.diff(signal)
        diff2 = np.diff(diff1)

        var_signal = np.var(signal)
        var_diff1 = np.var(diff1)
        var_diff2 = np.var(diff2)

        if var_signal < 1e-20 or var_diff1 < 1e-20:
            return 0.0

        mobility_signal = np.sqrt(var_diff1 / var_signal)
        mobility_diff1 = np.sqrt(var_diff2 / var_diff1)

        hjorth_complexity = mobility_diff1 / (mobility_signal + 1e-10)

        # Normalize: typical EEG Hjorth complexity is 1.0-3.0
        # Map to 0-1 range
        return float(np.clip(hjorth_complexity / 3.0, 0, 1))

    def _cross_channel_sync(self, signals: np.ndarray) -> float:
        """Compute mean pairwise correlation across channels.

        Pre-ictal state: synchronization INCREASES as cortical networks
        become more coupled before seizure onset.
        """
        n_channels = signals.shape[0]
        if n_channels < 2:
            return 0.5

        correlations = []
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                min_len = min(len(signals[i]), len(signals[j]))
                if min_len < 10:
                    continue
                s1 = signals[i, :min_len]
                s2 = signals[j, :min_len]
                std1, std2 = np.std(s1), np.std(s2)
                if std1 < 1e-10 or std2 < 1e-10:
                    correlations.append(0.0)
                    continue
                r = float(np.corrcoef(s1, s2)[0, 1])
                correlations.append(abs(r))

        return float(np.mean(correlations)) if correlations else 0.5

    # ── Private: Feature Change Analysis ─────────────────────────────

    def _compute_feature_changes(self, current: Dict) -> Dict:
        """Compute how current features deviate from baseline.

        Returns dict of signed change magnitudes for each feature.
        Positive values indicate change in the pre-ictal direction:
        - entropy_change < 0 means entropy decreased (pre-ictal sign)
        - synchrony_change > 0 means sync increased (pre-ictal sign)
        - theta_alpha_change > 0 means ratio increased (pre-ictal sign)
        - hfo_change > 0 means HFO power increased (pre-ictal sign)
        - complexity_change < 0 means complexity decreased (pre-ictal sign)
        """
        if self._baseline is None:
            return {
                "entropy_change": 0.0,
                "synchrony_change": 0.0,
                "theta_alpha_change": 0.0,
                "hfo_change": 0.0,
                "complexity_change": 0.0,
            }

        bl = self._baseline

        def _relative_change(current_val: float, baseline_val: float) -> float:
            if abs(baseline_val) < 1e-10:
                return 0.0
            return (current_val - baseline_val) / (abs(baseline_val) + 1e-10)

        return {
            "entropy_change": _relative_change(
                current["spectral_entropy"], bl["spectral_entropy"]
            ),
            "synchrony_change": _relative_change(
                current["mean_synchrony"], bl["mean_synchrony"]
            ),
            "theta_alpha_change": _relative_change(
                current["theta_alpha_ratio"], bl["theta_alpha_ratio"]
            ),
            "hfo_change": _relative_change(
                current["hfo_power"], bl["hfo_power"]
            ),
            "complexity_change": _relative_change(
                current["complexity"], bl["complexity"]
            ),
        }

    def _compute_risk(self, changes: Dict, features: Dict) -> float:
        """Compute overall pre-ictal risk from feature changes and absolutes.

        Combines five biomarkers with literature-informed weights:
        1. Spectral entropy decrease (25%) — Lehnertz & Elger, 1998
        2. Synchronization increase (25%) — Mormann et al., 2007
        3. Theta/alpha ratio increase (20%) — pre-ictal slowing
        4. HFO power increase (15%) — Gadhoumi et al., 2016
        5. Complexity decrease (15%) — Iasemidis et al., 2003
        """
        risk_components = []

        # 1. Entropy decrease: negative change = pre-ictal sign
        entropy_risk = float(np.clip(-changes["entropy_change"] * 2.0, 0, 1))
        risk_components.append(0.25 * entropy_risk)

        # 2. Synchrony increase: positive change = pre-ictal sign
        sync_risk = float(np.clip(changes["synchrony_change"] * 2.0, 0, 1))
        # Also use absolute synchrony — very high sync is suspicious
        abs_sync_risk = float(np.clip(
            (features["mean_synchrony"] - 0.7) / 0.3, 0, 1
        ))
        sync_combined = max(sync_risk, abs_sync_risk)
        risk_components.append(0.25 * sync_combined)

        # 3. Theta/alpha ratio increase: positive change = pre-ictal slowing
        ta_risk = float(np.clip(changes["theta_alpha_change"] * 1.5, 0, 1))
        risk_components.append(0.20 * ta_risk)

        # 4. HFO power increase
        hfo_risk = float(np.clip(changes["hfo_change"] * 1.5, 0, 1))
        risk_components.append(0.15 * hfo_risk)

        # 5. Complexity decrease: negative change = pre-ictal sign
        cplx_risk = float(np.clip(-changes["complexity_change"] * 2.0, 0, 1))
        risk_components.append(0.15 * cplx_risk)

        total_risk = float(np.sum(risk_components))

        # Without baseline, risk is attenuated (less confident)
        if self._baseline is None:
            # Use absolute feature values as weak signals
            # Low entropy + high sync is suspicious even without baseline
            abs_risk = 0.0
            if features["spectral_entropy"] < 0.4:
                abs_risk += 0.15
            if features["mean_synchrony"] > 0.85:
                abs_risk += 0.15
            total_risk = float(np.clip(abs_risk, 0, 1))

        return float(np.clip(total_risk, 0, 1))

    def _classify_risk(self, risk: float) -> str:
        """Classify risk value into named level."""
        if risk >= _RISK_WARNING:
            return "critical"
        elif risk >= _RISK_ELEVATED:
            return "warning"
        elif risk >= _RISK_NORMAL:
            return "elevated"
        else:
            return "normal"

    def _compute_entropy_trend(self, current_entropy: float) -> float:
        """Compute entropy trend from recent history.

        Returns slope of entropy over the trend window.
        Negative slope = entropy decreasing = pre-ictal sign.
        """
        # Collect recent entropy values
        recent_entropies = [current_entropy]
        window = min(self._trend_window, len(self._history))
        for i in range(window):
            idx = -(i + 1)
            if abs(idx) <= len(self._history):
                # Recover entropy from feature_changes + baseline
                # or approximate from the risk components
                entry = self._history[idx]
                # Use entropy_trend if available in history, else 0
                recent_entropies.append(entry.get("entropy_trend", 0.0))

        if len(recent_entropies) < 2:
            return 0.0

        # Simple linear trend: slope of entropy over assessments
        x = np.arange(len(recent_entropies), dtype=float)
        y = np.array(recent_entropies, dtype=float)

        # Least-squares slope
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        denom = np.sum((x - x_mean) ** 2)
        if denom < 1e-10:
            return 0.0

        slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
        return slope

    def _insufficient_data_result(self) -> Dict:
        """Return result for epochs that are too short to analyze."""
        return {
            "preictal_risk": 0.0,
            "risk_level": "normal",
            "entropy_trend": 0.0,
            "synchrony_index": 0.0,
            "feature_changes": {
                "entropy_change": 0.0,
                "synchrony_change": 0.0,
                "theta_alpha_change": 0.0,
                "hfo_change": 0.0,
                "complexity_change": 0.0,
            },
            "alert": False,
            "alert_message": None,
            "has_baseline": self._baseline is not None,
            "consecutive_elevated": self._consecutive_elevated,
            "disclaimer": MEDICAL_DISCLAIMER,
        }
