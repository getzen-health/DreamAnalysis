"""EEG-based deception detection via Concealed Information Test (CIT).

Detects concealed knowledge by comparing P300 ERP amplitude between
probe (known) and irrelevant (unknown) stimuli. The P300 is larger
for recognized items even when the subject tries to conceal knowledge.

Feature-based approach using validated biomarkers:
1. P300 amplitude difference (probe > irrelevant at 250-500ms)
2. Late positive potential (LPP) sustained positivity
3. Frontal theta increase during deception
4. Beta suppression during concealment effort

DISCLAIMER: Research tool only. Not validated for forensic or legal use.
EEG-based deception detection has inherent limitations and should never
be used as sole evidence.

References:
    2024 EEGNet CIT study — 86.67% accuracy
    Rosenfeld et al. (2008) — P300-based CIT review
"""
from typing import Dict, List, Optional

import numpy as np

DISCLAIMER = (
    "Research tool only. Not validated for forensic or legal use. "
    "EEG-based deception detection should never be used as sole evidence."
)


class DeceptionDetector:
    """EEG-based deception detection via P300 Concealed Information Test.

    Compares ERP responses to probe stimuli (items the subject knows)
    versus irrelevant stimuli (items unknown to the subject).
    """

    def __init__(self, fs: float = 256.0, p300_window: tuple = (250, 500)):
        self._fs = fs
        self._p300_window = p300_window  # ms post-stimulus
        self._history: Dict[str, List[Dict]] = {}

    def detect(
        self,
        probe_epoch: np.ndarray,
        irrelevant_epoch: np.ndarray,
        fs: Optional[float] = None,
        channel_idx: int = 2,
        user_id: str = "default",
    ) -> Dict:
        """Compare single probe vs irrelevant epoch for deception.

        Args:
            probe_epoch: (n_channels, n_samples) or (n_samples,) EEG epoch
                time-locked to probe stimulus.
            irrelevant_epoch: Same format, time-locked to irrelevant stimulus.
            fs: Sampling rate.
            channel_idx: Channel for P300 (default 2 = AF8).
            user_id: User identifier.

        Returns:
            Dict with deception_score, deception_detected, p300_difference,
            probe_p300_amplitude, irrelevant_p300_amplitude, confidence.
        """
        fs = fs or self._fs

        probe_signal = self._extract_channel(probe_epoch, channel_idx)
        irrel_signal = self._extract_channel(irrelevant_epoch, channel_idx)

        # Extract P300 amplitude
        probe_p300 = self._extract_p300(probe_signal, fs)
        irrel_p300 = self._extract_p300(irrel_signal, fs)

        # P300 difference (probe should be larger if concealing)
        p300_diff = probe_p300 - irrel_p300

        # Deception score via sigmoid
        deception_score = float(1.0 / (1.0 + np.exp(-p300_diff * 0.5)))

        # Additional features
        probe_theta = self._theta_power(probe_signal, fs)
        irrel_theta = self._theta_power(irrel_signal, fs)
        theta_increase = probe_theta - irrel_theta

        # Combine P300 + theta features
        combined_score = float(np.clip(
            0.70 * deception_score + 0.30 * min(max(theta_increase * 5 + 0.5, 0), 1),
            0, 1
        ))

        # Confidence based on P300 SNR
        snr = abs(p300_diff) / max(np.std(irrel_signal), 1e-10)
        if snr > 3.0:
            confidence = "high"
        elif snr > 1.5:
            confidence = "medium"
        else:
            confidence = "low"

        result = {
            "deception_score": round(combined_score, 4),
            "deception_detected": combined_score > 0.6,
            "p300_difference": round(float(p300_diff), 4),
            "probe_p300_amplitude": round(float(probe_p300), 4),
            "irrelevant_p300_amplitude": round(float(irrel_p300), 4),
            "theta_increase": round(float(theta_increase), 6),
            "confidence": confidence,
            "disclaimer": DISCLAIMER,
        }

        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > 500:
            self._history[user_id] = self._history[user_id][-500:]

        return result

    def detect_average(
        self,
        probe_epochs: List[np.ndarray],
        irrelevant_epochs: List[np.ndarray],
        fs: Optional[float] = None,
        channel_idx: int = 2,
        user_id: str = "default",
    ) -> Dict:
        """Average multiple trials for robust detection (recommended).

        ERP averaging improves SNR by sqrt(n_trials).
        Minimum 10 trials of each type recommended.

        Args:
            probe_epochs: List of probe epochs.
            irrelevant_epochs: List of irrelevant epochs.
            fs: Sampling rate.
            channel_idx: Channel index.
            user_id: User identifier.

        Returns:
            Same as detect() plus n_probe_trials, n_irrelevant_trials, snr_improvement.
        """
        fs = fs or self._fs

        if not probe_epochs or not irrelevant_epochs:
            return {
                "deception_score": 0.5,
                "deception_detected": False,
                "p300_difference": 0.0,
                "confidence": "low",
                "n_probe_trials": 0,
                "n_irrelevant_trials": 0,
                "disclaimer": DISCLAIMER,
            }

        # Average ERPs
        probe_signals = [self._extract_channel(e, channel_idx) for e in probe_epochs]
        irrel_signals = [self._extract_channel(e, channel_idx) for e in irrelevant_epochs]

        min_len_p = min(len(s) for s in probe_signals)
        min_len_i = min(len(s) for s in irrel_signals)

        avg_probe = np.mean([s[:min_len_p] for s in probe_signals], axis=0)
        avg_irrel = np.mean([s[:min_len_i] for s in irrel_signals], axis=0)

        # Run detection on averaged signals
        result = self.detect(avg_probe, avg_irrel, fs, 0, user_id)

        n_trials = min(len(probe_epochs), len(irrelevant_epochs))
        snr_improvement = float(np.sqrt(n_trials))

        result["n_probe_trials"] = len(probe_epochs)
        result["n_irrelevant_trials"] = len(irrelevant_epochs)
        result["snr_improvement"] = round(snr_improvement, 2)

        return result

    def get_history(self, user_id: str = "default", last_n: Optional[int] = None) -> List[Dict]:
        """Get detection history."""
        history = self._history.get(user_id, [])
        if last_n:
            history = history[-last_n:]
        return history

    def get_summary(self, user_id: str = "default") -> Dict:
        """Get session summary statistics."""
        history = self._history.get(user_id, [])
        if not history:
            return {"n_trials": 0}

        scores = [h["deception_score"] for h in history]
        detections = sum(1 for h in history if h["deception_detected"])

        return {
            "n_trials": len(history),
            "mean_deception_score": round(float(np.mean(scores)), 4),
            "detection_rate": round(detections / len(history), 4),
            "detections": detections,
            "high_confidence_detections": sum(
                1 for h in history if h["deception_detected"] and h["confidence"] == "high"
            ),
        }

    def reset(self, user_id: str = "default"):
        """Clear history."""
        self._history.pop(user_id, None)

    # ── Private helpers ──────────────────────────────────────────

    def _extract_channel(self, epoch: np.ndarray, channel_idx: int) -> np.ndarray:
        """Extract single channel from epoch."""
        epoch = np.asarray(epoch, dtype=float)
        if epoch.ndim == 2:
            idx = min(channel_idx, epoch.shape[0] - 1)
            return epoch[idx]
        return epoch

    def _extract_p300(self, signal: np.ndarray, fs: float) -> float:
        """Extract mean amplitude in P300 window."""
        start_sample = int(self._p300_window[0] / 1000 * fs)
        end_sample = int(self._p300_window[1] / 1000 * fs)

        if end_sample > len(signal):
            end_sample = len(signal)
        if start_sample >= end_sample:
            return 0.0

        return float(np.mean(signal[start_sample:end_sample]))

    def _theta_power(self, signal: np.ndarray, fs: float) -> float:
        """Compute theta band power (4-8 Hz)."""
        if len(signal) < int(fs * 0.25):
            return 0.0

        from scipy.signal import welch
        nperseg = min(len(signal), int(fs * 1))
        try:
            freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        except Exception:
            return 0.0

        mask = (freqs >= 4) & (freqs <= 8)
        if not np.any(mask):
            return 0.0
        return float(np.trapezoid(psd[mask], freqs[mask]) if hasattr(np, 'trapezoid')
                     else np.trapz(psd[mask], freqs[mask]))
