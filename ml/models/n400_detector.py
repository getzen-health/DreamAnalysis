"""N400 event-related potential (ERP) detection for semantic processing.

Detects the N400 component — a negative-going ERP peaking 300-500ms
after stimulus onset — as a marker of semantic processing difficulty.
Validated on Muse 2 at AF8 (Badolato et al. 2024, PMC11679099).

The N400 amplitude reflects semantic surprise:
- Large N400 → unexpected/incongruent stimulus (harder to process)
- Small N400 → expected/congruent stimulus (easy to process)

Applications: vocabulary learning, reading comprehension, language
assessment, semantic memory testing.

References:
    Badolato et al. (2024) — Muse 2 N400 validation at AF8
    Kutas & Hillyard (1980) — N400 discovery paper
"""
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal


class N400Detector:
    """Detect N400 ERP component from stimulus-locked EEG epochs.

    Designed for Muse 2 with AF8 (channel 2) as the primary detection
    site, validated by Badolato et al. (2024).
    """

    def __init__(
        self,
        baseline_window_ms: tuple = (-200, 0),
        n400_window_ms: tuple = (300, 500),
        threshold_uv: float = -2.0,
        af8_channel: int = 2,
    ):
        """Initialize N400 detector.

        Args:
            baseline_window_ms: Pre-stimulus baseline window in ms.
            n400_window_ms: N400 detection window in ms.
            threshold_uv: Amplitude threshold (negative) for N400 detection.
            af8_channel: Channel index for AF8 (default 2 for Muse 2).
        """
        self.baseline_window_ms = baseline_window_ms
        self.n400_window_ms = n400_window_ms
        self.threshold_uv = threshold_uv
        self.af8_channel = af8_channel
        self._history: Dict[str, List[Dict]] = {}

    def detect(
        self,
        epoch: np.ndarray,
        fs: float = 256,
        stimulus_onset_ms: float = 200,
        user_id: str = "default",
    ) -> Dict:
        """Detect N400 in a single stimulus-locked epoch.

        Args:
            epoch: EEG epoch array. Shapes accepted:
                - (n_samples,): single channel (assumed AF8)
                - (n_channels, n_samples): multichannel, uses af8_channel
            fs: Sampling rate in Hz.
            stimulus_onset_ms: Time of stimulus onset within epoch, in ms.
            user_id: User identifier for history tracking.

        Returns:
            Dict with n400_detected, n400_amplitude_uv, baseline_mean,
            n400_mean, peak_latency_ms, and semantic_difficulty fields.
        """
        # Extract AF8 channel
        if epoch.ndim == 2:
            if epoch.shape[0] > self.af8_channel:
                signal = epoch[self.af8_channel].astype(float)
            else:
                signal = epoch[0].astype(float)
        else:
            signal = epoch.astype(float)

        # Convert ms to samples
        onset_sample = int(stimulus_onset_ms * fs / 1000)

        # Baseline window
        bl_start = onset_sample + int(self.baseline_window_ms[0] * fs / 1000)
        bl_end = onset_sample + int(self.baseline_window_ms[1] * fs / 1000)
        bl_start = max(0, bl_start)
        bl_end = max(bl_start + 1, bl_end)

        # N400 window
        n4_start = onset_sample + int(self.n400_window_ms[0] * fs / 1000)
        n4_end = onset_sample + int(self.n400_window_ms[1] * fs / 1000)
        n4_start = min(n4_start, len(signal) - 1)
        n4_end = min(n4_end, len(signal))
        n4_end = max(n4_start + 1, n4_end)

        # Compute baseline-corrected N400
        baseline_mean = float(np.mean(signal[bl_start:bl_end]))
        corrected = signal - baseline_mean

        n400_region = corrected[n4_start:n4_end]
        n400_mean = float(np.mean(n400_region))

        # Peak (most negative) within N400 window
        peak_idx = int(np.argmin(n400_region))
        peak_amplitude = float(n400_region[peak_idx])
        peak_latency_ms = float((n4_start + peak_idx - onset_sample) * 1000 / fs)

        # Detection
        n400_detected = peak_amplitude < self.threshold_uv

        # Semantic difficulty score (0-1, higher = more difficulty)
        # Maps peak amplitude to a difficulty score via sigmoid
        difficulty = float(np.clip(1.0 / (1.0 + np.exp(peak_amplitude + 1)), 0, 1))

        result = {
            "n400_detected": bool(n400_detected),
            "n400_amplitude_uv": round(peak_amplitude, 3),
            "n400_mean_uv": round(n400_mean, 3),
            "baseline_mean_uv": round(baseline_mean, 3),
            "peak_latency_ms": round(peak_latency_ms, 1),
            "semantic_difficulty": round(difficulty, 4),
            "detection_channel": "AF8",
        }

        # Track history
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > 200:
            self._history[user_id] = self._history[user_id][-200:]

        return result

    def detect_average(
        self,
        epochs: List[np.ndarray],
        fs: float = 256,
        stimulus_onset_ms: float = 200,
    ) -> Dict:
        """Detect N400 from averaged epochs (more reliable).

        ERP averaging across trials reduces noise by sqrt(n_trials).
        Recommended: average 20-40 epochs for reliable N400 detection.

        Args:
            epochs: List of epoch arrays (all same shape).
            fs: Sampling rate.
            stimulus_onset_ms: Stimulus onset time within each epoch.

        Returns:
            Same as detect(), plus n_epochs and snr_improvement fields.
        """
        if not epochs:
            return {
                "n400_detected": False,
                "error": "No epochs provided",
                "n_epochs": 0,
            }

        # Average epochs
        stacked = np.stack(epochs, axis=0)
        averaged = np.mean(stacked, axis=0)

        result = self.detect(averaged, fs, stimulus_onset_ms)
        result["n_epochs"] = len(epochs)
        result["snr_improvement_db"] = round(10 * np.log10(len(epochs)), 1)
        return result

    def get_history(self, user_id: str = "default") -> List[Dict]:
        """Get N400 detection history for a user."""
        return list(self._history.get(user_id, []))

    def get_summary(self, user_id: str = "default") -> Dict:
        """Get summary statistics from detection history."""
        history = self._history.get(user_id, [])
        if not history:
            return {"n_trials": 0, "detection_rate": 0, "mean_amplitude": 0}

        amplitudes = [h["n400_amplitude_uv"] for h in history]
        detected = [h["n400_detected"] for h in history]

        return {
            "n_trials": len(history),
            "detection_rate": round(sum(detected) / len(detected), 3),
            "mean_amplitude_uv": round(float(np.mean(amplitudes)), 3),
            "std_amplitude_uv": round(float(np.std(amplitudes)), 3),
            "mean_latency_ms": round(
                float(np.mean([h["peak_latency_ms"] for h in history])), 1
            ),
            "mean_difficulty": round(
                float(np.mean([h["semantic_difficulty"] for h in history])), 4
            ),
        }

    def reset(self, user_id: str = "default"):
        """Clear detection history for a user."""
        self._history.pop(user_id, None)
