"""Slow Oscillation Detector — detection and SO-spindle coupling analysis.

Detects slow oscillations (SO, 0.5-1.25 Hz) in sleep EEG signals, computes
SO density, amplitude, and frequency, and measures SO-spindle coupling
strength to predict memory consolidation quality.

During NREM sleep (N2/N3), slow oscillations orchestrate the reactivation of
hippocampal memory traces. Thalamocortical spindles (11-16 Hz) preferentially
nest in the SO up-state, and this temporal coupling predicts overnight memory
retention (Staresina et al. 2015, Helfrich et al. 2018).

For Muse 2 (AF7/AF8 frontal, TP9/TP10 temporal), frontal channels capture SOs
most prominently. AF7 (ch1) is used by default.

Dependencies: numpy, scipy only.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SO_LOW = 0.5       # Hz — lower bound of SO band
_SO_HIGH = 1.25     # Hz — upper bound of SO band
_SPINDLE_LOW = 11.0  # Hz — spindle band lower bound
_SPINDLE_HIGH = 16.0 # Hz — spindle band upper bound

_DEFAULT_AMPLITUDE_THRESHOLD_UV = 30.0  # uV — minimum peak-to-trough for SO
_COUPLING_THRESHOLD = 0.15  # coupling_strength above this = coupling_detected

# Consolidation prediction thresholds (based on coupling_strength + SO density)
_CONSOLIDATION_THRESHOLDS = {
    "excellent": 0.65,
    "good": 0.40,
    "moderate": 0.15,
}


# ---------------------------------------------------------------------------
# SlowOscillationDetector
# ---------------------------------------------------------------------------

class SlowOscillationDetector:
    """Detect slow oscillations and measure SO-spindle coupling for memory
    consolidation prediction.

    Methods:
        detect(eeg_signals, fs) -> dict with SO metrics and coupling info
        get_coupling_score() -> float 0-1
        get_session_stats() -> aggregate stats across all epochs
        get_history() -> list of per-epoch results
        reset() -> clear session state
    """

    def __init__(
        self,
        amplitude_threshold_uv: float = _DEFAULT_AMPLITUDE_THRESHOLD_UV,
    ):
        """Initialize SlowOscillationDetector.

        Args:
            amplitude_threshold_uv: Minimum peak-to-trough amplitude (uV)
                for a candidate cycle to count as a slow oscillation.
                Lower = more sensitive; higher = fewer false positives.
                Default 30 uV is appropriate for Muse 2 dry electrodes.
        """
        self._amplitude_threshold = amplitude_threshold_uv
        self._history: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, eeg_signals: np.ndarray, fs: float = 256.0
    ) -> Dict:
        """Detect slow oscillations and compute SO-spindle coupling.

        Args:
            eeg_signals: 1D (n_samples,) or 2D (n_channels, n_samples) array.
                If 2D, uses channel 1 (AF7) for detection.
            fs: Sampling frequency in Hz.

        Returns:
            Dict with:
                so_count (int): number of SOs detected
                so_density (float): SOs per minute
                mean_amplitude (float): mean peak-to-trough amplitude (uV)
                mean_frequency (float): mean SO frequency (Hz)
                coupling_detected (bool): whether SO-spindle coupling found
                coupling_strength (float): 0-1 coupling index
                consolidation_prediction (str): poor/moderate/good/excellent
                so_events (list): list of individual SO event dicts
        """
        eeg_signals = np.asarray(eeg_signals, dtype=float)
        signal = self._to_1d(eeg_signals)

        duration_sec = len(signal) / fs
        duration_min = duration_sec / 60.0

        # Need at least 2 seconds for meaningful SO detection
        min_samples = int(fs * 2.0)
        if len(signal) < min_samples:
            result = self._empty_result()
            self._history.append(result)
            return result

        # Bandpass filter for SO band (0.5-1.25 Hz)
        so_filtered = self._bandpass(signal, _SO_LOW, _SO_HIGH, fs, order=2)
        if so_filtered is None:
            result = self._empty_result()
            self._history.append(result)
            return result

        # Detect individual SO events
        so_events = self._detect_so_events(so_filtered, fs)
        so_count = len(so_events)

        # Compute aggregate metrics
        if so_count > 0:
            so_density = so_count / max(duration_min, 1e-6)
            mean_amplitude = float(np.mean([e["amplitude"] for e in so_events]))
            mean_frequency = float(np.mean([e["frequency"] for e in so_events]))
        else:
            so_density = 0.0
            mean_amplitude = 0.0
            mean_frequency = 0.0

        # Compute SO-spindle coupling
        coupling_strength = self._compute_coupling(signal, so_filtered, fs)
        coupling_detected = coupling_strength > _COUPLING_THRESHOLD

        # Consolidation prediction
        consolidation_prediction = self._predict_consolidation(
            so_count, so_density, coupling_strength, mean_amplitude
        )

        result = {
            "so_count": so_count,
            "so_density": round(so_density, 2),
            "mean_amplitude": round(mean_amplitude, 4),
            "mean_frequency": round(mean_frequency, 4),
            "coupling_detected": coupling_detected,
            "coupling_strength": round(coupling_strength, 4),
            "consolidation_prediction": consolidation_prediction,
            "so_events": so_events,
        }

        self._history.append(result)
        return result

    def get_coupling_score(self) -> float:
        """Return the latest SO-spindle coupling strength (0-1).

        If no epochs analyzed, returns 0.0.
        """
        if not self._history:
            return 0.0
        return float(self._history[-1]["coupling_strength"])

    def get_session_stats(self) -> Dict:
        """Aggregate statistics across all analyzed epochs in this session.

        Returns:
            Dict with n_epochs, total_so_count, mean_density,
            mean_amplitude, mean_coupling_strength.
        """
        if not self._history:
            return {
                "n_epochs": 0,
                "total_so_count": 0,
                "mean_density": 0.0,
                "mean_amplitude": 0.0,
                "mean_coupling_strength": 0.0,
            }

        total_so = sum(h["so_count"] for h in self._history)
        densities = [h["so_density"] for h in self._history]
        amplitudes = [
            h["mean_amplitude"] for h in self._history if h["mean_amplitude"] > 0
        ]
        couplings = [h["coupling_strength"] for h in self._history]

        return {
            "n_epochs": len(self._history),
            "total_so_count": total_so,
            "mean_density": round(float(np.mean(densities)), 2),
            "mean_amplitude": round(
                float(np.mean(amplitudes)) if amplitudes else 0.0, 4
            ),
            "mean_coupling_strength": round(float(np.mean(couplings)), 4),
        }

    def get_history(self) -> List[Dict]:
        """Return list of all per-epoch analysis results."""
        return list(self._history)

    def reset(self) -> None:
        """Clear all session history."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> Dict:
        """Return a zeroed-out result dict for edge cases."""
        return {
            "so_count": 0,
            "so_density": 0.0,
            "mean_amplitude": 0.0,
            "mean_frequency": 0.0,
            "coupling_detected": False,
            "coupling_strength": 0.0,
            "consolidation_prediction": "poor",
            "so_events": [],
        }

    @staticmethod
    def _to_1d(signal: np.ndarray) -> np.ndarray:
        """Extract single channel from multichannel input.

        For Muse 2: ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10.
        Prefers AF7 (ch1) for frontal SO detection.
        """
        signal = np.asarray(signal, dtype=float)
        if signal.ndim == 2:
            if signal.shape[0] >= 2:
                return signal[1]  # AF7
            return signal[0]
        return signal

    @staticmethod
    def _bandpass(
        signal: np.ndarray,
        low: float,
        high: float,
        fs: float,
        order: int = 2,
    ) -> Optional[np.ndarray]:
        """Apply Butterworth bandpass filter. Returns None if signal too short."""
        nyq = fs / 2.0
        lo = max(low / nyq, 0.001)
        hi = min(high / nyq, 0.999)
        if lo >= hi:
            return None
        try:
            b, a = butter(order, [lo, hi], btype="band")
            padlen = 3 * max(len(a), len(b)) - 1
            if len(signal) <= padlen:
                return None
            return filtfilt(b, a, signal)
        except (ValueError, np.linalg.LinAlgError):
            return None

    def _detect_so_events(
        self, so_filtered: np.ndarray, fs: float
    ) -> List[Dict]:
        """Detect individual slow oscillation events by finding negative-to-positive
        zero crossings and measuring peak-to-trough amplitude.

        An SO is defined as a full cycle (negative half-wave followed by positive
        half-wave) with peak-to-trough amplitude exceeding the threshold.

        Returns:
            List of dicts with start_sample, amplitude, frequency.
        """
        events: List[Dict] = []

        # Find negative-to-positive zero crossings (up-going)
        # These mark the start of the SO up-state
        sign = np.sign(so_filtered)
        # Handle exact zeros: treat as positive
        sign[sign == 0] = 1
        diff_sign = np.diff(sign)
        # Negative-to-positive crossing: diff_sign == 2 (from -1 to +1)
        up_crossings = np.where(diff_sign > 0)[0]

        if len(up_crossings) < 2:
            return events

        # Each SO cycle spans from one up-crossing to the next
        for i in range(len(up_crossings) - 1):
            start = up_crossings[i]
            end = up_crossings[i + 1]
            cycle_len = end - start

            # Duration check: SO cycle should be 0.8-2.0 seconds (0.5-1.25 Hz)
            cycle_duration = cycle_len / fs
            if cycle_duration < 0.7 or cycle_duration > 2.5:
                continue

            segment = so_filtered[start:end]
            peak = float(np.max(segment))
            trough = float(np.min(segment))
            amplitude = peak - trough

            if amplitude < self._amplitude_threshold:
                continue

            frequency = 1.0 / cycle_duration

            events.append({
                "start_sample": int(start),
                "amplitude": round(amplitude, 4),
                "frequency": round(frequency, 4),
            })

        return events

    def _compute_coupling(
        self,
        raw_signal: np.ndarray,
        so_filtered: np.ndarray,
        fs: float,
    ) -> float:
        """Compute SO-spindle coupling strength.

        Measures whether spindle-band envelope power is preferentially elevated
        during SO up-states compared to down-states. This phase-amplitude
        coupling (PAC) proxy indicates functional coupling between slow
        oscillations and sleep spindles.

        Higher coupling = spindles are locked to SO up-states = better memory
        consolidation (Staresina et al. 2015).

        Returns:
            Coupling strength 0-1.
        """
        min_samples = int(fs * 3.0)
        if len(raw_signal) < min_samples:
            return 0.0

        # Get spindle-band envelope
        sp_filtered = self._bandpass(
            raw_signal, _SPINDLE_LOW, _SPINDLE_HIGH, fs, order=4
        )
        if sp_filtered is None:
            return 0.0

        sp_envelope = np.abs(hilbert(sp_filtered))

        # Identify SO up-states (positive phases) and down-states (negative phases)
        up_mask = so_filtered > 0
        down_mask = so_filtered < 0

        n_up = np.sum(up_mask)
        n_down = np.sum(down_mask)

        if n_up < 10 or n_down < 10:
            return 0.0

        # Mean spindle envelope during SO up-states vs down-states
        mean_up = float(np.mean(sp_envelope[up_mask]))
        mean_down = float(np.mean(sp_envelope[down_mask]))

        # Coupling index: normalized difference
        total = mean_up + mean_down
        if total < 1e-10:
            return 0.0

        # Positive coupling: spindle power higher during up-state
        coupling = (mean_up - mean_down) / total
        return float(np.clip(coupling, 0, 1))

    @staticmethod
    def _predict_consolidation(
        so_count: int,
        so_density: float,
        coupling_strength: float,
        mean_amplitude: float,
    ) -> str:
        """Predict memory consolidation quality from SO metrics.

        Combines coupling strength (primary driver) with SO density and
        amplitude to produce a qualitative label.

        Returns:
            One of: 'poor', 'moderate', 'good', 'excellent'
        """
        if so_count == 0:
            return "poor"

        # Composite score: 50% coupling, 30% density, 20% amplitude
        # Normalize density: 6+ SOs/min in N3 is physiologically healthy
        density_score = float(np.clip(so_density / 60.0, 0, 1))
        # Normalize amplitude: 75+ uV is a strong SO
        amplitude_score = float(np.clip(mean_amplitude / 150.0, 0, 1))

        composite = (
            0.50 * coupling_strength
            + 0.30 * density_score
            + 0.20 * amplitude_score
        )

        if composite >= _CONSOLIDATION_THRESHOLDS["excellent"]:
            return "excellent"
        elif composite >= _CONSOLIDATION_THRESHOLDS["good"]:
            return "good"
        elif composite >= _CONSOLIDATION_THRESHOLDS["moderate"]:
            return "moderate"
        else:
            return "poor"
