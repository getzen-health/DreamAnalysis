"""Sleep Spindle Analyzer — detection and characterization of sleep spindles.

Detects sleep spindles (11-16 Hz bursts, 0.5-2s duration) in EEG signals,
classifies them as slow (11-13 Hz, frontal) or fast (13-16 Hz, centroparietal),
and computes session-level metrics including a memory consolidation index.

Spindle density during N2/N3 sleep predicts next-day memory performance
(r=0.4-0.6, Mander et al. 2014). Slow spindles originate frontally and
fast spindles centroparietally; both couple with slow oscillations (<1 Hz)
to drive hippocampal-neocortical memory transfer (Luthi, 2014).

For Muse 2 (AF7/AF8 frontal, TP9/TP10 temporal), slow spindles are best
captured at AF7/AF8 and fast spindles at TP9/TP10, though the 4-channel
layout limits spatial separation.

Dependencies: numpy, scipy only.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SIGMA_LOW = 11.0   # Hz — lower bound of sigma (spindle) band
_SIGMA_HIGH = 16.0  # Hz — upper bound of sigma (spindle) band
_SLOW_FAST_BOUNDARY = 13.0  # Hz — slow spindles < 13 Hz, fast >= 13 Hz

_MIN_SPINDLE_DURATION = 0.5  # seconds
_MAX_SPINDLE_DURATION = 2.0  # seconds
# Relaxed detection bounds to avoid missing edge cases
_DETECT_MIN_DURATION = 0.3   # seconds — accept slightly short spindles
_DETECT_MAX_DURATION = 3.0   # seconds — accept slightly long spindles

_DEFAULT_THRESHOLD_Z = 2.0   # z-score threshold for spindle envelope
_SO_LOW = 0.5   # Hz — slow oscillation lower bound
_SO_HIGH = 1.5  # Hz — slow oscillation upper bound


# ---------------------------------------------------------------------------
# SpindleAnalyzer
# ---------------------------------------------------------------------------

class SpindleAnalyzer:
    """Detect, characterize, and track sleep spindles across a sleep session.

    Methods:
        detect_spindles(signal, fs) -> list of spindle dicts
        analyze(eeg_signals, fs) -> summary dict for one epoch
        get_session_stats() -> aggregate stats across all analyzed epochs
        get_consolidation_score() -> float 0-100
        get_history() -> list of per-epoch results
        reset() -> clear session state
    """

    def __init__(self, threshold_z: float = _DEFAULT_THRESHOLD_Z):
        """Initialize SpindleAnalyzer.

        Args:
            threshold_z: z-score threshold for spindle detection envelope.
                Higher = fewer false positives but may miss weak spindles.
                Default 2.0 balances sensitivity and specificity.
        """
        self._threshold_z = threshold_z
        self._history: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_spindles(
        self, signal: np.ndarray, fs: float = 256.0
    ) -> List[Dict]:
        """Detect and characterize individual sleep spindles.

        Args:
            signal: 1D EEG signal (single channel). If 2D, uses channel 1
                (AF7 for Muse 2).
            fs: Sampling frequency in Hz.

        Returns:
            List of dicts, each with:
                start_sample (int): sample index where spindle begins
                duration_s (float): spindle duration in seconds
                amplitude (float): peak envelope amplitude (uV)
                frequency (float): dominant frequency within spindle (Hz)
                type (str): 'slow' (11-13 Hz) or 'fast' (13-16 Hz)
        """
        signal = self._to_1d(signal)

        # Need at least 1 second of data for meaningful detection
        min_samples = int(fs * 1.0)
        if len(signal) < min_samples:
            return []

        # Bandpass filter in sigma band (11-16 Hz)
        filtered = self._bandpass(signal, _SIGMA_LOW, _SIGMA_HIGH, fs)
        if filtered is None:
            return []

        # Compute envelope via Hilbert transform
        analytic = hilbert(filtered)
        envelope = np.abs(analytic)

        # Threshold: mean + z * std
        env_mean = np.mean(envelope)
        env_std = np.std(envelope)
        if env_std < 1e-10:
            return []
        threshold = env_mean + self._threshold_z * env_std

        # Find contiguous regions above threshold
        above = envelope > threshold
        spindles: List[Dict] = []

        in_spindle = False
        start_idx = 0
        for i in range(len(above)):
            if above[i] and not in_spindle:
                in_spindle = True
                start_idx = i
            elif not above[i] and in_spindle:
                in_spindle = False
                self._characterize_spindle(
                    filtered, envelope, start_idx, i, fs, spindles
                )

        # Handle spindle ending at signal boundary
        if in_spindle:
            self._characterize_spindle(
                filtered, envelope, start_idx, len(above), fs, spindles
            )

        return spindles

    def analyze(
        self, eeg_signals: np.ndarray, fs: float = 256.0
    ) -> Dict:
        """Analyze an EEG epoch for spindle characteristics.

        Args:
            eeg_signals: 1D (n_samples,) or 2D (n_channels, n_samples) array.
                If 2D, uses channel 1 (AF7) for spindle detection.
            fs: Sampling frequency in Hz.

        Returns:
            Dict with:
                spindle_count (int): number of spindles detected
                spindle_density (float): spindles per minute
                mean_amplitude (float): mean peak amplitude across spindles
                mean_frequency (float): mean dominant frequency (Hz)
                mean_duration (float): mean spindle duration (seconds)
                spindle_type_distribution (dict): {'slow': frac, 'fast': frac}
                consolidation_index (float): 0-100 memory consolidation score
        """
        eeg_signals = np.asarray(eeg_signals, dtype=float)
        signal = self._to_1d(eeg_signals)

        duration_sec = len(signal) / fs
        duration_min = duration_sec / 60.0

        # Detect spindles
        spindles = self.detect_spindles(signal, fs)
        n_spindles = len(spindles)

        # Compute metrics
        if n_spindles > 0:
            density = n_spindles / max(duration_min, 1e-6)
            mean_amp = float(np.mean([s["amplitude"] for s in spindles]))
            mean_freq = float(np.mean([s["frequency"] for s in spindles]))
            mean_dur = float(np.mean([s["duration_s"] for s in spindles]))

            n_slow = sum(1 for s in spindles if s["type"] == "slow")
            n_fast = n_spindles - n_slow
            type_dist = {
                "slow": round(n_slow / n_spindles, 4),
                "fast": round(n_fast / n_spindles, 4),
            }
        else:
            density = 0.0
            mean_amp = 0.0
            mean_freq = 0.0
            mean_dur = 0.0
            type_dist = {"slow": 0.0, "fast": 0.0}

        # Memory consolidation index (0-100)
        consolidation = self._compute_consolidation_index(
            signal, fs, n_spindles, density, spindles
        )

        result = {
            "spindle_count": n_spindles,
            "spindle_density": round(density, 2),
            "mean_amplitude": round(mean_amp, 4),
            "mean_frequency": round(mean_freq, 2),
            "mean_duration": round(mean_dur, 4),
            "spindle_type_distribution": type_dist,
            "consolidation_index": round(consolidation, 2),
        }

        self._history.append(result)
        return result

    def get_session_stats(self) -> Dict:
        """Aggregate statistics across all analyzed epochs in this session.

        Returns:
            Dict with n_epochs, total_spindles, mean_density,
            mean_amplitude, mean_consolidation_index.
        """
        if not self._history:
            return {
                "n_epochs": 0,
                "total_spindles": 0,
                "mean_density": 0.0,
                "mean_amplitude": 0.0,
                "mean_consolidation_index": 0.0,
            }

        total_spindles = sum(h["spindle_count"] for h in self._history)
        densities = [h["spindle_density"] for h in self._history]
        amplitudes = [
            h["mean_amplitude"] for h in self._history if h["mean_amplitude"] > 0
        ]
        consol_indices = [h["consolidation_index"] for h in self._history]

        return {
            "n_epochs": len(self._history),
            "total_spindles": total_spindles,
            "mean_density": round(float(np.mean(densities)), 2),
            "mean_amplitude": round(
                float(np.mean(amplitudes)) if amplitudes else 0.0, 4
            ),
            "mean_consolidation_index": round(
                float(np.mean(consol_indices)), 2
            ),
        }

    def get_consolidation_score(self) -> float:
        """Return the latest memory consolidation index (0-100).

        If no epochs analyzed, returns 0.0.
        """
        if not self._history:
            return 0.0
        return float(self._history[-1]["consolidation_index"])

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
    def _to_1d(signal: np.ndarray) -> np.ndarray:
        """Extract single channel from multichannel input.

        For Muse 2: ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10.
        Prefers AF7 (ch1) for frontal spindle detection.
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
        order: int = 4,
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

    def _characterize_spindle(
        self,
        filtered: np.ndarray,
        envelope: np.ndarray,
        start: int,
        end: int,
        fs: float,
        spindles: List[Dict],
    ) -> None:
        """Characterize a single spindle candidate and append if valid.

        Checks duration constraints and computes frequency, amplitude, type.
        """
        duration_s = (end - start) / fs

        if duration_s < _DETECT_MIN_DURATION or duration_s > _DETECT_MAX_DURATION:
            return

        # Peak envelope amplitude
        amplitude = float(np.max(envelope[start:end]))

        # Dominant frequency via zero-crossing rate within the spindle segment
        segment = filtered[start:end]
        freq = self._estimate_frequency(segment, fs)

        # Classify type based on dominant frequency
        spindle_type = "slow" if freq < _SLOW_FAST_BOUNDARY else "fast"

        spindles.append({
            "start_sample": int(start),
            "duration_s": round(duration_s, 4),
            "amplitude": round(amplitude, 4),
            "frequency": round(freq, 2),
            "type": spindle_type,
        })

    @staticmethod
    def _estimate_frequency(segment: np.ndarray, fs: float) -> float:
        """Estimate dominant frequency of a spindle segment.

        Uses zero-crossing rate for short segments (more robust than FFT
        for 0.5-2s bursts). Falls back to midband if segment is too short.
        """
        if len(segment) < 4:
            return (_SIGMA_LOW + _SIGMA_HIGH) / 2.0

        # Zero-crossing count
        zero_crossings = np.where(np.diff(np.sign(segment)))[0]
        n_crossings = len(zero_crossings)

        if n_crossings < 2:
            return (_SIGMA_LOW + _SIGMA_HIGH) / 2.0

        # Each full cycle has 2 zero crossings
        duration_s = len(segment) / fs
        freq = n_crossings / (2.0 * duration_s)

        # Clamp to sigma band
        return float(np.clip(freq, _SIGMA_LOW, _SIGMA_HIGH))

    def _compute_consolidation_index(
        self,
        signal: np.ndarray,
        fs: float,
        n_spindles: int,
        density: float,
        spindles: List[Dict],
    ) -> float:
        """Compute memory consolidation index (0-100).

        Based on:
        1. Spindle density (40%) — >3/min is excellent for N2/N3
        2. SO-spindle coupling proxy (35%) — spindle envelope correlation with SO phase
        3. Spindle quality (25%) — duration and amplitude consistency

        References:
            Luthi (2014) — spindle generation mechanisms
            Mander et al. (2014) — spindle density predicts memory
        """
        if n_spindles == 0:
            return 0.0

        # Component 1: Spindle density score (0-1)
        # 3+ spindles/min is considered good; normalize to 4/min as ceiling
        density_score = float(np.clip(density / 4.0, 0, 1))

        # Component 2: SO-spindle coupling proxy (0-1)
        coupling_score = self._compute_so_coupling(signal, fs)

        # Component 3: Spindle quality (0-1)
        # Good spindles have consistent duration (0.5-2s) and adequate amplitude
        durations = [s["duration_s"] for s in spindles]
        mean_dur = float(np.mean(durations))
        # Optimal duration ~1.0s; penalize very short or very long
        dur_quality = 1.0 - abs(mean_dur - 1.0) / 1.5
        dur_quality = float(np.clip(dur_quality, 0, 1))

        # Amplitude consistency: lower CV = more consistent
        amplitudes = [s["amplitude"] for s in spindles]
        if len(amplitudes) > 1 and np.mean(amplitudes) > 0:
            cv = float(np.std(amplitudes) / np.mean(amplitudes))
            amp_quality = float(np.clip(1.0 - cv, 0, 1))
        else:
            amp_quality = 0.5  # single spindle — neutral

        quality_score = 0.6 * dur_quality + 0.4 * amp_quality

        # Weighted combination
        index = (
            0.40 * density_score
            + 0.35 * coupling_score
            + 0.25 * quality_score
        ) * 100.0

        return float(np.clip(index, 0, 100))

    def _compute_so_coupling(self, signal: np.ndarray, fs: float) -> float:
        """Compute SO-spindle coupling strength as a proxy for PAC.

        Measures whether spindle envelope power is elevated at SO up-states
        relative to baseline.

        Returns:
            Coupling score 0-1. Higher = spindles preferentially occur
            during SO up-states (favorable for memory consolidation).
        """
        min_samples = int(fs * 2.0)
        if len(signal) < min_samples:
            return 0.0

        # Bandpass for slow oscillations
        so_filtered = self._bandpass(signal, _SO_LOW, _SO_HIGH, fs, order=2)
        if so_filtered is None:
            return 0.0

        # Spindle envelope
        sp_filtered = self._bandpass(signal, _SIGMA_LOW, _SIGMA_HIGH, fs, order=4)
        if sp_filtered is None:
            return 0.0

        sp_envelope = np.abs(hilbert(sp_filtered))

        # SO up-states: positive zero crossings
        crossings = np.where(np.diff(np.sign(so_filtered)) > 0)[0]
        if len(crossings) < 2:
            return 0.0

        # Sample spindle envelope around SO up-states (500ms window)
        window = int(0.5 * fs)
        up_values = []
        for idx in crossings[:30]:  # cap for performance
            start = max(0, idx - window // 2)
            end = min(len(sp_envelope), idx + window // 2)
            if end > start:
                up_values.append(float(np.mean(sp_envelope[start:end])))

        if not up_values:
            return 0.0

        baseline = float(np.mean(sp_envelope))
        if baseline < 1e-10:
            return 0.0

        mean_up = float(np.mean(up_values))
        coupling = (mean_up - baseline) / (baseline + 1e-10)
        return float(np.clip(coupling, 0, 1))
