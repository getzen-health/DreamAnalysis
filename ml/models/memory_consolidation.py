"""Sleep memory consolidation tracker via spindle-slow oscillation coupling.

Monitors memory consolidation quality during sleep by detecting:
1. Sleep spindles (11-16 Hz bursts during N2/N3)
2. Slow oscillations (SO, 0.5-1.5 Hz during N3)
3. Spindle-SO coupling strength (r=0.4-0.6 correlates with memory retention)

Higher coupling = better memory consolidation during sleep.

References:
    npj Science of Learning (2025) — Personalized TMR with SO-spindle sync
    bioRxiv (2025) — Forehead EEG (AF7/AF8) validates spindle detection
    Staresina et al. (2015) — SO-spindle coupling and memory consolidation
"""
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal


class MemoryConsolidationTracker:
    """Track memory consolidation quality from sleep EEG.

    Detects sleep spindles and slow oscillations, then measures
    their temporal coupling as a proxy for memory consolidation.
    """

    def __init__(self):
        self._spindle_band = (11, 16)   # Hz — sleep spindle frequency
        self._so_band = (0.5, 1.5)      # Hz — slow oscillation frequency
        self._spindle_threshold_z = 1.5  # z-score threshold for spindle detection
        self._session_data: Dict[str, List[Dict]] = {}

    def analyze_epoch(
        self,
        signal: np.ndarray,
        fs: float = 256,
        sleep_stage: str = "N2",
        user_id: str = "default",
    ) -> Dict:
        """Analyze a single epoch for memory consolidation markers.

        Args:
            signal: 1D EEG signal array (single channel, preferably AF7/AF8).
            fs: Sampling rate in Hz.
            sleep_stage: Current sleep stage (N2/N3 are consolidation stages).
            user_id: User identifier.

        Returns:
            Dict with spindle_count, so_count, coupling_score,
            consolidation_quality, and spindle_density fields.
        """
        signal = np.asarray(signal, dtype=float)
        if signal.ndim == 2:
            signal = signal[1] if signal.shape[0] >= 2 else signal[0]  # prefer AF7

        duration_sec = len(signal) / fs

        # Detect spindles
        spindle_count, spindle_density = self._detect_spindles(signal, fs, duration_sec)

        # Detect slow oscillations
        so_count = self._detect_slow_oscillations(signal, fs, duration_sec)

        # Compute coupling score
        coupling = self._compute_coupling(signal, fs)

        # Consolidation quality depends on stage
        if sleep_stage in ("N2", "N3"):
            # Higher coupling + more spindles = better consolidation
            quality_raw = 0.5 * coupling + 0.3 * min(spindle_density / 3.0, 1.0) + 0.2 * min(so_count / max(duration_sec / 30, 1), 1.0)
            quality = float(np.clip(quality_raw, 0, 1))
        else:
            quality = 0.0  # REM/Wake don't contribute to spindle-mediated consolidation

        # Classify quality level
        if quality >= 0.7:
            quality_label = "excellent"
        elif quality >= 0.5:
            quality_label = "good"
        elif quality >= 0.3:
            quality_label = "moderate"
        else:
            quality_label = "low"

        result = {
            "spindle_count": spindle_count,
            "spindle_density_per_min": round(spindle_density, 2),
            "slow_oscillation_count": so_count,
            "coupling_score": round(coupling, 4),
            "consolidation_quality": round(quality, 4),
            "quality_label": quality_label,
            "sleep_stage": sleep_stage,
            "epoch_duration_sec": round(duration_sec, 2),
        }

        # Track session
        if user_id not in self._session_data:
            self._session_data[user_id] = []
        self._session_data[user_id].append(result)
        if len(self._session_data[user_id]) > 500:
            self._session_data[user_id] = self._session_data[user_id][-500:]

        return result

    def get_night_summary(self, user_id: str = "default") -> Dict:
        """Get summary statistics for the night's consolidation.

        Args:
            user_id: User identifier.

        Returns:
            Dict with total spindles, average coupling, quality distribution.
        """
        data = self._session_data.get(user_id, [])
        if not data:
            return {
                "total_epochs": 0,
                "total_spindles": 0,
                "avg_coupling": 0,
                "avg_quality": 0,
                "n2_n3_epochs": 0,
            }

        n2_n3 = [d for d in data if d["sleep_stage"] in ("N2", "N3")]
        total_spindles = sum(d["spindle_count"] for d in data)
        couplings = [d["coupling_score"] for d in n2_n3] if n2_n3 else [0]
        qualities = [d["consolidation_quality"] for d in n2_n3] if n2_n3 else [0]

        return {
            "total_epochs": len(data),
            "n2_n3_epochs": len(n2_n3),
            "total_spindles": total_spindles,
            "avg_spindle_density": round(
                float(np.mean([d["spindle_density_per_min"] for d in data])), 2
            ),
            "avg_coupling": round(float(np.mean(couplings)), 4),
            "avg_quality": round(float(np.mean(qualities)), 4),
            "quality_label": self._overall_label(float(np.mean(qualities))),
        }

    def reset(self, user_id: str = "default"):
        """Clear session data for a user."""
        self._session_data.pop(user_id, None)

    def _detect_spindles(self, signal: np.ndarray, fs: float, duration_sec: float):
        """Detect sleep spindles via bandpass + threshold.

        Returns:
            (count, density_per_min)
        """
        if len(signal) < int(fs * 0.5):
            return 0, 0.0

        # Bandpass 11-16 Hz
        nyq = fs / 2
        low = self._spindle_band[0] / nyq
        high = min(self._spindle_band[1] / nyq, 0.99)
        if low >= high:
            return 0, 0.0

        try:
            b, a = scipy_signal.butter(3, [low, high], btype="band")
            filtered = scipy_signal.filtfilt(b, a, signal)
        except (ValueError, np.linalg.LinAlgError):
            return 0, 0.0

        # Envelope via Hilbert transform
        analytic = scipy_signal.hilbert(filtered)
        envelope = np.abs(analytic)

        # Threshold: mean + z * std
        threshold = np.mean(envelope) + self._spindle_threshold_z * np.std(envelope)

        # Count peaks above threshold (minimum 0.5s apart)
        peaks, _ = scipy_signal.find_peaks(
            envelope, height=threshold, distance=int(fs * 0.5)
        )
        count = len(peaks)
        density = count / max(duration_sec / 60, 0.01)

        return count, float(density)

    def _detect_slow_oscillations(self, signal: np.ndarray, fs: float, duration_sec: float):
        """Detect slow oscillations (0.5-1.5 Hz) via bandpass + peak detection."""
        if len(signal) < int(fs * 2):
            return 0

        nyq = fs / 2
        low = self._so_band[0] / nyq
        high = min(self._so_band[1] / nyq, 0.99)
        if low >= high or low <= 0:
            return 0

        try:
            b, a = scipy_signal.butter(2, [low, high], btype="band")
            filtered = scipy_signal.filtfilt(b, a, signal)
        except (ValueError, np.linalg.LinAlgError):
            return 0

        # Count negative-to-positive zero crossings (each = one SO cycle)
        crossings = np.where(np.diff(np.sign(filtered)) > 0)[0]
        return len(crossings)

    def _compute_coupling(self, signal: np.ndarray, fs: float) -> float:
        """Compute spindle-SO coupling strength.

        Uses correlation between spindle envelope and SO phase as proxy.
        True phase-amplitude coupling (PAC) requires longer epochs.
        """
        if len(signal) < int(fs * 2):
            return 0.0

        nyq = fs / 2
        # SO phase
        so_low = self._so_band[0] / nyq
        so_high = min(self._so_band[1] / nyq, 0.99)
        # Spindle envelope
        sp_low = self._spindle_band[0] / nyq
        sp_high = min(self._spindle_band[1] / nyq, 0.99)

        if so_low <= 0 or so_low >= so_high or sp_low >= sp_high:
            return 0.0

        try:
            b_so, a_so = scipy_signal.butter(2, [so_low, so_high], btype="band")
            so_filtered = scipy_signal.filtfilt(b_so, a_so, signal)

            b_sp, a_sp = scipy_signal.butter(3, [sp_low, sp_high], btype="band")
            sp_filtered = scipy_signal.filtfilt(b_sp, a_sp, signal)
        except (ValueError, np.linalg.LinAlgError):
            return 0.0

        # Spindle envelope
        sp_envelope = np.abs(scipy_signal.hilbert(sp_filtered))

        # Correlation between SO signal and spindle envelope
        if np.std(so_filtered) < 1e-10 or np.std(sp_envelope) < 1e-10:
            return 0.0

        corr = float(np.abs(np.corrcoef(so_filtered, sp_envelope)[0, 1]))
        return float(np.clip(corr, 0, 1))

    def _overall_label(self, avg_quality: float) -> str:
        """Label overall night quality."""
        if avg_quality >= 0.7:
            return "excellent"
        elif avg_quality >= 0.5:
            return "good"
        elif avg_quality >= 0.3:
            return "moderate"
        return "low"
