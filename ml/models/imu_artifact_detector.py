"""IMU-based motion artifact detector for Muse 2 EEG.

Uses accelerometer (and optionally gyroscope) RMS magnitude from the Muse 2
ANCILLARY preset to flag EEG windows contaminated by head or body movement.

Science basis:
- Berg & Scherg (1994): motion artifact correlates with EMG and physical movement
- Vigon et al. (1999): accelerometer RMS magnitude predicts artifact presence
  with ~85% accuracy when properly threshold-calibrated
- Threshold approach: compute RMS acceleration minus gravity vector (9.81 m/s²),
  if residual exceeds ~0.15 g (1.47 m/s²), flag as motion-contaminated

Muse 2 IMU specs (BrainFlow ANCILLARY preset):
- Accelerometer: 3-axis, ~52 Hz, units m/s²
- Gyroscope: 3-axis, ~52 Hz, units deg/s
- Channel order: [x, y, z]
"""

import numpy as np
import threading
from typing import Dict, Optional

# Gravity constant in m/s²
_GRAVITY_MS2 = 9.81
# Gravity in g-units
_GRAVITY_G = 1.0

# Module-level singleton
_imu_detector_instance: Optional["IMUArtifactDetector"] = None
_imu_detector_lock = threading.Lock()


class IMUArtifactDetector:
    """Real-time motion artifact detection from Muse 2 IMU data.

    Uses accelerometer RMS magnitude to detect head/body movements
    that contaminate frontal EEG channels.

    Typical usage::

        detector = IMUArtifactDetector()
        # During 2-minute eyes-closed resting state:
        detector.calibrate_resting(acc_resting, fs=52.0)
        # During live recording:
        result = detector.detect(acc_window, gyro_window, fs=52.0)
        if result["motion_detected"]:
            reject_eeg_epoch()
    """

    # Thresholds in m/s² above resting baseline (1 g = 9.81 m/s²)
    # 0.15 g × 9.81 = 1.4715 m/s² — clear motion
    # 0.075 g × 9.81 = 0.7358 m/s² — mild movement

    # Gyroscope thresholds (deg/s)
    _MILD_GYRO_DPS = 5.0
    _SEVERE_GYRO_DPS = 15.0

    # Fraction of sub-windows with motion to classify as contaminated
    _CONTAMINATED_FRACTION_THRESHOLD = 0.25

    def __init__(self, threshold_g: float = 0.15, window_ms: int = 250):
        """Initialise the IMU artifact detector.

        Args:
            threshold_g: RMS acceleration above resting baseline (in g-units)
                that triggers motion detection.  Default 0.15 g is conservative
                enough to catch head nods while ignoring micro-tremor.
            window_ms: Sub-window length (ms) for RMS computation.  Multiple
                sub-windows are averaged to compute ``contaminated_fraction``.
        """
        # Store thresholds in m/s² internally for unit-consistent computation
        self._threshold_ms2 = float(threshold_g) * _GRAVITY_MS2  # severe
        self._mild_threshold_ms2 = float(threshold_g) * _GRAVITY_MS2 * 0.5  # mild
        self._threshold_g = float(threshold_g)   # kept for API exposure
        self._window_ms = int(window_ms)

        # Resting baseline in m/s² — updated by calibrate_resting / update_baseline
        # Default: pure gravity = 9.81 m/s²
        self._resting_baseline_ms2: float = _GRAVITY_MS2
        self._baseline_ready: bool = False

    # ── Calibration ────────────────────────────────────────────────────────────

    @property
    def _resting_baseline(self) -> float:
        """Public-facing resting baseline in g-units (for API / tests)."""
        return self._resting_baseline_ms2 / _GRAVITY_MS2

    @_resting_baseline.setter
    def _resting_baseline(self, value_g: float) -> None:
        """Accept assignment in g-units and store internally as m/s²."""
        self._resting_baseline_ms2 = float(value_g) * _GRAVITY_MS2

    def calibrate_resting(self, acc_data: np.ndarray, fs: float = 52.0) -> None:
        """Record resting-state baseline acceleration for normalisation.

        Call with ~30 seconds of eyes-closed, seated, still accelerometer data.
        Stores the median RMS magnitude (m/s²) across all sub-windows.

        Args:
            acc_data: Accelerometer array, shape ``(3, n_samples)`` or
                ``(n_samples, 3)``, units m/s².
            fs: IMU sampling rate in Hz.  Muse 2 default ~52 Hz.
        """
        acc = self._normalise_axes(acc_data)  # → (3, n_samples)
        rms_values_ms2 = self._compute_suwindow_rms(acc, fs)  # m/s²
        if len(rms_values_ms2) > 0:
            self._resting_baseline_ms2 = float(np.median(rms_values_ms2))
            self._baseline_ready = True

    def update_baseline(self, acc_rms: float, alpha: float = 0.01) -> None:
        """Exponential moving average update of resting baseline.

        Call once per second during known-still periods to track slow drift
        in headset orientation or body posture.

        Args:
            acc_rms: Current RMS acceleration magnitude in g-units.
            alpha: EMA smoothing factor (0 = no update, 1 = instant update).
        """
        acc_rms_ms2 = float(acc_rms) * _GRAVITY_MS2
        self._resting_baseline_ms2 = (
            alpha * acc_rms_ms2 + (1.0 - alpha) * self._resting_baseline_ms2
        )

    # ── Detection ──────────────────────────────────────────────────────────────

    def detect(
        self,
        acc_data: np.ndarray,
        gyro_data: Optional[np.ndarray] = None,
        fs: float = 52.0,
    ) -> Dict:
        """Detect motion artifacts in a window of IMU data.

        Args:
            acc_data: Accelerometer array, shape ``(3, n_samples)`` or
                ``(n_samples, 3)``, units m/s².
            gyro_data: Optional gyroscope array, same shape options, deg/s.
            fs: IMU sampling rate in Hz.

        Returns:
            Dict with keys:

            * ``motion_detected`` (bool) — True if motion artifact present
            * ``artifact_probability`` (float) — 0-1 probability
            * ``motion_rms_g`` (float) — acceleration magnitude above baseline (g)
            * ``gyro_rms_dps`` (float) — gyroscope RMS (deg/s); 0.0 if not provided
            * ``contaminated_fraction`` (float) — fraction of sub-windows with motion
            * ``recommendation`` (str) — "clean", "mild_motion", or "severe_motion"
        """
        acc = self._normalise_axes(acc_data)
        subrms_ms2 = self._compute_suwindow_rms(acc, fs)  # m/s²

        # Residual above baseline — all in m/s²
        residual_ms2 = np.maximum(0.0, subrms_ms2 - self._resting_baseline_ms2)
        mean_residual_ms2 = float(np.mean(residual_ms2)) if len(residual_ms2) > 0 else 0.0

        # Convert residual to g for output and threshold comparison
        mean_residual_g = mean_residual_ms2 / _GRAVITY_MS2
        residual_g = residual_ms2 / _GRAVITY_MS2

        # Peak residual (g) — used for classification; any single bad sub-window
        # contaminates the whole epoch, so peak is more appropriate than mean.
        peak_residual_g = float(np.max(residual_g)) if len(residual_g) > 0 else 0.0

        # Count sub-windows exceeding mild threshold (g-units)
        contaminated_count = int(np.sum(residual_g > (self._threshold_g * 0.5)))
        contaminated_fraction = (
            contaminated_count / len(residual_g) if len(residual_g) > 0 else 0.0
        )

        # Gyroscope contribution
        gyro_rms_dps = 0.0
        if gyro_data is not None:
            gyro = self._normalise_axes(gyro_data)
            gyro_mag = np.sqrt(np.sum(gyro ** 2, axis=0))
            gyro_rms_dps = float(np.sqrt(np.mean(gyro_mag ** 2)))

        # Combine into probability (uses peak for sensitivity)
        artifact_probability = self._compute_probability(
            peak_residual_g, gyro_rms_dps, contaminated_fraction
        )

        # Recommendation and binary flag (peak-based for correctness)
        recommendation, motion_detected = self._classify(
            peak_residual_g, gyro_rms_dps, contaminated_fraction
        )

        return {
            "motion_detected": motion_detected,
            "artifact_probability": round(artifact_probability, 4),
            "motion_rms_g": round(peak_residual_g, 5),
            "gyro_rms_dps": round(gyro_rms_dps, 3),
            "contaminated_fraction": round(contaminated_fraction, 4),
            "recommendation": recommendation,
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _normalise_axes(data: np.ndarray) -> np.ndarray:
        """Return data in shape (3, n_samples), transposing if needed."""
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            # Single-axis or flat — treat as 1-channel, shape (1, n)
            return arr.reshape(1, -1)
        if arr.shape[0] == 3:
            return arr
        if arr.shape[1] == 3:
            return arr.T
        # Fallback: keep as-is
        return arr

    def _compute_suwindow_rms(self, acc: np.ndarray, fs: float) -> np.ndarray:
        """Split acc into non-overlapping sub-windows and compute RMS per window.

        Args:
            acc: Shape (3, n_samples), m/s²
            fs: Sampling rate in Hz

        Returns:
            1D array of RMS magnitude values per sub-window (m/s²).
        """
        window_samples = max(1, int(self._window_ms * fs / 1000.0))
        n_samples = acc.shape[-1]
        mag = np.sqrt(np.sum(acc ** 2, axis=0))  # Euclidean magnitude per sample

        rms_values = []
        for start in range(0, n_samples - window_samples + 1, window_samples):
            seg = mag[start:start + window_samples]
            rms_values.append(float(np.sqrt(np.mean(seg ** 2))))

        if len(rms_values) == 0 and n_samples > 0:
            # Shorter than one sub-window — compute over all available samples
            rms_values.append(float(np.sqrt(np.mean(mag ** 2))))

        return np.array(rms_values)

    def _compute_probability(
        self, residual_g: float, gyro_dps: float, contaminated_fraction: float
    ) -> float:
        """Combine signals into a single [0, 1] artifact probability."""
        # Accelerometer score: sigmoid-like ramp from 0→1 over [0, threshold]
        acc_score = min(1.0, residual_g / max(self._threshold_g, 1e-6))

        # Gyroscope score: ramp from 0→1 over [0, severe gyro threshold]
        gyro_score = min(1.0, gyro_dps / max(self._SEVERE_GYRO_DPS, 1e-6))

        # Contaminated fraction score
        frac_score = min(1.0, contaminated_fraction / max(
            self._CONTAMINATED_FRACTION_THRESHOLD, 1e-6
        ))

        # Weighted combination: accelerometer is primary, gyro and fraction support
        weight_acc = 0.60
        weight_gyro = 0.20 if gyro_dps > 0 else 0.0
        weight_frac = 1.0 - weight_acc - weight_gyro

        prob = (
            weight_acc * acc_score
            + weight_gyro * gyro_score
            + weight_frac * frac_score
        )
        return float(np.clip(prob, 0.0, 1.0))

    def _classify(
        self, residual_g: float, gyro_dps: float, contaminated_fraction: float
    ) -> tuple:
        """Return (recommendation: str, motion_detected: bool)."""
        severe_acc = residual_g >= self._threshold_g
        severe_gyro = gyro_dps >= self._SEVERE_GYRO_DPS
        severe_frac = contaminated_fraction >= self._CONTAMINATED_FRACTION_THRESHOLD

        mild_acc = residual_g >= (self._threshold_g * 0.5)
        mild_gyro = gyro_dps >= self._MILD_GYRO_DPS

        if severe_acc or (severe_gyro and severe_frac):
            return "severe_motion", True
        if mild_acc or mild_gyro:
            return "mild_motion", True
        return "clean", False


def get_imu_detector() -> IMUArtifactDetector:
    """Return the module-level singleton IMUArtifactDetector.

    Thread-safe; creates the instance on first call.
    """
    global _imu_detector_instance
    if _imu_detector_instance is None:
        with _imu_detector_lock:
            if _imu_detector_instance is None:
                _imu_detector_instance = IMUArtifactDetector()
    return _imu_detector_instance
