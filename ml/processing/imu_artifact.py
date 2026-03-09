"""IMU-based motion artifact removal for Muse 2 EEG.

Uses accelerometer/gyroscope data as reference signals for adaptive LMS
filtering. Also classifies activity context (still/walking/active).
"""
import numpy as np


class IMUArtifactFilter:
    """Adaptive LMS filter using IMU reference to remove motion artifacts."""

    def __init__(self, n_taps: int = 32, mu: float = 0.01):
        """
        Args:
            n_taps: Number of filter taps (FIR filter length)
            mu: Learning rate for LMS adaptation (0.001-0.05)
        """
        self.n_taps = n_taps
        self.mu = mu
        self.weights = np.zeros(n_taps)

    def filter_channel(self, eeg_channel: np.ndarray, accel_reference: np.ndarray):
        """Remove motion-correlated component from single EEG channel.

        Args:
            eeg_channel: 1D array of EEG samples
            accel_reference: 1D array of accelerometer magnitude (same length)

        Returns:
            cleaned: 1D array of cleaned EEG
            artifact_estimate: 1D array of estimated artifact component
        """
        n = len(eeg_channel)
        cleaned = np.zeros(n)
        artifact_estimate = np.zeros(n)
        w = self.weights.copy()

        for i in range(self.n_taps, n):
            x = accel_reference[i - self.n_taps:i][::-1]  # reference window
            y_hat = np.dot(w, x)  # estimated artifact
            error = eeg_channel[i] - y_hat  # cleaned signal
            w += 2 * self.mu * error * x  # LMS weight update
            cleaned[i] = error
            artifact_estimate[i] = y_hat

        # Copy first n_taps samples unchanged
        cleaned[:self.n_taps] = eeg_channel[:self.n_taps]
        self.weights = w
        return cleaned, artifact_estimate

    def filter_multichannel(self, eeg_signals: np.ndarray, accel_data: np.ndarray):
        """Filter all EEG channels using accelerometer reference.

        Args:
            eeg_signals: (n_channels, n_samples) EEG array
            accel_data: (3, n_samples) accelerometer [x, y, z] or (n_samples,) magnitude

        Returns:
            cleaned_signals: (n_channels, n_samples) cleaned EEG
            artifact_ratio: fraction of signal power attributed to motion
        """
        if accel_data.ndim == 2:
            # Compute magnitude from xyz
            accel_mag = np.sqrt(np.sum(accel_data**2, axis=0))
        else:
            accel_mag = accel_data

        # Remove DC from accelerometer (gravity component)
        accel_ref = accel_mag - np.mean(accel_mag)

        n_channels = eeg_signals.shape[0]
        cleaned = np.zeros_like(eeg_signals)
        total_artifact_power = 0.0
        total_signal_power = 0.0

        for ch in range(n_channels):
            ch_cleaned, artifact = self.filter_channel(eeg_signals[ch], accel_ref)
            cleaned[ch] = ch_cleaned
            total_artifact_power += np.mean(artifact**2)
            total_signal_power += np.mean(eeg_signals[ch]**2)

        artifact_ratio = total_artifact_power / (total_signal_power + 1e-10)
        return cleaned, float(artifact_ratio)

    def reset(self):
        """Reset filter weights for new session."""
        self.weights = np.zeros(self.n_taps)


def classify_activity(accel_data: np.ndarray, fs: int = 52) -> dict:
    """Classify user activity from accelerometer data.

    Args:
        accel_data: (3, n_samples) accelerometer xyz or (n_samples,) magnitude
        fs: accelerometer sampling rate (Muse 2 auxiliary = ~52 Hz)

    Returns:
        dict with activity_label, confidence, accel_variance
    """
    if accel_data.ndim == 2:
        accel_mag = np.sqrt(np.sum(accel_data**2, axis=0))
    else:
        accel_mag = accel_data

    # Variance of magnitude (gravity-removed)
    accel_detrended = accel_mag - np.mean(accel_mag)
    variance = float(np.var(accel_detrended))

    # Thresholds (in g^2, Muse 2 accelerometer reports in g)
    if variance < 0.005:
        label, confidence = 'still', min(1.0, 1.0 - variance / 0.005)
    elif variance < 0.05:
        label, confidence = 'light_movement', 0.7
    elif variance < 0.2:
        label, confidence = 'walking', 0.7
    else:
        label, confidence = 'active', min(1.0, variance / 0.5)

    return {
        'activity_label': label,
        'confidence': float(confidence),
        'accel_variance': variance
    }
