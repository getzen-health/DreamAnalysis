"""Advanced artifact rejection for 4-channel consumer EEG (Muse 2).

ICA requires >4 channels, so we use statistical + spectral methods instead.

Methods:
1. Amplitude threshold (existing): reject if |signal| > threshold uV
2. Gradient threshold: reject if consecutive sample diff > 10 uV/ms
3. Spectral contamination: reject if broadband power (25-45 Hz) z-score > 2 vs rolling baseline
4. Flat channel detection: reject if std(channel) < 0.5 uV over epoch (electrode disconnected)
5. Epoch kurtosis: reject if kurtosis > 8 (impulsive artifacts)
6. Channel correlation: flag if inter-channel correlation drops below 0.1 (one channel bad)

Reference: Krigolson 2021 (Muse preprocessing validation).
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis as scipy_kurtosis
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# NumPy 2.0 renamed np.trapz -> np.trapezoid; 1.x only has np.trapz
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


@dataclass
class ArtifactResult:
    """Result of artifact detection for a single epoch."""

    is_clean: bool
    rejected_reasons: List[str]
    channel_quality: List[float]  # 0-1 per channel
    overall_quality: float  # 0-1
    artifact_types: Dict[str, int] = field(default_factory=dict)  # count per type


class MultiMethodArtifactDetector:
    """6-method artifact detector optimized for Muse 2 (4-channel, 256 Hz).

    Since ICA is mathematically impossible with only 4 channels (requires
    n_channels > n_artifacts), this detector uses statistical and spectral
    methods to identify contaminated epochs without decomposition.

    Each method targets a specific artifact class:
        - Amplitude: blinks, large movements
        - Gradient: electrode pops, sudden jumps
        - Spectral: sustained muscle (EMG) contamination
        - Flat: electrode disconnection / gel bridge
        - Kurtosis: impulsive transient artifacts
        - Correlation: single-channel failures
    """

    def __init__(
        self,
        fs: float = 256.0,
        amplitude_threshold: float = 75.0,
        gradient_threshold: float = 10.0,
        kurtosis_threshold: float = 8.0,
        flat_threshold: float = 0.5,
        spectral_z_threshold: float = 2.0,
        correlation_threshold: float = 0.1,
    ) -> None:
        self.fs = fs
        self.amplitude_threshold = amplitude_threshold
        self.gradient_threshold = gradient_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.flat_threshold = flat_threshold
        self.spectral_z_threshold = spectral_z_threshold
        self.correlation_threshold = correlation_threshold

        # Rolling baseline for spectral contamination (broadband 25-45 Hz power)
        self._spectral_baseline_powers: List[float] = []
        self._spectral_baseline_max: int = 100  # keep last 100 epochs

    def _check_amplitude(self, epoch: np.ndarray) -> Tuple[bool, List[bool]]:
        """Method 1: Reject if any sample exceeds amplitude threshold.

        Args:
            epoch: (n_channels, n_samples) array.

        Returns:
            (artifact_detected, per_channel_flags)
        """
        n_channels = epoch.shape[0]
        channel_flags = []
        for ch in range(n_channels):
            exceeded = bool(np.any(np.abs(epoch[ch]) > self.amplitude_threshold))
            channel_flags.append(exceeded)
        return any(channel_flags), channel_flags

    def _check_gradient(self, epoch: np.ndarray) -> Tuple[bool, List[bool]]:
        """Method 2: Reject if consecutive sample diff exceeds gradient threshold.

        Gradient is computed as uV per millisecond:
            gradient = |sample[i+1] - sample[i]| / (1000 / fs)
            = |diff| * fs / 1000

        Args:
            epoch: (n_channels, n_samples) array.

        Returns:
            (artifact_detected, per_channel_flags)
        """
        n_channels = epoch.shape[0]
        ms_per_sample = 1000.0 / self.fs  # ms between consecutive samples
        channel_flags = []
        for ch in range(n_channels):
            diff = np.abs(np.diff(epoch[ch]))
            gradient_uv_per_ms = diff / ms_per_sample
            exceeded = bool(np.any(gradient_uv_per_ms > self.gradient_threshold))
            channel_flags.append(exceeded)
        return any(channel_flags), channel_flags

    def _check_spectral(self, epoch: np.ndarray) -> Tuple[bool, List[bool]]:
        """Method 3: Reject if broadband (25-45 Hz) power z-score > threshold.

        Computes mean broadband power across channels, compares to rolling
        baseline. First few epochs (< 5) are never rejected by this method
        since no baseline exists yet.

        Args:
            epoch: (n_channels, n_samples) array.

        Returns:
            (artifact_detected, per_channel_flags)
        """
        n_channels = epoch.shape[0]
        n_samples = epoch.shape[1]

        # Need enough samples for meaningful PSD
        nperseg = min(n_samples, int(self.fs))
        if nperseg < 8:
            return False, [False] * n_channels

        channel_powers = []
        channel_flags = []

        for ch in range(n_channels):
            freqs, psd = scipy_signal.welch(
                epoch[ch], fs=self.fs, nperseg=nperseg
            )
            # Broadband power: 25-45 Hz (muscle contamination band)
            mask = (freqs >= 25.0) & (freqs <= 45.0)
            if not mask.any():
                channel_powers.append(0.0)
                channel_flags.append(False)
                continue
            bb_power = float(_trapezoid(psd[mask], freqs[mask]))
            channel_powers.append(bb_power)
            channel_flags.append(False)  # will update below

        mean_power = float(np.mean(channel_powers))

        # Update rolling baseline
        self._spectral_baseline_powers.append(mean_power)
        if len(self._spectral_baseline_powers) > self._spectral_baseline_max:
            self._spectral_baseline_powers = self._spectral_baseline_powers[
                -self._spectral_baseline_max :
            ]

        # Need at least 5 epochs for a meaningful baseline
        if len(self._spectral_baseline_powers) < 5:
            return False, [False] * n_channels

        baseline_mean = float(np.mean(self._spectral_baseline_powers[:-1]))
        baseline_std = float(np.std(self._spectral_baseline_powers[:-1]))

        if baseline_std < 1e-10:
            # If baseline has zero variance (all identical epochs), use a
            # relative threshold: flag if power exceeds 3x the baseline mean.
            if baseline_mean > 0 and mean_power > 3.0 * baseline_mean:
                artifact_detected = True
                for ch in range(n_channels):
                    if channel_powers[ch] > 3.0 * baseline_mean:
                        channel_flags[ch] = True
                return artifact_detected, channel_flags
            return False, [False] * n_channels

        z_score = (mean_power - baseline_mean) / baseline_std
        artifact_detected = z_score > self.spectral_z_threshold

        if artifact_detected:
            # Flag channels with above-average broadband power
            for ch in range(n_channels):
                if channel_powers[ch] > baseline_mean + self.spectral_z_threshold * baseline_std:
                    channel_flags[ch] = True

        return artifact_detected, channel_flags

    def _check_flat(self, epoch: np.ndarray) -> Tuple[bool, List[bool]]:
        """Method 4: Reject if any channel has std < flat_threshold.

        A flat channel indicates electrode disconnection or gel bridge.

        Args:
            epoch: (n_channels, n_samples) array.

        Returns:
            (artifact_detected, per_channel_flags)
        """
        n_channels = epoch.shape[0]
        channel_flags = []
        for ch in range(n_channels):
            is_flat = bool(np.std(epoch[ch]) < self.flat_threshold)
            channel_flags.append(is_flat)
        return any(channel_flags), channel_flags

    def _check_kurtosis(self, epoch: np.ndarray) -> Tuple[bool, List[bool]]:
        """Method 5: Reject if any channel kurtosis exceeds threshold.

        High kurtosis indicates impulsive artifacts (spikes, pops).
        Uses Fisher definition (normal = 0).

        Args:
            epoch: (n_channels, n_samples) array.

        Returns:
            (artifact_detected, per_channel_flags)
        """
        n_channels = epoch.shape[0]
        channel_flags = []
        for ch in range(n_channels):
            kurt = float(scipy_kurtosis(epoch[ch], fisher=True))
            exceeded = abs(kurt) > self.kurtosis_threshold
            channel_flags.append(exceeded)
        return any(channel_flags), channel_flags

    def _check_correlation(self, epoch: np.ndarray) -> Tuple[bool, List[bool]]:
        """Method 6: Flag if inter-channel correlation drops below threshold.

        For 4-channel Muse 2, nearby channels should show moderate correlation.
        A channel with near-zero correlation to all others is likely bad.

        Args:
            epoch: (n_channels, n_samples) array.

        Returns:
            (artifact_detected, per_channel_flags)
        """
        n_channels = epoch.shape[0]
        if n_channels < 2:
            return False, [False] * n_channels

        # Compute correlation matrix
        # Handle constant channels (would produce NaN in corrcoef)
        stds = np.std(epoch, axis=1)
        if np.any(stds < 1e-10):
            # If a channel is constant, it's already caught by flat detection
            channel_flags = [bool(s < 1e-10) for s in stds]
            return any(channel_flags), channel_flags

        corr_matrix = np.corrcoef(epoch)

        channel_flags = []
        for ch in range(n_channels):
            # Mean absolute correlation with all other channels
            other_corrs = []
            for other_ch in range(n_channels):
                if other_ch != ch:
                    other_corrs.append(abs(corr_matrix[ch, other_ch]))
            mean_corr = float(np.mean(other_corrs))
            is_low = mean_corr < self.correlation_threshold
            channel_flags.append(is_low)

        return any(channel_flags), channel_flags

    def detect(self, epoch: np.ndarray) -> ArtifactResult:
        """Analyze a single epoch (n_channels, n_samples) for artifacts.

        Runs all 6 detection methods and aggregates results.

        Args:
            epoch: 2D array of shape (n_channels, n_samples).

        Returns:
            ArtifactResult with per-channel quality scores and rejection reasons.
        """
        if epoch.ndim == 1:
            epoch = epoch.reshape(1, -1)

        n_channels = epoch.shape[0]
        rejected_reasons: List[str] = []
        artifact_types: Dict[str, int] = {}

        # Per-channel penalty accumulator (0 = perfect, higher = worse)
        channel_penalties = np.zeros(n_channels, dtype=float)

        # Run all 6 methods
        methods = [
            ("amplitude", self._check_amplitude),
            ("gradient", self._check_gradient),
            ("spectral", self._check_spectral),
            ("flat", self._check_flat),
            ("kurtosis", self._check_kurtosis),
            ("correlation", self._check_correlation),
        ]

        for method_name, method_fn in methods:
            detected, ch_flags = method_fn(epoch)
            if detected:
                rejected_reasons.append(method_name)
                bad_count = sum(1 for f in ch_flags if f)
                artifact_types[method_name] = bad_count
                for ch, flagged in enumerate(ch_flags):
                    if flagged:
                        channel_penalties[ch] += 1.0

        # Compute per-channel quality: 1.0 = clean, 0.0 = all methods flagged
        n_methods = len(methods)
        channel_quality = [
            max(0.0, 1.0 - channel_penalties[ch] / n_methods)
            for ch in range(n_channels)
        ]

        overall_quality = float(np.mean(channel_quality))
        is_clean = len(rejected_reasons) == 0

        return ArtifactResult(
            is_clean=is_clean,
            rejected_reasons=rejected_reasons,
            channel_quality=channel_quality,
            overall_quality=overall_quality,
            artifact_types=artifact_types,
        )

    def detect_batch(
        self, epochs: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[ArtifactResult]]:
        """Filter a batch of epochs, returning only clean ones + all results.

        Args:
            epochs: List of 2D arrays, each (n_channels, n_samples).

        Returns:
            Tuple of (clean_epochs, all_results).
        """
        clean_epochs: List[np.ndarray] = []
        all_results: List[ArtifactResult] = []

        for epoch in epochs:
            result = self.detect(epoch)
            all_results.append(result)
            if result.is_clean:
                clean_epochs.append(epoch)

        return clean_epochs, all_results

    def get_clean_ratio(self, results: List[ArtifactResult]) -> float:
        """Percentage of clean epochs in a batch.

        Args:
            results: List of ArtifactResult from detect() or detect_batch().

        Returns:
            Float between 0.0 and 1.0 representing the fraction of clean epochs.
        """
        if not results:
            return 0.0
        n_clean = sum(1 for r in results if r.is_clean)
        return n_clean / len(results)
