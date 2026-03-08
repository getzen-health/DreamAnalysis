"""Embedded Artifact Subspace Reconstruction (E-ASR).

Adapts ASR (traditionally multi-channel) to work on single channels via
time-delay embedding. Cleans artifacts in-place without discarding data.

Reference: IIT Guwahati, Sensors (Oct 2024)
           https://www.mdpi.com/1424-8220/24/20/6734

Algorithm:
1. Build calibration statistics from clean baseline data
2. For each incoming window:
   a. Create pseudo-multichannel via time-delay embedding
   b. Compute PCA on the embedded matrix
   c. Identify artifact components (those exceeding threshold * baseline SD)
   d. Reconstruct signal with artifact components removed
   e. Return cleaned single-channel signal

Pure numpy/scipy implementation -- no additional dependencies.
"""

import numpy as np
from typing import Optional, Tuple


class EmbeddedASR:
    """Single-channel artifact rejection via delay-vector embedding + ASR.

    Instead of discarding entire epochs when artifacts are detected (losing
    good data surrounding the artifact), E-ASR reconstructs a clean version
    of the signal by projecting out artifact subspace components.

    The key insight: a single channel can be turned into a pseudo-multichannel
    matrix via time-delay embedding (Takens' theorem). Standard ASR then
    operates on this matrix, comparing covariance structure against a
    known-clean baseline. Components whose variance exceeds the baseline
    by more than ``threshold`` standard deviations are projected back to
    baseline variance levels.

    Usage:
        easr = EmbeddedASR(fs=256)

        # During baseline (eyes-closed resting, 30-60s of clean data):
        easr.fit_baseline(clean_signal)

        # During live recording:
        cleaned = easr.clean(noisy_signal)
    """

    def __init__(
        self,
        fs: int = 256,
        embedding_dim: int = 10,
        threshold: float = 5.0,
        window_seconds: float = 0.5,
    ):
        """Initialize E-ASR.

        Args:
            fs: Sampling rate in Hz.
            embedding_dim: Number of delay taps for the embedding matrix.
                Higher values capture more temporal structure but increase
                computation. 10 works well for 256 Hz EEG.
            threshold: Rejection threshold in standard deviations of the
                baseline eigenvalues. Components exceeding this are cleaned.
                Lower = more aggressive cleaning, higher = more permissive.
                Default 5.0 follows the ASR literature.
            window_seconds: Not used in current implementation but reserved
                for future sliding-window processing.
        """
        self.fs = fs
        self.embedding_dim = embedding_dim
        self.threshold = threshold
        self.window_samples = int(window_seconds * fs)

        # Baseline statistics (set during fit_baseline)
        self._baseline_cov: Optional[np.ndarray] = None
        self._baseline_mean: Optional[np.ndarray] = None
        self._baseline_eigvals: Optional[np.ndarray] = None
        self._baseline_eigvecs: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """True when baseline statistics have been computed."""
        return self._is_fitted

    def _delay_embed(self, signal: np.ndarray) -> np.ndarray:
        """Create pseudo-multichannel matrix via time-delay embedding.

        Takes 1D signal of length N and creates (embedding_dim, N - embedding_dim + 1)
        matrix where each row is a delayed copy of the signal.

        This is Takens' embedding theorem applied to EEG: the delay-embedded
        matrix preserves the dynamical structure of the underlying system,
        allowing multivariate techniques (PCA/ASR) to operate on single-channel data.

        Args:
            signal: 1D array of length N.

        Returns:
            2D array of shape (embedding_dim, N - embedding_dim + 1).
        """
        n = len(signal)
        d = self.embedding_dim
        if n < d:
            return signal.reshape(1, -1)

        embedded = np.zeros((d, n - d + 1))
        for i in range(d):
            embedded[i] = signal[i : n - d + 1 + i]
        return embedded

    def fit_baseline(self, clean_signal: np.ndarray) -> None:
        """Compute baseline covariance statistics from known-clean EEG.

        Call this with 30-60 seconds of resting-state data that has been
        visually confirmed or threshold-checked to be artifact-free. The
        covariance of the delay-embedded baseline defines what "normal"
        signal structure looks like; deviations during live cleaning are
        interpreted as artifacts.

        Args:
            clean_signal: 1D array of clean EEG (at least embedding_dim samples).
        """
        if len(clean_signal) < self.embedding_dim * 2:
            return  # Too short to compute meaningful statistics

        embedded = self._delay_embed(clean_signal)
        self._baseline_mean = np.mean(embedded, axis=1, keepdims=True)
        centered = embedded - self._baseline_mean
        self._baseline_cov = np.cov(centered)

        # Pre-compute eigendecomposition (used in every clean() call)
        try:
            eigvals, eigvecs = np.linalg.eigh(self._baseline_cov)
            # Ensure non-negative eigenvalues (numerical stability)
            eigvals = np.maximum(eigvals, 1e-12)
            self._baseline_eigvals = eigvals
            self._baseline_eigvecs = eigvecs
            self._is_fitted = True
        except np.linalg.LinAlgError:
            self._is_fitted = False

    def clean(self, signal: np.ndarray) -> np.ndarray:
        """Clean artifacts from a single-channel EEG signal.

        If the baseline has been fitted, applies the full E-ASR algorithm:
        delay-embed the signal, compare its covariance structure against
        the baseline, and project artifact components back to baseline
        variance levels.

        If not fitted (no baseline available), falls back to simple
        amplitude-threshold interpolation so the method always returns
        a usable signal.

        Args:
            signal: 1D EEG signal array.

        Returns:
            Cleaned signal of the same length as input.
        """
        if not self._is_fitted:
            return self._simple_clean(signal)

        n = len(signal)
        d = self.embedding_dim
        if n < d * 2:
            return signal  # Too short for meaningful embedding

        # Step 1: Delay-embed the incoming signal
        embedded = self._delay_embed(signal)
        centered = embedded - self._baseline_mean

        # Step 2: Compute covariance of the current data
        data_cov = np.cov(centered)

        # Step 3: Eigendecomposition of the current data covariance
        try:
            data_eigvals, data_eigvecs = np.linalg.eigh(data_cov)
            data_eigvals = np.maximum(data_eigvals, 1e-12)
        except np.linalg.LinAlgError:
            return signal

        # Step 4: Identify artifact components
        # A component is artifactual if its variance exceeds
        # threshold * baseline variance for that component.
        # We compare in the baseline eigenvector space for consistency.
        artifact_mask = data_eigvals > (self.threshold * self._baseline_eigvals)

        if not np.any(artifact_mask):
            return signal  # No artifacts detected -- return unchanged

        # Step 5: Reconstruct with artifact components scaled back to baseline
        # For artifact components: scale their variance down to baseline level.
        # For clean components: leave unchanged.
        scaling = np.ones_like(data_eigvals)
        artifact_indices = np.where(artifact_mask)[0]
        for idx in artifact_indices:
            # Scale factor: sqrt(baseline_var / data_var)
            # This projects the artifact component back to baseline energy
            s = np.sqrt(self._baseline_eigvals[idx] / data_eigvals[idx])
            scaling[idx] = min(s, 1.0)  # Never amplify

        # Apply scaling in the principal component space
        projected = data_eigvecs.T @ centered
        projected = np.diag(scaling) @ projected
        reconstructed = data_eigvecs @ projected + self._baseline_mean

        # Step 6: Extract the cleaned signal
        # The first row of the reconstructed embedding = original time indices
        cleaned = np.zeros(n)
        n_embedded = reconstructed.shape[1]
        cleaned[:n_embedded] = reconstructed[0]

        # Fill the tail that delay embedding could not cover
        if n_embedded < n:
            cleaned[n_embedded:] = signal[n_embedded:]

        return cleaned

    def clean_multichannel(self, signals: np.ndarray) -> Tuple[np.ndarray, float]:
        """Clean each channel independently and return artifact ratio.

        Args:
            signals: 2D array (n_channels, n_samples).

        Returns:
            Tuple of (cleaned_signals, artifact_cleaned_ratio) where
            artifact_cleaned_ratio is the fraction of channels that had
            artifacts cleaned (0.0 to 1.0).
        """
        if signals.ndim == 1:
            cleaned = self.clean(signals)
            # Compute ratio: fraction of samples that changed significantly
            ratio = self._compute_change_ratio(signals, cleaned)
            return cleaned.reshape(1, -1), ratio

        n_channels = signals.shape[0]
        cleaned = np.zeros_like(signals)
        channels_with_artifacts = 0

        for ch in range(n_channels):
            cleaned[ch] = self.clean(signals[ch])
            if self._compute_change_ratio(signals[ch], cleaned[ch]) > 0.01:
                channels_with_artifacts += 1

        ratio = channels_with_artifacts / max(n_channels, 1)
        return cleaned, ratio

    def _compute_change_ratio(
        self, original: np.ndarray, cleaned: np.ndarray
    ) -> float:
        """Compute fraction of signal that was meaningfully altered.

        A sample is considered "cleaned" if the absolute difference exceeds
        1% of the original signal's standard deviation. This avoids counting
        floating-point noise as cleaning.

        Returns:
            Float in [0.0, 1.0].
        """
        if len(original) == 0:
            return 0.0
        orig_std = np.std(original)
        if orig_std < 1e-10:
            return 0.0
        diff = np.abs(original - cleaned)
        threshold = 0.01 * orig_std
        n_changed = np.sum(diff > threshold)
        return float(n_changed / len(original))

    def _simple_clean(
        self, signal: np.ndarray, threshold_uv: float = 75.0
    ) -> np.ndarray:
        """Fallback: interpolate samples exceeding amplitude threshold.

        Used when no baseline has been recorded. Identifies samples with
        amplitude above threshold_uv and replaces them via linear
        interpolation from neighboring clean samples.

        Args:
            signal: 1D EEG signal.
            threshold_uv: Amplitude threshold in microvolts.

        Returns:
            Cleaned signal (same length).
        """
        cleaned = signal.copy()
        bad = np.abs(cleaned) > threshold_uv
        if np.any(bad):
            good = ~bad
            if np.sum(good) > 2:
                cleaned[bad] = np.interp(
                    np.where(bad)[0], np.where(good)[0], cleaned[good]
                )
        return cleaned
