"""Signal Quality Gate for EEG Analysis.

Before running any ML model, check if the EEG signal is actually usable.
Bad electrode contact, muscle artifacts, or electrical noise will produce
garbage-in-garbage-out results. Better to say "signal too noisy" than to
output a confident but wrong prediction.

Quality Checks:
1. Amplitude range — reject flat-line (<1 uV) or rail (>200 uV)
2. Line noise (50/60 Hz) — reject if powerline dominates
3. Muscle artifact (EMG) — reject if high-frequency power is excessive
4. Eye blink rate — warn if too many blinks per window
5. Impedance proxy — estimate electrode contact quality from noise floor
6. Stationarity — reject if signal statistics change abruptly (motion artifact)

Output:
- quality_score: 0.0 (garbage) to 1.0 (clean)
- is_usable: True if quality_score >= threshold
- channel_qualities: per-channel scores
- rejection_reasons: list of why quality is low
"""

import numpy as np
from typing import Dict, List, Tuple


# Quality thresholds
MIN_AMPLITUDE_UV = 1.0       # Below this = flat line / disconnected
MAX_AMPLITUDE_UV = 200.0     # Above this = railed / saturated
MAX_LINE_NOISE_RATIO = 0.4   # Line noise power / total power
MAX_EMG_RATIO = 0.5          # High-freq (>40Hz) power / total power
MAX_BLINK_RATE_PER_MIN = 30  # Excessive blinks = bad forehead contact
MIN_QUALITY_THRESHOLD = 0.4  # Below this, don't run analysis


class SignalQualityChecker:
    """Checks EEG signal quality before analysis."""

    def __init__(self, fs: int = 256, line_freq: float = 60.0):
        """
        Args:
            fs: Sampling frequency in Hz.
            line_freq: Power line frequency (60 Hz Americas/Asia, 50 Hz Europe).
        """
        self.fs = fs
        self.line_freq = line_freq

    def check_quality(self, signal: np.ndarray) -> Dict:
        """Run all quality checks on a single-channel EEG signal.

        Args:
            signal: 1D array of EEG samples (typically 4 seconds).

        Returns:
            Dict with quality_score, is_usable, and detailed metrics.
        """
        signal = np.array(signal, dtype=float)
        reasons = []
        scores = []

        # 1. Amplitude check
        amp_score, amp_reasons = self._check_amplitude(signal)
        scores.append(amp_score)
        reasons.extend(amp_reasons)

        # 2. Line noise check
        noise_score, noise_reasons = self._check_line_noise(signal)
        scores.append(noise_score)
        reasons.extend(noise_reasons)

        # 3. EMG contamination check
        emg_score, emg_reasons = self._check_emg(signal)
        scores.append(emg_score)
        reasons.extend(emg_reasons)

        # 4. Stationarity check
        stat_score, stat_reasons = self._check_stationarity(signal)
        scores.append(stat_score)
        reasons.extend(stat_reasons)

        # 5. Eye blink check
        blink_score, blink_reasons = self._check_blinks(signal)
        scores.append(blink_score)
        reasons.extend(blink_reasons)

        # 6. Spectral flatness check
        flatness_score, flatness_reasons = self._check_spectral_flatness(signal)
        scores.append(flatness_score)
        reasons.extend(flatness_reasons)

        # 7. Alpha peak presence
        alpha_score, alpha_reasons = self._check_alpha_peak(signal)
        scores.append(alpha_score)
        reasons.extend(alpha_reasons)

        # Overall quality = weighted average (7 checks)
        weights = [0.20, 0.15, 0.15, 0.15, 0.10, 0.15, 0.10]
        quality_score = float(np.average(scores, weights=weights))

        return {
            "quality_score": round(quality_score, 3),
            "is_usable": quality_score >= MIN_QUALITY_THRESHOLD,
            "threshold": MIN_QUALITY_THRESHOLD,
            "rejection_reasons": reasons,
            "details": {
                "amplitude_score": round(scores[0], 3),
                "line_noise_score": round(scores[1], 3),
                "emg_score": round(scores[2], 3),
                "stationarity_score": round(scores[3], 3),
                "blink_score": round(scores[4], 3),
                "spectral_flatness_score": round(scores[5], 3),
                "alpha_peak_score": round(scores[6], 3),
            },
            "metrics": {
                "rms_amplitude": round(float(np.sqrt(np.mean(signal ** 2))), 2),
                "peak_amplitude": round(float(np.max(np.abs(signal))), 2),
                "std": round(float(np.std(signal)), 2),
            },
        }

    def check_multichannel(self, signals: np.ndarray) -> Dict:
        """Check quality for multi-channel EEG data.

        Args:
            signals: 2D array (n_channels, n_samples).

        Returns:
            Dict with overall quality and per-channel details.
        """
        signals = np.atleast_2d(signals)
        n_channels = signals.shape[0]

        channel_results = []
        for ch in range(n_channels):
            result = self.check_quality(signals[ch])
            channel_results.append(result)

        channel_scores = [r["quality_score"] for r in channel_results]

        # Overall: use median channel quality (robust to one bad channel)
        overall_score = float(np.median(channel_scores))
        usable_channels = sum(1 for r in channel_results if r["is_usable"])

        all_reasons = []
        for i, r in enumerate(channel_results):
            for reason in r["rejection_reasons"]:
                all_reasons.append(f"Ch{i}: {reason}")

        channel_quality = [round(s * 100, 1) for s in channel_scores]  # 0-100 scale for UI
        return {
            "quality_score": round(overall_score, 3),
            "sqi": round(overall_score, 3),
            "is_usable": usable_channels >= max(1, n_channels // 2),
            "usable_channels": usable_channels,
            "total_channels": n_channels,
            "channel_scores": [round(s, 3) for s in channel_scores],
            "channel_quality": channel_quality,  # 0-100 per channel for electrode status UI
            "channel_details": channel_results,
            "rejection_reasons": all_reasons,
            "clean_ratio": round(usable_channels / max(n_channels, 1), 3),
        }

    def _check_amplitude(self, signal: np.ndarray) -> Tuple[float, List[str]]:
        """Check if signal amplitude is in valid EEG range."""
        reasons = []
        rms = np.sqrt(np.mean(signal ** 2))
        peak = np.max(np.abs(signal))

        if rms < MIN_AMPLITUDE_UV:
            reasons.append(f"Flat signal (RMS={rms:.1f} uV < {MIN_AMPLITUDE_UV})")
            return 0.0, reasons

        if peak > MAX_AMPLITUDE_UV:
            reasons.append(f"Signal railed (peak={peak:.0f} uV > {MAX_AMPLITUDE_UV})")
            return 0.1, reasons

        # Check for saturation (clipping)
        clip_threshold = 0.95 * peak
        n_clipped = np.sum(np.abs(signal) > clip_threshold)
        clip_ratio = n_clipped / len(signal)
        if clip_ratio > 0.1:
            reasons.append(f"Signal clipping ({clip_ratio:.0%} of samples)")
            return 0.3, reasons

        # Score: best when RMS is in typical EEG range (5-50 uV)
        if 5 <= rms <= 50:
            score = 1.0
        elif rms < 5:
            score = rms / 5.0
        else:
            score = max(0.3, 1.0 - (rms - 50) / 150)

        return score, reasons

    def _check_line_noise(self, signal: np.ndarray) -> Tuple[float, List[str]]:
        """Check for power line interference (50/60 Hz)."""
        reasons = []
        n = len(signal)
        if n < self.fs:
            return 0.8, reasons  # Too short to reliably detect

        # FFT
        freqs = np.fft.rfftfreq(n, 1.0 / self.fs)
        fft_power = np.abs(np.fft.rfft(signal)) ** 2

        # Total power in EEG range (1-45 Hz)
        eeg_mask = (freqs >= 1) & (freqs <= 45)
        total_eeg_power = np.sum(fft_power[eeg_mask])

        # Power at line frequency (±2 Hz band)
        line_mask = (freqs >= self.line_freq - 2) & (freqs <= self.line_freq + 2)
        line_power = np.sum(fft_power[line_mask])

        if total_eeg_power == 0:
            return 0.5, reasons

        line_ratio = line_power / total_eeg_power

        if line_ratio > MAX_LINE_NOISE_RATIO:
            reasons.append(f"Excessive line noise ({self.line_freq}Hz: {line_ratio:.0%} of power)")
            return max(0.0, 1.0 - line_ratio * 2), reasons

        # Score: penalize proportionally
        score = max(0.3, 1.0 - line_ratio)
        return score, reasons

    def _check_emg(self, signal: np.ndarray) -> Tuple[float, List[str]]:
        """Check for muscle artifact contamination (high-frequency noise)."""
        reasons = []
        n = len(signal)
        if n < self.fs:
            return 0.8, reasons

        freqs = np.fft.rfftfreq(n, 1.0 / self.fs)
        fft_power = np.abs(np.fft.rfft(signal)) ** 2

        # EEG-relevant power (1-40 Hz)
        eeg_mask = (freqs >= 1) & (freqs <= 40)
        eeg_power = np.sum(fft_power[eeg_mask])

        # High-frequency power (40-100 Hz) = likely muscle
        emg_mask = (freqs >= 40) & (freqs <= min(100, self.fs / 2))
        emg_power = np.sum(fft_power[emg_mask])

        total = eeg_power + emg_power
        if total == 0:
            return 0.5, reasons

        emg_ratio = emg_power / total

        if emg_ratio > MAX_EMG_RATIO:
            reasons.append(f"Muscle artifact (EMG: {emg_ratio:.0%} of power)")
            return max(0.0, 1.0 - emg_ratio * 1.5), reasons

        score = max(0.3, 1.0 - emg_ratio)
        return score, reasons

    def _check_stationarity(self, signal: np.ndarray) -> Tuple[float, List[str]]:
        """Check if signal statistics are stable (no motion artifacts)."""
        reasons = []
        n = len(signal)
        if n < self.fs * 2:
            return 0.8, reasons  # Need at least 2 seconds

        # Split into quarters and compare statistics
        quarter = n // 4
        quarters = [signal[i * quarter:(i + 1) * quarter] for i in range(4)]

        stds = [np.std(q) for q in quarters]
        means = [np.mean(q) for q in quarters]

        # Coefficient of variation of the standard deviations
        std_cv = np.std(stds) / max(np.mean(stds), 1e-6)

        # Check for sudden jumps in mean (motion artifact)
        mean_jumps = [abs(means[i + 1] - means[i]) for i in range(3)]
        max_jump = max(mean_jumps) if mean_jumps else 0
        jump_ratio = max_jump / max(np.std(signal), 1e-6)

        if std_cv > 1.0:
            reasons.append(f"Non-stationary signal (variance changes {std_cv:.1f}x)")
            return 0.2, reasons

        if jump_ratio > 3.0:
            reasons.append(f"Motion artifact detected (jump={jump_ratio:.1f} SDs)")
            return 0.3, reasons

        # Score
        score = max(0.3, 1.0 - std_cv * 0.5 - jump_ratio * 0.1)
        return min(1.0, score), reasons

    def _check_blinks(self, signal: np.ndarray) -> Tuple[float, List[str]]:
        """Detect eye blinks (large slow transients in frontal channels)."""
        reasons = []
        n = len(signal)
        duration_s = n / self.fs

        if duration_s < 2:
            return 0.8, reasons

        # Eye blinks are large, slow (0.5-2 Hz) positive deflections > 3 SDs
        threshold = np.mean(signal) + 3 * np.std(signal)

        # Simple peak detection
        above_threshold = signal > threshold
        # Count rising edges (blink starts)
        transitions = np.diff(above_threshold.astype(int))
        n_blinks = np.sum(transitions > 0)

        blinks_per_min = (n_blinks / duration_s) * 60

        if blinks_per_min > MAX_BLINK_RATE_PER_MIN:
            reasons.append(f"Excessive eye blinks ({blinks_per_min:.0f}/min)")
            score = max(0.2, 1.0 - (blinks_per_min - MAX_BLINK_RATE_PER_MIN) / 60)
            return score, reasons

        # Normal blink rate: 15-20/min. Fewer blinks = cleaner signal
        if blinks_per_min <= 20:
            score = 1.0
        else:
            score = max(0.5, 1.0 - (blinks_per_min - 20) / 40)

        return score, reasons


    def _check_spectral_flatness(self, signal: np.ndarray) -> Tuple[float, List[str]]:
        """Check spectral flatness — flat spectrum indicates noise, not brain signal."""
        reasons = []
        from scipy.signal import welch
        freqs, psd = welch(signal, fs=self.fs, nperseg=min(len(signal), self.fs * 2))
        psd_positive = psd[psd > 0]
        if len(psd_positive) < 2:
            return 0.5, ["Insufficient spectral data"]

        geo_mean = np.exp(np.mean(np.log(psd_positive)))
        arith_mean = np.mean(psd_positive)
        flatness = geo_mean / (arith_mean + 1e-10)

        if flatness > 0.85:
            reasons.append(f"Spectrum too flat ({flatness:.2f}) — likely noise")
            return 0.2, reasons
        elif flatness > 0.7:
            reasons.append(f"Spectrum moderately flat ({flatness:.2f})")
            return 0.5, reasons
        elif flatness < 0.05:
            reasons.append(f"Spectrum too peaked ({flatness:.3f}) — possible line noise")
            return 0.4, reasons
        return 1.0, reasons

    def _check_alpha_peak(self, signal: np.ndarray) -> Tuple[float, List[str]]:
        """Check for alpha peak presence — indicates real EEG with good contact."""
        reasons = []
        from scipy.signal import welch
        freqs, psd = welch(signal, fs=self.fs, nperseg=min(len(signal), self.fs * 2))

        alpha_mask = (freqs >= 8.0) & (freqs <= 12.0)
        pre_mask = (freqs >= 4.0) & (freqs < 8.0)
        post_mask = (freqs > 12.0) & (freqs <= 20.0)

        if not alpha_mask.any() or not pre_mask.any() or not post_mask.any():
            return 0.5, ["Cannot assess alpha peak"]

        alpha_peak = np.max(psd[alpha_mask])
        neighbor_mean = (np.mean(psd[pre_mask]) + np.mean(psd[post_mask])) / 2

        if neighbor_mean > 0 and alpha_peak > 2.0 * neighbor_mean:
            return 1.0, []  # Clear alpha peak
        elif neighbor_mean > 0 and alpha_peak > 1.3 * neighbor_mean:
            reasons.append("Weak alpha peak detected")
            return 0.7, reasons
        else:
            reasons.append("No alpha peak — verify electrode contact")
            return 0.4, reasons


def gate_analysis(signal: np.ndarray, fs: int = 256, threshold: float = None) -> Dict:
    """Quick helper: check if signal is good enough for analysis.

    Returns:
        Dict with is_usable, quality_score, and reasons.

    Usage:
        quality = gate_analysis(eeg_signal, fs=256)
        if not quality["is_usable"]:
            return {"error": "Signal too noisy", "quality": quality}
        # ... proceed with analysis
    """
    checker = SignalQualityChecker(fs=fs)
    result = checker.check_quality(signal)

    if threshold is not None:
        result["is_usable"] = result["quality_score"] >= threshold

    return result
