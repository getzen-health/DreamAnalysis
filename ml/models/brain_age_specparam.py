"""SpecParam (Spectral Parameterization) for Brain Age Estimation.

Implements a from-scratch version of the FOOOF/specparam algorithm:
  1. Fit aperiodic (1/f) component via linear regression on log-log PSD
  2. Detect oscillatory peaks in the residual above the aperiodic fit
  3. Fit a Gaussian to each peak: center frequency, bandwidth, amplitude
  4. Estimate brain age from aperiodic exponent + alpha peak frequency

Scientific basis:
  - Donoghue et al. (2020, Nat Neurosci): SpecParam method
  - Voytek et al. (2015): aperiodic exponent increases with age
  - Klimesch (1999): alpha peak frequency decreases with age (~0.03 Hz/year)
  - Banville et al. (2024, MIT Press): brain age from 4-channel consumer EEG

DISCLAIMER: Wellness indicator only — not a medical device.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch, find_peaks
from scipy.optimize import curve_fit
from typing import Dict, List, Optional, Tuple


DISCLAIMER = (
    "SpecParam brain age is a research-grade wellness indicator derived from "
    "spectral EEG features. This is not a medical device. Individual variation "
    "is high (~8-year population SD). Consult a healthcare professional for "
    "medical concerns."
)

# ──────────────────────────────────────────────────────────────
# Population normative parameters (from Voytek 2015 + Klimesch 1999)
# ──────────────────────────────────────────────────────────────

# Aperiodic exponent norm: ~1.0 at age 18, increases ~0.012/year
# age_from_exponent = (exponent - 0.85) / 0.012 + 18
_EXPONENT_AT_18 = 0.85
_EXPONENT_SLOPE = 0.012   # per year

# Alpha peak frequency norm: ~10.8 Hz at age 18, decreases ~0.03 Hz/year
# age_from_alpha = (10.8 - alpha_peak) / 0.03 + 18
_ALPHA_PEAK_AT_18 = 10.8
_ALPHA_PEAK_SLOPE = 0.03  # Hz per year

# Feature weights for combined brain age estimate
_WEIGHT_EXPONENT = 0.55
_WEIGHT_ALPHA_PEAK = 0.45

# Peak detection parameters
_MIN_PEAK_HEIGHT = 0.1       # log10 power units above aperiodic baseline
_PEAK_DISTANCE_HZ = 1.5      # minimum Hz between distinct peaks
_GAUSSIAN_WIDTH_MAX = 6.0    # Hz — reject peaks wider than this (likely noise)

# Frequency range for aperiodic fit (Hz) — exclude DC, stay below high EMG
_FIT_FREQ_MIN = 2.0
_FIT_FREQ_MAX = 40.0

# Alpha band for peak search (Hz)
_ALPHA_SEARCH_MIN = 7.0
_ALPHA_SEARCH_MAX = 14.0


# ──────────────────────────────────────────────────────────────
# SpectralParameterizer
# ──────────────────────────────────────────────────────────────

class SpectralParameterizer:
    """Decompose EEG power spectrum into aperiodic + periodic components.

    Algorithm (mirrors FOOOF/specparam):
      1. Compute PSD via Welch method
      2. Fit aperiodic component: log10(PSD) ~ offset - exponent * log10(freq)
         using iterative peak exclusion (fit on full range, mask detected peaks,
         refit — one iteration is sufficient for 4-channel consumer EEG)
      3. Compute residual: log10(PSD) - aperiodic_fit
      4. Detect peaks in residual using scipy.signal.find_peaks
      5. Fit a Gaussian to each peak for precise center_freq / bandwidth / amplitude

    Usage::
        sp = SpectralParameterizer(fs=256)
        result = sp.fit(signal_1d)
        # result["aperiodic"]["exponent"], result["peaks"], ...
    """

    def __init__(self, fs: float = 256.0, nperseg: Optional[int] = None):
        self.fs = fs
        # 4-second window at 256 Hz = 1024 samples → freq resolution 0.25 Hz
        self.nperseg = nperseg or min(1024, int(fs * 4))

    # ── PSD ──────────────────────────────────────────────────

    def _compute_psd(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (freqs, psd) using Welch with 50% overlap."""
        nperseg = min(self.nperseg, len(signal))
        freqs, psd = welch(
            signal,
            fs=self.fs,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            window="hann",
        )
        return freqs, psd

    # ── Aperiodic fit ─────────────────────────────────────────

    def _fit_aperiodic(
        self,
        freqs: np.ndarray,
        log_psd: np.ndarray,
        peak_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float]:
        """Fit log-log linear model to aperiodic component.

        Returns (exponent, offset, r2).
        Excludes frequency bins covered by the peak_mask if provided.
        """
        band_mask = (freqs >= _FIT_FREQ_MIN) & (freqs <= _FIT_FREQ_MAX)
        if peak_mask is not None:
            fit_mask = band_mask & ~peak_mask
        else:
            fit_mask = band_mask

        if fit_mask.sum() < 5:
            fit_mask = band_mask  # fall back to full band

        log_f = np.log10(freqs[fit_mask])
        log_p = log_psd[fit_mask]

        # Linear regression: log_p = offset + slope * log_f
        # slope = -exponent (1/f → negative slope in log-log space)
        coeffs = np.polyfit(log_f, log_p, 1)
        slope, offset = float(coeffs[0]), float(coeffs[1])
        exponent = float(-slope)

        # R² on the full fit range
        all_log_f = np.log10(freqs[band_mask])
        all_log_p = log_psd[band_mask]
        predicted = np.polyval(coeffs, all_log_f)
        ss_res = np.sum((all_log_p - predicted) ** 2)
        ss_tot = np.sum((all_log_p - all_log_p.mean()) ** 2)
        r2 = float(np.clip(1.0 - ss_res / (ss_tot + 1e-10), 0.0, 1.0))

        return exponent, offset, r2

    def _aperiodic_curve(
        self, freqs: np.ndarray, offset: float, exponent: float
    ) -> np.ndarray:
        """Evaluate aperiodic model on given frequencies."""
        return offset - exponent * np.log10(freqs + 1e-10)

    # ── Peak detection ────────────────────────────────────────

    def _detect_peaks(
        self,
        freqs: np.ndarray,
        residual: np.ndarray,
        freq_resolution: float,
    ) -> List[Dict]:
        """Find oscillatory peaks in residual above aperiodic baseline.

        Returns list of dicts with gaussian-fitted peak parameters.
        """
        # Only search in the fitting range
        band_mask = (freqs >= _FIT_FREQ_MIN) & (freqs <= _FIT_FREQ_MAX)
        band_freqs = freqs[band_mask]
        band_residual = residual[band_mask]

        min_distance = max(1, int(_PEAK_DISTANCE_HZ / freq_resolution))

        peak_indices, peak_props = find_peaks(
            band_residual,
            height=_MIN_PEAK_HEIGHT,
            distance=min_distance,
        )

        peaks = []
        for idx in peak_indices:
            center_freq = float(band_freqs[idx])
            amplitude = float(band_residual[idx])

            # Fit Gaussian in a window around the peak for precise parameters
            gaussian_params = self._fit_gaussian(
                band_freqs, band_residual, idx, freq_resolution
            )
            if gaussian_params is not None:
                cf, amp, bw = gaussian_params
                if bw < _GAUSSIAN_WIDTH_MAX and amp >= _MIN_PEAK_HEIGHT:
                    peaks.append({
                        "center_freq": round(float(cf), 3),
                        "amplitude": round(float(amp), 4),
                        "bandwidth": round(float(bw), 3),
                        "band": _classify_peak_band(float(cf)),
                    })
            else:
                # Fallback: use raw peak values with half-power bandwidth estimate
                bw = self._half_power_bandwidth(band_freqs, band_residual, idx)
                peaks.append({
                    "center_freq": round(center_freq, 3),
                    "amplitude": round(amplitude, 4),
                    "bandwidth": round(bw, 3),
                    "band": _classify_peak_band(center_freq),
                })

        return peaks

    def _fit_gaussian(
        self,
        freqs: np.ndarray,
        residual: np.ndarray,
        peak_idx: int,
        freq_resolution: float,
    ) -> Optional[Tuple[float, float, float]]:
        """Fit a Gaussian (center, amplitude, sigma) around peak_idx."""
        # Window: ±6 Hz around the peak, or 12 bins minimum
        window_hz = 6.0
        window_bins = max(6, int(window_hz / freq_resolution))
        lo = max(0, peak_idx - window_bins)
        hi = min(len(freqs) - 1, peak_idx + window_bins)

        x = freqs[lo:hi + 1]
        y = residual[lo:hi + 1]

        if len(x) < 4:
            return None

        # Initial guess: center at peak, amplitude = peak height, sigma = 2 Hz
        p0 = [freqs[peak_idx], residual[peak_idx], 2.0]
        bounds = (
            [freqs[lo], 0.0, 0.1],
            [freqs[hi], residual[peak_idx] * 3, window_hz],
        )

        try:
            popt, _ = curve_fit(_gaussian, x, y, p0=p0, bounds=bounds, maxfev=500)
            center, amplitude, sigma = popt
            bandwidth = float(2.355 * abs(sigma))  # FWHM
            return float(center), float(amplitude), bandwidth
        except (RuntimeError, ValueError):
            return None

    def _half_power_bandwidth(
        self, freqs: np.ndarray, residual: np.ndarray, peak_idx: int
    ) -> float:
        """Estimate bandwidth as half-power (3 dB) width — fallback for Gaussian fit."""
        half_power = residual[peak_idx] / 2.0
        # Walk left
        lo = peak_idx
        while lo > 0 and residual[lo] > half_power:
            lo -= 1
        # Walk right
        hi = peak_idx
        while hi < len(residual) - 1 and residual[hi] > half_power:
            hi += 1
        bw = float(freqs[hi] - freqs[lo])
        return max(bw, 0.5)

    # ── Build peak mask for aperiodic re-fit ─────────────────

    def _peaks_to_mask(
        self,
        freqs: np.ndarray,
        peaks: List[Dict],
        freq_resolution: float,
    ) -> np.ndarray:
        """Create boolean mask over frequency bins covered by detected peaks."""
        mask = np.zeros(len(freqs), dtype=bool)
        for pk in peaks:
            cf = pk["center_freq"]
            bw = max(pk["bandwidth"], 1.0)
            half_bw = bw / 2.0 + freq_resolution
            mask |= (freqs >= cf - half_bw) & (freqs <= cf + half_bw)
        return mask

    # ── Main fit ──────────────────────────────────────────────

    def fit(self, signal: np.ndarray) -> Dict:
        """Decompose a 1-D EEG signal into aperiodic + periodic components.

        Args:
            signal: 1-D numpy array of EEG samples (µV), length ≥ 512 samples.

        Returns:
            dict with keys:
              aperiodic: {exponent, offset, r2, r2_after_peak_removal}
              peaks: list of {center_freq, amplitude, bandwidth, band}
              residual_rms: float — how well the model fits
              alpha_peak: optional dict for the dominant alpha peak
              spectrogram: {freqs, log_psd, aperiodic_fit, residual} as lists
        """
        freqs, psd = self._compute_psd(signal)
        freq_resolution = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.25

        # Guard against zero/negative PSD values before log
        psd = np.maximum(psd, 1e-30)
        log_psd = np.log10(psd)

        # Pass 1: aperiodic fit on full spectrum (no peak removal)
        exp1, off1, r2_1 = self._fit_aperiodic(freqs, log_psd, peak_mask=None)
        aperiodic_1 = self._aperiodic_curve(freqs, off1, exp1)
        residual_1 = log_psd - aperiodic_1

        # Detect candidate peaks on first-pass residual
        peaks = self._detect_peaks(freqs, residual_1, freq_resolution)

        # Pass 2: refit aperiodic with peak bins excluded (one iteration is enough)
        peak_mask = self._peaks_to_mask(freqs, peaks, freq_resolution)
        exp2, off2, r2_2 = self._fit_aperiodic(freqs, log_psd, peak_mask=peak_mask)
        aperiodic_2 = self._aperiodic_curve(freqs, off2, exp2)
        residual_2 = log_psd - aperiodic_2

        # Re-detect peaks on cleaner residual
        peaks = self._detect_peaks(freqs, residual_2, freq_resolution)

        # Identify dominant alpha peak
        alpha_peak = _find_dominant_peak(
            peaks, _ALPHA_SEARCH_MIN, _ALPHA_SEARCH_MAX
        )

        # Band-limited mask for residual RMS
        band_mask = (freqs >= _FIT_FREQ_MIN) & (freqs <= _FIT_FREQ_MAX)
        residual_rms = float(np.sqrt(np.mean(residual_2[band_mask] ** 2)))

        # Build spectrogram arrays for visualization (band-masked only)
        band_freqs = freqs[band_mask].tolist()
        band_log_psd = log_psd[band_mask].tolist()
        band_aperiodic = aperiodic_2[band_mask].tolist()
        band_residual = residual_2[band_mask].tolist()

        return {
            "aperiodic": {
                "exponent": round(exp2, 4),
                "offset": round(off2, 4),
                "r2": round(r2_2, 4),
                "r2_initial": round(r2_1, 4),
            },
            "peaks": peaks,
            "alpha_peak": alpha_peak,
            "residual_rms": round(residual_rms, 4),
            "spectrogram": {
                "freqs": [round(f, 3) for f in band_freqs],
                "log_psd": [round(v, 4) for v in band_log_psd],
                "aperiodic_fit": [round(v, 4) for v in band_aperiodic],
                "residual": [round(v, 4) for v in band_residual],
            },
        }

    def fit_multichannel(self, signals: np.ndarray) -> Dict:
        """Fit SpecParam across multiple channels and return averaged parameters.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) array

        Returns:
            Merged result: aperiodic averaged across channels,
            peaks from each channel listed, alpha_peak averaged.
        """
        if signals.ndim == 1:
            return self.fit(signals)

        n_channels = signals.shape[0]
        channel_results = [self.fit(signals[i]) for i in range(n_channels)]

        # Average aperiodic parameters across channels
        exponents = [r["aperiodic"]["exponent"] for r in channel_results]
        offsets = [r["aperiodic"]["offset"] for r in channel_results]
        r2s = [r["aperiodic"]["r2"] for r in channel_results]

        avg_exponent = float(np.mean(exponents))
        avg_offset = float(np.mean(offsets))
        avg_r2 = float(np.mean(r2s))

        # Alpha peaks from all channels — pick the one with highest amplitude
        alpha_peaks = [
            r["alpha_peak"] for r in channel_results if r["alpha_peak"] is not None
        ]
        best_alpha = None
        if alpha_peaks:
            best_alpha = max(alpha_peaks, key=lambda p: p["amplitude"])

        # Collect all peaks with channel labels
        all_peaks = []
        for ch_idx, r in enumerate(channel_results):
            for pk in r["peaks"]:
                pk_copy = dict(pk)
                pk_copy["channel"] = ch_idx
                all_peaks.append(pk_copy)

        # Average spectrogram from first channel (representative)
        spectrogram = channel_results[0]["spectrogram"]

        return {
            "aperiodic": {
                "exponent": round(avg_exponent, 4),
                "exponent_per_channel": [round(e, 4) for e in exponents],
                "offset": round(avg_offset, 4),
                "r2": round(avg_r2, 4),
            },
            "peaks": all_peaks,
            "alpha_peak": best_alpha,
            "residual_rms": round(
                float(np.mean([r["residual_rms"] for r in channel_results])), 4
            ),
            "spectrogram": spectrogram,
            "n_channels": n_channels,
        }


# ──────────────────────────────────────────────────────────────
# BrainAgeEstimator (SpecParam-based)
# ──────────────────────────────────────────────────────────────

class BrainAgeEstimator:
    """Estimate biological brain age from spectral parameters.

    Uses heuristic normative curves derived from the literature:
      - Aperiodic exponent (Voytek 2015, Banville 2024)
      - Alpha Individual Frequency / IAF (Klimesch 1999)

    No training data or external model needed — works immediately on any
    4-channel Muse 2 EEG signal.

    Brain Age Gap (BAG):
      - BAG > 0: brain appears older than chronological age
      - BAG < 0: brain appears younger (generally healthy)
      - |BAG| < 5 years: within normal population variation
    """

    def __init__(self, fs: float = 256.0):
        self.fs = fs
        self._sp = SpectralParameterizer(fs=fs)

    def _age_from_exponent(self, exponent: float) -> float:
        """Estimate age from aperiodic exponent using linear normative model."""
        # exponent(age) = _EXPONENT_AT_18 + _EXPONENT_SLOPE * (age - 18)
        # → age = (exponent - _EXPONENT_AT_18) / _EXPONENT_SLOPE + 18
        estimated = (exponent - _EXPONENT_AT_18) / _EXPONENT_SLOPE + 18.0
        return float(np.clip(estimated, 10.0, 95.0))

    def _age_from_alpha(self, alpha_freq: float) -> float:
        """Estimate age from alpha peak frequency using linear normative model."""
        # alpha_freq(age) = _ALPHA_PEAK_AT_18 - _ALPHA_PEAK_SLOPE * (age - 18)
        # → age = (_ALPHA_PEAK_AT_18 - alpha_freq) / _ALPHA_PEAK_SLOPE + 18
        estimated = (_ALPHA_PEAK_AT_18 - alpha_freq) / _ALPHA_PEAK_SLOPE + 18.0
        return float(np.clip(estimated, 10.0, 95.0))

    def estimate(
        self,
        signals: np.ndarray,
        chronological_age: Optional[float] = None,
    ) -> Dict:
        """Estimate brain age from EEG signals.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) at self.fs
            chronological_age: actual age in years for BAG calculation

        Returns:
            dict with predicted_age, brain_age_gap, specparam results,
            confidence, and disclaimer.
        """
        # Preprocess each channel before spectral fitting
        try:
            from processing.eeg_processor import preprocess
            if signals.ndim == 1:
                preprocessed = preprocess(signals, self.fs)
            else:
                preprocessed = np.stack(
                    [preprocess(signals[i], self.fs) for i in range(signals.shape[0])]
                )
        except Exception:
            # If preprocessing fails, use raw signal
            preprocessed = signals

        # Run SpecParam
        sp_result = self._sp.fit_multichannel(preprocessed)

        exponent = sp_result["aperiodic"]["exponent"]
        r2 = sp_result["aperiodic"]["r2"]
        alpha_peak = sp_result.get("alpha_peak")

        # Age estimate from aperiodic exponent (primary feature)
        age_from_exp = self._age_from_exponent(exponent)

        # Age estimate from alpha peak frequency (secondary feature)
        age_from_alpha = None
        alpha_cf = None
        if alpha_peak is not None:
            alpha_cf = alpha_peak["center_freq"]
            age_from_alpha = self._age_from_alpha(alpha_cf)

        # Combine estimates
        if age_from_alpha is not None:
            predicted_age = (
                _WEIGHT_EXPONENT * age_from_exp
                + _WEIGHT_ALPHA_PEAK * age_from_alpha
            )
        else:
            # Alpha peak not detected — fall back to exponent only
            predicted_age = age_from_exp
        predicted_age = float(np.clip(predicted_age, 10.0, 95.0))

        # Brain Age Gap
        gap = None
        gap_interpretation = None
        gap_severity = None
        if chronological_age is not None:
            gap = round(predicted_age - chronological_age, 1)
            if gap <= -10:
                gap_interpretation = "Brain appears significantly younger than chronological age."
                gap_severity = "very_young"
            elif gap <= -5:
                gap_interpretation = "Brain appears younger than average for chronological age."
                gap_severity = "young"
            elif gap < 5:
                gap_interpretation = "Brain age is within typical range for chronological age."
                gap_severity = "typical"
            elif gap < 10:
                gap_interpretation = "Brain appears slightly older than average for chronological age."
                gap_severity = "slightly_older"
            else:
                gap_interpretation = "Brain appears older than average for chronological age."
                gap_severity = "older"

        # Percentile among same-age population (population SD ≈ 8 years)
        percentile = None
        if chronological_age is not None and gap is not None:
            try:
                from scipy.stats import norm as _norm
                percentile = int(_norm.cdf(gap, loc=0, scale=8.0) * 100)
                percentile = int(np.clip(percentile, 1, 99))
            except Exception:
                pass

        # Confidence: R² of aperiodic fit + whether alpha peak was found
        base_confidence = float(np.clip(0.30 + 0.50 * r2, 0.30, 0.80))
        confidence = base_confidence + (0.10 if alpha_peak is not None else 0.0)
        confidence = round(float(np.clip(confidence, 0.30, 0.90)), 3)

        result = {
            "predicted_age": round(predicted_age, 1),
            "brain_age_gap": gap,
            "gap_interpretation": gap_interpretation,
            "gap_severity": gap_severity,
            "percentile": percentile,
            "confidence": confidence,
            "method": "specparam",
            "features": {
                "aperiodic_exponent": round(exponent, 4),
                "aperiodic_offset": round(sp_result["aperiodic"]["offset"], 4),
                "aperiodic_r2": round(r2, 4),
                "alpha_peak_freq": round(alpha_cf, 3) if alpha_cf is not None else None,
                "age_from_exponent": round(age_from_exp, 1),
                "age_from_alpha": round(age_from_alpha, 1) if age_from_alpha is not None else None,
            },
            "specparam": sp_result,
            "disclaimer": DISCLAIMER,
            "model_type": "specparam_heuristic",
        }
        return result


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _gaussian(x: np.ndarray, center: float, amplitude: float, sigma: float) -> np.ndarray:
    """1-D Gaussian function for curve fitting."""
    return amplitude * np.exp(-0.5 * ((x - center) / (sigma + 1e-10)) ** 2)


def _classify_peak_band(center_freq: float) -> str:
    """Return EEG band name for a given peak center frequency."""
    if center_freq < 4.0:
        return "delta"
    elif center_freq < 8.0:
        return "theta"
    elif center_freq < 12.0:
        return "alpha"
    elif center_freq < 20.0:
        return "low_beta"
    elif center_freq < 30.0:
        return "high_beta"
    else:
        return "gamma"


def _find_dominant_peak(
    peaks: List[Dict], freq_min: float, freq_max: float
) -> Optional[Dict]:
    """Return the highest-amplitude peak within the specified frequency range."""
    candidates = [
        pk for pk in peaks
        if freq_min <= pk["center_freq"] <= freq_max
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p["amplitude"])


# ──────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────

_estimator_instance: Optional[BrainAgeEstimator] = None


def get_brain_age_specparam(fs: float = 256.0) -> BrainAgeEstimator:
    """Return a module-level BrainAgeEstimator singleton."""
    global _estimator_instance
    if _estimator_instance is None or _estimator_instance.fs != fs:
        _estimator_instance = BrainAgeEstimator(fs=fs)
    return _estimator_instance
