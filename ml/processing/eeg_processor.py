"""EEG Signal Processing Module.

Provides bandpass/notch filtering, frequency band power extraction,
Hjorth parameters, spectral entropy, epoching, and wavelet
time-frequency analysis for EEG signals.
"""

import numpy as np
import pywt
from scipy import signal as scipy_signal
from scipy.stats import entropy
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from processing.e_asr import EmbeddedASR

# NumPy 2.0 renamed np.trapz → np.trapezoid; 1.x only has np.trapz
# Use the safe two-step lookup to avoid AttributeError on either version
_trapezoid = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)


# EEG frequency band definitions (Hz)
# low_alpha (8-10 Hz): general alertness, tonic arousal (Klimesch 1999)
# high_alpha (10-12 Hz): task-specific cortical processing, semantic memory,
#   MORE emotion-specific than full-band alpha (Bazanova & Vernon 2014,
#   2025 Scientific Reports). High-alpha suppression indexes emotional engagement.
# low_beta (12-20 Hz): active cognition, focus, motor planning
# high_beta (20-30 Hz): anxiety, stress, fear — strongest marker for fearful/anxious states
# gamma upper bound: capped at 50 Hz to match the preprocessing bandpass filter (1-50 Hz).
# The old 100 Hz upper bound included a dead zone (50-100 Hz) where power is zero after
# filtering, artificially diluting the gamma power estimate during Welch PSD integration.
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "low_alpha": (8.0, 10.0),
    "high_alpha": (10.0, 12.0),
    "beta": (12.0, 30.0),
    "low_beta": (12.0, 20.0),
    "high_beta": (20.0, 30.0),
    "gamma": (30.0, 50.0),
}


def bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5
) -> np.ndarray:
    """Apply Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = scipy_signal.butter(order, [low, high], btype="band")
    padlen = 3 * max(len(a), len(b)) - 1
    if data.shape[-1] <= padlen:
        return data  # signal too short to filter — return as-is
    return scipy_signal.filtfilt(b, a, data, axis=-1)


def notch_filter(data: np.ndarray, freq: float, fs: float, Q: float = 30.0) -> np.ndarray:
    """Apply notch filter to remove powerline interference."""
    nyq = 0.5 * fs
    w0 = freq / nyq
    if w0 >= 1.0:
        return data
    b, a = scipy_signal.iirnotch(w0, Q)
    padlen = 3 * max(len(a), len(b)) - 1
    if data.shape[-1] <= padlen:
        return data  # signal too short to filter — return as-is
    return scipy_signal.filtfilt(b, a, data, axis=-1)


def rereference_to_mastoid(
    signals: np.ndarray,
    left_mastoid_ch: int = 0,   # TP9
    right_mastoid_ch: int = 3,  # TP10
) -> np.ndarray:
    """Re-reference EEG to linked mastoid average (TP9 + TP10).

    Corrects the Muse 2 Fpz reference contamination that artificially suppresses
    AF7/AF8 amplitude. After re-referencing, FAA and DASM values become more
    accurate and comparable to the published FAA literature.

    BrainFlow Muse 2 defaults: left_mastoid_ch=0 (TP9), right_mastoid_ch=3 (TP10).
    Call BEFORE computing FAA or DASM/RASM.

    Args:
        signals: 2D array (n_channels, n_samples).
        left_mastoid_ch: Index of left mastoid channel (TP9 = 0 for Muse 2).
        right_mastoid_ch: Index of right mastoid channel (TP10 = 3 for Muse 2).

    Returns:
        Re-referenced signals (same shape). Mastoid channels become ~0.
    """
    if signals.ndim != 2 or signals.shape[0] <= max(left_mastoid_ch, right_mastoid_ch):
        return signals
    mastoid_ref = (signals[left_mastoid_ch] + signals[right_mastoid_ch]) / 2.0
    return signals - mastoid_ref[np.newaxis, :]


def _sanitize_nan(signal: np.ndarray) -> np.ndarray:
    """Replace NaN/inf values with linear interpolation (or zeros if all-NaN).

    BrainFlow Bluetooth packet drops inject NaN samples into the EEG stream.
    scipy.signal.filtfilt propagates NaN across the entire output, so these
    must be repaired BEFORE any filtering.  Linear interpolation is the
    standard EEG approach for short gaps (a few samples).

    Args:
        signal: 1D EEG array that may contain NaN or inf values.

    Returns:
        Signal with NaN/inf replaced.  Returns zeros if all values are bad.
    """
    bad = ~np.isfinite(signal)
    if not np.any(bad):
        return signal                       # fast path: nothing to fix
    signal = signal.copy()                  # don't mutate caller's array
    if np.all(bad):
        signal[:] = 0.0                     # entire signal is bad
        return signal
    good_idx = np.where(~bad)[0]
    signal[bad] = np.interp(
        np.where(bad)[0], good_idx, signal[good_idx]
    )
    return signal


def wavelet_denoise_channel(
    signal: np.ndarray, fs: int = 256, wavelet: str = "db4", level: int = 4
) -> np.ndarray:
    """Single-channel wavelet denoising for consumer EEG.

    Works per-channel -- suitable for Muse 2's 4 channels (no ICA needed).

    Method: DWT decomposition -> threshold detail coefficients -> reconstruct.
    Uses universal threshold (VisuShrink) with soft thresholding.

    Args:
        signal: 1D EEG signal.
        fs: Sample rate (256 Hz for Muse).
        wavelet: Wavelet family ('db4' is standard for EEG).
        level: Decomposition level (4 for 256 Hz captures delta through gamma).

    Returns:
        Denoised signal, same length as input.
    """
    # Sanitize NaN/inf before processing
    signal = _sanitize_nan(signal)

    n = len(signal)

    # Edge case: all-zero or constant signal
    if n == 0 or np.std(signal) < 1e-12:
        return signal.copy()

    # Edge case: signal too short for the requested decomposition
    wav = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(n, wav.dec_len)
    if max_level < 1:
        # Cannot decompose at all -- return as-is
        return signal.copy()
    use_level = min(level, max_level)

    # DWT decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=use_level)

    # Estimate noise sigma from finest detail coefficients using MAD
    # MAD = median(|d|) / 0.6745  (robust noise estimator)
    d1 = coeffs[-1]
    mad = np.median(np.abs(d1))
    sigma = mad / 0.6745 if mad > 0 else 1e-10

    # Universal (VisuShrink) threshold
    threshold = sigma * np.sqrt(2 * np.log(n))

    # Soft-threshold all detail coefficients (index 1..end); preserve approximation (index 0)
    denoised_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(detail, value=threshold, mode="soft"))

    # Reconstruct
    reconstructed = pywt.waverec(denoised_coeffs, wavelet)

    # waverec can produce an output 1 sample longer than the input due to
    # even/odd length rounding; trim to match original length
    return reconstructed[:n]


def adaptive_blink_filter(
    signal: np.ndarray, fs: int = 256, threshold_uv: float = 75.0
) -> np.ndarray:
    """Detect and interpolate blink artifacts using Savitzky-Golay reference.

    For frontal channels (AF7/AF8) where blinks are strongest.
    No separate EOG reference electrode needed.

    Method:
        1. Compute a slow envelope using Savitzky-Golay filter (~0.5s window).
        2. Detect blinks where |signal - envelope| > threshold.
        3. Interpolate blink segments using cubic interpolation.
        4. Return cleaned signal.

    Args:
        signal: 1D EEG signal.
        fs: Sample rate (256 Hz for Muse).
        threshold_uv: Amplitude threshold in microvolts for blink detection.

    Returns:
        Cleaned signal, same length as input.
    """
    # Sanitize NaN/inf before processing
    signal = _sanitize_nan(signal)

    n = len(signal)
    out = signal.copy()

    # Edge case: too short for Savitzky-Golay (needs odd window >= polyorder + 2)
    window_samples = int(0.5 * fs)  # ~0.5 seconds
    if window_samples % 2 == 0:
        window_samples += 1  # must be odd
    polyorder = 3

    if n < window_samples or n < polyorder + 2:
        return out

    # Compute slow envelope via Savitzky-Golay
    envelope = scipy_signal.savgol_filter(out, window_length=window_samples, polyorder=polyorder)

    # Detect blink samples: where residual exceeds threshold
    residual = np.abs(out - envelope)
    blink_mask = residual > threshold_uv

    if not np.any(blink_mask):
        return out

    # Expand blink regions slightly (20 ms margin on each side) to catch tails
    margin = max(1, int(0.02 * fs))
    expanded_mask = blink_mask.copy()
    for i in range(n):
        if blink_mask[i]:
            lo = max(0, i - margin)
            hi = min(n, i + margin + 1)
            expanded_mask[lo:hi] = True

    # Interpolate blink segments using cubic interpolation from clean samples
    clean_idx = np.where(~expanded_mask)[0]
    if len(clean_idx) < 2:
        # Almost entire signal is artifact -- fall back to envelope
        return envelope

    blink_idx = np.where(expanded_mask)[0]
    out[blink_idx] = np.interp(blink_idx, clean_idx, out[clean_idx])

    return out


def preprocess(
    raw_eeg: np.ndarray,
    fs: float = 256.0,
    clean_artifacts: bool = False,
    easr_instance: Optional["EmbeddedASR"] = None,
) -> np.ndarray:
    """Full preprocessing pipeline: bandpass 1-50Hz + notch 50/60Hz.

    Args:
        raw_eeg: 1D EEG signal array.
        fs: Sampling frequency in Hz.
        clean_artifacts: If True, apply E-ASR artifact cleaning after filtering.
            This preserves data instead of discarding epochs. Off by default
            to avoid breaking existing callers.
        easr_instance: Optional pre-configured EmbeddedASR instance. If None
            and clean_artifacts is True, a default instance is created (without
            baseline fitting, which uses the simple interpolation fallback).

    Returns:
        Filtered (and optionally artifact-cleaned) signal.
    """
    raw_eeg = _sanitize_nan(raw_eeg)
    filtered = bandpass_filter(raw_eeg, 1.0, 50.0, fs)
    filtered = notch_filter(filtered, 50.0, fs)
    filtered = notch_filter(filtered, 60.0, fs)

    # Wavelet denoising + blink artifact removal (per-channel, no ICA needed)
    filtered = wavelet_denoise_channel(filtered, fs=int(fs))
    filtered = adaptive_blink_filter(filtered, fs=int(fs))

    if clean_artifacts:
        filtered = clean_artifacts_easr(filtered, fs=fs, easr_instance=easr_instance)

    return filtered


def clean_artifacts_easr(
    signal: np.ndarray,
    fs: float = 256.0,
    easr_instance: Optional["EmbeddedASR"] = None,
) -> np.ndarray:
    """Clean artifacts using E-ASR (preserves data instead of discarding).

    E-ASR (Embedded Artifact Subspace Reconstruction) adapts the standard
    multi-channel ASR algorithm to work on single channels via time-delay
    embedding. Artifact components are projected back to baseline variance
    levels instead of being zeroed out or having the entire epoch discarded.

    Falls back to simple amplitude-threshold interpolation if the E-ASR
    instance has no fitted baseline.

    Args:
        signal: 1D EEG signal array.
        fs: Sampling frequency in Hz.
        easr_instance: Pre-configured EmbeddedASR instance. If None, a
            default (unfitted) instance is created, which will use the
            simple interpolation fallback.

    Returns:
        Cleaned signal (same length as input).
    """
    from processing.e_asr import EmbeddedASR

    if easr_instance is None:
        easr_instance = EmbeddedASR(fs=int(fs))

    return easr_instance.clean(signal)


# Cached instances for ML-based preprocessing (lazy loaded)
_denoiser_instance = None
_artifact_classifier_instance = None


def _get_denoiser():
    """Lazy-load the denoising autoencoder if available."""
    global _denoiser_instance
    if _denoiser_instance is not None:
        return _denoiser_instance

    from pathlib import Path
    model_path = Path("models/saved/denoiser_model.pt")
    if model_path.exists():
        try:
            from models.denoising_autoencoder import EEGDenoiser
            _denoiser_instance = EEGDenoiser(fs=256.0, model_path=str(model_path))
            return _denoiser_instance
        except Exception:
            pass
    return None


def _get_artifact_classifier():
    """Lazy-load the artifact classifier if available."""
    global _artifact_classifier_instance
    if _artifact_classifier_instance is not None:
        return _artifact_classifier_instance

    from pathlib import Path
    model_path = Path("models/saved/artifact_classifier_model.pkl")
    if model_path.exists():
        try:
            from models.artifact_classifier import ArtifactClassifier
            _artifact_classifier_instance = ArtifactClassifier(model_path=str(model_path))
            return _artifact_classifier_instance
        except Exception:
            pass
    return None


def preprocess_robust(
    raw_eeg: np.ndarray,
    fs: float = 256.0,
    use_denoiser: bool = True,
    use_artifact_classifier: bool = True,
    reject_artifacts: bool = False,
) -> np.ndarray:
    """Enhanced preprocessing with ML-based denoising and artifact handling.

    Pipeline:
        1. Classical bandpass + notch filtering (always)
        2. ML denoiser (if trained model available)
        3. Artifact classification (if trained model available)
        4. Optional artifact rejection (zeros out artifact segments)

    Falls back to classical preprocessing if ML models not available.

    Args:
        raw_eeg: Raw EEG signal (1D array).
        fs: Sampling frequency.
        use_denoiser: Try to use the trained denoising autoencoder.
        use_artifact_classifier: Try to use the artifact classifier.
        reject_artifacts: Zero out detected artifact segments.

    Returns:
        Preprocessed (and optionally denoised) signal.
    """
    # Step 1: Classical filtering (always applied)
    filtered = preprocess(raw_eeg, fs)

    # Step 2: ML denoiser
    if use_denoiser:
        denoiser = _get_denoiser()
        if denoiser is not None:
            try:
                filtered = denoiser.denoise(filtered)
            except Exception:
                pass  # Fall back to classical-only

    # Step 3: Artifact handling
    if use_artifact_classifier and reject_artifacts:
        classifier = _get_artifact_classifier()
        if classifier is not None:
            try:
                mask = classifier.get_clean_mask(filtered, fs, window_sec=1.0)
                filtered = filtered * mask  # Zero out artifact segments
            except Exception:
                pass

    return filtered


def estimate_iaf(
    signal: np.ndarray, fs: float = 256.0, search_range: tuple = (7.0, 14.0)
) -> float:
    """Estimate Individual Alpha Frequency from EEG signal.

    Finds the spectral peak in the 7-14 Hz range using Welch PSD.
    Typical IAF is 9-11 Hz for healthy adults.

    Args:
        signal: 1D EEG signal (preprocessed).
        fs: Sampling rate in Hz.
        search_range: Hz range to search for alpha peak.

    Returns:
        IAF in Hz (default 10.0 if no clear peak found or signal too short).
    """
    nperseg = min(int(fs * 2), len(signal))
    if nperseg < int(fs * 0.5):
        return 10.0  # signal too short for reliable PSD
    freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= search_range[0]) & (freqs <= search_range[1])
    if not np.any(mask):
        return 10.0
    alpha_psd = psd[mask]
    alpha_freqs = freqs[mask]
    return float(alpha_freqs[np.argmax(alpha_psd)])


def get_personalized_bands(iaf: float) -> Dict[str, tuple]:
    """Return frequency band boundaries personalized to user's IAF.

    Standard bands assume alpha = 8-12 Hz (IAF = 10 Hz).
    This shifts theta/alpha/beta/low_beta boundaries relative to the
    user's actual IAF while keeping delta, high_beta, and gamma fixed.

    Args:
        iaf: Individual Alpha Frequency in Hz.

    Returns:
        Dict mapping band names to (low_hz, high_hz) tuples.
    """
    return {
        "delta": (0.5, 4.0),
        "theta": (4.0, iaf - 2.0),
        "alpha": (iaf - 2.0, iaf + 2.0),
        "beta": (iaf + 2.0, 30.0),
        "low_beta": (iaf + 2.0, 20.0),
        "high_beta": (20.0, 30.0),
        "gamma": (30.0, 50.0),
    }


def _compute_psd(eeg: np.ndarray, fs: float = 256.0):
    """Compute Welch PSD once, to be shared across feature extractors."""
    return scipy_signal.welch(eeg, fs=fs, nperseg=min(len(eeg), int(fs * 2)))


def extract_band_powers(eeg: np.ndarray, fs: float = 256.0,
                        psd_cache: tuple = None,
                        bands: Dict[str, tuple] = None) -> Dict[str, float]:
    """Extract power spectral density for each EEG frequency band.

    Args:
        eeg: 1D EEG signal.
        fs: Sampling frequency.
        psd_cache: Optional pre-computed (freqs, psd) tuple to avoid re-computing Welch.
        bands: Optional custom band definitions (e.g. from get_personalized_bands).
               Falls back to module-level BANDS if not provided.
    """
    if psd_cache is not None:
        freqs, psd = psd_cache
    else:
        freqs, psd = _compute_psd(eeg, fs)

    total_power = _trapezoid(psd, freqs)
    if total_power == 0:
        total_power = 1e-10

    use_bands = bands if bands is not None else BANDS
    band_powers = {}
    for band_name, (low, high) in use_bands.items():
        mask = (freqs >= low) & (freqs <= high)
        band_power = _trapezoid(psd[mask], freqs[mask])
        band_powers[band_name] = float(band_power / total_power)

    return band_powers


def extract_band_powers_log(eeg: np.ndarray, fs: float = 256.0,
                             psd_cache: tuple = None) -> Dict[str, float]:
    """Extract log-transformed absolute band powers (µV²/Hz scale).

    Log-domain features follow a more normal distribution than raw power,
    improving classifier performance and reducing outlier sensitivity.

    Returns:
        Dict with keys like 'log_bp_alpha', 'log_bp_beta', etc.
    """
    if psd_cache is not None:
        freqs, psd = psd_cache
    else:
        freqs, psd = _compute_psd(eeg, fs)

    result = {}
    for band_name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        band_power = _trapezoid(psd[mask], freqs[mask]) if mask.any() else 0.0
        result[f"log_bp_{band_name}"] = float(np.log(max(band_power, 1e-12)))

    return result


def compute_hjorth_parameters(eeg: np.ndarray) -> Dict[str, float]:
    """Compute Hjorth activity, mobility, and complexity."""
    diff1 = np.diff(eeg)
    diff2 = np.diff(diff1)

    activity = np.var(eeg)
    mobility_var = np.var(diff1)
    if activity == 0:
        activity = 1e-10

    mobility = np.sqrt(mobility_var / activity)

    if mobility_var == 0:
        complexity = 0.0
    else:
        complexity = np.sqrt(np.var(diff2) / mobility_var) / mobility if mobility > 0 else 0.0

    return {
        "activity": float(activity),
        "mobility": float(mobility),
        "complexity": float(complexity),
    }


def compute_band_hjorth_mobility(
    eeg: np.ndarray,
    fs: float = 256.0,
    bands: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute Hjorth mobility on band-filtered signals.

    Standard Hjorth mobility is computed on the broadband signal.  This function
    first bandpass-filters the EEG into each requested frequency band, then
    computes Hjorth mobility on the filtered signal.  The result captures
    *how much* dynamic variation exists within a specific band — a richer
    feature than spectral power alone.

    Evidence: Hjorth mobility in the beta rhythm achieves 83.33% accuracy
    (AUC 0.904) on the SEED emotion dataset, the best single feature among
    18 examined (SVM, leave-one-subject-out).

    Args:
        eeg:   1-D array of EEG samples.
        fs:    Sampling frequency in Hz.
        bands: Which bands to compute.  Defaults to ["theta", "alpha", "beta"].
               Names must exist in the module-level ``BANDS`` dict.

    Returns:
        Dict mapping ``"<band>_mobility"`` to the Hjorth mobility value
        computed on the band-filtered signal.  Values are >= 0.
    """
    if bands is None:
        bands = ["theta", "alpha", "beta"]

    result: Dict[str, float] = {}
    for band_name in bands:
        low, high = BANDS.get(band_name, (8.0, 12.0))
        filtered = bandpass_filter(eeg, low, high, fs, order=4)

        # If the signal was too short for the filter, filtered == eeg (unfiltered).
        # Compute Hjorth on whatever we got — graceful degradation.
        activity = float(np.var(filtered))
        if activity < 1e-15:
            result[f"{band_name}_mobility"] = 0.0
            continue

        diff1 = np.diff(filtered)
        mobility = float(np.sqrt(np.var(diff1) / activity))
        result[f"{band_name}_mobility"] = mobility

    return result


def compute_hjorth_mobility_ratio(
    signals: np.ndarray,
    fs: float = 256.0,
    frontal_chs: Optional[List[int]] = None,
    temporal_chs: Optional[List[int]] = None,
    bands: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute frontal/temporal Hjorth mobility ratio per frequency band.

    For a 4-channel Muse 2 layout (TP9, AF7, AF8, TP10), this computes
    mean Hjorth mobility in band-filtered frontal channels divided by
    mean mobility in temporal channels.  A ratio > 1 indicates the
    frontal cortex is more dynamically active in that band than temporal.

    This ratio indicates *where* neural processing is happening.
    High frontal-to-temporal beta mobility → focused frontal engagement
    (attention, cognitive load, working memory).

    Args:
        signals:      2-D array (n_channels, n_samples).
        fs:           Sampling frequency in Hz.
        frontal_chs:  Indices of frontal channels.  Default [1, 2] (AF7, AF8).
        temporal_chs: Indices of temporal channels.  Default [0, 3] (TP9, TP10).
        bands:        Bands to compute ratio for.  Default ["beta"].

    Returns:
        Dict mapping ``"<band>_mobility_ratio_ft"`` to the ratio value.
        Returns empty dict if fewer than 2 channels.
    """
    if signals.ndim != 2 or signals.shape[0] < 2:
        return {}

    n_ch = signals.shape[0]
    if frontal_chs is None:
        frontal_chs = [1, 2] if n_ch >= 3 else [0]
    if temporal_chs is None:
        temporal_chs = [0, 3] if n_ch >= 4 else [n_ch - 1]
    if bands is None:
        bands = ["beta"]

    # Clamp channel indices to valid range
    frontal_chs = [min(c, n_ch - 1) for c in frontal_chs]
    temporal_chs = [min(c, n_ch - 1) for c in temporal_chs]

    result: Dict[str, float] = {}
    for band_name in bands:
        # Compute per-channel band mobility, then average per region
        frontal_mobs = []
        for ch in frontal_chs:
            mob = compute_band_hjorth_mobility(signals[ch], fs, bands=[band_name])
            frontal_mobs.append(mob.get(f"{band_name}_mobility", 0.0))

        temporal_mobs = []
        for ch in temporal_chs:
            mob = compute_band_hjorth_mobility(signals[ch], fs, bands=[band_name])
            temporal_mobs.append(mob.get(f"{band_name}_mobility", 0.0))

        mean_frontal = float(np.mean(frontal_mobs)) if frontal_mobs else 0.0
        mean_temporal = float(np.mean(temporal_mobs)) if temporal_mobs else 0.0

        # Ratio with epsilon to avoid division by zero
        ratio = mean_frontal / max(mean_temporal, 1e-10)
        result[f"{band_name}_mobility_ratio_ft"] = round(ratio, 4)

    return result


def spectral_entropy(eeg: np.ndarray, fs: float = 256.0,
                     psd_cache: tuple = None) -> float:
    """Compute normalized spectral entropy of the EEG signal."""
    if psd_cache is not None:
        freqs, psd = psd_cache
    else:
        freqs, psd = _compute_psd(eeg, fs)
    psd_norm = psd / (psd.sum() + 1e-10)
    se = entropy(psd_norm)
    max_entropy = np.log(len(psd_norm)) if len(psd_norm) > 0 else 1.0
    return float(se / max_entropy) if max_entropy > 0 else 0.0


def differential_entropy(eeg: np.ndarray, fs: float = 256.0) -> Dict[str, float]:
    """Compute differential entropy for each frequency band."""
    de = {}
    for band_name, (low, high) in BANDS.items():
        band_signal = bandpass_filter(eeg, low, high, fs)
        var = np.var(band_signal)
        de[band_name] = float(0.5 * np.log(2 * np.pi * np.e * max(var, 1e-10)))
    return de


def extract_features(eeg: np.ndarray, fs: float = 256.0) -> Dict[str, float]:
    """Extract comprehensive feature set from EEG signal.

    Computes the Welch PSD once and passes it to all sub-extractors to
    avoid redundant FFT computation (~30% speedup on repeated calls).
    """
    features = {}

    # Compute PSD once and share across extractors
    psd_cache = _compute_psd(eeg, fs)

    band_powers = extract_band_powers(eeg, fs, psd_cache=psd_cache)
    for k, v in band_powers.items():
        features[f"band_power_{k}"] = v

    # Log-domain band powers — better normality for ML classifiers
    log_powers = extract_band_powers_log(eeg, fs, psd_cache=psd_cache)
    features.update(log_powers)

    hjorth = compute_hjorth_parameters(eeg)
    for k, v in hjorth.items():
        features[f"hjorth_{k}"] = v

    features["spectral_entropy"] = spectral_entropy(eeg, fs, psd_cache=psd_cache)

    de = differential_entropy(eeg, fs)
    for k, v in de.items():
        features[f"de_{k}"] = v

    # Core ratios useful for emotion/state detection
    alpha = band_powers.get("alpha", 0)
    beta = band_powers.get("beta", 0)
    theta = band_powers.get("theta", 0)
    delta = band_powers.get("delta", 0)
    high_beta = band_powers.get("high_beta", 0)
    features["alpha_beta_ratio"] = alpha / max(beta, 1e-10)
    features["theta_beta_ratio"] = theta / max(beta, 1e-10)
    features["alpha_theta_ratio"] = alpha / max(theta, 1e-10)

    # Additional discriminative ratios
    features["delta_theta_ratio"] = delta / max(theta, 1e-10)   # drowsiness marker
    features["high_beta_frac"] = high_beta / max(beta, 1e-10)   # anxiety/fear marker (fraction of beta that is 20-30 Hz)
    features["theta_alpha_ratio"] = theta / max(alpha, 1e-10)   # meditation vs drowsiness

    return features


def extract_eye_movement_features(
    signal: np.ndarray,
    fs: float = 256.0,
    amplitude_threshold_uv: float = 50.0,
    window_ms: float = 300.0,
) -> Dict[str, float]:
    """Extract eye movement features from frontal EEG for REM detection.

    Detects rapid eye movements (saccades) from frontal EEG channels by
    identifying rapid amplitude deflections that exceed a threshold within
    a short time window. These eye-movement artifacts are informative for
    REM sleep detection since REMs are a hallmark of the REM stage.

    Supports two modes:
    - Single-channel (1D): high-pass filter + peak detection on one frontal channel.
    - Multichannel (2D, 2 rows = AF7 + AF8): uses AF7-AF8 difference for
      lateral eye movement detection (horizontal EOG proxy).

    Args:
        signal: 1D EEG array (single frontal channel) or 2D array of shape
                (2, n_samples) for AF7 + AF8 multichannel mode.
        fs: Sampling frequency in Hz.
        amplitude_threshold_uv: Minimum amplitude change (uV) to count as a
                                 saccade. Default 50 uV.
        window_ms: Maximum duration (ms) of a saccade deflection. Default 300 ms.

    Returns:
        Dict with:
            saccade_rate: Saccades per second (Hz).
            avg_saccade_amplitude: Mean absolute amplitude of detected saccades (uV).
            eye_movement_index: Composite index = saccade_rate * avg_saccade_amplitude / 100.
    """
    _zero = {"saccade_rate": 0.0, "avg_saccade_amplitude": 0.0, "eye_movement_index": 0.0}

    # Handle empty signal
    if signal.size == 0:
        return _zero

    # Determine working signal
    if signal.ndim == 2:
        # Multichannel: AF7 - AF8 difference (lateral eye movement proxy)
        if signal.shape[0] < 2 or signal.shape[1] == 0:
            return _zero
        working = signal[0] - signal[1]  # AF7 - AF8
    elif signal.ndim == 1:
        working = signal.copy()
    else:
        return _zero

    n_samples = len(working)
    if n_samples < 4:
        return _zero

    duration_sec = n_samples / fs
    if duration_sec <= 0:
        return _zero

    # High-pass filter at 0.5 Hz to isolate eye movement artifacts
    # (removes slow DC drift while preserving saccade waveforms)
    nyq = 0.5 * fs
    hp_freq = 0.5 / nyq
    if hp_freq < 0.999 and n_samples > 18:  # Need enough samples for filtfilt
        try:
            b, a = scipy_signal.butter(2, hp_freq, btype="high")
            padlen = 3 * max(len(a), len(b)) - 1
            if n_samples > padlen:
                working = scipy_signal.filtfilt(b, a, working)
        except Exception:
            pass  # If filtering fails, use unfiltered signal

    # Window size in samples
    window_samples = max(1, int((window_ms / 1000.0) * fs))

    # Detect rapid deflections: look for amplitude changes > threshold within window
    saccade_amplitudes = []
    i = 0
    while i < n_samples - window_samples:
        segment = working[i:i + window_samples]
        seg_range = np.max(segment) - np.min(segment)

        if seg_range >= amplitude_threshold_uv:
            saccade_amplitudes.append(float(seg_range))
            # Skip past this saccade to avoid double-counting
            i += window_samples
        else:
            i += 1

    n_saccades = len(saccade_amplitudes)
    saccade_rate = float(n_saccades / duration_sec)
    avg_amplitude = float(np.mean(saccade_amplitudes)) if n_saccades > 0 else 0.0
    eye_movement_index = float(saccade_rate * avg_amplitude / 100.0)

    return {
        "saccade_rate": saccade_rate,
        "avg_saccade_amplitude": avg_amplitude,
        "eye_movement_index": eye_movement_index,
    }


def compute_frontal_asymmetry(
    signals: np.ndarray, fs: float = 256.0,
    left_ch: int = 0, right_ch: int = 1
) -> Dict[str, float]:
    """Compute frontal alpha asymmetry (FAA) for 2+ channel EEG.

    FAA = log(right_alpha) - log(left_alpha)
    Positive FAA → approach motivation, positive affect (Davidson, 1992)
    Negative FAA → withdrawal motivation, negative affect / depression risk

    BrainFlow Muse 2 channel order (board_id 22/38):
        ch0 = TP9  (left temporal)
        ch1 = AF7  (left frontal)   ← left_ch default should be 1
        ch2 = AF8  (right frontal)  ← right_ch default should be 2
        ch3 = TP10 (right temporal)

    Callers must pass left_ch=1, right_ch=2 for Muse 2 data.
    The function defaults (0, 1) are kept for backwards compatibility with
    single-channel or reordered data — callers are responsible for correct indices.

    Args:
        signals: 2D array (n_channels, n_samples)
        fs: Sampling frequency
        left_ch: Index of left-frontal channel (AF7 = 1 for Muse 2)
        right_ch: Index of right-frontal channel (AF8 = 2 for Muse 2)

    Returns:
        Dict with 'frontal_asymmetry', 'temporal_asymmetry', 'asymmetry_valence'
    """
    n_channels = signals.shape[0]
    if n_channels < 2:
        return {"frontal_asymmetry": 0.0, "temporal_asymmetry": 0.0, "asymmetry_valence": 0.0}

    def _alpha_power(ch: int) -> float:
        """Alpha band power for a single channel."""
        proc = preprocess(signals[ch], fs)
        bp = extract_band_powers(proc, fs)
        return max(bp.get("alpha", 1e-12), 1e-12)

    r_alpha = _alpha_power(right_ch)
    l_alpha = _alpha_power(left_ch)
    frontal_asymmetry = float(np.log(r_alpha) - np.log(l_alpha))

    # Temporal asymmetry (TP9 vs TP10) if 4 channels available
    temporal_asymmetry = 0.0
    if n_channels >= 4:
        # For Muse 2: ch0=TP9 (left temporal), ch3=TP10 (right temporal)
        r_temporal = _alpha_power(3)   # TP10 right temporal
        l_temporal = _alpha_power(0)   # TP9  left temporal (was ch2=AF8, incorrect)
        temporal_asymmetry = float(np.log(r_temporal) - np.log(l_temporal))

    # Map asymmetry to valence signal (-1 to 1)
    # FAA > 0 → positive valence; FAA < 0 → negative valence
    asymmetry_valence = float(np.clip(np.tanh(frontal_asymmetry * 2.0), -1, 1))

    return {
        "frontal_asymmetry": round(frontal_asymmetry, 4),
        "temporal_asymmetry": round(temporal_asymmetry, 4),
        "asymmetry_valence": round(asymmetry_valence, 4),
    }


# Frequency bands used for DASM/RASM (5 core emotion bands, SJTU BCMI Lab convention)
_DASM_RASM_BANDS = ["delta", "theta", "alpha", "low_alpha", "high_alpha", "beta", "gamma"]


def compute_dasm_rasm(
    signals: np.ndarray, fs: float = 256.0,
    left_ch: int = 1, right_ch: int = 2
) -> Dict[str, float]:
    """Compute DASM and RASM inter-hemispheric asymmetry features across all bands.

    DASM (Differential Asymmetry): DE(right_ch) - DE(left_ch) per frequency band.
    RASM (Rational Asymmetry):     var(right_ch) / var(left_ch) per frequency band.

    Extends FAA from alpha-only to ALL five frequency bands. Introduced by the
    SJTU BCMI Lab (Zheng & Lu, 2015, SEED dataset). In the recommended 31-feature
    compact vector for Muse 2:
        5 DE × 4 channels  = 20 features
        5 DASM             =  5 features  ← this function
        5 RASM             =  5 features  ← this function
        1 FAA              =  1 feature   ← compute_frontal_asymmetry()
        Total: 31 features

    BrainFlow Muse 2 channel order:
        ch0 = TP9  (left temporal)
        ch1 = AF7  (left frontal)   ← left_ch default
        ch2 = AF8  (right frontal)  ← right_ch default
        ch3 = TP10 (right temporal)

    Args:
        signals: 2D array (n_channels, n_samples).
        fs: Sampling frequency (Hz).
        left_ch: Left-frontal channel index (AF7 = ch1 for Muse 2).
        right_ch: Right-frontal channel index (AF8 = ch2 for Muse 2).

    Returns:
        Dict with 10 keys:
            dasm_{band}  — DE(right) - DE(left) — signed; positive = right-dominant
            rasm_{band}  — var(right) / var(left) — ratio; 1.0 = symmetric
        Returns empty dict if signals lack required channels.
    """
    if signals.ndim < 2 or signals.shape[0] <= max(left_ch, right_ch):
        return {}

    left_signal = preprocess(signals[left_ch], fs)
    right_signal = preprocess(signals[right_ch], fs)

    de_left = differential_entropy(left_signal, fs)
    de_right = differential_entropy(right_signal, fs)

    result = {}
    for band in _DASM_RASM_BANDS:
        low, high = BANDS[band]

        # DASM: difference of differential entropies (signed asymmetry).
        # Positive → right-frontal more active → approach motivation / positive valence.
        result[f"dasm_{band}"] = float(de_right.get(band, 0.0) - de_left.get(band, 0.0))

        # RASM: variance ratio (always positive; avoids DE sign ambiguity at very low power).
        # Computed from raw band variance rather than DE to be numerically stable.
        left_band = bandpass_filter(left_signal, low, high, fs)
        right_band = bandpass_filter(right_signal, low, high, fs)
        var_l = float(np.var(left_band))
        var_r = float(np.var(right_band))
        result[f"rasm_{band}"] = var_r / max(var_l, 1e-12)

    return result


def compute_frontal_midline_theta(
    eeg: np.ndarray, fs: float = 256.0
) -> Dict[str, float]:
    """Compute Frontal Midline Theta (FMT) power and amplitude features.

    FMT sources from Anterior Cingulate Cortex (ACC) + medial PFC. Increases with:
      - Cognitive load / working memory demand (linear with n-back difficulty)
      - Positive emotional experience and relaxation from anxiety
      - Internalized attention and meditation depth

    Key advantage over FAA: FMT uses absolute power, not hemispheric asymmetry,
    making it less sensitive to reference electrode choice and more robust for
    real-time 4-channel systems like Muse 2.

    FP1/FP2 theta asymmetry (not computed here) can discriminate anger vs. fear.
    For single-channel use, compute on AF7 (ch1) or AF8 (ch2).

    Args:
        eeg: 1D EEG signal (single channel, preferably AF7 or AF8).
        fs: Sampling frequency (Hz).

    Returns:
        fmt_power:     Raw theta band variance (µV²) — absolute power estimate.
        fmt_de:        Differential entropy of theta band.
        fmt_amplitude: Mean instantaneous amplitude via Hilbert envelope.
        fmt_relative:  Theta as a fraction of total band power (0–1).
    """
    theta_filtered = bandpass_filter(eeg, 4.0, 8.0, fs)

    var = float(np.var(theta_filtered))
    de = float(0.5 * np.log(2 * np.pi * np.e * max(var, 1e-10)))

    try:
        from scipy.signal import hilbert as _hilbert
        amplitude = float(np.mean(np.abs(_hilbert(theta_filtered))))
    except Exception:
        amplitude = float(np.sqrt(max(var, 0)))

    psd_cache = _compute_psd(eeg, fs)
    theta_relative = extract_band_powers(eeg, fs, psd_cache=psd_cache).get("theta", 0.0)

    return {
        "fmt_power": var,
        "fmt_de": de,
        "fmt_amplitude": amplitude,
        "fmt_relative": float(theta_relative),
    }


def compute_theta_gamma_coupling(
    eeg: np.ndarray, fs: float = 256.0
) -> float:
    """Compute theta-gamma modulation index as a cross-frequency coupling proxy.

    Working memory and cognitive load are associated with theta-gamma coupling
    (Lisman & Jensen, 2013). Higher coupling = more active memory encoding.

    Uses a simplified power-envelope correlation approach.

    Returns:
        Coupling strength (0-1), higher = stronger theta-gamma coupling.
    """
    try:
        # Extract theta envelope
        theta_filtered = bandpass_filter(eeg, 4.0, 8.0, fs)
        from scipy.signal import hilbert
        theta_envelope = np.abs(hilbert(theta_filtered))

        # Extract gamma envelope
        gamma_filtered = bandpass_filter(eeg, 30.0, 80.0, fs)
        gamma_envelope = np.abs(hilbert(gamma_filtered))

        # Pearson correlation of envelopes (bounded to [0, 1])
        if theta_envelope.std() < 1e-10 or gamma_envelope.std() < 1e-10:
            return 0.0
        corr = float(np.corrcoef(theta_envelope, gamma_envelope)[0, 1])
        return float(np.clip((corr + 1) / 2, 0, 1))
    except Exception:
        return 0.0


def epoch_signal(
    eeg: np.ndarray, fs: float = 256.0, window_sec: float = 30.0, overlap: float = 0.0
) -> List[np.ndarray]:
    """Split continuous EEG into fixed-length epochs."""
    window_samples = int(window_sec * fs)
    step = int(window_samples * (1 - overlap))
    epochs = []
    for start in range(0, len(eeg) - window_samples + 1, step):
        epochs.append(eeg[start : start + window_samples])
    return epochs


# --- Multi-Channel Analysis ---


def extract_features_multichannel(
    signals: np.ndarray, fs: float = 256.0, method: str = "average",
    left_ch: int = 1, right_ch: int = 2,
) -> Dict[str, float]:
    """Extract features from multi-channel EEG, averaging across channels.

    Args:
        signals:   2D array (n_channels, n_samples)
        fs:        Sampling frequency
        method:    Aggregation method ("average" supported)
        left_ch:   Index of left-frontal electrode for DASM/RASM (default 1 = AF7 on Muse 2).
        right_ch:  Index of right-frontal electrode for DASM/RASM (default 2 = AF8 on Muse 2).

    Returns:
        Dict of averaged features (same 17 features as single-channel).
    """
    n_channels = signals.shape[0]
    all_features = []

    for ch in range(n_channels):
        processed = preprocess(signals[ch], fs)
        features = extract_features(processed, fs)
        all_features.append(features)

    if method == "average":
        avg_features = {}
        keys = all_features[0].keys()
        for key in keys:
            avg_features[key] = float(
                np.mean([f[key] for f in all_features])
            )

        # DASM/RASM: inter-hemispheric asymmetry across all bands.
        # left_ch / right_ch are device-specific (see processing/channel_maps.py).
        if n_channels >= 3:
            # Clamp to valid range in case caller passes unchecked indices
            lf = min(left_ch, n_channels - 1)
            rf = min(right_ch, n_channels - 1)
            dasm_rasm = compute_dasm_rasm(signals, fs, left_ch=lf, right_ch=rf)
            avg_features.update(dasm_rasm)

        return avg_features

    return all_features[0]


def compute_coherence(
    signals: np.ndarray, fs: float = 256.0, band: str = "alpha"
) -> float:
    """Compute mean inter-channel coherence in a frequency band.

    Args:
        signals: 2D array (n_channels, n_samples)
        fs: Sampling frequency
        band: Frequency band name (from BANDS dict)

    Returns:
        Mean coherence value (0-1) across all channel pairs in the band.
    """
    low, high = BANDS.get(band, (8.0, 12.0))
    n_channels = signals.shape[0]

    if n_channels < 2:
        return 1.0

    coherence_values = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            freqs, coh = scipy_signal.coherence(
                signals[i], signals[j], fs=fs,
                nperseg=min(len(signals[i]), int(fs * 2))
            )
            mask = (freqs >= low) & (freqs <= high)
            if mask.any():
                coherence_values.append(float(np.mean(coh[mask])))

    return float(np.mean(coherence_values)) if coherence_values else 0.0


def compute_phase_locking_value(
    signals: np.ndarray, fs: float = 256.0, band: str = "alpha"
) -> float:
    """Compute mean Phase Locking Value (PLV) across channel pairs.

    Uses Hilbert transform to extract instantaneous phase, then
    computes PLV as consistency of phase difference.

    Args:
        signals: 2D array (n_channels, n_samples)
        fs: Sampling frequency
        band: Frequency band name

    Returns:
        Mean PLV (0-1) across all channel pairs.
    """
    low, high = BANDS.get(band, (8.0, 12.0))
    n_channels = signals.shape[0]

    if n_channels < 2:
        return 1.0

    # Bandpass filter each channel
    filtered = np.array([bandpass_filter(signals[ch], low, high, fs) for ch in range(n_channels)])

    # Extract instantaneous phase via Hilbert transform
    from scipy.signal import hilbert
    analytic = hilbert(filtered, axis=1)
    phases = np.angle(analytic)

    plv_values = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = phases[i] - phases[j]
            plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
            plv_values.append(plv)

    return float(np.mean(plv_values)) if plv_values else 0.0


def compute_pairwise_plv(
    signals: np.ndarray, fs: float = 256.0,
    bands: Optional[Dict[str, tuple]] = None,
) -> Dict:
    """Compute per-pair, per-band Phase Locking Value for emotion classification.

    Unlike compute_phase_locking_value() which returns a single mean scalar,
    this returns the full 6-pair PLV matrix for theta/alpha/beta bands plus
    named summary features (frontal pair, fronto-temporal mean, global mean).

    Motivated by Wang et al. (2024), "Fusion of Multi-domain EEG Signatures
    Improves Emotion Recognition" -- PLV connectivity features combined with
    spectral/microstate features achieved 7%+ accuracy improvement over
    single-domain approaches.  High-frequency PLV in prefrontal and temporal
    regions particularly enhances emotional differentiation.

    Muse 2 channel order: TP9=0, AF7=1, AF8=2, TP10=3.
    6 pairs (4C2): (0,1) (0,2) (0,3) (1,2) (1,3) (2,3).
    Frontal pair: AF7-AF8 = pair index 3 → (1,2).
    Fronto-temporal: AF7-TP9 = pair 0 → (0,1), AF8-TP10 = pair 5 → (2,3).

    Args:
        signals: EEG data, shape (n_channels, n_samples).
        fs: Sampling rate in Hz.
        bands: Dict of band_name -> (low_hz, high_hz).  Defaults to
               theta (4-8), alpha (8-12), beta (12-30).

    Returns:
        Dict with keys:
            plv_theta, plv_alpha, plv_beta: list of per-pair PLV values.
            plv_frontal_alpha/theta/beta: AF7-AF8 PLV (pair index depends on n_channels).
            plv_fronto_temporal_alpha: mean of AF7-TP9 and AF8-TP10 PLV in alpha.
            plv_mean_alpha/theta/beta: global mean PLV across all pairs.
            feature_vector: flat list of all PLV features for ML pipelines.
            n_features: length of feature_vector.
    """
    if bands is None:
        bands = {
            "theta": (4.0, 8.0),
            "alpha": (8.0, 12.0),
            "beta": (12.0, 30.0),
        }

    n_channels = signals.shape[0]
    n_samples = signals.shape[1] if signals.ndim == 2 else 0

    # Build pair indices
    pairs = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            pairs.append((i, j))
    n_pairs = len(pairs)

    # Edge case: fewer than 2 channels or very short signal
    if n_channels < 2 or n_samples < 64:
        n_pairs_default = max(n_pairs, 1)
        empty: Dict = {}
        for band_name in bands:
            empty[f"plv_{band_name}"] = [0.0] * n_pairs_default
            empty[f"plv_mean_{band_name}"] = 0.0
        empty["plv_frontal_alpha"] = 0.0
        empty["plv_frontal_theta"] = 0.0
        empty["plv_frontal_beta"] = 0.0
        empty["plv_fronto_temporal_alpha"] = 0.0
        n_feats = n_pairs_default * len(bands) + 4  # 4 summary features
        empty["feature_vector"] = [0.0] * n_feats
        empty["n_features"] = n_feats
        return empty

    from scipy.signal import hilbert as _hilbert

    # Compute PLV per band per pair
    band_plvs: Dict[str, list] = {}
    for band_name, (low, high) in bands.items():
        # Bandpass filter each channel for this band
        filtered = np.array([
            bandpass_filter(signals[ch], low, high, fs)
            for ch in range(n_channels)
        ])

        # Extract instantaneous phase via Hilbert transform
        analytic = _hilbert(filtered, axis=1)
        phases = np.angle(analytic)

        pair_plvs = []
        for (i, j) in pairs:
            phase_diff = phases[i] - phases[j]
            plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
            pair_plvs.append(round(float(np.clip(plv, 0.0, 1.0)), 4))
        band_plvs[band_name] = pair_plvs

    # Build result dict
    result: Dict = {}
    for band_name in bands:
        result[f"plv_{band_name}"] = band_plvs[band_name]
        result[f"plv_mean_{band_name}"] = round(
            float(np.mean(band_plvs[band_name])), 4
        )

    # Named summary features based on Muse 2 channel layout
    # Frontal pair: AF7(1)-AF8(2).  Find the pair index.
    frontal_idx = None
    ft_left_idx = None   # AF7(1)-TP9(0)
    ft_right_idx = None  # AF8(2)-TP10(3)
    for idx, (i, j) in enumerate(pairs):
        if (i, j) == (1, 2):
            frontal_idx = idx
        if (i, j) == (0, 1):
            ft_left_idx = idx
        if (i, j) == (2, 3):
            ft_right_idx = idx

    for band_name in bands:
        key = f"plv_frontal_{band_name}"
        if frontal_idx is not None:
            result[key] = band_plvs[band_name][frontal_idx]
        else:
            result[key] = 0.0

    # Fronto-temporal: mean of left (AF7-TP9) and right (AF8-TP10) in alpha
    ft_vals = []
    if ft_left_idx is not None and "alpha" in band_plvs:
        ft_vals.append(band_plvs["alpha"][ft_left_idx])
    if ft_right_idx is not None and "alpha" in band_plvs:
        ft_vals.append(band_plvs["alpha"][ft_right_idx])
    result["plv_fronto_temporal_alpha"] = round(
        float(np.mean(ft_vals)), 4
    ) if ft_vals else 0.0

    # Build flat feature vector: all per-pair PLVs + 4 summary features
    fv: list = []
    for band_name in bands:
        fv.extend(band_plvs[band_name])
    fv.append(result.get("plv_frontal_alpha", 0.0))
    fv.append(result.get("plv_frontal_theta", 0.0))
    fv.append(result.get("plv_frontal_beta", 0.0))
    fv.append(result.get("plv_fronto_temporal_alpha", 0.0))

    result["feature_vector"] = fv
    result["n_features"] = len(fv)

    return result


# --- Wavelet Time-Frequency Analysis ---


def compute_cwt_spectrogram(
    signal_data: np.ndarray,
    fs: float = 256.0,
    wavelet: str = "morl",
    freqs: np.ndarray = None,
) -> Dict:
    """Compute Continuous Wavelet Transform spectrogram.

    Args:
        signal_data: 1D EEG signal array.
        fs: Sampling frequency in Hz.
        wavelet: Wavelet name (default Morlet).
        freqs: Frequencies of interest (default 1-50 Hz).

    Returns:
        Dict with 'coefficients' (2D power matrix), 'frequencies', 'times'.
    """
    if freqs is None:
        freqs = np.arange(1, 50)

    # pywt.cwt expects scales, not frequencies
    scales = pywt.central_frequency(wavelet) * fs / freqs
    coefficients, _ = pywt.cwt(signal_data, scales, wavelet, sampling_period=1.0 / fs)
    power = np.abs(coefficients) ** 2
    times = np.arange(len(signal_data)) / fs

    return {
        "coefficients": power.tolist(),
        "frequencies": freqs.tolist(),
        "times": times.tolist(),
    }


def compute_dwt_features(
    signal_data: np.ndarray, fs: float = 256.0, wavelet: str = "db4", level: int = 5
) -> Dict[str, float]:
    """Discrete Wavelet Transform band energy decomposition.

    Decomposition levels (at 256 Hz):
        D1 ~ 64-128 Hz (gamma+)
        D2 ~ 32-64 Hz  (gamma)
        D3 ~ 16-32 Hz  (beta)
        D4 ~ 8-16 Hz   (alpha)
        D5 ~ 4-8 Hz    (theta)
        A5 ~ 0-4 Hz    (delta)

    Returns:
        Dict mapping band names to relative energy.
    """
    max_level = pywt.dwt_max_level(len(signal_data), pywt.Wavelet(wavelet).dec_len)
    level = min(level, max_level)

    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    energies = [float(np.sum(c ** 2)) for c in coeffs]
    total = sum(energies) + 1e-10

    band_map = ["delta", "theta", "alpha", "beta", "gamma"]
    result = {}
    # A_n (approx) = lowest freq band (delta), then D_n...D_1
    result["delta"] = energies[0] / total
    for i, coeff_energy in enumerate(energies[1:]):
        if i < len(band_map) - 1:
            result[band_map[i + 1]] = coeff_energy / total
        else:
            result[f"detail_{i + 1}"] = coeff_energy / total

    return result


def detect_sleep_spindles(
    signal_data: np.ndarray, fs: float = 256.0, threshold: float = 2.0
) -> List[Dict]:
    """Detect sleep spindles (11-16 Hz bursts) using wavelet envelope.

    Args:
        signal_data: 1D preprocessed EEG signal.
        fs: Sampling frequency.
        threshold: Amplitude threshold in standard deviations.

    Returns:
        List of dicts with 'start', 'end', 'amplitude' for each spindle.
    """
    # Bandpass 11-16 Hz (sigma band)
    filtered = bandpass_filter(signal_data, 11.0, 16.0, fs, order=4)

    # Compute envelope via wavelet transform
    scales = pywt.central_frequency("morl") * fs / np.array([13.5])
    coeffs, _ = pywt.cwt(filtered, scales, "morl", sampling_period=1.0 / fs)
    envelope = np.abs(coeffs[0])

    # Threshold detection
    mean_env = np.mean(envelope)
    std_env = np.std(envelope)
    thresh_val = mean_env + threshold * std_env

    above = envelope > thresh_val
    spindles = []

    # Find contiguous regions above threshold
    in_spindle = False
    start_idx = 0
    for i in range(len(above)):
        if above[i] and not in_spindle:
            in_spindle = True
            start_idx = i
        elif not above[i] and in_spindle:
            in_spindle = False
            duration = (i - start_idx) / fs
            # Spindles are 0.5-2.0 seconds
            if 0.3 <= duration <= 3.0:
                spindles.append({
                    "start": float(start_idx / fs),
                    "end": float(i / fs),
                    "amplitude": float(np.max(envelope[start_idx:i])),
                })

    return spindles


def detect_k_complexes(
    signal_data: np.ndarray, fs: float = 256.0
) -> List[Dict]:
    """Detect K-complexes: high-amplitude slow waves (0.5-1.5 Hz).

    K-complexes have a sharp negative peak followed by a positive deflection,
    with total duration of 0.5-1.5 seconds and amplitude > 75 uV.

    Returns:
        List of dicts with 'time' and 'amplitude' for each K-complex.
    """
    filtered = bandpass_filter(signal_data, 0.5, 1.5, fs, order=3)

    # Find negative peaks
    neg_peaks, props = scipy_signal.find_peaks(-filtered, height=75, distance=int(fs * 0.5))

    k_complexes = []
    for peak_idx in neg_peaks:
        peak_time = peak_idx / fs
        # Check for positive deflection after the negative peak
        window_end = min(peak_idx + int(fs * 1.0), len(filtered))
        post_peak = filtered[peak_idx:window_end]
        if len(post_peak) > 0 and np.max(post_peak) > 30:
            k_complexes.append({
                "time": float(peak_time),
                "amplitude": float(filtered[peak_idx]),
            })

    return k_complexes


# ── Baseline Calibration ──────────────────────────────────────────────────────


class BaselineCalibrator:
    """Resting-state baseline normalization for within-session EEG features.

    Records a 2–3 minute resting baseline at session start, then normalizes
    all subsequent live features against it:

        corrected = (feature - baseline_mean) / baseline_std

    This corrects for inter-individual differences in skull thickness, hair,
    electrode fit, and session-to-session drift in absolute power levels.

    Published accuracy improvement: +15–29% over no baseline correction.
    (A Novel Baseline Removal Paradigm for Subject-Independent Features, PMC9854727)

    Usage:
        cal = BaselineCalibrator()
        # During 2-min eyes-closed resting state (call once per incoming epoch):
        for epoch in resting_epochs:
            cal.add_baseline_frame(epoch_4ch, fs=256.0)
        # During live session:
        if cal.is_ready:
            norm_features = cal.normalize(live_features)
    """

    _MIN_BASELINE_FRAMES = 30  # ~30 seconds at 1 fps; enough for stable mean/std

    def __init__(self):
        self._frames: List[Dict[str, float]] = []
        self._mean: Dict[str, float] = {}
        self._std: Dict[str, float] = {}
        self._ready = False

    @property
    def is_ready(self) -> bool:
        """True once enough baseline frames have been collected and stats computed."""
        return self._ready

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    def add_baseline_frame(self, signals: np.ndarray, fs: float = 256.0) -> bool:
        """Add one resting-state EEG epoch to the baseline buffer.

        Automatically computes statistics when _MIN_BASELINE_FRAMES are collected.

        Args:
            signals: 1D (single channel) or 2D (n_channels, n_samples) EEG array.
            fs: Sampling frequency.

        Returns:
            True if calibration is now ready (just crossed the threshold).
        """
        if signals.ndim == 2:
            features = extract_features_multichannel(signals, fs)
        else:
            features = extract_features(preprocess(signals, fs), fs)

        self._frames.append(features)

        if len(self._frames) >= self._MIN_BASELINE_FRAMES and not self._ready:
            self._compute_statistics()
            return True
        return False

    def _compute_statistics(self) -> None:
        """Compute per-feature mean and std across all baseline frames."""
        keys = self._frames[0].keys()
        for k in keys:
            vals = [f[k] for f in self._frames if k in f]
            self._mean[k] = float(np.mean(vals))
            self._std[k] = float(np.std(vals))
        self._ready = True

    def normalize(self, features: Dict[str, float]) -> Dict[str, float]:
        """Z-score normalize a feature dict against the resting-state baseline.

        Features with near-zero baseline variance (constant at rest) are set to 0
        to avoid division issues.

        Returns features unchanged if calibration is not yet ready.
        """
        if not self._ready:
            return features
        out = {}
        for k, v in features.items():
            std = self._std.get(k, 0.0)
            mean = self._mean.get(k, v)
            out[k] = float((v - mean) / std) if std > 1e-8 else 0.0
        return out

    def reset(self) -> None:
        """Clear all baseline data (call at the start of each new session)."""
        self._frames.clear()
        self._mean.clear()
        self._std.clear()
        self._ready = False

    def to_dict(self) -> Dict:
        """Serialize calibration state for storage/persistence."""
        return {
            "n_frames": len(self._frames),
            "is_ready": self._ready,
            "mean": self._mean,
            "std": self._std,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BaselineCalibrator":
        """Restore a previously saved calibrator (e.g., from session JSON)."""
        cal = cls()
        cal._mean = data.get("mean", {})
        cal._std = data.get("std", {})
        cal._ready = bool(data.get("is_ready", False))
        return cal


# ── RunningNormalizer — session drift correction ──────────────────────────────
import threading as _threading
from collections import deque as _deque


class RunningNormalizer:
    """Per-user rolling z-score normalizer for EEG features.

    Corrects within-session EEG signal drift (non-stationarity).
    Technique from SJTU SEED team and UESTC FACED paper:
    rolling normalization recovers +10-20 accuracy points on
    cross-subject emotion recognition.

    Returns raw features for the first 30 frames (insufficient statistics).
    After that, z-scores against a rolling 150-frame buffer (~5 min at 2s hop).
    """

    _MIN_SAMPLES: int = 30
    _BUFFER_SIZE: int = 150

    def __init__(self) -> None:
        self._buffers: Dict[str, _deque] = {}
        self._lock = _threading.Lock()

    def normalize(self, features: np.ndarray, user_id: str) -> np.ndarray:
        """Z-score features against the rolling buffer for this user.

        Returns raw features when buffer is below MIN_SAMPLES.
        Handles near-zero std by returning 0 for those dimensions.
        """
        with self._lock:
            buf = self._buffers.setdefault(user_id, _deque(maxlen=self._BUFFER_SIZE))
            buf.append(features.copy())
            n = len(buf)
            snapshot = list(buf) if n >= self._MIN_SAMPLES else None

        if snapshot is None:
            return features

        matrix = np.stack(snapshot)  # (n, n_features)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        # Avoid division by zero: where std < 1e-8, output 0
        safe_std = np.where(std < 1e-8, 1.0, std)
        normed = (features - mean) / safe_std
        normed = np.where(std < 1e-8, 0.0, normed)
        return normed


def apply_circadian_correction(
    features: Dict[str, float],
    hour_of_day: int,
    chronotype: str = "intermediate",
) -> Dict[str, float]:
    """Apply time-of-day circadian correction to EEG feature values.

    Alpha power follows a diurnal rhythm peaking in late morning and early
    evening (circadian rhythm research, Cajochen et al. 2002; Lim et al. 2020).
    Beta is anti-correlated with alpha. Theta is elevated at night.

    Corrections are multiplicative relative to a noon baseline (multiplier=1.0).
    Features not covered by the correction table are returned unchanged.

    Args:
        features: Dict of feature names to float values (e.g. from extract_features()).
        hour_of_day: Integer hour in 24h format (0–23).
        chronotype: "early" (morning type, -2h offset), "intermediate" (default),
                    or "late" (evening type, +2h offset).

    Returns:
        New feature dict with corrected values (input dict is not mutated).
    """
    # Multipliers for alpha-related features relative to noon baseline.
    # Derived from diurnal EEG alpha power patterns in published literature.
    ALPHA_CORRECTION: Dict[int, float] = {
        6: 0.85, 7: 0.88, 8: 0.92, 9: 0.96, 10: 1.0, 11: 1.02,
        12: 1.0, 13: 0.98, 14: 0.95, 15: 0.93, 16: 0.94, 17: 0.96,
        18: 1.0, 19: 1.03, 20: 1.05, 21: 1.02, 22: 0.95, 23: 0.88,
    }

    # Chronotype offset shifts the effective hour for the correction lookup.
    CHRONOTYPE_OFFSETS = {"early": -2, "intermediate": 0, "late": 2}
    offset = CHRONOTYPE_OFFSETS.get(chronotype, 0)
    effective_hour = (hour_of_day + offset) % 24

    alpha_mult = ALPHA_CORRECTION.get(effective_hour)
    if alpha_mult is None:
        # Hour not in table (e.g. 0–5) — no correction applied (safe fallback)
        return features

    corrected = dict(features)

    # Alpha-related features: multiply by alpha_mult (higher alpha in peak hours)
    alpha_keys = {"band_power_alpha", "alpha_power", "alpha_theta_ratio", "alpha_beta_ratio"}
    for key in alpha_keys:
        if key in corrected:
            corrected[key] = corrected[key] * alpha_mult

    # Beta-related and stress features: divide by alpha_mult (beta is anti-correlated)
    beta_keys = {"band_power_beta", "beta_power", "stress_index"}
    for key in beta_keys:
        if key in corrected:
            corrected[key] = corrected[key] / alpha_mult

    return corrected


def extract_spectral_microstate_features(
    signals: np.ndarray, fs: float = 256.0, window_ms: int = 250
) -> Dict:
    """Extract spectral microstate temporal features from EEG.

    Delegates to ``processing.spectral_microstates.extract_microstate_features``.
    Defines 4 microstates by dominant frequency band (delta/theta/alpha/beta)
    per 250ms window, then computes coverage, duration, occurrence,
    transition probabilities, and transition entropy features (entropy rate,
    excess entropy, Lempel-Ziv complexity).

    Args:
        signals: EEG data, shape (n_channels, n_samples) or (n_samples,).
        fs: Sampling rate in Hz.
        window_ms: Window duration in milliseconds (default 250).

    Returns:
        Dict with coverage, avg_duration, occurrence, transition_matrix,
        transition_entropy, dominant_state, state_diversity,
        feature_vector (31 elements), n_features, sequence_length.
    """
    from processing.spectral_microstates import extract_microstate_features
    return extract_microstate_features(signals, int(fs), window_ms)


# ── Euclidean Alignment (He & Wu, 2018, IEEE TBME) ──────────────────────────
#
# Reference: "Transfer Learning for Brain-Computer Interfaces: A Euclidean
#   Space Data Alignment Approach", He & Wu, IEEE Trans. Biomed. Eng., 2020.
# Confirmed effective: +4.33% cross-subject accuracy (arXiv:2401.10746, 2024).
# Revisited and recommended across 13 BCI paradigms (arXiv:2502.09203, 2025).
#
# The idea: each subject's EEG has a different covariance structure due to
# skull thickness, electrode impedance, brain geometry, etc. EA removes this
# by whitening each subject's trials so the mean covariance becomes identity.
#
# Algorithm for each subject:
#   1. Stack all epochs into (n_epochs, n_channels, n_samples) array
#   2. Compute mean covariance: R = mean(X_i @ X_i.T / n_samples)
#   3. Compute R^{-1/2} via eigendecomposition
#   4. Transform each epoch: X_aligned = R^{-1/2} @ X_i
#
# After alignment, covariance of each subject's data ≈ identity,
# removing inter-subject variability and improving cross-subject classifiers.


def _matrix_invsqrt(M: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Compute M^{-1/2} via eigendecomposition.

    Args:
        M: Symmetric positive semi-definite matrix (n, n).
        reg: Regularization added to eigenvalues (avoids division by zero).

    Returns:
        M^{-1/2} of shape (n, n).
    """
    eigvals, eigvecs = np.linalg.eigh(M)
    # Clamp eigenvalues to avoid negative values from numerical noise
    eigvals = np.maximum(eigvals, reg)
    inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
    return (eigvecs * inv_sqrt_eigvals[np.newaxis, :]) @ eigvecs.T


def euclidean_align(
    epochs: np.ndarray,
    ref_matrix: Optional[np.ndarray] = None,
    reg: float = 1e-6,
) -> tuple:
    """Apply Euclidean Alignment to a set of EEG epochs from one subject.

    Whitens the data so the mean covariance becomes approximately identity,
    removing subject-specific spatial patterns caused by skull thickness,
    electrode impedance, and brain geometry.

    Args:
        epochs: Array of shape (n_epochs, n_channels, n_samples).
                Each epoch is a multi-channel EEG segment.
        ref_matrix: Optional pre-computed reference covariance matrix R
                    of shape (n_channels, n_channels). If None, computed
                    from the provided epochs.
        reg: Regularization for eigenvalue clamping (default 1e-6).

    Returns:
        Tuple of (aligned_epochs, ref_matrix):
            aligned_epochs: Same shape as input, EA-transformed.
            ref_matrix: The reference covariance used (for reuse at inference).

    Example:
        # During training — align each subject's data
        aligned, R = euclidean_align(subject_epochs)

        # During live inference — use pre-computed R from calibration
        aligned, _ = euclidean_align(live_epochs, ref_matrix=R)
    """
    n_epochs, n_ch, n_samples = epochs.shape

    if ref_matrix is None:
        # Compute mean covariance across all epochs
        cov_sum = np.zeros((n_ch, n_ch))
        for i in range(n_epochs):
            X_i = epochs[i]  # (n_ch, n_samples)
            cov_sum += X_i @ X_i.T / n_samples
        ref_matrix = cov_sum / n_epochs

    # Compute whitening matrix R^{-1/2}
    R_invsqrt = _matrix_invsqrt(ref_matrix, reg=reg)

    # Align each epoch
    aligned = np.empty_like(epochs)
    for i in range(n_epochs):
        aligned[i] = R_invsqrt @ epochs[i]

    return aligned, ref_matrix


class EuclideanAligner:
    """Online Euclidean Alignment for live EEG streaming.

    Accumulates covariance from incoming epochs and applies EA transformation.
    Requires a minimum number of epochs before alignment activates (before that,
    returns data unchanged).

    Usage:
        aligner = EuclideanAligner(n_channels=4, min_epochs=10)

        # During calibration phase (first 30-60 seconds):
        for epoch in calibration_epochs:
            aligned = aligner.update_and_align(epoch)  # accumulates cov

        # During live session:
        for epoch in live_epochs:
            aligned = aligner.align(epoch)  # uses frozen reference
    """

    def __init__(self, n_channels: int = 4, min_epochs: int = 10, reg: float = 1e-6):
        self.n_ch = n_channels
        self.min_epochs = min_epochs
        self.reg = reg
        self._cov_sum = np.zeros((n_channels, n_channels))
        self._n_epochs = 0
        self._R_invsqrt: Optional[np.ndarray] = None
        self._ref_matrix: Optional[np.ndarray] = None

    @property
    def is_ready(self) -> bool:
        """True once enough epochs have been accumulated for alignment."""
        return self._n_epochs >= self.min_epochs

    @property
    def n_epochs_seen(self) -> int:
        return self._n_epochs

    def add_epoch(self, epoch: np.ndarray) -> None:
        """Add an epoch to the running covariance estimate.

        Args:
            epoch: Array of shape (n_channels, n_samples).
        """
        if epoch.ndim != 2 or epoch.shape[0] != self.n_ch:
            return
        n_samples = epoch.shape[1]
        if n_samples < 2:
            return
        self._cov_sum += epoch @ epoch.T / n_samples
        self._n_epochs += 1
        # Recompute whitening matrix
        self._ref_matrix = self._cov_sum / self._n_epochs
        self._R_invsqrt = _matrix_invsqrt(self._ref_matrix, reg=self.reg)

    def align(self, epoch: np.ndarray) -> np.ndarray:
        """Apply Euclidean Alignment to a single epoch.

        Args:
            epoch: Array of shape (n_channels, n_samples).

        Returns:
            Aligned epoch (same shape). Returns unchanged if not ready.
        """
        if self._R_invsqrt is None or not self.is_ready:
            return epoch
        return self._R_invsqrt @ epoch

    def update_and_align(self, epoch: np.ndarray) -> np.ndarray:
        """Add epoch to covariance estimate AND return aligned version.

        Convenience method for the calibration phase where you want to
        both update the reference and get aligned output.
        """
        self.add_epoch(epoch)
        return self.align(epoch)

    def get_ref_matrix(self) -> Optional[np.ndarray]:
        """Return the current reference covariance matrix (for serialization)."""
        return self._ref_matrix

    def set_ref_matrix(self, R: np.ndarray) -> None:
        """Load a pre-computed reference matrix (e.g., from a previous session).

        Args:
            R: Covariance matrix of shape (n_channels, n_channels).
        """
        self._ref_matrix = R
        self._R_invsqrt = _matrix_invsqrt(R, reg=self.reg)
        self._n_epochs = self.min_epochs  # mark as ready

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._cov_sum = np.zeros((self.n_ch, self.n_ch))
        self._n_epochs = 0
        self._R_invsqrt = None
        self._ref_matrix = None
