"""EEG Signal Processing Module.

Provides bandpass/notch filtering, frequency band power extraction,
Hjorth parameters, spectral entropy, epoching, and wavelet
time-frequency analysis for EEG signals.
"""

import numpy as np
import pywt
from scipy import signal as scipy_signal
from scipy.stats import entropy
from typing import Dict, List

# Numpy 2.x renamed np.trapz to np.trapezoid
_trapezoid = getattr(np, 'trapezoid', np.trapz)


# EEG frequency band definitions (Hz)
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 100.0),
}


def bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5
) -> np.ndarray:
    """Apply Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = scipy_signal.butter(order, [low, high], btype="band")
    return scipy_signal.filtfilt(b, a, data, axis=-1)


def notch_filter(data: np.ndarray, freq: float, fs: float, Q: float = 30.0) -> np.ndarray:
    """Apply notch filter to remove powerline interference."""
    nyq = 0.5 * fs
    w0 = freq / nyq
    if w0 >= 1.0:
        return data
    b, a = scipy_signal.iirnotch(w0, Q)
    return scipy_signal.filtfilt(b, a, data, axis=-1)


def preprocess(raw_eeg: np.ndarray, fs: float = 256.0) -> np.ndarray:
    """Full preprocessing pipeline: bandpass 1-50Hz + notch 50/60Hz."""
    filtered = bandpass_filter(raw_eeg, 1.0, 50.0, fs)
    filtered = notch_filter(filtered, 50.0, fs)
    filtered = notch_filter(filtered, 60.0, fs)
    return filtered


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


def extract_band_powers(eeg: np.ndarray, fs: float = 256.0) -> Dict[str, float]:
    """Extract power spectral density for each EEG frequency band."""
    freqs, psd = scipy_signal.welch(eeg, fs=fs, nperseg=min(len(eeg), int(fs * 2)))
    total_power = _trapezoid(psd, freqs)
    if total_power == 0:
        total_power = 1e-10

    band_powers = {}
    for band_name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        band_power = _trapezoid(psd[mask], freqs[mask])
        band_powers[band_name] = float(band_power / total_power)

    return band_powers


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


def spectral_entropy(eeg: np.ndarray, fs: float = 256.0) -> float:
    """Compute normalized spectral entropy of the EEG signal."""
    freqs, psd = scipy_signal.welch(eeg, fs=fs, nperseg=min(len(eeg), int(fs * 2)))
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
    """Extract comprehensive feature set from EEG signal."""
    features = {}

    band_powers = extract_band_powers(eeg, fs)
    for k, v in band_powers.items():
        features[f"band_power_{k}"] = v

    hjorth = compute_hjorth_parameters(eeg)
    for k, v in hjorth.items():
        features[f"hjorth_{k}"] = v

    features["spectral_entropy"] = spectral_entropy(eeg, fs)

    de = differential_entropy(eeg, fs)
    for k, v in de.items():
        features[f"de_{k}"] = v

    # Ratios useful for emotion/state detection
    alpha = band_powers.get("alpha", 0)
    beta = band_powers.get("beta", 0)
    theta = band_powers.get("theta", 0)
    features["alpha_beta_ratio"] = alpha / max(beta, 1e-10)
    features["theta_beta_ratio"] = theta / max(beta, 1e-10)
    features["alpha_theta_ratio"] = alpha / max(theta, 1e-10)

    return features


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
    signals: np.ndarray, fs: float = 256.0, method: str = "average"
) -> Dict[str, float]:
    """Extract features from multi-channel EEG, averaging across channels.

    Args:
        signals: 2D array (n_channels, n_samples)
        fs: Sampling frequency
        method: Aggregation method ("average" supported)

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
