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
# low_beta (12-20 Hz): active cognition, focus, motor planning
# high_beta (20-30 Hz): anxiety, stress, fear — strongest marker for fearful/anxious states
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "low_beta": (12.0, 20.0),
    "high_beta": (20.0, 30.0),
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
    filtered = bandpass_filter(raw_eeg, 1.0, 50.0, fs)
    filtered = notch_filter(filtered, 50.0, fs)
    filtered = notch_filter(filtered, 60.0, fs)

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
        "gamma": (30.0, 100.0),
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
_DASM_RASM_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


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
