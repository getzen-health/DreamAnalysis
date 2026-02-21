"""Artifact Detection & Signal Quality Module.

Detects eye blinks, muscle artifacts, electrode pops, and computes
Signal Quality Index (SQI) for EEG channels. Includes ICA-based
artifact removal and automatic epoch rejection.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Dict, List, Tuple, Optional

# Numpy 2.x renamed np.trapz to np.trapezoid
_trapezoid = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)


def detect_eye_blinks(
    signal_data: np.ndarray, fs: float = 256.0, threshold: float = 100.0
) -> List[Dict]:
    """Detect eye blink artifacts via amplitude threshold on frontal channels.

    Eye blinks produce >100 uV deflections lasting 200-400ms.

    Returns:
        List of dicts with 'time', 'amplitude' for each blink.
    """
    peaks, props = scipy_signal.find_peaks(
        np.abs(signal_data),
        height=threshold,
        distance=int(fs * 0.3),  # Minimum 300ms between blinks
    )

    blinks = []
    for i, peak_idx in enumerate(peaks):
        blinks.append({
            "time": float(peak_idx / fs),
            "amplitude": float(props["peak_heights"][i]),
        })

    return blinks


def detect_muscle_artifacts(
    signal_data: np.ndarray, fs: float = 256.0, threshold: float = 0.4
) -> List[Dict]:
    """Detect muscle (EMG) artifacts via high-frequency power ratio.

    Muscle activity contaminates the gamma band (>30 Hz). A high ratio
    of high-frequency power to total power indicates muscle artifact.

    Returns:
        List of dicts with 'start', 'end', 'hf_ratio' for contaminated segments.
    """
    window_samples = int(fs * 2)  # 2-second windows
    step = int(fs * 1)  # 1-second step
    artifacts = []

    for start in range(0, len(signal_data) - window_samples + 1, step):
        segment = signal_data[start : start + window_samples]
        freqs, psd = scipy_signal.welch(segment, fs=fs, nperseg=min(len(segment), int(fs)))
        total_power = _trapezoid(psd, freqs) + 1e-10
        hf_mask = freqs >= 30.0
        hf_power = _trapezoid(psd[hf_mask], freqs[hf_mask])
        ratio = float(hf_power / total_power)

        if ratio > threshold:
            artifacts.append({
                "start": float(start / fs),
                "end": float((start + window_samples) / fs),
                "hf_ratio": ratio,
            })

    return artifacts


def detect_electrode_pops(
    signal_data: np.ndarray, fs: float = 256.0, threshold: float = 5.0
) -> List[Dict]:
    """Detect electrode pop artifacts via first-derivative threshold.

    Sudden jumps in the signal indicate electrode contact issues.

    Returns:
        List of dicts with 'time', 'magnitude' for each pop.
    """
    derivative = np.diff(signal_data)
    mean_d = np.mean(np.abs(derivative))
    std_d = np.std(np.abs(derivative))
    thresh_val = mean_d + threshold * std_d

    pop_indices = np.where(np.abs(derivative) > thresh_val)[0]

    # Group nearby pops (within 50ms)
    pops = []
    min_gap = int(fs * 0.05)
    if len(pop_indices) > 0:
        groups = [[pop_indices[0]]]
        for idx in pop_indices[1:]:
            if idx - groups[-1][-1] <= min_gap:
                groups[-1].append(idx)
            else:
                groups.append([idx])

        for group in groups:
            center = group[len(group) // 2]
            pops.append({
                "time": float(center / fs),
                "magnitude": float(np.max(np.abs(derivative[group]))),
            })

    return pops


def compute_signal_quality_index(
    signal_data: np.ndarray, fs: float = 256.0
) -> float:
    """Compute Signal Quality Index (0-100) for an EEG channel.

    Criteria:
        - Amplitude range (should be 10-200 uV for EEG)
        - Line noise ratio (50/60 Hz power relative to total)
        - High-frequency contamination (>40 Hz muscle noise)
        - Flatline detection (near-zero variance segments)

    Returns:
        SQI score from 0 (unusable) to 100 (excellent).
    """
    score = 100.0

    # 1. Amplitude range check (penalize extremes)
    amp_range = np.ptp(signal_data)
    if amp_range < 5:
        score -= 40  # Flatline or near-zero
    elif amp_range > 500:
        score -= 30  # Excessive amplitude (artifact)
    elif amp_range > 200:
        score -= 10

    # 2. Line noise ratio (50/60 Hz)
    freqs, psd = scipy_signal.welch(signal_data, fs=fs, nperseg=min(len(signal_data), int(fs * 2)))
    total_power = _trapezoid(psd, freqs) + 1e-10

    for line_freq in [50.0, 60.0]:
        if line_freq < fs / 2:
            mask = (freqs >= line_freq - 2) & (freqs <= line_freq + 2)
            if mask.any():
                line_power = _trapezoid(psd[mask], freqs[mask])
                line_ratio = line_power / total_power
                if line_ratio > 0.2:
                    score -= 20
                elif line_ratio > 0.1:
                    score -= 10

    # 3. High-frequency contamination
    hf_mask = freqs >= 40.0
    if hf_mask.any():
        hf_ratio = _trapezoid(psd[hf_mask], freqs[hf_mask]) / total_power
        if hf_ratio > 0.4:
            score -= 15
        elif hf_ratio > 0.25:
            score -= 8

    # 4. Flatline detection (check variance in sliding windows)
    window = int(fs * 1)  # 1-second windows
    for start in range(0, len(signal_data) - window, window):
        seg_var = np.var(signal_data[start : start + window])
        if seg_var < 0.1:
            score -= 5

    return max(0.0, min(100.0, score))


def auto_reject_epochs(
    signals: np.ndarray,
    fs: float = 256.0,
    epoch_sec: float = 5.0,
    sqi_threshold: float = 60.0,
) -> Tuple[List[np.ndarray], List[int]]:
    """Epoch the signal and reject epochs below SQI threshold.

    Args:
        signals: 1D or 2D array of EEG data.
        fs: Sampling frequency.
        epoch_sec: Epoch duration in seconds.
        sqi_threshold: Minimum SQI to keep an epoch.

    Returns:
        Tuple of (clean_epochs, rejected_indices).
    """
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    n_samples = signals.shape[1]
    epoch_samples = int(epoch_sec * fs)
    clean_epochs = []
    rejected = []

    for i, start in enumerate(range(0, n_samples - epoch_samples + 1, epoch_samples)):
        epoch = signals[:, start : start + epoch_samples]
        # Average SQI across channels
        sqis = [compute_signal_quality_index(epoch[ch], fs) for ch in range(epoch.shape[0])]
        avg_sqi = np.mean(sqis)

        if avg_sqi >= sqi_threshold:
            clean_epochs.append(epoch)
        else:
            rejected.append(i)

    return clean_epochs, rejected


def ica_artifact_removal(
    signals: np.ndarray, fs: float = 256.0, n_components: Optional[int] = None
) -> Dict:
    """Remove artifacts using Independent Component Analysis.

    Uses sklearn's FastICA to decompose signals and remove components
    with artifact characteristics (high kurtosis = eye blinks,
    high HF power = muscle artifacts).

    Args:
        signals: 2D array (n_channels, n_samples).
        fs: Sampling frequency.
        n_components: Number of ICA components (default: n_channels).

    Returns:
        Dict with 'cleaned_signals', 'removed_components', 'n_components'.
    """
    n_channels = signals.shape[0]
    if n_components is None:
        n_components = min(n_channels, signals.shape[1] // 10)
        n_components = max(2, n_components)

    try:
        from sklearn.decomposition import FastICA

        ica = FastICA(n_components=n_components, max_iter=500, random_state=42)
        sources = ica.fit_transform(signals.T)  # (n_samples, n_components)
        mixing = ica.mixing_  # (n_channels, n_components)

        # Identify artifact components
        removed = []
        for comp in range(n_components):
            component = sources[:, comp]

            # High kurtosis → likely eye blink artifact
            from scipy.stats import kurtosis
            kurt = kurtosis(component)
            if abs(kurt) > 5.0:
                sources[:, comp] = 0
                removed.append(comp)
                continue

            # High HF power ratio → muscle artifact
            freqs, psd = scipy_signal.welch(component, fs=fs, nperseg=min(len(component), int(fs)))
            total = _trapezoid(psd, freqs) + 1e-10
            hf_mask = freqs >= 30.0
            if hf_mask.any():
                hf_ratio = _trapezoid(psd[hf_mask], freqs[hf_mask]) / total
                if hf_ratio > 0.5:
                    sources[:, comp] = 0
                    removed.append(comp)

        # Reconstruct cleaned signals
        cleaned = (sources @ mixing.T).T  # Back to (n_channels, n_samples)

        return {
            "cleaned_signals": cleaned,
            "removed_components": removed,
            "n_components": n_components,
        }

    except ImportError:
        # Fallback: simple high-pass filter to remove slow drifts
        cleaned = np.array([
            scipy_signal.filtfilt(
                *scipy_signal.butter(3, 1.0 / (0.5 * fs), btype="high"),
                signals[ch],
            )
            for ch in range(n_channels)
        ])
        return {
            "cleaned_signals": cleaned,
            "removed_components": [],
            "n_components": 0,
        }
