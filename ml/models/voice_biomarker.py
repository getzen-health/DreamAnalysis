"""
voice_biomarker.py — Open-source voice biomarker feature extraction.

Kintsugi's voice biomarker models are proprietary and NOT open source.
This module implements an open-source alternative using the eGeMAPS
(extended Geneva Minimalistic Acoustic Parameter Set) feature set,
which is the standard for voice emotion recognition research.

The eGeMAPS features we extract here are the same acoustic features
that Kintsugi and similar clinical voice analysis tools likely use
internally, based on published research:
  - Eyben et al. (2016): "The Geneva Minimalistic Acoustic Parameter Set
    (GeMAPS) for Voice Research and Affective Computing"
  - IEEE Transactions on Affective Computing

Feature categories:
  1. Frequency (F0): jitter, F0 mean/std/range, F0 percentiles
  2. Amplitude: shimmer, loudness (HNR), amplitude perturbation
  3. Spectral: MFCC (13 coefficients), spectral centroid, spectral flux
  4. Temporal: speech rate, pause ratio, voiced/unvoiced ratio

Usage:
    from ml.models.voice_biomarker import extract_egemaps_features
    features = extract_egemaps_features(audio_samples, sample_rate=16000)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Types ────────────────────────────────────────────────────────────────────


@dataclass
class VoiceBiomarkerFeatures:
    """Complete voice biomarker feature set (eGeMAPS-inspired)."""

    # ── F0 (fundamental frequency) features ──
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_min: float = 0.0
    f0_max: float = 0.0
    f0_range: float = 0.0

    # ── Perturbation features ──
    jitter_local: float = 0.0  # cycle-to-cycle F0 variation (%)
    jitter_rap: float = 0.0  # relative average perturbation
    shimmer_local: float = 0.0  # cycle-to-cycle amplitude variation (%)
    shimmer_apq3: float = 0.0  # amplitude perturbation quotient (3-point)

    # ── Harmonics-to-noise ratio ──
    hnr_mean: float = 0.0  # higher = cleaner voice

    # ── Spectral features ──
    mfcc: list[float] = field(default_factory=lambda: [0.0] * 13)
    spectral_centroid: float = 0.0
    spectral_flux: float = 0.0

    # ── Temporal features ──
    speech_rate: float = 0.0  # syllables per second (estimated)
    voiced_fraction: float = 0.0  # fraction of frames that are voiced
    pause_rate: float = 0.0  # pauses per second

    # ── Metadata ──
    duration_sec: float = 0.0
    sample_rate: int = 16000
    n_voiced_frames: int = 0


# ── Constants ────────────────────────────────────────────────────────────────

MIN_F0_HZ = 75
MAX_F0_HZ = 500
FRAME_DURATION_SEC = 0.025  # 25ms frames (standard for speech)
FRAME_HOP_SEC = 0.010  # 10ms hop
SILENCE_THRESHOLD = 0.01  # RMS below this is silence
PRE_EMPHASIS_COEFF = 0.97


# ── Helpers ──────────────────────────────────────────────────────────────────


def _pre_emphasis(signal: np.ndarray, coeff: float = PRE_EMPHASIS_COEFF) -> np.ndarray:
    """Apply pre-emphasis filter to boost high frequencies."""
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def _frame_signal(
    signal: np.ndarray, frame_size: int, hop_size: int
) -> np.ndarray:
    """Split signal into overlapping frames."""
    n_frames = 1 + (len(signal) - frame_size) // hop_size
    if n_frames <= 0:
        return np.empty((0, frame_size))
    indices = (
        np.arange(frame_size)[None, :] + np.arange(n_frames)[:, None] * hop_size
    )
    return signal[indices]


def _rms(frame: np.ndarray) -> float:
    """Compute RMS energy of a frame."""
    return float(np.sqrt(np.mean(frame**2)))


def _autocorrelation_pitch(
    frame: np.ndarray, sr: int, min_f0: float = MIN_F0_HZ, max_f0: float = MAX_F0_HZ
) -> float:
    """Detect pitch period via normalized autocorrelation. Returns F0 in Hz or 0."""
    min_lag = int(sr / max_f0)
    max_lag = int(sr / min_f0)
    n = len(frame)

    if max_lag >= n or min_lag >= max_lag:
        return 0.0

    # Normalized autocorrelation
    best_lag = 0
    best_corr = -1.0

    for lag in range(min_lag, min(max_lag + 1, n)):
        x = frame[: n - lag]
        y = frame[lag:]
        norm = np.sqrt(np.sum(x**2) * np.sum(y**2))
        if norm < 1e-10:
            continue
        corr = float(np.sum(x * y) / norm)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    if best_corr < 0.3 or best_lag == 0:
        return 0.0

    return sr / best_lag


def _compute_hnr(frame: np.ndarray, sr: int, f0: float) -> float:
    """
    Compute Harmonics-to-Noise Ratio for a single frame.

    HNR = 10 * log10(autocorrelation_at_pitch_period / (1 - autocorrelation_at_pitch_period))
    """
    if f0 <= 0:
        return 0.0

    period = int(sr / f0)
    n = len(frame)
    if period >= n or period == 0:
        return 0.0

    x = frame[:n - period]
    y = frame[period:]
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    norm = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if norm < 1e-10:
        return 0.0

    r = float(np.sum(x * y) / norm)
    r = max(min(r, 0.9999), 0.0001)  # clamp to avoid log(0)

    return 10 * np.log10(r / (1 - r))


def _compute_mfcc(signal: np.ndarray, sr: int, n_mfcc: int = 13) -> list[float]:
    """
    Compute MFCC coefficients using a simple DFT + mel filterbank approach.

    This is a lightweight implementation. For production, use librosa or openSMILE.
    """
    n = len(signal)
    if n < 256:
        return [0.0] * n_mfcc

    # Windowed FFT
    fft_size = min(2048, n)
    windowed = signal[:fft_size] * np.hamming(fft_size)
    spectrum = np.abs(np.fft.rfft(windowed))
    power_spectrum = spectrum**2 / fft_size

    # Mel filterbank (simplified: 26 triangular filters)
    n_filters = 26
    low_mel = 0
    high_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((fft_size + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_filters, len(power_spectrum)))
    for i in range(n_filters):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        if center <= left or right <= center:
            continue
        for j in range(left, center):
            if j < len(power_spectrum):
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if j < len(power_spectrum):
                filterbank[i, j] = (right - j) / (right - center)

    mel_energies = np.dot(filterbank, power_spectrum)
    mel_energies = np.where(mel_energies > 1e-10, mel_energies, 1e-10)
    log_mel = np.log(mel_energies)

    # DCT to get MFCCs
    n_mel = len(log_mel)
    mfcc = np.zeros(n_mfcc)
    for k in range(n_mfcc):
        mfcc[k] = np.sum(
            log_mel * np.cos(np.pi * k * (np.arange(n_mel) + 0.5) / n_mel)
        )

    return mfcc.tolist()


# ── Main extraction ──────────────────────────────────────────────────────────


def extract_egemaps_features(
    samples: np.ndarray,
    sample_rate: int = 16000,
) -> VoiceBiomarkerFeatures:
    """
    Extract eGeMAPS-inspired voice biomarker features from raw audio.

    Parameters
    ----------
    samples : np.ndarray
        Mono audio samples as float32/float64, range [-1, 1].
    sample_rate : int
        Audio sample rate in Hz (default 16000).

    Returns
    -------
    VoiceBiomarkerFeatures
        Complete feature set for downstream classification.
    """
    result = VoiceBiomarkerFeatures()
    result.sample_rate = sample_rate
    result.duration_sec = len(samples) / sample_rate

    if len(samples) < sample_rate * 0.1:  # need at least 100ms
        return result

    # Ensure float64 for computation
    signal = samples.astype(np.float64)

    # Pre-emphasis
    signal = _pre_emphasis(signal)

    # Frame the signal
    frame_size = int(sample_rate * FRAME_DURATION_SEC)
    hop_size = int(sample_rate * FRAME_HOP_SEC)
    frames = _frame_signal(signal, frame_size, hop_size)

    if len(frames) == 0:
        return result

    # ── Per-frame analysis ──────────────────────────────────────────────
    f0_values: list[float] = []
    peak_amplitudes: list[float] = []
    hnr_values: list[float] = []
    voiced_count = 0
    pause_count = 0
    in_pause = False

    for frame in frames:
        energy = _rms(frame)

        if energy < SILENCE_THRESHOLD:
            if not in_pause:
                pause_count += 1
                in_pause = True
            continue
        else:
            in_pause = False

        f0 = _autocorrelation_pitch(frame, sample_rate)
        if f0 > 0:
            voiced_count += 1
            f0_values.append(f0)
            peak_amplitudes.append(float(np.max(np.abs(frame))))

            hnr = _compute_hnr(frame, sample_rate, f0)
            hnr_values.append(hnr)

    result.n_voiced_frames = voiced_count
    result.voiced_fraction = voiced_count / len(frames) if len(frames) > 0 else 0.0
    result.pause_rate = pause_count / result.duration_sec if result.duration_sec > 0 else 0.0

    # ── F0 statistics ──────────────────────────────────────────────────
    if len(f0_values) >= 2:
        f0_arr = np.array(f0_values)
        result.f0_mean = float(np.mean(f0_arr))
        result.f0_std = float(np.std(f0_arr, ddof=1))
        result.f0_min = float(np.min(f0_arr))
        result.f0_max = float(np.max(f0_arr))
        result.f0_range = result.f0_max - result.f0_min

    # ── Jitter (cycle-to-cycle F0 perturbation) ────────────────────────
    if len(f0_values) >= 2:
        periods = [sample_rate / f for f in f0_values if f > 0]
        if len(periods) >= 2:
            diffs = [abs(periods[i] - periods[i - 1]) for i in range(1, len(periods))]
            mean_period = np.mean(periods)
            if mean_period > 0:
                result.jitter_local = float(np.mean(diffs) / mean_period) * 100

            # RAP: 3-point running average perturbation
            if len(periods) >= 3:
                rap_diffs = []
                for i in range(1, len(periods) - 1):
                    avg3 = (periods[i - 1] + periods[i] + periods[i + 1]) / 3
                    rap_diffs.append(abs(periods[i] - avg3))
                if mean_period > 0:
                    result.jitter_rap = float(np.mean(rap_diffs) / mean_period) * 100

    # ── Shimmer (cycle-to-cycle amplitude perturbation) ────────────────
    if len(peak_amplitudes) >= 2:
        amp_arr = np.array(peak_amplitudes)
        diffs = np.abs(np.diff(amp_arr))
        mean_amp = np.mean(amp_arr)
        if mean_amp > 0:
            result.shimmer_local = float(np.mean(diffs) / mean_amp) * 100

        # APQ3: 3-point amplitude perturbation quotient
        if len(peak_amplitudes) >= 3:
            apq3_diffs = []
            for i in range(1, len(peak_amplitudes) - 1):
                avg3 = (peak_amplitudes[i - 1] + peak_amplitudes[i] + peak_amplitudes[i + 1]) / 3
                apq3_diffs.append(abs(peak_amplitudes[i] - avg3))
            if mean_amp > 0:
                result.shimmer_apq3 = float(np.mean(apq3_diffs) / mean_amp) * 100

    # ── HNR ────────────────────────────────────────────────────────────
    if len(hnr_values) > 0:
        result.hnr_mean = float(np.mean(hnr_values))

    # ── MFCC ───────────────────────────────────────────────────────────
    result.mfcc = _compute_mfcc(signal, sample_rate, n_mfcc=13)

    # ── Spectral centroid ──────────────────────────────────────────────
    fft_size = min(2048, len(signal))
    if fft_size > 0:
        spectrum = np.abs(np.fft.rfft(signal[:fft_size]))
        freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        total = np.sum(spectrum)
        if total > 0:
            result.spectral_centroid = float(np.sum(freqs * spectrum) / total)

    # ── Speech rate (approximate syllable count) ───────────────────────
    # Energy envelope -> count peaks above threshold
    frame_energies = np.array([_rms(f) for f in frames])
    if len(frame_energies) > 0:
        threshold = np.mean(frame_energies) * 0.5
        syllables = 0
        was_below = True
        for e in frame_energies:
            if e > threshold and was_below:
                syllables += 1
                was_below = False
            elif e <= threshold:
                was_below = True
        result.speech_rate = syllables / result.duration_sec if result.duration_sec > 0 else 0.0

    return result


def features_to_array(features: VoiceBiomarkerFeatures) -> np.ndarray:
    """
    Convert VoiceBiomarkerFeatures to a flat numpy array for ML input.

    Returns a 25-element feature vector:
      [f0_mean, f0_std, f0_range, jitter_local, jitter_rap,
       shimmer_local, shimmer_apq3, hnr_mean, spectral_centroid,
       speech_rate, voiced_fraction, pause_rate,
       mfcc_0 ... mfcc_12]
    """
    return np.array(
        [
            features.f0_mean,
            features.f0_std,
            features.f0_range,
            features.jitter_local,
            features.jitter_rap,
            features.shimmer_local,
            features.shimmer_apq3,
            features.hnr_mean,
            features.spectral_centroid,
            features.speech_rate,
            features.voiced_fraction,
            features.pause_rate,
        ]
        + features.mfcc,
        dtype=np.float32,
    )
