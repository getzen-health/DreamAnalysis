"""Synthetic EEG generation and augmentation engine.

Solves the small-dataset bottleneck by generating realistic parametric
synthetic EEG signals with controllable spectral properties. No deep
learning required -- all generation is based on band-power-controlled
sinusoidal superposition with 1/f aperiodic background.

Capabilities:
  - Band-power controlled generation: specify target alpha, beta, theta,
    delta power and generate a signal that matches.
  - Emotion-conditioned generation: generate EEG that would produce a
    specific emotion classification (maps emotion -> band-power profile).
  - Noise injection: add realistic artifacts (eye blinks, muscle, electrode
    pop) at configurable rates.
  - Augmentation pipeline: take real EEG and produce augmented variants
    (time shift, amplitude scale, additive noise, band-power perturbation).
  - Quality validation: verify synthetic data has realistic spectral
    properties compared to known physiological ranges.

Functions:
  generate_synthetic_eeg()
  generate_emotion_conditioned_eeg()
  inject_artifacts()
  augment_eeg()
  validate_synthetic_quality()
  compute_generation_stats()
  stats_to_dict()

GitHub issue: #445
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal

# numpy >=2.0 renamed trapz -> trapezoid; fall back for older versions
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

log = logging.getLogger(__name__)

# ---- EEG frequency bands (Hz) ------------------------------------------------
BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 45.0),
}

# ---- Default band-power profiles (relative, sum to ~1) -----------------------
# Each profile is {band_name: relative_power} representing canonical spectral
# shapes for different emotional/cognitive states.  Values are approximate
# relative band-powers derived from literature.

_EMOTION_PROFILES: Dict[str, Dict[str, float]] = {
    "happy": {
        "delta": 0.10,
        "theta": 0.12,
        "alpha": 0.35,
        "beta": 0.30,
        "gamma": 0.13,
    },
    "sad": {
        "delta": 0.18,
        "theta": 0.22,
        "alpha": 0.30,
        "beta": 0.18,
        "gamma": 0.12,
    },
    "angry": {
        "delta": 0.10,
        "theta": 0.10,
        "alpha": 0.15,
        "beta": 0.40,
        "gamma": 0.25,
    },
    "fear": {
        "delta": 0.12,
        "theta": 0.18,
        "alpha": 0.15,
        "beta": 0.35,
        "gamma": 0.20,
    },
    "neutral": {
        "delta": 0.15,
        "theta": 0.15,
        "alpha": 0.30,
        "beta": 0.25,
        "gamma": 0.15,
    },
    "surprise": {
        "delta": 0.10,
        "theta": 0.15,
        "alpha": 0.20,
        "beta": 0.35,
        "gamma": 0.20,
    },
    "relaxed": {
        "delta": 0.15,
        "theta": 0.20,
        "alpha": 0.40,
        "beta": 0.15,
        "gamma": 0.10,
    },
    "focused": {
        "delta": 0.08,
        "theta": 0.10,
        "alpha": 0.15,
        "beta": 0.42,
        "gamma": 0.25,
    },
}

# ---- Physiological spectral ranges for quality validation --------------------
# Relative band-power ranges observed in healthy adults (eyes-closed resting).
_VALID_RANGES: Dict[str, Tuple[float, float]] = {
    "delta": (0.05, 0.50),
    "theta": (0.05, 0.40),
    "alpha": (0.05, 0.55),
    "beta":  (0.05, 0.50),
    "gamma": (0.02, 0.35),
}

# Total absolute power range (uV^2) for a realistic EEG epoch
_TOTAL_POWER_RANGE: Tuple[float, float] = (5.0, 5000.0)


# ==============================================================================
#  Core generation
# ==============================================================================

def generate_synthetic_eeg(
    duration: float = 4.0,
    fs: float = 256.0,
    n_channels: int = 4,
    band_powers: Optional[Dict[str, float]] = None,
    amplitude_uv: float = 20.0,
    aperiodic_exponent: float = 1.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate synthetic EEG with controllable spectral properties.

    Builds a signal as the sum of:
      1. 1/f aperiodic (pink-noise) background
      2. Band-specific oscillatory components at amplitudes controlled by
         *band_powers*

    Args:
        duration: Signal length in seconds.
        fs: Sampling rate (Hz).
        n_channels: Number of EEG channels to generate.
        band_powers: Dict mapping band name -> relative power (0-1).
            If None a neutral/resting-state profile is used.
        amplitude_uv: Overall RMS amplitude in micro-volts.
        aperiodic_exponent: Exponent of the 1/f background (1.0 = pink,
            2.0 = brown).  Healthy adults typically 1.0-2.0.
        seed: Optional random seed for reproducibility.

    Returns:
        ndarray of shape (n_channels, n_samples).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration * fs)

    if band_powers is None:
        band_powers = _EMOTION_PROFILES["neutral"]

    # Normalise band powers so they sum to 1
    total = sum(band_powers.values()) or 1.0
    bp_norm = {k: v / total for k, v in band_powers.items()}

    signals = np.zeros((n_channels, n_samples), dtype=np.float64)

    for ch in range(n_channels):
        # -- Aperiodic background (1/f^exponent) --
        white = rng.standard_normal(n_samples)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
        freqs[0] = 1.0  # avoid division by zero at DC
        fft_white = np.fft.rfft(white)
        pink_filter = 1.0 / (freqs ** (aperiodic_exponent / 2.0))
        pink = np.fft.irfft(fft_white * pink_filter, n=n_samples)
        # Scale aperiodic to ~30% of total amplitude
        pink *= 0.30 * amplitude_uv / (np.std(pink) + 1e-12)

        # -- Oscillatory components --
        t = np.arange(n_samples) / fs
        osc = np.zeros(n_samples, dtype=np.float64)
        for band_name, (lo, hi) in BANDS.items():
            rel = bp_norm.get(band_name, 0.0)
            if rel < 1e-6:
                continue
            # Place 2-3 sinusoidal components within the band
            n_components = rng.integers(2, 4)
            for _ in range(n_components):
                freq = rng.uniform(lo, hi)
                phase = rng.uniform(0, 2 * np.pi)
                amp = amplitude_uv * np.sqrt(rel) / np.sqrt(n_components)
                osc += amp * np.sin(2 * np.pi * freq * t + phase)

        # Combine
        combined = pink + osc
        # Scale to desired RMS
        rms = np.sqrt(np.mean(combined ** 2)) + 1e-12
        combined = combined * (amplitude_uv / rms)

        # Small inter-channel jitter (phase/amplitude variation)
        combined *= rng.uniform(0.85, 1.15)
        signals[ch] = combined

    return signals


def generate_emotion_conditioned_eeg(
    emotion: str,
    duration: float = 4.0,
    fs: float = 256.0,
    n_channels: int = 4,
    amplitude_uv: float = 20.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate EEG that would produce a specific emotion classification.

    Args:
        emotion: Target emotion label (happy, sad, angry, fear, neutral,
            surprise, relaxed, focused).
        duration, fs, n_channels, amplitude_uv, seed: Forwarded to
            :func:`generate_synthetic_eeg`.

    Returns:
        Dict with keys ``signals``, ``emotion``, ``band_profile``,
        ``n_channels``, ``n_samples``, ``fs``.

    Raises:
        ValueError: If *emotion* is not in the known profiles.
    """
    emotion_lower = emotion.lower().strip()
    if emotion_lower not in _EMOTION_PROFILES:
        raise ValueError(
            f"Unknown emotion '{emotion}'. "
            f"Choose from: {sorted(_EMOTION_PROFILES)}"
        )

    profile = _EMOTION_PROFILES[emotion_lower]
    signals = generate_synthetic_eeg(
        duration=duration,
        fs=fs,
        n_channels=n_channels,
        band_powers=profile,
        amplitude_uv=amplitude_uv,
        seed=seed,
    )

    return {
        "signals": signals,
        "emotion": emotion_lower,
        "band_profile": profile,
        "n_channels": n_channels,
        "n_samples": signals.shape[1],
        "fs": fs,
    }


# ==============================================================================
#  Artifact injection
# ==============================================================================

def inject_artifacts(
    signals: np.ndarray,
    fs: float = 256.0,
    blink_rate: float = 0.3,
    muscle_rate: float = 0.1,
    electrode_pop_rate: float = 0.05,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Inject realistic artifacts into EEG signals.

    Modifies a *copy* of the input -- the original is not mutated.

    Artifact types:
      - **Eye blinks**: large (~100-300 uV) slow deflections lasting
        200-400 ms, primarily on frontal channels.
      - **Muscle artifacts**: high-frequency (20-60 Hz) bursts lasting
        50-200 ms.
      - **Electrode pops**: sharp transients (1-5 ms) with exponential
        recovery.

    Args:
        signals: (n_channels, n_samples) EEG array.
        fs: Sampling rate.
        blink_rate: Mean blinks per second (Poisson rate).
        muscle_rate: Mean muscle bursts per second.
        electrode_pop_rate: Mean electrode pops per second.
        seed: Random seed.

    Returns:
        Dict with ``signals`` (corrupted copy), ``artifact_log`` (list of
        injected artifact descriptions), ``n_artifacts``.
    """
    rng = np.random.default_rng(seed)
    out = signals.copy().astype(np.float64)
    n_ch, n_samples = out.shape
    duration = n_samples / fs
    artifact_log: List[Dict[str, Any]] = []

    # -- Eye blinks --
    n_blinks = rng.poisson(blink_rate * duration)
    for _ in range(n_blinks):
        onset = rng.integers(0, max(1, n_samples - int(0.4 * fs)))
        blink_dur = int(rng.uniform(0.2, 0.4) * fs)
        blink_dur = min(blink_dur, n_samples - onset)
        amp = rng.uniform(100, 300)
        # Half-sine shape
        t_blink = np.linspace(0, np.pi, blink_dur)
        waveform = amp * np.sin(t_blink)
        # Apply primarily to frontal channels (first half)
        frontal_channels = list(range(min(2, n_ch)))
        for ch in frontal_channels:
            out[ch, onset: onset + blink_dur] += waveform
        artifact_log.append({
            "type": "eye_blink",
            "onset_sample": int(onset),
            "duration_samples": int(blink_dur),
            "amplitude_uv": float(amp),
            "channels": frontal_channels,
        })

    # -- Muscle artifacts --
    n_muscle = rng.poisson(muscle_rate * duration)
    for _ in range(n_muscle):
        onset = rng.integers(0, max(1, n_samples - int(0.2 * fs)))
        burst_dur = int(rng.uniform(0.05, 0.2) * fs)
        burst_dur = min(burst_dur, n_samples - onset)
        amp = rng.uniform(20, 80)
        # High-frequency burst (20-60 Hz)
        t_burst = np.arange(burst_dur) / fs
        freq = rng.uniform(20, 60)
        envelope = np.hanning(burst_dur)
        waveform = amp * envelope * np.sin(2 * np.pi * freq * t_burst)
        ch = rng.integers(0, n_ch)
        out[ch, onset: onset + burst_dur] += waveform
        artifact_log.append({
            "type": "muscle",
            "onset_sample": int(onset),
            "duration_samples": int(burst_dur),
            "amplitude_uv": float(amp),
            "channels": [int(ch)],
        })

    # -- Electrode pops --
    n_pops = rng.poisson(electrode_pop_rate * duration)
    for _ in range(n_pops):
        onset = rng.integers(0, max(1, n_samples - int(0.05 * fs)))
        pop_dur = int(rng.uniform(0.001, 0.005) * fs)
        pop_dur = max(pop_dur, 2)
        pop_dur = min(pop_dur, n_samples - onset)
        amp = rng.uniform(200, 500) * rng.choice([-1, 1])
        # Exponential decay
        decay = np.exp(-np.linspace(0, 5, pop_dur))
        waveform = amp * decay
        ch = rng.integers(0, n_ch)
        out[ch, onset: onset + pop_dur] += waveform
        artifact_log.append({
            "type": "electrode_pop",
            "onset_sample": int(onset),
            "duration_samples": int(pop_dur),
            "amplitude_uv": float(abs(amp)),
            "channels": [int(ch)],
        })

    return {
        "signals": out,
        "artifact_log": artifact_log,
        "n_artifacts": len(artifact_log),
    }


# ==============================================================================
#  Augmentation pipeline
# ==============================================================================

def augment_eeg(
    signals: np.ndarray,
    fs: float = 256.0,
    n_augmentations: int = 5,
    time_shift: bool = True,
    amplitude_scale: bool = True,
    additive_noise: bool = True,
    band_perturbation: bool = True,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Produce augmented variants of real EEG data.

    Each augmentation applies a random subset of the enabled transforms.

    Args:
        signals: (n_channels, n_samples) original EEG.
        fs: Sampling rate.
        n_augmentations: Number of augmented copies to produce.
        time_shift: Enable random circular time shift.
        amplitude_scale: Enable random per-channel amplitude scaling.
        additive_noise: Enable Gaussian noise injection.
        band_perturbation: Enable band-specific power perturbation.
        seed: Random seed.

    Returns:
        List of dicts, each with ``signals`` (augmented array) and
        ``transforms`` (list of applied transform descriptions).
    """
    rng = np.random.default_rng(seed)
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    n_ch, n_samples = signals.shape
    results: List[Dict[str, Any]] = []

    for i in range(n_augmentations):
        aug = signals.copy()
        transforms: List[str] = []

        # -- Time shift (circular) --
        if time_shift:
            max_shift = int(0.1 * n_samples)  # up to 10%
            shift = rng.integers(-max_shift, max_shift + 1)
            aug = np.roll(aug, shift, axis=-1)
            transforms.append(f"time_shift={shift}")

        # -- Amplitude scaling --
        if amplitude_scale:
            for ch in range(n_ch):
                scale = rng.uniform(0.8, 1.2)
                aug[ch] *= scale
            transforms.append("amplitude_scale")

        # -- Additive Gaussian noise --
        if additive_noise:
            noise_std = rng.uniform(0.05, 0.15) * np.std(aug)
            aug += rng.standard_normal(aug.shape) * noise_std
            transforms.append(f"additive_noise_std={noise_std:.4f}")

        # -- Band-power perturbation --
        if band_perturbation:
            # Boost or attenuate a random band by filtering and mixing back
            band_name = rng.choice(list(BANDS.keys()))
            lo, hi = BANDS[band_name]
            gain = rng.uniform(0.7, 1.3)
            try:
                sos = scipy_signal.butter(
                    4, [lo, min(hi, fs / 2 - 1)], btype="bandpass",
                    fs=fs, output="sos",
                )
                for ch in range(n_ch):
                    band_component = scipy_signal.sosfiltfilt(sos, aug[ch])
                    aug[ch] += band_component * (gain - 1.0)
                transforms.append(f"band_perturb_{band_name}_gain={gain:.2f}")
            except Exception:
                pass  # skip if filter design fails (edge-case fs)

        results.append({
            "signals": aug,
            "transforms": transforms,
            "augmentation_index": i,
        })

    return results


# ==============================================================================
#  Quality validation
# ==============================================================================

@dataclass
class GenerationStats:
    """Quality statistics for a batch of synthetic EEG signals."""

    n_signals: int = 0
    n_passed: int = 0
    n_failed: int = 0
    mean_band_powers: Dict[str, float] = field(default_factory=dict)
    power_in_range: Dict[str, bool] = field(default_factory=dict)
    total_power: float = 0.0
    total_power_valid: bool = False
    spectral_entropy: float = 0.0
    is_valid: bool = False
    failure_reasons: List[str] = field(default_factory=list)


def validate_synthetic_quality(
    signals: np.ndarray,
    fs: float = 256.0,
) -> GenerationStats:
    """Validate that synthetic EEG has realistic spectral properties.

    Checks:
      - Relative band powers fall within physiological ranges
      - Total power is within realistic bounds
      - Spectral entropy is reasonable (not too flat, not too peaked)

    Args:
        signals: (n_channels, n_samples) or (n_samples,) EEG array.
        fs: Sampling rate.

    Returns:
        :class:`GenerationStats` with validation results.
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    n_ch, n_samples = signals.shape
    stats = GenerationStats(n_signals=n_ch)

    # Compute PSD via Welch for each channel, average across channels
    nperseg = min(n_samples, int(fs * 2))
    all_psd = []
    for ch in range(n_ch):
        f, pxx = scipy_signal.welch(signals[ch], fs=fs, nperseg=nperseg)
        all_psd.append(pxx)
    mean_psd = np.mean(all_psd, axis=0)

    # Band powers
    band_abs: Dict[str, float] = {}
    for band_name, (lo, hi) in BANDS.items():
        mask = (f >= lo) & (f <= hi)
        if mask.any():
            band_abs[band_name] = float(_trapezoid(mean_psd[mask], f[mask]))
        else:
            band_abs[band_name] = 0.0

    total_power = sum(band_abs.values()) or 1e-12
    stats.total_power = float(total_power)

    # Relative band powers
    rel_powers: Dict[str, float] = {}
    for band_name in BANDS:
        rel_powers[band_name] = band_abs[band_name] / total_power

    stats.mean_band_powers = {k: round(v, 4) for k, v in rel_powers.items()}

    # Check ranges
    failure_reasons: List[str] = []
    power_in_range: Dict[str, bool] = {}
    for band_name, rel in rel_powers.items():
        lo, hi = _VALID_RANGES[band_name]
        in_range = lo <= rel <= hi
        power_in_range[band_name] = in_range
        if not in_range:
            failure_reasons.append(
                f"{band_name} relative power {rel:.3f} outside "
                f"range [{lo}, {hi}]"
            )

    stats.power_in_range = power_in_range

    # Total power check
    stats.total_power_valid = (
        _TOTAL_POWER_RANGE[0] <= total_power <= _TOTAL_POWER_RANGE[1]
    )
    if not stats.total_power_valid:
        failure_reasons.append(
            f"Total power {total_power:.1f} outside range "
            f"[{_TOTAL_POWER_RANGE[0]}, {_TOTAL_POWER_RANGE[1]}]"
        )

    # Spectral entropy
    psd_norm = mean_psd / (np.sum(mean_psd) + 1e-12)
    psd_norm = psd_norm[psd_norm > 0]
    se = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-20)))
    max_se = np.log2(len(psd_norm)) if len(psd_norm) > 0 else 1.0
    stats.spectral_entropy = round(se / max_se, 4) if max_se > 0 else 0.0

    if stats.spectral_entropy < 0.3:
        failure_reasons.append(
            f"Spectral entropy {stats.spectral_entropy:.3f} too low "
            f"(signal too peaked/narrow)"
        )
    elif stats.spectral_entropy > 0.98:
        failure_reasons.append(
            f"Spectral entropy {stats.spectral_entropy:.3f} too high "
            f"(signal too flat/white)"
        )

    stats.failure_reasons = failure_reasons
    stats.is_valid = len(failure_reasons) == 0
    stats.n_passed = 1 if stats.is_valid else 0
    stats.n_failed = 0 if stats.is_valid else 1

    return stats


def compute_generation_stats(
    signals_batch: List[np.ndarray],
    fs: float = 256.0,
) -> GenerationStats:
    """Compute aggregate quality stats over a batch of synthetic signals.

    Args:
        signals_batch: List of (n_channels, n_samples) arrays.
        fs: Sampling rate.

    Returns:
        Aggregated :class:`GenerationStats`.
    """
    if not signals_batch:
        return GenerationStats()

    agg = GenerationStats(n_signals=len(signals_batch))
    all_band_powers: Dict[str, List[float]] = {b: [] for b in BANDS}
    total_powers: List[float] = []
    entropies: List[float] = []
    all_failures: List[str] = []
    passed = 0
    failed = 0

    for sig in signals_batch:
        st = validate_synthetic_quality(sig, fs=fs)
        for b in BANDS:
            all_band_powers[b].append(st.mean_band_powers.get(b, 0.0))
        total_powers.append(st.total_power)
        entropies.append(st.spectral_entropy)
        if st.is_valid:
            passed += 1
        else:
            failed += 1
            all_failures.extend(st.failure_reasons)

    agg.n_passed = passed
    agg.n_failed = failed
    agg.mean_band_powers = {
        b: round(float(np.mean(vals)), 4) for b, vals in all_band_powers.items()
    }
    agg.total_power = float(np.mean(total_powers))
    agg.total_power_valid = (
        _TOTAL_POWER_RANGE[0] <= agg.total_power <= _TOTAL_POWER_RANGE[1]
    )
    agg.spectral_entropy = round(float(np.mean(entropies)), 4)
    agg.is_valid = failed == 0
    agg.failure_reasons = all_failures[:20]  # cap to avoid huge responses
    agg.power_in_range = {
        b: all(
            _VALID_RANGES[b][0] <= v <= _VALID_RANGES[b][1]
            for v in all_band_powers[b]
        )
        for b in BANDS
    }

    return agg


def stats_to_dict(stats: GenerationStats) -> Dict[str, Any]:
    """Convert a :class:`GenerationStats` to a JSON-serialisable dict."""
    return {
        "n_signals": stats.n_signals,
        "n_passed": stats.n_passed,
        "n_failed": stats.n_failed,
        "mean_band_powers": stats.mean_band_powers,
        "power_in_range": stats.power_in_range,
        "total_power": round(stats.total_power, 4),
        "total_power_valid": stats.total_power_valid,
        "spectral_entropy": stats.spectral_entropy,
        "is_valid": stats.is_valid,
        "failure_reasons": stats.failure_reasons,
    }
