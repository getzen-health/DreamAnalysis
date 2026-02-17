"""Physiologically Realistic EEG Simulator.

Generates synthetic EEG signals that match real brain state frequency profiles.
The same ML pipeline works for both simulated and real data.
"""

import numpy as np
from typing import Dict


# Frequency profiles for different brain states (relative power in each band)
STATE_PROFILES = {
    "rest": {
        "delta": {"power": 0.15, "freq_center": 2.0},
        "theta": {"power": 0.15, "freq_center": 6.0},
        "alpha": {"power": 0.45, "freq_center": 10.0},
        "beta": {"power": 0.15, "freq_center": 20.0},
        "gamma": {"power": 0.10, "freq_center": 40.0},
    },
    "focus": {
        "delta": {"power": 0.05, "freq_center": 2.0},
        "theta": {"power": 0.10, "freq_center": 6.0},
        "alpha": {"power": 0.15, "freq_center": 10.0},
        "beta": {"power": 0.50, "freq_center": 22.0},
        "gamma": {"power": 0.20, "freq_center": 45.0},
    },
    "rem": {
        "delta": {"power": 0.10, "freq_center": 2.5},
        "theta": {"power": 0.35, "freq_center": 5.5},
        "alpha": {"power": 0.10, "freq_center": 9.0},
        "beta": {"power": 0.30, "freq_center": 18.0},
        "gamma": {"power": 0.15, "freq_center": 35.0},
    },
    "deep_sleep": {
        "delta": {"power": 0.65, "freq_center": 1.5},
        "theta": {"power": 0.20, "freq_center": 5.0},
        "alpha": {"power": 0.05, "freq_center": 9.0},
        "beta": {"power": 0.05, "freq_center": 16.0},
        "gamma": {"power": 0.05, "freq_center": 35.0},
    },
    "light_sleep": {
        "delta": {"power": 0.25, "freq_center": 2.0},
        "theta": {"power": 0.35, "freq_center": 5.5},
        "alpha": {"power": 0.15, "freq_center": 10.0},
        "beta": {"power": 0.15, "freq_center": 14.0},
        "gamma": {"power": 0.10, "freq_center": 35.0},
    },
    "meditation": {
        "delta": {"power": 0.10, "freq_center": 2.0},
        "theta": {"power": 0.40, "freq_center": 7.0},
        "alpha": {"power": 0.35, "freq_center": 10.5},
        "beta": {"power": 0.10, "freq_center": 15.0},
        "gamma": {"power": 0.05, "freq_center": 40.0},
    },
    "stress": {
        "delta": {"power": 0.05, "freq_center": 2.0},
        "theta": {"power": 0.10, "freq_center": 6.0},
        "alpha": {"power": 0.10, "freq_center": 10.0},
        "beta": {"power": 0.55, "freq_center": 25.0},
        "gamma": {"power": 0.20, "freq_center": 50.0},
    },
}


def generate_band_signal(
    freq_center: float,
    power: float,
    duration: float,
    fs: float,
    bandwidth: float = 2.0,
) -> np.ndarray:
    """Generate a narrow-band EEG-like signal with realistic characteristics."""
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    # Multiple overlapping oscillators within the band
    signal = np.zeros(n_samples)
    n_oscillators = 3
    for i in range(n_oscillators):
        freq = freq_center + np.random.uniform(-bandwidth, bandwidth)
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = power * np.random.uniform(0.5, 1.5) / n_oscillators
        # Amplitude modulation for realism
        mod_freq = np.random.uniform(0.1, 0.5)
        modulation = 1.0 + 0.3 * np.sin(2 * np.pi * mod_freq * t + np.random.uniform(0, 2 * np.pi))
        signal += amplitude * modulation * np.sin(2 * np.pi * freq * t + phase)

    return signal


def add_artifacts(eeg: np.ndarray, fs: float, artifact_rate: float = 0.02) -> np.ndarray:
    """Add realistic artifacts: eye blinks, muscle, and electrode noise."""
    n_samples = len(eeg)
    result = eeg.copy()

    # Eye blink artifacts (low-freq, high-amplitude)
    n_blinks = int(artifact_rate * n_samples / fs)
    for _ in range(n_blinks):
        pos = np.random.randint(0, max(1, n_samples - int(fs * 0.3)))
        blink_duration = int(fs * np.random.uniform(0.15, 0.35))
        blink = np.exp(-0.5 * np.linspace(-2, 2, blink_duration) ** 2) * np.random.uniform(50, 150)
        end = min(pos + blink_duration, n_samples)
        result[pos:end] += blink[: end - pos]

    # Background noise (1/f pink noise approximation)
    white = np.random.randn(n_samples)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    freqs[0] = 1.0  # avoid division by zero
    pink_filter = 1.0 / np.sqrt(freqs)
    pink = np.fft.irfft(np.fft.rfft(white) * pink_filter, n=n_samples)
    result += pink * 2.0

    return result


def simulate_eeg(
    state: str = "rest",
    duration: float = 30.0,
    fs: float = 256.0,
    n_channels: int = 1,
    add_noise: bool = True,
) -> Dict:
    """Generate physiologically realistic EEG for a given brain state.

    Args:
        state: Brain state ('rest', 'focus', 'rem', 'deep_sleep', 'light_sleep', 'meditation', 'stress')
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        n_channels: Number of EEG channels
        add_noise: Whether to add realistic noise/artifacts

    Returns:
        Dictionary with 'signals' (n_channels x n_samples), 'fs', 'state', 'timestamps'
    """
    if state not in STATE_PROFILES:
        state = "rest"

    profile = STATE_PROFILES[state]
    n_samples = int(duration * fs)
    signals = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        channel_signal = np.zeros(n_samples)
        for band_name, params in profile.items():
            band_signal = generate_band_signal(
                freq_center=params["freq_center"],
                power=params["power"] * 50.0,  # Scale to microvolts
                duration=duration,
                fs=fs,
            )
            channel_signal += band_signal

        if add_noise:
            channel_signal = add_artifacts(channel_signal, fs)

        signals[ch] = channel_signal

    timestamps = np.arange(n_samples) / fs

    return {
        "signals": signals.tolist(),
        "fs": fs,
        "state": state,
        "duration": duration,
        "n_channels": n_channels,
        "n_samples": n_samples,
        "timestamps": timestamps.tolist(),
    }


def simulate_sleep_session(
    total_duration_hours: float = 8.0, fs: float = 256.0
) -> Dict:
    """Simulate a full night's sleep with realistic sleep stage transitions."""
    total_seconds = total_duration_hours * 3600
    epoch_duration = 30.0  # Standard 30-second sleep epochs

    # Simplified hypnogram: cycle through sleep stages
    cycle_duration = 90 * 60  # 90-minute sleep cycles
    n_cycles = int(total_seconds / cycle_duration)

    stages = []
    signals_all = []

    for cycle in range(max(n_cycles, 1)):
        # Each cycle: Light -> Deep -> Light -> REM
        cycle_stages = [
            ("light_sleep", 0.25),
            ("deep_sleep", 0.30),
            ("light_sleep", 0.15),
            ("rem", 0.30),
        ]
        # Later cycles have more REM, less deep sleep
        if cycle > 2:
            cycle_stages = [
                ("light_sleep", 0.20),
                ("deep_sleep", 0.10),
                ("light_sleep", 0.20),
                ("rem", 0.50),
            ]

        for stage_name, fraction in cycle_stages:
            stage_duration = cycle_duration * fraction
            remaining = total_seconds - len(signals_all) * epoch_duration
            if remaining <= 0:
                break
            actual_duration = min(stage_duration, remaining)
            n_epochs = max(1, int(actual_duration / epoch_duration))

            for _ in range(n_epochs):
                if len(stages) * epoch_duration >= total_seconds:
                    break
                epoch = simulate_eeg(state=stage_name, duration=epoch_duration, fs=fs)
                signals_all.append(epoch["signals"][0])
                stages.append(stage_name)

    return {
        "stages": stages,
        "epoch_duration": epoch_duration,
        "n_epochs": len(stages),
        "fs": fs,
        "total_duration_hours": len(stages) * epoch_duration / 3600,
    }
