"""Noise Augmentation for Robust EEG Training.

Simulates real-world dry electrode noise conditions:
- Gaussian sensor noise (dry electrode baseline)
- Electrode drift (DC offset changes)
- Motion artifacts (head movement, jaw clench)
- 50/60Hz powerline interference
- EMG muscle contamination
- Eye blink artifacts
- Channel dropout (loose electrode simulation)

These augmentations make models trained on lab data
generalize to consumer-grade EEG devices (Muse, Emotiv, OpenBCI).
"""

import numpy as np
from scipy import signal as scipy_signal


def add_gaussian_noise(eeg, snr_db=10.0):
    """Add Gaussian white noise at a given SNR (dB).

    Consumer dry electrodes typically have 5-15 dB SNR
    vs lab wet electrodes at 20-40 dB.
    """
    signal_power = np.mean(eeg ** 2) + 1e-10
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(*eeg.shape) * np.sqrt(noise_power)
    return eeg + noise


def add_electrode_drift(eeg, fs=128, max_drift_uv=50.0):
    """Add slow DC drift (common with dry electrodes).

    Dry electrodes have poor skin contact, causing slow
    voltage drifts over seconds.
    """
    n_samples = eeg.shape[-1] if eeg.ndim > 1 else len(eeg)
    # Random low-frequency drift (0.01 - 0.5 Hz)
    t = np.linspace(0, n_samples / fs, n_samples)
    drift_freq = np.random.uniform(0.01, 0.5)
    drift_phase = np.random.uniform(0, 2 * np.pi)
    drift_amp = np.random.uniform(0, max_drift_uv)
    drift = drift_amp * np.sin(2 * np.pi * drift_freq * t + drift_phase)

    if eeg.ndim > 1:
        # Different drift per channel
        result = eeg.copy()
        for ch in range(eeg.shape[0]):
            ch_drift_amp = np.random.uniform(0, max_drift_uv)
            ch_drift_freq = np.random.uniform(0.01, 0.5)
            ch_drift = ch_drift_amp * np.sin(2 * np.pi * ch_drift_freq * t + np.random.uniform(0, 2 * np.pi))
            result[ch] += ch_drift
        return result
    return eeg + drift


def add_motion_artifact(eeg, fs=128, n_artifacts=3, max_amplitude=200.0):
    """Add sudden motion artifacts (head movement, jaw clench).

    Appears as high-amplitude, short-duration transients.
    """
    result = eeg.copy()
    n_samples = eeg.shape[-1] if eeg.ndim > 1 else len(eeg)

    for _ in range(n_artifacts):
        # Random position and duration (50-500ms)
        duration = int(np.random.uniform(0.05, 0.5) * fs)
        start = np.random.randint(0, max(1, n_samples - duration))
        amplitude = np.random.uniform(50, max_amplitude)

        # Create artifact shape (sharp rise, exponential decay)
        t_art = np.linspace(0, 1, duration)
        artifact = amplitude * t_art * np.exp(-3 * t_art) * np.random.choice([-1, 1])

        if eeg.ndim > 1:
            # Affect random subset of channels
            n_affected = np.random.randint(1, eeg.shape[0] + 1)
            affected = np.random.choice(eeg.shape[0], n_affected, replace=False)
            for ch in affected:
                result[ch, start:start + duration] += artifact[:min(duration, n_samples - start)]
        else:
            result[start:start + duration] += artifact[:min(duration, n_samples - start)]

    return result


def add_powerline_noise(eeg, fs=128, freq=50.0, amplitude=5.0):
    """Add 50/60Hz powerline interference.

    Common in unshielded consumer EEG environments.
    """
    n_samples = eeg.shape[-1] if eeg.ndim > 1 else len(eeg)
    t = np.arange(n_samples) / fs
    noise = amplitude * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))

    # Add harmonics
    noise += (amplitude * 0.3) * np.sin(2 * np.pi * freq * 2 * t + np.random.uniform(0, 2 * np.pi))

    if eeg.ndim > 1:
        return eeg + noise[np.newaxis, :]
    return eeg + noise


def add_emg_contamination(eeg, fs=128, intensity=0.3):
    """Add EMG (muscle) noise — broadband high-frequency contamination.

    Common from facial muscles, neck tension. Dominates > 20Hz.
    """
    n_samples = eeg.shape[-1] if eeg.ndim > 1 else len(eeg)
    signal_std = np.std(eeg) + 1e-10

    # EMG is broadband but concentrated 20-300Hz
    emg_raw = np.random.randn(n_samples) * signal_std * intensity

    # Highpass at 20Hz to simulate EMG spectrum
    nyq = fs / 2
    if 20 / nyq < 1.0:
        b, a = scipy_signal.butter(2, 20 / nyq, btype='high')
        emg = scipy_signal.filtfilt(b, a, emg_raw)
    else:
        emg = emg_raw

    if eeg.ndim > 1:
        result = eeg.copy()
        for ch in range(eeg.shape[0]):
            ch_emg = np.random.randn(n_samples) * signal_std * intensity * np.random.uniform(0.5, 1.5)
            if 20 / nyq < 1.0:
                ch_emg = scipy_signal.filtfilt(b, a, ch_emg)
            result[ch] += ch_emg
        return result
    return eeg + emg


def add_eye_blink(eeg, fs=128, n_blinks=2, amplitude=150.0):
    """Add eye blink artifacts (frontal channels most affected).

    Blinks produce large slow positive deflections, ~200-400ms duration.
    """
    result = eeg.copy()
    n_samples = eeg.shape[-1] if eeg.ndim > 1 else len(eeg)

    for _ in range(n_blinks):
        duration = int(np.random.uniform(0.2, 0.4) * fs)
        start = np.random.randint(0, max(1, n_samples - duration))
        amp = np.random.uniform(amplitude * 0.5, amplitude * 1.5)

        # Blink shape: half-sine positive deflection
        blink = amp * np.sin(np.linspace(0, np.pi, duration))

        if eeg.ndim > 1:
            # Frontal channels affected most (first 1/3 of channels)
            n_frontal = max(1, eeg.shape[0] // 3)
            for ch in range(eeg.shape[0]):
                # Amplitude decreases for posterior channels
                scale = 1.0 if ch < n_frontal else 0.3
                end_idx = min(start + duration, n_samples)
                result[ch, start:end_idx] += blink[:end_idx - start] * scale
        else:
            end_idx = min(start + duration, n_samples)
            result[start:end_idx] += blink[:end_idx - start]

    return result


def add_channel_dropout(eeg, dropout_prob=0.1):
    """Simulate loose/disconnected electrodes by zeroing random channels.

    Common with dry electrode headsets.
    """
    if eeg.ndim < 2:
        return eeg

    result = eeg.copy()
    for ch in range(eeg.shape[0]):
        if np.random.random() < dropout_prob:
            result[ch] = 0.0
    return result


def augment_eeg(eeg, fs=128, difficulty='medium'):
    """Apply random combination of noise augmentations.

    Args:
        eeg: EEG data (channels x samples) or (samples,)
        fs: Sampling frequency
        difficulty: 'easy' (lab-like), 'medium' (good consumer), 'hard' (worst case)

    Returns:
        Augmented EEG data
    """
    configs = {
        'easy': {
            'gaussian_snr': (15, 30),
            'drift_prob': 0.2, 'drift_amp': 20,
            'motion_prob': 0.1, 'motion_n': 1,
            'powerline_prob': 0.3, 'powerline_amp': 2,
            'emg_prob': 0.1, 'emg_intensity': 0.1,
            'blink_prob': 0.2, 'blink_n': 1,
            'dropout_prob': 0.0,
        },
        'medium': {
            'gaussian_snr': (8, 20),
            'drift_prob': 0.5, 'drift_amp': 50,
            'motion_prob': 0.3, 'motion_n': 3,
            'powerline_prob': 0.5, 'powerline_amp': 5,
            'emg_prob': 0.3, 'emg_intensity': 0.3,
            'blink_prob': 0.4, 'blink_n': 2,
            'dropout_prob': 0.05,
        },
        'hard': {
            'gaussian_snr': (3, 12),
            'drift_prob': 0.8, 'drift_amp': 100,
            'motion_prob': 0.5, 'motion_n': 5,
            'powerline_prob': 0.7, 'powerline_amp': 10,
            'emg_prob': 0.5, 'emg_intensity': 0.5,
            'blink_prob': 0.6, 'blink_n': 3,
            'dropout_prob': 0.1,
        },
    }

    cfg = configs.get(difficulty, configs['medium'])
    result = eeg.copy().astype(np.float64)

    # Always add some Gaussian noise
    snr = np.random.uniform(*cfg['gaussian_snr'])
    result = add_gaussian_noise(result, snr_db=snr)

    if np.random.random() < cfg['drift_prob']:
        result = add_electrode_drift(result, fs, cfg['drift_amp'])

    if np.random.random() < cfg['motion_prob']:
        result = add_motion_artifact(result, fs, cfg['motion_n'])

    if np.random.random() < cfg['powerline_prob']:
        freq = np.random.choice([50.0, 60.0])
        result = add_powerline_noise(result, fs, freq, cfg['powerline_amp'])

    if np.random.random() < cfg['emg_prob']:
        result = add_emg_contamination(result, fs, cfg['emg_intensity'])

    if np.random.random() < cfg['blink_prob']:
        result = add_eye_blink(result, fs, cfg['blink_n'])

    if np.random.random() < cfg['dropout_prob']:
        result = add_channel_dropout(result, cfg['dropout_prob'])

    return result
