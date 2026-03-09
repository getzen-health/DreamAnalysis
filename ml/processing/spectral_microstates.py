"""Spectral microstate analysis for 4-channel consumer EEG.

Adapted from classical microstate analysis (which requires 32+ channels).
Defines microstates by dominant frequency band per 250ms window, then
extracts temporal dynamics: coverage, duration, occurrence, transitions.

Classical microstates cluster spatial topographies across 32+ electrodes.
With only 4 Muse 2 channels, spatial clustering is not viable. Instead,
we classify each short time window by its dominant spectral band and
analyze the temporal dynamics of the resulting state sequence.

Microstate classes:
    D = delta dominant (0.5-4 Hz)  -- drowsiness, deep relaxation
    T = theta dominant (4-8 Hz)    -- meditation, creativity
    A = alpha dominant (8-12 Hz)   -- calm, relaxation
    B = beta dominant (12-30 Hz)   -- active thinking, stress

From the state sequence (e.g., AAAT -> ABBB -> BBBA -> AAAA), we extract:
    1. Coverage     -- fraction of time in each state
    2. Duration     -- average consecutive time in each state (seconds)
    3. Occurrence   -- transitions per second into each state
    4. Transitions  -- 4x4 probability matrix P(next | current)

These temporal features capture dynamics that epoch-averaged band powers
miss entirely.

Reference: Frontiers in Neuroscience (2024) -- 70-75% DEAP accuracy with
microstate-derived features alone.

Dependencies: numpy, scipy (already in the project).
"""

import numpy as np
from typing import Dict, List
from scipy.signal import welch


MICROSTATE_BANDS = {
    "D": (0.5, 4.0),    # delta
    "T": (4.0, 8.0),    # theta
    "A": (8.0, 12.0),   # alpha
    "B": (12.0, 30.0),  # beta
}
STATE_NAMES = list(MICROSTATE_BANDS.keys())  # ["D", "T", "A", "B"]


def classify_window(signal: np.ndarray, fs: int = 256) -> str:
    """Classify a short EEG window by its dominant frequency band.

    Computes the Welch PSD of the window and sums power in each of the
    four microstate bands (delta, theta, alpha, beta). Returns the label
    of the band with the highest total power.

    Args:
        signal: 1D EEG signal for a single window.
        fs: Sampling rate in Hz.

    Returns:
        One of "D", "T", "A", "B".
    """
    # Use the full window length for maximum frequency resolution.
    # With 64 samples (250ms @ 256Hz), resolution is 4 Hz -- adequate for
    # separating delta/theta/alpha/beta. Shorter windows get even coarser
    # resolution but still produce a usable dominant-band classification.
    nperseg = len(signal)
    if nperseg < 4:
        return "A"  # default when window is too short for meaningful PSD

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    band_powers = {}
    for state, (lo, hi) in MICROSTATE_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_powers[state] = float(np.sum(psd[mask])) if np.any(mask) else 0.0

    return max(band_powers, key=band_powers.get)


def extract_microstate_sequence(
    signals: np.ndarray, fs: int = 256, window_ms: int = 250
) -> List[str]:
    """Convert multichannel EEG into a sequence of spectral microstates.

    Divides the signal into non-overlapping windows of ``window_ms``
    milliseconds. For each window, averages across channels and classifies
    by dominant frequency band.

    Args:
        signals: EEG data, shape (n_channels, n_samples) or (n_samples,).
        fs: Sampling rate in Hz.
        window_ms: Window duration in milliseconds.

    Returns:
        List of state labels, one per window (e.g., ["A", "A", "B", "T"]).
    """
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    window_samples = int(window_ms * fs / 1000)
    n_windows = signals.shape[1] // window_samples

    sequence: List[str] = []
    for w in range(n_windows):
        start = w * window_samples
        end = start + window_samples
        # Average band powers across channels, then classify
        avg_signal = np.mean(signals[:, start:end], axis=0)
        state = classify_window(avg_signal, fs)
        sequence.append(state)

    return sequence


def extract_microstate_features(
    signals: np.ndarray, fs: int = 256, window_ms: int = 250
) -> Dict:
    """Extract temporal microstate features from EEG data.

    Computes the microstate sequence, then derives coverage, average
    duration, occurrence rate, transition probabilities, and summary
    statistics. Returns a flat 28-element feature vector suitable for
    ML pipelines alongside the human-readable breakdowns.

    Feature vector layout (28 elements):
        [0:4]   coverage for D, T, A, B
        [4:8]   avg_duration for D, T, A, B (seconds)
        [8:12]  occurrence for D, T, A, B (transitions/sec)
        [12:28] transition matrix flattened row-major (4x4)

    Args:
        signals: EEG data, shape (n_channels, n_samples) or (n_samples,).
        fs: Sampling rate in Hz.
        window_ms: Window duration in milliseconds.

    Returns:
        Dict with keys: coverage, avg_duration, occurrence,
        transition_matrix, dominant_state, state_diversity,
        feature_vector, n_features, sequence_length.
    """
    sequence = extract_microstate_sequence(signals, fs, window_ms)

    if len(sequence) < 2:
        return _empty_features()

    n = len(sequence)
    window_sec = window_ms / 1000.0
    total_duration = n * window_sec

    # Coverage: fraction of time in each state
    coverage = {}
    for s in STATE_NAMES:
        coverage[s] = sequence.count(s) / n

    # Duration: average consecutive run length in each state
    runs: Dict[str, List[float]] = {s: [] for s in STATE_NAMES}
    current_state = sequence[0]
    current_run = 1
    for i in range(1, n):
        if sequence[i] == current_state:
            current_run += 1
        else:
            runs[current_state].append(current_run * window_sec)
            current_state = sequence[i]
            current_run = 1
    # Append the final run
    runs[current_state].append(current_run * window_sec)

    avg_duration = {}
    for s in STATE_NAMES:
        avg_duration[s] = float(np.mean(runs[s])) if runs[s] else 0.0

    # Occurrence: transitions into each state per second
    occurrence = {s: 0 for s in STATE_NAMES}
    for i in range(1, n):
        if sequence[i] != sequence[i - 1]:
            occurrence[sequence[i]] += 1
    occurrence = {s: v / total_duration for s, v in occurrence.items()}

    # Transition probabilities: 4x4 matrix
    transitions = np.zeros((4, 4))
    for i in range(1, n):
        from_idx = STATE_NAMES.index(sequence[i - 1])
        to_idx = STATE_NAMES.index(sequence[i])
        transitions[from_idx][to_idx] += 1

    # Normalize rows (rows with zero counts stay zero)
    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transitions = transitions / row_sums

    # Flatten to feature vector
    # 4 coverage + 4 duration + 4 occurrence + 16 transitions = 28
    feature_vector = (
        [coverage[s] for s in STATE_NAMES]
        + [avg_duration[s] for s in STATE_NAMES]
        + [occurrence[s] for s in STATE_NAMES]
        + transitions.flatten().tolist()
    )

    return {
        "coverage": coverage,
        "avg_duration": avg_duration,
        "occurrence": occurrence,
        "transition_matrix": {
            f"{STATE_NAMES[i]}->{STATE_NAMES[j]}": round(float(transitions[i][j]), 3)
            for i in range(4)
            for j in range(4)
        },
        "dominant_state": max(coverage, key=coverage.get),
        "state_diversity": float(max(0.0,
            -sum(c * np.log2(c + 1e-10) for c in coverage.values())
        )),
        "feature_vector": [round(float(f), 4) for f in feature_vector],
        "n_features": len(feature_vector),
        "sequence_length": n,
    }


def _empty_features() -> Dict:
    """Return zero-valued feature dict when input is too short to analyze."""
    return {
        "coverage": {s: 0.25 for s in STATE_NAMES},
        "avg_duration": {s: 0.0 for s in STATE_NAMES},
        "occurrence": {s: 0.0 for s in STATE_NAMES},
        "transition_matrix": {},
        "dominant_state": "A",
        "state_diversity": 0.0,
        "feature_vector": [0.0] * 28,
        "n_features": 28,
        "sequence_length": 0,
    }
