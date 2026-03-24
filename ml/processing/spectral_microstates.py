"""Spectral microstate analysis for 4-channel consumer EEG.

Adapted from classical microstate analysis (which requires 32+ channels).
Defines microstates by dominant frequency band per 250ms window, then
extracts temporal dynamics: coverage, duration, occurrence, transitions,
and information-theoretic measures of transition complexity.

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
    5. Transition entropy -- entropy rate, excess entropy, LZ complexity

Transition entropy features (improvement #40):
    - Entropy rate: conditional entropy of next state given current state.
      High = chaotic/random transitions (anxiety, agitation).
      Low = predictable/stable transitions (calm, focused).
    - Excess entropy: H_1 - H_rate. Measures temporal structure beyond
      single-symbol statistics. High = structured transition patterns.
    - Lempel-Ziv complexity: normalized complexity of the state sequence.
      High = irregular switching. Low = repetitive patterns.

References:
    - Weng et al. (2025, Sleep Med): entropy rate + excess entropy from
      microstate transitions → 82.8% insomnia vs controls accuracy.
    - Bai et al. (2024): unique transition pairs distinguish positive vs
      negative emotions in VR EEG microstate analysis.
    - von Wegner et al. (2018, Front Comput Neurosci): information-theoretic
      framework for microstate sequence analysis.
    - Frontiers in Neuroscience (2024): 70-75% DEAP accuracy with
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


def _lempel_ziv_complexity(sequence: List[str]) -> float:
    """Compute normalized Lempel-Ziv complexity of a symbol sequence.

    Uses the LZ76 algorithm: scan left to right, increment complexity
    counter each time a new subsequence is encountered. Normalize by
    n / log_k(n) where k = alphabet size and n = sequence length, so
    the result is in [0, 1] with 1 = maximally complex (random).

    Args:
        sequence: List of state labels.

    Returns:
        Normalized LZ complexity in [0, 1].
    """
    n = len(sequence)
    if n <= 1:
        return 0.0

    # Count unique symbols actually present
    alphabet_size = len(set(sequence))
    if alphabet_size <= 1:
        return 0.0

    # LZ76 complexity count
    s = "".join(sequence)
    complexity = 1
    i = 0
    k = 1
    while i + k <= n:
        # Check if s[i:i+k] has appeared in s[0:i+k-1]
        substring = s[i: i + k]
        search_space = s[: i + k - 1]  # exclude last char to avoid trivial match
        if substring in search_space:
            k += 1
            if i + k > n:
                complexity += 1
                break
        else:
            complexity += 1
            i += k
            k = 1
    # Normalize: theoretical upper bound for iid sequence is
    # n / log_k(n) where k = alphabet_size
    log_base = np.log(n) / np.log(alphabet_size) if alphabet_size > 1 else n
    normalized = complexity / (n / log_base) if log_base > 0 else 0.0
    return float(min(normalized, 1.0))


def compute_transition_entropy(sequence: List[str]) -> Dict:
    """Compute information-theoretic features from a microstate sequence.

    Given a sequence of microstate labels, computes:

    1. **Entropy rate** of the first-order Markov chain:
       H_rate = -sum_i pi_i * sum_j P(j|i) * log2(P(j|i))
       where pi is the stationary distribution and P is the transition
       matrix. Measures unpredictability of the next state given the
       current state.

    2. **Excess entropy** (also called predictive information):
       E = H_1 - H_rate
       where H_1 is the single-symbol Shannon entropy. Measures how much
       temporal structure exists beyond what single-symbol statistics
       capture. High excess entropy = structured transition patterns.

    3. **Lempel-Ziv complexity**: normalized LZ76 complexity of the
       raw sequence. Model-free measure of sequence regularity.

    Args:
        sequence: List of microstate labels (e.g., ["A", "B", "A", "T"]).

    Returns:
        Dict with keys: entropy_rate, excess_entropy, lz_complexity.
        All values are non-negative floats (except excess_entropy which
        can theoretically be slightly negative for very short sequences).

    References:
        Weng et al. (2025, Sleep Med): entropy rate + excess entropy
        from microstate transitions → 82.8% insomnia classification.
        von Wegner et al. (2018, Front Comput Neurosci): information-
        theoretic framework for EEG microstate sequences.
    """
    if len(sequence) < 2:
        return {"entropy_rate": 0.0, "excess_entropy": 0.0, "lz_complexity": 0.0}

    # Map labels to integer indices for the unique states present
    unique_states = sorted(set(sequence))
    n_states = len(unique_states)
    state_to_idx = {s: i for i, s in enumerate(unique_states)}

    if n_states < 2:
        # Only one state ever appears -- no transitions, zero entropy
        return {"entropy_rate": 0.0, "excess_entropy": 0.0, "lz_complexity": 0.0}

    # Build transition count matrix
    trans_counts = np.zeros((n_states, n_states))
    for i in range(len(sequence) - 1):
        from_idx = state_to_idx[sequence[i]]
        to_idx = state_to_idx[sequence[i + 1]]
        trans_counts[from_idx, to_idx] += 1

    # Normalize to transition probability matrix P(j | i)
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    P = trans_counts / row_sums

    # Stationary distribution: left eigenvector of P with eigenvalue 1.
    # For a finite ergodic Markov chain, this is the normalized row sums
    # of the transition counts (empirical stationary distribution).
    total_transitions = trans_counts.sum()
    if total_transitions == 0:
        return {"entropy_rate": 0.0, "excess_entropy": 0.0, "lz_complexity": 0.0}
    pi = trans_counts.sum(axis=1) / total_transitions

    # Entropy rate: H_rate = -sum_i pi_i * sum_j P_ij * log2(P_ij)
    entropy_rate = 0.0
    for i in range(n_states):
        if pi[i] > 0:
            for j in range(n_states):
                if P[i, j] > 0:
                    entropy_rate -= pi[i] * P[i, j] * np.log2(P[i, j])

    # Single-symbol Shannon entropy: H_1 = -sum_i p_i * log2(p_i)
    # where p_i is the marginal frequency of each state
    n = len(sequence)
    freqs = np.array([sequence.count(s) / n for s in unique_states])
    h1 = -float(np.sum(freqs * np.log2(freqs + 1e-15)))

    # Excess entropy: E = H_1 - H_rate
    # Measures how much temporal structure beyond marginal statistics
    excess_entropy = max(0.0, h1 - entropy_rate)

    # Lempel-Ziv complexity
    lz = _lempel_ziv_complexity(sequence)

    return {
        "entropy_rate": round(float(entropy_rate), 6),
        "excess_entropy": round(float(excess_entropy), 6),
        "lz_complexity": round(float(lz), 6),
    }


def extract_microstate_features(
    signals: np.ndarray, fs: int = 256, window_ms: int = 250
) -> Dict:
    """Extract temporal microstate features from EEG data.

    Computes the microstate sequence, then derives coverage, average
    duration, occurrence rate, transition probabilities, transition
    entropy features, and summary statistics. Returns a flat 31-element
    feature vector suitable for ML pipelines alongside human-readable
    breakdowns.

    Feature vector layout (31 elements):
        [0:4]   coverage for D, T, A, B
        [4:8]   avg_duration for D, T, A, B (seconds)
        [8:12]  occurrence for D, T, A, B (transitions/sec)
        [12:28] transition matrix flattened row-major (4x4)
        [28]    entropy_rate (Markov chain entropy rate)
        [29]    excess_entropy (H_1 - H_rate)
        [30]    lz_complexity (normalized Lempel-Ziv)

    Args:
        signals: EEG data, shape (n_channels, n_samples) or (n_samples,).
        fs: Sampling rate in Hz.
        window_ms: Window duration in milliseconds.

    Returns:
        Dict with keys: coverage, avg_duration, occurrence,
        transition_matrix, transition_entropy, dominant_state,
        state_diversity, feature_vector, n_features, sequence_length.
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

    # Transition entropy features (improvement #40)
    te = compute_transition_entropy(sequence)

    # Flatten to feature vector
    # 4 coverage + 4 duration + 4 occurrence + 16 transitions + 3 entropy = 31
    feature_vector = (
        [coverage[s] for s in STATE_NAMES]
        + [avg_duration[s] for s in STATE_NAMES]
        + [occurrence[s] for s in STATE_NAMES]
        + transitions.flatten().tolist()
        + [te["entropy_rate"], te["excess_entropy"], te["lz_complexity"]]
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
        "transition_entropy": te,
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
        "transition_entropy": {
            "entropy_rate": 0.0,
            "excess_entropy": 0.0,
            "lz_complexity": 0.0,
        },
        "dominant_state": "A",
        "state_diversity": 0.0,
        "feature_vector": [0.0] * 31,
        "n_features": 31,
        "sequence_length": 0,
    }
