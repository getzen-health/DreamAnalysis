"""Neural Complexity Analyzer for EEG signals.

Computes multiple nonlinear complexity measures from EEG data and provides
a consciousness-level estimate based on a perturbational complexity index
(PCI) proxy. Tracks complexity over time to detect state transitions
(awake -> drowsy -> asleep).

Complexity measures:
  - Sample Entropy (SampEn): regularity measure, higher = more complex
  - Permutation Entropy (PE): ordinal pattern diversity, normalized [0,1]
  - Lempel-Ziv Complexity (LZc): algorithmic compressibility, normalized [0,1]
  - Hurst Exponent (H): long-range dependence, 0.5 = random, >0.5 = persistent
  - Higuchi Fractal Dimension (HFD): waveform complexity, range [1,2]
  - Detrended Fluctuation Analysis (DFA) exponent: scaling behavior

Consciousness estimation:
  The composite complexity_index (0-100) is a weighted blend of all metrics,
  inspired by Casali et al. (2013) perturbational complexity index. Higher
  complexity correlates with higher consciousness levels.

References:
    Casali et al. (2013) — A theoretically based index of consciousness
    Richman & Moorman (2000) — Physiological time-series analysis using
        approximate entropy and sample entropy
    Bandt & Pompe (2002) — Permutation entropy
    Lempel & Ziv (1976) — On the complexity of finite sequences
    Higuchi (1988) — Approach to an irregular time series
    Peng et al. (1994) — Mosaic organization of DNA nucleotides

Usage:
    analyzer = NeuralComplexityAnalyzer()
    result = analyzer.analyze(eeg_signals, fs=256)
    # result["complexity_index"] -> 0-100
    # result["consciousness_level"] -> "conscious"/"reduced"/"minimal"
    estimate = analyzer.get_consciousness_estimate()
    stats = analyzer.get_session_stats()
    history = analyzer.get_history()
    analyzer.reset()
"""

import math
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal


CONSCIOUSNESS_LEVELS = ("conscious", "reduced", "minimal")

# Thresholds for consciousness classification (complexity_index)
_CONSCIOUS_THRESHOLD = 45
_REDUCED_THRESHOLD = 20

# Transition detection threshold (absolute change in complexity_index)
_TRANSITION_THRESHOLD = 20.0

# DFA and Hurst defaults
_DFA_MIN_WINDOW = 4
_DFA_MAX_WINDOW_FRAC = 0.25


class NeuralComplexityAnalyzer:
    """Compute nonlinear complexity measures from EEG and estimate
    consciousness level.

    Maintains an internal history of complexity readings for session
    statistics and state-transition detection.

    Thread safety: not thread-safe. Use one instance per worker or wrap
    calls with a lock in production.
    """

    def __init__(self) -> None:
        self._history: List[Dict] = []
        self._last_complexity_index: Optional[float] = None
        self._last_consciousness_level: Optional[str] = None

    # ── Public API ──────────────────────────────────────────────────────

    def analyze(
        self,
        eeg_signals: np.ndarray,
        fs: float = 256.0,
    ) -> Dict:
        """Compute all complexity measures from EEG.

        Args:
            eeg_signals: EEG data. Accepts:
                - 1D array (n_samples,) for single channel
                - 2D array (n_channels, n_samples) for multichannel
            fs: Sampling rate in Hz.

        Returns:
            Dict with keys:
                sample_entropy, permutation_entropy, lz_complexity,
                hurst_exponent, fractal_dimension, dfa_exponent,
                complexity_index (0-100), consciousness_level (str),
                state_transition_detected (bool)
        """
        signals = np.asarray(eeg_signals, dtype=np.float64)

        # Normalize shape to (n_channels, n_samples)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)
        elif signals.ndim == 2 and signals.shape[0] > signals.shape[1]:
            # Likely (n_samples, n_channels) — transpose
            signals = signals.T

        n_channels, n_samples = signals.shape

        # Compute per-channel metrics, then average
        all_se = []
        all_pe = []
        all_lz = []
        all_hurst = []
        all_fd = []
        all_dfa = []

        for ch in range(n_channels):
            x = signals[ch]
            all_se.append(_sample_entropy(x))
            all_pe.append(_permutation_entropy(x))
            all_lz.append(_lempel_ziv_complexity(x))
            all_hurst.append(_hurst_exponent(x))
            all_fd.append(_higuchi_fractal_dimension(x))
            all_dfa.append(_dfa_exponent(x, fs))

        sample_entropy = float(np.mean(all_se))
        permutation_entropy = float(np.mean(all_pe))
        lz_complexity = float(np.mean(all_lz))
        hurst_exponent = float(np.clip(np.mean(all_hurst), 0.0, 1.0))
        fractal_dimension = float(np.clip(np.mean(all_fd), 1.0, 2.0))
        dfa_exponent = float(np.mean(all_dfa))

        # Composite complexity index (0-100)
        complexity_index = self._compute_composite(
            sample_entropy,
            permutation_entropy,
            lz_complexity,
            hurst_exponent,
            fractal_dimension,
            dfa_exponent,
        )

        # Consciousness level
        consciousness_level = self._classify_consciousness(complexity_index)

        # State transition detection
        state_transition_detected = False
        if self._last_complexity_index is not None:
            delta = abs(complexity_index - self._last_complexity_index)
            if delta >= _TRANSITION_THRESHOLD:
                state_transition_detected = True

        # Update internal state
        self._last_complexity_index = complexity_index
        self._last_consciousness_level = consciousness_level

        result = {
            "sample_entropy": round(sample_entropy, 6),
            "permutation_entropy": round(permutation_entropy, 6),
            "lz_complexity": round(lz_complexity, 6),
            "hurst_exponent": round(hurst_exponent, 6),
            "fractal_dimension": round(fractal_dimension, 6),
            "dfa_exponent": round(dfa_exponent, 6),
            "complexity_index": round(complexity_index, 2),
            "consciousness_level": consciousness_level,
            "state_transition_detected": state_transition_detected,
        }

        self._history.append(result.copy())
        return result

    def get_consciousness_estimate(self) -> Optional[Dict]:
        """Return the latest consciousness estimate.

        Returns:
            Dict with consciousness_level and complexity_index, or None
            if no analysis has been performed yet.
        """
        if self._last_consciousness_level is None:
            return None
        return {
            "consciousness_level": self._last_consciousness_level,
            "complexity_index": self._last_complexity_index,
        }

    def get_session_stats(self) -> Dict:
        """Return aggregate statistics for all analyses in this session.

        Returns:
            Dict with n_epochs, mean_complexity, std_complexity,
            min_complexity, max_complexity. If no epochs recorded,
            n_epochs=0 and other fields are absent.
        """
        n = len(self._history)
        if n == 0:
            return {"n_epochs": 0}

        indices = [entry["complexity_index"] for entry in self._history]
        return {
            "n_epochs": n,
            "mean_complexity": round(float(np.mean(indices)), 4),
            "std_complexity": round(float(np.std(indices)), 4),
            "min_complexity": round(float(np.min(indices)), 4),
            "max_complexity": round(float(np.max(indices)), 4),
        }

    def get_history(self) -> List[Dict]:
        """Return full chronological history of analysis results.

        Returns:
            List of dicts, each from an analyze() call, in order.
        """
        return list(self._history)

    def reset(self) -> None:
        """Clear all internal state (history, last estimate)."""
        self._history.clear()
        self._last_complexity_index = None
        self._last_consciousness_level = None

    # ── Private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _compute_composite(
        se: float,
        pe: float,
        lz: float,
        hurst: float,
        fd: float,
        dfa: float,
    ) -> float:
        """Blend individual metrics into a 0-100 composite index.

        Weights inspired by PCI proxy: entropy and compressibility measures
        dominate, with fractal/scaling measures as secondary indicators.

        Each metric is first mapped to a [0, 1] contribution, then the
        weighted sum is scaled to [0, 100].
        """
        # Map sample entropy to [0, 1]: SampEn of 0 = regular, ~2+ = complex
        se_norm = float(np.clip(se / 2.5, 0.0, 1.0))

        # PE is already [0, 1]
        pe_norm = float(np.clip(pe, 0.0, 1.0))

        # LZ is already [0, 1]
        lz_norm = float(np.clip(lz, 0.0, 1.0))

        # Hurst: 0.5 = random (max complexity), 0 or 1 = deterministic
        # Map deviation from 0.5 as anti-complexity indicator
        hurst_contrib = 1.0 - 2.0 * abs(hurst - 0.5)
        hurst_norm = float(np.clip(hurst_contrib, 0.0, 1.0))

        # Fractal dimension: 1.0 = simple, 2.0 = maximally complex
        fd_norm = float(np.clip((fd - 1.0), 0.0, 1.0))

        # DFA: ~0.5 = white noise (high complexity), ~1.5 = very correlated
        # Map closer to 0.5 as higher complexity
        dfa_contrib = 1.0 - abs(dfa - 0.5) / 1.0
        dfa_norm = float(np.clip(dfa_contrib, 0.0, 1.0))

        # Weighted sum
        composite = (
            0.25 * se_norm
            + 0.20 * pe_norm
            + 0.20 * lz_norm
            + 0.10 * hurst_norm
            + 0.15 * fd_norm
            + 0.10 * dfa_norm
        )

        return float(np.clip(composite * 100, 0.0, 100.0))

    @staticmethod
    def _classify_consciousness(complexity_index: float) -> str:
        """Classify consciousness level from composite complexity index."""
        if complexity_index >= _CONSCIOUS_THRESHOLD:
            return "conscious"
        elif complexity_index >= _REDUCED_THRESHOLD:
            return "reduced"
        return "minimal"


# ── Standalone complexity functions ─────────────────────────────────────────

def _sample_entropy(
    x: np.ndarray,
    m: int = 2,
    r_frac: float = 0.2,
) -> float:
    """Compute sample entropy (SampEn) of a time series.

    SampEn measures the conditional probability that two sequences that are
    similar for m points remain similar for m+1 points. Lower values
    indicate more regularity; higher values indicate more complexity.

    Args:
        x: 1D signal.
        m: Embedding dimension.
        r_frac: Tolerance as fraction of signal std.

    Returns:
        Sample entropy (non-negative float). Returns 0.0 for degenerate
        cases (flat signal, too-short signal).

    Reference:
        Richman & Moorman (2000), Am J Physiol Heart Circ Physiol.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)

    if n < m + 2:
        return 0.0

    std = np.std(x)
    if std < 1e-10:
        return 0.0

    r = r_frac * std

    def _count_matches(template_len: int) -> int:
        """Count template matches using Chebyshev distance."""
        count = 0
        templates = np.array([
            x[i:i + template_len] for i in range(n - template_len)
        ])
        n_templates = len(templates)
        for i in range(n_templates):
            # Exclude self-match by starting from i+1
            diffs = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += np.sum(diffs <= r)
        return count

    b = _count_matches(m)      # matches of length m
    a = _count_matches(m + 1)  # matches of length m+1

    if b == 0:
        return 0.0
    if a == 0:
        # No matches at m+1 — maximum entropy for this embedding
        return float(np.log(b))

    return float(-np.log(a / b))


def _permutation_entropy(
    x: np.ndarray,
    order: int = 3,
    delay: int = 1,
) -> float:
    """Compute normalized permutation entropy.

    PE quantifies the diversity of ordinal patterns in a time series.
    Normalized to [0, 1] by dividing by log(order!).

    Args:
        x: 1D signal.
        order: Order of permutation patterns (3-7 typical).
        delay: Time delay between points in each pattern.

    Returns:
        Normalized permutation entropy in [0, 1].

    Reference:
        Bandt & Pompe (2002), Phys Rev Lett.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)

    n_patterns = n - (order - 1) * delay
    if n_patterns < 1:
        return 0.0

    # Build ordinal patterns
    pattern_counts: Dict[tuple, int] = {}
    for i in range(n_patterns):
        indices = [i + j * delay for j in range(order)]
        pattern = tuple(np.argsort(x[indices]))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Shannon entropy of pattern distribution
    total = sum(pattern_counts.values())
    probs = np.array(list(pattern_counts.values()), dtype=np.float64) / total
    entropy = -np.sum(probs * np.log(probs))

    # Normalize by maximum possible entropy: log(order!)
    max_entropy = np.log(float(math.factorial(order)))
    if max_entropy < 1e-10:
        return 0.0

    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def _lempel_ziv_complexity(x: np.ndarray) -> float:
    """Compute normalized Lempel-Ziv complexity of a binarized signal.

    The signal is binarized by thresholding at its median. LZ complexity
    counts the number of distinct substrings encountered during a sequential
    scan. Normalized by n / log2(n) to yield [0, 1].

    Args:
        x: 1D signal.

    Returns:
        Normalized LZ complexity in [0, ~1].

    Reference:
        Lempel & Ziv (1976), IEEE Trans Inf Theory.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)

    if n < 2:
        return 0.0

    # Binarize at median
    median_val = np.median(x)
    binary = (x > median_val).astype(np.int8)

    # If all same value after binarization, minimal complexity
    if np.all(binary == binary[0]):
        return 0.0

    # LZ76 algorithm
    complexity = 1
    prefix_len = 1
    component_len = 1
    i = 0
    i_max = 1

    while i_max + component_len <= n:
        # Check if current component exists in the prefix
        if binary[i + component_len - 1] == binary[i_max + component_len - 1]:
            component_len += 1
        else:
            # Try next position in prefix
            i += 1
            if i == i_max:
                # New word found
                complexity += 1
                i_max += component_len
                i = 0
                component_len = 1
    complexity += 1

    # Normalize: theoretical upper bound for random binary is n / log2(n)
    if n <= 1:
        return 0.0
    normalizer = n / np.log2(n)
    if normalizer < 1e-10:
        return 0.0

    return float(np.clip(complexity / normalizer, 0.0, 1.0))


def _hurst_exponent(x: np.ndarray) -> float:
    """Estimate Hurst exponent using rescaled range (R/S) analysis.

    H < 0.5: anti-persistent (mean-reverting)
    H = 0.5: random walk (white noise)
    H > 0.5: persistent (trending)

    Args:
        x: 1D signal.

    Returns:
        Hurst exponent in [0, 1]. Returns 0.5 for degenerate cases.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)

    if n < 20:
        return 0.5

    # Use a range of sub-series lengths
    max_k = n // 2
    min_k = 8
    if min_k > max_k:
        return 0.5

    # Generate log-spaced window sizes
    ks = np.unique(np.logspace(
        np.log10(min_k), np.log10(max_k), num=15, dtype=int
    ))
    ks = ks[ks >= min_k]
    if len(ks) < 2:
        return 0.5

    rs_values = []
    for k in ks:
        n_segments = n // k
        if n_segments < 1:
            continue

        rs_seg = []
        for seg_i in range(n_segments):
            segment = x[seg_i * k:(seg_i + 1) * k]
            mean_seg = np.mean(segment)
            deviations = np.cumsum(segment - mean_seg)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(segment, ddof=0)
            if s > 1e-10:
                rs_seg.append(r / s)

        if rs_seg:
            rs_values.append((float(k), float(np.mean(rs_seg))))

    if len(rs_values) < 2:
        return 0.5

    # Linear regression in log-log space
    log_k = np.log(np.array([v[0] for v in rs_values]))
    log_rs = np.log(np.array([v[1] for v in rs_values]))

    # Filter out non-finite values
    valid = np.isfinite(log_k) & np.isfinite(log_rs)
    if np.sum(valid) < 2:
        return 0.5

    log_k = log_k[valid]
    log_rs = log_rs[valid]

    # Least-squares slope
    coeffs = np.polyfit(log_k, log_rs, 1)
    hurst = float(coeffs[0])

    return float(np.clip(hurst, 0.0, 1.0))


def _higuchi_fractal_dimension(
    x: np.ndarray,
    k_max: int = 10,
) -> float:
    """Compute Higuchi fractal dimension of a time series.

    HFD measures the fractal dimension of a waveform. For EEG:
      - Simple periodic signal: FD ~ 1.0
      - Complex EEG: FD ~ 1.3-1.8
      - Random noise: FD ~ 2.0

    Args:
        x: 1D signal.
        k_max: Maximum interval length for curve construction.

    Returns:
        Fractal dimension in [1.0, 2.0].

    Reference:
        Higuchi (1988), Physica D.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)

    if n < k_max + 1:
        k_max = max(2, n // 2)

    lengths = []
    ks = np.arange(1, k_max + 1)

    for k in ks:
        lk = []
        for m in range(1, k + 1):
            # Build sub-series starting at m with interval k
            indices = np.arange(m - 1, n, k)
            if len(indices) < 2:
                continue
            sub = x[indices]
            # Curve length for this (m, k) pair
            diff_sum = np.sum(np.abs(np.diff(sub)))
            # Normalize
            norm_factor = (n - 1) / (k * len(sub) * k) if len(sub) > 0 else 1.0
            if norm_factor < 1e-15:
                norm_factor = 1e-15
            length = diff_sum * ((n - 1) / (((len(indices) - 1) * k) * k))
            lk.append(length)

        if lk:
            lengths.append((float(k), float(np.mean(lk))))

    if len(lengths) < 2:
        return 1.0

    # Linear regression in log-log space
    log_k = np.log(np.array([1.0 / v[0] for v in lengths]))
    log_l = np.log(np.array([max(v[1], 1e-15) for v in lengths]))

    valid = np.isfinite(log_k) & np.isfinite(log_l)
    if np.sum(valid) < 2:
        return 1.0

    log_k = log_k[valid]
    log_l = log_l[valid]

    coeffs = np.polyfit(log_k, log_l, 1)
    fd = float(coeffs[0])

    return float(np.clip(fd, 1.0, 2.0))


def _dfa_exponent(
    x: np.ndarray,
    fs: float = 256.0,
) -> float:
    """Compute Detrended Fluctuation Analysis (DFA) scaling exponent.

    DFA characterizes long-range temporal correlations:
      - alpha ~ 0.5: uncorrelated (white noise)
      - alpha ~ 1.0: 1/f noise (pink noise)
      - alpha ~ 1.5: Brownian motion (integrated white noise)

    Args:
        x: 1D signal.
        fs: Sampling rate (unused but kept for API consistency).

    Returns:
        DFA exponent (positive float). Returns 0.5 for degenerate cases.

    Reference:
        Peng et al. (1994), Phys Rev E.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)

    if n < 16:
        return 0.5

    # Integrate the signal (cumulative sum of mean-subtracted series)
    y = np.cumsum(x - np.mean(x))

    # Generate window sizes (log-spaced)
    min_win = max(_DFA_MIN_WINDOW, 4)
    max_win = max(min_win + 1, int(n * _DFA_MAX_WINDOW_FRAC))

    if max_win <= min_win:
        return 0.5

    window_sizes = np.unique(np.logspace(
        np.log10(min_win), np.log10(max_win), num=15, dtype=int
    ))
    window_sizes = window_sizes[window_sizes >= min_win]

    if len(window_sizes) < 2:
        return 0.5

    fluctuations = []
    for win in window_sizes:
        n_segments = n // win
        if n_segments < 1:
            continue

        rms_values = []
        for seg_i in range(n_segments):
            segment = y[seg_i * win:(seg_i + 1) * win]
            # Detrend with linear fit
            t_seg = np.arange(len(segment), dtype=np.float64)
            coeffs = np.polyfit(t_seg, segment, 1)
            trend = np.polyval(coeffs, t_seg)
            residual = segment - trend
            rms = np.sqrt(np.mean(residual ** 2))
            rms_values.append(rms)

        if rms_values:
            fluctuations.append((float(win), float(np.mean(rms_values))))

    if len(fluctuations) < 2:
        return 0.5

    # Linear regression in log-log space
    log_n = np.log(np.array([f[0] for f in fluctuations]))
    log_f = np.log(np.array([max(f[1], 1e-15) for f in fluctuations]))

    valid = np.isfinite(log_n) & np.isfinite(log_f)
    if np.sum(valid) < 2:
        return 0.5

    log_n = log_n[valid]
    log_f = log_f[valid]

    coeffs = np.polyfit(log_n, log_f, 1)
    alpha = float(coeffs[0])

    return max(alpha, 0.01)  # Ensure positive
