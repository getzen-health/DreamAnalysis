"""Consciousness State Detector from EEG signals.

Detects altered states of consciousness from 4-channel Muse 2 EEG data
(ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10, 256 Hz) using signal complexity
and spectral features.

Scientific basis:
  - Lempel-Ziv complexity increases in psychedelic/altered states
    (Schartner et al. 2017, Timmermann et al. 2019)
  - Alpha power suppression is a hallmark of psychedelic states
    (Carhart-Harris et al. 2016)
  - Spectral entropy increases in altered states (more random/complex signal)
  - Frontal theta may increase during mystical/transcendent experiences
    (Beauregard & Paquette 2006)
  - Signal diversity (permutation entropy) correlates with consciousness level
    (Bandt & Pompe 2002)

States (by diversity_score):
  ordinary:       0-30
  relaxed:       30-45
  meditative:    45-60
  altered:       60-80
  deeply_altered: 80-100

Usage:
    detector = ConsciousnessDetector(fs=256.0)
    detector.set_baseline(resting_eeg)
    result = detector.assess(live_eeg)
    # result["consciousness_state"] -> "ordinary" / "relaxed" / etc.
    # result["diversity_score"] -> 0-100
    stats = detector.get_session_stats()
    history = detector.get_history(last_n=10)
    detector.reset()
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal


CONSCIOUSNESS_STATES = ("ordinary", "relaxed", "meditative", "altered", "deeply_altered")

# Diversity score thresholds for state classification
_STATE_THRESHOLDS = {
    "ordinary": (0, 30),
    "relaxed": (30, 45),
    "meditative": (45, 60),
    "altered": (60, 80),
    "deeply_altered": (80, 100),
}

# Maximum history entries per user
_MAX_HISTORY = 500


class ConsciousnessDetector:
    """Detect altered states of consciousness from EEG signals.

    Uses Lempel-Ziv complexity, permutation entropy, spectral entropy,
    and alpha suppression to compute a composite diversity score that maps
    to consciousness states.

    Supports multiple users via user_id parameter. Each user has independent
    baseline, history, and session stats.
    """

    def __init__(self, fs: float = 256.0) -> None:
        self._fs = fs
        # Per-user state: keyed by user_id
        self._baselines: Dict[str, Dict] = {}
        self._histories: Dict[str, List[Dict]] = defaultdict(list)

    # ── Public API ──────────────────────────────────────────────────────

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record a resting-state baseline for a user.

        Args:
            signals: EEG data (1D or 2D).
            fs: Sampling rate. Defaults to instance fs.
            user_id: User identifier for multi-user support.

        Returns:
            Dict with baseline_set (bool), baseline_complexity (float),
            baseline_entropy (float).
        """
        fs = fs or self._fs
        signals = self._normalize_input(signals)
        n_channels, n_samples = signals.shape

        # Compute baseline metrics across channels
        lz_values = []
        pe_values = []
        se_values = []
        alpha_powers = []

        for ch in range(n_channels):
            x = signals[ch]
            lz_values.append(_lempel_ziv_complexity(x))
            pe_values.append(_permutation_entropy(x))
            se_values.append(_spectral_entropy(x, fs))
            alpha_powers.append(_alpha_band_power(x, fs))

        baseline_complexity = float(np.mean(lz_values))
        baseline_entropy = float(np.mean(se_values))
        baseline_alpha = float(np.mean(alpha_powers))

        self._baselines[user_id] = {
            "complexity": baseline_complexity,
            "entropy": baseline_entropy,
            "alpha_power": baseline_alpha,
            "pe": float(np.mean(pe_values)),
        }

        return {
            "baseline_set": True,
            "baseline_complexity": round(baseline_complexity, 6),
            "baseline_entropy": round(baseline_entropy, 6),
        }

    def assess(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Assess consciousness state from EEG signals.

        Args:
            signals: EEG data. Accepts:
                - 1D array (n_samples,) for single channel
                - 2D array (n_channels, n_samples) for multichannel
            fs: Sampling rate in Hz. Defaults to instance fs.
            user_id: User identifier for multi-user support.

        Returns:
            Dict with:
                consciousness_state: str - one of CONSCIOUSNESS_STATES
                complexity_index: float - Lempel-Ziv complexity, 0-1
                spectral_entropy: float - normalized spectral entropy, 0-1
                alpha_suppression: float - alpha suppression index, 0-1
                diversity_score: float - composite score, 0-100
                state_stability: float - stability over recent epochs, 0-1
                has_baseline: bool
        """
        fs = fs or self._fs
        signals = self._normalize_input(signals)
        n_channels, n_samples = signals.shape

        has_baseline = user_id in self._baselines

        # Compute per-channel metrics, then average
        lz_values = []
        pe_values = []
        se_values = []
        alpha_powers = []

        for ch in range(n_channels):
            x = signals[ch]
            lz_values.append(_lempel_ziv_complexity(x))
            pe_values.append(_permutation_entropy(x))
            se_values.append(_spectral_entropy(x, fs))
            alpha_powers.append(_alpha_band_power(x, fs))

        complexity_index = float(np.clip(np.mean(lz_values), 0.0, 1.0))
        perm_entropy = float(np.clip(np.mean(pe_values), 0.0, 1.0))
        spec_entropy = float(np.clip(np.mean(se_values), 0.0, 1.0))
        alpha_power = float(np.mean(alpha_powers))

        # Alpha suppression: measures how much alpha is reduced
        # relative to baseline (if available) or in absolute terms
        alpha_suppression = self._compute_alpha_suppression(
            alpha_power, user_id
        )

        # Composite diversity score (0-100)
        diversity_score = self._compute_diversity_score(
            complexity_index, perm_entropy, spec_entropy, alpha_suppression
        )

        # Classify state
        consciousness_state = self._classify_state(diversity_score)

        # State stability
        history = self._histories[user_id]
        state_stability = self._compute_stability(history, diversity_score)

        result = {
            "consciousness_state": consciousness_state,
            "complexity_index": round(complexity_index, 6),
            "spectral_entropy": round(spec_entropy, 6),
            "alpha_suppression": round(alpha_suppression, 6),
            "diversity_score": round(diversity_score, 4),
            "state_stability": round(state_stability, 6),
            "has_baseline": has_baseline,
        }

        # Append to history (capped at _MAX_HISTORY)
        self._histories[user_id].append(result.copy())
        if len(self._histories[user_id]) > _MAX_HISTORY:
            self._histories[user_id] = self._histories[user_id][-_MAX_HISTORY:]

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get aggregate session statistics for a user.

        Args:
            user_id: User identifier.

        Returns:
            Dict with n_epochs, mean_diversity, dominant_state,
            state_distribution. If no epochs, returns n_epochs=0.
        """
        history = self._histories.get(user_id, [])
        n = len(history)

        if n == 0:
            return {"n_epochs": 0}

        diversity_scores = [entry["diversity_score"] for entry in history]
        states = [entry["consciousness_state"] for entry in history]

        # State distribution
        state_dist: Dict[str, int] = {}
        for s in states:
            state_dist[s] = state_dist.get(s, 0) + 1

        # Dominant state
        dominant_state = max(state_dist, key=state_dist.get)

        return {
            "n_epochs": n,
            "mean_diversity": round(float(np.mean(diversity_scores)), 4),
            "dominant_state": dominant_state,
            "state_distribution": state_dist,
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Return chronological history of assessments.

        Args:
            user_id: User identifier.
            last_n: If provided, return only the last N entries.

        Returns:
            List of result dicts from assess() calls, in order.
        """
        history = self._histories.get(user_id, [])
        if last_n is not None and last_n > 0:
            return list(history[-last_n:])
        return list(history)

    def reset(self, user_id: str = "default") -> None:
        """Clear all internal state for a user.

        Args:
            user_id: User identifier. Clears baseline, history.
        """
        if user_id in self._baselines:
            del self._baselines[user_id]
        if user_id in self._histories:
            del self._histories[user_id]

    # ── Private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _normalize_input(signals: np.ndarray) -> np.ndarray:
        """Normalize input to (n_channels, n_samples) shape."""
        signals = np.asarray(signals, dtype=np.float64)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)
        elif signals.ndim == 2 and signals.shape[0] > signals.shape[1]:
            signals = signals.T
        return signals

    def _compute_alpha_suppression(
        self, alpha_power: float, user_id: str
    ) -> float:
        """Compute alpha suppression index (0-1).

        If baseline exists: relative suppression = 1 - (current / baseline).
        Otherwise: absolute suppression based on typical alpha range.
        """
        if user_id in self._baselines:
            baseline_alpha = self._baselines[user_id]["alpha_power"]
            if baseline_alpha > 1e-10:
                ratio = alpha_power / baseline_alpha
                # suppression = how much alpha has decreased
                suppression = float(np.clip(1.0 - ratio, 0.0, 1.0))
                return suppression
        # Absolute: high alpha = low suppression, low alpha = high suppression
        # Typical resting alpha relative power ~0.2-0.4 of total
        # Map so that alpha_power of 0.3+ gives ~0 suppression, near 0 gives ~1
        suppression = float(np.clip(1.0 - alpha_power / 0.35, 0.0, 1.0))
        return suppression

    @staticmethod
    def _compute_diversity_score(
        complexity: float,
        perm_entropy: float,
        spec_entropy: float,
        alpha_suppression: float,
    ) -> float:
        """Compute composite diversity score (0-100).

        Weights:
          - Lempel-Ziv complexity: 30% (primary altered state marker)
          - Permutation entropy: 25% (ordinal pattern diversity)
          - Spectral entropy: 25% (spectral randomness)
          - Alpha suppression: 20% (hallmark of psychedelic states)
        """
        composite = (
            0.30 * complexity
            + 0.25 * perm_entropy
            + 0.25 * spec_entropy
            + 0.20 * alpha_suppression
        )
        return float(np.clip(composite * 100, 0.0, 100.0))

    @staticmethod
    def _classify_state(diversity_score: float) -> str:
        """Classify consciousness state from diversity score."""
        if diversity_score >= 80:
            return "deeply_altered"
        elif diversity_score >= 60:
            return "altered"
        elif diversity_score >= 45:
            return "meditative"
        elif diversity_score >= 30:
            return "relaxed"
        return "ordinary"

    @staticmethod
    def _compute_stability(
        history: List[Dict], current_diversity: float
    ) -> float:
        """Compute state stability from recent history.

        Uses the inverse of the standard deviation of recent diversity scores.
        More consistent scores = higher stability.
        """
        if len(history) < 1:
            return 0.0

        # Use last 10 entries + current
        recent = [entry["diversity_score"] for entry in history[-10:]]
        recent.append(current_diversity)

        if len(recent) < 2:
            return 0.0

        std = float(np.std(recent))
        # Map std to stability: low std = high stability
        # std of 0 = perfect stability (1.0)
        # std of 20+ = very unstable (near 0)
        stability = float(np.clip(1.0 - std / 20.0, 0.0, 1.0))
        return stability


# ── Standalone signal analysis functions ──────────────────────────────────


def _lempel_ziv_complexity(x: np.ndarray) -> float:
    """Compute normalized Lempel-Ziv complexity of a binarized signal.

    The signal is binarized at the median. LZ complexity counts distinct
    substrings during a sequential scan. Normalized by n / log2(n).

    Args:
        x: 1D signal.

    Returns:
        Normalized LZ complexity in [0, ~1].

    Reference:
        Lempel & Ziv (1976), IEEE Trans Inf Theory.
        Schartner et al. (2017), Sci Rep.
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
        if binary[i + component_len - 1] == binary[i_max + component_len - 1]:
            component_len += 1
        else:
            i += 1
            if i == i_max:
                complexity += 1
                i_max += component_len
                i = 0
                component_len = 1
    complexity += 1

    # Normalize by theoretical upper bound for random binary: n / log2(n)
    if n <= 1:
        return 0.0
    normalizer = n / np.log2(n)
    if normalizer < 1e-10:
        return 0.0

    return float(np.clip(complexity / normalizer, 0.0, 1.0))


def _permutation_entropy(
    x: np.ndarray, order: int = 3, delay: int = 1
) -> float:
    """Compute normalized permutation entropy.

    PE quantifies the diversity of ordinal patterns. Normalized to [0, 1]
    by dividing by log(order!).

    Args:
        x: 1D signal.
        order: Order of permutation patterns.
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
    entropy = -float(np.sum(probs * np.log(probs)))

    # Normalize by maximum possible entropy: log(order!)
    max_entropy = np.log(float(math.factorial(order)))
    if max_entropy < 1e-10:
        return 0.0

    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def _spectral_entropy(x: np.ndarray, fs: float = 256.0) -> float:
    """Compute normalized spectral entropy.

    Uses Welch's method to estimate PSD, then computes the Shannon
    entropy of the normalized power spectrum. Normalized to [0, 1].

    Args:
        x: 1D signal.
        fs: Sampling rate in Hz.

    Returns:
        Normalized spectral entropy in [0, 1].
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)

    if n < 4:
        return 0.0

    # Check for flat/constant signal
    if np.std(x) < 1e-10:
        return 0.0

    # Welch PSD
    nperseg = min(256, n)
    freqs, psd = scipy_signal.welch(x, fs=fs, nperseg=nperseg)

    # Restrict to 1-45 Hz (physiological EEG range)
    mask = (freqs >= 1.0) & (freqs <= 45.0)
    psd = psd[mask]

    if len(psd) == 0 or np.sum(psd) < 1e-15:
        return 0.0

    # Normalize to probability distribution
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]

    if len(psd_norm) < 2:
        return 0.0

    # Shannon entropy
    entropy = -float(np.sum(psd_norm * np.log(psd_norm)))

    # Normalize by max entropy (uniform distribution)
    max_entropy = np.log(len(psd_norm))
    if max_entropy < 1e-10:
        return 0.0

    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def _alpha_band_power(x: np.ndarray, fs: float = 256.0) -> float:
    """Compute relative alpha band power (8-12 Hz).

    Uses Welch PSD and returns the fraction of total power in 1-45 Hz
    that falls in the alpha band.

    Args:
        x: 1D signal.
        fs: Sampling rate in Hz.

    Returns:
        Relative alpha power (0-1).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)

    if n < 4:
        return 0.0

    if np.std(x) < 1e-10:
        return 0.0

    nperseg = min(256, n)
    freqs, psd = scipy_signal.welch(x, fs=fs, nperseg=nperseg)

    # Total power 1-45 Hz
    total_mask = (freqs >= 1.0) & (freqs <= 45.0)
    total_power = np.sum(psd[total_mask])

    if total_power < 1e-15:
        return 0.0

    # Alpha band 8-12 Hz
    alpha_mask = (freqs >= 8.0) & (freqs <= 12.0)
    alpha_power = np.sum(psd[alpha_mask])

    return float(np.clip(alpha_power / total_power, 0.0, 1.0))
