"""Emotional granularity estimator from EEG pattern diversity.

Measures how finely a user differentiates between emotional states by
analyzing the diversity of EEG feature patterns across multiple emotional
episodes. Higher granularity means the brain produces more distinct neural
signatures for different emotions -- a marker of emotional intelligence and
affect differentiation (Tugade, Fredrickson & Barrett, 2004).

Scientific basis:
- Barrett (2006): emotional granularity = ability to make fine-grained
  distinctions among feelings. Measured via intra-individual covariance of
  self-reported affect. Extended here to EEG feature space.
- Cosine distance between episode feature vectors captures how different
  the neural patterns are across emotional experiences.
- Spectral entropy variability reflects diversity in overall brain dynamics.
- Differential entropy variability captures band-specific pattern changes.

Usage:
    estimator = EmotionalGranularityEstimator(fs=256.0)
    for eeg_epoch in episodes:
        estimator.add_episode(eeg_epoch)
    result = estimator.compute_granularity()
    # result["granularity"] in [0, 1], higher = more granular
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)

# --- Band definitions (Hz) matching eeg_processor.py ---
_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Maximum episodes stored per user to prevent unbounded memory growth.
_MAX_EPISODES = 500

# Maximum granularity results stored per user.
_MAX_HISTORY = 500

# Minimum episodes needed before granularity can be computed.
_MIN_EPISODES = 10


def _bandpass_filter(
    signal: np.ndarray, fs: float, low: float = 1.0, high: float = 45.0, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    low_n = max(low / nyq, 1e-5)
    high_n = min(high / nyq, 0.9999)
    b, a = butter(order, [low_n, high_n], btype="band")
    # Need at least 3*max(len(a), len(b)) samples for filtfilt
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        return signal.copy()
    return filtfilt(b, a, signal)


def _extract_band_powers(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """Extract log band powers via Welch PSD."""
    nperseg = min(len(signal), int(fs * 2))
    if nperseg < 4:
        return {band: 0.0 for band in _BANDS}

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    powers = {}
    for band, (lo, hi) in _BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.any():
            power = float(np.mean(psd[mask]))
            powers[band] = float(np.log1p(max(power, 0.0)))
        else:
            powers[band] = 0.0
    return powers


def _spectral_entropy(signal: np.ndarray, fs: float) -> float:
    """Normalized spectral entropy of a signal."""
    nperseg = min(len(signal), int(fs * 2))
    if nperseg < 4:
        return 0.0
    _, psd = welch(signal, fs=fs, nperseg=nperseg)
    psd = psd + 1e-12  # avoid log(0)
    psd_norm = psd / psd.sum()
    entropy = -float(np.sum(psd_norm * np.log2(psd_norm)))
    max_entropy = np.log2(len(psd_norm))
    if max_entropy < 1e-12:
        return 0.0
    return float(entropy / max_entropy)


def _differential_entropy_per_band(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """Differential entropy per frequency band: 0.5 * log(2*pi*e*var)."""
    de = {}
    nyq = fs / 2.0
    for band, (lo, hi) in _BANDS.items():
        low_n = max(lo / nyq, 1e-5)
        high_n = min(hi / nyq, 0.9999)
        try:
            b, a = butter(4, [low_n, high_n], btype="band")
            padlen = 3 * max(len(a), len(b))
            if len(signal) <= padlen:
                de[band] = 0.0
                continue
            filtered = filtfilt(b, a, signal)
            var = float(np.var(filtered))
            if var > 1e-20:
                de[band] = 0.5 * float(np.log(2 * np.pi * np.e * var))
            else:
                de[band] = 0.0
        except Exception:
            de[band] = 0.0
    return de


def _extract_episode_features(eeg: np.ndarray, fs: float) -> np.ndarray:
    """Extract a feature vector from one EEG episode.

    Features per channel: 5 band powers + spectral entropy + 5 DE values
        = 11 features per channel.
    Plus inter-channel features if multichannel:
        alpha/beta ratio (ch-avg), theta/beta ratio (ch-avg),
        frontal asymmetry (if >= 3 channels for AF7/AF8).
    """
    if eeg.ndim == 1:
        eeg = eeg.reshape(1, -1)

    n_channels = eeg.shape[0]
    all_features: List[float] = []

    for ch in range(n_channels):
        sig = eeg[ch].copy()
        # Filter
        sig = _bandpass_filter(sig, fs, low=1.0, high=45.0)
        # Band powers
        bp = _extract_band_powers(sig, fs)
        all_features.extend(bp[b] for b in _BANDS)
        # Spectral entropy
        se = _spectral_entropy(sig, fs)
        all_features.append(se)
        # Differential entropy
        de = _differential_entropy_per_band(sig, fs)
        all_features.extend(de[b] for b in _BANDS)

    # Cross-channel ratios (averaged across channels)
    avg_bp: Dict[str, float] = {b: 0.0 for b in _BANDS}
    for ch in range(n_channels):
        sig = _bandpass_filter(eeg[ch].copy(), fs)
        bp = _extract_band_powers(sig, fs)
        for b in _BANDS:
            avg_bp[b] += bp[b]
    for b in _BANDS:
        avg_bp[b] /= max(n_channels, 1)

    alpha = max(avg_bp["alpha"], 1e-10)
    beta = max(avg_bp["beta"], 1e-10)
    theta = max(avg_bp["theta"], 1e-10)
    all_features.append(alpha / beta)   # alpha/beta ratio
    all_features.append(theta / beta)   # theta/beta ratio

    # Frontal asymmetry if we have AF7 (ch1) and AF8 (ch2)
    if n_channels >= 3:
        left = _bandpass_filter(eeg[1].copy(), fs)
        right = _bandpass_filter(eeg[2].copy(), fs)
        left_bp = _extract_band_powers(left, fs)
        right_bp = _extract_band_powers(right, fs)
        left_alpha = max(left_bp["alpha"], 1e-10)
        right_alpha = max(right_bp["alpha"], 1e-10)
        faa = float(np.log(right_alpha) - np.log(left_alpha))
        all_features.append(faa)
    else:
        all_features.append(0.0)

    feature_vec = np.array(all_features, dtype=np.float64)
    # Sanitize: replace NaN/inf with 0
    feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_vec


class EmotionalGranularityEstimator:
    """Measures how finely a user differentiates emotional states via EEG.

    Collects EEG episodes over time. After enough episodes (default 10),
    computes a granularity score based on the diversity of neural patterns.

    Higher granularity = more distinct neural signatures across episodes
    = finer emotional differentiation ability.

    Args:
        fs: Default sampling rate in Hz.
    """

    def __init__(self, fs: float = 256.0) -> None:
        self._fs = fs
        # Per-user storage: user_id -> list of feature vectors
        self._episodes: Dict[str, List[np.ndarray]] = {}
        # Per-user storage: user_id -> list of episode labels (optional)
        self._labels: Dict[str, List[Optional[str]]] = {}
        # Per-user storage: user_id -> list of spectral entropies (for variability)
        self._entropies: Dict[str, List[float]] = {}
        # Per-user storage: user_id -> list of DE dicts
        self._des: Dict[str, List[Dict[str, float]]] = {}
        # Per-user granularity history
        self._history: Dict[str, List[Dict]] = {}

    def _ensure_user(self, user_id: str) -> None:
        """Initialize storage for a user if not already present."""
        if user_id not in self._episodes:
            self._episodes[user_id] = []
            self._labels[user_id] = []
            self._entropies[user_id] = []
            self._des[user_id] = []
            self._history[user_id] = []

    def add_episode(
        self,
        eeg: np.ndarray,
        fs: Optional[float] = None,
        label: Optional[str] = None,
        user_id: str = "default",
    ) -> None:
        """Add one emotional episode's EEG data.

        Extracts features and stores them for later granularity computation.

        Args:
            eeg: EEG array, shape (n_channels, n_samples) or (n_samples,).
            fs: Sampling rate. If None, uses the default set in __init__.
            label: Optional emotion label for this episode.
            user_id: User identifier for multi-user support.
        """
        self._ensure_user(user_id)
        sample_rate = fs if fs is not None else self._fs

        if eeg.ndim == 1:
            eeg = eeg.reshape(1, -1)

        # Extract feature vector
        features = _extract_episode_features(eeg, sample_rate)
        self._episodes[user_id].append(features)
        self._labels[user_id].append(label)

        # Store per-episode spectral entropy (average across channels)
        entropies_ch = []
        for ch in range(eeg.shape[0]):
            sig = _bandpass_filter(eeg[ch].copy(), sample_rate)
            entropies_ch.append(_spectral_entropy(sig, sample_rate))
        self._entropies[user_id].append(float(np.mean(entropies_ch)))

        # Store per-episode differential entropy (average across channels)
        de_avg: Dict[str, float] = {b: 0.0 for b in _BANDS}
        for ch in range(eeg.shape[0]):
            sig = _bandpass_filter(eeg[ch].copy(), sample_rate)
            de_ch = _differential_entropy_per_band(sig, sample_rate)
            for b in _BANDS:
                de_avg[b] += de_ch[b]
        for b in _BANDS:
            de_avg[b] /= max(eeg.shape[0], 1)
        self._des[user_id].append(de_avg)

        # Enforce cap
        if len(self._episodes[user_id]) > _MAX_EPISODES:
            self._episodes[user_id] = self._episodes[user_id][-_MAX_EPISODES:]
            self._labels[user_id] = self._labels[user_id][-_MAX_EPISODES:]
            self._entropies[user_id] = self._entropies[user_id][-_MAX_EPISODES:]
            self._des[user_id] = self._des[user_id][-_MAX_EPISODES:]

    def compute_granularity(self, user_id: str = "default") -> Dict:
        """Compute emotional granularity from accumulated episodes.

        Requires at least 10 episodes. Returns a dict with the granularity
        score and supporting metrics.

        Args:
            user_id: User identifier.

        Returns:
            Dict with keys: granularity, pattern_diversity, entropy_variability,
            de_variability, episodes_collected, ready, granularity_level.
            granularity is None if fewer than 10 episodes collected.
        """
        self._ensure_user(user_id)
        n_episodes = len(self._episodes[user_id])

        if n_episodes < _MIN_EPISODES:
            return {
                "granularity": None,
                "pattern_diversity": None,
                "entropy_variability": None,
                "de_variability": None,
                "episodes_collected": n_episodes,
                "ready": False,
                "granularity_level": "insufficient_data",
            }

        # --- Pattern diversity: mean pairwise cosine distance ---
        feature_matrix = np.array(self._episodes[user_id])
        # Normalize rows to unit length for cosine distance
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        feature_matrix_normed = feature_matrix / norms
        cosine_distances = pdist(feature_matrix_normed, metric="cosine")
        # Replace NaN from zero-vectors with 0
        cosine_distances = np.nan_to_num(cosine_distances, nan=0.0)
        pattern_diversity = float(np.mean(cosine_distances))

        # --- Entropy variability: std of spectral entropy across episodes ---
        entropy_arr = np.array(self._entropies[user_id])
        entropy_variability = float(np.std(entropy_arr))

        # --- DE variability: mean std of differential entropy across bands ---
        de_matrix = np.array([
            [d[b] for b in _BANDS] for d in self._des[user_id]
        ])
        de_stds = np.std(de_matrix, axis=0)  # std per band
        de_variability = float(np.mean(de_stds))

        # --- Composite granularity score ---
        # Normalize each component to [0, 1] range using sigmoid-like mapping
        # Pattern diversity: cosine distance in [0, 2], typical range [0, 0.5]
        pd_norm = float(1.0 / (1.0 + np.exp(-10.0 * (pattern_diversity - 0.15))))
        # Entropy variability: typical range [0, 0.1]
        ev_norm = float(1.0 / (1.0 + np.exp(-40.0 * (entropy_variability - 0.03))))
        # DE variability: typical range [0, 3]
        dev_norm = float(1.0 / (1.0 + np.exp(-2.0 * (de_variability - 0.5))))

        granularity = float(np.clip(
            0.50 * pd_norm + 0.30 * ev_norm + 0.20 * dev_norm,
            0.0, 1.0,
        ))

        # Classify level
        if granularity >= 0.65:
            level = "high"
        elif granularity >= 0.35:
            level = "moderate"
        else:
            level = "low"

        result = {
            "granularity": round(granularity, 4),
            "pattern_diversity": round(pattern_diversity, 4),
            "entropy_variability": round(entropy_variability, 4),
            "de_variability": round(de_variability, 4),
            "episodes_collected": n_episodes,
            "ready": True,
            "granularity_level": level,
        }

        # Store in history
        self._history[user_id].append(result)
        if len(self._history[user_id]) > _MAX_HISTORY:
            self._history[user_id] = self._history[user_id][-_MAX_HISTORY:]

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get session statistics for a user.

        Args:
            user_id: User identifier.

        Returns:
            Dict with n_episodes, has_episodes, mean_granularity (if computed).
        """
        self._ensure_user(user_id)
        n_episodes = len(self._episodes[user_id])
        history = self._history[user_id]

        stats: Dict = {
            "n_episodes": n_episodes,
            "has_episodes": n_episodes > 0,
        }

        if history:
            granularity_values = [
                h["granularity"] for h in history if h["granularity"] is not None
            ]
            if granularity_values:
                stats["mean_granularity"] = round(
                    float(np.mean(granularity_values)), 4
                )
            else:
                stats["mean_granularity"] = None
        else:
            stats["mean_granularity"] = None

        return stats

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Get granularity computation history for a user.

        Args:
            user_id: User identifier.
            last_n: If provided, return only the last N entries.

        Returns:
            List of granularity result dicts.
        """
        self._ensure_user(user_id)
        history = self._history[user_id]
        if last_n is not None and last_n > 0:
            return list(history[-last_n:])
        return list(history)

    def reset(self, user_id: str = "default") -> None:
        """Clear all data for a user.

        Args:
            user_id: User identifier.
        """
        self._episodes.pop(user_id, None)
        self._labels.pop(user_id, None)
        self._entropies.pop(user_id, None)
        self._des.pop(user_id, None)
        self._history.pop(user_id, None)
