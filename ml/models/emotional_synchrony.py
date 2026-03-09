"""Emotional synchrony detector using single-brain EEG phase-locking and coherence.

Detects emotional engagement with content (music, video, conversation) by
measuring inter-channel phase synchronization and coherence across the
Muse 2's 4-channel layout.

When a person is emotionally engaged with external content, fronto-temporal
phase-locking increases (affective network synchronization) and inter-hemispheric
frontal coherence rises (bilateral prefrontal integration).

Channel layout (Muse 2, BrainFlow board_id 38):
    ch0 = TP9   (left temporal)
    ch1 = AF7   (left frontal)
    ch2 = AF8   (right frontal)
    ch3 = TP10  (right temporal)

Synchrony metrics computed:
    - Fronto-temporal PLV (alpha, beta): AF7<->TP9, AF8<->TP10
    - Frontal interhemispheric PLV (alpha): AF7<->AF8
    - Mean alpha/beta coherence across all channel pairs
    - Composite synchrony score (0-1)

References:
    Lindenberger et al. (2009) -- Brains swinging in concert: cortical
        phase synchronization while playing in pairs
    Dikker et al. (2017) -- Brain-to-brain synchrony tracks real-world
        dynamic group interactions in the classroom
    Perez et al. (2017) -- Brain-to-brain entrainment: EEG interbrain
        synchronization while speaking and listening
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal


# -- Band definitions (Hz) ---------------------------------------------------
_BANDS = {
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
}

# -- Engagement thresholds ----------------------------------------------------
_DEEPLY_ENGAGED_THRESHOLD = 0.65
_MODERATELY_ENGAGED_THRESHOLD = 0.45
_MILDLY_ENGAGED_THRESHOLD = 0.25

# -- History cap per user -----------------------------------------------------
_MAX_HISTORY = 500

# -- Minimum samples for meaningful computation --------------------------------
_MIN_SAMPLES = 64


def _bandpass_filter(
    data: np.ndarray,
    low: float,
    high: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Args:
        data: 1D signal array.
        low: Lower cutoff frequency in Hz.
        high: Upper cutoff frequency in Hz.
        fs: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Filtered signal of same shape.
    """
    nyq = fs / 2.0
    low_n = low / nyq
    high_n = high / nyq
    # Clamp to valid range
    low_n = max(low_n, 1e-5)
    high_n = min(high_n, 1.0 - 1e-5)
    if low_n >= high_n:
        return data
    b, a = scipy_signal.butter(order, [low_n, high_n], btype="band")
    return scipy_signal.filtfilt(b, a, data)


def _compute_plv(
    signal1: np.ndarray,
    signal2: np.ndarray,
    fs: float,
    band: Tuple[float, float],
) -> float:
    """Compute phase-locking value between two signals in a frequency band.

    Steps:
        1. Bandpass filter both signals to the target band.
        2. Hilbert transform to extract instantaneous phase.
        3. PLV = |mean(exp(j * (phase1 - phase2)))|

    Args:
        signal1: 1D array, first channel.
        signal2: 1D array, second channel.
        fs: Sampling rate in Hz.
        band: (low_freq, high_freq) tuple.

    Returns:
        PLV value in [0, 1]. 1 = perfect phase locking, 0 = random.
    """
    if len(signal1) < _MIN_SAMPLES or len(signal2) < _MIN_SAMPLES:
        return 0.0

    low, high = band
    filtered1 = _bandpass_filter(signal1, low, high, fs)
    filtered2 = _bandpass_filter(signal2, low, high, fs)

    analytic1 = scipy_signal.hilbert(filtered1)
    analytic2 = scipy_signal.hilbert(filtered2)

    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)

    phase_diff = phase1 - phase2
    plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
    if np.isnan(plv):
        return 0.0
    return np.clip(plv, 0.0, 1.0)


def _compute_mean_coherence(
    signal1: np.ndarray,
    signal2: np.ndarray,
    fs: float,
    band: Tuple[float, float],
) -> float:
    """Compute mean coherence between two signals in a frequency band.

    Uses scipy.signal.coherence with Welch method, then averages
    coherence values within the specified band.

    Args:
        signal1: 1D array, first channel.
        signal2: 1D array, second channel.
        fs: Sampling rate in Hz.
        band: (low_freq, high_freq) tuple.

    Returns:
        Mean coherence in [0, 1].
    """
    if len(signal1) < _MIN_SAMPLES or len(signal2) < _MIN_SAMPLES:
        return 0.0

    nperseg = min(len(signal1), int(fs * 2))
    if nperseg < 4:
        return 0.0

    try:
        freqs, coh = scipy_signal.coherence(
            signal1, signal2, fs=fs, nperseg=nperseg
        )
    except Exception:
        return 0.0

    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0

    coh_band = coh[mask]
    # Handle NaN from zero-variance signals (scipy divides by zero PSD)
    coh_band = np.nan_to_num(coh_band, nan=0.0)
    mean_coh = float(np.mean(coh_band))
    return np.clip(mean_coh, 0.0, 1.0)


def _spectral_entropy(signal: np.ndarray, fs: float) -> float:
    """Compute normalized spectral entropy as a single-channel synchrony proxy.

    Lower entropy = more concentrated power = more organized oscillation.
    Returns value in [0, 1] where 0 = single frequency, 1 = white noise.
    """
    if len(signal) < _MIN_SAMPLES:
        return 0.5  # neutral default

    nperseg = min(len(signal), int(fs * 2))
    if nperseg < 4:
        return 0.5

    try:
        freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)
    except Exception:
        return 0.5

    # Normalize PSD to probability distribution
    psd_sum = np.sum(psd)
    if psd_sum < 1e-12:
        return 0.5

    p = psd / psd_sum
    # Remove zeros to avoid log(0)
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))
    # Normalize by max possible entropy (uniform distribution)
    max_entropy = np.log2(len(p)) if len(p) > 0 else 1.0
    if max_entropy < 1e-12:
        return 0.5

    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


class EmotionalSynchronyDetector:
    """Detect emotional synchrony with content from single-brain EEG.

    Uses phase-locking value (PLV) and coherence between Muse 2 channels
    to measure how synchronized brain regions are during emotional engagement.

    Higher synchrony = deeper emotional engagement with the stimulus.

    Args:
        fs: Default sampling rate in Hz.

    Channel pairs analyzed:
        - Left fronto-temporal: AF7 (ch1) <-> TP9 (ch0)
        - Right fronto-temporal: AF8 (ch2) <-> TP10 (ch3)
        - Frontal interhemispheric: AF7 (ch1) <-> AF8 (ch2)
    """

    def __init__(self, fs: float = 256.0):
        self._fs = fs
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record baseline synchrony from resting-state EEG.

        Call during 2-3 min eyes-closed resting state before the task.
        Baseline synchrony is subtracted from task-state to detect
        content-driven engagement above resting levels.

        Args:
            signals: EEG array. (n_channels, n_samples) or (n_samples,) for 1D.
            fs: Sampling rate override (uses default if None).
            user_id: User identifier.

        Returns:
            Dict with baseline_set (bool), n_channels (int).
        """
        fs = fs if fs is not None else self._fs
        signals = np.asarray(signals, dtype=np.float64)

        if signals.ndim == 1:
            signals = signals[np.newaxis, :]

        n_channels = signals.shape[0]

        # Compute baseline synchrony metrics
        baseline_metrics = self._compute_synchrony_metrics(signals, fs)

        self._baselines[user_id] = {
            "metrics": baseline_metrics,
            "n_channels": n_channels,
            "fs": fs,
        }

        return {
            "baseline_set": True,
            "n_channels": n_channels,
        }

    def analyze(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Analyze emotional synchrony from a 4-channel EEG epoch.

        Computes PLV, coherence, and composite synchrony score.
        Optionally adjusts for resting baseline if set_baseline() was called.

        Args:
            signals: EEG array. (n_channels, n_samples) or (n_samples,) for 1D.
            fs: Sampling rate override (uses default if None).
            user_id: User identifier.

        Returns:
            Dict with:
            - synchrony_score: 0-1 composite score
            - fronto_temporal_plv_alpha: PLV between frontal-temporal in alpha
            - fronto_temporal_plv_beta: PLV in beta band
            - frontal_interhemispheric_plv: PLV between AF7-AF8 in alpha
            - alpha_coherence: mean alpha coherence across pairs
            - beta_coherence: mean beta coherence across pairs
            - engagement_level: "deeply_engaged" / "moderately_engaged" /
              "mildly_engaged" / "disengaged"
            - has_baseline: whether baseline was set for this user
        """
        fs = fs if fs is not None else self._fs
        signals = np.asarray(signals, dtype=np.float64)

        if signals.ndim == 1:
            signals = signals[np.newaxis, :]

        n_channels = signals.shape[0]
        n_samples = signals.shape[1]

        has_baseline = user_id in self._baselines

        # Very short signals: return defaults
        if n_samples < _MIN_SAMPLES:
            result = self._default_result(has_baseline)
            self._append_history(user_id, result)
            return result

        # Compute raw synchrony metrics
        metrics = self._compute_synchrony_metrics(signals, fs)

        # Apply baseline correction if available
        if has_baseline:
            baseline = self._baselines[user_id]["metrics"]
            # Subtract baseline, clip to [0, 1]
            corrected_score = float(np.clip(
                metrics["synchrony_score"] - baseline["synchrony_score"] * 0.5 + 0.5,
                0.0, 1.0,
            ))
        else:
            corrected_score = metrics["synchrony_score"]

        # Classify engagement level
        engagement_level = self._classify_engagement(corrected_score)

        result = {
            "synchrony_score": round(corrected_score, 4),
            "fronto_temporal_plv_alpha": round(
                metrics["fronto_temporal_plv_alpha"], 4
            ),
            "fronto_temporal_plv_beta": round(
                metrics["fronto_temporal_plv_beta"], 4
            ),
            "frontal_interhemispheric_plv": round(
                metrics["frontal_interhemispheric_plv"], 4
            ),
            "alpha_coherence": round(metrics["alpha_coherence"], 4),
            "beta_coherence": round(metrics["beta_coherence"], 4),
            "engagement_level": engagement_level,
            "has_baseline": has_baseline,
        }

        self._append_history(user_id, result)
        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get session-level statistics.

        Returns:
            Dict with n_epochs, mean_synchrony, dominant_engagement_level.
            If no data, returns n_epochs=0.
        """
        history = self._history.get(user_id, [])
        if not history:
            return {"n_epochs": 0}

        scores = [h["synchrony_score"] for h in history]
        levels = [h["engagement_level"] for h in history]

        # Find dominant engagement level (most frequent)
        level_counts: Dict[str, int] = {}
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        dominant = max(level_counts, key=level_counts.get)

        return {
            "n_epochs": len(history),
            "mean_synchrony": round(float(np.mean(scores)), 4),
            "dominant_engagement_level": dominant,
        }

    def get_history(
        self,
        user_id: str = "default",
        last_n: Optional[int] = None,
    ) -> List[Dict]:
        """Get analysis history for a user.

        Args:
            user_id: User identifier.
            last_n: If provided, return only the last N entries.

        Returns:
            List of analysis result dicts.
        """
        history = self._history.get(user_id, [])
        if last_n is not None:
            history = history[-last_n:]
        return list(history)

    def reset(self, user_id: str = "default") -> None:
        """Clear all data (baseline + history) for a user."""
        self._baselines.pop(user_id, None)
        self._history.pop(user_id, None)

    # -- Private helpers -------------------------------------------------------

    def _compute_synchrony_metrics(
        self, signals: np.ndarray, fs: float
    ) -> Dict:
        """Compute all synchrony metrics from multichannel EEG.

        For < 4 channels, computes what's possible. For 1 channel, uses
        spectral entropy as a proxy.

        Returns dict with all metric keys.
        """
        n_channels = signals.shape[0]

        # Initialize all metrics to 0
        ft_plv_alpha = 0.0
        ft_plv_beta = 0.0
        frontal_ih_plv = 0.0
        alpha_coherence = 0.0
        beta_coherence = 0.0

        if n_channels >= 4:
            # Full 4-channel Muse 2 layout
            tp9 = signals[0]
            af7 = signals[1]
            af8 = signals[2]
            tp10 = signals[3]

            # Fronto-temporal PLV (alpha band)
            left_ft_plv_alpha = _compute_plv(af7, tp9, fs, _BANDS["alpha"])
            right_ft_plv_alpha = _compute_plv(af8, tp10, fs, _BANDS["alpha"])
            ft_plv_alpha = (left_ft_plv_alpha + right_ft_plv_alpha) / 2.0

            # Fronto-temporal PLV (beta band)
            left_ft_plv_beta = _compute_plv(af7, tp9, fs, _BANDS["beta"])
            right_ft_plv_beta = _compute_plv(af8, tp10, fs, _BANDS["beta"])
            ft_plv_beta = (left_ft_plv_beta + right_ft_plv_beta) / 2.0

            # Frontal interhemispheric PLV (alpha)
            frontal_ih_plv = _compute_plv(af7, af8, fs, _BANDS["alpha"])

            # Alpha coherence: mean across all 6 channel pairs
            alpha_coh_pairs = []
            beta_coh_pairs = []
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    alpha_coh_pairs.append(
                        _compute_mean_coherence(
                            signals[i], signals[j], fs, _BANDS["alpha"]
                        )
                    )
                    beta_coh_pairs.append(
                        _compute_mean_coherence(
                            signals[i], signals[j], fs, _BANDS["beta"]
                        )
                    )
            alpha_coherence = float(np.mean(alpha_coh_pairs))
            beta_coherence = float(np.mean(beta_coh_pairs))

        elif n_channels >= 2:
            # Partial: compute PLV and coherence for available pairs
            plv_alpha_pairs = []
            plv_beta_pairs = []
            coh_alpha_pairs = []
            coh_beta_pairs = []
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    plv_alpha_pairs.append(
                        _compute_plv(signals[i], signals[j], fs, _BANDS["alpha"])
                    )
                    plv_beta_pairs.append(
                        _compute_plv(signals[i], signals[j], fs, _BANDS["beta"])
                    )
                    coh_alpha_pairs.append(
                        _compute_mean_coherence(
                            signals[i], signals[j], fs, _BANDS["alpha"]
                        )
                    )
                    coh_beta_pairs.append(
                        _compute_mean_coherence(
                            signals[i], signals[j], fs, _BANDS["beta"]
                        )
                    )
            ft_plv_alpha = float(np.mean(plv_alpha_pairs))
            ft_plv_beta = float(np.mean(plv_beta_pairs))
            frontal_ih_plv = plv_alpha_pairs[0]  # first pair
            alpha_coherence = float(np.mean(coh_alpha_pairs))
            beta_coherence = float(np.mean(coh_beta_pairs))

        else:
            # Single channel: use spectral entropy as proxy
            # Low entropy = concentrated power = internal synchrony
            se = _spectral_entropy(signals[0], fs)
            # Invert: low entropy -> high "synchrony"
            single_ch_sync = 1.0 - se
            ft_plv_alpha = single_ch_sync
            ft_plv_beta = single_ch_sync
            alpha_coherence = single_ch_sync
            beta_coherence = single_ch_sync
            # No interhemispheric PLV possible
            frontal_ih_plv = 0.0

        # Composite synchrony score
        synchrony_score = (
            0.25 * ft_plv_alpha
            + 0.25 * ft_plv_beta
            + 0.25 * alpha_coherence
            + 0.25 * frontal_ih_plv
        )
        synchrony_score = float(np.clip(synchrony_score, 0.0, 1.0))

        return {
            "synchrony_score": synchrony_score,
            "fronto_temporal_plv_alpha": ft_plv_alpha,
            "fronto_temporal_plv_beta": ft_plv_beta,
            "frontal_interhemispheric_plv": frontal_ih_plv,
            "alpha_coherence": alpha_coherence,
            "beta_coherence": beta_coherence,
        }

    def _classify_engagement(self, synchrony_score: float) -> str:
        """Classify engagement level from composite synchrony score."""
        if synchrony_score >= _DEEPLY_ENGAGED_THRESHOLD:
            return "deeply_engaged"
        elif synchrony_score >= _MODERATELY_ENGAGED_THRESHOLD:
            return "moderately_engaged"
        elif synchrony_score >= _MILDLY_ENGAGED_THRESHOLD:
            return "mildly_engaged"
        else:
            return "disengaged"

    def _default_result(self, has_baseline: bool) -> Dict:
        """Return default result for very short or invalid signals."""
        return {
            "synchrony_score": 0.0,
            "fronto_temporal_plv_alpha": 0.0,
            "fronto_temporal_plv_beta": 0.0,
            "frontal_interhemispheric_plv": 0.0,
            "alpha_coherence": 0.0,
            "beta_coherence": 0.0,
            "engagement_level": "disengaged",
            "has_baseline": has_baseline,
        }

    def _append_history(self, user_id: str, result: Dict) -> None:
        """Append result to user history, enforcing cap."""
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > _MAX_HISTORY:
            self._history[user_id] = self._history[user_id][-_MAX_HISTORY:]
