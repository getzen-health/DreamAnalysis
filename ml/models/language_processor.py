"""EEG-based language processing and reading comprehension detector.

Detects language processing states from 4-channel Muse 2 EEG using
spectral markers that correlate with linguistic cognition:

- Frontal theta (4-8 Hz) increase during sentence comprehension and
  semantic processing (N400-related spectral signature)
- Left-hemisphere dominance: AF7 > AF8 activation during language tasks
- Temporal alpha (8-12 Hz) suppression at TP9/TP10 during auditory
  language processing
- Beta desynchronization during semantic integration
- Frontal theta intensity as working memory load marker

Note: Muse 2 cannot detect single-trial ERPs (N400/P600) reliably due
to low electrode count and dry electrode noise. This detector uses
sustained spectral markers that correlate with language processing
states rather than direct ERP detection.

States (by comprehension_index):
    disengaged (0-25): No language processing detected
    passive_listening (25-50): Hearing but shallow processing
    active_processing (50-75): Engaged comprehension
    deep_comprehension (75-100): Deep semantic integration

Scientific basis:
    Bastiaansen et al. (2005) — Theta increases during sentence processing
    Weiss & Mueller (2012) — Theta-gamma coupling in language comprehension
    Salmelin (2007) — Left-hemisphere dominance in language processing
    Klimesch (1999) — Alpha suppression during cognitive engagement
    Hald et al. (2006) — Theta power and semantic integration
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.signal import welch

# NumPy 2.0 renamed np.trapz -> np.trapezoid; 1.x only has np.trapz
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

# Muse 2 channel layout (BrainFlow board_id 22/38)
CH_TP9 = 0   # Left temporal
CH_AF7 = 1   # Left frontal
CH_AF8 = 2   # Right frontal
CH_TP10 = 3  # Right temporal

# EEG frequency band definitions (Hz)
BAND_THETA = (4.0, 8.0)
BAND_ALPHA = (8.0, 12.0)
BAND_BETA = (12.0, 30.0)
BAND_LOW_BETA = (12.0, 20.0)

# Processing state labels in order of comprehension_index thresholds
PROCESSING_STATES = [
    "disengaged",          # 0-25
    "passive_listening",   # 25-50
    "active_processing",   # 50-75
    "deep_comprehension",  # 75-100
]

HISTORY_CAP = 500


class LanguageProcessor:
    """EEG-based language processing and reading comprehension detector.

    Designed for Muse 2 (4-channel, 256 Hz). Uses spectral markers —
    frontal theta, hemispheric asymmetry, temporal alpha suppression,
    and beta desynchronization — to estimate language processing state.
    """

    def __init__(self, fs: float = 256.0):
        """Initialize the language processor.

        Args:
            fs: Default sampling rate in Hz (Muse 2 = 256).
        """
        self.fs = fs
        self._baselines: Dict[str, Dict[str, float]] = {}
        self._history: Dict[str, List[Dict]] = {}

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record resting-state baseline for normalization.

        Should be called with 30-120 seconds of resting EEG (eyes open,
        no reading or listening). Features are normalized against this
        baseline during assess().

        Args:
            signals: EEG array — (4, n_samples) or (n_samples,).
            fs: Sampling rate override. Uses constructor fs if None.
            user_id: User identifier for multi-user support.

        Returns:
            Dict with baseline_set (bool) and baseline_metrics (dict).
        """
        use_fs = fs if fs is not None else self.fs
        signals = self._sanitize_input(signals)
        channels, primary = self._extract_channels(signals)

        metrics = self._compute_spectral_features(primary, channels, use_fs)
        self._baselines[user_id] = metrics

        return {
            "baseline_set": True,
            "baseline_metrics": {k: round(v, 4) for k, v in metrics.items()},
        }

    def assess(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Assess language processing state from current EEG.

        Args:
            signals: EEG array — (4, n_samples), (n_channels, n_samples),
                     or (n_samples,) for single-channel.
            fs: Sampling rate override. Uses constructor fs if None.
            user_id: User identifier for multi-user support.

        Returns:
            Dict with processing_state, comprehension_index (0-100),
            semantic_load (0-1), left_lateralization (-1 to 1),
            temporal_engagement (0-1), working_memory_load (0-1),
            and has_baseline (bool).
        """
        use_fs = fs if fs is not None else self.fs
        signals = self._sanitize_input(signals)
        channels, primary = self._extract_channels(signals)

        features = self._compute_spectral_features(primary, channels, use_fs)
        baseline = self._baselines.get(user_id)
        has_baseline = baseline is not None

        # --- Compute component scores ---

        # 1. Semantic load: frontal theta relative to total power
        #    Higher frontal theta = more semantic processing
        frontal_theta = features["frontal_theta"]
        if has_baseline:
            bl_theta = max(baseline["frontal_theta"], 1e-10)
            theta_ratio = frontal_theta / bl_theta
            semantic_load = float(np.clip(np.tanh((theta_ratio - 1.0) * 2.0), 0, 1))
        else:
            # Without baseline, use absolute theta fraction
            semantic_load = float(np.clip(np.tanh(frontal_theta * 4.0), 0, 1))

        # 2. Left lateralization: AF7 vs AF8 theta power
        #    Positive = left dominant (language active)
        left_lat = features["left_lateralization"]

        # 3. Temporal engagement: alpha suppression at TP9/TP10
        #    Low temporal alpha = engaged in auditory/language processing
        #    Only computable with 4-channel data (need TP9/TP10)
        temporal_alpha = features["temporal_alpha"]
        has_temporal = features.get("has_temporal", False)
        if not has_temporal:
            # Cannot compute temporal engagement without TP9/TP10
            temporal_engagement = 0.0
        elif has_baseline:
            bl_temp_alpha = max(baseline["temporal_alpha"], 1e-10)
            alpha_ratio = temporal_alpha / bl_temp_alpha
            # Suppression: lower alpha relative to baseline = more engaged
            temporal_engagement = float(np.clip(1.0 - np.tanh(alpha_ratio * 1.5), 0, 1))
        else:
            # Without baseline: lower alpha = more engaged
            temporal_engagement = float(np.clip(1.0 - np.tanh(temporal_alpha * 3.0), 0, 1))

        # 4. Working memory load: average frontal theta intensity
        frontal_theta_avg = features["frontal_theta_avg"]
        if has_baseline:
            bl_theta_avg = max(baseline["frontal_theta_avg"], 1e-10)
            wm_ratio = frontal_theta_avg / bl_theta_avg
            working_memory_load = float(np.clip(np.tanh((wm_ratio - 1.0) * 2.0), 0, 1))
        else:
            working_memory_load = float(np.clip(np.tanh(frontal_theta_avg * 4.0), 0, 1))

        # 5. Beta desynchronization: lower beta during semantic processing
        beta_desynch = features["beta_desynchronization"]

        # --- Comprehension index (0-100) ---
        # Weighted sum of all component scores
        raw_score = (
            0.30 * semantic_load
            + 0.25 * max(left_lat, 0.0)  # Only positive lateralization contributes
            + 0.20 * temporal_engagement
            + 0.15 * working_memory_load
            + 0.10 * beta_desynch
        )
        comprehension_index = float(np.clip(raw_score * 100, 0, 100))

        # --- State classification ---
        if comprehension_index >= 75:
            state = "deep_comprehension"
        elif comprehension_index >= 50:
            state = "active_processing"
        elif comprehension_index >= 25:
            state = "passive_listening"
        else:
            state = "disengaged"

        result = {
            "processing_state": state,
            "comprehension_index": round(comprehension_index, 1),
            "semantic_load": round(semantic_load, 4),
            "left_lateralization": round(left_lat, 4),
            "temporal_engagement": round(temporal_engagement, 4),
            "working_memory_load": round(working_memory_load, 4),
            "has_baseline": has_baseline,
        }

        # Track history
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > HISTORY_CAP:
            self._history[user_id] = self._history[user_id][-HISTORY_CAP:]

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get aggregate statistics for the current session.

        Args:
            user_id: User identifier.

        Returns:
            Dict with n_epochs, mean_comprehension, dominant_state,
            and state_distribution.
        """
        history = self._history.get(user_id, [])
        if not history:
            return {
                "n_epochs": 0,
                "mean_comprehension": 0.0,
                "dominant_state": "disengaged",
                "state_distribution": {s: 0.0 for s in PROCESSING_STATES},
            }

        comprehensions = [h["comprehension_index"] for h in history]
        states = [h["processing_state"] for h in history]
        n = len(history)

        # State distribution
        dist = {}
        for s in PROCESSING_STATES:
            dist[s] = round(states.count(s) / n, 4)

        # Dominant state = most frequent
        dominant = max(dist, key=dist.get)

        return {
            "n_epochs": n,
            "mean_comprehension": round(float(np.mean(comprehensions)), 1),
            "dominant_state": dominant,
            "state_distribution": dist,
        }

    def get_history(
        self,
        user_id: str = "default",
        last_n: Optional[int] = None,
    ) -> List[Dict]:
        """Get assessment history for a user.

        Args:
            user_id: User identifier.
            last_n: If provided, return only the last N entries.

        Returns:
            List of assessment result dicts.
        """
        history = list(self._history.get(user_id, []))
        if last_n is not None and last_n < len(history):
            return history[-last_n:]
        return history

    def reset(self, user_id: str = "default"):
        """Clear history and baseline for a user.

        Args:
            user_id: User identifier.
        """
        self._history.pop(user_id, None)
        self._baselines.pop(user_id, None)

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _sanitize_input(self, signals: np.ndarray) -> np.ndarray:
        """Convert to float64 and replace NaN with zero."""
        out = np.array(signals, dtype=np.float64)
        if np.any(np.isnan(out)):
            out = np.nan_to_num(out, nan=0.0)
        return out

    def _extract_channels(self, signals: np.ndarray):
        """Extract channel arrays and primary frontal signal.

        Returns:
            (channels_2d_or_None, primary_1d)
            channels is the full multichannel array if ndim==2, else None.
            primary is AF7 (ch1) if available, else first/only channel.
        """
        if signals.ndim == 2 and signals.shape[0] >= 2:
            # Multichannel — use AF7 as primary for frontal theta
            primary_idx = min(CH_AF7, signals.shape[0] - 1)
            return signals, signals[primary_idx]
        elif signals.ndim == 2 and signals.shape[0] == 1:
            return None, signals[0]
        else:
            return None, signals

    def _compute_band_power(
        self,
        signal_1d: np.ndarray,
        fs: float,
        band: tuple,
    ) -> float:
        """Compute relative band power using Welch PSD.

        Args:
            signal_1d: 1D EEG signal.
            fs: Sampling rate.
            band: (low_freq, high_freq) tuple.

        Returns:
            Relative band power (0-1 range).
        """
        nperseg = min(len(signal_1d), int(fs * 2))
        if nperseg < 4:
            return 0.0

        freqs, psd = welch(signal_1d, fs=fs, nperseg=nperseg)
        total_power = _trapezoid(psd, freqs)
        if total_power <= 0:
            return 0.0

        mask = (freqs >= band[0]) & (freqs <= band[1])
        if not np.any(mask):
            return 0.0
        band_power = _trapezoid(psd[mask], freqs[mask])
        return float(band_power / total_power)

    def _compute_spectral_features(
        self,
        primary: np.ndarray,
        channels: Optional[np.ndarray],
        fs: float,
    ) -> Dict[str, float]:
        """Compute all spectral features needed for language processing assessment.

        Args:
            primary: 1D EEG signal (AF7 or only channel).
            channels: Full multichannel array or None.
            fs: Sampling rate.

        Returns:
            Dict with frontal_theta, frontal_theta_avg, left_lateralization,
            temporal_alpha, beta_desynchronization.
        """
        # Frontal theta from primary (AF7) channel
        frontal_theta = self._compute_band_power(primary, fs, BAND_THETA)

        # Default values for single-channel case
        left_lat = 0.0
        temporal_alpha = 0.0
        frontal_theta_avg = frontal_theta
        beta_desynch = 0.0
        has_temporal = False

        if channels is not None and channels.shape[0] >= 2:
            n_ch = channels.shape[0]

            # Frontal theta average (AF7 + AF8 if available)
            af7_idx = min(CH_AF7, n_ch - 1)
            af8_idx = min(CH_AF8, n_ch - 1)

            af7_theta = self._compute_band_power(channels[af7_idx], fs, BAND_THETA)
            af8_theta = self._compute_band_power(channels[af8_idx], fs, BAND_THETA)

            # Only compute lateralization if we have distinct AF7 and AF8
            if n_ch >= 3:  # At least 3 channels means AF7 and AF8 are distinct
                frontal_theta_avg = (af7_theta + af8_theta) / 2.0

                # Left lateralization:
                # Positive = left (AF7) has more theta power = language active
                # Use log ratio for normalization (like FAA formula)
                log_af7 = np.log(max(af7_theta, 1e-10))
                log_af8 = np.log(max(af8_theta, 1e-10))
                raw_lat = log_af7 - log_af8
                left_lat = float(np.clip(np.tanh(raw_lat * 2.0), -1.0, 1.0))
            else:
                # 2-channel: cannot distinguish AF7 from AF8
                frontal_theta_avg = (af7_theta + af8_theta) / 2.0

            # Temporal alpha suppression (TP9/TP10 if we have 4 channels)
            if n_ch >= 4:
                tp9_alpha = self._compute_band_power(channels[CH_TP9], fs, BAND_ALPHA)
                tp10_alpha = self._compute_band_power(channels[CH_TP10], fs, BAND_ALPHA)
                temporal_alpha = (tp9_alpha + tp10_alpha) / 2.0
                has_temporal = True

            # Beta desynchronization at frontal sites
            af7_beta = self._compute_band_power(channels[af7_idx], fs, BAND_BETA)
            af8_beta = self._compute_band_power(channels[af8_idx], fs, BAND_BETA) if n_ch >= 3 else af7_beta
            frontal_beta = (af7_beta + af8_beta) / 2.0
            # Lower beta = more desynchronization = more semantic processing
            beta_desynch = float(np.clip(1.0 - np.tanh(frontal_beta * 4.0), 0, 1))

        return {
            "frontal_theta": frontal_theta,
            "frontal_theta_avg": frontal_theta_avg,
            "left_lateralization": left_lat,
            "temporal_alpha": temporal_alpha,
            "beta_desynchronization": beta_desynch,
            "has_temporal": has_temporal,
        }
