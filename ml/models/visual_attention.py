"""Visual Attention Detector — EEG-based gaze direction and focus estimation.

Detects visual attention direction (left / right / center / diffuse)
and visual focus intensity from 4-channel Muse 2 EEG using alpha
lateralization and event-related desynchronization (ERD).

Scientific basis:
- Alpha lateralization: attending left suppresses right-hemisphere alpha
  (TP10/AF8), attending right suppresses left-hemisphere alpha (TP9/AF7).
  Worden et al. (2000), Thut et al. (2006).
- Alpha ERD: active visual processing desynchronises posterior alpha
  relative to a resting (eyes-closed) baseline. Pfurtscheller & Lopes
  da Silva (1999).
- SSVEP: steady-state visually evoked potentials at stimulus frequency
  indicate sustained visual focus. Not usable without known stimulus,
  so this module focuses on alpha metrics.
- Frontal theta increase during sustained visual attention —
  Cavanagh & Frank (2014).

Muse 2 channel order (BrainFlow board 38):
  ch0 = TP9  (left temporal)
  ch1 = AF7  (left frontal)
  ch2 = AF8  (right frontal)
  ch3 = TP10 (right temporal)

References:
    Worden et al. (2000) — Anticipatory biasing of visuospatial attention
    Thut et al. (2006) — Alpha-band EEG activity over ipsilateral cortex
    Pfurtscheller & Lopes da Silva (1999) — Event-related EEG/MEG
    Cavanagh & Frank (2014) — Frontal theta as a mechanism for cognitive control
"""

import numpy as np
from typing import Dict, List, Optional

from scipy.signal import welch

# Portable trapezoid integration (numpy >= 2.0 renames trapz)
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

# Channel indices for Muse 2
_CH_TP9 = 0
_CH_AF7 = 1
_CH_AF8 = 2
_CH_TP10 = 3

# Frequency bands (Hz)
_ALPHA_LOW = 8.0
_ALPHA_HIGH = 12.0
_THETA_LOW = 4.0
_THETA_HIGH = 8.0
_BETA_LOW = 12.0
_BETA_HIGH = 30.0

# Thresholds
_EYES_CLOSED_ALPHA_THRESHOLD = 0.45  # relative alpha > this = eyes closed
_LATERALITY_CENTER_BAND = 0.15       # |laterality| < this = "center"
_FOCUS_HIGH_THRESHOLD = 60           # score >= this = "focused"
_FOCUS_LOW_THRESHOLD = 30            # score < this = "unfocused"

_HISTORY_CAP = 500


class VisualAttentionDetector:
    """Detect visual attention direction and focus from Muse 2 EEG.

    Uses alpha lateralization (left vs. right hemisphere alpha power)
    to estimate gaze direction, and alpha suppression relative to
    baseline to estimate visual focus intensity.
    """

    def __init__(self, fs: float = 256.0):
        self._fs = fs
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}

    # ── Public API ──────────────────────────────────────────────────

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record resting-state baseline (ideally eyes-closed).

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate. Falls back to constructor value.
            user_id: User identifier for multi-user support.

        Returns:
            Dict with baseline_set (bool) and baseline_alpha (dict).
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        alpha_per_ch = self._alpha_per_channel(signals, fs)
        total_alpha = float(np.mean(list(alpha_per_ch.values())))

        self._baselines[user_id] = {
            "alpha_per_channel": alpha_per_ch,
            "total_alpha": total_alpha,
        }

        return {
            "baseline_set": True,
            "baseline_alpha": {
                k: round(v, 6) for k, v in alpha_per_ch.items()
            },
        }

    def assess(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Assess current visual attention direction and focus.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with attention_direction, laterality_index, visual_focus_score,
            alpha_suppression, attention_state, has_baseline.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        n_channels = signals.shape[0]
        baseline = self._baselines.get(user_id)
        has_baseline = baseline is not None

        # Extract alpha power per channel
        alpha_per_ch = self._alpha_per_channel(signals, fs)

        # Extract band powers for attention state classification
        band_powers = self._extract_band_powers(signals, fs)
        total_power = sum(band_powers.values()) + 1e-10
        relative_alpha = band_powers["alpha"] / total_power

        # ── Laterality index ────────────────────────────────────────
        laterality_index = self._compute_laterality(alpha_per_ch, n_channels)

        # ── Attention direction ─────────────────────────────────────
        attention_direction = self._classify_direction(
            laterality_index, relative_alpha, n_channels
        )

        # ── Alpha suppression (relative to baseline) ───────────────
        alpha_suppression = self._compute_alpha_suppression(
            alpha_per_ch, baseline
        )

        # ── Attention state ─────────────────────────────────────────
        attention_state = self._classify_state(
            relative_alpha, alpha_suppression, band_powers, has_baseline
        )

        # ── Visual focus score (0-100) ──────────────────────────────
        visual_focus_score = self._compute_focus_score(
            alpha_suppression, band_powers, relative_alpha, has_baseline
        )

        result = {
            "attention_direction": attention_direction,
            "laterality_index": round(float(laterality_index), 4),
            "visual_focus_score": round(float(visual_focus_score), 1),
            "alpha_suppression": round(float(alpha_suppression), 4),
            "attention_state": attention_state,
            "has_baseline": has_baseline,
        }

        # Store in history
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > _HISTORY_CAP:
            self._history[user_id] = self._history[user_id][-_HISTORY_CAP:]

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get session statistics for a user.

        Returns:
            Dict with n_epochs, mean_focus, dominant_direction,
            direction_distribution.
        """
        history = self._history.get(user_id, [])
        if not history:
            return {"n_epochs": 0}

        focus_scores = [h["visual_focus_score"] for h in history]
        directions = [h["attention_direction"] for h in history]
        dir_counts: Dict[str, int] = {}
        for d in directions:
            dir_counts[d] = dir_counts.get(d, 0) + 1

        return {
            "n_epochs": len(history),
            "mean_focus": round(float(np.mean(focus_scores)), 1),
            "dominant_direction": max(dir_counts, key=dir_counts.get),
            "direction_distribution": dir_counts,
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Get assessment history.

        Args:
            user_id: User identifier.
            last_n: If set, return only the last N entries.

        Returns:
            List of assessment result dicts.
        """
        history = self._history.get(user_id, [])
        if last_n:
            history = history[-last_n:]
        return history

    def reset(self, user_id: str = "default"):
        """Clear baseline and history for a user."""
        self._baselines.pop(user_id, None)
        self._history.pop(user_id, None)

    # ── Private helpers ─────────────────────────────────────────────

    def _alpha_per_channel(
        self, signals: np.ndarray, fs: float
    ) -> Dict[str, float]:
        """Extract alpha power for each available channel."""
        channel_names = ["TP9", "AF7", "AF8", "TP10"]
        result: Dict[str, float] = {}
        n_ch = min(signals.shape[0], len(channel_names))

        for i in range(n_ch):
            power = self._band_power(signals[i], fs, _ALPHA_LOW, _ALPHA_HIGH)
            result[channel_names[i]] = power

        # If fewer than 4 channels, label generically
        for i in range(n_ch, signals.shape[0]):
            power = self._band_power(signals[i], fs, _ALPHA_LOW, _ALPHA_HIGH)
            result[f"ch{i}"] = power

        return result

    def _band_power(
        self, signal: np.ndarray, fs: float, low: float, high: float
    ) -> float:
        """Compute band power using Welch PSD."""
        nperseg = min(len(signal), int(fs * 2))
        if nperseg < 4:
            return 0.0
        try:
            freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
            mask = (freqs >= low) & (freqs <= high)
            if not np.any(mask):
                return 0.0
            return float(_trapezoid(psd[mask], freqs[mask]))
        except Exception:
            return 0.0

    def _extract_band_powers(
        self, signals: np.ndarray, fs: float
    ) -> Dict[str, float]:
        """Extract average band powers across all channels."""
        bands = {
            "theta": (_THETA_LOW, _THETA_HIGH),
            "alpha": (_ALPHA_LOW, _ALPHA_HIGH),
            "beta": (_BETA_LOW, _BETA_HIGH),
        }
        result: Dict[str, float] = {}
        for band_name, (low, high) in bands.items():
            powers = []
            for ch in range(signals.shape[0]):
                p = self._band_power(signals[ch], fs, low, high)
                powers.append(p)
            result[band_name] = float(np.mean(powers)) if powers else 0.0
        return result

    def _compute_laterality(
        self, alpha_per_ch: Dict[str, float], n_channels: int
    ) -> float:
        """Compute laterality index from alpha asymmetry.

        Convention: negative = attending left, positive = attending right.

        Alpha lateralization: attending left suppresses right-hemisphere
        alpha (AF8/TP10). So when right alpha is LOW relative to left,
        the subject is looking left -> laterality < 0.

        laterality = (left_alpha - right_alpha) / (left_alpha + right_alpha)
        Attending left -> right suppressed -> left > right -> positive raw
        We negate to follow the convention: negative = left attention.
        """
        if n_channels < 2:
            return 0.0

        # Use frontal channels (AF7=left, AF8=right) if available
        if "AF7" in alpha_per_ch and "AF8" in alpha_per_ch:
            left_alpha = alpha_per_ch["AF7"]
            right_alpha = alpha_per_ch["AF8"]
        elif len(alpha_per_ch) >= 2:
            keys = list(alpha_per_ch.keys())
            left_alpha = alpha_per_ch[keys[0]]
            right_alpha = alpha_per_ch[keys[1]]
        else:
            return 0.0

        # Also incorporate temporal if available
        if "TP9" in alpha_per_ch and "TP10" in alpha_per_ch:
            left_alpha = 0.6 * alpha_per_ch["AF7"] + 0.4 * alpha_per_ch["TP9"]
            right_alpha = 0.6 * alpha_per_ch["AF8"] + 0.4 * alpha_per_ch["TP10"]

        denom = left_alpha + right_alpha + 1e-10

        # Raw ratio: (left - right) / (left + right)
        # When attending LEFT, right alpha suppressed -> left > right -> positive
        # Convention: negative = left, positive = right
        # So negate: laterality = -(left - right) / denom = (right - left) / denom
        raw = (right_alpha - left_alpha) / denom
        return float(np.clip(raw, -1.0, 1.0))

    def _classify_direction(
        self, laterality: float, relative_alpha: float, n_channels: int
    ) -> str:
        """Classify attention direction from laterality index."""
        if n_channels < 2:
            return "center"

        # Very low alpha = diffuse / no clear spatial focus
        if relative_alpha < 0.05:
            return "diffuse"

        if abs(laterality) < _LATERALITY_CENTER_BAND:
            return "center"
        elif laterality < 0:
            return "left"
        else:
            return "right"

    def _compute_alpha_suppression(
        self,
        alpha_per_ch: Dict[str, float],
        baseline: Optional[Dict],
    ) -> float:
        """Compute alpha suppression (ERD) relative to baseline.

        0 = no suppression (alpha same as baseline or no baseline).
        1 = complete suppression (alpha = 0).
        """
        current_alpha = float(np.mean(list(alpha_per_ch.values()))) if alpha_per_ch else 0.0

        if baseline is None:
            # Without baseline, estimate suppression from absolute level.
            # Lower alpha = more suppression. Use a heuristic reference.
            # Typical resting alpha ~ 0.3 relative power; we use absolute here.
            # Just return a modest estimate based on current level.
            return float(np.clip(1.0 - current_alpha / (current_alpha + 1e-2), 0.0, 1.0))

        baseline_alpha = baseline.get("total_alpha", current_alpha)
        if baseline_alpha <= 1e-10:
            return 0.0

        # ERD = (baseline - current) / baseline, clipped to [0, 1]
        suppression = (baseline_alpha - current_alpha) / baseline_alpha
        return float(np.clip(suppression, 0.0, 1.0))

    def _classify_state(
        self,
        relative_alpha: float,
        alpha_suppression: float,
        band_powers: Dict[str, float],
        has_baseline: bool,
    ) -> str:
        """Classify visual attention state.

        - eyes_closed: very high bilateral alpha (> threshold)
        - focused: moderate alpha suppression, good beta engagement
        - scanning: low alpha, moderate beta, diffuse
        - unfocused: high alpha without eyes-closed pattern, low beta
        """
        # Eyes closed: dominant alpha
        if relative_alpha > _EYES_CLOSED_ALPHA_THRESHOLD:
            return "eyes_closed"

        # Compute beta engagement indicator
        total = sum(band_powers.values()) + 1e-10
        beta_frac = band_powers["beta"] / total
        theta_frac = band_powers["theta"] / total

        # Focused: good beta, suppressed alpha, low theta
        if beta_frac > 0.3 and relative_alpha < 0.3:
            return "focused"

        # Scanning: moderate beta, low alpha
        if relative_alpha < 0.15 and beta_frac > 0.15:
            return "scanning"

        # High theta, low beta = unfocused / mind wandering
        if theta_frac > 0.4 or beta_frac < 0.15:
            return "unfocused"

        return "scanning"

    def _compute_focus_score(
        self,
        alpha_suppression: float,
        band_powers: Dict[str, float],
        relative_alpha: float,
        has_baseline: bool,
    ) -> float:
        """Compute visual focus score (0-100).

        Components:
        - Alpha suppression (ERD): more suppression = more focused (40%)
        - Beta engagement: higher beta fraction = more engaged (30%)
        - Theta/beta ratio: lower = better sustained attention (30%)
        """
        total = sum(band_powers.values()) + 1e-10
        beta_frac = band_powers["beta"] / total
        theta = band_powers["theta"]
        beta = band_powers["beta"]

        # Alpha suppression component (0-1)
        erd_component = alpha_suppression if has_baseline else float(
            np.clip(1.0 - relative_alpha / 0.5, 0.0, 1.0)
        )

        # Beta engagement component (0-1)
        beta_component = float(np.clip(beta_frac / 0.5, 0.0, 1.0))

        # Theta/beta ratio component (lower = better)
        tbr = theta / (beta + 1e-10)
        tbr_component = float(np.clip(1.0 - tbr / 3.0, 0.0, 1.0))

        score = 100.0 * (
            0.40 * erd_component
            + 0.30 * beta_component
            + 0.30 * tbr_component
        )

        return float(np.clip(score, 0.0, 100.0))
