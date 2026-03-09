"""Affect labeling efficacy tracker from EEG pre/post emotional labeling.

Measures the neural efficacy of affect labeling (putting feelings into words)
by tracking EEG changes before vs after labeling emotional states.

Scientific basis:
- Lieberman et al. (2007): affect labeling activates right VLPFC (near AF8),
  reduces amygdala activation, and attenuates the Late Positive Potential (LPP).
- Kircanski et al. (2012): repeated affect labeling produces habituation of
  emotional arousal, indexed by decreasing LPP amplitude over trials.
- Torre & Lieberman (2018): putting feelings into words dampens emotional
  reactivity as measured by frontal alpha/beta changes.
- Burklund et al. (2014): neural correlates of implicit affect labeling.

Key biomarkers tracked:
1. LPP reduction: lower post-label beta amplitude at frontal channels
   indicates successful emotion regulation via labeling.
2. Prefrontal increase: left-frontal alpha suppression after labeling
   (AF7 alpha decreases) reflects VLPFC engagement.
3. Alpha/beta power changes: overall calming indicated by alpha increase
   or beta decrease post-labeling.
4. Composite efficacy: weighted combination of LPP reduction + prefrontal
   activation.

Note: Without event-locked ERPs (no stimulus onset markers from Muse 2),
we approximate LPP as mean beta-band amplitude in the epoch. The proxy
captures the same general phenomenon: sustained emotional processing
amplitude that should decrease after labeling.

Channel layout (Muse 2, BrainFlow order):
    ch0 = TP9  (left temporal)
    ch1 = AF7  (left frontal)   -- VLPFC proxy
    ch2 = AF8  (right frontal)
    ch3 = TP10 (right temporal)
"""

import numpy as np
from typing import Dict, List, Optional

from scipy.signal import butter, filtfilt, welch


# Muse 2 channel indices
_CH_TP9 = 0
_CH_AF7 = 1
_CH_AF8 = 2
_CH_TP10 = 3

# Maximum history entries per user
_MAX_HISTORY = 500

# Efficacy level thresholds
_THRESH_HIGHLY_EFFECTIVE = 0.7
_THRESH_EFFECTIVE = 0.5
_THRESH_MODERATE = 0.3

# Exported for tests
EFFICACY_LEVELS = [
    "minimal_effect",
    "moderately_effective",
    "effective",
    "highly_effective",
]


def _band_power(signal: np.ndarray, fs: float, low: float, high: float) -> float:
    """Mean PSD in [low, high] Hz via Welch's method."""
    n = len(signal)
    if n < 16:
        return float(np.mean(signal ** 2))
    nperseg = min(256, n // 2)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return float(np.mean(psd))
    return float(np.mean(psd[mask]))


def _extract_metrics(signals: np.ndarray, fs: float) -> Dict:
    """Extract EEG metrics from a multichannel epoch.

    Returns dict with:
        frontal_beta_amp: mean beta amplitude at frontal channels (LPP proxy)
        af7_alpha: alpha power at AF7 (left-frontal)
        af8_alpha: alpha power at AF8 (right-frontal)
        overall_alpha: mean alpha power across all channels
        overall_beta: mean beta power across all channels
    """
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    n_ch = signals.shape[0]

    # Compute per-channel alpha (8-12 Hz) and beta (12-30 Hz) powers
    alphas = []
    betas = []
    for ch in range(n_ch):
        alphas.append(_band_power(signals[ch], fs, 8.0, 12.0))
        betas.append(_band_power(signals[ch], fs, 12.0, 30.0))

    # Frontal beta amplitude (LPP proxy)
    if n_ch >= 3:
        frontal_beta = (betas[_CH_AF7] + betas[_CH_AF8]) / 2.0
    else:
        frontal_beta = betas[0]

    # AF7 and AF8 alpha
    af7_alpha = alphas[_CH_AF7] if n_ch >= 2 else alphas[0]
    af8_alpha = alphas[_CH_AF8] if n_ch >= 3 else alphas[0]

    return {
        "frontal_beta_amp": frontal_beta,
        "af7_alpha": af7_alpha,
        "af8_alpha": af8_alpha,
        "overall_alpha": float(np.mean(alphas)),
        "overall_beta": float(np.mean(betas)),
    }


class AffectLabelingTracker:
    """Track neural efficacy of affect labeling across sessions.

    Maintains per-user state: baseline, pre-label recordings, and
    post-label trial history. Each user_id gets independent storage.

    Usage:
        tracker = AffectLabelingTracker(fs=256.0)
        tracker.set_baseline(resting_eeg, user_id="alice")
        tracker.record_pre_label(pre_eeg, user_id="alice")
        result = tracker.record_post_label(post_eeg, label="anxious", user_id="alice")
        stats = tracker.get_session_stats(user_id="alice")
    """

    def __init__(self, fs: float = 256.0) -> None:
        self.fs = fs
        # Per-user storage: user_id -> dict of state
        self._users: Dict[str, Dict] = {}

    def _ensure_user(self, user_id: str) -> Dict:
        """Lazily initialize per-user storage."""
        if user_id not in self._users:
            self._users[user_id] = {
                "baseline": None,         # Dict from _extract_metrics
                "pre_label": None,        # Dict from _extract_metrics
                "history": [],            # List of post-label result dicts
            }
        return self._users[user_id]

    def set_baseline(self, signals: np.ndarray, fs: Optional[float] = None,
                     user_id: str = "default") -> Dict:
        """Record resting-state baseline EEG.

        Call with 2-3 minutes of eyes-closed resting EEG for best results.
        The baseline is used to normalize pre/post-label metrics.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate. Uses self.fs if None.
            user_id: User identifier for per-user storage.

        Returns:
            Dict with baseline_set=True and n_channels.
        """
        fs = fs if fs is not None else self.fs
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        user = self._ensure_user(user_id)
        user["baseline"] = _extract_metrics(signals, fs)

        return {
            "baseline_set": True,
            "n_channels": signals.shape[0],
        }

    def record_pre_label(self, signals: np.ndarray, fs: Optional[float] = None,
                         user_id: str = "default") -> Dict:
        """Record EEG epoch BEFORE user labels their emotion.

        Captures emotional reactivity state. Call this right before
        presenting the labeling prompt.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate. Uses self.fs if None.
            user_id: User identifier.

        Returns:
            Dict confirming pre-label was recorded.
        """
        fs = fs if fs is not None else self.fs
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        user = self._ensure_user(user_id)
        user["pre_label"] = _extract_metrics(signals, fs)

        return {
            "pre_label_recorded": True,
            "n_channels": signals.shape[0],
        }

    def record_post_label(self, signals: np.ndarray, label: Optional[str] = None,
                          fs: Optional[float] = None,
                          user_id: str = "default") -> Dict:
        """Record EEG epoch AFTER user labels their emotion.

        Computes labeling efficacy by comparing pre-label vs post-label
        metrics. If no pre-label was recorded, uses baseline or zero
        defaults.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
            label: Optional text of the emotion label the user chose.
            fs: Sampling rate. Uses self.fs if None.
            user_id: User identifier.

        Returns:
            Dict with labeling_efficacy, lpp_reduction, prefrontal_increase,
            alpha_change, beta_change, efficacy_level, has_baseline.
        """
        fs = fs if fs is not None else self.fs
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        user = self._ensure_user(user_id)
        post = _extract_metrics(signals, fs)

        # Get pre-label metrics (fall back to defaults if not recorded)
        pre = user.get("pre_label")
        if pre is None:
            pre = {
                "frontal_beta_amp": 0.0,
                "af7_alpha": 0.0,
                "af8_alpha": 0.0,
                "overall_alpha": 0.0,
                "overall_beta": 0.0,
            }

        baseline = user.get("baseline")
        has_baseline = baseline is not None

        # --- LPP reduction (frontal beta decrease) ---
        # Positive = beta decreased after labeling = good
        pre_beta = pre["frontal_beta_amp"]
        post_beta = post["frontal_beta_amp"]
        if pre_beta > 1e-12:
            lpp_reduction = float((pre_beta - post_beta) / pre_beta)
        else:
            lpp_reduction = 0.0
        lpp_reduction = float(np.clip(lpp_reduction, -1.0, 1.0))

        # --- Prefrontal increase (AF7 alpha suppression = left VLPFC activation) ---
        # Alpha suppression at AF7 means MORE activation.
        # Positive = alpha decreased at AF7 after labeling = VLPFC active = good
        pre_af7_alpha = pre["af7_alpha"]
        post_af7_alpha = post["af7_alpha"]
        if pre_af7_alpha > 1e-12:
            prefrontal_increase = float(
                (pre_af7_alpha - post_af7_alpha) / pre_af7_alpha
            )
        else:
            prefrontal_increase = 0.0
        prefrontal_increase = float(np.clip(prefrontal_increase, -1.0, 1.0))

        # --- Alpha change (overall) ---
        pre_alpha = pre["overall_alpha"]
        post_alpha = post["overall_alpha"]
        if pre_alpha > 1e-12:
            alpha_change = float((post_alpha - pre_alpha) / pre_alpha)
        else:
            alpha_change = 0.0
        alpha_change = float(np.clip(alpha_change, -2.0, 2.0))

        # --- Beta change (overall) ---
        pre_overall_beta = pre["overall_beta"]
        post_overall_beta = post["overall_beta"]
        if pre_overall_beta > 1e-12:
            beta_change = float((post_overall_beta - pre_overall_beta) / pre_overall_beta)
        else:
            beta_change = 0.0
        beta_change = float(np.clip(beta_change, -2.0, 2.0))

        # --- Composite efficacy ---
        # 60% LPP reduction (primary biomarker) + 40% prefrontal increase
        # Both are positive when labeling works, so composite is 0-1
        efficacy_raw = (
            0.60 * max(0.0, lpp_reduction)
            + 0.40 * max(0.0, prefrontal_increase)
        )
        labeling_efficacy = float(np.clip(efficacy_raw, 0.0, 1.0))

        # --- Efficacy level ---
        if labeling_efficacy >= _THRESH_HIGHLY_EFFECTIVE:
            efficacy_level = "highly_effective"
        elif labeling_efficacy >= _THRESH_EFFECTIVE:
            efficacy_level = "effective"
        elif labeling_efficacy >= _THRESH_MODERATE:
            efficacy_level = "moderately_effective"
        else:
            efficacy_level = "minimal_effect"

        result: Dict = {
            "labeling_efficacy": round(labeling_efficacy, 4),
            "lpp_reduction": round(lpp_reduction, 4),
            "prefrontal_increase": round(prefrontal_increase, 4),
            "alpha_change": round(alpha_change, 4),
            "beta_change": round(beta_change, 4),
            "efficacy_level": efficacy_level,
            "has_baseline": has_baseline,
        }
        if label is not None:
            result["label"] = label

        # Store in history (cap at _MAX_HISTORY)
        history = user["history"]
        history.append(result)
        if len(history) > _MAX_HISTORY:
            user["history"] = history[-_MAX_HISTORY:]

        # Clear pre-label after use
        user["pre_label"] = None

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get session statistics for a user.

        Returns:
            Dict with n_trials, mean_efficacy, and improvement_over_session.
            improvement_over_session compares the mean efficacy of the last
            third of trials to the first third (positive = improving).
        """
        user = self._ensure_user(user_id)
        history = user["history"]

        if not history:
            return {
                "n_trials": 0,
                "mean_efficacy": 0.0,
                "improvement_over_session": 0.0,
            }

        efficacies = [h["labeling_efficacy"] for h in history]
        n = len(efficacies)
        mean_eff = float(np.mean(efficacies))

        # Improvement: compare last third vs first third
        if n >= 3:
            third = max(1, n // 3)
            first_third = float(np.mean(efficacies[:third]))
            last_third = float(np.mean(efficacies[-third:]))
            improvement = last_third - first_third
        else:
            improvement = 0.0

        return {
            "n_trials": n,
            "mean_efficacy": round(mean_eff, 4),
            "improvement_over_session": round(improvement, 4),
        }

    def get_history(self, user_id: str = "default",
                    last_n: Optional[int] = None) -> List[Dict]:
        """Get trial history for a user.

        Args:
            user_id: User identifier.
            last_n: If provided, return only the last N entries.

        Returns:
            List of post-label result dicts.
        """
        user = self._ensure_user(user_id)
        history = user["history"]
        if last_n is not None and last_n > 0:
            return list(history[-last_n:])
        return list(history)

    def reset(self, user_id: str = "default") -> None:
        """Clear all data for a user."""
        if user_id in self._users:
            del self._users[user_id]
