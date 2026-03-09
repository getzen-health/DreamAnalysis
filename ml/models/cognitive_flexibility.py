"""Cognitive Flexibility Detector via frontal theta and alpha desynchronization.

Measures cognitive flexibility / task-switching ability from EEG signals.
Frontal theta power (AF7/AF8) increases during task switching in flexible
individuals. Alpha desynchronization (event-related decrease) reflects
attentional reorientation during set-shifting.

Key markers:
  - Frontal theta power (4-8 Hz at AF7/AF8): primary marker of cognitive
    control during task switching. Higher theta during switch trials
    indicates greater executive control engagement.
  - Switch cost: theta power increase when switching vs sustaining attention.
    Lower switch cost = higher flexibility.
  - Alpha desynchronization: alpha power decrease during task switching
    reflects attentional disengagement from the prior task set.
  - Frontal-parietal theta coherence: inter-regional coordination during
    cognitive control. With Muse 2 we approximate this using AF7/AF8
    (frontal) vs TP9/TP10 (temporal-parietal proxy).

References:
    Monsell, S. (2003). Task switching. Trends in Cognitive Sciences, 7(3),
        134-140. Foundational review of task-switching paradigms and switch
        cost as a measure of cognitive flexibility.
    Sauseng, P., Klimesch, W., Freunberger, R., Pecherstorfer, T.,
        Hanslmayr, S., & Doppelmayr, M. (2006). Relevance of EEG alpha and
        theta oscillations during task switching. Experimental Brain
        Research, 170(3), 295-301. Frontal theta increases and posterior
        alpha decreases during task switching.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal

# NumPy 2.0 renamed np.trapz -> np.trapezoid
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


def _band_power(
    signal: np.ndarray, fs: float, low: float, high: float
) -> float:
    """Compute band power via Welch PSD for a single channel."""
    nperseg = min(len(signal), int(fs * 2))
    if nperseg < 4:
        return 0.0
    try:
        freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)
    except Exception:
        return 0.0
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(_trapezoid(psd[mask], freqs[mask]))


def _theta_power(signal: np.ndarray, fs: float) -> float:
    """Theta (4-8 Hz) band power."""
    return _band_power(signal, fs, 4.0, 8.0)


def _alpha_power(signal: np.ndarray, fs: float) -> float:
    """Alpha (8-12 Hz) band power."""
    return _band_power(signal, fs, 8.0, 12.0)


def _beta_power(signal: np.ndarray, fs: float) -> float:
    """Beta (12-30 Hz) band power."""
    return _band_power(signal, fs, 12.0, 30.0)


def _frontal_theta_coherence(
    signals: np.ndarray, fs: float
) -> float:
    """Approximate frontal-parietal theta coherence.

    Uses AF7/AF8 (frontal, ch1/ch2) vs TP9/TP10 (temporal-parietal proxy,
    ch0/ch3). Averages coherence across frontal-temporal channel pairs in
    the theta band (4-8 Hz).
    """
    if signals.ndim != 2 or signals.shape[0] < 4:
        return 0.0

    frontal_chs = [1, 2]   # AF7, AF8
    parietal_chs = [0, 3]  # TP9, TP10 (temporal-parietal proxy)
    coh_values = []
    nperseg = min(signals.shape[1], int(fs * 2))
    if nperseg < 4:
        return 0.0

    for f_ch in frontal_chs:
        for p_ch in parietal_chs:
            try:
                freqs, coh = scipy_signal.coherence(
                    signals[f_ch], signals[p_ch], fs=fs, nperseg=nperseg
                )
                mask = (freqs >= 4.0) & (freqs <= 8.0)
                if np.any(mask):
                    coh_values.append(float(np.mean(coh[mask])))
            except Exception:
                pass

    return float(np.mean(coh_values)) if coh_values else 0.0


class CognitiveFlexibilityDetector:
    """Detect cognitive flexibility / task-switching ability from EEG.

    Tracks switch vs sustain trials separately and computes switch cost
    as the theta power difference between switching and sustaining attention.

    Usage:
        detector = CognitiveFlexibilityDetector()
        # Optional: set resting baseline
        detector.set_baseline(resting_eeg, fs=256)
        # Assess each trial
        result = detector.assess(eeg_signals, fs=256, is_switch_trial=True)

    Cognitive state thresholds (flexibility_score 0-100):
        0-25:   rigid
        25-50:  moderate
        50-75:  flexible
        75-100: highly_flexible
    """

    _STATE_THRESHOLDS = [
        (75, "highly_flexible"),
        (50, "flexible"),
        (25, "moderate"),
        (0, "rigid"),
    ]

    def __init__(self) -> None:
        self._baseline_theta: Optional[float] = None
        self._baseline_alpha: Optional[float] = None
        self._baseline_set: bool = False

        # Per-trial type history
        self._switch_history: List[Dict] = []
        self._sustain_history: List[Dict] = []
        self._all_history: List[Dict] = []

    # ---- Public API -------------------------------------------------

    def set_baseline(
        self,
        eeg: np.ndarray,
        fs: float = 256,
        user_id: str = "default",
    ) -> Dict:
        """Record resting-state baseline theta and alpha power.

        Call during a passive rest period before task-switching trials.

        Args:
            eeg: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate in Hz.
            user_id: User identifier (reserved for future multi-user).

        Returns:
            Dict with baseline_theta, baseline_alpha, baseline_set.
        """
        eeg = np.asarray(eeg, dtype=float)
        frontal = self._get_frontal_signal(eeg)

        self._baseline_theta = _theta_power(frontal, fs)
        self._baseline_alpha = _alpha_power(frontal, fs)
        self._baseline_set = True

        return {
            "baseline_theta": round(self._baseline_theta, 6),
            "baseline_alpha": round(self._baseline_alpha, 6),
            "baseline_set": True,
        }

    def assess(
        self,
        eeg_signals: np.ndarray,
        fs: float = 256,
        is_switch_trial: bool = False,
    ) -> Dict:
        """Assess cognitive flexibility from an EEG epoch.

        Args:
            eeg_signals: (n_channels, n_samples) or (n_samples,) EEG.
            fs: Sampling rate in Hz.
            is_switch_trial: True if this epoch is during a task-switch,
                False if sustaining the same task set.

        Returns:
            Dict with flexibility_score (0-100), switch_cost (0-1),
            frontal_theta_power, alpha_suppression, theta_coherence,
            cognitive_state, recommendations, trial_type.
        """
        eeg_signals = np.asarray(eeg_signals, dtype=float)
        frontal = self._get_frontal_signal(eeg_signals)

        # -- Core features --
        theta = _theta_power(frontal, fs)
        alpha = _alpha_power(frontal, fs)
        beta = _beta_power(frontal, fs)

        # Theta coherence (frontal-parietal proxy)
        theta_coh = _frontal_theta_coherence(eeg_signals, fs)

        # -- Alpha suppression (relative to baseline) --
        if self._baseline_alpha and self._baseline_alpha > 1e-12:
            alpha_suppression = float(
                np.clip(1.0 - alpha / self._baseline_alpha, 0.0, 1.0)
            )
        else:
            # No baseline: estimate from alpha/beta ratio
            total = alpha + beta + 1e-12
            alpha_suppression = float(np.clip(1.0 - alpha / total, 0.0, 1.0))

        # -- Switch cost --
        switch_cost = self._compute_switch_cost(theta, is_switch_trial)

        # -- Flexibility score (0-100) --
        flexibility_score = self._compute_flexibility_score(
            theta, alpha_suppression, theta_coh, switch_cost
        )

        # -- Cognitive state label --
        cognitive_state = self._score_to_state(flexibility_score)

        # -- Recommendations --
        recommendations = self._generate_recommendations(
            flexibility_score, cognitive_state, switch_cost, alpha_suppression
        )

        result = {
            "flexibility_score": round(flexibility_score, 2),
            "switch_cost": round(switch_cost, 4),
            "frontal_theta_power": round(theta, 6),
            "alpha_suppression": round(alpha_suppression, 4),
            "theta_coherence": round(theta_coh, 4),
            "cognitive_state": cognitive_state,
            "recommendations": recommendations,
            "trial_type": "switch" if is_switch_trial else "sustain",
            "has_baseline": self._baseline_set,
        }

        # Record in history
        self._all_history.append(result)
        if is_switch_trial:
            self._switch_history.append(result)
        else:
            self._sustain_history.append(result)

        # Cap history length
        max_history = 1000
        if len(self._all_history) > max_history:
            self._all_history = self._all_history[-max_history:]
        if len(self._switch_history) > max_history:
            self._switch_history = self._switch_history[-max_history:]
        if len(self._sustain_history) > max_history:
            self._sustain_history = self._sustain_history[-max_history:]

        return result

    def get_session_stats(self) -> Dict:
        """Get aggregate statistics for the current session.

        Returns:
            Dict with n_trials, n_switch, n_sustain, mean_flexibility,
            mean_switch_cost, has_baseline.
        """
        if not self._all_history:
            return {
                "n_trials": 0,
                "n_switch": 0,
                "n_sustain": 0,
                "has_baseline": self._baseline_set,
            }

        scores = [h["flexibility_score"] for h in self._all_history]
        costs = [h["switch_cost"] for h in self._all_history]

        return {
            "n_trials": len(self._all_history),
            "n_switch": len(self._switch_history),
            "n_sustain": len(self._sustain_history),
            "mean_flexibility": round(float(np.mean(scores)), 2),
            "mean_switch_cost": round(float(np.mean(costs)), 4),
            "min_flexibility": round(float(np.min(scores)), 2),
            "max_flexibility": round(float(np.max(scores)), 2),
            "has_baseline": self._baseline_set,
        }

    def get_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """Get assessment history.

        Args:
            last_n: If provided, return only the last N entries.

        Returns:
            List of assessment result dicts.
        """
        if last_n is not None and last_n > 0:
            return list(self._all_history[-last_n:])
        return list(self._all_history)

    def reset(self) -> None:
        """Clear all state: baseline, history, and trial records."""
        self._baseline_theta = None
        self._baseline_alpha = None
        self._baseline_set = False
        self._switch_history.clear()
        self._sustain_history.clear()
        self._all_history.clear()

    # ---- Private helpers -------------------------------------------

    def _get_frontal_signal(self, eeg: np.ndarray) -> np.ndarray:
        """Extract frontal signal from multichannel EEG.

        For 4-channel Muse 2: averages AF7 (ch1) and AF8 (ch2).
        For 2-3 channels: uses ch1 (AF7) if available, else ch0.
        For 1D: returns as-is.
        """
        if eeg.ndim == 1:
            return eeg
        if eeg.shape[0] >= 3:
            return (eeg[1] + eeg[2]) / 2.0  # AF7 + AF8 average
        if eeg.shape[0] >= 2:
            return eeg[1]  # AF7
        return eeg[0]

    def _compute_switch_cost(
        self, current_theta: float, is_switch_trial: bool
    ) -> float:
        """Compute switch cost as normalized theta difference.

        Switch cost = (mean_switch_theta - mean_sustain_theta) / max_theta.
        Higher switch cost = more effort to switch = lower flexibility.
        Returns 0.0 when insufficient data.
        """
        if not self._switch_history and not self._sustain_history:
            # No prior data: return moderate default
            return 0.5

        if is_switch_trial:
            switch_thetas = [h["frontal_theta_power"] for h in self._switch_history]
            switch_thetas.append(current_theta)
        else:
            switch_thetas = [h["frontal_theta_power"] for h in self._switch_history]

        if not is_switch_trial:
            sustain_thetas = [h["frontal_theta_power"] for h in self._sustain_history]
            sustain_thetas.append(current_theta)
        else:
            sustain_thetas = [h["frontal_theta_power"] for h in self._sustain_history]

        if not switch_thetas or not sustain_thetas:
            return 0.5

        mean_switch = float(np.mean(switch_thetas))
        mean_sustain = float(np.mean(sustain_thetas))
        max_theta = max(mean_switch, mean_sustain, 1e-12)

        raw_cost = (mean_switch - mean_sustain) / max_theta
        return float(np.clip(raw_cost, 0.0, 1.0))

    def _compute_flexibility_score(
        self,
        theta: float,
        alpha_suppression: float,
        theta_coh: float,
        switch_cost: float,
    ) -> float:
        """Compute composite flexibility score (0-100).

        Components:
          - Theta engagement (35%): higher frontal theta during cognitive
            control = more flexible executive function.
          - Alpha suppression (25%): greater alpha decrease = better
            attentional reorientation.
          - Theta coherence (20%): higher frontal-parietal coordination
            = better top-down control.
          - Switch cost inverted (20%): lower switch cost = higher
            flexibility.
        """
        # Theta engagement score (0-1)
        if self._baseline_theta and self._baseline_theta > 1e-12:
            theta_ratio = theta / self._baseline_theta
            # Score peaks at 1.5x baseline, saturates at 2x
            theta_score = float(np.clip((theta_ratio - 0.5) / 1.5, 0.0, 1.0))
        else:
            # Without baseline, normalize against typical theta power range
            # Typical frontal theta power: 0.001 - 0.1 (Welch PSD units)
            theta_score = float(np.clip(theta * 50.0, 0.0, 1.0))

        # Alpha suppression score (already 0-1)
        alpha_score = alpha_suppression

        # Theta coherence score (already ~0-1)
        coh_score = float(np.clip(theta_coh, 0.0, 1.0))

        # Switch cost score (inverted: low cost = high flexibility)
        cost_score = 1.0 - switch_cost

        # Weighted composite
        raw = (
            0.35 * theta_score
            + 0.25 * alpha_score
            + 0.20 * coh_score
            + 0.20 * cost_score
        )

        return float(np.clip(raw * 100.0, 0.0, 100.0))

    def _score_to_state(self, score: float) -> str:
        """Map flexibility score to cognitive state label."""
        for threshold, label in self._STATE_THRESHOLDS:
            if score >= threshold:
                return label
        return "rigid"

    def _generate_recommendations(
        self,
        score: float,
        state: str,
        switch_cost: float,
        alpha_suppression: float,
    ) -> List[str]:
        """Generate actionable recommendations based on assessment."""
        recs: List[str] = []

        if state == "rigid":
            recs.append(
                "Practice task-switching exercises to improve cognitive flexibility."
            )
            recs.append(
                "Try alternating between different types of mental tasks every 5-10 minutes."
            )
        elif state == "moderate":
            recs.append(
                "Flexibility is developing. Continue varied cognitive challenges."
            )

        if switch_cost > 0.7:
            recs.append(
                "High switch cost detected. Brief mindfulness breaks between "
                "tasks may reduce transition friction."
            )

        if alpha_suppression < 0.2:
            recs.append(
                "Low alpha suppression suggests difficulty disengaging from "
                "prior task. Practice focused attention meditation."
            )

        if state in ("flexible", "highly_flexible") and not recs:
            recs.append(
                "Strong cognitive flexibility. Maintain with varied intellectual activities."
            )

        return recs
