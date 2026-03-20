"""Objective Meditation Depth Quantifier using validated EEG markers.

Quantifies meditation depth on a 0-100 scale using four research-validated
EEG markers, each measurable from the Muse 2's 4-channel dry electrode system:

1. Frontal Midline Theta (FMT) power -- primary marker of internalized
   attention and meditation depth (Lomas et al., 2015; Kubota et al., 2001).
   Approximated on Muse 2 by averaging AF7+AF8 theta.

2. Alpha coherence between hemispheres -- increases with meditation
   experience and indicates whole-brain synchrony (Travis & Shear, 2010).

3. Theta/alpha ratio progression -- tracks deepening meditation as theta
   rises and alpha stabilizes across the session.

4. Gamma bursts (40 Hz) -- marker of advanced non-dual practice; 25x higher
   in experienced meditators (Lutz et al., 2004).

5. Nonstationarity index -- stable EEG = deeper meditation; quantified via
   sliding-window variance of band powers.

References:
    Lutz, A. et al. (2004). Long-term meditators self-induce high-amplitude
        gamma synchrony. PNAS 101(46), 16369-16373.
    Lomas, T. et al. (2015). A systematic review of the neurophysiology of
        mindfulness. Neuroscience & Biobehavioral Reviews 57, 401-410.
    Brandmeyer, T. & Delorme, A. (2018). Reduced mind wandering in experienced
        meditators. Experimental Brain Research 236, 2519-2528.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal

# NumPy 2.0 renamed np.trapz -> np.trapezoid
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

# ---- Depth level definitions --------------------------------------------------
DEPTH_LEVELS = ["surface", "light", "moderate", "deep", "absorbed"]

# Score thresholds mapping depth_score (0-100) to levels
_LEVEL_THRESHOLDS = [
    (0, "surface"),
    (20, "light"),
    (40, "moderate"),
    (65, "deep"),
    (85, "absorbed"),
]

# EEG frequency bands (Hz)
_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 50.0),
}

# Gamma burst detection: narrow band around 40 Hz
_GAMMA_BURST_BAND = (38.0, 42.0)


class MeditationDepthQuantifier:
    """Objective meditation depth quantifier from 4-channel Muse 2 EEG.

    Supports multiple independent users via user_id parameter. Each user
    gets their own baseline, session timeline, and history.

    Typical usage:
        q = MeditationDepthQuantifier()
        q.set_baseline(resting_eeg, fs=256)
        result = q.assess(meditation_eeg, fs=256)
        print(result["depth_score"], result["depth_level"])
    """

    def __init__(self, fs: float = 256.0):
        self._fs = fs
        # Per-user state: baseline, timeline, history
        self._baselines: Dict[str, Dict[str, float]] = {}
        self._timelines: Dict[str, List[Dict]] = {}
        self._histories: Dict[str, List[Dict]] = {}
        # Time tracking per depth level per user (seconds)
        self._time_in_depth: Dict[str, Dict[str, float]] = {}
        # Epoch duration estimate (updated from signal length in assess)
        self._epoch_seconds: float = 4.0

    # ---- Baseline -----------------------------------------------------------

    def set_baseline(
        self,
        eeg_signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record resting-state baseline for normalization.

        Args:
            eeg_signals: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate in Hz. Falls back to constructor default.
            user_id: User identifier for multi-user support.

        Returns:
            Dict with baseline band powers and confirmation flag.
        """
        fs = fs or self._fs
        signals = np.asarray(eeg_signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        # Compute per-band baseline power (mean across channels)
        baseline = {}
        for band_name, (low, high) in _BANDS.items():
            powers = []
            for ch in range(signals.shape[0]):
                powers.append(self._band_power(signals[ch], low, high, fs))
            baseline[band_name] = float(np.mean(powers)) if powers else 0.0

        # Alpha coherence baseline (if multichannel)
        if signals.shape[0] >= 2:
            baseline["alpha_coherence"] = self._alpha_coherence(signals, fs)
        else:
            baseline["alpha_coherence"] = 0.0

        self._baselines[user_id] = baseline

        # Initialize time tracking
        if user_id not in self._time_in_depth:
            self._time_in_depth[user_id] = {level: 0.0 for level in DEPTH_LEVELS}

        return {
            "baseline_set": True,
            "baseline_powers": {k: round(v, 6) for k, v in baseline.items()},
        }

    # ---- Assess -------------------------------------------------------------

    def assess(
        self,
        eeg_signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Assess current meditation depth from an EEG epoch.

        Args:
            eeg_signals: (n_channels, n_samples) or (n_samples,) EEG.
            fs: Sampling rate in Hz.
            user_id: User identifier.

        Returns:
            Dict with depth_score (0-100), depth_level, fmt_power,
            alpha_coherence, theta_alpha_ratio, gamma_bursts_detected,
            stability_index, time_in_depth, recommendations.
        """
        fs = fs or self._fs
        signals = np.asarray(eeg_signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        n_samples = signals.shape[1]
        self._epoch_seconds = n_samples / fs

        baseline = self._baselines.get(user_id, {})

        # ---- Compute markers ------------------------------------------------

        # 1. FMT power: average theta at AF7 (ch1) + AF8 (ch2), or all channels
        fmt_power = self._compute_fmt(signals, fs)

        # 2. Alpha coherence (inter-hemispheric)
        alpha_coh = self._alpha_coherence(signals, fs)

        # 3. Band powers (mean across channels)
        band_powers = {}
        for band_name, (low, high) in _BANDS.items():
            ch_powers = []
            for ch in range(signals.shape[0]):
                ch_powers.append(self._band_power(signals[ch], low, high, fs))
            band_powers[band_name] = float(np.mean(ch_powers))

        theta = band_powers.get("theta", 0.0)
        alpha = band_powers.get("alpha", 0.0)
        beta = band_powers.get("beta", 0.0)
        gamma = band_powers.get("gamma", 0.0)

        theta_alpha_ratio = theta / max(alpha, 1e-10)

        # 4. Gamma bursts (40 Hz narrow-band)
        gamma_bursts_detected = self._detect_gamma_bursts(signals, fs, baseline)

        # 5. Stability index: inverse of band-power variance across
        #    short sliding windows within this epoch
        stability_index = self._compute_stability(signals, fs)

        # ---- Compute sub-scores (each 0-1) ----------------------------------

        # FMT sub-score: normalized against baseline theta if available
        base_theta = baseline.get("theta", 0.0)
        if base_theta > 1e-10:
            fmt_ratio = fmt_power / base_theta
            fmt_score = float(np.clip(np.tanh((fmt_ratio - 1.0) * 1.5), 0, 1))
        else:
            # No baseline: use absolute FMT level (relative within band powers)
            fmt_score = float(np.clip(np.tanh(fmt_power * 8), 0, 1))

        # Theta/alpha ratio sub-score: higher = deeper
        tar_score = float(np.clip(np.tanh((theta_alpha_ratio - 0.5) * 1.2), 0, 1))

        # Alpha coherence sub-score
        base_coh = baseline.get("alpha_coherence", 0.0)
        if base_coh > 0.1:
            coh_increase = alpha_coh - base_coh
            coh_score = float(np.clip(0.5 + np.tanh(coh_increase * 5) * 0.5, 0, 1))
        else:
            coh_score = float(np.clip(alpha_coh, 0, 1))

        # Beta quieting sub-score: lower beta relative to baseline = deeper
        base_beta = baseline.get("beta", 0.0)
        if base_beta > 1e-10:
            beta_ratio = beta / base_beta
            beta_quiet_score = float(np.clip(1.0 - np.tanh(beta_ratio * 0.8), 0, 1))
        else:
            beta_quiet_score = float(np.clip(1.0 - np.tanh(beta * 5), 0, 1))

        # Stability sub-score
        stability_score = stability_index  # already 0-1

        # Gamma burst bonus (advanced practice indicator)
        gamma_bonus = 0.1 if gamma_bursts_detected else 0.0

        # ---- Weighted depth score (0-100) ------------------------------------

        raw_score = (
            0.30 * fmt_score
            + 0.25 * tar_score
            + 0.20 * coh_score
            + 0.15 * beta_quiet_score
            + 0.10 * stability_score
        )
        # Add gamma bonus, capped at 1.0
        raw_score = float(np.clip(raw_score + gamma_bonus, 0, 1))
        depth_score = round(raw_score * 100, 1)

        # ---- Map to depth level ---------------------------------------------
        depth_level = self._score_to_level(depth_score)

        # ---- Track time in depth ---------------------------------------------
        if user_id not in self._time_in_depth:
            self._time_in_depth[user_id] = {level: 0.0 for level in DEPTH_LEVELS}
        self._time_in_depth[user_id][depth_level] += self._epoch_seconds

        # ---- Recommendations ------------------------------------------------
        recommendations = self._generate_recommendations(
            depth_level, fmt_score, tar_score, coh_score, beta_quiet_score,
            stability_score, gamma_bursts_detected,
        )

        # ---- Build result ----------------------------------------------------
        result = {
            "depth_score": depth_score,
            "depth_level": depth_level,
            "fmt_power": round(fmt_power, 6),
            "alpha_coherence": round(alpha_coh, 4),
            "theta_alpha_ratio": round(theta_alpha_ratio, 4),
            "gamma_bursts_detected": gamma_bursts_detected,
            "stability_index": round(stability_index, 4),
            "time_in_depth": dict(self._time_in_depth.get(user_id, {})),
            "recommendations": recommendations,
        }

        # ---- Update timeline and history -------------------------------------
        if user_id not in self._timelines:
            self._timelines[user_id] = []
        self._timelines[user_id].append({
            "depth_score": depth_score,
            "depth_level": depth_level,
            "fmt_power": round(fmt_power, 6),
            "alpha_coherence": round(alpha_coh, 4),
            "theta_alpha_ratio": round(theta_alpha_ratio, 4),
        })
        # Cap timeline to ~30 min at 4-sec epochs
        if len(self._timelines[user_id]) > 450:
            self._timelines[user_id] = self._timelines[user_id][-450:]

        if user_id not in self._histories:
            self._histories[user_id] = []
        self._histories[user_id].append(result)
        if len(self._histories[user_id]) > 1000:
            self._histories[user_id] = self._histories[user_id][-1000:]

        return result

    # ---- Session timeline / stats / history ---------------------------------

    def get_session_timeline(self, user_id: str = "default") -> List[Dict]:
        """Return the depth timeline for visualization.

        Each entry contains depth_score, depth_level, fmt_power,
        alpha_coherence, theta_alpha_ratio -- one per assess() call.
        """
        return list(self._timelines.get(user_id, []))

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Return aggregate session statistics.

        Returns:
            Dict with n_assessments, mean_depth, max_depth, deepest_level,
            time_in_depth, and trend.
        """
        timeline = self._timelines.get(user_id, [])
        if not timeline:
            return {
                "n_assessments": 0,
                "has_baseline": user_id in self._baselines,
            }

        scores = [e["depth_score"] for e in timeline]
        levels = [e["depth_level"] for e in timeline]

        # Determine deepest level reached
        max_level_idx = max(DEPTH_LEVELS.index(lv) for lv in levels)
        deepest_level = DEPTH_LEVELS[max_level_idx]

        # Trend: compare first vs second half
        trend = "insufficient_data"
        if len(scores) >= 10:
            mid = len(scores) // 2
            first_half = float(np.mean(scores[:mid]))
            second_half = float(np.mean(scores[mid:]))
            diff = second_half - first_half
            if diff > 3.0:
                trend = "deepening"
            elif diff < -3.0:
                trend = "surfacing"
            else:
                trend = "stable"

        return {
            "n_assessments": len(timeline),
            "mean_depth": round(float(np.mean(scores)), 1),
            "max_depth": round(float(np.max(scores)), 1),
            "deepest_level": deepest_level,
            "time_in_depth": dict(self._time_in_depth.get(user_id, {})),
            "trend": trend,
            "has_baseline": user_id in self._baselines,
        }

    def get_history(
        self,
        user_id: str = "default",
        last_n: Optional[int] = None,
    ) -> List[Dict]:
        """Return assessment history.

        Args:
            user_id: User identifier.
            last_n: If set, return only the last N entries.

        Returns:
            List of assessment result dicts.
        """
        history = self._histories.get(user_id, [])
        if last_n is not None and last_n > 0:
            return list(history[-last_n:])
        return list(history)

    def reset(self, user_id: str = "default"):
        """Clear all state for a user (baseline, timeline, history)."""
        self._baselines.pop(user_id, None)
        self._timelines.pop(user_id, None)
        self._histories.pop(user_id, None)
        self._time_in_depth.pop(user_id, None)

    # ---- Private helpers ----------------------------------------------------

    def _band_power(
        self, signal: np.ndarray, low: float, high: float, fs: float
    ) -> float:
        """Compute band power via Welch PSD."""
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

    def _compute_fmt(self, signals: np.ndarray, fs: float) -> float:
        """Compute Frontal Midline Theta power.

        On Muse 2: average AF7 (ch1) + AF8 (ch2) theta power.
        Falls back to mean of all channels if fewer than 3 channels.
        """
        theta_low, theta_high = _BANDS["theta"]

        if signals.shape[0] >= 3:
            # AF7 = ch1, AF8 = ch2
            af7_theta = self._band_power(signals[1], theta_low, theta_high, fs)
            af8_theta = self._band_power(signals[2], theta_low, theta_high, fs)
            return float((af7_theta + af8_theta) / 2.0)
        else:
            # Average across all available channels
            powers = []
            for ch in range(signals.shape[0]):
                powers.append(
                    self._band_power(signals[ch], theta_low, theta_high, fs)
                )
            return float(np.mean(powers)) if powers else 0.0

    def _alpha_coherence(self, signals: np.ndarray, fs: float) -> float:
        """Compute mean alpha-band coherence across channel pairs."""
        if signals.shape[0] < 2:
            return 0.0

        alpha_low, alpha_high = _BANDS["alpha"]
        nperseg = min(signals.shape[1], int(fs * 2))
        if nperseg < 4:
            return 0.0

        coherence_values = []
        n_ch = signals.shape[0]
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                try:
                    freqs, coh = scipy_signal.coherence(
                        signals[i], signals[j], fs=fs, nperseg=nperseg
                    )
                    mask = (freqs >= alpha_low) & (freqs <= alpha_high)
                    if mask.any():
                        coherence_values.append(float(np.mean(coh[mask])))
                except Exception:
                    continue

        return float(np.mean(coherence_values)) if coherence_values else 0.0

    def _detect_gamma_bursts(
        self, signals: np.ndarray, fs: float, baseline: Dict
    ) -> bool:
        """Detect 40 Hz gamma bursts indicative of advanced meditation.

        Returns True if gamma power in the 38-42 Hz band exceeds baseline
        gamma by at least 2x, or if absolute gamma is high in the absence
        of a baseline.
        """
        gb_low, gb_high = _GAMMA_BURST_BAND
        # Need sufficient sampling rate for 40 Hz
        if fs < 80:
            return False

        gamma_powers = []
        for ch in range(signals.shape[0]):
            gamma_powers.append(self._band_power(signals[ch], gb_low, gb_high, fs))

        mean_gamma = float(np.mean(gamma_powers)) if gamma_powers else 0.0

        base_gamma = baseline.get("gamma", 0.0)
        if base_gamma > 1e-10:
            return mean_gamma > base_gamma * 2.0
        else:
            # No baseline: use absolute threshold
            return mean_gamma > 0.05

    def _compute_stability(self, signals: np.ndarray, fs: float) -> float:
        """Compute nonstationarity index as 1 - normalized variance of
        sliding-window band powers. Higher = more stable = deeper meditation.

        Uses 1-second windows with 50% overlap within the epoch.
        """
        window_samples = int(fs)  # 1-second window
        n_samples = signals.shape[1]

        if n_samples < window_samples * 2:
            # Not enough data for multiple windows; assume moderate stability
            return 0.5

        hop = window_samples // 2
        # Compute theta+alpha power for each window
        window_powers = []
        start = 0
        while start + window_samples <= n_samples:
            segment_powers = []
            for ch in range(signals.shape[0]):
                seg = signals[ch, start: start + window_samples]
                theta_p = self._band_power(seg, 4.0, 8.0, fs)
                alpha_p = self._band_power(seg, 8.0, 12.0, fs)
                segment_powers.append(theta_p + alpha_p)
            window_powers.append(float(np.mean(segment_powers)))
            start += hop

        if len(window_powers) < 2:
            return 0.5

        arr = np.array(window_powers)
        mean_p = float(np.mean(arr))
        if mean_p < 1e-10:
            return 0.5

        cv = float(np.std(arr) / mean_p)  # coefficient of variation
        # Map CV to stability: CV=0 -> stability=1, CV>=1 -> stability~0
        stability = float(np.clip(1.0 - np.tanh(cv * 2), 0, 1))
        return stability

    @staticmethod
    def _score_to_level(score: float) -> str:
        """Map depth score (0-100) to a depth level string."""
        level = "surface"
        for threshold, name in _LEVEL_THRESHOLDS:
            if score >= threshold:
                level = name
        return level

    @staticmethod
    def _generate_recommendations(
        depth_level: str,
        fmt_score: float,
        tar_score: float,
        coh_score: float,
        beta_quiet_score: float,
        stability_score: float,
        gamma_bursts: bool,
    ) -> List[str]:
        """Generate actionable recommendations based on current state."""
        recs = []

        if depth_level == "surface":
            recs.append("Focus on slow, steady breathing to calm the mind.")
            if beta_quiet_score < 0.3:
                recs.append(
                    "High beta detected -- try releasing tension in the jaw and forehead."
                )

        if depth_level in ("surface", "light"):
            if fmt_score < 0.3:
                recs.append(
                    "Frontal theta is low -- try directing attention to a single point of focus."
                )
            if stability_score < 0.4:
                recs.append(
                    "EEG is unstable -- maintain stillness and reduce eye movements."
                )

        if depth_level in ("moderate", "deep"):
            if coh_score < 0.4:
                recs.append(
                    "Alpha coherence is low -- try a whole-body awareness technique."
                )
            if tar_score > 0.8 and beta_quiet_score > 0.6:
                recs.append(
                    "Strong theta with quiet beta -- conditions are favorable for deepening."
                )

        if depth_level == "absorbed":
            recs.append("Excellent depth -- maintain without effort.")
            if gamma_bursts:
                recs.append(
                    "Gamma bursts detected -- characteristic of advanced non-dual awareness."
                )

        if not recs:
            recs.append("Continue current practice.")

        return recs
