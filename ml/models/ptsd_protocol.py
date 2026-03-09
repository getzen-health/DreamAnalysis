"""Neurofeedback protocol for PTSD symptom reduction.

Implements alpha asymmetry normalization training targeting the
right-frontal hyperactivation pattern characteristic of PTSD.
PTSD patients typically show excess right-frontal activation
(negative FAA or reduced left-frontal alpha), along with
high-beta hyperarousal and dissociative theta/alpha shifts.

Protocol:
1. Baseline: record 2-3 min resting state to establish individual FAA
2. Training: reward when FAA moves toward balanced or left-dominant
3. Monitor for hyperarousal (high-beta excess) and dissociation
   (sudden theta dominance / alpha collapse)
4. Provide grounding cues during hyperarousal episodes

Muse 2 channel layout (BrainFlow board_id 38):
    ch0 = TP9  (left temporal)
    ch1 = AF7  (left frontal)   <- FAA left channel
    ch2 = AF8  (right frontal)  <- FAA right channel
    ch3 = TP10 (right temporal)

References:
    van der Kolk et al. (2016) — Neurofeedback for PTSD treatment
    Nicholson et al. (2020) — Alpha asymmetry neurofeedback for PTSD
    Ros et al. (2017) — Neurofeedback augmentation of psychotherapy
    Lanius et al. (2010) — Dissociation and the default mode network

CLINICAL DISCLAIMER: This is a research/educational tool only.
It is NOT a substitute for professional clinical diagnosis or
treatment of PTSD. Always work with a licensed mental health
professional for PTSD treatment.
"""
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal

# FAA threshold: values above this indicate right-frontal excess
# (the PTSD pattern). Values near zero or negative are "normalized."
_FAA_ABNORMAL_THRESHOLD = 0.3

# High-beta / alpha ratio threshold for hyperarousal detection
_HYPERAROUSAL_RATIO_THRESHOLD = 1.5

# Clinical disclaimer included in every response
_CLINICAL_DISCLAIMER = (
    "Clinical disclaimer: This neurofeedback protocol is a research "
    "and educational tool only. It is NOT a substitute for professional "
    "clinical diagnosis or treatment of PTSD. Do not use this as a "
    "standalone intervention. Always work with a licensed mental health "
    "professional for PTSD assessment and treatment."
)


class PTSDProtocol:
    """Alpha asymmetry normalization neurofeedback for PTSD.

    Target: normalize right-frontal alpha asymmetry by training
    FAA toward balanced (near zero) or left-dominant (negative FAA).

    PTSD pattern: excess right-frontal activation -> positive FAA
    (more right alpha = less right activation, so PTSD has *less*
    right alpha -> negative FAA in some formulations; here we use
    FAA = ln(right) - ln(left), so PTSD with *reduced left alpha*
    relative to right shows positive FAA).

    Also monitors:
    - Hyperarousal: high-beta (20-30 Hz) excess at frontal sites
    - Dissociation: sudden theta dominance with alpha collapse
    """

    def __init__(
        self,
        faa_target: float = 0.0,
        hyperarousal_threshold: float = _HYPERAROUSAL_RATIO_THRESHOLD,
        fs: float = 256.0,
    ):
        """
        Args:
            faa_target: Target FAA value. 0.0 = balanced, negative = left-dominant.
            hyperarousal_threshold: High-beta/alpha ratio above this triggers alert.
            fs: Default sampling rate.
        """
        self._faa_target = faa_target
        self._hyperarousal_threshold = hyperarousal_threshold
        self._fs = fs
        self._baselines: Dict[str, Dict] = {}
        self._sessions: Dict[str, List[Dict]] = {}

    # ── Public API ─────────────────────────────────────────────────

    def set_baseline(
        self,
        eeg: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record baseline FAA and band powers from resting state.

        Args:
            eeg: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate (defaults to self._fs).
            user_id: User identifier for multi-user support.

        Returns:
            Dict with baseline_set, faa_baseline, alpha_left, alpha_right,
            high_beta, theta.
        """
        fs = fs or self._fs
        eeg = np.asarray(eeg, dtype=float)
        if eeg.ndim == 1:
            eeg = eeg.reshape(1, -1)

        alpha_left, alpha_right = self._frontal_alpha_pair(eeg, fs)
        faa = self._compute_faa(alpha_left, alpha_right)
        hbeta = self._frontal_band_power(eeg, fs, 20, 30)
        theta = self._frontal_band_power(eeg, fs, 4, 8)

        baseline = {
            "faa": faa,
            "alpha_left": alpha_left,
            "alpha_right": alpha_right,
            "high_beta": hbeta,
            "theta": theta,
        }
        self._baselines[user_id] = baseline

        return {
            "baseline_set": True,
            "faa_baseline": round(faa, 4),
            "alpha_left": round(alpha_left, 6),
            "alpha_right": round(alpha_right, 6),
            "high_beta": round(hbeta, 6),
            "theta": round(theta, 6),
        }

    def evaluate(
        self,
        eeg: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Evaluate a training epoch and provide neurofeedback.

        Args:
            eeg: (n_channels, n_samples) or (n_samples,) EEG epoch.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with:
                faa_current: Current FAA value.
                faa_baseline: Baseline FAA (0.0 if no baseline).
                asymmetry_normalized: Whether FAA is in healthy range.
                hyperarousal_detected: Whether high-beta is excessive.
                dissociation_risk: 0-1 risk score.
                regulation_score: 0-100 composite score.
                feedback_message: Guidance text.
                clinical_disclaimer: Always-present disclaimer.
        """
        fs = fs or self._fs
        eeg = np.asarray(eeg, dtype=float)
        if eeg.ndim == 1:
            eeg = eeg.reshape(1, -1)

        baseline = self._baselines.get(user_id, {})
        bl_faa = baseline.get("faa", 0.0)
        bl_hbeta = baseline.get("high_beta", 0.0)
        bl_theta = baseline.get("theta", 0.0)

        # Current values
        alpha_left, alpha_right = self._frontal_alpha_pair(eeg, fs)
        faa = self._compute_faa(alpha_left, alpha_right)
        hbeta = self._frontal_band_power(eeg, fs, 20, 30)
        theta = self._frontal_band_power(eeg, fs, 4, 8)
        alpha_total = self._frontal_band_power(eeg, fs, 8, 12)

        # Asymmetry assessment
        asymmetry_normalized = faa <= _FAA_ABNORMAL_THRESHOLD

        # Hyperarousal: high-beta / alpha ratio
        ha_result = self._assess_hyperarousal(hbeta, alpha_total, bl_hbeta)
        hyperarousal_detected = ha_result["hyperarousal_detected"]

        # Dissociation: theta/alpha shift
        diss_result = self._assess_dissociation(theta, alpha_total, bl_theta)
        dissociation_risk = diss_result["dissociation_risk"]

        # Regulation score (0-100)
        regulation_score = self._compute_regulation_score(
            faa, hyperarousal_detected, dissociation_risk, bl_faa
        )

        # Feedback message
        feedback_message = self._generate_feedback(
            faa, asymmetry_normalized, hyperarousal_detected, dissociation_risk
        )

        result = {
            "faa_current": round(faa, 4),
            "faa_baseline": round(bl_faa, 4),
            "asymmetry_normalized": asymmetry_normalized,
            "hyperarousal_detected": hyperarousal_detected,
            "hyperarousal_intensity": round(ha_result["hyperarousal_intensity"], 4),
            "dissociation_risk": round(dissociation_risk, 4),
            "regulation_score": round(regulation_score, 2),
            "feedback_message": feedback_message,
            "clinical_disclaimer": _CLINICAL_DISCLAIMER,
            "alpha_left": round(alpha_left, 6),
            "alpha_right": round(alpha_right, 6),
            "high_beta": round(hbeta, 6),
            "theta": round(theta, 6),
            "has_baseline": bool(baseline),
        }

        # Track session
        if user_id not in self._sessions:
            self._sessions[user_id] = []
        self._sessions[user_id].append(result)
        if len(self._sessions[user_id]) > 1000:
            self._sessions[user_id] = self._sessions[user_id][-1000:]

        return result

    def detect_hyperarousal(
        self,
        eeg: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Standalone hyperarousal detection.

        Args:
            eeg: (n_channels, n_samples) or (n_samples,) EEG.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with hyperarousal_detected (bool), hyperarousal_intensity (0-1),
            high_beta, alpha, ratio.
        """
        fs = fs or self._fs
        eeg = np.asarray(eeg, dtype=float)
        if eeg.ndim == 1:
            eeg = eeg.reshape(1, -1)

        baseline = self._baselines.get(user_id, {})
        bl_hbeta = baseline.get("high_beta", 0.0)

        hbeta = self._frontal_band_power(eeg, fs, 20, 30)
        alpha = self._frontal_band_power(eeg, fs, 8, 12)

        result = self._assess_hyperarousal(hbeta, alpha, bl_hbeta)
        result["high_beta"] = round(hbeta, 6)
        result["alpha"] = round(alpha, 6)
        result["clinical_disclaimer"] = _CLINICAL_DISCLAIMER
        return result

    def detect_dissociation(
        self,
        eeg: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Standalone dissociation risk detection.

        Args:
            eeg: (n_channels, n_samples) or (n_samples,) EEG.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with dissociation_risk (0-1), theta, alpha, theta_alpha_ratio.
        """
        fs = fs or self._fs
        eeg = np.asarray(eeg, dtype=float)
        if eeg.ndim == 1:
            eeg = eeg.reshape(1, -1)

        baseline = self._baselines.get(user_id, {})
        bl_theta = baseline.get("theta", 0.0)

        theta = self._frontal_band_power(eeg, fs, 4, 8)
        alpha = self._frontal_band_power(eeg, fs, 8, 12)

        result = self._assess_dissociation(theta, alpha, bl_theta)
        result["theta"] = round(theta, 6)
        result["alpha"] = round(alpha, 6)
        result["clinical_disclaimer"] = _CLINICAL_DISCLAIMER
        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get training session statistics.

        Returns:
            Dict with n_epochs, mean_regulation_score, normalization_rate,
            hyperarousal_count, trend, has_baseline.
        """
        history = self._sessions.get(user_id, [])
        if not history:
            return {
                "n_epochs": 0,
                "has_baseline": user_id in self._baselines,
            }

        regulation_scores = [h["regulation_score"] for h in history]
        normalized_count = sum(1 for h in history if h["asymmetry_normalized"])
        hyperarousal_count = sum(1 for h in history if h["hyperarousal_detected"])

        # Trend computation
        if len(regulation_scores) >= 10:
            first_half = float(np.mean(regulation_scores[: len(regulation_scores) // 2]))
            second_half = float(np.mean(regulation_scores[len(regulation_scores) // 2 :]))
            diff = second_half - first_half
            if diff > 3.0:
                trend = "improving"
            elif diff < -3.0:
                trend = "worsening"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "n_epochs": len(history),
            "mean_regulation_score": round(float(np.mean(regulation_scores)), 2),
            "normalization_rate": round(normalized_count / len(history), 4),
            "hyperarousal_count": hyperarousal_count,
            "trend": trend,
            "has_baseline": user_id in self._baselines,
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Get epoch-by-epoch history.

        Args:
            user_id: User identifier.
            last_n: Return only the last N entries.

        Returns:
            List of evaluation result dicts.
        """
        history = self._sessions.get(user_id, [])
        if last_n:
            history = history[-last_n:]
        return history

    def reset(self, user_id: str = "default"):
        """Clear baseline and session data for a user.

        Args:
            user_id: User identifier.
        """
        self._baselines.pop(user_id, None)
        self._sessions.pop(user_id, None)

    # ── Private helpers ────────────────────────────────────────────

    def _frontal_alpha_pair(
        self, signals: np.ndarray, fs: float
    ) -> tuple:
        """Compute alpha power at AF7 (ch1) and AF8 (ch2) separately.

        Returns:
            (alpha_left, alpha_right) tuple of float.
        """
        if signals.shape[0] >= 3:
            left_ch, right_ch = 1, 2
        else:
            # Fallback: single channel, use same for both
            left_ch, right_ch = 0, 0

        alpha_left = self._channel_band_power(signals[left_ch], fs, 8, 12)
        alpha_right = self._channel_band_power(
            signals[right_ch] if right_ch < signals.shape[0] else signals[0],
            fs, 8, 12,
        )
        return alpha_left, alpha_right

    def _compute_faa(self, alpha_left: float, alpha_right: float) -> float:
        """Compute frontal alpha asymmetry.

        FAA = ln(right_alpha) - ln(left_alpha)
        Positive FAA = more right alpha = less right activation = healthy
        Negative FAA = less right alpha = more right activation = PTSD pattern

        Note: In PTSD literature, excess right-frontal *activation* means
        *less* right-frontal alpha (alpha is inversely related to activation).
        The PTSD pattern shows as right_alpha < left_alpha -> negative FAA.
        However, we also see PTSD patients with abnormally high positive FAA
        due to excessive left-frontal deactivation. The protocol targets
        normalization toward FAA near zero.
        """
        eps = 1e-10
        return float(np.log(alpha_right + eps) - np.log(alpha_left + eps))

    def _frontal_band_power(
        self, signals: np.ndarray, fs: float, low: float, high: float
    ) -> float:
        """Compute mean band power across frontal channels (AF7=ch1, AF8=ch2)."""
        frontal_channels = [1, 2] if signals.shape[0] >= 3 else [0]
        powers = []
        for ch in frontal_channels:
            if ch >= signals.shape[0]:
                continue
            power = self._channel_band_power(signals[ch], fs, low, high)
            if power > 0:
                powers.append(power)
        return float(np.mean(powers)) if powers else 0.0

    def _channel_band_power(
        self, channel: np.ndarray, fs: float, low: float, high: float
    ) -> float:
        """Compute band power for a single channel via Welch PSD."""
        nperseg = min(len(channel), int(fs * 2))
        if nperseg < 4:
            return 0.0

        try:
            freqs, psd = scipy_signal.welch(channel, fs=fs, nperseg=nperseg)
        except Exception:
            return 0.0

        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 0.0

        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(psd[mask], freqs[mask]))
        return float(np.trapz(psd[mask], freqs[mask]))

    def _assess_hyperarousal(
        self, hbeta: float, alpha: float, bl_hbeta: float
    ) -> Dict:
        """Assess hyperarousal from high-beta / alpha ratio.

        Hyperarousal in PTSD: excessive high-beta (20-30 Hz) at frontal sites,
        reflecting sympathetic nervous system overdrive and hypervigilance.

        Returns:
            Dict with hyperarousal_detected and hyperarousal_intensity.
        """
        eps = 1e-10
        ratio = hbeta / (alpha + eps)
        detected = ratio > self._hyperarousal_threshold

        # Intensity: how far above threshold (0-1 scale)
        if ratio <= 1.0:
            intensity = 0.0
        else:
            intensity = float(np.clip((ratio - 1.0) / 2.0, 0, 1))

        return {
            "hyperarousal_detected": bool(detected),
            "hyperarousal_intensity": intensity,
            "hbeta_alpha_ratio": round(ratio, 4),
        }

    def _assess_dissociation(
        self, theta: float, alpha: float, bl_theta: float
    ) -> Dict:
        """Assess dissociation risk from theta/alpha dynamics.

        Dissociation in PTSD (Lanius et al., 2010): sudden shift to
        theta/alpha dominance, reflecting cortical deafferentation and
        emotional numbing. Marked by theta surge with alpha collapse.

        Returns:
            Dict with dissociation_risk (0-1) and theta_alpha_ratio.
        """
        eps = 1e-10
        theta_alpha_ratio = theta / (alpha + eps)

        # Baseline-relative theta change
        if bl_theta > eps:
            theta_change = theta / bl_theta
        else:
            theta_change = 1.0

        # Dissociation risk: combination of theta/alpha ratio and theta surge
        # High theta/alpha ratio + sudden theta increase -> higher risk
        ratio_component = float(np.clip((theta_alpha_ratio - 1.0) / 2.0, 0, 1))
        surge_component = float(np.clip((theta_change - 1.5) / 2.0, 0, 1))

        risk = float(np.clip(0.6 * ratio_component + 0.4 * surge_component, 0, 1))

        return {
            "dissociation_risk": risk,
            "theta_alpha_ratio": round(theta_alpha_ratio, 4),
        }

    def _compute_regulation_score(
        self,
        faa: float,
        hyperarousal: bool,
        dissociation_risk: float,
        bl_faa: float,
    ) -> float:
        """Compute composite regulation score (0-100).

        Components:
        - Asymmetry normalization (40%): FAA near target
        - Arousal regulation (30%): no hyperarousal
        - Dissociation safety (30%): low dissociation risk

        Higher = better self-regulation.
        """
        # Asymmetry component (40 points): how close FAA is to target
        faa_distance = abs(faa - self._faa_target)
        asymmetry_score = float(np.clip(1.0 - faa_distance / 1.0, 0, 1)) * 40

        # Arousal component (30 points)
        if hyperarousal:
            arousal_score = 0.0
        else:
            arousal_score = 30.0

        # Dissociation component (30 points)
        dissociation_score = (1.0 - dissociation_risk) * 30

        total = asymmetry_score + arousal_score + dissociation_score
        return float(np.clip(total, 0, 100))

    def _generate_feedback(
        self,
        faa: float,
        asymmetry_normalized: bool,
        hyperarousal: bool,
        dissociation_risk: float,
    ) -> str:
        """Generate feedback message based on current state.

        Priority order: hyperarousal > dissociation > asymmetry training.
        """
        if hyperarousal:
            return (
                "Grounding: breathe slowly in for 4 counts, out for 6 counts. "
                "Feel your feet on the ground. You are safe in this moment."
            )

        if dissociation_risk > 0.6:
            return (
                "Gentle re-orientation: notice 5 things you can see, "
                "4 you can touch, 3 you can hear. Stay present."
            )

        if dissociation_risk > 0.3:
            return (
                "Stay engaged: focus on your breath, feel the chair "
                "supporting you. Maintain gentle awareness."
            )

        if asymmetry_normalized:
            return (
                "Good regulation: your frontal activity is balanced. "
                "Continue this calm, centered state."
            )

        if faa > _FAA_ABNORMAL_THRESHOLD:
            return (
                "Shift attention: try gentle left-hand movements or "
                "focus on a calming memory. This activates left-frontal regions."
            )

        return (
            "Maintain: your brain activity is moving toward balance. "
            "Continue with relaxed, steady breathing."
        )
