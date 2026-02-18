"""Emotional Shift Detector — Pre-conscious emotion change awareness.

Animals sense human emotional shifts before we consciously notice them.
They read physiological signals — heart rate, skin conductance, posture,
scent — that change seconds before conscious awareness.

EEG does the same. The brain's electrical signature shifts BEFORE you
realize your mood is changing:
- Frontal alpha asymmetry tilts toward withdrawal before you feel sad
- Beta spikes in temporal regions before anxiety becomes conscious
- Theta/alpha ratio shifts before you notice relaxation setting in
- Gamma coherence drops before focus breaks

This module watches for those pre-conscious signatures and alerts:
"Your emotional state is shifting — pay attention to what you're feeling."

The goal: give humans the same emotional awareness that animals have.
"""

import time
import numpy as np
from typing import Dict, List, Optional
from collections import deque

from processing.eeg_processor import extract_band_powers, preprocess, spectral_entropy


# How quickly each EEG marker typically changes before a conscious emotion shift.
# Based on neuroscience literature: EEG changes precede self-reported
# emotion shifts by 2-8 seconds on average.
PRECURSOR_WINDOW_SEC = 8.0

# Minimum change magnitude to count as a real shift (not noise)
SHIFT_THRESHOLDS = {
    "valence": 0.15,         # -1 to 1 scale — 0.15 is noticeable
    "arousal": 0.12,         # 0 to 1 scale
    "alpha_asymmetry": 0.10, # frontal alpha asymmetry
    "stress_index": 12.0,    # 0-100 scale
    "calm_ratio": 0.3,       # (alpha+theta)/beta
}

# EEG precursors for specific emotional transitions
# Each maps (from_direction) -> list of EEG signatures to watch
EMOTION_PRECURSORS = {
    "approaching_anxiety": {
        "description": "Rising tension — beta increasing, alpha dropping",
        "eeg_signs": ["beta_rising", "alpha_dropping", "gamma_rising"],
        "body_feeling": "You may notice tightness in chest, shallow breathing",
        "guidance": "Take 3 slow breaths. Ground yourself in the present moment.",
    },
    "approaching_sadness": {
        "description": "Withdrawal pattern — right frontal activation increasing",
        "eeg_signs": ["valence_dropping", "arousal_dropping", "theta_rising"],
        "body_feeling": "You may notice heaviness, slower thoughts",
        "guidance": "Notice this feeling without judgment. It's information, not identity.",
    },
    "approaching_calm": {
        "description": "Settling pattern — alpha rising, beta decreasing",
        "eeg_signs": ["alpha_rising", "beta_dropping", "calm_ratio_rising"],
        "body_feeling": "You may notice your shoulders relaxing, breathing deepening",
        "guidance": "Beautiful — your nervous system is settling. Let it happen.",
    },
    "approaching_focus": {
        "description": "Engagement pattern — beta structured, theta dropping",
        "eeg_signs": ["beta_structured", "theta_dropping", "entropy_dropping"],
        "body_feeling": "You may notice heightened attention, narrowing awareness",
        "guidance": "You're entering a focused state. Channel it toward what matters.",
    },
    "approaching_joy": {
        "description": "Approach pattern — left frontal activation, gamma bursts",
        "eeg_signs": ["valence_rising", "arousal_moderate", "gamma_bursts"],
        "body_feeling": "You may notice lightness, warmth, or energy rising",
        "guidance": "Savor this. Consciously noting positive states strengthens them.",
    },
    "emotional_turbulence": {
        "description": "Rapid fluctuation — emotional state is unstable",
        "eeg_signs": ["high_variability", "rapid_band_shifts"],
        "body_feeling": "You may feel unsettled, reactive, or overwhelmed",
        "guidance": "Your system is processing something. Pause. Breathe. Give yourself space.",
    },
}


class EmotionShiftDetector:
    """Detects pre-conscious emotional shifts from EEG time series.

    Maintains a sliding window of emotional indicators and watches for
    rate-of-change patterns that precede conscious emotion transitions.

    Usage:
        detector = EmotionShiftDetector(fs=256)

        # Feed it EEG data continuously (every ~0.25-1 sec):
        result = detector.update(eeg_chunk)

        if result["shift_detected"]:
            print(f"Alert: {result['shift_type']} — {result['description']}")
    """

    def __init__(self, fs: float = 256.0, window_seconds: float = 30.0):
        self.fs = fs
        self.window_seconds = window_seconds

        # Rolling history of emotional indicators
        self._max_history = int(window_seconds * 4)  # at ~4Hz update rate
        self._timestamps = deque(maxlen=self._max_history)
        self._valence_history = deque(maxlen=self._max_history)
        self._arousal_history = deque(maxlen=self._max_history)
        self._band_history = deque(maxlen=self._max_history)
        self._entropy_history = deque(maxlen=self._max_history)
        self._stress_history = deque(maxlen=self._max_history)
        self._calm_ratio_history = deque(maxlen=self._max_history)

        # Shift detection state
        self._last_shift_time = 0.0
        self._cooldown_sec = 10.0  # Don't alert more than once per 10s
        self._previous_emotion = "unknown"
        self._current_emotion = "unknown"
        self._shift_count = 0

        # Accumulated shift log for session review
        self.shift_log: List[Dict] = []

    def update(
        self,
        eeg: np.ndarray,
        emotion_prediction: Optional[Dict] = None,
    ) -> Dict:
        """Process new EEG data and check for emotional shifts.

        Args:
            eeg: Raw EEG signal chunk (1D array, typically 0.25-1 sec).
            emotion_prediction: Optional current emotion model output
                (with 'emotion', 'valence', 'arousal', etc.)

        Returns:
            Dict with shift detection results.
        """
        now = time.time()
        processed = preprocess(eeg, self.fs)
        bands = extract_band_powers(processed, self.fs)
        se = spectral_entropy(processed, self.fs)

        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        theta = bands.get("theta", 0)
        gamma = bands.get("gamma", 0)
        delta = bands.get("delta", 0)

        # Compute emotional indicators from EEG
        valence = float(np.tanh((alpha - beta) * 2 + (theta - gamma) * 0.5))
        arousal = float(np.clip(beta + gamma, 0, 1))
        stress = float(np.clip(beta / max(alpha, 1e-10) * 25, 0, 100))
        calm_ratio = (alpha + theta) / max(beta, 1e-10)

        # Use model predictions if available (more accurate)
        if emotion_prediction:
            valence = emotion_prediction.get("valence", valence)
            arousal = emotion_prediction.get("arousal", arousal)
            stress = emotion_prediction.get("stress_index", stress)
            new_emotion = emotion_prediction.get("emotion", self._current_emotion)
            if new_emotion != self._current_emotion:
                self._previous_emotion = self._current_emotion
            self._current_emotion = new_emotion

        # Store in history
        self._timestamps.append(now)
        self._valence_history.append(valence)
        self._arousal_history.append(arousal)
        self._band_history.append(bands)
        self._entropy_history.append(se)
        self._stress_history.append(stress)
        self._calm_ratio_history.append(calm_ratio)

        # Need at least 3 seconds of data to detect shifts
        if len(self._timestamps) < 12:
            return {
                "shift_detected": False,
                "buffering": True,
                "samples_collected": len(self._timestamps),
                "samples_needed": 12,
                "current_indicators": {
                    "valence": round(valence, 3),
                    "arousal": round(arousal, 3),
                    "stress_index": round(stress, 1),
                    "calm_ratio": round(calm_ratio, 3),
                },
            }

        # Analyze rate of change over recent window
        shift_analysis = self._analyze_shifts()

        # Check cooldown
        in_cooldown = (now - self._last_shift_time) < self._cooldown_sec
        shift_detected = shift_analysis["shift_detected"] and not in_cooldown

        if shift_detected:
            self._last_shift_time = now
            self._shift_count += 1

            # Log the shift
            shift_entry = {
                "timestamp": now,
                "shift_type": shift_analysis["shift_type"],
                "description": shift_analysis["description"],
                "magnitude": shift_analysis["magnitude"],
                "previous_emotion": self._previous_emotion,
                "current_emotion": self._current_emotion,
                "reason": shift_analysis.get("reason", ""),
                "indicators": shift_analysis["indicators"],
            }
            self.shift_log.append(shift_entry)
            # Keep last 100 shifts
            if len(self.shift_log) > 100:
                self.shift_log = self.shift_log[-100:]

        return {
            "shift_detected": shift_detected,
            "shift_type": shift_analysis.get("shift_type"),
            "description": shift_analysis.get("description"),
            "body_feeling": shift_analysis.get("body_feeling"),
            "guidance": shift_analysis.get("guidance"),
            "reason": shift_analysis.get("reason", ""),
            "magnitude": shift_analysis.get("magnitude", 0),
            "confidence": shift_analysis.get("confidence", 0),
            "previous_emotion": self._previous_emotion,
            "current_emotion": self._current_emotion,
            "indicators": {
                "valence": round(valence, 3),
                "arousal": round(arousal, 3),
                "stress_index": round(stress, 1),
                "calm_ratio": round(calm_ratio, 3),
                "spectral_entropy": round(se, 3),
            },
            "trends": shift_analysis.get("trends", {}),
            "total_shifts_detected": self._shift_count,
            "in_cooldown": in_cooldown,
        }

    def _analyze_shifts(self) -> Dict:
        """Analyze recent history for emotional shift patterns."""
        n = len(self._valence_history)
        if n < 8:
            return {"shift_detected": False}

        # Split into "recent" (last 2 sec) and "baseline" (2-8 sec ago)
        recent_n = min(8, n // 3)
        baseline_n = min(n - recent_n, 24)

        recent_valence = list(self._valence_history)[-recent_n:]
        baseline_valence = list(self._valence_history)[-(recent_n + baseline_n):-recent_n]

        recent_arousal = list(self._arousal_history)[-recent_n:]
        baseline_arousal = list(self._arousal_history)[-(recent_n + baseline_n):-recent_n]

        recent_stress = list(self._stress_history)[-recent_n:]
        baseline_stress = list(self._stress_history)[-(recent_n + baseline_n):-recent_n]

        recent_calm = list(self._calm_ratio_history)[-recent_n:]
        baseline_calm = list(self._calm_ratio_history)[-(recent_n + baseline_n):-recent_n]

        recent_entropy = list(self._entropy_history)[-recent_n:]
        baseline_entropy = list(self._entropy_history)[-(recent_n + baseline_n):-recent_n]

        if not baseline_valence:
            return {"shift_detected": False}

        # Compute deltas
        d_valence = np.mean(recent_valence) - np.mean(baseline_valence)
        d_arousal = np.mean(recent_arousal) - np.mean(baseline_arousal)
        d_stress = np.mean(recent_stress) - np.mean(baseline_stress)
        d_calm = np.mean(recent_calm) - np.mean(baseline_calm)
        d_entropy = np.mean(recent_entropy) - np.mean(baseline_entropy)

        # Compute variability (turbulence indicator)
        valence_var = np.std(recent_valence)
        arousal_var = np.std(recent_arousal)

        # Recent band trends
        recent_bands = list(self._band_history)[-recent_n:]
        baseline_bands = list(self._band_history)[-(recent_n + baseline_n):-recent_n]

        band_deltas = {}
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            r = np.mean([b.get(band, 0) for b in recent_bands])
            b = np.mean([b.get(band, 0) for b in baseline_bands]) if baseline_bands else r
            band_deltas[band] = r - b

        trends = {
            "valence_delta": round(d_valence, 4),
            "arousal_delta": round(d_arousal, 4),
            "stress_delta": round(d_stress, 2),
            "calm_ratio_delta": round(d_calm, 4),
            "entropy_delta": round(d_entropy, 4),
            "valence_variability": round(valence_var, 4),
            "band_deltas": {k: round(v, 4) for k, v in band_deltas.items()},
        }

        # --- Pattern matching for specific emotional shifts ---

        prev = self._previous_emotion
        curr = self._current_emotion

        # Approaching anxiety: beta up, alpha down, stress rising
        if (d_stress > SHIFT_THRESHOLDS["stress_index"]
                and band_deltas.get("beta", 0) > 0.02
                and band_deltas.get("alpha", 0) < -0.01):
            info = EMOTION_PRECURSORS["approaching_anxiety"]
            magnitude = abs(d_stress) / 100 + abs(band_deltas.get("beta", 0))
            reason = f"Beta power increased by {abs(band_deltas.get('beta', 0)):.0%} while alpha dropped {abs(band_deltas.get('alpha', 0)):.0%} — your brain shifted from {prev} to {curr}"
            return {
                "shift_detected": True,
                "shift_type": "approaching_anxiety",
                "magnitude": round(min(1.0, magnitude), 3),
                "confidence": round(min(1.0, abs(d_stress) / 30), 3),
                "reason": reason,
                "trends": trends,
                "indicators": {"stress_delta": d_stress, "beta_delta": band_deltas.get("beta", 0)},
                **info,
            }

        # Approaching sadness: valence dropping, arousal dropping, theta rising
        if (d_valence < -SHIFT_THRESHOLDS["valence"]
                and d_arousal < -0.05
                and band_deltas.get("theta", 0) > 0.01):
            info = EMOTION_PRECURSORS["approaching_sadness"]
            magnitude = abs(d_valence) + abs(d_arousal)
            reason = f"Theta power rose by {abs(band_deltas.get('theta', 0)):.0%} as arousal dropped {abs(d_arousal):.0%} — your brain shifted from {prev} to {curr}"
            return {
                "shift_detected": True,
                "shift_type": "approaching_sadness",
                "magnitude": round(min(1.0, magnitude), 3),
                "confidence": round(min(1.0, abs(d_valence) / 0.4), 3),
                "reason": reason,
                "trends": trends,
                "indicators": {"valence_delta": d_valence, "arousal_delta": d_arousal},
                **info,
            }

        # Approaching calm: alpha rising, beta dropping, calm ratio up
        if (d_calm > SHIFT_THRESHOLDS["calm_ratio"]
                and band_deltas.get("alpha", 0) > 0.02
                and band_deltas.get("beta", 0) < -0.01):
            info = EMOTION_PRECURSORS["approaching_calm"]
            magnitude = abs(d_calm) / 2 + abs(band_deltas.get("alpha", 0))
            reason = f"Alpha power increased by {abs(band_deltas.get('alpha', 0)):.0%} while beta dropped {abs(band_deltas.get('beta', 0)):.0%} — your brain shifted from {prev} to {curr}"
            return {
                "shift_detected": True,
                "shift_type": "approaching_calm",
                "magnitude": round(min(1.0, magnitude), 3),
                "confidence": round(min(1.0, d_calm / 0.8), 3),
                "reason": reason,
                "trends": trends,
                "indicators": {"calm_ratio_delta": d_calm, "alpha_delta": band_deltas.get("alpha", 0)},
                **info,
            }

        # Approaching focus: entropy dropping, beta structured, theta down
        if (d_entropy < -0.05
                and band_deltas.get("beta", 0) > 0.01
                and band_deltas.get("theta", 0) < -0.01):
            info = EMOTION_PRECURSORS["approaching_focus"]
            magnitude = abs(d_entropy) + abs(band_deltas.get("beta", 0))
            reason = f"Beta structured up {abs(band_deltas.get('beta', 0)):.0%} and theta dropped {abs(band_deltas.get('theta', 0)):.0%} — your brain shifted from {prev} to {curr}"
            return {
                "shift_detected": True,
                "shift_type": "approaching_focus",
                "magnitude": round(min(1.0, magnitude), 3),
                "confidence": round(min(1.0, abs(d_entropy) / 0.15), 3),
                "reason": reason,
                "trends": trends,
                "indicators": {"entropy_delta": d_entropy, "beta_delta": band_deltas.get("beta", 0)},
                **info,
            }

        # Approaching joy: valence rising, moderate arousal, gamma activity
        if (d_valence > SHIFT_THRESHOLDS["valence"]
                and band_deltas.get("gamma", 0) > 0.005):
            info = EMOTION_PRECURSORS["approaching_joy"]
            magnitude = abs(d_valence) + abs(band_deltas.get("gamma", 0))
            reason = f"Gamma bursts up {abs(band_deltas.get('gamma', 0)):.0%} with valence rising {abs(d_valence):.0%} — your brain shifted from {prev} to {curr}"
            return {
                "shift_detected": True,
                "shift_type": "approaching_joy",
                "magnitude": round(min(1.0, magnitude), 3),
                "confidence": round(min(1.0, d_valence / 0.3), 3),
                "reason": reason,
                "trends": trends,
                "indicators": {"valence_delta": d_valence, "gamma_delta": band_deltas.get("gamma", 0)},
                **info,
            }

        # Emotional turbulence: high variability in valence and arousal
        if valence_var > 0.15 and arousal_var > 0.08:
            info = EMOTION_PRECURSORS["emotional_turbulence"]
            magnitude = valence_var + arousal_var
            reason = f"Rapid band fluctuations — valence variability {valence_var:.0%}, arousal variability {arousal_var:.0%} — shifting between {prev} and {curr}"
            return {
                "shift_detected": True,
                "shift_type": "emotional_turbulence",
                "magnitude": round(min(1.0, magnitude), 3),
                "confidence": round(min(1.0, valence_var / 0.3), 3),
                "reason": reason,
                "trends": trends,
                "indicators": {"valence_variability": valence_var, "arousal_variability": arousal_var},
                **info,
            }

        # General shift: significant valence or arousal change that doesn't
        # match a specific pattern
        if abs(d_valence) > SHIFT_THRESHOLDS["valence"] or abs(d_arousal) > SHIFT_THRESHOLDS["arousal"]:
            direction = "positive" if d_valence > 0 else "negative"
            energy = "energizing" if d_arousal > 0 else "calming"
            reason = f"Valence changed by {abs(d_valence):.0%} and arousal by {abs(d_arousal):.0%} — your brain shifted from {prev} to {curr}"
            return {
                "shift_detected": True,
                "shift_type": "general_shift",
                "description": f"Emotional state shifting — {direction} valence, {energy} arousal",
                "body_feeling": "Something is changing in your emotional landscape",
                "guidance": "Pause and check in with yourself. What are you feeling right now?",
                "reason": reason,
                "magnitude": round(min(1.0, abs(d_valence) + abs(d_arousal)), 3),
                "confidence": round(min(1.0, max(abs(d_valence), abs(d_arousal)) / 0.3), 3),
                "trends": trends,
                "indicators": {"valence_delta": d_valence, "arousal_delta": d_arousal},
            }

        return {"shift_detected": False, "trends": trends}

    def get_session_summary(self) -> Dict:
        """Get summary of all emotional shifts detected in this session."""
        if not self.shift_log:
            return {
                "total_shifts": 0,
                "session_stability": "stable",
                "dominant_patterns": [],
                "insights": ["No significant emotional shifts detected yet."],
            }

        # Count shift types
        type_counts = {}
        for entry in self.shift_log:
            st = entry.get("shift_type", "unknown")
            type_counts[st] = type_counts.get(st, 0) + 1

        dominant = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)

        # Compute session duration
        if len(self.shift_log) >= 2:
            duration = self.shift_log[-1]["timestamp"] - self.shift_log[0]["timestamp"]
            shifts_per_min = len(self.shift_log) / max(duration / 60, 0.5)
        else:
            duration = 0
            shifts_per_min = 0

        # Stability assessment
        if shifts_per_min > 3:
            stability = "turbulent"
            stability_insight = "Your emotional state has been highly dynamic. This is common during processing or stressful periods."
        elif shifts_per_min > 1:
            stability = "active"
            stability_insight = "Moderate emotional movement — your system is actively processing."
        elif shifts_per_min > 0.3:
            stability = "gentle"
            stability_insight = "Gentle emotional flow — natural, healthy variation."
        else:
            stability = "stable"
            stability_insight = "Very stable emotional state throughout the session."

        # Generate insights
        insights = [stability_insight]

        if type_counts.get("approaching_anxiety", 0) >= 2:
            insights.append(
                f"Anxiety patterns appeared {type_counts['approaching_anxiety']} times. "
                "Consider what situations triggered these. Regular breathwork can reduce frequency."
            )

        if type_counts.get("approaching_calm", 0) >= 2:
            insights.append(
                f"Your system moved toward calm {type_counts['approaching_calm']} times. "
                "Your body knows how to settle — trust it."
            )

        if type_counts.get("emotional_turbulence", 0) >= 2:
            insights.append(
                "Multiple turbulence episodes suggest emotional processing. "
                "This is healthy — your brain is working through something."
            )

        # Average magnitude
        magnitudes = [e.get("magnitude", 0) for e in self.shift_log]
        avg_magnitude = np.mean(magnitudes) if magnitudes else 0

        return {
            "total_shifts": len(self.shift_log),
            "shifts_per_minute": round(shifts_per_min, 2),
            "session_stability": stability,
            "average_magnitude": round(avg_magnitude, 3),
            "dominant_patterns": [
                {"type": t, "count": c, "description": EMOTION_PRECURSORS.get(t, {}).get("description", t)}
                for t, c in dominant[:3]
            ],
            "shift_timeline": [
                {
                    "time_offset_sec": round(e["timestamp"] - self.shift_log[0]["timestamp"], 1),
                    "type": e["shift_type"],
                    "magnitude": e.get("magnitude", 0),
                    "emotion_at_time": e.get("current_emotion", "unknown"),
                }
                for e in self.shift_log
            ],
            "insights": insights,
        }

    def get_emotional_awareness_score(self) -> Dict:
        """Compute an emotional awareness score for the session.

        The idea: the more shifts you observe consciously (by using
        this tool), the more emotionally aware you become over time.
        Like training a muscle.
        """
        n_shifts = len(self.shift_log)

        if n_shifts == 0:
            return {
                "awareness_score": 0,
                "level": "Beginning",
                "message": "Start a session to begin building emotional awareness.",
            }

        # Variety of emotions detected
        unique_types = len(set(e.get("shift_type") for e in self.shift_log))

        # Score based on observation (more shifts noticed = more awareness)
        # Real awareness comes from the user paying attention to alerts
        raw_score = min(100, n_shifts * 8 + unique_types * 15)

        if raw_score >= 80:
            level = "Deep Awareness"
            message = "You're developing the emotional sensitivity of a seasoned meditator."
        elif raw_score >= 60:
            level = "Growing Awareness"
            message = "You're building real-time emotional intelligence. Keep observing."
        elif raw_score >= 30:
            level = "Awakening"
            message = "You're starting to notice what was previously invisible."
        else:
            level = "Beginning"
            message = "Every journey starts here. Each noticed shift builds awareness."

        return {
            "awareness_score": raw_score,
            "level": level,
            "message": message,
            "shifts_observed": n_shifts,
            "emotion_types_encountered": unique_types,
        }
