"""Brain State Transition Engine — Temporal smoothing for EEG predictions.

Problem: Raw per-epoch predictions are noisy. A person doesn't flip from
"deep flow" to "no flow" and back every 4 seconds. Real brain states have
inertia — transitions happen gradually and follow physiological rules.

Solution: Markov-style transition model that:
1. Smooths predictions using transition probabilities
2. Blocks impossible transitions (e.g., deep_sleep → flow)
3. Requires minimum dwell time before accepting a state change
4. Uses exponential moving average to prevent flip-flopping

This sits between the raw model output and what we show the user.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple


# ─── Transition matrices ────────────────────────────────────────────
# Each matrix[i][j] = probability of transitioning FROM state i TO state j.
# Rows must sum to 1.0. Low off-diagonal = hard to leave current state.

SLEEP_STATES = ["Wake", "N1", "N2", "N3", "REM"]
SLEEP_TRANSITIONS = np.array([
    # Wake   N1    N2    N3    REM
    [0.85, 0.10, 0.03, 0.00, 0.02],  # Wake → mostly stays Wake
    [0.15, 0.60, 0.20, 0.02, 0.03],  # N1 → can go Wake, N2, or stay
    [0.05, 0.10, 0.65, 0.15, 0.05],  # N2 → mostly stays, can deepen
    [0.01, 0.02, 0.15, 0.80, 0.02],  # N3 → hard to leave deep sleep
    [0.10, 0.05, 0.05, 0.00, 0.80],  # REM → stays or goes to Wake/light
])

FLOW_STATES = ["no_flow", "micro_flow", "flow", "deep_flow"]
FLOW_TRANSITIONS = np.array([
    # no_flow  micro   flow   deep
    [0.75, 0.20, 0.04, 0.01],  # no_flow → gradual entry
    [0.15, 0.60, 0.22, 0.03],  # micro → can go either way
    [0.03, 0.12, 0.70, 0.15],  # flow → tends to stay or deepen
    [0.02, 0.05, 0.18, 0.75],  # deep_flow → hard to leave
])

CREATIVITY_STATES = ["analytical", "transitional", "creative", "insight"]
CREATIVITY_TRANSITIONS = np.array([
    # analyt  trans  creat  insight
    [0.70, 0.25, 0.04, 0.01],  # analytical → gradual shift
    [0.15, 0.55, 0.25, 0.05],  # transitional → either direction
    [0.05, 0.15, 0.65, 0.15],  # creative → tends to stay
    [0.10, 0.10, 0.30, 0.50],  # insight → fleeting, often drops
])

MEMORY_STATES = ["poor_encoding", "weak_encoding", "active_encoding", "deep_encoding"]
MEMORY_TRANSITIONS = np.array([
    # poor    weak   active  deep
    [0.70, 0.25, 0.04, 0.01],
    [0.15, 0.60, 0.22, 0.03],
    [0.03, 0.12, 0.70, 0.15],
    [0.02, 0.08, 0.20, 0.70],
])

EMOTION_STATES = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]
EMOTION_TRANSITIONS = np.array([
    # happy  sad    angry  fear   relax  focus
    [0.65, 0.05, 0.03, 0.02, 0.15, 0.10],  # happy
    [0.05, 0.65, 0.10, 0.08, 0.07, 0.05],  # sad
    [0.05, 0.08, 0.60, 0.10, 0.07, 0.10],  # angry
    [0.03, 0.10, 0.10, 0.62, 0.10, 0.05],  # fearful
    [0.12, 0.05, 0.03, 0.03, 0.65, 0.12],  # relaxed
    [0.08, 0.04, 0.05, 0.03, 0.12, 0.68],  # focused
])


# ─── Impossible transitions (blocked) ──────────────────────────────
# These can NEVER happen in a single step, regardless of model output.

BLOCKED_TRANSITIONS = {
    "sleep": {
        ("Wake", "N3"),     # Can't jump to deep sleep
        ("N3", "Wake"),     # Can't jump from deep sleep to awake
        ("N3", "REM"),      # N3 → REM doesn't happen directly
        ("REM", "N3"),      # REM → N3 doesn't happen directly
    },
    "flow": {
        ("no_flow", "deep_flow"),   # Can't jump to deep flow
        ("deep_flow", "no_flow"),   # Can't crash out of deep flow instantly
    },
    "creativity": {
        ("analytical", "insight"),  # Can't jump to insight
    },
    "memory": {
        ("poor_encoding", "deep_encoding"),
        ("deep_encoding", "poor_encoding"),
    },
}

# Minimum dwell time in seconds before accepting a state change.
# Prevents rapid flip-flopping between states.
MIN_DWELL_SECONDS = {
    "sleep": 30,       # Sleep stages last minutes, not seconds
    "flow": 10,        # Flow needs some stability
    "creativity": 8,   # Creativity states shift more readily
    "memory": 8,
    "emotion": 6,      # Emotions can shift faster
}


class StateTracker:
    """Tracks one model's state with temporal smoothing."""

    def __init__(self, model_name: str, states: List[str],
                 transition_matrix: np.ndarray, min_dwell: float = 10.0):
        self.model_name = model_name
        self.states = states
        self.n_states = len(states)
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        self.transition_matrix = transition_matrix
        self.min_dwell = min_dwell

        # Current state
        self.current_state = states[0]
        self.current_idx = 0
        self.state_entered_at = time.time()

        # Smoothed probability distribution
        self.smoothed_probs = np.ones(self.n_states) / self.n_states
        self.ema_alpha = 0.3  # Exponential moving average weight (0=full smooth, 1=no smooth)

        # History for analysis
        self.history: List[Tuple[float, str, float]] = []  # (timestamp, state, confidence)

    def update(self, raw_state: str, raw_confidence: float = 1.0,
               raw_probs: Optional[np.ndarray] = None) -> Dict:
        """Update state given new model prediction.

        Args:
            raw_state: Model's raw predicted state.
            raw_confidence: Model's confidence in prediction.
            raw_probs: Optional full probability distribution over states.

        Returns:
            Dict with smoothed_state, smoothed_confidence, was_overridden.
        """
        now = time.time()
        raw_idx = self.state_to_idx.get(raw_state, 0)

        # Build raw probability distribution
        # Handle dict-style probabilities (e.g., {"Wake": 0.5, "N1": 0.2})
        if isinstance(raw_probs, dict):
            raw_probs = np.array(list(raw_probs.values()), dtype=float)
        elif isinstance(raw_probs, list):
            raw_probs = np.array(raw_probs, dtype=float)

        if raw_probs is not None and len(raw_probs) == self.n_states:
            raw_dist = np.array(raw_probs, dtype=float)
        else:
            # Create distribution from single prediction + confidence
            raw_dist = np.full(self.n_states, (1 - raw_confidence) / max(1, self.n_states - 1))
            raw_dist[raw_idx] = raw_confidence

        # Normalize
        raw_dist = np.maximum(raw_dist, 0)
        if raw_dist.sum() > 0:
            raw_dist /= raw_dist.sum()
        else:
            raw_dist = np.ones(self.n_states) / self.n_states

        # Apply transition probability weighting
        # Prior from current state's transition probabilities
        transition_prior = self.transition_matrix[self.current_idx]
        weighted = raw_dist * transition_prior
        if weighted.sum() > 0:
            weighted /= weighted.sum()

        # Exponential moving average with previous smoothed distribution
        self.smoothed_probs = self.ema_alpha * weighted + (1 - self.ema_alpha) * self.smoothed_probs
        self.smoothed_probs /= self.smoothed_probs.sum()

        # Pick the most likely state from smoothed distribution
        candidate_idx = int(np.argmax(self.smoothed_probs))
        candidate_state = self.states[candidate_idx]
        candidate_confidence = float(self.smoothed_probs[candidate_idx])

        # Check if transition is blocked
        was_blocked = False
        blocked_set = BLOCKED_TRANSITIONS.get(self.model_name, set())
        if (self.current_state, candidate_state) in blocked_set:
            was_blocked = True
            candidate_state = self.current_state
            candidate_idx = self.current_idx
            candidate_confidence = float(self.smoothed_probs[candidate_idx])

        # Check minimum dwell time
        was_held = False
        time_in_state = now - self.state_entered_at
        if candidate_state != self.current_state and time_in_state < self.min_dwell:
            was_held = True
            candidate_state = self.current_state
            candidate_idx = self.current_idx
            candidate_confidence = float(self.smoothed_probs[candidate_idx])

        # Accept the state
        was_overridden = (candidate_state != raw_state)
        if candidate_state != self.current_state:
            self.state_entered_at = now
            self.current_state = candidate_state
            self.current_idx = candidate_idx

        # Record history
        self.history.append((now, candidate_state, candidate_confidence))
        # Keep last 500 entries
        if len(self.history) > 500:
            self.history = self.history[-500:]

        return {
            "smoothed_state": candidate_state,
            "smoothed_confidence": round(candidate_confidence, 3),
            "raw_state": raw_state,
            "raw_confidence": round(raw_confidence, 3),
            "was_overridden": was_overridden,
            "override_reason": (
                "blocked_transition" if was_blocked
                else "min_dwell" if was_held
                else "smoothing" if was_overridden
                else None
            ),
            "time_in_state": round(time_in_state, 1),
            "state_distribution": {
                self.states[i]: round(float(self.smoothed_probs[i]), 3)
                for i in range(self.n_states)
            },
        }

    def get_state_summary(self) -> Dict:
        """Get summary of state history."""
        if not self.history:
            return {"current_state": self.current_state, "total_updates": 0}

        state_times = {s: 0.0 for s in self.states}
        for i in range(1, len(self.history)):
            dt = self.history[i][0] - self.history[i - 1][0]
            state_times[self.history[i - 1][1]] += dt

        total_time = sum(state_times.values()) or 1.0

        return {
            "current_state": self.current_state,
            "total_updates": len(self.history),
            "time_in_current": round(time.time() - self.state_entered_at, 1),
            "state_percentages": {
                s: round(t / total_time * 100, 1)
                for s, t in state_times.items()
            },
        }


class BrainStateEngine:
    """Unified temporal smoothing for all 6 models.

    Usage:
        engine = BrainStateEngine()

        # Each time models produce predictions:
        smoothed = engine.update({
            "sleep": {"state": "N2", "confidence": 0.7, "probabilities": [...]},
            "flow": {"state": "flow", "flow_score": 0.8},
            "emotion": {"emotion": "focused", "confidence": 0.6},
            "creativity": {"state": "creative", "creativity_score": 0.7},
            "memory": {"state": "active_encoding", "encoding_score": 0.65},
            "dream": {"is_dreaming": False, "probability": 0.1},
        })
    """

    def __init__(self):
        self.trackers = {
            "sleep": StateTracker(
                "sleep", SLEEP_STATES, SLEEP_TRANSITIONS,
                MIN_DWELL_SECONDS["sleep"]
            ),
            "flow": StateTracker(
                "flow", FLOW_STATES, FLOW_TRANSITIONS,
                MIN_DWELL_SECONDS["flow"]
            ),
            "creativity": StateTracker(
                "creativity", CREATIVITY_STATES, CREATIVITY_TRANSITIONS,
                MIN_DWELL_SECONDS["creativity"]
            ),
            "memory": StateTracker(
                "memory", MEMORY_STATES, MEMORY_TRANSITIONS,
                MIN_DWELL_SECONDS["memory"]
            ),
            "emotion": StateTracker(
                "emotion", EMOTION_STATES, EMOTION_TRANSITIONS,
                MIN_DWELL_SECONDS["emotion"]
            ),
        }

        # Dream is binary, track separately with simple smoothing
        self._dream_ema = 0.0
        self._dream_alpha = 0.3

    def update(self, raw_predictions: Dict) -> Dict:
        """Update all state trackers with new model predictions.

        Args:
            raw_predictions: Dict with keys matching model names.
                Each value is the raw prediction dict from that model.

        Returns:
            Dict with smoothed predictions for all models.
        """
        results = {}

        # Sleep
        sleep_pred = raw_predictions.get("sleep", {})
        if sleep_pred:
            sleep_result = self.trackers["sleep"].update(
                raw_state=sleep_pred.get("stage", "Wake"),
                raw_confidence=sleep_pred.get("confidence", 0.5),
                raw_probs=sleep_pred.get("probabilities"),
            )
            results["sleep"] = sleep_result

        # Flow
        flow_pred = raw_predictions.get("flow", {})
        if flow_pred:
            flow_result = self.trackers["flow"].update(
                raw_state=flow_pred.get("state", "no_flow"),
                raw_confidence=flow_pred.get("confidence", flow_pred.get("flow_score", 0.5)),
            )
            results["flow"] = flow_result

        # Emotion
        emo_pred = raw_predictions.get("emotion", {})
        if emo_pred:
            emo_result = self.trackers["emotion"].update(
                raw_state=emo_pred.get("emotion", "relaxed"),
                raw_confidence=emo_pred.get("confidence", 0.5),
                raw_probs=emo_pred.get("probabilities"),
            )
            results["emotion"] = emo_result

        # Creativity
        cre_pred = raw_predictions.get("creativity", {})
        if cre_pred:
            cre_result = self.trackers["creativity"].update(
                raw_state=cre_pred.get("state", "analytical"),
                raw_confidence=cre_pred.get("confidence", cre_pred.get("creativity_score", 0.5)),
            )
            results["creativity"] = cre_result

        # Memory
        mem_pred = raw_predictions.get("memory", {})
        if mem_pred:
            mem_result = self.trackers["memory"].update(
                raw_state=mem_pred.get("state", "weak_encoding"),
                raw_confidence=mem_pred.get("confidence", mem_pred.get("encoding_score", 0.5)),
            )
            results["memory"] = mem_result

        # Dream (binary — simple EMA smoothing)
        dream_pred = raw_predictions.get("dream", {})
        if dream_pred:
            raw_prob = dream_pred.get("probability", 0.0)
            self._dream_ema = self._dream_alpha * raw_prob + (1 - self._dream_alpha) * self._dream_ema
            results["dream"] = {
                "smoothed_probability": round(self._dream_ema, 3),
                "is_dreaming": self._dream_ema > 0.5,
                "raw_probability": round(raw_prob, 3),
                "was_overridden": (self._dream_ema > 0.5) != dream_pred.get("is_dreaming", False),
            }

        return results

    def get_summary(self) -> Dict:
        """Get summary of all state trackers."""
        return {
            name: tracker.get_state_summary()
            for name, tracker in self.trackers.items()
        }

    def get_cross_state_coherence(self) -> Dict:
        """Check if current states across models are physiologically coherent.

        Returns warnings if states are contradictory.
        """
        warnings = []

        sleep_state = self.trackers["sleep"].current_state
        flow_state = self.trackers["flow"].current_state
        creativity_state = self.trackers["creativity"].current_state
        emotion_state = self.trackers["emotion"].current_state

        # Deep sleep + flow = impossible
        if sleep_state in ("N2", "N3") and flow_state in ("flow", "deep_flow"):
            warnings.append("Flow state during deep sleep is physiologically impossible")

        # Deep sleep + focused emotion = unlikely
        if sleep_state in ("N2", "N3") and emotion_state == "focused":
            warnings.append("Focused emotion during deep sleep is unlikely")

        # REM + deep flow = very unlikely (unless lucid dreaming)
        if sleep_state == "REM" and flow_state == "deep_flow":
            warnings.append("Deep flow during REM suggests possible lucid dream state")

        # Dreaming while fully awake with high flow = contradictory
        if sleep_state == "Wake" and self._dream_ema > 0.7 and flow_state in ("flow", "deep_flow"):
            warnings.append("High dream probability during waking flow — likely artifact")

        return {
            "is_coherent": len(warnings) == 0,
            "warnings": warnings,
            "states": {
                "sleep": sleep_state,
                "flow": flow_state,
                "creativity": creativity_state,
                "emotion": emotion_state,
                "memory": self.trackers["memory"].current_state,
                "dream_probability": round(self._dream_ema, 3),
            },
        }
