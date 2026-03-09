"""EEG-guided binaural beat and music neurofeedback.

Based on: Brain Sciences 2025 — closed-loop binaural beat entrainment.
The system reads real-time EEG state and computes optimal binaural beat
frequency and target brainwave band.

Binaural beat: play slightly different tone in each ear (e.g. 200 Hz left,
210 Hz right) → brain perceives 10 Hz beat → entrains toward alpha.
"""
from __future__ import annotations
import logging
from typing import Dict, Optional

log = logging.getLogger(__name__)


# Binaural beat targets by desired mental state
ENTRAINMENT_TARGETS = {
    "focus":      {"band": "beta",  "beat_hz": 15.0, "carrier_hz": 200.0, "description": "Mental focus and concentration"},
    "relax":      {"band": "alpha", "beat_hz": 10.0, "carrier_hz": 200.0, "description": "Relaxed awareness"},
    "meditation": {"band": "theta", "beat_hz": 6.0,  "carrier_hz": 200.0, "description": "Deep meditation and creativity"},
    "sleep":      {"band": "delta", "beat_hz": 2.0,  "carrier_hz": 100.0, "description": "Deep sleep induction"},
    "flow":       {"band": "alpha", "beat_hz": 8.0,  "carrier_hz": 200.0, "description": "Flow state and creativity"},
    "calm":       {"band": "alpha", "beat_hz": 9.0,  "carrier_hz": 180.0, "description": "Stress reduction"},
}


class BinauralFeedbackController:
    """Closed-loop binaural beat neurofeedback.

    Reads EEG state → selects optimal beat frequency → adjusts adaptively.
    """

    def __init__(self):
        self.target_state = "relax"
        self.current_beat_hz = 10.0
        self.carrier_hz = 200.0
        self.volume = 0.3  # 0-1
        self.session_active = False
        self.adaptation_rate = 0.1  # how fast to adjust beat frequency

    def start_session(self, target_state: str = "relax", volume: float = 0.3) -> Dict:
        """Start a binaural beat neurofeedback session."""
        if target_state not in ENTRAINMENT_TARGETS:
            target_state = "relax"
        self.target_state = target_state
        self.volume = max(0.0, min(1.0, volume))
        target = ENTRAINMENT_TARGETS[target_state]
        self.current_beat_hz = target["beat_hz"]
        self.carrier_hz = target["carrier_hz"]
        self.session_active = True
        return self._current_params()

    def stop_session(self) -> Dict:
        self.session_active = False
        return {"status": "stopped"}

    def update_from_eeg(self, eeg_features: Dict) -> Dict:
        """Adjust beat frequency based on current EEG state.

        Args:
            eeg_features: dict with band powers (alpha, beta, theta, delta)

        Returns:
            Updated binaural beat parameters
        """
        if not self.session_active:
            return self._current_params()

        target = ENTRAINMENT_TARGETS[self.target_state]
        target_band = target["band"]
        target_beat = target["beat_hz"]

        # Read current band power dominance
        alpha = eeg_features.get("alpha", 0.5)
        beta = eeg_features.get("beta", 0.5)
        theta = eeg_features.get("theta", 0.5)
        delta = eeg_features.get("delta", 0.5)
        total = alpha + beta + theta + delta + 1e-10

        band_fractions = {
            "alpha": alpha / total,
            "beta": beta / total,
            "theta": theta / total,
            "delta": delta / total,
        }

        current_fraction = band_fractions.get(target_band, 0.0)

        # If current band fraction is below 30%, nudge beat frequency closer to target
        # If above 50%, we're already in the zone — maintain
        if current_fraction < 0.25:
            # User is far from target — increase beat intensity (move beat freq toward target)
            adjustment = self.adaptation_rate * (target_beat - self.current_beat_hz)
            self.current_beat_hz = max(0.5, min(40.0, self.current_beat_hz + adjustment))
        elif current_fraction > 0.45:
            # User is in target band — gently oscillate to maintain engagement
            self.current_beat_hz = target_beat * (1.0 + 0.05 * (current_fraction - 0.45))

        return self._current_params(band_fractions=band_fractions, entrainment_score=current_fraction)

    def _current_params(self, band_fractions: Optional[Dict] = None, entrainment_score: float = 0.0) -> Dict:
        return {
            "session_active": self.session_active,
            "target_state": self.target_state,
            "beat_frequency_hz": round(self.current_beat_hz, 2),
            "carrier_hz": self.carrier_hz,
            "left_ear_hz": self.carrier_hz,
            "right_ear_hz": round(self.carrier_hz + self.current_beat_hz, 2),
            "volume": self.volume,
            "description": ENTRAINMENT_TARGETS.get(self.target_state, {}).get("description", ""),
            "entrainment_score": round(entrainment_score, 3),
            "band_fractions": band_fractions or {},
        }

    def get_status(self) -> Dict:
        return self._current_params()
