"""Brain music sonification mapper.

Maps EEG emotional state to musical parameters:
- FAA (valence) → major/minor scale selection + key
- Arousal → tempo (60-140 BPM)
- Alpha power → dynamics (volume/intensity)
- Theta power → harmonic complexity

References:
    Miranda & Brouse (2005) — Brain-computer music interface
    Mealla et al. (2014) — Sonification of EEG signals
"""
from typing import Dict, Optional

import numpy as np


# Musical keys mapped to valence spectrum
VALENCE_KEY_MAP = [
    # (threshold, key, scale_type)
    (-0.8, "A", "minor"),     # deeply sad/negative
    (-0.6, "D", "minor"),     # melancholic
    (-0.4, "E", "minor"),     # somber
    (-0.2, "B", "minor"),     # pensive
    (0.0, "A", "major"),      # neutral-calm
    (0.2, "C", "major"),      # mildly positive
    (0.4, "D", "major"),      # happy
    (0.6, "G", "major"),      # joyful
    (0.8, "E", "major"),      # euphoric
]

# Complexity levels mapped to theta power
COMPLEXITY_MAP = {
    "simple": {"intervals": [0, 4, 7], "description": "root-third-fifth"},
    "moderate": {"intervals": [0, 2, 4, 7], "description": "root-second-third-fifth"},
    "complex": {"intervals": [0, 2, 4, 5, 7, 9], "description": "hexatonic"},
    "rich": {"intervals": [0, 2, 3, 5, 7, 8, 10, 11], "description": "chromatic-mix"},
}


class BrainMusicMapper:
    """Map EEG brain states to musical parameters for sonification."""

    def __init__(self):
        self._min_tempo = 60
        self._max_tempo = 140
        self._history: list = []

    def map(self, emotion_result: Dict) -> Dict:
        """Map emotion classifier output to musical parameters.

        Args:
            emotion_result: Output from emotion classifier containing
                frontal_asymmetry, arousal, valence, and band powers.

        Returns:
            Dict with scale, key, tempo_bpm, dynamics, complexity,
            intervals, and mood_color fields.
        """
        faa = emotion_result.get("frontal_asymmetry", 0.0)
        arousal = emotion_result.get("arousal", 0.5)
        valence = emotion_result.get("valence", 0.0)
        alpha = emotion_result.get("alpha_power", 0.0)
        theta = emotion_result.get("theta_power", 0.0)

        # Use valence if available, otherwise derive from FAA
        effective_valence = valence if valence != 0.0 else float(np.tanh(faa * 2.0))

        # Scale and key from valence
        key, scale_type = self._map_key(effective_valence)

        # Tempo from arousal (60-140 BPM)
        tempo_bpm = int(self._min_tempo + float(np.clip(arousal, 0, 1)) * (self._max_tempo - self._min_tempo))

        # Dynamics from alpha power (calm = louder sustained notes)
        dynamics = float(np.clip(alpha * 2.0, 0.1, 1.0)) if alpha > 0 else 0.5

        # Harmonic complexity from theta (high theta = more complex harmonics)
        complexity, intervals = self._map_complexity(theta)

        # Mood color for visualization
        mood_color = self._mood_to_color(effective_valence, arousal)

        result = {
            "key": key,
            "scale": scale_type,
            "tempo_bpm": tempo_bpm,
            "dynamics": dynamics,
            "complexity": complexity,
            "intervals": intervals,
            "effective_valence": float(effective_valence),
            "arousal": float(arousal),
            "mood_color": mood_color,
        }

        self._history.append(result)
        if len(self._history) > 100:
            self._history = self._history[-100:]

        return result

    def _map_key(self, valence: float):
        """Map valence to musical key and scale type."""
        for threshold, key, scale_type in VALENCE_KEY_MAP:
            if valence <= threshold:
                return key, scale_type
        return "E", "major"  # very positive default

    def _map_complexity(self, theta: float):
        """Map theta power to harmonic complexity."""
        if theta <= 0.1:
            level = "simple"
        elif theta <= 0.3:
            level = "moderate"
        elif theta <= 0.6:
            level = "complex"
        else:
            level = "rich"
        return level, COMPLEXITY_MAP[level]["intervals"]

    def _mood_to_color(self, valence: float, arousal: float) -> str:
        """Map valence-arousal to a hex color for visualization."""
        # Valence: red (negative) → green (positive)
        # Arousal: dark (low) → bright (high)
        r = int(max(0, min(255, 128 - valence * 127)))
        g = int(max(0, min(255, 128 + valence * 127)))
        b = int(max(0, min(255, 100 + arousal * 155)))
        return f"#{r:02x}{g:02x}{b:02x}"

    def get_history(self) -> list:
        """Get recent mapping history for visualization."""
        return list(self._history)

    def get_average_mood(self) -> Dict:
        """Get average musical parameters from recent history."""
        if not self._history:
            return {"key": "C", "scale": "major", "tempo_bpm": 100, "dynamics": 0.5}
        recent = self._history[-10:]
        return {
            "avg_tempo": int(np.mean([h["tempo_bpm"] for h in recent])),
            "avg_dynamics": float(np.mean([h["dynamics"] for h in recent])),
            "avg_valence": float(np.mean([h["effective_valence"] for h in recent])),
            "dominant_scale": max(set(h["scale"] for h in recent), key=lambda s: sum(1 for h in recent if h["scale"] == s)),
        }
