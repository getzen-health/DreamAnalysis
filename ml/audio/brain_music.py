"""Brain Music Generator — maps EEG emotional state to MIDI/note events.

Scientific basis:
- IEEE (2024): Neurophone BCMI — FAA-to-music mapping on Muse S (same layout as Muse 2)
- Wiley (2024): Mind to Music — valence→scale, arousal→tempo, alpha→dynamics

Mapping:
- valence (−1 to 1) → scale selection (chromatic/dorian/minor/major pentatonic)
- arousal (0 to 1) → tempo (40–180 BPM) and note density (1–8 notes/bar)
- alpha_power → dynamics (higher alpha = softer notes, higher velocity when low alpha)
- Markov chain on scale degrees for melodic coherence

No external audio library dependency — returns pure Python dicts with note events.
All note values are MIDI note numbers (60–84, C4–C6).
"""

import math
import random
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Scale definitions — intervals in semitones relative to root (C4 = MIDI 60)
# ---------------------------------------------------------------------------
SCALES: Dict[str, List[int]] = {
    # Bright, uplifting — positive high-arousal states
    "major_pentatonic": [0, 2, 4, 7, 9],
    # Warm, resolved — positive low-arousal states
    "minor_pentatonic": [0, 3, 5, 7, 10],
    # Jazzy, ambiguous — mildly negative states
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    # Tense, unresolved — negative states
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

# Root note: C4 (MIDI 60). Notes span C4–C6 (MIDI 60–84, two octaves).
_ROOT = 60
_MAX_NOTE = 84  # C6

# Markov transition weights: prefer stepwise motion (adjacent scale degrees)
_MARKOV_WEIGHTS: List[float] = [0.05, 0.20, 0.40, 0.20, 0.10, 0.04, 0.01]


def _markov_next(current_idx: int, scale_len: int) -> int:
    """Return next scale-degree index using a step-motion-preferring distribution."""
    n_weights = len(_MARKOV_WEIGHTS)
    half = n_weights // 2
    weights: List[float] = []
    indices: List[int] = []
    for offset in range(-half, half + 1):
        target = current_idx + offset
        if 0 <= target < scale_len:
            indices.append(target)
            w_idx = offset + half
            weights.append(_MARKOV_WEIGHTS[w_idx] if w_idx < n_weights else 0.01)
    if not indices:
        return current_idx
    total = sum(weights)
    weights = [w / total for w in weights]
    r = random.random()
    cumulative = 0.0
    for idx, w in zip(indices, weights):
        cumulative += w
        if r <= cumulative:
            return idx
    return indices[-1]


class BrainMusicGenerator:
    """Maps EEG emotional state to MIDI note events.

    All outputs are pure Python dicts — no external audio library required.
    """

    SCALES = SCALES

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map_emotion_to_params(
        self,
        valence: float,
        arousal: float,
        alpha_power: float,
    ) -> Dict[str, Any]:
        """Map EEG state to music generation parameters.

        Args:
            valence: −1.0 (very negative) to 1.0 (very positive)
            arousal: 0.0 (calm) to 1.0 (highly energised)
            alpha_power: instantaneous alpha band power (non-negative float,
                         typically 0.0–1.0 for normalised features)

        Returns:
            {
                "scale": str,           # name of selected scale
                "tempo_bpm": float,     # 40–180 BPM
                "note_density": int,    # notes per bar (1–8)
                "velocity": int,        # MIDI velocity (30–100)
            }
        """
        valence = float(max(-1.0, min(1.0, valence)))
        arousal = float(max(0.0, min(1.0, arousal)))
        alpha_power = float(max(0.0, alpha_power))

        scale = self._select_scale(valence, arousal)
        tempo_bpm = self._compute_tempo(arousal)
        note_density = self._compute_note_density(arousal)
        velocity = self._compute_velocity(arousal, alpha_power)

        return {
            "scale": scale,
            "tempo_bpm": round(tempo_bpm, 1),
            "note_density": note_density,
            "velocity": velocity,
        }

    def generate_notes(
        self,
        valence: float,
        arousal: float,
        alpha_power: float,
        n_notes: int = 4,
    ) -> List[Dict[str, Any]]:
        """Generate a sequence of MIDI note events from EEG state.

        Args:
            valence: −1.0 to 1.0
            arousal: 0.0 to 1.0
            alpha_power: alpha band power (≥ 0.0)
            n_notes: number of notes to generate (1–16)

        Returns:
            List of dicts, each containing:
                {
                    "note": int,        # MIDI note number (60–84)
                    "velocity": int,    # MIDI velocity (30–100)
                    "duration": float,  # note duration in seconds
                }
        """
        n_notes = max(1, min(16, int(n_notes)))
        params = self.map_emotion_to_params(valence, arousal, alpha_power)

        scale_name = params["scale"]
        scale_intervals = SCALES[scale_name]
        scale_len = len(scale_intervals)

        tempo_bpm = params["tempo_bpm"]
        base_velocity = params["velocity"]

        # Beat duration in seconds
        beat_duration = 60.0 / tempo_bpm
        # Quarter-note duration (basis for note length)
        note_duration = self._compute_duration(arousal, beat_duration)

        # Start from a scale degree near the middle of the scale
        current_idx = scale_len // 2
        # Choose a starting octave: keep notes in 60–84 range
        octave_shift = self._choose_octave_shift(valence)

        notes: List[Dict[str, Any]] = []
        for _ in range(n_notes):
            current_idx = _markov_next(current_idx, scale_len)
            semitone = scale_intervals[current_idx]
            midi_note = _ROOT + octave_shift * 12 + semitone
            # Clamp to valid MIDI range
            midi_note = max(60, min(_MAX_NOTE, midi_note))

            # Add slight velocity humanisation (±5)
            vel_jitter = random.randint(-5, 5)
            velocity = max(30, min(100, base_velocity + vel_jitter))

            notes.append(
                {
                    "note": midi_note,
                    "velocity": velocity,
                    "duration": round(note_duration, 3),
                }
            )

        return notes

    def get_state_description(self, valence: float, arousal: float) -> str:
        """Return a human-readable description of the emotional state.

        Examples: "calm and positive", "tense and energetic",
                  "anxious and agitated", "serene and content"
        """
        valence = float(max(-1.0, min(1.0, valence)))
        arousal = float(max(0.0, min(1.0, arousal)))

        # Valence dimension
        if valence > 0.5:
            valence_word = "joyful"
        elif valence > 0.15:
            valence_word = "positive"
        elif valence > -0.15:
            valence_word = "neutral"
        elif valence > -0.5:
            valence_word = "tense"
        else:
            valence_word = "distressed"

        # Arousal dimension
        if arousal > 0.75:
            arousal_word = "highly energetic"
        elif arousal > 0.55:
            arousal_word = "energetic"
        elif arousal > 0.35:
            arousal_word = "alert"
        elif arousal > 0.15:
            arousal_word = "relaxed"
        else:
            arousal_word = "very calm"

        return f"{valence_word} and {arousal_word}"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_scale(valence: float, arousal: float) -> str:
        """Choose a scale name based on valence and arousal.

        Mapping (Neurophone BCMI / Mind-to-Music):
            valence > 0.3               → major_pentatonic  (bright, positive)
            0.0 < valence ≤ 0.3         → minor_pentatonic  (warm, mild)
            -0.3 < valence ≤ 0.0        → dorian            (ambiguous, jazzy)
            valence ≤ -0.3              → chromatic         (tense, dark)
        """
        if valence > 0.3:
            return "major_pentatonic"
        elif valence > 0.0:
            return "minor_pentatonic"
        elif valence > -0.3:
            return "dorian"
        else:
            return "chromatic"

    @staticmethod
    def _compute_tempo(arousal: float) -> float:
        """Map arousal [0, 1] linearly to tempo [40, 180] BPM."""
        return 40.0 + arousal * 140.0

    @staticmethod
    def _compute_note_density(arousal: float) -> int:
        """Map arousal [0, 1] to note density [1, 8] notes/bar."""
        raw = 1.0 + arousal * 7.0
        return max(1, min(8, int(round(raw))))

    @staticmethod
    def _compute_velocity(arousal: float, alpha_power: float) -> int:
        """Compute MIDI velocity from arousal and alpha_power.

        Higher alpha → softer (lower velocity).
        Higher arousal → louder (higher velocity).

        Base range: 30–100.
        """
        # Arousal contribution: 0→30, 1→70
        arousal_contrib = 30.0 + arousal * 40.0
        # Alpha contribution: high alpha reduces velocity
        # Clamp alpha_power: treat values > 1 as saturated at 1
        alpha_clamped = min(1.0, alpha_power)
        alpha_reduction = alpha_clamped * 30.0
        raw = arousal_contrib - alpha_reduction
        return max(30, min(100, int(round(raw))))

    @staticmethod
    def _compute_duration(arousal: float, beat_duration: float) -> float:
        """Compute note duration in seconds.

        High arousal → shorter, detached notes (0.5 × beat).
        Low arousal  → longer, sustained notes (1.5 × beat).
        """
        # Linear interpolation: arousal=0 → 1.5×beat, arousal=1 → 0.5×beat
        multiplier = 1.5 - arousal * 1.0
        return beat_duration * multiplier

    @staticmethod
    def _choose_octave_shift(valence: float) -> int:
        """Pick an octave offset (0 or 1) to keep notes bright vs grounded.

        Positive valence → higher octave (octave shift 1, notes in C5–C6).
        Non-positive     → lower octave (octave shift 0, notes in C4–C5).
        """
        return 1 if valence > 0.0 else 0
