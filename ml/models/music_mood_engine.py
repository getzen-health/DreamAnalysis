"""Adaptive Music Mood Engine — ISO Principle Controller (#284).

Implements the ISO principle for emotion-adaptive music therapy:
  Phase 1: Match current emotional state (prevent abrupt change)
  Phase 2: Gradually shift toward target state
  Phase 3: Sustain target state

Evidence base:
  - de Witte et al. 2020/2025: d=0.723 stress reduction (51 RCTs, 6147 records)
  - Neurophone 2024: EEG→real-time music parameter control (Muse S)
  - HeartDJ 2025: HRV-adaptive music generation (Dartmouth)

Does NOT integrate MusicGen (requires GPU/large model download).
Instead outputs: music search queries, Spotify-compatible parameters,
and tempo/key/mode recommendations for the current ISO phase.
Frontend can use these parameters to search Spotify/Apple Music/YouTube.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ── Valence/Arousal → Music Parameter Mapping ────────────────────────────────
# Based on Russell's circumplex model + psychoacoustics research

@dataclass
class MusicParameters:
    tempo_bpm: float          # 40-200 BPM
    key: str                  # e.g. "C major", "A minor"
    mode: str                 # "major" or "minor"
    energy: float             # 0-1 (maps to loudness/activity)
    valence_label: str        # "high-positive" / "positive" / "neutral" / "negative" / "high-negative"
    arousal_label: str        # "high" / "medium" / "low"
    search_query: str         # Ready-to-use search query
    genre_suggestions: List[str]
    avoid_features: List[str] # Anti-recommendations (e.g. avoid lyrics when meditating)


def _valence_bucket(v: float) -> str:
    if v > 0.5: return "high-positive"
    if v > 0.15: return "positive"
    if v > -0.15: return "neutral"
    if v > -0.5: return "negative"
    return "high-negative"


def _arousal_bucket(a: float) -> str:
    if a > 0.65: return "high"
    if a > 0.35: return "medium"
    return "low"


# Mapping from (valence_bucket, arousal_bucket) → MusicParameters
_PARAM_MAP: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("high-positive", "high"):   {"tempo": 140, "key": "G major", "mode": "major", "energy": 0.85, "genres": ["pop", "funk", "upbeat electronic"], "avoid": []},
    ("high-positive", "medium"): {"tempo": 110, "key": "D major", "mode": "major", "energy": 0.65, "genres": ["indie pop", "acoustic", "classical light"], "avoid": []},
    ("high-positive", "low"):    {"tempo": 75,  "key": "F major", "mode": "major", "energy": 0.40, "genres": ["acoustic", "ambient", "lo-fi"], "avoid": []},
    ("positive",      "high"):   {"tempo": 120, "key": "A major", "mode": "major", "energy": 0.75, "genres": ["dance", "reggae", "upbeat pop"], "avoid": []},
    ("positive",      "medium"): {"tempo": 100, "key": "C major", "mode": "major", "energy": 0.55, "genres": ["pop", "acoustic", "singer-songwriter"], "avoid": []},
    ("positive",      "low"):    {"tempo": 70,  "key": "G major", "mode": "major", "energy": 0.35, "genres": ["ambient", "folk", "new age"], "avoid": ["loud", "aggressive"]},
    ("neutral",       "high"):   {"tempo": 120, "key": "E minor", "mode": "minor", "energy": 0.65, "genres": ["electronic", "cinematic", "rock"], "avoid": []},
    ("neutral",       "medium"): {"tempo": 90,  "key": "C major", "mode": "major", "energy": 0.50, "genres": ["jazz", "classical", "folk"], "avoid": []},
    ("neutral",       "low"):    {"tempo": 65,  "key": "D minor", "mode": "minor", "energy": 0.30, "genres": ["ambient", "meditation", "piano"], "avoid": ["lyrics", "upbeat"]},
    ("negative",      "high"):   {"tempo": 80,  "key": "A minor", "mode": "minor", "energy": 0.55, "genres": ["blues", "soul", "cinematic"], "avoid": ["aggressive", "loud"]},
    ("negative",      "medium"): {"tempo": 70,  "key": "D minor", "mode": "minor", "energy": 0.40, "genres": ["classical", "ambient", "indie acoustic"], "avoid": ["upbeat", "party"]},
    ("negative",      "low"):    {"tempo": 60,  "key": "C minor", "mode": "minor", "energy": 0.25, "genres": ["ambient", "drone", "meditation"], "avoid": ["lyrics", "upbeat", "energetic"]},
    ("high-negative", "high"):   {"tempo": 75,  "key": "F minor", "mode": "minor", "energy": 0.50, "genres": ["ambient", "nature sounds", "soft piano"], "avoid": ["lyrics", "upbeat", "aggressive"]},
    ("high-negative", "medium"): {"tempo": 60,  "key": "A minor", "mode": "minor", "energy": 0.30, "genres": ["meditation", "ambient", "binaural alpha"], "avoid": ["lyrics", "drums", "energetic"]},
    ("high-negative", "low"):    {"tempo": 50,  "key": "C minor", "mode": "minor", "energy": 0.20, "genres": ["sleep music", "delta waves", "deep ambient"], "avoid": ["all stimulating"]},
}

_SEARCH_TEMPLATES: Dict[Tuple[str, str], str] = {
    ("high-positive", "high"):   "upbeat happy energetic pop dance",
    ("high-positive", "medium"): "happy positive acoustic cheerful",
    ("high-positive", "low"):    "calming happy peaceful acoustic",
    ("positive",      "high"):   "positive energetic uplifting",
    ("positive",      "medium"): "positive mood background music",
    ("positive",      "low"):    "gentle positive calm ambient",
    ("neutral",       "high"):   "focus background electronic instrumental",
    ("neutral",       "medium"): "neutral background jazz study music",
    ("neutral",       "low"):    "ambient relaxing neutral instrumental",
    ("negative",      "high"):   "calming stress relief instrumental",
    ("negative",      "medium"): "soothing relaxing calm music",
    ("negative",      "low"):    "deep relaxation meditation ambient",
    ("high-negative", "high"):   "anxiety relief nature sounds",
    ("high-negative", "medium"): "stress relief meditation music",
    ("high-negative", "low"):    "sleep music delta waves deep rest",
}


def get_music_parameters(valence: float, arousal: float) -> MusicParameters:
    vb = _valence_bucket(valence)
    ab = _arousal_bucket(arousal)
    params = _PARAM_MAP.get((vb, ab), _PARAM_MAP[("neutral", "medium")])
    query = _SEARCH_TEMPLATES.get((vb, ab), "relaxing background music")

    return MusicParameters(
        tempo_bpm=params["tempo"],
        key=params["key"],
        mode=params["mode"],
        energy=params["energy"],
        valence_label=vb,
        arousal_label=ab,
        search_query=query,
        genre_suggestions=params["genres"],
        avoid_features=params.get("avoid", []),
    )


# ── ISO Principle Controller ──────────────────────────────────────────────────

class ISOController:
    """Implements the ISO principle for music therapy sessions.

    ISO principle: match current emotional state first, then gradually
    shift toward target state through intermediate musical steps.

    Session structure (default 20 min):
      Phase 1 (match):      6 min  — same valence/arousal as current state
      Phase 2 (transition): 8 min  — midpoint between current and target
      Phase 3 (target):     6 min  — target emotional state
    """

    DEFAULT_SESSION_MIN = 20  # Evidence: effects plateau after 45 min

    def prescribe(
        self,
        current_valence: float,
        current_arousal: float,
        target_valence: float = 0.4,
        target_arousal: Optional[float] = None,
        session_min: int = DEFAULT_SESSION_MIN,
    ) -> Dict[str, Any]:
        """Generate full ISO session prescription.

        Parameters
        ----------
        current_valence : float  -1 to 1
        current_arousal : float   0 to 1
        target_valence  : float  -1 to 1  (default: mildly positive)
        target_arousal  : float   0 to 1  (default: auto from target_valence)
        session_min     : int    Total session length in minutes
        """
        if target_arousal is None:
            # High positive → medium arousal; very negative → low arousal
            target_arousal = 0.5 + target_valence * 0.2

        target_arousal = max(0.0, min(1.0, target_arousal))

        # ISO midpoint
        mid_valence = (current_valence + target_valence) / 2
        mid_arousal = (current_arousal + target_arousal) / 2

        phase_1 = get_music_parameters(current_valence, current_arousal)
        phase_2 = get_music_parameters(mid_valence, mid_arousal)
        phase_3 = get_music_parameters(target_valence, target_arousal)

        phase_min = [
            round(session_min * 0.30),   # 30% match
            round(session_min * 0.40),   # 40% transition
            round(session_min * 0.30),   # 30% target
        ]

        stress_before = max(0.0, min(1.0, (-current_valence + current_arousal) / 2))
        stress_after_est = max(0.0, min(1.0, (-target_valence + target_arousal) / 2))
        expected_stress_delta = round((stress_before - stress_after_est) * 100, 1)

        return {
            "session_duration_min": session_min,
            "evidence_grade": "A",
            "evidence_summary": "d=0.723 stress reduction (de Witte 2020/2025, 51 RCTs, 6147 participants)",
            "current_state": {
                "valence": round(current_valence, 3),
                "arousal": round(current_arousal, 3),
                "label": f"{phase_1.valence_label} valence, {phase_1.arousal_label} arousal",
            },
            "target_state": {
                "valence": round(target_valence, 3),
                "arousal": round(target_arousal, 3),
                "label": f"{phase_3.valence_label} valence, {phase_3.arousal_label} arousal",
            },
            "expected_stress_reduction_pct": expected_stress_delta,
            "phases": [
                {
                    "phase": 1,
                    "name": "Match",
                    "description": "Music matches your current emotional state to create rapport",
                    "duration_min": phase_min[0],
                    "parameters": self._format_phase(phase_1),
                },
                {
                    "phase": 2,
                    "name": "Transition",
                    "description": "Gradual shift toward target state",
                    "duration_min": phase_min[1],
                    "parameters": self._format_phase(phase_2),
                },
                {
                    "phase": 3,
                    "name": "Target",
                    "description": "Sustained target emotional state",
                    "duration_min": phase_min[2],
                    "parameters": self._format_phase(phase_3),
                },
            ],
        }

    @staticmethod
    def _format_phase(p: MusicParameters) -> Dict[str, Any]:
        return {
            "tempo_bpm": round(p.tempo_bpm),
            "key": p.key,
            "mode": p.mode,
            "energy_level": round(p.energy, 2),
            "search_query": p.search_query,
            "genre_suggestions": p.genre_suggestions,
            "avoid_features": p.avoid_features,
        }


_controller: Optional[ISOController] = None


def get_iso_controller() -> ISOController:
    global _controller
    if _controller is None:
        _controller = ISOController()
    return _controller
