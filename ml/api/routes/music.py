"""Brain music sonification endpoints.

POST /brain-music/generate  — generate MIDI note events from EEG emotional state
GET  /brain-music/scales    — list available scales and their intervals
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Any, Dict, List

log = logging.getLogger(__name__)
router = APIRouter()


def _get_generator():
    """Lazy-load BrainMusicGenerator to avoid import-time side-effects."""
    from audio.brain_music import BrainMusicGenerator
    return BrainMusicGenerator()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class EmotionMusicRequest(BaseModel):
    valence: float = Field(..., ge=-1.0, le=1.0, description="Valence score −1 to 1")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Arousal score 0 to 1")
    alpha_power: float = Field(
        default=0.2, ge=0.0, description="Alpha band power (normalised, ≥ 0)"
    )
    user_id: str = Field(default="default", description="User identifier")
    n_notes: int = Field(default=4, ge=1, le=16, description="Number of notes to generate")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/brain-music/generate")
async def generate_brain_music(req: EmotionMusicRequest) -> Dict[str, Any]:
    """Generate music notes from EEG emotional state.

    Returns:
        scale: name of the selected scale
        tempo_bpm: beats per minute (40–180)
        note_density: notes per bar (1–8)
        velocity: base MIDI velocity (30–100)
        notes: list of {note (MIDI 60–84), velocity, duration (seconds)}
        state_description: human-readable description of emotional state
        user_id: echoed from request
    """
    gen = _get_generator()
    params = gen.map_emotion_to_params(req.valence, req.arousal, req.alpha_power)
    notes = gen.generate_notes(req.valence, req.arousal, req.alpha_power, req.n_notes)
    description = gen.get_state_description(req.valence, req.arousal)

    return {
        "scale": params["scale"],
        "tempo_bpm": params["tempo_bpm"],
        "note_density": params["note_density"],
        "velocity": params["velocity"],
        "notes": notes,
        "state_description": description,
        "user_id": req.user_id,
    }


@router.get("/brain-music/scales")
async def list_scales() -> Dict[str, Any]:
    """Return available scale names and their semitone intervals relative to C4."""
    from audio.brain_music import SCALES
    return {
        "scales": {
            name: {
                "intervals": intervals,
                "description": _scale_descriptions().get(name, ""),
            }
            for name, intervals in SCALES.items()
        }
    }


def _scale_descriptions() -> Dict[str, str]:
    return {
        "major_pentatonic": "Bright and uplifting — positive high-arousal states",
        "minor_pentatonic": "Warm and resolved — positive low-arousal states",
        "dorian": "Jazzy and ambiguous — mildly negative or neutral states",
        "chromatic": "Tense and unresolved — negative or highly stressed states",
    }
