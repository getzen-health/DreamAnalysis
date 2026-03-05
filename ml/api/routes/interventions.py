"""Real-time intervention engine — closed-loop stress / focus / food triggers.

Logic:
  HIGH stress (≥0.65) for >60 s  → breathing exercise
  HIGH stress (≥0.65)            → calming music recommendation
  focus < 0.35 for >120 s        → focus music recommendation
  meal gap ≥ 240 min + stress    → food / protein snack suggestion

Each intervention has a 10-minute per-user cooldown to prevent spam.
Outcomes are recorded 5 minutes after trigger and fed back for personalisation.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

# ── Thresholds ────────────────────────────────────────────────────────────────
STRESS_HIGH       = 0.65   # stress_index threshold for "high" stress
STRESS_MODERATE   = 0.45   # threshold for moderate stress (food trigger)
FOCUS_LOW         = 0.35   # focus_index below this = focus problem
BREATHING_SECS    = 60     # seconds of sustained high stress before breathing trigger
FOCUS_PROBLEM_SECS = 120   # seconds of sustained low focus before music trigger
COOLDOWN_SECS     = 600    # 10-minute cooldown between interventions per user
FOOD_GAP_MINS     = 240    # 4 hours without eating + any stress → food suggestion

# ── Per-user state ────────────────────────────────────────────────────────────
# Thread-safe dict keyed by user_id
_lock = threading.Lock()

_state: Dict[str, Dict] = {}
# Schema per user:
# {
#   "stress_elevated_since": float | None,   # epoch when stress first crossed high threshold
#   "focus_low_since":       float | None,   # epoch when focus first dropped low
#   "last_triggered_ts":     float,          # epoch of last intervention trigger
#   "last_triggered_type":   str,            # type of last trigger
#   "last_snooze_until":     float,          # epoch until interventions are snoozed
#   "history":               list[dict],     # [{type, ts, outcome_stress_before, outcome_stress_after}]
# }

_intervention_history: Dict[str, List[Dict]] = {}  # separate so history survives state reset


def _user_state(user_id: str) -> Dict:
    """Return (or create) mutable per-user state dict."""
    with _lock:
        if user_id not in _state:
            _state[user_id] = {
                "stress_elevated_since": None,
                "focus_low_since": None,
                "last_triggered_ts": 0.0,
                "last_triggered_type": "",
                "last_snooze_until": 0.0,
            }
        return _state[user_id]


def _user_history(user_id: str) -> List[Dict]:
    with _lock:
        if user_id not in _intervention_history:
            _intervention_history[user_id] = []
        return _intervention_history[user_id]


# ── Intervention catalogue ────────────────────────────────────────────────────

INTERVENTIONS = {
    "breathing": {
        "type": "breathing",
        "title": "Time to breathe",
        "body": "Your stress has been elevated for over a minute. A 4-minute coherence breathing session can cut cortisol by 15–20%.",
        "action_label": "Start breathing",
        "action_url": "/biofeedback?protocol=coherence&auto=true",
        "icon": "wind",
        "evidence": "Lehrer & Vaschillo (2000): HRV biofeedback reduces cortisol. 4–6 min coherence breathing (5.5 BPM) most effective.",
        "priority": 1,
    },
    "music_calm": {
        "type": "music_calm",
        "title": "Calming music can help",
        "body": "High stress detected. Slow-tempo music (60–80 BPM) activates your parasympathetic system within minutes.",
        "action_label": "Open music",
        "action_url": "/biofeedback?tab=music&mood=calm",
        "icon": "music",
        "evidence": "Thoma et al. (2013): Self-selected calming music reduces cortisol during acute stress recovery.",
        "priority": 2,
    },
    "music_focus": {
        "type": "music_focus",
        "title": "Focus music to re-engage",
        "body": "Your focus has been low for 2+ minutes. Binaural beats at 40 Hz (gamma) can increase sustained attention within 10 minutes.",
        "action_label": "Focus sounds",
        "action_url": "/biofeedback?tab=music&mood=focus",
        "icon": "headphones",
        "evidence": "Kraus et al. (2021): Gamma-frequency binaural beats improve selective attention and working memory.",
        "priority": 2,
    },
    "food": {
        "type": "food",
        "title": "Low blood sugar may be raising your stress",
        "body": "You haven't eaten in 4+ hours and your stress is rising. Have a protein snack — not sugar, which will spike and crash.",
        "action_label": "Food tips",
        "action_url": "/food?alert=protein_snack",
        "icon": "apple",
        "evidence": "Macht (2008): Hunger increases negative affect and cortisol. Protein stabilises blood glucose better than carbohydrates alone.",
        "priority": 3,
    },
    "walk": {
        "type": "walk",
        "title": "A short walk could break this stress pattern",
        "body": "Sustained high stress detected. Even a 5-minute walk outside reduces cortisol and improves mood for hours.",
        "action_label": "Set walk reminder",
        "action_url": "/brain-report?action=walk_reminder",
        "icon": "footprints",
        "evidence": "Kang et al. (2020): 5-min nature walks significantly reduce perceived stress and improve affect.",
        "priority": 3,
    },
}


# ── Core decision logic ───────────────────────────────────────────────────────

def _decide_intervention(
    user_id: str,
    stress_index: float,
    focus_index: float,
    minutes_since_last_meal: Optional[float],
) -> Optional[Dict]:
    """Evaluate current brain state and return an intervention dict or None."""
    now = time.time()
    st = _user_state(user_id)

    # Respect snooze
    if now < st["last_snooze_until"]:
        return None

    # Enforce per-user cooldown
    if now - st["last_triggered_ts"] < COOLDOWN_SECS:
        return None

    # ── Track stress elevation duration ──────────────────────────────────────
    if stress_index >= STRESS_HIGH:
        if st["stress_elevated_since"] is None:
            st["stress_elevated_since"] = now
    else:
        st["stress_elevated_since"] = None

    # ── Track focus low duration ──────────────────────────────────────────────
    if focus_index < FOCUS_LOW:
        if st["focus_low_since"] is None:
            st["focus_low_since"] = now
    else:
        st["focus_low_since"] = None

    # ── Decision tree (priority order) ───────────────────────────────────────

    # Priority 1: sustained high stress → breathing (most evidence-backed)
    stress_duration = (now - st["stress_elevated_since"]) if st["stress_elevated_since"] else 0
    if stress_index >= STRESS_HIGH and stress_duration >= BREATHING_SECS:
        return INTERVENTIONS["breathing"]

    # Priority 2: food (hunger amplifies stress — address root cause first)
    if (
        minutes_since_last_meal is not None
        and minutes_since_last_meal >= FOOD_GAP_MINS
        and stress_index >= STRESS_MODERATE
        and st["last_triggered_type"] != "food"  # don't repeat food immediately
    ):
        return INTERVENTIONS["food"]

    # Priority 3: sustained high stress → calming music (if breathing already done)
    if stress_index >= STRESS_HIGH and st["last_triggered_type"] == "breathing":
        return INTERVENTIONS["music_calm"]

    # Priority 4: sustained low focus → focus music
    focus_duration = (now - st["focus_low_since"]) if st["focus_low_since"] else 0
    if focus_index < FOCUS_LOW and focus_duration >= FOCUS_PROBLEM_SECS:
        return INTERVENTIONS["music_focus"]

    # Priority 5: prolonged high stress with no recent intervention → walk
    if stress_index >= STRESS_HIGH and stress_duration >= BREATHING_SECS * 3:
        return INTERVENTIONS["walk"]

    return None


# ── Pydantic models ───────────────────────────────────────────────────────────

class CheckRequest(BaseModel):
    user_id: str = Field(default="default")
    stress_index: float = Field(default=0.0, ge=0.0, le=1.0)
    focus_index: float = Field(default=0.5, ge=0.0, le=1.0)
    minutes_since_last_meal: Optional[float] = Field(
        default=None, description="Minutes since last meal (None = unknown)"
    )
    voice_emotion: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Voice emotion reading from voice_emotion2vec or equivalent. "
            "Expected keys: emotion (str), valence (float -1..1), "
            "arousal (float 0..1), confidence (float 0..1), "
            "probabilities (dict), model_type (str)."
        ),
    )


class TriggerRequest(BaseModel):
    user_id: str = Field(default="default")
    intervention_type: str = Field(..., description="Type from INTERVENTIONS keys")
    stress_before: float = Field(default=0.0, ge=0.0, le=1.0)
    focus_before: float = Field(default=0.5, ge=0.0, le=1.0)


class OutcomeRequest(BaseModel):
    user_id: str = Field(default="default")
    intervention_type: str
    stress_after: float = Field(ge=0.0, le=1.0)
    focus_after: float = Field(ge=0.0, le=1.0)
    felt_helpful: Optional[bool] = Field(default=None)


class SnoozeRequest(BaseModel):
    user_id: str = Field(default="default")
    minutes: int = Field(default=10, ge=1, le=120)


# ── Endpoints ─────────────────────────────────────────────────────────────────

VOICE_STRESS_INTERVENTION = {
    "type": "voice_stress",
    "title": "Your voice signals elevated stress",
    "body": (
        "Voice analysis detected elevated stress or negative emotion. "
        "A short breathing exercise can reduce cortisol within minutes."
    ),
    "action_label": "Start breathing",
    "action_url": "/biofeedback?protocol=coherence&auto=true",
    "icon": "mic",
    "evidence": (
        "Schuller et al. (2013): Acoustic features reliably index emotional arousal "
        "and can serve as a stress proxy when EEG is unavailable or low-confidence."
    ),
    "priority": 2,
    "source": "voice",
}


def _check_intervention_logic(req: CheckRequest) -> Dict:
    """Sync core of the /check endpoint — separated so tests can call it directly."""
    intervention = _decide_intervention(
        req.user_id,
        req.stress_index,
        req.focus_index,
        req.minutes_since_last_meal,
    )

    # ── Voice emotion triggers ────────────────────────────────────────────────
    # Only evaluate voice if no EEG-based intervention is already queued and a
    # voice reading is present with sufficient model confidence.
    if intervention is None and req.voice_emotion:
        voice_valence = float(req.voice_emotion.get("valence", 0.0))
        voice_arousal = float(req.voice_emotion.get("arousal", 0.5))
        voice_label   = req.voice_emotion.get("emotion", "neutral")
        voice_conf    = float(req.voice_emotion.get("confidence", 0.0))

        # Only act on confident voice readings
        if voice_conf >= 0.5:
            if voice_arousal >= 0.70 or voice_valence <= -0.30:
                # Build a copy of the template so callers can mutate freely
                intervention = dict(VOICE_STRESS_INTERVENTION)
                intervention["body"] = (
                    f"Voice detected {voice_label} "
                    f"(valence {voice_valence:+.2f}, arousal {voice_arousal:.2f}). "
                    "Consider a breathing exercise."
                )

    return {
        "intervention": intervention,
        "has_recommendation": intervention is not None,
    }


@router.post("/interventions/check")
async def check_intervention(req: CheckRequest):
    """Evaluate current brain state and return an intervention recommendation.

    Call this every 30 seconds from the frontend.  Returns null when no
    intervention is needed or the cooldown is active.
    """
    return _check_intervention_logic(req)


@router.post("/interventions/trigger")
async def record_trigger(req: TriggerRequest):
    """Record that the user acted on an intervention (opened the screen / started exercise).

    Updates cooldown timer and logs stress_before for later outcome tracking.
    """
    if req.intervention_type not in INTERVENTIONS:
        return {"ok": False, "error": f"Unknown intervention type: {req.intervention_type}"}

    now = time.time()
    st = _user_state(req.user_id)
    st["last_triggered_ts"] = now
    st["last_triggered_type"] = req.intervention_type

    entry = {
        "type": req.intervention_type,
        "triggered_at": now,
        "stress_before": req.stress_before,
        "focus_before": req.focus_before,
        "stress_after": None,
        "focus_after": None,
        "felt_helpful": None,
        "outcome_recorded_at": None,
    }
    _user_history(req.user_id).append(entry)

    return {
        "ok": True,
        "cooldown_until": now + COOLDOWN_SECS,
        "logged": entry,
    }


@router.post("/interventions/outcome")
async def record_outcome(req: OutcomeRequest):
    """Record the outcome of the most recent intervention (call 5 min after trigger).

    Measures whether stress actually dropped.  Logged data feeds future
    personalisation — what works for THIS user.
    """
    history = _user_history(req.user_id)
    # Find the most recent matching entry with no outcome yet
    matched = None
    for entry in reversed(history):
        if entry["type"] == req.intervention_type and entry["outcome_recorded_at"] is None:
            matched = entry
            break

    if matched is None:
        return {"ok": False, "error": "No pending intervention of that type found"}

    matched["stress_after"]         = req.stress_after
    matched["focus_after"]          = req.focus_after
    matched["felt_helpful"]         = req.felt_helpful
    matched["outcome_recorded_at"]  = time.time()

    stress_delta = matched["stress_before"] - req.stress_after   # positive = improved
    focus_delta  = req.focus_after - matched["focus_before"]     # positive = improved

    return {
        "ok": True,
        "stress_delta": round(stress_delta, 3),
        "focus_delta":  round(focus_delta, 3),
        "worked": stress_delta > 0.05 or (req.felt_helpful is True),
    }


@router.post("/interventions/snooze")
async def snooze_interventions(req: SnoozeRequest):
    """Snooze all interventions for this user for N minutes."""
    st = _user_state(req.user_id)
    snooze_until = time.time() + req.minutes * 60
    st["last_snooze_until"] = snooze_until
    return {"ok": True, "snoozed_until": snooze_until, "minutes": req.minutes}


@router.get("/interventions/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    """Return recent intervention history for a user."""
    history = _user_history(user_id)
    return {
        "user_id": user_id,
        "total": len(history),
        "history": list(reversed(history[-limit:])),
    }


@router.get("/interventions/effectiveness/{user_id}")
async def get_effectiveness(user_id: str):
    """Aggregate which intervention types actually reduced stress for this user."""
    history = _user_history(user_id)
    outcomes = [e for e in history if e.get("outcome_recorded_at") is not None]

    by_type: Dict[str, Dict] = {}
    for e in outcomes:
        t = e["type"]
        if t not in by_type:
            by_type[t] = {"count": 0, "worked": 0, "avg_stress_delta": 0.0, "deltas": []}
        by_type[t]["count"] += 1
        delta = (e["stress_before"] or 0) - (e["stress_after"] or 0)
        by_type[t]["deltas"].append(delta)
        if delta > 0.05 or e.get("felt_helpful"):
            by_type[t]["worked"] += 1

    for t, d in by_type.items():
        deltas = d.pop("deltas")
        d["avg_stress_delta"] = round(sum(deltas) / len(deltas), 3) if deltas else 0.0
        d["success_rate"] = round(d["worked"] / d["count"], 2) if d["count"] else 0.0

    return {
        "user_id": user_id,
        "total_outcomes": len(outcomes),
        "by_type": by_type,
    }


@router.get("/interventions/catalogue")
async def get_catalogue():
    """Return all available intervention types with metadata."""
    return {"interventions": list(INTERVENTIONS.values())}
