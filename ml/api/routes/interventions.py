"""Real-time intervention engine — closed-loop stress / focus / food triggers.

Logic:
  HIGH stress (≥0.65) for >60 s  → breathing exercise
  HIGH stress (≥0.65)            → calming music recommendation
  focus < 0.35 for >120 s        → focus music recommendation
  meal gap ≥ 240 min + stress    → food / protein snack suggestion

Each intervention has a 10-minute per-user cooldown to prevent spam.
Outcomes are recorded 5 minutes after trigger and fed back for personalisation.

JITAI (Just-In-Time Adaptive Intervention) extensions:
  - Thompson Sampling bandit for personalized intervention selection
  - HRV-based stress trigger detection (RMSSD drop from baseline)
  - Evidence-based new interventions: cyclic sighing, grounding, body scan,
    cognitive reappraisal, slow breathing
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.jitai_engine import HRVTriggerDetector, InterventionBandit

logger = logging.getLogger(__name__)

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
    # ── JITAI interventions ─────────────────────────────────────────────────
    "cyclic_sighing": {
        "type": "cyclic_sighing",
        "title": "Try cyclic sighing",
        "body": (
            "Double inhale through nose, long exhale through mouth. "
            "Repeat 3-4 times. Stanford research shows this outperforms "
            "meditation for mood improvement."
        ),
        "action_label": "Start sighing",
        "action_url": "/biofeedback?protocol=cyclic_sighing&auto=true",
        "icon": "wind",
        "evidence": (
            "Balban et al. (2023, Cell Reports Medicine): Cyclic sighing "
            "produced greater positive affect improvement than mindfulness "
            "meditation (p<.05). N=108 RCT."
        ),
        "priority": 1,
        "duration_seconds": 60,
        "intensity_range": [0.7, 1.0],
    },
    "grounding_54321": {
        "type": "grounding_54321",
        "title": "5-4-3-2-1 grounding",
        "body": (
            "Name 5 things you see, 4 you hear, 3 you touch, 2 you smell, "
            "1 you taste. Reestablishes sensory connection when overwhelmed."
        ),
        "action_label": "Start grounding",
        "action_url": "/biofeedback?protocol=grounding&auto=true",
        "icon": "hand",
        "evidence": (
            "Evidence-informed low-barrier technique for anxiety, panic, "
            "and dissociation. Operates through sensory modality engagement."
        ),
        "priority": 1,
        "duration_seconds": 90,
        "intensity_range": [0.7, 1.0],
    },
    "body_scan": {
        "type": "body_scan",
        "title": "Quick body scan",
        "body": (
            "Focus attention on 3-4 body regions for 20 seconds each. "
            "Notice tension without judging. Brief body scans produce "
            "measurable HRV changes."
        ),
        "action_label": "Start body scan",
        "action_url": "/biofeedback?protocol=body_scan&auto=true",
        "icon": "scan",
        "evidence": (
            "PMC 11519409 (2024): Brief mindfulness exercises including "
            "body scan produced significant HRV changes."
        ),
        "priority": 2,
        "duration_seconds": 90,
        "intensity_range": [0.4, 0.7],
    },
    "cognitive_reappraisal": {
        "type": "cognitive_reappraisal",
        "title": "Reframe this moment",
        "body": (
            "What is the situation? What is another way to interpret it? "
            "Most validated emotion regulation strategy for mild-to-moderate "
            "distress."
        ),
        "action_label": "Start reframing",
        "action_url": "/biofeedback?protocol=reappraisal&auto=true",
        "icon": "lightbulb",
        "evidence": (
            "30+ years of research. Most effective at low-to-moderate "
            "intensity. Less effective under high intensity (prefrontal "
            "demands too high)."
        ),
        "priority": 2,
        "duration_seconds": 60,
        "intensity_range": [0.0, 0.4],
    },
    "slow_breathing": {
        "type": "slow_breathing",
        "title": "Slow breathing (6 breaths/min)",
        "body": (
            "Inhale 5 seconds, exhale 5 seconds. Six breaths per minute "
            "maximizes heart rate variability and activates the "
            "parasympathetic system."
        ),
        "action_label": "Start breathing",
        "action_url": "/biofeedback?protocol=slow_breathing&auto=true",
        "icon": "wind",
        "evidence": (
            "Strong meta-analytic evidence. Significant HRV increases in "
            "single 2-minute sessions. A52 Breath Method variant supported "
            "by 2025 narrative review."
        ),
        "priority": 1,
        "duration_seconds": 120,
        "intensity_range": [0.4, 0.7],
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


# ── JITAI module-level instances ─────────────────────────────────────────────

_bandit = InterventionBandit()
_hrv_detector = HRVTriggerDetector()

# All intervention types eligible for bandit selection
_JITAI_ELIGIBLE = [
    "breathing", "music_calm", "music_focus", "food", "walk",
    "cyclic_sighing", "grounding_54321", "body_scan",
    "cognitive_reappraisal", "slow_breathing",
]


# ── JITAI Pydantic models ───────────────────────────────────────────────────

class JITAICheckRequest(BaseModel):
    user_id: str = Field(default="default")
    stress_index: float = Field(default=0.0, ge=0.0, le=1.0)
    focus_index: float = Field(default=0.5, ge=0.0, le=1.0)
    emotion_type: Optional[str] = Field(
        default=None, description="Dominant emotion: happy, sad, angry, fear, etc."
    )
    emotion_intensity: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Distress intensity for bandit context."
    )
    rmssd: Optional[float] = Field(
        default=None, description="HRV RMSSD in milliseconds."
    )
    heart_rate: Optional[float] = Field(
        default=None, description="Heart rate in BPM."
    )
    minutes_since_last_meal: Optional[float] = Field(
        default=None, description="Minutes since last meal (None = unknown)."
    )
    voice_emotion: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Voice emotion reading (same format as CheckRequest).",
    )


class HRVReadingRequest(BaseModel):
    user_id: str = Field(default="default")
    rmssd: float = Field(..., description="RMSSD in milliseconds")
    heart_rate: Optional[float] = Field(
        default=None, description="Heart rate in BPM."
    )


# ── JITAI helper ─────────────────────────────────────────────────────────────

def _jitai_check_logic(req: JITAICheckRequest) -> Dict:
    """Sync core of the /jitai/check endpoint — separated for testability."""
    now = time.time()
    st = _user_state(req.user_id)

    # Respect snooze
    if now < st["last_snooze_until"]:
        return {
            "intervention": None,
            "has_recommendation": False,
            "trigger_source": None,
            "snoozed": True,
        }

    # Enforce per-user cooldown
    if now - st["last_triggered_ts"] < COOLDOWN_SECS:
        return {
            "intervention": None,
            "has_recommendation": False,
            "trigger_source": None,
            "cooldown_active": True,
        }

    trigger_source: Optional[str] = None

    # ── HRV trigger detection ──────────────────────────────────────────────
    hrv_triggered = False
    if req.rmssd is not None:
        # Feed reading to build baseline
        _hrv_detector.add_reading(req.user_id, req.rmssd, req.heart_rate)
        # Check if this reading triggers
        hrv_result = _hrv_detector.check_trigger(req.user_id, req.rmssd)
        hrv_triggered = hrv_result.get("triggered", False)
        if hrv_triggered:
            trigger_source = "hrv"

    # ── Standard stress/focus triggers ─────────────────────────────────────
    stress_triggered = req.stress_index >= STRESS_HIGH
    focus_triggered = req.focus_index < FOCUS_LOW

    # ── Voice emotion trigger ──────────────────────────────────────────────
    voice_triggered = False
    if req.voice_emotion:
        voice_conf = float(req.voice_emotion.get("confidence", 0.0))
        voice_arousal = float(req.voice_emotion.get("arousal", 0.5))
        voice_valence = float(req.voice_emotion.get("valence", 0.0))
        if voice_conf >= 0.5 and (voice_arousal >= 0.70 or voice_valence <= -0.30):
            voice_triggered = True
            if trigger_source is None:
                trigger_source = "voice"

    # Determine if any trigger fired
    any_trigger = hrv_triggered or stress_triggered or voice_triggered
    if not any_trigger and not focus_triggered:
        return {
            "intervention": None,
            "has_recommendation": False,
            "trigger_source": None,
        }

    if trigger_source is None:
        trigger_source = "eeg" if stress_triggered else "focus"

    # ── Determine intensity for bandit context ─────────────────────────────
    intensity = req.emotion_intensity
    if stress_triggered:
        intensity = max(intensity, req.stress_index)

    # ── Filter available interventions ─────────────────────────────────────
    available = list(_JITAI_ELIGIBLE)

    # Remove food unless hunger is a factor
    if req.minutes_since_last_meal is None or req.minutes_since_last_meal < FOOD_GAP_MINS:
        available = [a for a in available if a != "food"]

    # Remove focus music if focus is not the issue
    if not focus_triggered:
        available = [a for a in available if a != "music_focus"]

    if not available:
        available = ["breathing"]  # absolute fallback

    # ── Bandit selection ───────────────────────────────────────────────────
    selected_type = _bandit.select(
        user_id=req.user_id,
        available_interventions=available,
        intensity=intensity,
        emotion_type=req.emotion_type,
    )

    intervention = dict(INTERVENTIONS.get(selected_type, INTERVENTIONS["breathing"]))
    intervention["selection_method"] = (
        "bandit" if _bandit._outcome_counts.get(req.user_id, 0) >= 10
        else "evidence_default"
    )

    return {
        "intervention": intervention,
        "has_recommendation": True,
        "trigger_source": trigger_source,
        "selected_by": intervention.get("selection_method", "evidence_default"),
    }


# ── JITAI endpoints ─────────────────────────────────────────────────────────

@router.post("/interventions/jitai/check")
async def jitai_check(req: JITAICheckRequest):
    """JITAI-enhanced intervention check with HRV triggering + bandit selection.

    Accepts HRV features (rmssd, heart_rate) alongside stress/focus.
    Uses Thompson Sampling to select optimal intervention for this user.
    Does NOT modify existing /interventions/check behavior.
    """
    return _jitai_check_logic(req)


@router.post("/interventions/jitai/hrv-reading")
async def add_hrv_reading(req: HRVReadingRequest):
    """Feed HRV reading to build per-user baseline for trigger detection.

    Call this every 30 seconds with the latest RMSSD value. After 5 readings,
    the baseline is considered stable and trigger detection activates.
    """
    status = _hrv_detector.add_reading(req.user_id, req.rmssd, req.heart_rate)
    return {"ok": True, "baseline": status}


@router.get("/interventions/jitai/bandit-stats/{user_id}")
async def get_bandit_stats(user_id: str):
    """Return current bandit priors and selection stats for a user.

    Shows per-arm Alpha/Beta parameters, estimated reward, and whether
    the user is still in cold-start mode.
    """
    return _bandit.get_stats(user_id)


@router.post("/interventions/jitai/outcome")
async def jitai_outcome(req: OutcomeRequest):
    """Record outcome and feed reward to the bandit.

    This wraps the existing outcome logic and additionally updates the
    Thompson Sampling bandit with a normalized reward signal.
    """
    # Use existing outcome recording logic
    history = _user_history(req.user_id)
    matched = None
    for entry in reversed(history):
        if entry["type"] == req.intervention_type and entry["outcome_recorded_at"] is None:
            matched = entry
            break

    if matched is None:
        return {"ok": False, "error": "No pending intervention of that type found"}

    matched["stress_after"] = req.stress_after
    matched["focus_after"] = req.focus_after
    matched["felt_helpful"] = req.felt_helpful
    matched["outcome_recorded_at"] = time.time()

    stress_delta = matched["stress_before"] - req.stress_after  # positive = improved
    focus_delta = req.focus_after - matched["focus_before"]      # positive = improved

    # Compute bandit reward: normalize stress reduction to [0, 1]
    # stress_delta range is roughly [-1, 1], map to [0, 1]
    reward = max(0.0, min(1.0, (stress_delta + 1.0) / 2.0))
    # Bonus for self-reported helpfulness
    if req.felt_helpful is True:
        reward = min(1.0, reward + 0.15)
    elif req.felt_helpful is False:
        reward = max(0.0, reward - 0.15)

    _bandit.update(req.user_id, req.intervention_type, reward)

    return {
        "ok": True,
        "stress_delta": round(stress_delta, 3),
        "focus_delta": round(focus_delta, 3),
        "worked": stress_delta > 0.05 or (req.felt_helpful is True),
        "bandit_reward": round(reward, 3),
    }
