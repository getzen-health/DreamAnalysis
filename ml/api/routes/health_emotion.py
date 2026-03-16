"""Passive emotion estimation from wearable health data (HR/HRV/sleep/activity).

POST /health-emotion/estimate
    Accept watch-level biometrics (HR, HRV/RMSSD, respiratory rate, steps, sleep
    hours) and return a passive emotion estimate with explanation.

No ML model is loaded — purely feature-based heuristics grounded in published
autonomic nervous system research:
  - Task Force ESC/NASPE (1996): HRV standard measures
  - McCraty et al. (2009): HRV and emotional states
  - Thayer & Lane (2000): A model of neurovisceral integration
  - Kim & André (2008): Physiological signals for emotion recognition
  - Åkerstedt & Gillberg (1990): sleep and daytime affect

Population reference ranges used as thresholds (healthy adults at rest):
  HR:       55-100 BPM (normal); resting: 60-80 BPM
  RMSSD:    20-60 ms (normal vagal tone)
  Resp:     12-20 br/min
  Sleep:    7-9 h (recommended adult range)
  Steps/h:  0-150 typical sedentary; >500 = recent exercise
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(prefix="/health-emotion", tags=["health_emotion"])


# ── Request / response models ─────────────────────────────────────────────────

class HealthEmotionRequest(BaseModel):
    hr_bpm: float = Field(..., ge=20, le=250, description="Current or resting heart rate (BPM)")
    hrv_rmssd_ms: Optional[float] = Field(None, ge=0, le=300, description="HRV RMSSD in milliseconds")
    respiratory_rate: Optional[float] = Field(None, ge=4, le=60, description="Respiratory rate (breaths/min)")
    steps_last_hour: Optional[int] = Field(None, ge=0, description="Step count in the last hour (recent activity)")
    sleep_hours: Optional[float] = Field(None, ge=0, le=24, description="Hours slept last night")
    timestamp: Optional[str] = Field(None, description="ISO 8601 timestamp of the measurement")


class HealthEmotionResponse(BaseModel):
    emotion: str
    valence: float
    arousal: float
    stress: float
    confidence: float
    source: str
    watch_says: str
    explanation: str


# ── Heuristic mapping ─────────────────────────────────────────────────────────

# Emotion label lookup from (valence_zone, arousal_zone)
# valence_zone: "positive" | "neutral" | "negative"
# arousal_zone: "high" | "medium" | "low"
_EMOTION_TABLE: dict[tuple[str, str], str] = {
    ("positive", "high"):   "excited",
    ("positive", "medium"): "happy",
    ("positive", "low"):    "calm",
    ("neutral",  "high"):   "alert",
    ("neutral",  "medium"): "neutral",
    ("neutral",  "low"):    "relaxed",
    ("negative", "high"):   "stressed",
    ("negative", "medium"): "tense",
    ("negative", "low"):    "drained",
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(np.clip(v, lo, hi))


def estimate(
    hr_bpm: float,
    hrv_rmssd_ms: Optional[float],
    respiratory_rate: Optional[float],
    steps_last_hour: Optional[int],
    sleep_hours: Optional[float],
) -> HealthEmotionResponse:
    """Map health biometrics to emotion estimate.

    Returns a HealthEmotionResponse without requiring any trained model.
    """

    # ── Arousal (0–1): how activated / energised ─────────────────────────────
    # Primary driver: HR elevation above comfortable resting baseline (70 BPM).
    # Higher HR → higher arousal.
    hr_ref = 70.0  # typical resting HR
    hr_max = 120.0  # HR at which arousal saturates to 1.0
    hr_arousal = _clamp((hr_bpm - hr_ref) / (hr_max - hr_ref), 0.0, 1.0)

    # HRV contribution: low RMSSD → high sympathetic tone → elevated arousal
    if hrv_rmssd_ms is not None and hrv_rmssd_ms > 0:
        rmssd_arousal = _clamp(1.0 - hrv_rmssd_ms / 55.0, 0.0, 1.0)
        arousal_raw = 0.55 * hr_arousal + 0.45 * rmssd_arousal
    else:
        arousal_raw = hr_arousal

    # Respiratory rate boosts arousal when elevated
    if respiratory_rate is not None and respiratory_rate > 0:
        resp_boost = _clamp((respiratory_rate - 15.0) / 15.0, 0.0, 0.25)
        arousal_raw = _clamp(arousal_raw + resp_boost * 0.15, 0.0, 1.0)

    # Recent physical activity elevates arousal but in a healthy way
    # (steps_last_hour > 500 → moderately active; does not push arousal past 0.75)
    if steps_last_hour is not None and steps_last_hour > 0:
        step_boost = _clamp(steps_last_hour / 1500.0, 0.0, 0.20)
        arousal_raw = _clamp(arousal_raw + step_boost, 0.0, 0.85)

    arousal = round(arousal_raw, 3)

    # ── Stress (0–1): negative high-arousal state ─────────────────────────────
    # Stress = arousal with a negative valence modifier.
    # High HR + low HRV → stress.
    # Recent exercise with decent HRV → arousal WITHOUT stress.
    hr_stress = _clamp((hr_bpm - 60.0) / 60.0, 0.0, 1.0)

    if hrv_rmssd_ms is not None and hrv_rmssd_ms > 0:
        rmssd_stress = _clamp(1.0 - hrv_rmssd_ms / 75.0, 0.0, 1.0)
        stress_raw = 0.45 * hr_stress + 0.55 * rmssd_stress
    else:
        # Without HRV, use only HR with lower confidence
        stress_raw = hr_stress * 0.6

    # Sleep deprivation adds to stress baseline
    if sleep_hours is not None:
        if sleep_hours < 6.0:
            sleep_penalty = _clamp((6.0 - sleep_hours) / 3.0, 0.0, 0.30)
            stress_raw = _clamp(stress_raw + sleep_penalty * 0.20, 0.0, 1.0)

    # Physical activity context: if step count is high, some of the "high HR"
    # is exercise-driven, not stress-driven. Dampen stress proportionally.
    if steps_last_hour is not None and steps_last_hour > 400:
        exercise_fraction = _clamp(steps_last_hour / 1200.0, 0.0, 0.40)
        stress_raw = _clamp(stress_raw * (1.0 - exercise_fraction * 0.5), 0.0, 1.0)

    stress = round(stress_raw, 3)

    # ── Valence (−1 to +1): negative ↔ positive feeling ──────────────────────
    # HRV is the best non-EEG valence predictor: high vagal tone correlates with
    # positive affect and emotional regulation capacity.
    if hrv_rmssd_ms is not None and hrv_rmssd_ms > 0:
        # RMSSD ~50 ms → neutral (0). Higher → positive; lower → negative.
        rmssd_valence = _clamp((hrv_rmssd_ms - 25.0) / 30.0, -1.0, 1.0)
        # Scale to ±0.7 to leave room for sleep modifier
        valence_raw = rmssd_valence * 0.70
    else:
        # Without HRV, infer valence from stress (stressed → negative)
        valence_raw = -stress_raw * 0.5

    # Sleep quality modifier: poor sleep → negative valence
    if sleep_hours is not None:
        if sleep_hours < 6.0:
            sleep_penalty = _clamp((6.0 - sleep_hours) / 3.0, 0.0, 1.0)
            valence_raw = _clamp(valence_raw - sleep_penalty * 0.30, -1.0, 1.0)
        elif sleep_hours >= 7.0 and sleep_hours <= 9.5:
            # Well-slept → small positive valence boost
            valence_raw = _clamp(valence_raw + 0.10, -1.0, 1.0)

    # Exercise bonus: recent steps (400-1500+) → mild positive valence
    if steps_last_hour is not None and steps_last_hour >= 400:
        activity_bonus = _clamp(steps_last_hour / 3000.0, 0.0, 0.15)
        valence_raw = _clamp(valence_raw + activity_bonus, -1.0, 1.0)

    valence = round(valence_raw, 3)

    # ── Confidence ────────────────────────────────────────────────────────────
    # Base confidence from number of signals available
    n_signals = 1  # HR always present
    if hrv_rmssd_ms is not None:   n_signals += 1
    if respiratory_rate is not None: n_signals += 1
    if steps_last_hour is not None:  n_signals += 1
    if sleep_hours is not None:      n_signals += 1

    # 1 signal (HR only): 40%; 2: 55%; 3: 65%; 4: 72%; 5: 78%
    signal_conf = _clamp(0.30 + (n_signals - 1) * 0.12, 0.0, 0.80)

    # HRV is the most important single contributor — bonus for having it
    hrv_bonus = 0.10 if hrv_rmssd_ms is not None else 0.0
    confidence = round(_clamp(signal_conf + hrv_bonus, 0.0, 0.85), 3)

    # ── Emotion label from (valence, arousal) ─────────────────────────────────
    v_zone = "positive" if valence > 0.15 else "negative" if valence < -0.15 else "neutral"
    a_zone = "high" if arousal > 0.60 else "low" if arousal < 0.30 else "medium"
    emotion = _EMOTION_TABLE.get((v_zone, a_zone), "neutral")

    # ── Human-readable "Your watch says you're …" ─────────────────────────────
    watch_says = _build_watch_says(
        emotion=emotion,
        stress=stress,
        arousal=arousal,
        valence=valence,
        steps_last_hour=steps_last_hour,
        sleep_hours=sleep_hours,
        hr_bpm=hr_bpm,
        hrv_rmssd_ms=hrv_rmssd_ms,
    )

    explanation = _build_explanation(
        hr_bpm=hr_bpm,
        hrv_rmssd_ms=hrv_rmssd_ms,
        respiratory_rate=respiratory_rate,
        steps_last_hour=steps_last_hour,
        sleep_hours=sleep_hours,
        stress=stress,
        arousal=arousal,
    )

    return HealthEmotionResponse(
        emotion=emotion,
        valence=valence,
        arousal=arousal,
        stress=stress,
        confidence=confidence,
        source="health",
        watch_says=watch_says,
        explanation=explanation,
    )


def _build_watch_says(
    emotion: str,
    stress: float,
    arousal: float,
    valence: float,
    steps_last_hour: Optional[int],
    sleep_hours: Optional[float],
    hr_bpm: float,
    hrv_rmssd_ms: Optional[float],
) -> str:
    # Exercise context takes priority over generic arousal messaging
    if steps_last_hour is not None and steps_last_hour >= 400 and arousal > 0.5:
        return "Your watch says you're active — elevated HR from recent movement."

    if stress > 0.65:
        hrv_note = ""
        if hrv_rmssd_ms is not None and hrv_rmssd_ms < 25:
            hrv_note = f" (HRV: {hrv_rmssd_ms:.0f} ms)"
        return f"Your watch says you're stressed{hrv_note} — HR {hr_bpm:.0f} BPM with reduced heart rate variability."

    if stress < 0.25 and arousal < 0.35:
        if hrv_rmssd_ms is not None and hrv_rmssd_ms >= 40:
            return f"Your watch says you're calm — HRV {hrv_rmssd_ms:.0f} ms suggests strong parasympathetic tone."
        return "Your watch says you're calm — low HR and stable rhythms."

    if sleep_hours is not None and sleep_hours < 6.0:
        return f"Your watch says you're drained — only {sleep_hours:.1f}h of sleep detected last night."

    if valence > 0.25 and arousal > 0.45:
        return f"Your watch says you're energised — HR {hr_bpm:.0f} BPM with good heart rate variability."

    if emotion == "relaxed":
        return "Your watch says you're relaxed — your autonomic nervous system is in recovery mode."

    if emotion == "neutral":
        return "Your watch says you're in a balanced state — no notable physiological stress signals."

    # Generic fallback
    return f"Your watch says you're {emotion}."


def _build_explanation(
    hr_bpm: float,
    hrv_rmssd_ms: Optional[float],
    respiratory_rate: Optional[float],
    steps_last_hour: Optional[int],
    sleep_hours: Optional[float],
    stress: float,
    arousal: float,
) -> str:
    parts: list[str] = []

    # HR
    if hr_bpm > 90:
        parts.append(f"HR {hr_bpm:.0f} BPM (elevated)")
    elif hr_bpm < 60:
        parts.append(f"HR {hr_bpm:.0f} BPM (low resting)")
    else:
        parts.append(f"HR {hr_bpm:.0f} BPM")

    # HRV
    if hrv_rmssd_ms is not None:
        if hrv_rmssd_ms < 20:
            parts.append(f"HRV {hrv_rmssd_ms:.0f} ms (low — reduced vagal tone)")
        elif hrv_rmssd_ms > 50:
            parts.append(f"HRV {hrv_rmssd_ms:.0f} ms (high — strong vagal tone)")
        else:
            parts.append(f"HRV {hrv_rmssd_ms:.0f} ms")

    # Respiratory rate
    if respiratory_rate is not None:
        if respiratory_rate > 20:
            parts.append(f"resp {respiratory_rate:.0f} br/min (elevated)")
        else:
            parts.append(f"resp {respiratory_rate:.0f} br/min")

    # Activity
    if steps_last_hour is not None and steps_last_hour > 0:
        if steps_last_hour >= 500:
            parts.append(f"{steps_last_hour} steps/h (active)")
        else:
            parts.append(f"{steps_last_hour} steps/h")

    # Sleep
    if sleep_hours is not None:
        if sleep_hours < 6:
            parts.append(f"{sleep_hours:.1f}h sleep (insufficient)")
        elif sleep_hours >= 7:
            parts.append(f"{sleep_hours:.1f}h sleep")
        else:
            parts.append(f"{sleep_hours:.1f}h sleep (marginal)")

    return "; ".join(parts) if parts else "health data"


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/estimate", response_model=HealthEmotionResponse)
async def estimate_health_emotion(body: HealthEmotionRequest) -> HealthEmotionResponse:
    """Estimate emotional state passively from wearable health data.

    This endpoint requires no EEG device or voice input — it works with
    data routinely collected by Apple Watch / Garmin / Fitbit.

    **Physiological grounding:**
    - High HR (>90 BPM) + low HRV (<25 ms) → high stress / negative arousal
    - Low HR (<65 BPM) + high HRV (>50 ms) → parasympathetic / calm / positive
    - Recent exercise (steps_last_hour > 500) → arousal without stress penalty
    - Sleep < 6 h → lowered valence baseline + stress penalty

    **Confidence interpretation:**
    - 0.40–0.55: HR-only estimate (low confidence)
    - 0.55–0.70: HR + HRV or HR + sleep (moderate)
    - 0.70–0.85: full multi-signal estimate
    """
    try:
        return estimate(
            hr_bpm=body.hr_bpm,
            hrv_rmssd_ms=body.hrv_rmssd_ms,
            respiratory_rate=body.respiratory_rate,
            steps_last_hour=body.steps_last_hour,
            sleep_hours=body.sleep_hours,
        )
    except Exception as exc:
        log.exception("health-emotion/estimate failed: %s", exc)
        return HealthEmotionResponse(
            emotion="unknown",
            valence=0.0,
            arousal=0.0,
            stress=0.0,
            confidence=0.0,
            source="health",
            watch_says="Unable to estimate — check health data availability.",
            explanation=str(exc),
        )
