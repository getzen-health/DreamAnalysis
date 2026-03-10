"""Daily Brain Report — no-EEG mode.

Generates a daily brain report from voice check-ins + Apple Health data when
no EEG session is available.  Falls back gracefully when health or voice data
is sparse.

Endpoint
--------
GET /brain-report/{user_id}
    Returns a structured report with:
    - data_sources: list of what was used ("voice", "health", "eeg")
    - sleep_quality: 0-100 or null
    - focus_forecast: 0-100
    - stress_risk: 0-100
    - dominant_mood: emotion label or null
    - recommended_action: string
    - peak_focus_window: "HH:MM–HH:MM" or null
    - insight: one-liner about yesterday's patterns
"""
from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from .voice_watch import _VOICE_CACHE, _VOICE_CACHE_TTL, _VOICE_HISTORY

log = logging.getLogger(__name__)
router = APIRouter(prefix="/brain-report", tags=["Brain Report"])


# ── Response model ────────────────────────────────────────────────────────────

class BrainReport(BaseModel):
    user_id: str
    date: str
    data_sources: List[str]          # ["voice", "health"] or ["eeg", "voice", "health"]
    sleep_quality: Optional[float]   # 0-100 or null
    focus_forecast: float            # 0-100
    stress_risk: float               # 0-100
    dominant_mood: Optional[str]
    mood_valence: Optional[float]    # -1 to 1
    recommended_action: str
    peak_focus_window: Optional[str] # "09:00–11:00" or null
    insight: str
    has_eeg: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _yesterday_voice(user_id: str) -> Dict[str, Any]:
    """Pull latest voice result from the canonical voice-watch cache."""
    import time
    entry = _VOICE_CACHE.get(user_id)
    if not entry:
        return {}
    # Honour TTL — stale cache is treated as no data
    if time.time() - entry.get("ts", 0) > _VOICE_CACHE_TTL:
        return {}
    r = entry.get("result", {})
    stress_raw = r.get("stress_from_watch")
    stress_index = min(1.0, float(stress_raw) / 10.0) if stress_raw is not None else r.get("stress_index", 0.0)
    return {
        "avg_valence":      float(r.get("valence", 0.0)),
        "avg_stress":       float(stress_index),
        "avg_arousal":      float(r.get("arousal", 0.5)),
        "evening_stress":   float(stress_index),
        "dominant_emotion": r.get("emotion", "neutral"),
        "count": 1,
    }


def _peak_focus_from_arousal(user_id: str) -> Optional[str]:
    """Estimate peak focus window from the current voice-watch cache entry."""
    import time
    entry = _VOICE_CACHE.get(user_id)
    if not entry or time.time() - entry.get("ts", 0) > _VOICE_CACHE_TTL:
        return None
    arousal = float(entry.get("result", {}).get("arousal", 0.5))
    # High arousal → morning peak; lower arousal → later morning window
    base_hour = 9 if arousal >= 0.6 else 10
    return f"{base_hour:02d}:00–{base_hour + 2:02d}:00"


def _health_daily_summary(user_id: str) -> Dict[str, Any]:
    """Pull today's health summary from the health DB (best-effort)."""
    try:
        from .health import _health_db  # type: ignore[attr-defined]
        result = _health_db.get_daily_summary(user_id, None)
        if isinstance(result, dict):
            return result
    except Exception as exc:
        log.debug("Health DB not available: %s", exc)
    return {}


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.get("/{user_id}", response_model=BrainReport)
async def get_brain_report(user_id: str) -> BrainReport:
    """Generate a daily brain report using voice + health (EEG optional)."""
    today = datetime.date.today().isoformat()
    data_sources: List[str] = []

    # ── Voice data ────────────────────────────────────────────────────────────
    voice = _yesterday_voice(user_id)
    if voice:
        data_sources.append("voice")

    # ── Health data ───────────────────────────────────────────────────────────
    health = _health_daily_summary(user_id)
    if health:
        data_sources.append("health")

    # ── EEG presence check (simple: any EEG in health summary) ───────────────
    has_eeg = bool(health.get("eeg_sessions") or health.get("alpha_power"))
    if has_eeg:
        data_sources.append("eeg")

    # ── Sleep quality ─────────────────────────────────────────────────────────
    sleep_quality: Optional[float] = None
    if "sleep_efficiency" in health and health["sleep_efficiency"] is not None:
        sleep_quality = _clamp(float(health["sleep_efficiency"]))
    elif "sleep_score" in health and health["sleep_score"] is not None:
        sleep_quality = _clamp(float(health["sleep_score"]))

    # ── HRV ───────────────────────────────────────────────────────────────────
    hrv_sdnn: Optional[float] = health.get("hrv_sdnn")
    hrv_norm: float = 0.5  # neutral default
    if hrv_sdnn and hrv_sdnn > 0:
        # Normalise: 70 ms = excellent (1.0), 20 ms = poor (0.0)
        hrv_norm = _clamp((float(hrv_sdnn) - 20) / 50, 0.0, 1.0)

    # ── Focus forecast (0-100) ────────────────────────────────────────────────
    focus_components: List[float] = []
    weights: List[float] = []

    if sleep_quality is not None:
        focus_components.append(sleep_quality / 100)
        weights.append(0.40)

    if hrv_sdnn:
        focus_components.append(hrv_norm)
        weights.append(0.30)

    if voice.get("avg_valence") is not None:
        # Map valence [-1, 1] → [0, 1]
        focus_components.append((voice["avg_valence"] + 1) / 2)
        weights.append(0.30)

    if focus_components:
        total_w = sum(weights)
        focus_forecast = _clamp(sum(c * w for c, w in zip(focus_components, weights)) / total_w * 100)
    else:
        focus_forecast = 50.0  # neutral default

    # ── Stress risk (0-100) ───────────────────────────────────────────────────
    stress_components: List[float] = []
    stress_weights: List[float] = []

    stress_components.append(1 - hrv_norm)
    stress_weights.append(0.50)

    if voice.get("evening_stress") is not None:
        stress_components.append(float(voice["evening_stress"]))
        stress_weights.append(0.30)

    if sleep_quality is not None:
        poor_sleep_flag = _clamp(1 - sleep_quality / 100, 0.0, 1.0)
        stress_components.append(poor_sleep_flag)
        stress_weights.append(0.20)

    total_sw = sum(stress_weights)
    stress_risk = _clamp(sum(c * w for c, w in zip(stress_components, stress_weights)) / total_sw * 100)

    # ── Dominant mood ─────────────────────────────────────────────────────────
    dominant_mood: Optional[str] = voice.get("dominant_emotion")
    mood_valence: Optional[float] = voice.get("avg_valence")

    # ── Recommended action ────────────────────────────────────────────────────
    if stress_risk > 65:
        recommended_action = "Take a 5-minute breathing break before starting work"
    elif focus_forecast < 45:
        recommended_action = "Start with simple tasks — today's focus forecast is low"
    elif sleep_quality is not None and sleep_quality < 60:
        recommended_action = "Short nap (20 min) may help recover from poor sleep"
    elif focus_forecast > 75:
        recommended_action = "Good time for deep work — block 90 minutes this morning"
    else:
        recommended_action = "Steady day ahead — maintain regular breaks"

    # ── Peak focus window ─────────────────────────────────────────────────────
    peak_focus_window = _peak_focus_from_arousal(user_id)

    # ── Insight ───────────────────────────────────────────────────────────────
    if not voice and not health:
        insight = "Complete a voice check-in or sync health data for a personalised report"
    elif voice and not health:
        n = voice["count"]
        mood = voice.get("dominant_emotion", "neutral")
        insight = f"Yesterday's {n} voice check-in{'s' if n != 1 else ''} showed {mood} as the dominant mood"
    elif health and not voice:
        insight = f"Sleep efficiency {sleep_quality:.0f}%" if sleep_quality else "Health data synced — add voice check-ins for mood insights"
    else:
        v = voice.get("avg_valence", 0.0)
        polarity = "positive" if v > 0.1 else "negative" if v < -0.1 else "neutral"
        insight = f"Yesterday's voice mood was {polarity}; focus forecast today is {focus_forecast:.0f}/100"

    return BrainReport(
        user_id=user_id,
        date=today,
        data_sources=data_sources,
        sleep_quality=round(sleep_quality, 1) if sleep_quality is not None else None,
        focus_forecast=round(focus_forecast, 1),
        stress_risk=round(stress_risk, 1),
        dominant_mood=dominant_mood,
        mood_valence=round(mood_valence, 3) if mood_valence is not None else None,
        recommended_action=recommended_action,
        peak_focus_window=peak_focus_window,
        insight=insight,
        has_eeg=has_eeg,
    )
