"""Sleep-to-mood predictor — forecast next-day emotional state from sleep metrics.

Science basis: npj Digital Medicine (2024) showed 36 sleep/circadian features
predict mood episodes. Sleep quality is the strongest single predictor of next-day
mood — stronger than any real-time sensor reading.

Key correlations:
- Deep sleep (N3) ↑ → valence ↑ (memory consolidation, emotional processing)
- REM ↑ → emotional resilience ↑ (affective regulation during REM)
- Sleep efficiency < 85% → stress risk ↑
- Sleep debt > 2hrs → arousal ↓, stress ↑
- WASO > 30min → next-day valence ↓
- HRV during sleep ↑ → stress resistance ↑, focus ↑
- Irregular bedtime → mood instability ↑
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# Maximum history entries stored per user
_MAX_HISTORY = 90

# Population reference values used for normalisation
_REF_DEEP_PCT = 0.20     # 20% deep sleep
_REF_REM_PCT = 0.22      # 22% REM sleep
_REF_EFFICIENCY = 0.85   # 85% sleep efficiency
_REF_TOTAL_HOURS = 8.0   # optimal total sleep
_REF_HRV_MS = 50.0       # reference RMSSD in ms


class SleepMoodPredictor:
    """Evidence-based heuristic model: last night's sleep → next-day mood forecast.

    No ML weights needed — all computation is deterministic given the inputs.
    History is stored in-memory (per-user, capped at 90 entries).
    """

    def __init__(self) -> None:
        # user_id → list of prediction records
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # ── Core prediction ────────────────────────────────────────────────────────

    def predict_next_day(
        self,
        sleep_data: Dict[str, Any],
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Predict next-day emotional state from last night's sleep metrics.

        Parameters (all optional — sensible defaults assumed):
            total_sleep_hours        float     hours of sleep (default 7.0)
            deep_sleep_pct           float     N3 fraction 0–1 (default 0.18)
            rem_pct                  float     REM fraction 0–1 (default 0.22)
            sleep_efficiency         float     time-asleep/time-in-bed 0–1 (default 0.85)
            waso_minutes             float     wake-after-sleep-onset in minutes (default 0)
            sleep_onset_latency      float     minutes to fall asleep (default 15)
            bedtime_regularity       float     std-dev of bedtime in hours (default 0.5)
            hrv_ms                   float|None  overnight RMSSD in ms (default None)
            resting_hr_during_sleep  float|None  mean HR during sleep (default None)

        Returns a dict with predicted_valence, predicted_arousal, predicted_stress_risk,
        predicted_focus_score, predicted_focus_window, confidence, key_factor,
        mood_label, sleep_score, and timestamp.
        """
        # ── Extract and bound-check inputs ─────────────────────────────────────
        total_hours = float(sleep_data.get("total_sleep_hours", 7.0) or 7.0)
        deep_pct = float(sleep_data.get("deep_sleep_pct", 0.18) or 0.18)
        rem_pct = float(sleep_data.get("rem_pct", 0.22) or 0.22)
        efficiency = float(sleep_data.get("sleep_efficiency", 0.85) or 0.85)
        waso = float(sleep_data.get("waso_minutes", 0.0) or 0.0)
        sol = float(sleep_data.get("sleep_onset_latency", 15.0) or 15.0)
        regularity = float(sleep_data.get("bedtime_regularity", 0.5) or 0.5)
        hrv_ms: Optional[float] = sleep_data.get("hrv_ms")
        rhr: Optional[float] = sleep_data.get("resting_hr_during_sleep")

        # Clamp to valid physiological ranges
        total_hours = float(np.clip(total_hours, 0.0, 14.0))
        deep_pct = float(np.clip(deep_pct, 0.0, 1.0))
        rem_pct = float(np.clip(rem_pct, 0.0, 1.0))
        efficiency = float(np.clip(efficiency, 0.0, 1.0))
        waso = float(np.clip(waso, 0.0, 300.0))
        sol = float(np.clip(sol, 0.0, 120.0))
        regularity = float(np.clip(regularity, 0.0, 6.0))
        if hrv_ms is not None:
            hrv_ms = float(np.clip(float(hrv_ms), 0.0, 300.0))
        if rhr is not None:
            rhr = float(np.clip(float(rhr), 20.0, 120.0))

        # ── Sleep debt (hours short of 8h reference) ───────────────────────────
        sleep_debt = max(0.0, _REF_TOTAL_HOURS - total_hours)

        # ── Predicted valence  (-1 to 1) ───────────────────────────────────────
        # Contributions:
        #   35% deep sleep above/below reference
        #   25% REM above/below reference
        #   20% sleep efficiency bonus/penalty
        #   20% WASO penalty (capped at 60 min)
        valence_raw = (
            0.35 * (deep_pct / _REF_DEEP_PCT - 1.0)
            + 0.25 * (rem_pct / _REF_REM_PCT - 1.0)
            + 0.20 * (efficiency - _REF_EFFICIENCY) * 4.0
            + 0.20 * (1.0 - min(waso, 60.0) / 60.0)
        )
        predicted_valence = float(np.clip(valence_raw, -1.0, 1.0))

        # ── Predicted arousal (0 to 1) ─────────────────────────────────────────
        # More sleep → sustainable arousal; sleep debt suppresses arousal.
        # HRV bonus when available.
        hrv_contribution = 0.0
        if hrv_ms is not None:
            hrv_contribution = (hrv_ms / _REF_HRV_MS - 1.0) * 0.5

        arousal_raw = (
            0.40 * min(total_hours, 9.0) / 9.0
            + 0.30 * (1.0 - max(0.0, 9.0 - total_hours) / 9.0)
            + 0.30 * hrv_contribution
        )
        predicted_arousal = float(np.clip(arousal_raw, 0.0, 1.0))

        # ── Predicted stress risk (0 to 1) ─────────────────────────────────────
        valence_pos_frac = (predicted_valence + 1.0) / 2.0  # map -1..1 → 0..1
        predicted_stress_risk = float(
            np.clip(1.0 - valence_pos_frac * efficiency, 0.0, 1.0)
        )

        # ── Predicted focus score (0 to 1) ─────────────────────────────────────
        # Good sleep + low stress + low debt = high focus
        debt_penalty = min(sleep_debt / 4.0, 1.0)
        predicted_focus_score = float(
            np.clip(
                (1.0 - predicted_stress_risk) * (1.0 - debt_penalty * 0.5),
                0.0,
                1.0,
            )
        )

        # ── Focus window prediction ────────────────────────────────────────────
        # Circadian default peak: ~90 min after wake (assume 7 am wake).
        # Poor sleep shifts peak or compresses window.
        predicted_focus_window = self._compute_focus_window(
            total_hours=total_hours,
            sleep_debt=sleep_debt,
            efficiency=efficiency,
        )

        # ── Confidence: 0.5–0.9 based on input completeness ───────────────────
        completeness_score = self._compute_confidence(
            hrv_ms=hrv_ms,
            rhr=rhr,
            waso=waso,
            sol=sol,
            sleep_data=sleep_data,
        )

        # ── Key factor: largest absolute contributor ───────────────────────────
        key_factor = self._identify_key_factor(
            deep_pct=deep_pct,
            rem_pct=rem_pct,
            efficiency=efficiency,
            waso=waso,
            total_hours=total_hours,
            sleep_debt=sleep_debt,
            hrv_ms=hrv_ms,
            regularity=regularity,
        )

        # ── Mood label ─────────────────────────────────────────────────────────
        if predicted_valence > 0.15:
            mood_label = "positive"
        elif predicted_valence < -0.15:
            mood_label = "challenging"
        else:
            mood_label = "neutral"

        # ── Overall sleep score (0–100) ────────────────────────────────────────
        sleep_score = self._compute_sleep_score(
            total_hours=total_hours,
            deep_pct=deep_pct,
            rem_pct=rem_pct,
            efficiency=efficiency,
            waso=waso,
            sol=sol,
            regularity=regularity,
        )

        ts = time.time()

        result: Dict[str, Any] = {
            "predicted_valence": round(predicted_valence, 4),
            "predicted_arousal": round(predicted_arousal, 4),
            "predicted_stress_risk": round(predicted_stress_risk, 4),
            "predicted_focus_score": round(predicted_focus_score, 4),
            "predicted_focus_window": predicted_focus_window,
            "confidence": round(completeness_score, 4),
            "key_factor": key_factor,
            "mood_label": mood_label,
            "sleep_score": sleep_score,
            "timestamp": ts,
        }

        # Persist to in-memory history
        history_entry: Dict[str, Any] = {
            "timestamp": ts,
            "inputs": {
                "total_sleep_hours": total_hours,
                "deep_sleep_pct": deep_pct,
                "rem_pct": rem_pct,
                "sleep_efficiency": efficiency,
                "waso_minutes": waso,
                "sleep_onset_latency": sol,
                "bedtime_regularity": regularity,
                "hrv_ms": hrv_ms,
                "resting_hr_during_sleep": rhr,
            },
            "outputs": {k: v for k, v in result.items() if k != "timestamp"},
        }
        user_history = self._history[user_id]
        user_history.append(history_entry)
        if len(user_history) > _MAX_HISTORY:
            self._history[user_id] = user_history[-_MAX_HISTORY:]

        return result

    # ── History retrieval ──────────────────────────────────────────────────────

    def get_history(
        self,
        user_id: str = "default",
        last_n: int = 14,
    ) -> List[Dict[str, Any]]:
        """Return last N prediction records for a user."""
        history = self._history.get(user_id, [])
        return history[-last_n:] if len(history) > last_n else list(history)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_focus_window(
        self,
        total_hours: float,
        sleep_debt: float,
        efficiency: float,
    ) -> str:
        """Return a human-readable best focus window based on sleep quality."""
        # Assume nominal wake time 7 am; first peak ~90 min later.
        # Poor sleep → delayed or shortened window.
        if total_hours >= 7.0 and efficiency >= 0.85 and sleep_debt <= 1.0:
            return "9:00am–12:00pm"
        elif total_hours >= 6.0 and efficiency >= 0.75:
            return "10:00am–1:00pm"
        elif total_hours >= 5.0:
            return "11:00am–1:00pm"
        else:
            # Severely sleep-deprived: afternoon secondary peak sometimes works better
            return "2:00pm–4:00pm"

    def _compute_confidence(
        self,
        hrv_ms: Optional[float],
        rhr: Optional[float],
        waso: float,
        sol: float,
        sleep_data: Dict[str, Any],
    ) -> float:
        """Return confidence 0.5–0.9 based on how many metrics were provided."""
        # Base confidence with just the core four metrics
        score = 0.60
        if hrv_ms is not None:
            score += 0.10   # HRV is a high-value supplementary signal
        if rhr is not None:
            score += 0.05
        if sleep_data.get("waso_minutes") is not None:
            score += 0.05
        if sleep_data.get("sleep_onset_latency") is not None:
            score += 0.05
        if sleep_data.get("bedtime_regularity") is not None:
            score += 0.05
        return float(np.clip(score, 0.5, 0.9))

    def _identify_key_factor(
        self,
        deep_pct: float,
        rem_pct: float,
        efficiency: float,
        waso: float,
        total_hours: float,
        sleep_debt: float,
        hrv_ms: Optional[float],
        regularity: float,
    ) -> str:
        """Return the label of the most impactful sleep factor for this prediction."""
        magnitudes: Dict[str, float] = {}

        # Deep sleep deviation from reference
        magnitudes["deep_sleep"] = abs(deep_pct / _REF_DEEP_PCT - 1.0) * 0.35

        # REM deviation
        magnitudes["rem_sleep"] = abs(rem_pct / _REF_REM_PCT - 1.0) * 0.25

        # Efficiency penalty
        magnitudes["sleep_efficiency"] = abs(efficiency - _REF_EFFICIENCY) * 4.0 * 0.20

        # WASO penalty
        magnitudes["waso"] = (min(waso, 60.0) / 60.0) * 0.20

        # Sleep debt (secondary signal, weighted lower in display)
        magnitudes["sleep_debt"] = min(sleep_debt / 4.0, 1.0) * 0.15

        # HRV when available
        if hrv_ms is not None:
            magnitudes["hrv"] = abs(hrv_ms / _REF_HRV_MS - 1.0) * 0.10

        # Bedtime irregularity
        magnitudes["bedtime_regularity"] = min(regularity / 2.0, 1.0) * 0.10

        key = max(magnitudes, key=lambda k: magnitudes[k])
        _LABELS: Dict[str, str] = {
            "deep_sleep": "Deep sleep (N3) level",
            "rem_sleep": "REM sleep proportion",
            "sleep_efficiency": "Sleep efficiency",
            "waso": "Wake after sleep onset",
            "sleep_debt": "Accumulated sleep debt",
            "hrv": "Overnight HRV",
            "bedtime_regularity": "Bedtime irregularity",
        }
        return _LABELS.get(key, key)

    def _compute_sleep_score(
        self,
        total_hours: float,
        deep_pct: float,
        rem_pct: float,
        efficiency: float,
        waso: float,
        sol: float,
        regularity: float,
    ) -> int:
        """Return an overall sleep quality score from 0 to 100."""
        # Duration score: optimal 7-9 h
        if 7.0 <= total_hours <= 9.0:
            duration_score = 1.0
        elif total_hours < 7.0:
            duration_score = max(0.0, total_hours / 7.0)
        else:
            duration_score = max(0.0, 1.0 - (total_hours - 9.0) / 3.0)

        # Deep sleep score: reference 20%
        deep_score = float(np.clip(deep_pct / _REF_DEEP_PCT, 0.0, 1.0))

        # REM score: reference 22%
        rem_score = float(np.clip(rem_pct / _REF_REM_PCT, 0.0, 1.0))

        # Efficiency score: 1 at 100%, 0 at 70%
        eff_score = float(np.clip((efficiency - 0.70) / 0.30, 0.0, 1.0))

        # WASO score: 1 at 0 min, 0 at 60 min
        waso_score = float(np.clip(1.0 - waso / 60.0, 0.0, 1.0))

        # SOL score: 1 at 0 min, 0 at 45 min
        sol_score = float(np.clip(1.0 - sol / 45.0, 0.0, 1.0))

        # Regularity score: 1 at 0 std, 0 at 2 h std
        reg_score = float(np.clip(1.0 - regularity / 2.0, 0.0, 1.0))

        composite = (
            0.25 * duration_score
            + 0.20 * deep_score
            + 0.20 * rem_score
            + 0.15 * eff_score
            + 0.10 * waso_score
            + 0.05 * sol_score
            + 0.05 * reg_score
        )
        return int(round(float(np.clip(composite * 100.0, 0.0, 100.0))))


# ── Module-level singleton ─────────────────────────────────────────────────────

_predictor: Optional[SleepMoodPredictor] = None


def get_sleep_mood_predictor() -> SleepMoodPredictor:
    """Return (or lazily create) the module-level SleepMoodPredictor singleton."""
    global _predictor
    if _predictor is None:
        _predictor = SleepMoodPredictor()
        log.info("SleepMoodPredictor initialised (evidence-based heuristics)")
    return _predictor
