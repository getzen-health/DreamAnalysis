"""Sleep quality predictor — predict next-day cognitive performance.

Combines sleep EEG features (spindle density, slow oscillation power,
sleep efficiency, stage distribution) with health metrics to predict
a restorative sleep quality score and next-day readiness forecast.

Features used:
- N3 percentage (deep sleep fraction)
- Spindle density (consolidation quality)
- Sleep efficiency (time asleep / time in bed)
- Wake after sleep onset (WASO)
- REM percentage
- HRV during sleep (if available)

References:
    Mander et al. (2017) — Sleep quality predicts next-day cognition
    Walker (2017) — Why We Sleep: sleep architecture and performance
"""
from typing import Dict, List, Optional

import numpy as np

QUALITY_LABELS = ["poor", "fair", "good", "excellent"]


class SleepQualityPredictor:
    """Predict sleep quality and next-day cognitive readiness.

    Takes sleep stage distribution and optional health metrics,
    outputs a restorative quality score (0-100) and readiness forecast.
    """

    def __init__(self):
        self._history: Dict[str, List[Dict]] = {}

    def predict(
        self,
        n3_pct: float = 0.0,
        rem_pct: float = 0.0,
        n2_pct: float = 0.0,
        n1_pct: float = 0.0,
        wake_pct: float = 0.0,
        sleep_efficiency: float = 0.85,
        spindle_density: float = 0.0,
        total_sleep_hours: float = 7.0,
        waso_minutes: float = 0.0,
        hrv_ms: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Predict sleep quality and next-day readiness.

        Args:
            n3_pct: Deep sleep percentage (0-1). Optimal: 0.15-0.25.
            rem_pct: REM percentage (0-1). Optimal: 0.20-0.25.
            n2_pct: N2 percentage (0-1).
            n1_pct: N1 percentage (0-1).
            wake_pct: Wake percentage (0-1).
            sleep_efficiency: Time asleep / time in bed (0-1).
            spindle_density: Sleep spindles per minute (0-10).
            total_sleep_hours: Total time asleep.
            waso_minutes: Wake after sleep onset in minutes.
            hrv_ms: HRV RMSSD during sleep (optional).
            user_id: User identifier.

        Returns:
            Dict with quality_score (0-100), quality_label, readiness_forecast,
            component scores, and improvement suggestions.
        """
        # Component scores (0-100 each)
        deep_score = self._score_deep_sleep(n3_pct)
        rem_score = self._score_rem(rem_pct)
        efficiency_score = self._score_efficiency(sleep_efficiency)
        duration_score = self._score_duration(total_sleep_hours)
        continuity_score = self._score_continuity(waso_minutes, total_sleep_hours)
        spindle_score = self._score_spindles(spindle_density)

        # HRV bonus (if available)
        hrv_score = self._score_hrv(hrv_ms) if hrv_ms is not None else None

        # Weighted combination
        if hrv_score is not None:
            quality = (
                0.25 * deep_score + 0.20 * rem_score + 0.15 * efficiency_score
                + 0.15 * duration_score + 0.10 * continuity_score
                + 0.05 * spindle_score + 0.10 * hrv_score
            )
        else:
            quality = (
                0.28 * deep_score + 0.22 * rem_score + 0.18 * efficiency_score
                + 0.17 * duration_score + 0.10 * continuity_score
                + 0.05 * spindle_score
            )

        quality = float(np.clip(quality, 0, 100))

        # Labels
        if quality >= 80:
            label = "excellent"
        elif quality >= 60:
            label = "good"
        elif quality >= 40:
            label = "fair"
        else:
            label = "poor"

        # Readiness forecast
        readiness = self._forecast_readiness(quality, total_sleep_hours)

        # Improvement suggestions
        suggestions = self._get_suggestions(
            deep_score, rem_score, efficiency_score, duration_score,
            continuity_score, total_sleep_hours
        )

        result = {
            "quality_score": round(quality, 1),
            "quality_label": label,
            "readiness_forecast": readiness,
            "components": {
                "deep_sleep": round(deep_score, 1),
                "rem_sleep": round(rem_score, 1),
                "efficiency": round(efficiency_score, 1),
                "duration": round(duration_score, 1),
                "continuity": round(continuity_score, 1),
                "spindle_density": round(spindle_score, 1),
            },
            "suggestions": suggestions,
        }

        if hrv_score is not None:
            result["components"]["hrv"] = round(hrv_score, 1)

        # Store history
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > 365:
            self._history[user_id] = self._history[user_id][-365:]

        return result

    def get_trend(self, user_id: str = "default", window: int = 7) -> Dict:
        """Get sleep quality trend over recent nights."""
        history = self._history.get(user_id, [])
        if len(history) < 2:
            return {"trend": "insufficient_data", "n_nights": len(history)}

        recent = history[-window:]
        scores = [h["quality_score"] for h in recent]
        slope = float(np.polyfit(range(len(scores)), scores, 1)[0])

        if slope > 1.0:
            trend = "improving"
        elif slope < -1.0:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": round(slope, 2),
            "mean_quality": round(float(np.mean(scores)), 1),
            "n_nights": len(recent),
        }

    def get_history(self, user_id: str = "default", last_n: Optional[int] = None) -> List[Dict]:
        """Get quality history."""
        history = self._history.get(user_id, [])
        if last_n:
            history = history[-last_n:]
        return history

    def reset(self, user_id: str = "default"):
        """Clear history."""
        self._history.pop(user_id, None)

    # ── Scoring functions ───────────────────────────────────────

    def _score_deep_sleep(self, n3_pct: float) -> float:
        """Score deep sleep (N3). Optimal: 15-25%."""
        if n3_pct >= 0.15 and n3_pct <= 0.25:
            return 100.0
        elif n3_pct > 0.25:
            return max(100 - (n3_pct - 0.25) * 200, 60)
        else:
            return float(np.clip(n3_pct / 0.15 * 100, 0, 100))

    def _score_rem(self, rem_pct: float) -> float:
        """Score REM sleep. Optimal: 20-25%."""
        if rem_pct >= 0.20 and rem_pct <= 0.25:
            return 100.0
        elif rem_pct > 0.25:
            return max(100 - (rem_pct - 0.25) * 200, 60)
        else:
            return float(np.clip(rem_pct / 0.20 * 100, 0, 100))

    def _score_efficiency(self, efficiency: float) -> float:
        """Score sleep efficiency. Optimal: >85%."""
        return float(np.clip(efficiency / 0.85 * 100, 0, 100))

    def _score_duration(self, hours: float) -> float:
        """Score total sleep duration. Optimal: 7-9 hours."""
        if 7 <= hours <= 9:
            return 100.0
        elif 6 <= hours < 7 or 9 < hours <= 10:
            return 70.0
        elif 5 <= hours < 6:
            return 40.0
        elif hours < 5:
            return 20.0
        else:
            return 60.0

    def _score_continuity(self, waso_min: float, total_hours: float) -> float:
        """Score sleep continuity. Less WASO = better."""
        if total_hours < 0.1:
            return 50.0
        waso_pct = waso_min / (total_hours * 60)
        return float(np.clip((1 - waso_pct * 5) * 100, 0, 100))

    def _score_spindles(self, density: float) -> float:
        """Score spindle density. Optimal: 2-4 per minute."""
        if density >= 2 and density <= 4:
            return 100.0
        elif density > 0:
            return float(np.clip(density / 2 * 100, 0, 100))
        return 50.0  # no spindle data

    def _score_hrv(self, hrv_ms: float) -> float:
        """Score sleep HRV. Higher = better recovery."""
        return float(np.clip(hrv_ms / 60 * 100, 0, 100))

    def _forecast_readiness(self, quality: float, hours: float) -> Dict:
        """Forecast next-day cognitive readiness."""
        if quality >= 80 and hours >= 7:
            return {"level": "high", "focus_estimate": "90%+", "recommendation": "tackle_hard_tasks"}
        elif quality >= 60:
            return {"level": "moderate", "focus_estimate": "70-85%", "recommendation": "normal_workload"}
        elif quality >= 40:
            return {"level": "low", "focus_estimate": "50-70%", "recommendation": "light_tasks_and_nap"}
        else:
            return {"level": "very_low", "focus_estimate": "<50%", "recommendation": "rest_and_recover"}

    def _get_suggestions(self, deep, rem, eff, dur, cont, hours) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        if deep < 50:
            suggestions.append("increase_deep_sleep: avoid alcohol/caffeine after 2pm")
        if rem < 50:
            suggestions.append("increase_rem: maintain consistent wake time")
        if eff < 70:
            suggestions.append("improve_efficiency: only go to bed when sleepy")
        if dur < 60 and hours < 7:
            suggestions.append("extend_sleep: aim for 7-9 hours")
        if cont < 60:
            suggestions.append("reduce_awakenings: keep bedroom cool and dark")
        return suggestions
