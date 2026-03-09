"""Digital phenotyping — EEG + health data fusion for mental wellness tracking.

Fuses EEG session summaries with daily health metrics (steps, sleep,
heart rate, HRV) into a unified mental health phenotype score. Tracks
longitudinal trends over weeks/months.

References:
    JMIR 2025 — 112-study review: EEG + wearable fusion improves depression detection
    JMIR Mental Health 2025 — Multimodal digital phenotyping in MDD validates longitudinally
"""
from typing import Dict, List, Optional

import numpy as np

TREND_LABELS = ["improving", "stable", "declining"]
RISK_LEVELS = ["low", "moderate", "elevated", "high"]


class DigitalPhenotyper:
    """Fuse EEG sessions + health metrics into mental wellness phenotype.

    Computes a daily mental health score (0-100) and weekly/monthly trends.
    """

    def __init__(self):
        self._daily_scores: Dict[str, List[Dict]] = {}

    def compute_daily_score(
        self,
        eeg_summary: Optional[Dict] = None,
        health_data: Optional[Dict] = None,
        user_id: str = "default",
    ) -> Dict:
        """Compute a single day's mental health phenotype score.

        Args:
            eeg_summary: Dict with keys from EEG session aggregation:
                - mean_valence: float (-1 to 1)
                - mean_arousal: float (0 to 1)
                - mean_stress: float (0 to 1)
                - session_count: int
                - mean_faa: float (optional)
            health_data: Dict with daily health metrics:
                - steps: int
                - sleep_hours: float
                - resting_hr: float (bpm)
                - hrv_ms: float (RMSSD in ms)
                - active_energy_kcal: float

        Returns:
            Dict with mental_health_score (0-100), component scores,
            risk_flags, and data quality indicator.
        """
        eeg_score, eeg_quality = self._score_eeg(eeg_summary)
        health_score, health_quality = self._score_health(health_data)

        # Weighted combination based on data availability
        if eeg_quality > 0 and health_quality > 0:
            combined = 0.45 * eeg_score + 0.55 * health_score
            quality = "full"
        elif eeg_quality > 0:
            combined = eeg_score
            quality = "eeg_only"
        elif health_quality > 0:
            combined = health_score
            quality = "health_only"
        else:
            combined = 50.0
            quality = "no_data"

        score = float(np.clip(combined, 0, 100))

        # Risk flags
        risk_flags = self._assess_risks(eeg_summary, health_data)

        # Risk level
        if score >= 70:
            risk_level = "low"
        elif score >= 50:
            risk_level = "moderate"
        elif score >= 30:
            risk_level = "elevated"
        else:
            risk_level = "high"

        result = {
            "mental_health_score": round(score, 1),
            "eeg_component": round(eeg_score, 1),
            "health_component": round(health_score, 1),
            "risk_level": risk_level,
            "risk_flags": risk_flags,
            "data_quality": quality,
        }

        # Store for trend analysis
        if user_id not in self._daily_scores:
            self._daily_scores[user_id] = []
        self._daily_scores[user_id].append(result)
        if len(self._daily_scores[user_id]) > 365:
            self._daily_scores[user_id] = self._daily_scores[user_id][-365:]

        return result

    def compute_trend(self, user_id: str = "default", window: int = 7) -> Dict:
        """Compute trend over a rolling window.

        Args:
            user_id: User identifier.
            window: Number of days for trend (default 7).

        Returns:
            Dict with trend label, slope, mean score, and score history.
        """
        scores = self._daily_scores.get(user_id, [])
        if len(scores) < 2:
            return {
                "trend": "stable",
                "slope": 0.0,
                "mean_score": scores[0]["mental_health_score"] if scores else 50.0,
                "n_days": len(scores),
            }

        recent = scores[-window:]
        values = [s["mental_health_score"] for s in recent]

        # Linear regression slope
        x = np.arange(len(values))
        slope = float(np.polyfit(x, values, 1)[0])

        if slope > 1.0:
            trend = "improving"
        elif slope < -1.0:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": round(slope, 2),
            "mean_score": round(float(np.mean(values)), 1),
            "min_score": round(float(np.min(values)), 1),
            "max_score": round(float(np.max(values)), 1),
            "n_days": len(recent),
        }

    def get_monthly_report(self, user_id: str = "default") -> Dict:
        """Get 30-day mental health report."""
        scores = self._daily_scores.get(user_id, [])
        if not scores:
            return {"n_days": 0}

        last_30 = scores[-30:]
        values = [s["mental_health_score"] for s in last_30]
        risk_levels = [s["risk_level"] for s in last_30]

        from collections import Counter
        risk_dist = Counter(risk_levels)

        return {
            "n_days": len(last_30),
            "mean_score": round(float(np.mean(values)), 1),
            "std_score": round(float(np.std(values)), 1),
            "best_day_score": round(float(np.max(values)), 1),
            "worst_day_score": round(float(np.min(values)), 1),
            "trend": self.compute_trend(user_id, window=30)["trend"],
            "risk_distribution": dict(risk_dist),
            "days_at_risk": risk_dist.get("elevated", 0) + risk_dist.get("high", 0),
        }

    def get_scores(self, user_id: str = "default", last_n: Optional[int] = None) -> List[Dict]:
        """Get raw daily scores."""
        scores = self._daily_scores.get(user_id, [])
        if last_n:
            scores = scores[-last_n:]
        return scores

    def reset(self, user_id: str = "default"):
        """Clear user data."""
        self._daily_scores.pop(user_id, None)

    # ── Private scoring helpers ──────────────────────────────────

    def _score_eeg(self, eeg: Optional[Dict]) -> tuple:
        """Score EEG component (0-100)."""
        if not eeg:
            return 50.0, 0

        valence = eeg.get("mean_valence", 0)
        stress = eeg.get("mean_stress", 0.5)
        sessions = eeg.get("session_count", 0)

        # Valence: -1 to 1 → 0 to 100
        valence_score = (valence + 1) / 2 * 100

        # Stress: 0 to 1 → 100 to 0 (inverse)
        stress_score = (1 - stress) * 100

        # Session engagement bonus (up to 10 points for 3+ sessions)
        engagement_bonus = min(sessions * 3.3, 10)

        score = 0.45 * valence_score + 0.40 * stress_score + 0.15 * engagement_bonus
        return float(np.clip(score, 0, 100)), 1

    def _score_health(self, health: Optional[Dict]) -> tuple:
        """Score health component (0-100)."""
        if not health:
            return 50.0, 0

        scores = []

        # Steps: 10k = 100, 5k = 50, 0 = 0
        steps = health.get("steps", 0)
        scores.append(min(steps / 10000 * 100, 100))

        # Sleep: 7-9h optimal
        sleep = health.get("sleep_hours", 0)
        if 7 <= sleep <= 9:
            scores.append(100)
        elif 6 <= sleep < 7 or 9 < sleep <= 10:
            scores.append(70)
        elif 5 <= sleep < 6 or 10 < sleep <= 11:
            scores.append(40)
        else:
            scores.append(20)

        # Resting HR: lower is better (50-60 = 100, >80 = 30)
        hr = health.get("resting_hr", 0)
        if hr > 0:
            hr_score = float(np.clip((90 - hr) / 40 * 100, 0, 100))
            scores.append(hr_score)

        # HRV: higher is better (>50ms = good)
        hrv = health.get("hrv_ms", 0)
        if hrv > 0:
            hrv_score = float(np.clip(hrv / 60 * 100, 0, 100))
            scores.append(hrv_score)

        return float(np.mean(scores)) if scores else 50.0, 1 if scores else 0

    def _assess_risks(self, eeg: Optional[Dict], health: Optional[Dict]) -> List[str]:
        """Identify specific risk flags."""
        flags = []

        if eeg:
            if eeg.get("mean_valence", 0) < -0.3:
                flags.append("persistent_negative_valence")
            if eeg.get("mean_stress", 0) > 0.7:
                flags.append("elevated_stress")

        if health:
            if health.get("sleep_hours", 8) < 5:
                flags.append("severe_sleep_deficit")
            if health.get("steps", 10000) < 2000:
                flags.append("very_low_activity")
            if health.get("resting_hr", 60) > 90:
                flags.append("elevated_resting_hr")

        return flags
