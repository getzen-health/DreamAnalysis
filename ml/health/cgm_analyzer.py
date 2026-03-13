"""CGM (Continuous Glucose Monitor) Analyzer.

Correlates overnight glucose dynamics with sleep quality metrics.
Glucose readings are expected in mg/dL.

Normal overnight glucose range: 70–120 mg/dL.
Readings above 120 mg/dL are counted as excursions.
"""

import logging
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Glucose range thresholds (mg/dL)
GLUCOSE_LOW = 70
GLUCOSE_TARGET_HIGH = 120  # used for excursion counting and time-in-range
GLUCOSE_HIGH = 140          # red zone for colour-coding


@dataclass
class OvernightGlucoseMetrics:
    """Computed metrics for a single overnight glucose window."""

    mean_glucose: float         # mg/dL — average overnight level
    glucose_std: float          # variability (standard deviation)
    excursion_count: int        # number of readings above 120 mg/dL
    time_in_range_pct: float    # % of readings between 70–120 mg/dL
    min_glucose: float
    max_glucose: float
    readings_count: int


class CGMAnalyzer:
    """Analyze overnight CGM data and correlate it with sleep quality."""

    # ------------------------------------------------------------------ #
    #  Core analysis                                                        #
    # ------------------------------------------------------------------ #

    def analyze_overnight(
        self,
        glucose_readings: List[Dict[str, Any]],
        sleep_start: datetime,
        wake_time: datetime,
    ) -> OvernightGlucoseMetrics:
        """Filter readings to the overnight window and compute metrics.

        Args:
            glucose_readings: List of dicts, each with at least:
                - 'timestamp': datetime or ISO-8601 str
                - 'value': glucose level in mg/dL
            sleep_start: Datetime when the subject went to sleep.
            wake_time: Datetime when the subject woke up.

        Returns:
            OvernightGlucoseMetrics computed from the filtered window.

        Raises:
            ValueError: If no readings fall within the overnight window.
        """
        filtered = self._filter_to_window(glucose_readings, sleep_start, wake_time)

        if not filtered:
            raise ValueError(
                f"No glucose readings found between {sleep_start} and {wake_time}"
            )

        values = [r["value"] for r in filtered]
        n = len(values)

        mean_gl = statistics.mean(values)
        std_gl = statistics.stdev(values) if n > 1 else 0.0
        excursions = sum(1 for v in values if v > GLUCOSE_TARGET_HIGH)
        in_range = sum(1 for v in values if GLUCOSE_LOW <= v <= GLUCOSE_TARGET_HIGH)
        time_in_range_pct = (in_range / n) * 100.0

        return OvernightGlucoseMetrics(
            mean_glucose=round(mean_gl, 2),
            glucose_std=round(std_gl, 2),
            excursion_count=excursions,
            time_in_range_pct=round(time_in_range_pct, 2),
            min_glucose=round(min(values), 2),
            max_glucose=round(max(values), 2),
            readings_count=n,
        )

    # ------------------------------------------------------------------ #
    #  Sleep correlation                                                    #
    # ------------------------------------------------------------------ #

    def correlate_with_sleep(
        self,
        glucose_metrics: OvernightGlucoseMetrics,
        sleep_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Correlate overnight glucose metrics with sleep quality indicators.

        Args:
            glucose_metrics: Precomputed OvernightGlucoseMetrics.
            sleep_data: Dict containing any of:
                - 'deep_sleep_pct'    (float 0–100)
                - 'awakenings'        (int)
                - 'sleep_efficiency'  (float 0–100)
                - 'total_sleep_min'   (float)
                - 'rem_pct'           (float 0–100)

        Returns:
            Dict with three core correlation insights plus a summary score.
        """
        insights: Dict[str, Any] = {}

        deep_sleep_pct = sleep_data.get("deep_sleep_pct")
        awakenings = sleep_data.get("awakenings")
        sleep_efficiency = sleep_data.get("sleep_efficiency")

        # 1. Glucose variability vs deep sleep
        #    Higher variability → less deep sleep
        if deep_sleep_pct is not None:
            # Variability > 20 mg/dL is considered high for overnight
            variability_score = min(1.0, glucose_metrics.glucose_std / 20.0)
            predicted_deep_impact = variability_score * -1.0  # negative = reduces deep
            insights["glucose_variability_vs_deep_sleep"] = {
                "glucose_std_mgdl": glucose_metrics.glucose_std,
                "deep_sleep_pct": deep_sleep_pct,
                "variability_score": round(variability_score, 3),
                "direction": "negative" if variability_score > 0.3 else "neutral",
                "interpretation": (
                    "High glucose variability is associated with reduced deep sleep."
                    if variability_score > 0.3
                    else "Glucose variability within normal range."
                ),
                "predicted_deep_impact": round(predicted_deep_impact, 3),
            }

        # 2. Excursions vs awakenings
        #    Glucose spikes can coincide with micro-arousals / wake events
        if awakenings is not None:
            excursion_awakening_ratio = (
                glucose_metrics.excursion_count / max(awakenings, 1)
                if awakenings > 0
                else float(glucose_metrics.excursion_count > 0)
            )
            insights["excursions_vs_awakenings"] = {
                "excursion_count": glucose_metrics.excursion_count,
                "awakenings": awakenings,
                "excursion_awakening_ratio": round(excursion_awakening_ratio, 3),
                "interpretation": (
                    "Glucose excursions may be contributing to sleep disruptions."
                    if glucose_metrics.excursion_count > 0 and awakenings > 0
                    else "No significant glucose-awakening pattern detected."
                ),
            }

        # 3. Mean glucose vs sleep efficiency
        #    Elevated mean glucose is associated with reduced sleep efficiency
        if sleep_efficiency is not None:
            # 100 mg/dL is a neutral reference; above it we penalise efficiency
            mean_elevation = max(0.0, glucose_metrics.mean_glucose - 100.0)
            efficiency_penalty = min(1.0, mean_elevation / 40.0)  # 140 mg/dL → full penalty
            insights["mean_glucose_vs_sleep_efficiency"] = {
                "mean_glucose_mgdl": glucose_metrics.mean_glucose,
                "sleep_efficiency_pct": sleep_efficiency,
                "mean_elevation_above_100": round(mean_elevation, 2),
                "efficiency_penalty_score": round(efficiency_penalty, 3),
                "interpretation": (
                    "Elevated mean glucose is associated with lower sleep efficiency."
                    if mean_elevation > 10
                    else "Mean glucose within range associated with normal sleep efficiency."
                ),
            }

        # Overall glucose-sleep summary score (0 = poor, 1 = good)
        time_in_range_score = glucose_metrics.time_in_range_pct / 100.0
        variability_penalty = min(1.0, glucose_metrics.glucose_std / 30.0)
        summary_score = round(time_in_range_score * (1.0 - 0.4 * variability_penalty), 3)

        insights["summary"] = {
            "glucose_sleep_score": summary_score,
            "time_in_range_pct": glucose_metrics.time_in_range_pct,
            "assessment": self._score_to_assessment(summary_score),
        }

        return insights

    # ------------------------------------------------------------------ #
    #  Weekly patterns                                                      #
    # ------------------------------------------------------------------ #

    def get_weekly_patterns(
        self,
        daily_metrics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Summarise 7-day glucose/sleep patterns.

        Args:
            daily_metrics: List of up to 7 dicts, each with:
                - 'date'              (str, ISO-8601 date)
                - 'glucose_metrics'   (OvernightGlucoseMetrics or equivalent dict)
                - 'sleep_quality'     (float 0–100, optional)
                - 'sleep_efficiency'  (float 0–100, optional)
                - 'deep_sleep_pct'    (float 0–100, optional)

        Returns:
            Dict with trend arrays, best/worst nights, and a 7-day summary.
        """
        if not daily_metrics:
            return {"error": "No daily metrics provided"}

        variabilities: List[float] = []
        time_in_ranges: List[float] = []
        mean_glucoses: List[float] = []
        sleep_qualities: List[Optional[float]] = []
        dates: List[str] = []

        for day in daily_metrics:
            gm = day.get("glucose_metrics", {})
            # Accept both OvernightGlucoseMetrics dataclass and plain dict
            if isinstance(gm, OvernightGlucoseMetrics):
                std = gm.glucose_std
                tir = gm.time_in_range_pct
                mean_gl = gm.mean_glucose
            else:
                std = gm.get("glucose_std", 0.0)
                tir = gm.get("time_in_range_pct", 0.0)
                mean_gl = gm.get("mean_glucose", 0.0)

            variabilities.append(std)
            time_in_ranges.append(tir)
            mean_glucoses.append(mean_gl)
            sleep_qualities.append(day.get("sleep_quality"))
            dates.append(day.get("date", ""))

        # Identify best night (lowest variability + highest TIR)
        composite_scores = [
            tir - (std / 30.0) * 100
            for tir, std in zip(time_in_ranges, variabilities)
        ]
        best_idx = composite_scores.index(max(composite_scores))
        worst_idx = composite_scores.index(min(composite_scores))

        # Correlation direction (simple linear trend: positive means improving)
        variability_trend = self._linear_trend(variabilities)
        tir_trend = self._linear_trend(time_in_ranges)

        valid_sleep_qualities = [q for q in sleep_qualities if q is not None]
        sleep_glucose_correlation: Optional[float] = None
        if len(valid_sleep_qualities) >= 3:
            # Simple Pearson-style correlation between TIR and sleep quality
            sleep_glucose_correlation = self._pearson_correlation(
                time_in_ranges[: len(valid_sleep_qualities)],
                valid_sleep_qualities,
            )

        return {
            "days_analyzed": len(daily_metrics),
            "dates": dates,
            "variability_trend": {
                "values": variabilities,
                "direction": "improving" if variability_trend < 0 else "worsening",
                "slope": round(variability_trend, 4),
            },
            "time_in_range_trend": {
                "values": time_in_ranges,
                "direction": "improving" if tir_trend > 0 else "worsening",
                "slope": round(tir_trend, 4),
            },
            "mean_glucose_values": mean_glucoses,
            "sleep_quality_values": sleep_qualities,
            "best_night": {
                "date": dates[best_idx] if dates else None,
                "variability": variabilities[best_idx],
                "time_in_range_pct": time_in_ranges[best_idx],
            },
            "worst_night": {
                "date": dates[worst_idx] if dates else None,
                "variability": variabilities[worst_idx],
                "time_in_range_pct": time_in_ranges[worst_idx],
            },
            "sleep_glucose_correlation": (
                round(sleep_glucose_correlation, 3)
                if sleep_glucose_correlation is not None
                else None
            ),
            "weekly_summary": {
                "avg_variability": round(statistics.mean(variabilities), 2),
                "avg_time_in_range_pct": round(statistics.mean(time_in_ranges), 2),
                "avg_mean_glucose": round(statistics.mean(mean_glucoses), 2),
            },
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _filter_to_window(
        self,
        readings: List[Dict[str, Any]],
        start: datetime,
        end: datetime,
    ) -> List[Dict[str, Any]]:
        """Return only readings whose timestamp falls within [start, end]."""
        filtered = []
        for r in readings:
            ts = r.get("timestamp")
            if ts is None:
                continue
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if start <= ts <= end:
                filtered.append({**r, "timestamp": ts})
        return filtered

    @staticmethod
    def _score_to_assessment(score: float) -> str:
        if score >= 0.85:
            return "Excellent — glucose well-controlled overnight"
        if score >= 0.70:
            return "Good — minor glucose variability"
        if score >= 0.50:
            return "Fair — moderate glucose excursions may be affecting sleep"
        return "Poor — high glucose variability likely disrupting sleep quality"

    @staticmethod
    def _linear_trend(values: List[float]) -> float:
        """Return the slope of the best-fit line (positive = upward trend)."""
        n = len(values)
        if n < 2:
            return 0.0
        x = list(range(n))
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
        denominator = sum((xi - x_mean) ** 2 for xi in x)
        return numerator / denominator if denominator != 0 else 0.0

    @staticmethod
    def _pearson_correlation(x: List[float], y: List[float]) -> Optional[float]:
        """Compute Pearson r between two equal-length lists."""
        n = len(x)
        if n < 2 or len(y) != n:
            return None
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        den_x = sum((xi - x_mean) ** 2 for xi in x) ** 0.5
        den_y = sum((yi - y_mean) ** 2 for yi in y) ** 0.5
        if den_x == 0 or den_y == 0:
            return None
        return num / (den_x * den_y)
