"""Evidence-based Sleep Restriction Therapy (SRT) engine.

Implements the Spielman et al. (1987) CBT-i Sleep Restriction Therapy protocol.
Sleep restriction consolidates sleep by temporarily reducing time in bed to match
actual sleep time, then gradually expanding it as efficiency improves.

Clinical references:
- Spielman AJ et al. (1987) — original SRT protocol
- Morin CM et al. (2006) — CBT-i multicomponent efficacy
- Kyle SD et al. (2014) — digitally-delivered CBT-i outcomes

IMPORTANT: This module is a wellness tool only. It does NOT constitute
medical advice and must not replace consultation with a licensed sleep
clinician, especially for patients with seizure disorders, bipolar disorder,
sleep apnea, or occupational safety concerns.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional


class SleepRestrictionProtocol:
    """Evidence-based Sleep Restriction Therapy (SRT) engine.

    Calculates and adjusts a user's recommended sleep window each week
    based on their rolling sleep efficiency. The core rule:

    - Week 1: time-in-bed = average total sleep time (minimum 5.5 h)
    - Each subsequent week: expand/contract the window by 15 min
      depending on whether sleep efficiency meets thresholds

    Sleep efficiency (SE) = total sleep time / time in bed × 100
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.min_window_hours: float = 5.5
        self.efficiency_expand_threshold: float = 0.85   # SE ≥ 85 % → expand
        self.efficiency_contract_threshold: float = 0.80  # SE < 80 % → contract
        self.window_adjustment_minutes: int = 15

    # ── Core calculations ─────────────────────────────────────────────────────

    def calculate_sleep_efficiency(self, sleep_data: list[dict]) -> float:
        """Calculate mean sleep efficiency over the supplied records.

        Args:
            sleep_data: List of dicts, each with:
                - ``time_asleep_hours``  (float) — actual sleep time
                - ``time_in_bed_hours``  (float) — total time in bed
                Records with zero or missing time-in-bed are skipped.

        Returns:
            Mean sleep efficiency as a fraction (0.0–1.0).
            Returns 0.0 if no valid records are supplied.
        """
        valid: list[float] = []
        for record in sleep_data:
            tib = record.get("time_in_bed_hours", 0.0)
            tst = record.get("time_asleep_hours", 0.0)
            if tib and tib > 0:
                valid.append(min(tst / tib, 1.0))
        return sum(valid) / len(valid) if valid else 0.0

    def get_current_window(self, enrollment_data: dict) -> dict:
        """Return the recommended sleep window for the current program week.

        Args:
            enrollment_data: Dict with:
                - ``baseline_sleep_hours`` (float) — mean TST from first 7 days
                - ``enrolled_at``          (str, ISO-8601) — enrollment timestamp
                - ``weekly_adjustments``   (list[dict], optional) —
                  [{week, adjustment_hours}] cumulative adjustments applied so far

        Returns:
            Dict with keys:
                - ``bedtime``      (str) "HH:MM" in 24-h format
                - ``wake_time``    (str) "HH:MM" in 24-h format
                - ``window_hours`` (float)
                - ``week``         (int) current program week (1–6)
        """
        baseline = float(enrollment_data.get("baseline_sleep_hours", 7.0))
        enrolled_at_raw = enrollment_data.get("enrolled_at", datetime.utcnow().isoformat())

        try:
            enrolled_at = datetime.fromisoformat(enrolled_at_raw.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            enrolled_at = datetime.utcnow()

        days_elapsed = (datetime.utcnow() - enrolled_at.replace(tzinfo=None)).days
        week = min(max(1, days_elapsed // 7 + 1), 6)

        # Starting window: max(baseline, min_window)
        window_hours = max(baseline, self.min_window_hours)

        # Apply any recorded weekly adjustments
        for adj in enrollment_data.get("weekly_adjustments", []):
            window_hours += adj.get("adjustment_hours", 0.0)

        # Clamp to minimum
        window_hours = max(window_hours, self.min_window_hours)

        # Fixed wake time at 06:30 (clinical default — patients keep this constant)
        wake_hour, wake_minute = 6, 30
        wake_minutes_from_midnight = wake_hour * 60 + wake_minute
        bed_minutes_from_midnight = wake_minutes_from_midnight - int(window_hours * 60)

        # Handle midnight crossover
        if bed_minutes_from_midnight < 0:
            bed_minutes_from_midnight += 24 * 60

        bed_hour = bed_minutes_from_midnight // 60
        bed_minute = bed_minutes_from_midnight % 60

        return {
            "bedtime": f"{bed_hour:02d}:{bed_minute:02d}",
            "wake_time": f"{wake_hour:02d}:{wake_minute:02d}",
            "window_hours": round(window_hours, 2),
            "week": week,
        }

    def should_adjust_window(self, efficiency: float, current_week: int) -> str:
        """Decide whether to expand, contract, or maintain the sleep window.

        Protocol rules (applied at week boundary review):
        - SE ≥ 85 %  → expand window by 15 min (sleep is consolidated)
        - SE < 80 %  → contract window by 15 min (still too fragmented)
        - 80 % ≤ SE < 85 % → maintain current window

        Week 1 always maintains (baseline establishment week).

        Args:
            efficiency:   Sleep efficiency as a fraction (0.0–1.0).
            current_week: Current program week number (1–6).

        Returns:
            ``"expand"``, ``"contract"``, or ``"maintain"``.
        """
        if current_week <= 1:
            return "maintain"
        if efficiency >= self.efficiency_expand_threshold:
            return "expand"
        if efficiency < self.efficiency_contract_threshold:
            return "contract"
        return "maintain"

    # ── Behavioral prompts ────────────────────────────────────────────────────

    def get_daily_checkin_prompts(self) -> list[dict]:
        """Return the stimulus control behavioral prompts for the daily check-in.

        These are the core Bootzin (1972) stimulus-control instructions adapted
        for digital self-monitoring. Each prompt has a yes/no response.

        Returns:
            List of dicts with keys: ``id``, ``prompt``, ``weight``.
            ``weight`` is the contribution to the compliance score (sum = 1.0).
        """
        return [
            {
                "id": "bedtime_adherence",
                "prompt": "Did you go to bed at your recommended time?",
                "weight": 0.40,
            },
            {
                "id": "get_up_if_awake",
                "prompt": "Did you get up if you were awake for more than 20 minutes?",
                "weight": 0.35,
            },
            {
                "id": "bed_for_sleep_only",
                "prompt": "Did you use your bed only for sleep (no screens, reading, or worrying)?",
                "weight": 0.25,
            },
        ]

    def calculate_compliance_score(self, checkin_responses: dict[str, bool]) -> float:
        """Calculate weighted compliance score from a set of check-in responses.

        Args:
            checkin_responses: Dict mapping prompt ``id`` → ``True``/``False``.

        Returns:
            Compliance score as a fraction (0.0–1.0).
        """
        prompts = {p["id"]: p["weight"] for p in self.get_daily_checkin_prompts()}
        score = 0.0
        for prompt_id, weight in prompts.items():
            if checkin_responses.get(prompt_id, False):
                score += weight
        return round(score, 4)

    # ── Progress summary ──────────────────────────────────────────────────────

    def get_progress_summary(self, weekly_data: list[dict]) -> dict:
        """Build a 6-week progress summary for the progress chart.

        Args:
            weekly_data: List of weekly summary dicts (up to 6), each with:
                - ``week``               (int)
                - ``sleep_data``         (list[dict]) — raw nightly records
                - ``compliance_scores``  (list[float]) — daily 0–1 scores
                - ``window_hours``       (float) — recommended window that week

        Returns:
            Dict with:
                - ``weeks``               (list[dict]) — per-week stats
                - ``efficiency_trend``    (list[float]) — SE per week
                - ``latency_improvement`` (float) — minutes improvement in SOL
                  from week 1 to latest week
                - ``compliance_rate``     (float) — mean compliance across all weeks
                - ``current_week``        (int)
                - ``total_weeks``         (int) — 6 always
        """
        weeks_out: list[dict] = []
        efficiency_trend: list[float] = []
        all_compliance: list[float] = []
        first_week_latency: Optional[float] = None
        last_week_latency: Optional[float] = None

        for entry in weekly_data[:6]:
            week_num = entry.get("week", 1)
            sleep_data = entry.get("sleep_data", [])
            compliance_scores = entry.get("compliance_scores", [])
            window_hours = entry.get("window_hours", 0.0)

            se = self.calculate_sleep_efficiency(sleep_data)
            efficiency_trend.append(round(se * 100, 1))

            mean_compliance = (
                sum(compliance_scores) / len(compliance_scores)
                if compliance_scores
                else 0.0
            )
            all_compliance.extend(compliance_scores)

            # Sleep onset latency from raw records
            latencies = [
                r.get("sleep_onset_latency_minutes", 0)
                for r in sleep_data
                if r.get("sleep_onset_latency_minutes") is not None
            ]
            mean_latency = sum(latencies) / len(latencies) if latencies else None

            if week_num == 1 and mean_latency is not None:
                first_week_latency = mean_latency
            if mean_latency is not None:
                last_week_latency = mean_latency

            adjustment = self.should_adjust_window(se, week_num)
            weeks_out.append(
                {
                    "week": week_num,
                    "efficiency": round(se * 100, 1),
                    "window_hours": round(window_hours, 2),
                    "compliance": round(mean_compliance * 100, 1),
                    "adjustment": adjustment,
                    "mean_latency_minutes": round(mean_latency, 1) if mean_latency else None,
                }
            )

        latency_improvement: Optional[float] = None
        if first_week_latency is not None and last_week_latency is not None:
            latency_improvement = round(first_week_latency - last_week_latency, 1)

        overall_compliance = (
            round(sum(all_compliance) / len(all_compliance) * 100, 1)
            if all_compliance
            else 0.0
        )

        return {
            "weeks": weeks_out,
            "efficiency_trend": efficiency_trend,
            "latency_improvement": latency_improvement,
            "compliance_rate": overall_compliance,
            "current_week": len(weekly_data),
            "total_weeks": 6,
        }
