"""Sleep-to-next-day-mood predictor.

Forecasts next-day emotional state (valence, arousal, stress risk, focus window)
from last night's sleep metrics.

Research basis:
  - npj Digital Medicine (2024): 36 sleep/circadian features predict mood episodes
    (168 patients, 587 days follow-up)
  - MDPI (2024): HRV during sleep + skin temperature predict next-day state
  - Frontiers Psychiatry (2025): ANN models with digital biomarkers for sleep
  - Established sleep architecture research: Hobson (2005), Dang-Vu et al. (2010)

Model approach:
  Population-level heuristics calibrated to published effect sizes.
  Falls back gracefully when data is missing (unknown → neutral with low confidence).
  After ≥14 days of user data, predictions can be further calibrated to the
  individual (not implemented here — future: personal linear regression).

Usage:
    predictor = SleepMoodPredictor()
    result = predictor.predict_next_day({
        "total_sleep_hours": 7.2,
        "deep_sleep_pct": 0.20,
        "rem_pct": 0.22,
        "sleep_efficiency": 0.88,
        "wake_after_sleep_onset": 15.0,
        "sleep_onset_latency": 12.0,
        "bedtime_regularity": 25.0,
        "hrv_during_sleep": 45.0,
        "resting_hr_during_sleep": 58.0,
        "sleep_debt": 1.5,
    })
"""
from __future__ import annotations

import math
from typing import Optional


# ---------------------------------------------------------------------------
# Population-level reference values
# (from published meta-analyses and normative datasets)
# ---------------------------------------------------------------------------

_REF = {
    # Adults: 7-9h recommended (NSF 2015)
    "total_sleep_hours_optimal": 7.5,
    "total_sleep_hours_min": 6.0,

    # Sleep architecture (% of total sleep time)
    "deep_sleep_pct_optimal": 0.20,    # 20% N3 = healthy
    "deep_sleep_pct_low": 0.10,        # <10% → significant mood risk
    "rem_pct_optimal": 0.22,           # 22% REM = healthy
    "rem_pct_low": 0.15,               # <15% → emotional processing impaired

    # Sleep quality
    "efficiency_good": 0.85,           # >85% is good
    "waso_normal": 20.0,               # minutes of waking after sleep onset
    "sol_normal": 15.0,                # minutes to fall asleep

    # Regularity (std dev of bedtime in minutes)
    "regularity_good": 30.0,           # <30 min variability
    "regularity_poor": 60.0,           # >60 min variability

    # HRV (RMSSD) thresholds — age 25-45 reference
    "hrv_high": 55.0,                  # RMSSD ms — good recovery
    "hrv_low": 25.0,                   # RMSSD ms — poor recovery

    # Resting HR during sleep
    "rhr_optimal": 55.0,               # BPM — endurance athlete baseline
    "rhr_elevated": 70.0,              # BPM — elevated stress indicator

    # Sleep debt (cumulative hours)
    "debt_low": 1.0,                   # ≤1h — manageable
    "debt_high": 3.0,                  # ≥3h — significant impairment
}


# ---------------------------------------------------------------------------
# SleepMoodPredictor
# ---------------------------------------------------------------------------


class SleepMoodPredictor:
    """Predict next-day emotional state from last night's sleep metrics.

    Uses population-level evidence-based heuristics. Each sleep feature
    contributes an independent weight to the outcome scores, then they
    are blended and normalised.

    Attributes:
        None — stateless; all parameters passed per call.
    """

    def predict_next_day(
        self,
        sleep_data: dict,
        user_id: str = "default",
    ) -> dict:
        """Predict next-day valence, arousal, stress risk, and focus window.

        Args:
            sleep_data: Dict with any subset of the following keys:
                total_sleep_hours     — float, hours slept
                deep_sleep_pct        — float, fraction of total sleep in N3 (0-1)
                rem_pct               — float, fraction of total sleep in REM (0-1)
                sleep_efficiency      — float, 0-1 (time asleep / time in bed)
                wake_after_sleep_onset — float, minutes
                sleep_onset_latency   — float, minutes to fall asleep
                bedtime_regularity    — float, std dev of bedtime in minutes
                hrv_during_sleep      — float, mean RMSSD ms
                resting_hr_during_sleep — float, BPM
                sleep_debt            — float, cumulative hours deficit over 7 days
            user_id: Ignored currently (for future personalisation hook).

        Returns:
            Dict with:
                predicted_valence     — float, -1 to 1
                predicted_arousal     — float, 0 to 1
                predicted_stress_risk — float, 0 to 1
                predicted_focus_window — str, e.g. "9:30am – 12:00pm"
                confidence            — float, 0 to 1 (scales with data completeness)
                key_factor            — str, most influential sleep metric
                factors               — list of dicts, all contributing factors ranked
                interpretation        — str, human-readable summary
        """
        scores: dict = {
            "valence": 0.0,
            "arousal": 0.0,
            "stress_risk": 0.0,
        }
        factors = []
        n_available = 0

        def _add(metric_name: str, v_delta: float, a_delta: float, s_delta: float,
                 label: str) -> None:
            """Apply a factor delta and record it."""
            scores["valence"] += v_delta
            scores["arousal"] += a_delta
            scores["stress_risk"] += s_delta
            factors.append({
                "metric": metric_name,
                "label": label,
                "valence_impact": round(v_delta, 3),
                "stress_impact": round(s_delta, 3),
            })

        # ── Total sleep hours ────────────────────────────────────────────
        hours = sleep_data.get("total_sleep_hours")
        if hours is not None:
            n_available += 1
            hours = float(hours)
            if hours >= _REF["total_sleep_hours_optimal"]:
                _add("total_sleep_hours", +0.12, +0.05, -0.10,
                     f"Good sleep duration ({hours:.1f}h)")
            elif hours >= _REF["total_sleep_hours_min"]:
                deficit = (_REF["total_sleep_hours_optimal"] - hours)
                _add("total_sleep_hours", -0.05 * deficit, -0.03 * deficit,
                     +0.05 * deficit, f"Slightly short sleep ({hours:.1f}h)")
            else:
                deficit = (_REF["total_sleep_hours_optimal"] - hours)
                _add("total_sleep_hours", -0.15, -0.10, +0.18,
                     f"Short sleep ({hours:.1f}h — below {_REF['total_sleep_hours_min']}h)")

        # ── Deep sleep (N3) ──────────────────────────────────────────────
        deep = sleep_data.get("deep_sleep_pct")
        if deep is not None:
            n_available += 1
            deep = float(deep)
            if deep >= _REF["deep_sleep_pct_optimal"]:
                _add("deep_sleep_pct", +0.08, +0.03, -0.06,
                     f"Adequate deep sleep ({deep:.0%})")
            elif deep >= _REF["deep_sleep_pct_low"]:
                _add("deep_sleep_pct", -0.04, -0.02, +0.05,
                     f"Low deep sleep ({deep:.0%})")
            else:
                _add("deep_sleep_pct", -0.10, -0.05, +0.12,
                     f"Very low deep sleep ({deep:.0%}) — memory/restoration impaired")

        # ── REM sleep ────────────────────────────────────────────────────
        rem = sleep_data.get("rem_pct")
        if rem is not None:
            n_available += 1
            rem = float(rem)
            if rem >= _REF["rem_pct_optimal"]:
                _add("rem_pct", +0.10, +0.02, -0.08,
                     f"Adequate REM ({rem:.0%}) — emotional memory consolidation")
            elif rem >= _REF["rem_pct_low"]:
                _add("rem_pct", -0.03, -0.01, +0.04,
                     f"Low REM ({rem:.0%})")
            else:
                _add("rem_pct", -0.12, -0.03, +0.15,
                     f"Very low REM ({rem:.0%}) — emotional reactivity expected")

        # ── Sleep efficiency ─────────────────────────────────────────────
        eff = sleep_data.get("sleep_efficiency")
        if eff is not None:
            n_available += 1
            eff = float(eff)
            if eff >= _REF["efficiency_good"]:
                _add("sleep_efficiency", +0.05, +0.01, -0.04,
                     f"Good sleep efficiency ({eff:.0%})")
            else:
                delta = _REF["efficiency_good"] - eff
                _add("sleep_efficiency", -0.08 * delta / 0.15, 0.0,
                     +0.08 * delta / 0.15,
                     f"Poor sleep efficiency ({eff:.0%})")

        # ── Wake after sleep onset ───────────────────────────────────────
        waso = sleep_data.get("wake_after_sleep_onset")
        if waso is not None:
            n_available += 1
            waso = float(waso)
            excess = max(0.0, waso - _REF["waso_normal"])
            if excess > 0:
                impact = min(0.15, excess / 60.0 * 0.15)
                _add("wake_after_sleep_onset", -impact, 0.0, +impact,
                     f"High WASO ({waso:.0f} min)")
            else:
                _add("wake_after_sleep_onset", +0.03, 0.0, -0.02,
                     f"Low WASO ({waso:.0f} min)")

        # ── Sleep onset latency ──────────────────────────────────────────
        sol = sleep_data.get("sleep_onset_latency")
        if sol is not None:
            n_available += 1
            sol = float(sol)
            if sol > 30.0:
                _add("sleep_onset_latency", -0.05, 0.0, +0.07,
                     f"High sleep latency ({sol:.0f} min) — possible pre-sleep stress")
            elif sol < 5.0:
                _add("sleep_onset_latency", -0.02, 0.0, +0.03,
                     f"Very short latency ({sol:.0f} min) — may indicate sleep debt")

        # ── Bedtime regularity ───────────────────────────────────────────
        reg = sleep_data.get("bedtime_regularity")
        if reg is not None:
            n_available += 1
            reg = float(reg)
            if reg <= _REF["regularity_good"]:
                _add("bedtime_regularity", +0.04, 0.0, -0.03,
                     f"Regular bedtime (±{reg:.0f} min)")
            elif reg >= _REF["regularity_poor"]:
                _add("bedtime_regularity", -0.08, -0.02, +0.07,
                     f"Irregular bedtime (±{reg:.0f} min) — circadian disruption")

        # ── HRV during sleep ─────────────────────────────────────────────
        hrv = sleep_data.get("hrv_during_sleep")
        if hrv is not None:
            n_available += 1
            hrv = float(hrv)
            if hrv >= _REF["hrv_high"]:
                _add("hrv_during_sleep", +0.10, +0.03, -0.10,
                     f"High HRV ({hrv:.0f} ms) — good autonomic recovery")
            elif hrv >= _REF["hrv_low"]:
                norm = (hrv - _REF["hrv_low"]) / (_REF["hrv_high"] - _REF["hrv_low"])
                _add("hrv_during_sleep", +0.05 * norm, +0.01 * norm, -0.05 * norm,
                     f"Moderate HRV ({hrv:.0f} ms)")
            else:
                _add("hrv_during_sleep", -0.10, -0.05, +0.12,
                     f"Low HRV ({hrv:.0f} ms) — poor autonomic recovery")

        # ── Resting heart rate ───────────────────────────────────────────
        rhr = sleep_data.get("resting_hr_during_sleep")
        if rhr is not None:
            n_available += 1
            rhr = float(rhr)
            if rhr <= _REF["rhr_optimal"]:
                _add("resting_hr_during_sleep", +0.04, 0.0, -0.04,
                     f"Low resting HR ({rhr:.0f} BPM)")
            elif rhr >= _REF["rhr_elevated"]:
                _add("resting_hr_during_sleep", -0.07, +0.03, +0.08,
                     f"Elevated resting HR ({rhr:.0f} BPM)")

        # ── Sleep debt ───────────────────────────────────────────────────
        debt = sleep_data.get("sleep_debt")
        if debt is not None:
            n_available += 1
            debt = float(debt)
            if debt >= _REF["debt_high"]:
                _add("sleep_debt", -0.15, -0.10, +0.18,
                     f"High sleep debt ({debt:.1f}h)")
            elif debt >= _REF["debt_low"]:
                _add("sleep_debt", -0.05, -0.03, +0.06,
                     f"Moderate sleep debt ({debt:.1f}h)")

        # ── Compute confidence ───────────────────────────────────────────
        max_features = 10
        confidence = min(0.90, 0.30 + 0.065 * n_available)  # 0.30 with 0, 0.95 with 10

        # ── Normalise scores ─────────────────────────────────────────────
        # Raw deltas can stack; scale down to sensible range
        scale_v = max(1.0, abs(scores["valence"]) / 0.80)
        scale_s = max(1.0, abs(scores["stress_risk"]) / 0.80)

        valence = max(-1.0, min(1.0, scores["valence"] / scale_v))
        arousal = max(0.0, min(1.0, 0.5 + scores["arousal"] / 2.0))
        stress_risk = max(0.0, min(1.0, 0.30 + scores["stress_risk"] / scale_s))

        # ── Key factor ───────────────────────────────────────────────────
        if factors:
            key = max(factors, key=lambda f: abs(f["valence_impact"]) + abs(f["stress_impact"]))
            key_factor = key["metric"]
            key_label = key["label"]
        else:
            key_factor = "no_data"
            key_label = "Insufficient sleep data"

        # ── Focus window prediction ──────────────────────────────────────
        focus_window = self._predict_focus_window(
            valence=valence,
            stress_risk=stress_risk,
            total_sleep_hours=sleep_data.get("total_sleep_hours"),
            hrv=sleep_data.get("hrv_during_sleep"),
        )

        # ── Interpretation ───────────────────────────────────────────────
        interpretation = self._interpret(valence, stress_risk, key_label, confidence)

        # Sort factors by absolute impact
        factors_sorted = sorted(
            factors,
            key=lambda f: abs(f["valence_impact"]) + abs(f["stress_impact"]),
            reverse=True,
        )

        return {
            "predicted_valence": round(valence, 3),
            "predicted_arousal": round(arousal, 3),
            "predicted_stress_risk": round(stress_risk, 3),
            "predicted_focus_window": focus_window,
            "confidence": round(confidence, 3),
            "key_factor": key_factor,
            "key_factor_label": key_label,
            "factors": factors_sorted,
            "interpretation": interpretation,
            "n_features_used": n_available,
        }

    def _predict_focus_window(
        self,
        valence: float,
        stress_risk: float,
        total_sleep_hours: Optional[float],
        hrv: Optional[float],
    ) -> str:
        """Predict the best focus window given sleep quality.

        Bad sleep shifts peak focus later; good sleep and high HRV
        give an earlier, longer focus window.
        """
        # Base window: 9:30am to 12:00pm (circadian peak)
        start_hour = 9.5
        end_hour = 12.0

        # Adjust for sleep quality
        if total_sleep_hours is not None:
            if total_sleep_hours < 6.0:
                start_hour += 1.0   # push focus window later
                end_hour += 0.5
            elif total_sleep_hours < 7.0:
                start_hour += 0.5

        if stress_risk > 0.65:
            # High stress → shorter focus window
            end_hour -= 0.5

        if valence < -0.2:
            start_hour += 0.5

        if hrv is not None and hrv >= 50.0:
            # Great HRV → extend window
            end_hour += 0.5

        start_hour = max(7.0, min(12.0, start_hour))
        end_hour = max(start_hour + 1.0, min(15.0, end_hour))

        def _fmt(h: float) -> str:
            hour = int(h)
            mins = int((h - hour) * 60)
            period = "am" if hour < 12 else "pm"
            if hour > 12:
                hour -= 12
            return f"{hour}:{mins:02d}{period}"

        return f"{_fmt(start_hour)} – {_fmt(end_hour)}"

    def _interpret(
        self,
        valence: float,
        stress_risk: float,
        key_label: str,
        confidence: float,
    ) -> str:
        """Generate a human-readable mood forecast."""
        if confidence < 0.35:
            return (
                "Sleep data is limited — mood forecast is a rough estimate. "
                "Log more sleep metrics for better predictions."
            )

        mood = "positive" if valence > 0.15 else ("neutral" if valence > -0.15 else "lower")
        stress = "low" if stress_risk < 0.35 else ("moderate" if stress_risk < 0.60 else "elevated")

        base = (
            f"Based on last night's sleep, expect {mood} mood with {stress} stress risk today."
        )
        detail = f" Key influence: {key_label}."

        if stress_risk > 0.65:
            tip = (
                " Consider a short morning breathing exercise or brief walk "
                "to reduce cortisol before high-demand tasks."
            )
        elif valence > 0.15 and stress_risk < 0.40:
            tip = " Good recovery — schedule demanding creative or analytical work in your focus window."
        else:
            tip = ""

        return base + detail + tip
