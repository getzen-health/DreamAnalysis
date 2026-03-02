"""Multimodal Emotion & Dream State Fusion.

Combines EEG + every available biometric signal to maximise emotion and
dream-state prediction accuracy.

Signal contributions (research-backed):
────────────────────────────────────────────────────────────────────
EEG (Muse 2)
  - Valence / FAA                → ±15–25% accuracy gain over HRV alone
  - Arousal / beta-alpha ratio   → real-time, high temporal resolution

HRV / Heart signals (Apple Watch / Polar)
  - SDNN, RMSSD, LF/HF           → autonomic balance, stress, vagal tone
  - Current HR, resting HR       → sympathetic activation proxy
  - HF power (0.15–0.4 Hz)       → parasympathetic (calm ↔ anxious)

Respiratory (Apple Watch, chest strap)
  - Respiratory rate             → anxiety: fast + shallow; calm: slow + deep
  - Breath coherence (during biofeedback) → resonant freq. training
  - HRV coherence ratio          → coherence > 0.8 → parasympathetic dominance

Sleep (Apple Health / Fitbit / Oura)
  - Previous night total sleep   → mood predictor (strongest next-day effect)
  - REM duration                 → emotional memory consolidation
  - Deep sleep (SWS) duration    → cortisol recovery
  - Sleep onset latency          → stress / anxiety marker

Activity (Apple Health / Google Fit)
  - Step count today             → physical activity → serotonin / mood boost
  - Active energy burned         → exercise reduces stress
  - Exercise intensity (METs)    → post-workout endorphin state
  - Movement variance            → restlessness correlates with anxiety

Skin temperature (Apple Watch Ultra / Oura ring)
  - Wrist skin temperature       → slight elevation = stress or illness
  - Temp deviation from baseline → fever, ovulation, emotional arousal

Blood oxygen (SpO2, Apple Watch)
  - SpO2                         → chronic low SpO2 correlates with fatigue / anxiety

Temporal / contextual
  - Time of day                  → circadian: cortisol peaks 30–45 min post-wakeup
  - Day of week                  → weekday vs weekend mood baseline
  - Days since last workout      → detraining → mood degradation
  - Caffeine timing (from food log) → peak caffeine 30–60 min post-ingestion

Food state (project-specific EEG feature)
  - Minutes since last meal      → hungry / post-meal / fasting states differ
  - Glucose load estimate        → high GI → blood sugar spike → mood swing

────────────────────────────────────────────────────────────────────
Expected accuracy improvements (from literature):
  EEG alone (feature heuristics): 60–74%
  + HRV:                          +8–12 pts  → ~78–82%
  + Respiratory rate:             +2–4 pts   → ~80–85%
  + Sleep quality (prev night):   +3–5 pts   → ~83–88%
  + Activity:                     +1–2 pts   → ~84–89%
  + Skin temp deviation:          +1–2 pts   → ~85–90%
  Full multimodal fusion:         ~85–90% arousal; ~75–82% valence
────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


# ── Dataclass for all possible biometric inputs ────────────────────────────

@dataclass
class BiometricSnapshot:
    """All biometric values available at a given moment.

    All fields are Optional — only pass what you have.
    The fusion model gracefully degrades when signals are missing.
    """

    # ── HRV / Heart ──────────────────────────────────────────────────────
    hrv_sdnn: Optional[float] = None          # ms — general autonomic balance
    hrv_rmssd: Optional[float] = None         # ms — parasympathetic (vagal tone)
    hrv_lf_hf_ratio: Optional[float] = None   # LF/HF — sympathovagal balance (>2 = stressed)
    hrv_hf_power: Optional[float] = None      # ms² — HF band power (parasympathetic)
    resting_heart_rate: Optional[float] = None # bpm (from last completed rest period)
    current_heart_rate: Optional[float] = None # bpm (live)
    heart_rate_variability_coherence: Optional[float] = None  # 0–1 (from biofeedback)

    # ── Respiratory ───────────────────────────────────────────────────────
    respiratory_rate: Optional[float] = None  # breaths / min
    breath_coherence: Optional[float] = None  # 0–1 (biofeedback session metric)

    # ── Blood Oxygen ──────────────────────────────────────────────────────
    spo2: Optional[float] = None              # % (95–100 normal; <95 = concerning)

    # ── Skin temperature ─────────────────────────────────────────────────
    skin_temperature_deviation: Optional[float] = None  # °C deviation from personal baseline

    # ── Sleep (previous night) ───────────────────────────────────────────
    sleep_total_hours: Optional[float] = None          # hours
    sleep_rem_hours: Optional[float] = None            # hours REM
    sleep_deep_hours: Optional[float] = None           # hours SWS / N3
    sleep_efficiency: Optional[float] = None           # % (time asleep / time in bed)
    sleep_onset_latency_min: Optional[float] = None   # min to fall asleep
    hours_since_wake: Optional[float] = None           # circadian context

    # ── Physical activity (today so far) ─────────────────────────────────
    steps_today: Optional[float] = None               # step count
    active_energy_kcal: Optional[float] = None        # kcal burned (active)
    exercise_minutes_today: Optional[float] = None    # minutes
    days_since_last_workout: Optional[float] = None   # 0 = today

    # ── Nutrition / food state ────────────────────────────────────────────
    minutes_since_last_meal: Optional[float] = None   # 0–720+ min
    estimated_glucose_load: Optional[float] = None    # arbitrary 0–100 scale

    # ── Contextual ────────────────────────────────────────────────────────
    hour_of_day: Optional[float] = None               # 0–23
    day_of_week: Optional[float] = None               # 0=Mon, 6=Sun
    caffeine_minutes_ago: Optional[float] = None      # time since last caffeine intake


# ── Fusion model ──────────────────────────────────────────────────────────

class MultimodalEmotionFusion:
    """Fuse EEG emotion output with all available biometric signals.

    Usage::

        from ml.models.multimodal_emotion_fusion import (
            MultimodalEmotionFusion, BiometricSnapshot
        )

        fusion = MultimodalEmotionFusion()

        eeg_result = emotion_model.predict(eeg_array, fs=256)
        bio = BiometricSnapshot(
            hrv_sdnn=42.0,
            hrv_rmssd=35.0,
            resting_heart_rate=63.0,
            current_heart_rate=72.0,
            respiratory_rate=15.0,
            sleep_total_hours=7.2,
            sleep_rem_hours=1.8,
            steps_today=4500,
            hours_since_wake=3.5,
            hour_of_day=10.5,
        )

        result = fusion.fuse(eeg_result, bio)
        # → { stress_index, valence, arousal, emotion, dream_readiness, ... }
    """

    # Weights for each signal layer — sum per dimension need not equal 1
    # because we normalise at the end.
    _EEG_WEIGHT_STRESS  = 0.60
    _HRV_WEIGHT_STRESS  = 0.25
    _RESP_WEIGHT_STRESS = 0.08
    _SLEEP_WEIGHT_STRESS = 0.07

    _EEG_WEIGHT_VALENCE  = 0.55
    _HRV_WEIGHT_VALENCE  = 0.25
    _SLEEP_WEIGHT_VALENCE = 0.12
    _ACTIVITY_WEIGHT_VALENCE = 0.08

    _EEG_WEIGHT_AROUSAL  = 0.50
    _HRV_WEIGHT_AROUSAL  = 0.30
    _RESP_WEIGHT_AROUSAL = 0.12
    _TEMP_WEIGHT_AROUSAL = 0.08

    # ── Public interface ──────────────────────────────────────────────────

    def fuse(
        self,
        eeg_result: Dict[str, Any],
        bio: BiometricSnapshot,
    ) -> Dict[str, Any]:
        """Return fused emotion + biometric state dict.

        Parameters
        ----------
        eeg_result:
            Output of ``EmotionClassifier.predict()`` — must contain at least
            ``stress_index``, ``valence``, ``arousal``.
        bio:
            Snapshot of all available biometric readings.

        Returns
        -------
        dict with keys: stress_index, valence, arousal, emotion,
        focus_index, relaxation_index, dream_readiness, sleep_debt,
        circadian_phase, biometric_confidence, signal_count, notes.
        """
        eeg_stress  = float(eeg_result.get("stress_index", 0.5))
        eeg_valence = float(eeg_result.get("valence", 0.0))
        eeg_arousal = float(eeg_result.get("arousal", 0.5))
        eeg_focus   = float(eeg_result.get("focus_index", 0.5))
        eeg_relax   = float(eeg_result.get("relaxation_index", 0.5))
        eeg_emotion = eeg_result.get("emotion", "neutral")

        signals_used: list[str] = ["eeg"]

        # ── 1. HRV layer ─────────────────────────────────────────────────
        hrv_stress, hrv_arousal, hrv_valence, hrv_signals = self._hrv_scores(bio)
        signals_used.extend(hrv_signals)

        # ── 2. Respiratory layer ──────────────────────────────────────────
        resp_stress, resp_arousal = self._respiratory_scores(bio)
        if bio.respiratory_rate is not None:
            signals_used.append("respiratory_rate")

        # ── 3. Sleep layer ────────────────────────────────────────────────
        sleep_stress, sleep_valence, sleep_debt = self._sleep_scores(bio)
        if bio.sleep_total_hours is not None:
            signals_used.append("sleep")

        # ── 4. Activity layer ─────────────────────────────────────────────
        activity_valence, activity_arousal = self._activity_scores(bio)
        if bio.steps_today is not None or bio.active_energy_kcal is not None:
            signals_used.append("activity")

        # ── 5. Skin temperature layer ─────────────────────────────────────
        temp_arousal = self._temperature_scores(bio)
        if bio.skin_temperature_deviation is not None:
            signals_used.append("skin_temperature")

        # ── 6. Contextual / circadian layer ──────────────────────────────
        circadian_phase, circadian_valence = self._circadian_scores(bio)
        if bio.hour_of_day is not None:
            signals_used.append("circadian")

        # ── 7. Food / metabolic layer ─────────────────────────────────────
        food_stress, food_valence = self._food_scores(bio)
        if bio.minutes_since_last_meal is not None:
            signals_used.append("food_state")

        # ── Fuse stress ───────────────────────────────────────────────────
        stress_index = self._weighted_avg([
            (eeg_stress,   self._EEG_WEIGHT_STRESS),
            (hrv_stress,   self._HRV_WEIGHT_STRESS),
            (resp_stress,  self._RESP_WEIGHT_STRESS),
            (sleep_stress, self._SLEEP_WEIGHT_STRESS),
            (food_stress,  0.05),
        ])

        # ── Fuse valence ──────────────────────────────────────────────────
        # valence is −1 to +1; convert to 0–1 for weighted_avg then back
        eeg_v01    = (eeg_valence + 1) / 2
        hrv_v01    = (hrv_valence + 1) / 2
        sleep_v01  = (sleep_valence + 1) / 2
        circ_v01   = (circadian_valence + 1) / 2
        food_v01   = (food_valence + 1) / 2
        act_v01    = (activity_valence + 1) / 2

        valence_01 = self._weighted_avg([
            (eeg_v01,    self._EEG_WEIGHT_VALENCE),
            (hrv_v01,    self._HRV_WEIGHT_VALENCE),
            (sleep_v01,  self._SLEEP_WEIGHT_VALENCE),
            (act_v01,    self._ACTIVITY_WEIGHT_VALENCE),
            (circ_v01,   0.05),
            (food_v01,   0.05),
        ])
        valence = valence_01 * 2 - 1  # back to −1..+1

        # ── Fuse arousal ──────────────────────────────────────────────────
        arousal = self._weighted_avg([
            (eeg_arousal,      self._EEG_WEIGHT_AROUSAL),
            (hrv_arousal,      self._HRV_WEIGHT_AROUSAL),
            (resp_arousal,     self._RESP_WEIGHT_AROUSAL),
            (temp_arousal,     self._TEMP_WEIGHT_AROUSAL),
            (activity_arousal, 0.05),
        ])

        # ── Dream readiness (for sleep-mode sessions) ─────────────────────
        dream_readiness = self._dream_readiness_score(bio, eeg_result)

        # ── Confidence — how many real signals do we have? ────────────────
        n_signals = len(signals_used)
        # 1 signal (EEG only) = 0.40 confidence; full panel (10+) = 1.0
        biometric_confidence = min(1.0, 0.40 + (n_signals - 1) * 0.07)

        return {
            "stress_index":          round(float(stress_index), 3),
            "valence":               round(float(valence), 3),
            "arousal":               round(float(arousal), 3),
            "focus_index":           round(float(eeg_focus), 3),    # EEG-only for now
            "relaxation_index":      round(float(eeg_relax), 3),
            "emotion":               eeg_emotion,
            "sleep_debt":            round(float(sleep_debt), 2),   # hours missing vs 8h
            "circadian_phase":       circadian_phase,                # "morning_peak" etc.
            "dream_readiness":       round(float(dream_readiness), 3),
            "biometric_confidence":  round(float(biometric_confidence), 2),
            "signal_count":          n_signals,
            "signals_used":          signals_used,
            # Preserve EEG model metadata so callers can see which model produced the result
            "model_type":            eeg_result.get("model_type", "multimodal-fusion"),
            "confidence":            eeg_result.get("confidence", round(float(biometric_confidence), 2)),
            "probabilities":         eeg_result.get("probabilities", {}),
        }

    # ── HRV sub-model ─────────────────────────────────────────────────────

    def _hrv_scores(
        self, bio: BiometricSnapshot
    ) -> tuple[float, float, float, list[str]]:
        """Return (stress, arousal, valence, signals_used) from HRV data."""
        stress = 0.5
        arousal = 0.5
        valence_01 = 0.5
        used: list[str] = []
        n = 0

        if bio.hrv_sdnn is not None:
            # SDNN: low → high stress; high → low stress
            # Normal range 50–100 ms.  <30 ms = high stress.
            s = 1.0 - min(1.0, max(0.0, (bio.hrv_sdnn - 20) / 80))
            stress  = (stress * n + s) / (n + 1)
            arousal = (arousal * n + s) / (n + 1)   # low SDNN = high arousal too
            # High SDNN → parasympathetic → positive valence proxy
            v = min(1.0, max(0.0, (bio.hrv_sdnn - 30) / 70))
            valence_01 = (valence_01 * n + v) / (n + 1)
            n += 1
            used.append("hrv_sdnn")

        if bio.hrv_rmssd is not None:
            # RMSSD: sharp parasympathetic marker.  Normal 20–60 ms.
            s = 1.0 - min(1.0, max(0.0, (bio.hrv_rmssd - 15) / 60))
            stress  = (stress * n + s) / (n + 1)
            v = min(1.0, max(0.0, (bio.hrv_rmssd - 20) / 55))
            valence_01 = (valence_01 * n + v) / (n + 1)
            n += 1
            used.append("hrv_rmssd")

        if bio.hrv_lf_hf_ratio is not None:
            # LF/HF > 2 = sympathetically dominant (stressed / aroused)
            ratio = bio.hrv_lf_hf_ratio
            s = min(1.0, max(0.0, (ratio - 0.5) / 4.0))
            stress  = (stress * n + s) / (n + 1)
            arousal = (arousal * n + s) / (n + 1)
            n += 1
            used.append("hrv_lf_hf_ratio")

        if bio.current_heart_rate is not None:
            # >80 bpm → elevated sympathetic tone
            hr = bio.current_heart_rate
            s = min(1.0, max(0.0, (hr - 55) / 55))
            stress  = (stress * n + s) / (n + 1)
            arousal = (arousal * n + s) / (n + 1)
            n += 1
            used.append("current_heart_rate")

        if bio.resting_heart_rate is not None:
            # High resting HR → chronically elevated sympathetic tone
            rhr = bio.resting_heart_rate
            s = min(1.0, max(0.0, (rhr - 55) / 40))
            stress = (stress * n + s) / (n + 1)
            n += 1
            used.append("resting_heart_rate")

        if bio.heart_rate_variability_coherence is not None:
            # High coherence (> 0.8) during biofeedback = deep calm
            coh = bio.heart_rate_variability_coherence
            stress  = (stress * n + (1 - coh)) / (n + 1)
            v = min(1.0, coh)
            valence_01 = (valence_01 * n + v) / (n + 1)
            n += 1
            used.append("hrv_coherence")

        if bio.spo2 is not None:
            # SpO2 < 95% → fatigue / mild hypoxia → slight stress increase
            spo2_stress = max(0.0, (95.0 - bio.spo2) / 5.0)
            stress = (stress * n + min(1.0, stress + spo2_stress * 0.3)) / (n + 1)
            n += 1
            used.append("spo2")

        valence = valence_01 * 2 - 1
        return stress, arousal, valence, used

    # ── Respiratory sub-model ─────────────────────────────────────────────

    def _respiratory_scores(
        self, bio: BiometricSnapshot
    ) -> tuple[float, float]:
        """Return (stress, arousal) from respiratory signals."""
        stress = 0.5
        arousal = 0.5

        if bio.respiratory_rate is not None:
            rr = bio.respiratory_rate
            # Normal rest: 12–20 br/min.
            # Anxiety → >20 (hyperventilation).  Calm → 10–14.
            if rr > 20:
                s = min(1.0, (rr - 20) / 10)
                stress  = min(1.0, stress + s * 0.4)
                arousal = min(1.0, arousal + s * 0.3)
            elif rr < 12:
                # Very slow breathing → deep calm or drowsiness
                stress  = max(0.0, stress - 0.15)
                arousal = max(0.0, arousal - 0.15)

        if bio.breath_coherence is not None:
            # Coherent breathing (≈ 5–6 br/min, in resonant frequency) = calm
            coh = bio.breath_coherence
            stress  = stress * (1 - coh * 0.4)
            arousal = max(0.0, arousal - coh * 0.2)

        return stress, arousal

    # ── Sleep sub-model ───────────────────────────────────────────────────

    def _sleep_scores(
        self, bio: BiometricSnapshot
    ) -> tuple[float, float, float]:
        """Return (stress, valence, sleep_debt_hours)."""
        stress = 0.5
        valence = 0.0
        debt = 0.0

        if bio.sleep_total_hours is not None:
            debt = max(0.0, 8.0 - bio.sleep_total_hours)
            # Each hour of debt → +0.08 stress, −0.12 valence
            stress  = min(1.0, 0.5 + debt * 0.08)
            valence = max(-1.0, 0.0 - debt * 0.12)

        if bio.sleep_rem_hours is not None:
            # REM < 1.5h → emotional processing impaired → lower positive valence
            rem_deficit = max(0.0, 1.5 - bio.sleep_rem_hours)
            valence = max(-1.0, valence - rem_deficit * 0.1)

        if bio.sleep_deep_hours is not None:
            # Deep sleep < 1h → cortisol not fully cleared → higher stress
            deep_deficit = max(0.0, 1.0 - bio.sleep_deep_hours)
            stress = min(1.0, stress + deep_deficit * 0.06)

        if bio.sleep_onset_latency_min is not None:
            # Long latency (> 30 min) → anxiety
            lat = bio.sleep_onset_latency_min
            if lat > 30:
                stress = min(1.0, stress + (lat - 30) / 90 * 0.15)

        return stress, valence, debt

    # ── Activity sub-model ────────────────────────────────────────────────

    def _activity_scores(
        self, bio: BiometricSnapshot
    ) -> tuple[float, float]:
        """Return (valence, arousal) from physical activity signals."""
        valence = 0.0
        arousal = 0.5

        if bio.steps_today is not None:
            # 8000+ steps → good mood boost.  0 steps → slight negative.
            steps = bio.steps_today
            if steps >= 8000:
                valence = min(1.0, valence + 0.20)
            elif steps >= 4000:
                valence = min(1.0, valence + 0.10)
            elif steps < 1000:
                valence = max(-1.0, valence - 0.10)

        if bio.active_energy_kcal is not None:
            # > 300 kcal active energy → endorphin effect → positive valence
            if bio.active_energy_kcal > 300:
                valence = min(1.0, valence + 0.15)
                arousal = min(1.0, arousal + 0.10)

        if bio.days_since_last_workout is not None:
            days = bio.days_since_last_workout
            if days >= 3:
                valence = max(-1.0, valence - 0.08 * min(days - 2, 4))

        return valence, arousal

    # ── Skin temperature sub-model ────────────────────────────────────────

    def _temperature_scores(self, bio: BiometricSnapshot) -> float:
        """Return arousal from skin temperature deviation."""
        if bio.skin_temperature_deviation is None:
            return 0.5
        dev = bio.skin_temperature_deviation
        # Positive deviation (fever / stress) → elevated arousal
        return min(1.0, max(0.0, 0.5 + dev * 0.2))

    # ── Circadian / contextual sub-model ─────────────────────────────────

    def _circadian_scores(
        self, bio: BiometricSnapshot
    ) -> tuple[str, float]:
        """Return (phase_label, valence_modifier)."""
        if bio.hour_of_day is None:
            return "unknown", 0.0

        h = bio.hour_of_day
        if 6 <= h < 10:
            # Cortisol awakening response → highest alertness + slightly higher stress
            return "morning_peak", 0.05
        elif 10 <= h < 13:
            # Late-morning peak: cognitive peak, positive valence
            return "mid_morning", 0.15
        elif 13 <= h < 15:
            # Post-lunch dip: slight negative valence, lower arousal
            return "afternoon_dip", -0.10
        elif 15 <= h < 18:
            # Second wind: motor speed/reaction time peak
            return "afternoon_peak", 0.10
        elif 18 <= h < 21:
            # Evening: melatonin rising, moderate mood
            return "evening", 0.0
        elif h >= 21 or h < 2:
            # Late night: stress if awake, moderate negative valence
            return "late_night", -0.15
        else:
            return "night", -0.05

    # ── Food / metabolic sub-model ────────────────────────────────────────

    def _food_scores(
        self, bio: BiometricSnapshot
    ) -> tuple[float, float]:
        """Return (stress, valence) from food state."""
        stress = 0.0
        valence = 0.0

        if bio.minutes_since_last_meal is not None:
            mins = bio.minutes_since_last_meal
            if mins < 30:
                # Post-meal → slight parasympathetic (fed/satisfied) unless high GI
                stress  = -0.05
                valence = 0.05
            elif 30 <= mins < 90:
                # Blood sugar peak → slightly elevated mood if healthy meal
                valence = 0.10
            elif 90 <= mins < 180:
                # Blood sugar returning to baseline
                valence = 0.0
            elif mins >= 240:
                # Hunger signal → cortisol uptick, mild negative valence
                stress  = 0.10
                valence = -0.10
            if mins >= 480:
                # Extended fast (8+ hours) → significant cortisol → stress
                stress  = 0.20
                valence = -0.15

        if bio.estimated_glucose_load is not None and bio.minutes_since_last_meal is not None:
            # High GI meal → spike + crash → increased mood variability
            gl = bio.estimated_glucose_load
            mins = bio.minutes_since_last_meal
            if gl > 70 and 60 <= mins <= 180:
                # Post-spike phase — likely in crash
                stress  += 0.10
                valence -= 0.10

        if bio.caffeine_minutes_ago is not None:
            caf = bio.caffeine_minutes_ago
            if 30 <= caf <= 90:
                # Peak caffeine → arousal boost, slight anxiety in sensitive people
                valence = valence + 0.05  # net positive for most people

        return stress, valence

    # ── Dream readiness score ─────────────────────────────────────────────

    def _dream_readiness_score(
        self,
        bio: BiometricSnapshot,
        eeg_result: Dict[str, Any],
    ) -> float:
        """Probability that the user is in or approaching REM / dream state.

        Used during sleep sessions to trigger dream detection UI.
        Scale: 0 = definitely awake, 1 = in REM-like state.
        """
        score = 0.0
        n = 0

        # EEG sleep stage hint
        sleep_stage = eeg_result.get("sleep_stage", "wake")
        if sleep_stage == "rem":
            score += 1.0; n += 1
        elif sleep_stage in ("n1", "n2"):
            score += 0.4; n += 1
        elif sleep_stage == "n3":
            score += 0.1; n += 1
        else:
            score += 0.0; n += 1

        # Low current HR → sleep state
        if bio.current_heart_rate is not None:
            if bio.current_heart_rate < 60:
                score += 0.8; n += 1
            elif bio.current_heart_rate < 70:
                score += 0.4; n += 1
            else:
                score += 0.0; n += 1

        # Sleep time context
        if bio.hour_of_day is not None:
            if 22 <= bio.hour_of_day or bio.hour_of_day < 7:
                score += 0.6; n += 1
            else:
                score += 0.0; n += 1

        # High REM duration last night → more dream-prone this session
        if bio.sleep_rem_hours is not None:
            if bio.sleep_rem_hours >= 2.0:
                score += 0.3; n += 1

        return score / max(n, 1)

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _weighted_avg(pairs: list[tuple[float, float]]) -> float:
        """Compute weighted average of (value, weight) pairs, clipped 0–1."""
        total_w = sum(w for _, w in pairs)
        if total_w == 0:
            return 0.5
        result = sum(v * w for v, w in pairs) / total_w
        return min(1.0, max(0.0, result))
