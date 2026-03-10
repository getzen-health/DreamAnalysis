"""Context-aware emotion prior adjustment.

Provides Bayesian-style prior adjustments for emotion predictions based on
contextual signals that have no new hardware requirements:
  - Time of day (circadian baseline)
  - Sleep quality (recovery state)
  - Activity level (arousal context)
  - Recent caffeine intake (arousal modifier via supplement tracker)
  - Day of week (weekend vs workday patterns)

Research basis:
  - Circadian valence/arousal curves: Stone et al. (2006), Wilhelm et al. (2011)
  - HRV-sleep-mood coupling: Kahn et al. (2013), CANMAT 2022
  - Caffeine arousal timeline: Nehlig (2010), 30-60 min post-dose peak
  - Post-lunch dip: Monk (2005), Lavie (2001)

Usage:
    prior = ContextPrior()
    ctx = prior.compute_prior(hour=14, sleep_score=60, steps=3000,
                               recent_caffeine=True, day_of_week=2)
    adjusted = prior.adjust_prediction(raw, ctx, confidence=0.55)
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Circadian curve coefficients
# (fitted to Stone et al. 2006 experience sampling data)
# ---------------------------------------------------------------------------
# Valence: gentle rise from 6am to ~10am peak, dip 3pm, recovery evening.
# Arousal: bimodal with peaks near 10am and 6pm.
_VALENCE_PEAK_HOUR = 10.5    # morning peak (valence)
_VALENCE_DIP_HOUR  = 15.5    # post-lunch dip
_AROUSAL_PEAK1     = 10.0    # morning alertness peak
_AROUSAL_PEAK2     = 18.0    # evening second wind
_AROUSAL_TROUGH    = 3.0     # lowest point (early morning)

# Amplitude of circadian modulation (in valence/arousal units: -1 to 1 scale)
_VALENCE_AMPLITUDE = 0.12    # ±0.12 around neutral
_AROUSAL_AMPLITUDE = 0.15    # ±0.15 around neutral


def _circadian_valence(hour: float) -> float:
    """Return circadian valence offset for a given hour of day (0-24).

    Returns a value in [-_VALENCE_AMPLITUDE, +_VALENCE_AMPLITUDE].
    Positive = above neutral mood baseline, negative = below.
    """
    # Weighted combination of two Gaussians: morning peak and dip
    peak_weight  = math.exp(-0.5 * ((hour - _VALENCE_PEAK_HOUR) / 3.0) ** 2)
    dip_weight   = math.exp(-0.5 * ((hour - _VALENCE_DIP_HOUR) / 2.0) ** 2)
    offset = _VALENCE_AMPLITUDE * (peak_weight - 0.6 * dip_weight)
    return max(-_VALENCE_AMPLITUDE, min(_VALENCE_AMPLITUDE, offset))


def _circadian_arousal(hour: float) -> float:
    """Return circadian arousal offset for a given hour of day (0-24).

    Returns a value in [-_AROUSAL_AMPLITUDE, +_AROUSAL_AMPLITUDE].
    Bimodal: peaks at ~10am and ~6pm, trough at ~3am.
    """
    peak1 = math.exp(-0.5 * ((hour - _AROUSAL_PEAK1) / 2.5) ** 2)
    peak2 = math.exp(-0.5 * ((hour - _AROUSAL_PEAK2) / 2.0) ** 2)
    trough = math.exp(-0.5 * ((hour - _AROUSAL_TROUGH) / 2.0) ** 2)
    raw = _AROUSAL_AMPLITUDE * (0.7 * peak1 + 0.5 * peak2 - 0.4 * trough)
    return max(-_AROUSAL_AMPLITUDE, min(_AROUSAL_AMPLITUDE, raw))


def _circadian_stress(hour: float) -> float:
    """Return circadian stress offset.

    Stress tends to be lowest in early morning, builds mid-day (cortisol
    awakening response peaks 30-45 min after waking), drops in evening.
    """
    # Rough cosine: low at night, moderate morning, peak mid-afternoon
    # Using hour 13 (1pm) as approximate daily stress peak
    phase = 2 * math.pi * (hour - 13.0) / 24.0
    # Invert cosine so peak is around 13h
    raw = -0.08 * math.cos(phase)
    return max(-0.10, min(0.10, raw))


# ---------------------------------------------------------------------------
# ContextPrior
# ---------------------------------------------------------------------------


class ContextPrior:
    """Bayesian-style context prior for emotion adjustment.

    Computes expected emotion offsets from contextual signals, then
    blends them into raw model predictions weighted by confidence.
    When confidence is high (>0.8), the raw prediction dominates.
    When confidence is low (<0.4), context contributes up to 60%.
    """

    # Maximum blend weight given to context (at zero confidence)
    _MAX_CONTEXT_WEIGHT = 0.60

    def compute_prior(
        self,
        hour: Optional[float] = None,
        sleep_score: Optional[float] = None,
        steps: Optional[float] = None,
        recent_caffeine: bool = False,
        day_of_week: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> dict:
        """Compute context-derived emotion prior offsets.

        Args:
            hour: Hour of day (0-24). Inferred from timestamp if not given.
            sleep_score: Sleep quality (0-100). None = unknown (neutral).
            steps: Step count so far today. None = unknown (neutral).
            recent_caffeine: True if caffeine consumed in last 2 hours.
            day_of_week: 0=Monday … 6=Sunday. None = unknown.
            timestamp: datetime object; used to infer hour if hour is None.

        Returns:
            Dict with prior offsets:
                valence_offset  — expected valence deviation from neutral
                arousal_offset  — expected arousal deviation from neutral
                stress_offset   — expected stress deviation from neutral
                focus_offset    — expected focus deviation from neutral
                context_weight  — recommended blend weight for this prior
                                  (higher = more confident in context)
                reasons         — list of human-readable reasons applied
        """
        if hour is None and timestamp is not None:
            hour = timestamp.hour + timestamp.minute / 60.0

        # Default to neutral midday if no time information
        if hour is None:
            hour = 12.0

        reasons = []

        # -- Circadian component --
        v_offset = _circadian_valence(hour)
        a_offset = _circadian_arousal(hour)
        s_offset = _circadian_stress(hour)
        f_offset = 0.0

        reasons.append(f"circadian (hour={hour:.1f}h)")

        # Focus has a separate pattern: peaks 9-11am and 2-4pm (ultradian)
        focus_peak1 = math.exp(-0.5 * ((hour - 10.0) / 1.5) ** 2)
        focus_peak2 = math.exp(-0.5 * ((hour - 15.0) / 1.5) ** 2)
        f_offset = 0.12 * (0.6 * focus_peak1 + 0.4 * focus_peak2) - 0.04

        # -- Sleep quality modifier --
        if sleep_score is not None:
            norm_sleep = float(sleep_score) / 100.0  # 0.0 - 1.0
            # Poor sleep (<50/100) lowers valence and raises stress
            if norm_sleep < 0.50:
                delta = 0.10 * (0.50 - norm_sleep) / 0.50  # up to -0.10
                v_offset -= delta
                a_offset -= delta * 0.5
                s_offset += delta
                f_offset -= delta * 0.8
                reasons.append(f"poor sleep ({sleep_score:.0f}/100)")
            elif norm_sleep > 0.75:
                delta = 0.06 * (norm_sleep - 0.75) / 0.25  # up to +0.06
                v_offset += delta
                f_offset += delta
                reasons.append(f"good sleep ({sleep_score:.0f}/100)")

        # -- Activity level modifier --
        if steps is not None:
            steps_f = float(steps)
            if steps_f > 8000:
                # High activity → slight positive valence, lower stress
                v_offset += 0.05
                s_offset -= 0.04
                reasons.append(f"high activity ({steps_f:.0f} steps)")
            elif steps_f < 1500:
                # Sedentary → mild negative valence, mild stress increase
                v_offset -= 0.03
                reasons.append(f"low activity ({steps_f:.0f} steps)")

        # -- Caffeine modifier --
        if recent_caffeine:
            # Caffeine → higher arousal, slight positive valence at moderate dose
            a_offset += 0.12
            f_offset += 0.10
            s_offset += 0.04  # can increase anxiety
            reasons.append("recent caffeine")

        # -- Weekend effect (mild positive valence on Sat/Sun) --
        if day_of_week is not None and day_of_week >= 5:
            v_offset += 0.04
            s_offset -= 0.03
            reasons.append("weekend")

        # -- Compute context weight (0 = ignore context, 1 = trust fully) --
        # Increases with number of available signals (max ~0.65)
        n_signals = sum([
            hour is not None,
            sleep_score is not None,
            steps is not None,
            recent_caffeine,
            day_of_week is not None,
        ])
        context_weight = min(0.65, 0.10 + 0.12 * n_signals)

        return {
            "valence_offset":  round(max(-0.30, min(0.30, v_offset)), 4),
            "arousal_offset":  round(max(-0.25, min(0.25, a_offset)), 4),
            "stress_offset":   round(max(-0.20, min(0.20, s_offset)), 4),
            "focus_offset":    round(max(-0.20, min(0.20, f_offset)), 4),
            "context_weight":  round(context_weight, 3),
            "reasons":         reasons,
        }

    def adjust_prediction(
        self,
        raw_prediction: dict,
        prior: dict,
        confidence: float,
    ) -> dict:
        """Blend raw model prediction with context prior.

        High confidence (>0.8) → raw prediction dominates.
        Low confidence (<0.4) → context contributes up to MAX_CONTEXT_WEIGHT.

        Args:
            raw_prediction: Dict with keys valence, arousal, stress_index,
                            focus_index (and any others, passed through).
            prior: Output of compute_prior().
            confidence: Model prediction confidence (0-1).

        Returns:
            New dict with adjusted values. Adds a ``context_adjustment``
            key with debug information.
        """
        # Clamp confidence
        conf = max(0.0, min(1.0, float(confidence)))

        # Context weight scales linearly from MAX_CONTEXT_WEIGHT at conf=0
        # to 0 at conf=1.0, modulated by prior's own context_weight.
        max_w = self._MAX_CONTEXT_WEIGHT * prior.get("context_weight", 0.5)
        ctx_w = max_w * (1.0 - conf)

        result = dict(raw_prediction)
        adjustments = {}

        _map = {
            "valence":     "valence_offset",
            "arousal":     "arousal_offset",
            "stress_index": "stress_offset",
            "focus_index":  "focus_offset",
        }

        for key, offset_key in _map.items():
            raw_val = float(raw_prediction.get(key, 0.0))
            offset  = float(prior.get(offset_key, 0.0))
            # Blend: (1 - ctx_w) * raw + ctx_w * (raw + offset)
            #      = raw + ctx_w * offset
            adjusted = raw_val + ctx_w * offset
            # Keep in valid range
            if key == "valence":
                adjusted = max(-1.0, min(1.0, adjusted))
            else:
                adjusted = max(0.0, min(1.0, adjusted))
            result[key] = round(adjusted, 4)
            adjustments[key] = round(ctx_w * offset, 4)

        result["context_adjustment"] = {
            "context_weight_applied": round(ctx_w, 3),
            "confidence": round(conf, 3),
            "offsets_applied": adjustments,
            "reasons": prior.get("reasons", []),
        }

        return result

    # ------------------------------------------------------------------
    # Convenience API used by voice_checkin.py
    # ------------------------------------------------------------------

    def get_prior(
        self,
        hour: int,
        sleep_quality: Optional[float] = None,
        steps_today: Optional[int] = None,
        caffeine_logged: bool = False,
        previous_checkin_valence: Optional[float] = None,
    ) -> dict:
        """Thin wrapper around compute_prior() for the voice check-in route.

        Translates the voice check-in's 0-10 sleep_quality scale to the
        internal 0-100 scale and maps field names.  Returns a dict with
        keys: valence_prior, arousal_prior, confidence, adjustments.
        """
        # Convert sleep_quality (0-10) to internal (0-100)
        sleep_score: Optional[float] = None
        if sleep_quality is not None:
            sleep_score = float(sleep_quality) * 10.0

        prior = self.compute_prior(
            hour=float(hour),
            sleep_score=sleep_score,
            steps=float(steps_today) if steps_today is not None else None,
            recent_caffeine=caffeine_logged,
        )

        # Derive valence/arousal priors from neutral (0.0 / 0.5) + offsets
        valence_prior = max(-1.0, min(1.0, prior["valence_offset"]))
        arousal_prior = max(0.0, min(1.0, 0.5 + prior["arousal_offset"]))

        # Blend in momentum from previous check-in (20% weight)
        adjustments = list(prior["reasons"])
        if previous_checkin_valence is not None:
            momentum = 0.20 * float(previous_checkin_valence)
            valence_prior = max(-1.0, min(1.0, valence_prior + momentum))
            adjustments.append(
                f"previous check-in valence {previous_checkin_valence:.2f} — 20% emotional momentum"
            )

        return {
            "valence_prior": round(valence_prior, 4),
            "arousal_prior": round(arousal_prior, 4),
            "confidence": round(prior["context_weight"], 4),
            "adjustments": adjustments,
        }


def blend_with_prior(
    prediction: dict,
    prior: dict,
    prior_weight: float = 0.20,
) -> dict:
    """Blend an ML emotion prediction with contextual priors.

    Parameters
    ----------
    prediction:
        Output dict from VoiceEmotionModel (must contain 'valence' and 'arousal').
    prior:
        Output dict from ContextPrior.get_prior().
    prior_weight:
        How much weight to give the context prior (default 0.20 = 20%).

    Returns
    -------
    Copy of prediction with adjusted valence and arousal, plus:
        context_adjusted: True
        adjustments: list of human-readable reasons
    """
    result = dict(prediction)

    predicted_valence = float(prediction.get("valence", 0.0))
    predicted_arousal = float(prediction.get("arousal", 0.5))
    valence_prior = float(prior.get("valence_prior", 0.0))
    arousal_prior = float(prior.get("arousal_prior", 0.5))

    blended_valence = (1.0 - prior_weight) * predicted_valence + prior_weight * valence_prior
    blended_arousal = (1.0 - prior_weight) * predicted_arousal + prior_weight * arousal_prior

    result["valence"] = float(max(-1.0, min(1.0, blended_valence)))
    result["arousal"] = float(max(0.0, min(1.0, blended_arousal)))
    result["context_adjusted"] = True
    result["adjustments"] = prior.get("adjustments", [])

    return result
