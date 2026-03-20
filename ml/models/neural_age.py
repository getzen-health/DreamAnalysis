"""Neural Age Biomarker — EEG-derived emotional brain age as a wellness metric.

Computes a "brain age" from EEG features that reflect emotional and cognitive
neural health, then compares it to chronological age. The Brain Age Gap (BAG)
indicates whether your brain is aging faster or slower than expected.

Key features used:
- Alpha peak frequency: declines ~0.05 Hz/decade after age 30
- Theta/beta ratio: increases with age (reduced executive control)
- Alpha power: declines with age (reduced resting-state coherence)
- Reaction time proxies: beta latency, P300-like features
- Emotional range: narrowing emotional bandwidth correlates with aging

Scientific basis:
- Scally et al. (2018): Alpha peak frequency declines with healthy aging
- Hashemi et al. (2016): Normative EEG database across lifespan
- Cole & Franke (2017): Brain Age Gap as a biomarker for neurodegeneration
- Liem et al. (2017): Lifestyle factors predict Brain Age Gap

DISCLAIMER: This is a wellness indicator only, not a medical device.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


DISCLAIMER = (
    "Neural age estimation is a research-grade wellness indicator based on "
    "EEG spectral features. This is not a medical device. Individual "
    "variation is high. Consult a healthcare professional for medical concerns."
)


# ------------------------------------------------------------------ #
# Dataclasses
# ------------------------------------------------------------------ #


@dataclass
class EEGAgeFeatures:
    """EEG features relevant to neural age estimation.

    All values should be extracted from a resting-state or task-based
    EEG recording of at least 2 minutes duration.
    """

    alpha_peak_freq: float  # Hz, typically 8-13 Hz, declines with age
    theta_beta_ratio: float  # unitless, increases with age
    alpha_power: float  # relative power (0-1), declines with age
    emotional_range: float  # 0-1, breadth of emotional response (valence variance)
    reaction_time_ms: float  # milliseconds, proxy from beta latency

    def validate(self) -> List[str]:
        """Return list of validation warnings (empty if all OK)."""
        warnings = []
        if not 4.0 <= self.alpha_peak_freq <= 16.0:
            warnings.append(
                f"alpha_peak_freq={self.alpha_peak_freq} outside expected range [4-16] Hz"
            )
        if self.theta_beta_ratio < 0:
            warnings.append("theta_beta_ratio must be non-negative")
        if not 0.0 <= self.alpha_power <= 1.0:
            warnings.append(
                f"alpha_power={self.alpha_power} outside expected range [0-1]"
            )
        if not 0.0 <= self.emotional_range <= 1.0:
            warnings.append(
                f"emotional_range={self.emotional_range} outside expected range [0-1]"
            )
        if self.reaction_time_ms < 0:
            warnings.append("reaction_time_ms must be non-negative")
        return warnings


@dataclass
class NeuralAgeProfile:
    """Full neural age assessment with longitudinal tracking."""

    neural_age: float  # estimated brain age in years
    chronological_age: float  # user-reported age in years
    brain_age_gap: float  # neural_age - chronological_age
    gap_interpretation: str  # human-readable interpretation
    aging_rate: Optional[float]  # years of brain age per calendar year (None if < 2 measurements)
    aging_rate_interpretation: Optional[str]
    feature_contributions: Dict[str, float]  # which features push age up/down
    modifiable_factors: List[Dict[str, str]]  # lifestyle factors that affect brain age
    percentile: int  # percentile rank vs population (0-100)
    confidence: float  # 0-1 confidence in the estimate
    warnings: List[str]  # data quality warnings
    disclaimer: str = field(default=DISCLAIMER)


# ------------------------------------------------------------------ #
# Population norms by decade
# ------------------------------------------------------------------ #

# Each tuple: (mean, std) for the given feature in the given decade.
# Sources: composite from Hashemi 2016, Scally 2018, normative databases.

_NORMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "20s": {
        "alpha_peak_freq": (10.5, 0.8),
        "theta_beta_ratio": (1.2, 0.3),
        "alpha_power": (0.35, 0.08),
        "emotional_range": (0.75, 0.12),
        "reaction_time_ms": (250.0, 30.0),
    },
    "30s": {
        "alpha_peak_freq": (10.2, 0.8),
        "theta_beta_ratio": (1.35, 0.35),
        "alpha_power": (0.32, 0.08),
        "emotional_range": (0.70, 0.12),
        "reaction_time_ms": (270.0, 35.0),
    },
    "40s": {
        "alpha_peak_freq": (9.8, 0.9),
        "theta_beta_ratio": (1.5, 0.4),
        "alpha_power": (0.28, 0.07),
        "emotional_range": (0.65, 0.13),
        "reaction_time_ms": (295.0, 40.0),
    },
    "50s": {
        "alpha_peak_freq": (9.4, 0.9),
        "theta_beta_ratio": (1.7, 0.45),
        "alpha_power": (0.24, 0.07),
        "emotional_range": (0.58, 0.14),
        "reaction_time_ms": (320.0, 45.0),
    },
    "60s+": {
        "alpha_peak_freq": (9.0, 1.0),
        "theta_beta_ratio": (1.9, 0.5),
        "alpha_power": (0.20, 0.06),
        "emotional_range": (0.50, 0.15),
        "reaction_time_ms": (355.0, 55.0),
    },
}

# Decade midpoints for interpolation
_DECADE_AGES = {"20s": 25, "30s": 35, "40s": 45, "50s": 55, "60s+": 67}

# Feature weights for age estimation (sum to 1.0)
_FEATURE_WEIGHTS = {
    "alpha_peak_freq": 0.30,
    "theta_beta_ratio": 0.25,
    "alpha_power": 0.20,
    "emotional_range": 0.15,
    "reaction_time_ms": 0.10,
}


# ------------------------------------------------------------------ #
# Modifiable factors affecting brain age
# ------------------------------------------------------------------ #

_MODIFIABLE_FACTORS = [
    {
        "factor": "sleep_quality",
        "label": "Sleep Quality",
        "direction": "Poor sleep accelerates brain aging by 1-3 years",
        "recommendation": "Aim for 7-9 hours of consistent, quality sleep",
    },
    {
        "factor": "exercise",
        "label": "Physical Exercise",
        "direction": "Regular aerobic exercise can reduce brain age by 2-5 years",
        "recommendation": "150 minutes of moderate aerobic exercise per week",
    },
    {
        "factor": "meditation",
        "label": "Meditation Practice",
        "direction": "Regular meditation preserves cortical thickness and alpha power",
        "recommendation": "20 minutes of daily mindfulness or focused-attention meditation",
    },
    {
        "factor": "stress",
        "label": "Chronic Stress",
        "direction": "Sustained stress accelerates neural aging by 2-4 years",
        "recommendation": "Practice stress management techniques; monitor HRV",
    },
    {
        "factor": "social_connection",
        "label": "Social Connection",
        "direction": "Social isolation is associated with accelerated cognitive decline",
        "recommendation": "Maintain regular social interactions and meaningful relationships",
    },
]


# ------------------------------------------------------------------ #
# Core functions
# ------------------------------------------------------------------ #


def _get_norm_value(feature: str, age: float) -> Tuple[float, float]:
    """Interpolate population norm (mean, std) for a feature at a given age.

    Uses linear interpolation between decade midpoints.
    Clamps to boundary decades for ages outside the 20-70 range.
    """
    decades = list(_DECADE_AGES.keys())
    ages = [_DECADE_AGES[d] for d in decades]

    # Clamp
    if age <= ages[0]:
        d = decades[0]
        return _NORMS[d][feature]
    if age >= ages[-1]:
        d = decades[-1]
        return _NORMS[d][feature]

    # Find bracketing decades
    for i in range(len(ages) - 1):
        if ages[i] <= age <= ages[i + 1]:
            t = (age - ages[i]) / (ages[i + 1] - ages[i])
            mean_lo, std_lo = _NORMS[decades[i]][feature]
            mean_hi, std_hi = _NORMS[decades[i + 1]][feature]
            mean_interp = mean_lo + t * (mean_hi - mean_lo)
            std_interp = std_lo + t * (std_hi - std_lo)
            return (mean_interp, std_interp)

    # Fallback (should never reach here)
    return _NORMS["40s"][feature]


def _feature_to_age(feature_name: str, value: float) -> float:
    """Map a single feature value to an estimated age using inverse norm lookup.

    For each decade, compute how well the value matches the population norm.
    Use weighted centroid of decade ages based on Gaussian likelihood.
    """
    decades = list(_DECADE_AGES.keys())
    likelihoods = []

    for d in decades:
        mean, std = _NORMS[d][feature_name]
        if std < 1e-10:
            std = 1e-10
        z = (value - mean) / std
        likelihood = float(np.exp(-0.5 * z * z))
        likelihoods.append(likelihood)

    total = sum(likelihoods)
    if total < 1e-10:
        return 45.0  # uninformative -> midpoint

    weighted_age = sum(
        _DECADE_AGES[d] * lik for d, lik in zip(decades, likelihoods)
    )
    return weighted_age / total


def estimate_neural_age(
    features: EEGAgeFeatures,
    chronological_age: Optional[float] = None,
) -> Dict:
    """Estimate neural age from EEG features.

    Uses a weighted ensemble of per-feature age estimates. Each feature
    is mapped to an age via likelihood matching against population norms,
    then combined using predefined weights.

    Args:
        features: EEG age-relevant features extracted from recording.
        chronological_age: User's actual age (for gap calculation).

    Returns:
        Dict with neural_age, per-feature age estimates, confidence,
        brain_age_gap (if chronological_age provided), and warnings.
    """
    warnings = features.validate()

    feature_ages = {}
    feature_values = {
        "alpha_peak_freq": features.alpha_peak_freq,
        "theta_beta_ratio": features.theta_beta_ratio,
        "alpha_power": features.alpha_power,
        "emotional_range": features.emotional_range,
        "reaction_time_ms": features.reaction_time_ms,
    }

    for fname, fval in feature_values.items():
        feature_ages[fname] = _feature_to_age(fname, fval)

    # Weighted combination
    neural_age = sum(
        _FEATURE_WEIGHTS[f] * feature_ages[f] for f in _FEATURE_WEIGHTS
    )
    neural_age = float(np.clip(neural_age, 18.0, 85.0))

    # Confidence: based on how consistent per-feature ages are
    age_values = list(feature_ages.values())
    age_std = float(np.std(age_values))
    # Low std = high agreement = high confidence
    confidence = float(np.clip(1.0 - age_std / 20.0, 0.2, 0.95))

    result = {
        "neural_age": round(neural_age, 1),
        "feature_ages": {k: round(v, 1) for k, v in feature_ages.items()},
        "confidence": round(confidence, 3),
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    }

    if chronological_age is not None:
        gap = round(neural_age - chronological_age, 1)
        result["brain_age_gap"] = gap
        result["gap_interpretation"] = _interpret_gap(gap)

    return result


def compute_brain_age_gap(
    features: EEGAgeFeatures,
    chronological_age: float,
) -> Dict:
    """Compute Brain Age Gap with detailed interpretation.

    BAG = neural_age - chronological_age
    Positive = brain older than body; negative = brain younger.

    Args:
        features: EEG features for neural age estimation.
        chronological_age: User's actual age in years.

    Returns:
        Dict with gap value, severity, interpretation, percentile, and
        per-feature contributions showing which features push age up/down.
    """
    estimation = estimate_neural_age(features, chronological_age)
    neural_age = estimation["neural_age"]
    gap = round(neural_age - chronological_age, 1)

    # Per-feature contribution: how much each feature adds/subtracts
    # relative to chronological-age norm
    contributions = {}
    feature_values = {
        "alpha_peak_freq": features.alpha_peak_freq,
        "theta_beta_ratio": features.theta_beta_ratio,
        "alpha_power": features.alpha_power,
        "emotional_range": features.emotional_range,
        "reaction_time_ms": features.reaction_time_ms,
    }
    for fname, fval in feature_values.items():
        norm_mean, norm_std = _get_norm_value(fname, chronological_age)
        if norm_std < 1e-10:
            norm_std = 1e-10
        z = (fval - norm_mean) / norm_std
        # For features that decrease with age (alpha_peak_freq, alpha_power,
        # emotional_range), a LOWER value means OLDER brain.
        if fname in ("alpha_peak_freq", "alpha_power", "emotional_range"):
            z = -z  # flip: below-norm = positive contribution to age gap
        contributions[fname] = round(float(z * _FEATURE_WEIGHTS[fname] * 5.0), 2)

    # Percentile: using normal distribution with population SD ~6 years for BAG
    from scipy.stats import norm

    percentile = int(norm.cdf(gap, loc=0, scale=6.0) * 100)
    percentile = max(1, min(99, percentile))

    return {
        "neural_age": neural_age,
        "chronological_age": chronological_age,
        "brain_age_gap": gap,
        "gap_interpretation": _interpret_gap(gap),
        "gap_severity": _severity_from_gap(gap),
        "feature_contributions": contributions,
        "percentile": percentile,
        "confidence": estimation["confidence"],
        "feature_ages": estimation["feature_ages"],
        "warnings": estimation["warnings"],
        "disclaimer": DISCLAIMER,
    }


def compute_aging_rate(
    history: List[Dict],
) -> Dict:
    """Compute brain aging rate from longitudinal measurements.

    Takes a list of historical measurements, each containing at minimum:
    - "neural_age": estimated neural age at that time
    - "timestamp": measurement time (unix seconds or ISO string)
      OR "elapsed_days": days since first measurement

    Aging rate = change in neural age / change in calendar time.
    Rate of 1.0 = normal aging. >1.0 = accelerated. <1.0 = decelerated.

    Args:
        history: List of past measurements in chronological order.

    Returns:
        Dict with aging_rate, interpretation, trend data, and
        recommendation.
    """
    if len(history) < 2:
        return {
            "aging_rate": None,
            "interpretation": "Need at least 2 measurements to compute aging rate",
            "data_points": len(history),
            "sufficient_data": False,
        }

    # Extract neural ages and time deltas
    neural_ages = []
    days = []

    for i, entry in enumerate(history):
        neural_ages.append(float(entry["neural_age"]))
        if "elapsed_days" in entry:
            days.append(float(entry["elapsed_days"]))
        elif "timestamp" in entry:
            if i == 0:
                days.append(0.0)
            else:
                dt = float(entry["timestamp"]) - float(history[0]["timestamp"])
                days.append(dt / 86400.0)
        else:
            # Assume equal spacing of 30 days if no time info
            days.append(i * 30.0)

    neural_ages_arr = np.array(neural_ages)
    days_arr = np.array(days)

    # Total time span in years
    time_span_years = (days_arr[-1] - days_arr[0]) / 365.25
    if time_span_years < 0.01:
        return {
            "aging_rate": None,
            "interpretation": "Measurements too close together for rate calculation",
            "data_points": len(history),
            "sufficient_data": False,
            "time_span_days": round(float(days_arr[-1] - days_arr[0]), 1),
        }

    # Linear regression: neural_age = slope * days + intercept
    # slope = change in neural age per day
    coeffs = np.polyfit(days_arr, neural_ages_arr, 1)
    slope_per_day = float(coeffs[0])
    slope_per_year = slope_per_day * 365.25

    # Aging rate: brain years per calendar year
    # Normal = 1.0 (brain ages 1 year per calendar year)
    aging_rate = round(slope_per_year, 3)

    # Residuals for confidence
    predicted = np.polyval(coeffs, days_arr)
    residuals = neural_ages_arr - predicted
    rmse = float(np.sqrt(np.mean(residuals**2)))

    interpretation = _interpret_aging_rate(aging_rate)

    return {
        "aging_rate": aging_rate,
        "interpretation": interpretation,
        "data_points": len(history),
        "time_span_years": round(time_span_years, 2),
        "rmse": round(rmse, 2),
        "trend_neural_ages": [round(float(v), 1) for v in neural_ages_arr],
        "trend_days": [round(float(d), 1) for d in days_arr],
        "sufficient_data": True,
    }


def identify_aging_factors(
    features: EEGAgeFeatures,
    chronological_age: float,
    lifestyle: Optional[Dict] = None,
) -> Dict:
    """Identify which factors are contributing to brain aging.

    Compares each EEG feature against population norms for the user's
    age group and flags areas where the brain appears older than expected.

    Args:
        features: Current EEG features.
        chronological_age: User's actual age.
        lifestyle: Optional dict with keys like "sleep_quality" (0-1),
                   "exercise_hours_weekly", "meditation_minutes_daily",
                   "stress_level" (0-1), "social_hours_weekly".

    Returns:
        Dict with identified aging contributors, protective factors,
        and personalized recommendations.
    """
    feature_values = {
        "alpha_peak_freq": features.alpha_peak_freq,
        "theta_beta_ratio": features.theta_beta_ratio,
        "alpha_power": features.alpha_power,
        "emotional_range": features.emotional_range,
        "reaction_time_ms": features.reaction_time_ms,
    }

    aging_contributors = []
    protective_factors = []

    for fname, fval in feature_values.items():
        norm_mean, norm_std = _get_norm_value(fname, chronological_age)
        if norm_std < 1e-10:
            norm_std = 1e-10
        z = (fval - norm_mean) / norm_std

        # Determine direction: for some features, lower = older
        if fname in ("alpha_peak_freq", "alpha_power", "emotional_range"):
            age_z = -z  # below norm = aging
        else:
            age_z = z  # above norm = aging (theta/beta ratio, reaction time)

        entry = {
            "feature": fname,
            "value": round(fval, 3),
            "population_mean": round(norm_mean, 3),
            "z_score": round(float(z), 2),
            "age_effect_z": round(float(age_z), 2),
        }

        if age_z > 0.5:
            entry["status"] = "aging_contributor"
            entry["description"] = _feature_aging_description(fname, "aging")
            aging_contributors.append(entry)
        elif age_z < -0.5:
            entry["status"] = "protective"
            entry["description"] = _feature_aging_description(fname, "protective")
            protective_factors.append(entry)

    # Sort by impact magnitude
    aging_contributors.sort(key=lambda x: x["age_effect_z"], reverse=True)
    protective_factors.sort(key=lambda x: x["age_effect_z"])

    # Lifestyle-based recommendations
    recommendations = list(_MODIFIABLE_FACTORS)
    if lifestyle:
        recommendations = _personalize_recommendations(lifestyle, recommendations)

    return {
        "aging_contributors": aging_contributors,
        "protective_factors": protective_factors,
        "modifiable_factors": recommendations,
        "summary": _aging_summary(aging_contributors, protective_factors),
    }


def compute_neural_age_profile(
    features: EEGAgeFeatures,
    chronological_age: float,
    history: Optional[List[Dict]] = None,
    lifestyle: Optional[Dict] = None,
) -> NeuralAgeProfile:
    """Compute a complete neural age profile.

    Combines estimation, gap analysis, aging rate (if history available),
    and factor identification into a single comprehensive profile.

    Args:
        features: EEG features for age estimation.
        chronological_age: User's actual age in years.
        history: Optional list of past measurements for aging rate.
        lifestyle: Optional lifestyle factors dict.

    Returns:
        NeuralAgeProfile dataclass with all assessment data.
    """
    gap_result = compute_brain_age_gap(features, chronological_age)

    aging_rate = None
    aging_rate_interpretation = None
    if history and len(history) >= 2:
        rate_result = compute_aging_rate(history)
        if rate_result.get("sufficient_data"):
            aging_rate = rate_result["aging_rate"]
            aging_rate_interpretation = rate_result["interpretation"]

    factors = identify_aging_factors(features, chronological_age, lifestyle)

    from scipy.stats import norm

    percentile = int(
        norm.cdf(gap_result["brain_age_gap"], loc=0, scale=6.0) * 100
    )
    percentile = max(1, min(99, percentile))

    return NeuralAgeProfile(
        neural_age=gap_result["neural_age"],
        chronological_age=chronological_age,
        brain_age_gap=gap_result["brain_age_gap"],
        gap_interpretation=gap_result["gap_interpretation"],
        aging_rate=aging_rate,
        aging_rate_interpretation=aging_rate_interpretation,
        feature_contributions=gap_result["feature_contributions"],
        modifiable_factors=factors["modifiable_factors"],
        percentile=percentile,
        confidence=gap_result["confidence"],
        warnings=gap_result["warnings"],
    )


def profile_to_dict(profile: NeuralAgeProfile) -> Dict:
    """Convert a NeuralAgeProfile to a JSON-serializable dict."""
    return {
        "neural_age": profile.neural_age,
        "chronological_age": profile.chronological_age,
        "brain_age_gap": profile.brain_age_gap,
        "gap_interpretation": profile.gap_interpretation,
        "aging_rate": profile.aging_rate,
        "aging_rate_interpretation": profile.aging_rate_interpretation,
        "feature_contributions": profile.feature_contributions,
        "modifiable_factors": profile.modifiable_factors,
        "percentile": profile.percentile,
        "confidence": profile.confidence,
        "warnings": profile.warnings,
        "disclaimer": profile.disclaimer,
    }


# ------------------------------------------------------------------ #
# Interpretation helpers
# ------------------------------------------------------------------ #


def _interpret_gap(gap: float) -> str:
    """Return human-readable interpretation of Brain Age Gap."""
    if gap <= -5:
        return (
            "Your brain appears significantly younger than your chronological age. "
            "This suggests strong neural health and cognitive reserve."
        )
    if gap <= -2:
        return (
            "Your brain appears somewhat younger than your age. "
            "This is a positive indicator of neural wellness."
        )
    if gap <= 2:
        return (
            "Your brain age is within the typical range for your chronological age."
        )
    if gap <= 5:
        return (
            "Your brain appears somewhat older than your age. "
            "Consider lifestyle adjustments to support neural health."
        )
    return (
        "Your brain appears significantly older than your chronological age. "
        "We recommend reviewing lifestyle factors and consulting a healthcare provider."
    )


def _severity_from_gap(gap: float) -> str:
    """Return severity label from Brain Age Gap."""
    abs_gap = abs(gap)
    if abs_gap <= 2:
        return "normal"
    if abs_gap <= 5:
        return "mild"
    if abs_gap <= 10:
        return "moderate"
    return "significant"


def _interpret_aging_rate(rate: float) -> str:
    """Return human-readable interpretation of aging rate."""
    if rate < 0.5:
        return "Your brain aging has significantly decelerated. Excellent neural maintenance."
    if rate < 0.8:
        return "Your brain is aging slower than average. Good neural health trajectory."
    if rate <= 1.2:
        return "Your brain is aging at a normal rate."
    if rate <= 1.5:
        return "Your brain appears to be aging slightly faster than average. Review lifestyle factors."
    return "Your brain appears to be aging faster than expected. Consider lifestyle changes and medical consultation."


def _feature_aging_description(feature: str, direction: str) -> str:
    """Return description of how a feature contributes to aging."""
    descriptions = {
        ("alpha_peak_freq", "aging"): (
            "Alpha peak frequency is lower than expected, suggesting reduced cortical processing speed."
        ),
        ("alpha_peak_freq", "protective"): (
            "Alpha peak frequency is higher than expected, indicating strong cortical processing speed."
        ),
        ("theta_beta_ratio", "aging"): (
            "Elevated theta/beta ratio suggests reduced executive control, typical of older brains."
        ),
        ("theta_beta_ratio", "protective"): (
            "Low theta/beta ratio indicates strong executive function and cognitive control."
        ),
        ("alpha_power", "aging"): (
            "Reduced alpha power suggests diminished resting-state cortical coherence."
        ),
        ("alpha_power", "protective"): (
            "Strong alpha power indicates healthy resting-state neural synchronization."
        ),
        ("emotional_range", "aging"): (
            "Narrow emotional range may indicate reduced emotional flexibility."
        ),
        ("emotional_range", "protective"): (
            "Broad emotional range suggests preserved emotional flexibility and reactivity."
        ),
        ("reaction_time_ms", "aging"): (
            "Slower neural response times suggest reduced processing efficiency."
        ),
        ("reaction_time_ms", "protective"): (
            "Fast neural response times indicate efficient cognitive processing."
        ),
    }
    return descriptions.get(
        (feature, direction),
        f"{feature} is {'contributing to' if direction == 'aging' else 'protecting against'} neural aging.",
    )


def _aging_summary(
    contributors: List[Dict],
    protective: List[Dict],
) -> str:
    """Generate a one-sentence summary of aging factors."""
    n_aging = len(contributors)
    n_protect = len(protective)

    if n_aging == 0 and n_protect == 0:
        return "All neural markers are within the normal range for your age."
    if n_aging == 0:
        return f"{n_protect} neural marker(s) are performing better than expected for your age."
    if n_protect == 0:
        return f"{n_aging} neural marker(s) suggest accelerated aging."
    return (
        f"{n_aging} marker(s) suggest accelerated aging, "
        f"while {n_protect} marker(s) are performing above expectations."
    )


def _personalize_recommendations(
    lifestyle: Dict,
    base_recommendations: List[Dict],
) -> List[Dict]:
    """Reorder and annotate recommendations based on lifestyle data."""
    recommendations = []
    for rec in base_recommendations:
        entry = dict(rec)
        factor = rec["factor"]

        if factor == "sleep_quality" and "sleep_quality" in lifestyle:
            sq = lifestyle["sleep_quality"]
            if sq < 0.5:
                entry["priority"] = "high"
                entry["note"] = f"Your sleep quality ({sq:.0%}) needs improvement."
            else:
                entry["priority"] = "low"
                entry["note"] = f"Your sleep quality ({sq:.0%}) is adequate."

        elif factor == "exercise" and "exercise_hours_weekly" in lifestyle:
            eh = lifestyle["exercise_hours_weekly"]
            if eh < 2.5:
                entry["priority"] = "high"
                entry["note"] = f"You exercise {eh:.1f}h/week; target is 2.5h+."
            else:
                entry["priority"] = "low"
                entry["note"] = f"Your exercise level ({eh:.1f}h/week) meets guidelines."

        elif factor == "meditation" and "meditation_minutes_daily" in lifestyle:
            mm = lifestyle["meditation_minutes_daily"]
            if mm < 10:
                entry["priority"] = "medium"
                entry["note"] = f"You meditate {mm:.0f} min/day; 20 min recommended."
            else:
                entry["priority"] = "low"
                entry["note"] = f"Your meditation practice ({mm:.0f} min/day) is beneficial."

        elif factor == "stress" and "stress_level" in lifestyle:
            sl = lifestyle["stress_level"]
            if sl > 0.6:
                entry["priority"] = "high"
                entry["note"] = f"Your stress level ({sl:.0%}) is elevated."
            else:
                entry["priority"] = "low"
                entry["note"] = f"Your stress level ({sl:.0%}) is manageable."

        elif factor == "social_connection" and "social_hours_weekly" in lifestyle:
            sh = lifestyle["social_hours_weekly"]
            if sh < 3:
                entry["priority"] = "medium"
                entry["note"] = f"You spend {sh:.1f}h/week socializing; aim for 5h+."
            else:
                entry["priority"] = "low"
                entry["note"] = f"Your social engagement ({sh:.1f}h/week) is healthy."

        recommendations.append(entry)

    # Sort by priority: high > medium > low
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))
    return recommendations
