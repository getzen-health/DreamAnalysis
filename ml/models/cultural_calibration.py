"""Cultural emotion calibration model (issue #436).

Adapts emotion recognition outputs to account for cultural display rules,
affect valuation differences, and self-report response biases across major
cultural clusters.

Core insight: the *same* facial expression or EEG arousal level maps to
different subjective experiences and social meanings depending on the
person's cultural context.  A restrained smile in East Asia may reflect
the same internal positive state as an effusive grin in Latin America.

References:
- Matsumoto et al. (2008) -- Cultural display rules and emotion regulation
- Tsai (2007) -- Ideal affect: cultural causes and behavioural consequences
- Harzing (2006) -- Response styles in cross-national survey research
- Mesquita & Frijda (1992) -- Cultural variations in emotions
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cultural cluster profiles
# ---------------------------------------------------------------------------

# Each profile encodes empirically-documented tendencies for a broad cultural
# cluster.  These are population-level priors -- individual variation within
# cultures is large.  Values are normalised to [-1, 1] or [0, 1] scales.


@dataclass(frozen=True)
class CulturalProfile:
    """Static cultural-cluster profile."""

    cluster_name: str
    description: str

    # Display rules ----------------------------------------------------------
    # masking_tendency: how much the culture suppresses outward emotion display
    #   0 = fully expressive, 1 = highly masked
    masking_tendency: float

    # expressivity_baseline: average outward expressivity in neutral settings
    #   0 = very restrained, 1 = very expressive
    expressivity_baseline: float

    # negative_suppression: tendency to suppress negative emotions publicly
    #   0 = none, 1 = strong suppression
    negative_suppression: float

    # positive_amplification: tendency to amplify positive display
    #   0 = none, 1 = strong amplification
    positive_amplification: float

    # Affect valuation -------------------------------------------------------
    # ideal_arousal: culturally ideal arousal level (0 = calm, 1 = excited)
    #   East Asian cultures value low-arousal positive (calm, serene)
    #   Western cultures value high-arousal positive (excited, enthusiastic)
    ideal_arousal: float

    # ideal_valence_type: "calm_positive" or "excited_positive"
    ideal_valence_type: str

    # Self-report biases -----------------------------------------------------
    # acquiescence_bias: tendency to agree with statements regardless of
    #   content.  0 = none, 1 = strong
    acquiescence_bias: float

    # extreme_response_style: tendency to use extreme scale endpoints
    #   0 = midpoint tendency, 1 = extreme tendency
    extreme_response_style: float

    # social_desirability_bias: tendency to report socially desirable emotions
    #   0 = none, 1 = strong
    social_desirability_bias: float


# -- 8 cultural cluster constants -------------------------------------------

EAST_ASIAN = CulturalProfile(
    cluster_name="east_asian",
    description="East Asian (China, Japan, Korea, Taiwan)",
    masking_tendency=0.75,
    expressivity_baseline=0.30,
    negative_suppression=0.80,
    positive_amplification=0.15,
    ideal_arousal=0.30,
    ideal_valence_type="calm_positive",
    acquiescence_bias=0.55,
    extreme_response_style=0.20,
    social_desirability_bias=0.65,
)

SOUTH_ASIAN = CulturalProfile(
    cluster_name="south_asian",
    description="South Asian (India, Pakistan, Bangladesh, Sri Lanka)",
    masking_tendency=0.50,
    expressivity_baseline=0.55,
    negative_suppression=0.60,
    positive_amplification=0.45,
    ideal_arousal=0.50,
    ideal_valence_type="calm_positive",
    acquiescence_bias=0.65,
    extreme_response_style=0.40,
    social_desirability_bias=0.60,
)

LATIN_AMERICAN = CulturalProfile(
    cluster_name="latin_american",
    description="Latin American (Mexico, Brazil, Argentina, Colombia, etc.)",
    masking_tendency=0.20,
    expressivity_baseline=0.80,
    negative_suppression=0.25,
    positive_amplification=0.75,
    ideal_arousal=0.75,
    ideal_valence_type="excited_positive",
    acquiescence_bias=0.50,
    extreme_response_style=0.70,
    social_desirability_bias=0.40,
)

NORDIC = CulturalProfile(
    cluster_name="nordic",
    description="Nordic (Sweden, Norway, Denmark, Finland, Iceland)",
    masking_tendency=0.55,
    expressivity_baseline=0.40,
    negative_suppression=0.50,
    positive_amplification=0.20,
    ideal_arousal=0.35,
    ideal_valence_type="calm_positive",
    acquiescence_bias=0.15,
    extreme_response_style=0.25,
    social_desirability_bias=0.30,
)

NORTH_AMERICAN = CulturalProfile(
    cluster_name="north_american",
    description="North American (USA, Canada)",
    masking_tendency=0.30,
    expressivity_baseline=0.65,
    negative_suppression=0.40,
    positive_amplification=0.60,
    ideal_arousal=0.70,
    ideal_valence_type="excited_positive",
    acquiescence_bias=0.30,
    extreme_response_style=0.55,
    social_desirability_bias=0.45,
)

MIDDLE_EASTERN = CulturalProfile(
    cluster_name="middle_eastern",
    description="Middle Eastern (Saudi Arabia, UAE, Iran, Egypt, Turkey, etc.)",
    masking_tendency=0.45,
    expressivity_baseline=0.60,
    negative_suppression=0.55,
    positive_amplification=0.50,
    ideal_arousal=0.55,
    ideal_valence_type="excited_positive",
    acquiescence_bias=0.55,
    extreme_response_style=0.60,
    social_desirability_bias=0.65,
)

SUB_SAHARAN_AFRICAN = CulturalProfile(
    cluster_name="sub_saharan_african",
    description="Sub-Saharan African (Nigeria, Kenya, South Africa, Ghana, etc.)",
    masking_tendency=0.30,
    expressivity_baseline=0.70,
    negative_suppression=0.35,
    positive_amplification=0.65,
    ideal_arousal=0.65,
    ideal_valence_type="excited_positive",
    acquiescence_bias=0.60,
    extreme_response_style=0.65,
    social_desirability_bias=0.50,
)

WESTERN_EUROPEAN = CulturalProfile(
    cluster_name="western_european",
    description="Western European (UK, Germany, France, Netherlands, etc.)",
    masking_tendency=0.40,
    expressivity_baseline=0.50,
    negative_suppression=0.45,
    positive_amplification=0.35,
    ideal_arousal=0.50,
    ideal_valence_type="calm_positive",
    acquiescence_bias=0.20,
    extreme_response_style=0.30,
    social_desirability_bias=0.35,
)

# Lookup table
CULTURAL_PROFILES: Dict[str, CulturalProfile] = {
    "east_asian": EAST_ASIAN,
    "south_asian": SOUTH_ASIAN,
    "latin_american": LATIN_AMERICAN,
    "nordic": NORDIC,
    "north_american": NORTH_AMERICAN,
    "middle_eastern": MIDDLE_EASTERN,
    "sub_saharan_african": SUB_SAHARAN_AFRICAN,
    "western_european": WESTERN_EUROPEAN,
}


# ---------------------------------------------------------------------------
# Data classes for calibration results
# ---------------------------------------------------------------------------


@dataclass
class DisplayRuleCorrection:
    """Result of applying display-rule correction to an emotion reading."""

    original_valence: float
    original_arousal: float
    corrected_valence: float
    corrected_arousal: float
    valence_adjustment: float
    arousal_adjustment: float
    correction_rationale: str


@dataclass
class SelfReportCalibration:
    """Calibrated self-report values correcting for cultural response biases."""

    original_valence: float
    original_arousal: float
    calibrated_valence: float
    calibrated_arousal: float
    bias_corrections_applied: List[str]


@dataclass
class AffectValuation:
    """How well an emotional state matches the culturally ideal affect."""

    ideal_arousal: float
    ideal_valence_type: str
    current_arousal: float
    current_valence: float
    alignment_score: float  # 0-1: how well current state matches cultural ideal
    interpretation: str


@dataclass
class CulturalCalibrationResult:
    """Full cultural calibration output."""

    culture: str
    display_rule_correction: DisplayRuleCorrection
    self_report_calibration: SelfReportCalibration
    affect_valuation: AffectValuation
    adapted_thresholds: Dict[str, float]
    profile_summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def get_cultural_profile(culture: str) -> CulturalProfile:
    """Retrieve a cultural profile by cluster name.

    Parameters
    ----------
    culture : str
        One of the registered cultural cluster names (e.g. "east_asian").

    Returns
    -------
    CulturalProfile

    Raises
    ------
    ValueError
        If *culture* is not a recognised cluster name.
    """
    key = culture.lower().strip().replace(" ", "_").replace("-", "_")
    if key not in CULTURAL_PROFILES:
        available = ", ".join(sorted(CULTURAL_PROFILES.keys()))
        raise ValueError(
            f"Unknown cultural cluster '{culture}'. "
            f"Available clusters: {available}"
        )
    return CULTURAL_PROFILES[key]


def apply_display_rule_correction(
    valence: float,
    arousal: float,
    profile: CulturalProfile,
) -> DisplayRuleCorrection:
    """Correct an observed emotion reading for cultural display rules.

    Cultures that suppress negative emotions publicly will show less negative
    valence than actually felt -- so we shift the raw reading further negative
    to estimate the true underlying state.  Cultures that amplify positive
    display show more positivity than actually felt -- so we attenuate.

    Parameters
    ----------
    valence : float  (-1 to 1)
    arousal : float  (0 to 1)
    profile : CulturalProfile

    Returns
    -------
    DisplayRuleCorrection
    """
    valence = max(-1.0, min(1.0, valence))
    arousal = max(0.0, min(1.0, arousal))

    valence_adj = 0.0
    arousal_adj = 0.0
    rationale_parts: List[str] = []

    # --- Negative-suppression correction ---
    # If the culture suppresses negative displays, a neutral/mild-negative
    # reading likely understates the true negative state.
    if valence < 0.1 and profile.negative_suppression > 0.3:
        # Shift valence more negative in proportion to suppression tendency
        shift = -profile.negative_suppression * 0.3 * (1.0 - valence)
        valence_adj += shift
        rationale_parts.append(
            f"negative suppression correction ({profile.negative_suppression:.2f})"
        )

    # --- Positive-amplification correction ---
    # If the culture amplifies positive display, a highly positive reading
    # may overstate the true positive state.
    if valence > 0.0 and profile.positive_amplification > 0.3:
        shift = -profile.positive_amplification * 0.25 * valence
        valence_adj += shift
        rationale_parts.append(
            f"positive amplification correction ({profile.positive_amplification:.2f})"
        )

    # --- Masking correction for arousal ---
    # High-masking cultures display lower arousal than felt internally.
    if profile.masking_tendency > 0.4:
        arousal_boost = profile.masking_tendency * 0.2
        arousal_adj += arousal_boost
        rationale_parts.append(
            f"arousal masking correction ({profile.masking_tendency:.2f})"
        )

    corrected_v = max(-1.0, min(1.0, valence + valence_adj))
    corrected_a = max(0.0, min(1.0, arousal + arousal_adj))

    return DisplayRuleCorrection(
        original_valence=valence,
        original_arousal=arousal,
        corrected_valence=round(corrected_v, 4),
        corrected_arousal=round(corrected_a, 4),
        valence_adjustment=round(valence_adj, 4),
        arousal_adjustment=round(arousal_adj, 4),
        correction_rationale="; ".join(rationale_parts) if rationale_parts else "no correction needed",
    )


def calibrate_self_report(
    reported_valence: float,
    reported_arousal: float,
    profile: CulturalProfile,
) -> SelfReportCalibration:
    """Adjust self-reported emotion values for known cultural response biases.

    Three biases are corrected:
    1. Acquiescence bias -- tendency to agree / rate positively.
    2. Extreme response style -- tendency to use scale endpoints.
    3. Social desirability -- tendency to report socially approved emotions.

    Parameters
    ----------
    reported_valence : float  (-1 to 1)
    reported_arousal : float  (0 to 1)
    profile : CulturalProfile
    """
    v = max(-1.0, min(1.0, reported_valence))
    a = max(0.0, min(1.0, reported_arousal))
    corrections: List[str] = []

    # 1. Acquiescence bias: shifts the midpoint upward.
    #    To correct, pull toward centre proportional to bias strength.
    if profile.acquiescence_bias > 0.3:
        pull_factor = profile.acquiescence_bias * 0.15
        v = v - pull_factor * v  # attenuate away from extremes
        corrections.append(f"acquiescence ({profile.acquiescence_bias:.2f})")

    # 2. Extreme response style: inflates extremes.
    #    To correct, compress toward centre.
    if profile.extreme_response_style > 0.4:
        compress = 1.0 - profile.extreme_response_style * 0.2
        v = v * compress
        a = 0.5 + (a - 0.5) * compress
        corrections.append(f"extreme response ({profile.extreme_response_style:.2f})")

    # 3. Social desirability: inflates positive valence.
    #    To correct, attenuate positive reports.
    if profile.social_desirability_bias > 0.3 and v > 0:
        attenuation = profile.social_desirability_bias * 0.2
        v = v * (1.0 - attenuation)
        corrections.append(f"social desirability ({profile.social_desirability_bias:.2f})")

    v = max(-1.0, min(1.0, v))
    a = max(0.0, min(1.0, a))

    return SelfReportCalibration(
        original_valence=reported_valence,
        original_arousal=reported_arousal,
        calibrated_valence=round(v, 4),
        calibrated_arousal=round(a, 4),
        bias_corrections_applied=corrections,
    )


def compute_affect_valuation(
    valence: float,
    arousal: float,
    profile: CulturalProfile,
) -> AffectValuation:
    """Compute how well a current emotional state matches the culturally ideal affect.

    Based on Tsai (2007): cultures differ in which affective states are
    considered ideal/desirable.  East Asian cultures value calm positive affect
    (serenity, contentment); Western cultures value high-arousal positive affect
    (excitement, enthusiasm).

    Parameters
    ----------
    valence : float (-1 to 1)
    arousal : float (0 to 1)
    profile : CulturalProfile

    Returns
    -------
    AffectValuation with alignment_score 0-1.
    """
    valence = max(-1.0, min(1.0, valence))
    arousal = max(0.0, min(1.0, arousal))

    # Define the ideal point in valence-arousal space for this culture
    ideal_v = 0.7  # universally positive is ideal
    ideal_a = profile.ideal_arousal

    # Euclidean distance in VA space (normalised to 0-1 range)
    dist = math.sqrt((valence - ideal_v) ** 2 + (arousal - ideal_a) ** 2)
    max_dist = math.sqrt(1.7 ** 2 + 1.0 ** 2)  # worst case: v=-1, a is opposite
    alignment = max(0.0, 1.0 - dist / max_dist)

    # Interpretation
    if alignment > 0.75:
        interp = "Current state closely matches cultural ideal affect"
    elif alignment > 0.50:
        interp = "Current state moderately aligns with cultural ideal"
    elif alignment > 0.25:
        interp = "Current state diverges from cultural ideal affect"
    else:
        interp = "Current state is far from cultural ideal affect"

    return AffectValuation(
        ideal_arousal=profile.ideal_arousal,
        ideal_valence_type=profile.ideal_valence_type,
        current_arousal=arousal,
        current_valence=valence,
        alignment_score=round(alignment, 4),
        interpretation=interp,
    )


def adapt_thresholds(
    profile: CulturalProfile,
    base_valence_threshold: float = 0.0,
    base_arousal_threshold: float = 0.5,
) -> Dict[str, float]:
    """Adapt emotion detection thresholds for a cultural profile.

    High-masking cultures need lower thresholds (smaller displays are
    significant).  Highly expressive cultures need higher thresholds
    (large displays are normative, not necessarily indicative of intense
    internal state).

    Parameters
    ----------
    profile : CulturalProfile
    base_valence_threshold : float
        Default valence threshold for detecting negative emotion (default 0.0).
    base_arousal_threshold : float
        Default arousal threshold for detecting high arousal (default 0.5).

    Returns
    -------
    Dict with adjusted thresholds.
    """
    # Expressivity scales thresholds: more expressive = need higher bar
    expressivity_factor = 1.0 + (profile.expressivity_baseline - 0.5) * 0.4

    # Masking inverts: more masking = lower threshold (subtle signals matter)
    masking_factor = 1.0 - profile.masking_tendency * 0.3

    combined_factor = expressivity_factor * masking_factor

    adjusted_valence = round(base_valence_threshold * combined_factor, 4)
    adjusted_arousal = round(base_arousal_threshold * combined_factor, 4)

    # Negative emotion detection threshold
    neg_threshold = round(-0.3 + profile.negative_suppression * 0.2, 4)

    # Positive emotion detection threshold
    pos_threshold = round(0.3 + profile.positive_amplification * 0.15, 4)

    return {
        "valence_threshold": adjusted_valence,
        "arousal_threshold": max(0.0, min(1.0, adjusted_arousal)),
        "negative_emotion_threshold": neg_threshold,
        "positive_emotion_threshold": pos_threshold,
        "expressivity_factor": round(expressivity_factor, 4),
        "masking_factor": round(masking_factor, 4),
    }


def profile_to_dict(profile: CulturalProfile) -> Dict[str, Any]:
    """Serialise a CulturalProfile to a JSON-safe dict."""
    return {
        "cluster_name": profile.cluster_name,
        "description": profile.description,
        "display_rules": {
            "masking_tendency": profile.masking_tendency,
            "expressivity_baseline": profile.expressivity_baseline,
            "negative_suppression": profile.negative_suppression,
            "positive_amplification": profile.positive_amplification,
        },
        "affect_valuation": {
            "ideal_arousal": profile.ideal_arousal,
            "ideal_valence_type": profile.ideal_valence_type,
        },
        "self_report_biases": {
            "acquiescence_bias": profile.acquiescence_bias,
            "extreme_response_style": profile.extreme_response_style,
            "social_desirability_bias": profile.social_desirability_bias,
        },
    }


def calibrate(
    valence: float,
    arousal: float,
    culture: str,
    reported_valence: Optional[float] = None,
    reported_arousal: Optional[float] = None,
) -> CulturalCalibrationResult:
    """Full cultural calibration pipeline.

    Applies display-rule correction to the observed/measured values,
    optionally calibrates self-report values, computes affect valuation,
    and adapts detection thresholds.

    Parameters
    ----------
    valence : float
        Measured/observed emotional valence (-1 to 1).
    arousal : float
        Measured/observed arousal (0 to 1).
    culture : str
        Cultural cluster name.
    reported_valence : float, optional
        Self-reported valence for calibration.
    reported_arousal : float, optional
        Self-reported arousal for calibration.

    Returns
    -------
    CulturalCalibrationResult
    """
    profile = get_cultural_profile(culture)

    display_correction = apply_display_rule_correction(valence, arousal, profile)

    # Self-report calibration uses reported values if available, else measured
    rep_v = reported_valence if reported_valence is not None else valence
    rep_a = reported_arousal if reported_arousal is not None else arousal
    self_report_cal = calibrate_self_report(rep_v, rep_a, profile)

    affect_val = compute_affect_valuation(
        display_correction.corrected_valence,
        display_correction.corrected_arousal,
        profile,
    )

    thresholds = adapt_thresholds(profile)

    return CulturalCalibrationResult(
        culture=profile.cluster_name,
        display_rule_correction=display_correction,
        self_report_calibration=self_report_cal,
        affect_valuation=affect_val,
        adapted_thresholds=thresholds,
        profile_summary=profile_to_dict(profile),
    )
