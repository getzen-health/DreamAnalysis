"""EEG-guided perinatal emotional intelligence — pregnancy & postpartum mood tracking.

Provides trimester-aware emotional baselines, postpartum depression (PPD) risk
scoring compatible with the Edinburgh Postnatal Depression Scale (EPDS),
baby-blues vs PPD distinction, hormonal-phase emotional pattern tracking,
bonding readiness assessment, and partner stress detection.

Trimester emotional baselines:
  1st (weeks 1-13):  elevated anxiety, nausea-related irritability, fatigue
  2nd (weeks 14-27): emotional stabilization, improved energy, "honeymoon"
  3rd (weeks 28-40): rising anxiety, nesting urge, sleep disruption

Postpartum phases:
  Baby blues (0-2 weeks): mild mood swings, tearfulness, resolves spontaneously
  PPD risk window (2-52 weeks): persistent low mood, functional impairment,
      escalating severity warrants clinical referral

Edinburgh Postnatal Depression Scale (EPDS):
  10-item self-report; scores 0-30.  Threshold >= 13 = likely PPD.
  This model produces an EPDS-compatible risk score from EEG-derived emotional
  features, NOT a replacement for the clinical EPDS questionnaire.

Hormonal context:
  Estrogen and progesterone drop sharply within 48 hours of delivery.
  The magnitude of this drop correlates with mood vulnerability.  We track
  emotional patterns against estimated hormonal phases (no direct hormone
  measurement — inferred from weeks postpartum).

References:
    Cox et al. (1987) — Edinburgh Postnatal Depression Scale
    O'Hara & McCabe (2013) — Postpartum depression: current status and future
    Gavin et al. (2005) — Perinatal depression: a systematic review
    Wisner et al. (2013) — Onset timing, thoughts of self-harm, and diagnoses
    Feldman (2007) — Parent-infant synchrony: biological foundations

CLINICAL DISCLAIMER: This is a research and educational tool only.
It is NOT a substitute for clinical screening, the EPDS questionnaire,
or professional perinatal mental health care.  For any genuine mental
health crisis, contact a licensed professional or emergency services.

Issue #451.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clinical disclaimer & resources
# ---------------------------------------------------------------------------

_CLINICAL_DISCLAIMER = (
    "Clinical disclaimer: This perinatal emotional intelligence module is a "
    "research and educational tool only. It is NOT a substitute for the Edinburgh "
    "Postnatal Depression Scale (EPDS), clinical screening, or professional "
    "perinatal mental health care. For any genuine mental health crisis, contact "
    "a licensed mental health professional or call the 988 Suicide & Crisis "
    "Lifeline (call or text 988 in the US), or your local equivalent."
)

_CRISIS_RESOURCES = (
    "If you or someone you know is in crisis: "
    "Postpartum Support International Helpline: 1-800-944-4773 (call or text). "
    "988 Suicide & Crisis Lifeline (US): call or text 988. "
    "Crisis Text Line: text HOME to 741741. "
    "International Association for Suicide Prevention: "
    "https://www.iasp.info/resources/Crisis_Centres/"
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PerinatalPhase(str, Enum):
    """Broad perinatal phase."""
    FIRST_TRIMESTER = "first_trimester"
    SECOND_TRIMESTER = "second_trimester"
    THIRD_TRIMESTER = "third_trimester"
    POSTPARTUM_EARLY = "postpartum_early"      # 0-2 weeks
    POSTPARTUM_LATE = "postpartum_late"         # 2-52 weeks
    UNKNOWN = "unknown"


class PPDRiskLevel(str, Enum):
    """PPD risk classification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class BluesVsPPD(str, Enum):
    """Distinction between baby blues and PPD."""
    BABY_BLUES = "baby_blues"
    POSSIBLE_PPD = "possible_ppd"
    LIKELY_PPD = "likely_ppd"
    NOT_APPLICABLE = "not_applicable"   # still pregnant or too early
    INSUFFICIENT_DATA = "insufficient_data"


class HormonalPhase(str, Enum):
    """Estimated hormonal phase based on weeks postpartum."""
    PREGNANCY_RISING = "pregnancy_rising"        # during pregnancy
    ACUTE_DROP = "acute_drop"                     # 0-1 weeks postpartum
    ADJUSTMENT = "adjustment"                     # 1-6 weeks postpartum
    STABILIZING = "stabilizing"                   # 6-12 weeks postpartum
    STABILIZED = "stabilized"                     # 12+ weeks postpartum
    UNKNOWN = "unknown"


class BondingLevel(str, Enum):
    """Bonding readiness assessment."""
    STRONG = "strong"
    DEVELOPING = "developing"
    DELAYED = "delayed"
    CONCERN = "concern"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PerinatalState:
    """Current perinatal state for a user."""
    weeks_pregnant: Optional[int] = None        # None if postpartum
    weeks_postpartum: Optional[int] = None      # None if pregnant
    trimester: Optional[int] = None             # 1, 2, or 3 (None if postpartum)
    phase: PerinatalPhase = PerinatalPhase.UNKNOWN

    def __post_init__(self) -> None:
        """Derive phase and trimester from week counts."""
        if self.weeks_pregnant is not None and self.weeks_pregnant >= 0:
            if self.weeks_pregnant <= 13:
                self.trimester = 1
                self.phase = PerinatalPhase.FIRST_TRIMESTER
            elif self.weeks_pregnant <= 27:
                self.trimester = 2
                self.phase = PerinatalPhase.SECOND_TRIMESTER
            else:
                self.trimester = 3
                self.phase = PerinatalPhase.THIRD_TRIMESTER
        elif self.weeks_postpartum is not None and self.weeks_postpartum >= 0:
            self.trimester = None
            if self.weeks_postpartum <= 2:
                self.phase = PerinatalPhase.POSTPARTUM_EARLY
            else:
                self.phase = PerinatalPhase.POSTPARTUM_LATE


@dataclass
class PPDRiskScore:
    """PPD risk assessment result."""
    epds_proxy_score: float = 0.0       # 0-30 scale (EPDS-compatible)
    risk_level: PPDRiskLevel = PPDRiskLevel.LOW
    confidence: float = 0.0
    indicators: List[str] = field(default_factory=list)
    recommendation: str = ""
    clinical_disclaimer: str = _CLINICAL_DISCLAIMER


@dataclass
class PerinatalProfile:
    """Complete perinatal emotional profile."""
    state: PerinatalState = field(default_factory=PerinatalState)
    phase: PerinatalPhase = PerinatalPhase.UNKNOWN
    trimester_baseline: Dict[str, float] = field(default_factory=dict)
    ppd_risk: PPDRiskScore = field(default_factory=PPDRiskScore)
    blues_vs_ppd: BluesVsPPD = BluesVsPPD.NOT_APPLICABLE
    hormonal_phase: HormonalPhase = HormonalPhase.UNKNOWN
    hormonal_emotional_pattern: Dict[str, Any] = field(default_factory=dict)
    bonding_level: BondingLevel = BondingLevel.DEVELOPING
    bonding_score: float = 0.0
    partner_support_score: float = 0.0
    partner_stress_detected: bool = False
    clinical_disclaimer: str = _CLINICAL_DISCLAIMER
    crisis_resources: str = _CRISIS_RESOURCES


# ---------------------------------------------------------------------------
# Emotional reading input
# ---------------------------------------------------------------------------


@dataclass
class PerinatalReading:
    """A single emotional reading in perinatal context."""
    valence: float = 0.0            # -1 to 1
    arousal: float = 0.0            # 0 to 1
    stress_index: float = 0.0       # 0 to 1
    anxiety_index: float = 0.0      # 0 to 1
    irritability_index: float = 0.0 # 0 to 1
    fatigue_index: float = 0.0      # 0 to 1
    tearfulness_index: float = 0.0  # 0 to 1
    bonding_response: float = 0.5   # 0 to 1 — response to baby-related stimuli
    partner_valence: Optional[float] = None   # partner's emotional valence if available
    partner_stress: Optional[float] = None    # partner's stress if available
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "stress_index": round(self.stress_index, 4),
            "anxiety_index": round(self.anxiety_index, 4),
            "irritability_index": round(self.irritability_index, 4),
            "fatigue_index": round(self.fatigue_index, 4),
            "tearfulness_index": round(self.tearfulness_index, 4),
            "bonding_response": round(self.bonding_response, 4),
            "partner_valence": round(self.partner_valence, 4) if self.partner_valence is not None else None,
            "partner_stress": round(self.partner_stress, 4) if self.partner_stress is not None else None,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Trimester baseline thresholds
# ---------------------------------------------------------------------------

# Expected emotional norms per trimester and postpartum phases.
# Format: {metric: (expected_mean, acceptable_deviation)}
_TRIMESTER_BASELINES: Dict[str, Dict[str, tuple]] = {
    "first_trimester": {
        "valence": (-0.05, 0.30),       # slightly negative, wide range
        "arousal": (0.45, 0.20),
        "anxiety": (0.50, 0.20),         # elevated anxiety normal
        "irritability": (0.40, 0.20),    # nausea-related irritability
        "fatigue": (0.60, 0.20),         # high fatigue normal
    },
    "second_trimester": {
        "valence": (0.15, 0.25),         # improved mood ("honeymoon")
        "arousal": (0.40, 0.15),
        "anxiety": (0.25, 0.15),         # reduced anxiety
        "irritability": (0.20, 0.15),    # reduced irritability
        "fatigue": (0.35, 0.20),         # improved energy
    },
    "third_trimester": {
        "valence": (0.0, 0.30),          # mixed
        "arousal": (0.50, 0.20),
        "anxiety": (0.55, 0.20),         # rising anxiety, nesting
        "irritability": (0.35, 0.20),
        "fatigue": (0.65, 0.20),         # significant fatigue
    },
    "postpartum_early": {
        "valence": (-0.10, 0.35),        # mood lability normal
        "arousal": (0.50, 0.25),
        "anxiety": (0.45, 0.25),
        "irritability": (0.35, 0.25),
        "fatigue": (0.70, 0.20),         # extreme fatigue normal
    },
    "postpartum_late": {
        "valence": (0.10, 0.25),         # should be improving
        "arousal": (0.40, 0.20),
        "anxiety": (0.30, 0.20),
        "irritability": (0.25, 0.20),
        "fatigue": (0.50, 0.25),
    },
}

# PPD scoring weights
_PPD_WEIGHTS = {
    "negative_valence": 3.0,
    "low_arousal": 2.0,
    "high_anxiety": 2.5,
    "high_tearfulness": 2.0,
    "high_fatigue": 1.5,
    "high_irritability": 2.0,
    "low_bonding": 2.5,
    "high_stress": 2.0,
}

# EPDS thresholds
_EPDS_LOW_THRESHOLD = 9          # 0-9: low risk
_EPDS_MODERATE_THRESHOLD = 13    # 10-12: moderate risk
_EPDS_HIGH_THRESHOLD = 20        # 13-19: high risk (clinical cutoff is 13)
                                 # 20-30: critical

# Blues: resolves within 2 weeks, mild
_BLUES_MAX_WEEKS = 2
_BLUES_SEVERITY_CEILING = 12     # EPDS proxy below this = blues-level

# Bonding thresholds
_BONDING_STRONG = 0.70
_BONDING_DEVELOPING = 0.45
_BONDING_DELAYED = 0.25


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_trimester_baseline(state: PerinatalState) -> Dict[str, Any]:
    """Compute expected emotional baseline for the current perinatal phase.

    Returns expected norms and deviation thresholds so readings can be
    compared against phase-appropriate expectations.
    """
    phase_key = state.phase.value
    baselines = _TRIMESTER_BASELINES.get(phase_key)

    if baselines is None:
        return {
            "phase": state.phase.value,
            "baselines": {},
            "note": "Unknown phase; no baseline available.",
        }

    result: Dict[str, Any] = {
        "phase": state.phase.value,
        "trimester": state.trimester,
        "weeks_pregnant": state.weeks_pregnant,
        "weeks_postpartum": state.weeks_postpartum,
        "baselines": {},
    }

    for metric, (mean, deviation) in baselines.items():
        result["baselines"][metric] = {
            "expected_mean": round(mean, 4),
            "acceptable_range": [
                round(mean - deviation, 4),
                round(mean + deviation, 4),
            ],
        }

    # Phase-specific notes
    notes = {
        "first_trimester": (
            "First trimester: elevated anxiety and fatigue are normal. "
            "Nausea-related irritability is common."
        ),
        "second_trimester": (
            "Second trimester: emotional stabilization expected. "
            "Often called the 'honeymoon' trimester."
        ),
        "third_trimester": (
            "Third trimester: rising anxiety and nesting urge are normal. "
            "Sleep disruption increases fatigue."
        ),
        "postpartum_early": (
            "Early postpartum (0-2 weeks): mood lability and tearfulness "
            "are common (baby blues). Monitor for escalation."
        ),
        "postpartum_late": (
            "Late postpartum (2+ weeks): mood should be stabilizing. "
            "Persistent low mood may indicate PPD."
        ),
    }
    result["note"] = notes.get(phase_key, "")
    return result


def score_ppd_risk(
    reading: PerinatalReading,
    state: PerinatalState,
    recent_readings: Optional[List[PerinatalReading]] = None,
) -> PPDRiskScore:
    """Compute PPD risk score from emotional reading and perinatal state.

    Produces an EPDS-compatible proxy score (0-30 scale) from EEG-derived
    emotional features.  This is NOT a replacement for the clinical EPDS.

    The scoring algorithm weights emotional indicators based on their
    clinical correlation with PPD:
    - Persistent negative valence (strongest predictor)
    - Low bonding response to baby stimuli
    - High anxiety with low arousal (anhedonia pattern)
    - Tearfulness beyond normal baby-blues window
    - Functional impairment markers (fatigue + low focus)
    """
    indicators: List[str] = []
    raw_score = 0.0

    # 1. Negative valence contribution (0-3 points mapped to 0-9 EPDS)
    neg_val = max(0.0, -reading.valence)
    val_contribution = neg_val * _PPD_WEIGHTS["negative_valence"] * 3.0
    if neg_val > 0.3:
        indicators.append("persistent_negative_mood")
    raw_score += val_contribution

    # 2. Low arousal / anhedonia (0-6 EPDS)
    low_arousal = max(0.0, 0.4 - reading.arousal)
    arousal_contribution = low_arousal * _PPD_WEIGHTS["low_arousal"] * 5.0
    if low_arousal > 0.15:
        indicators.append("low_arousal_anhedonia")
    raw_score += arousal_contribution

    # 3. Anxiety (0-5 EPDS)
    anxiety_contribution = reading.anxiety_index * _PPD_WEIGHTS["high_anxiety"] * 2.0
    if reading.anxiety_index > 0.6:
        indicators.append("elevated_anxiety")
    raw_score += anxiety_contribution

    # 4. Tearfulness (0-3 EPDS)
    tear_contribution = reading.tearfulness_index * _PPD_WEIGHTS["high_tearfulness"] * 1.5
    if reading.tearfulness_index > 0.5:
        indicators.append("frequent_tearfulness")
    raw_score += tear_contribution

    # 5. Fatigue beyond phase norm (0-3 EPDS)
    phase_key = state.phase.value
    phase_baselines = _TRIMESTER_BASELINES.get(phase_key, {})
    expected_fatigue_mean = phase_baselines.get("fatigue", (0.5, 0.2))[0]
    excess_fatigue = max(0.0, reading.fatigue_index - expected_fatigue_mean - 0.1)
    fatigue_contribution = excess_fatigue * _PPD_WEIGHTS["high_fatigue"] * 3.0
    if excess_fatigue > 0.15:
        indicators.append("excess_fatigue")
    raw_score += fatigue_contribution

    # 6. Irritability (0-2 EPDS)
    irr_contribution = reading.irritability_index * _PPD_WEIGHTS["high_irritability"] * 1.0
    if reading.irritability_index > 0.6:
        indicators.append("elevated_irritability")
    raw_score += irr_contribution

    # 7. Low bonding response (0-5 EPDS) — only postpartum
    if state.weeks_postpartum is not None:
        low_bonding = max(0.0, 0.5 - reading.bonding_response)
        bonding_contribution = low_bonding * _PPD_WEIGHTS["low_bonding"] * 4.0
        if low_bonding > 0.2:
            indicators.append("low_bonding_response")
        raw_score += bonding_contribution

    # 8. Stress (0-4 EPDS)
    stress_contribution = reading.stress_index * _PPD_WEIGHTS["high_stress"] * 2.0
    if reading.stress_index > 0.6:
        indicators.append("high_stress")
    raw_score += stress_contribution

    # Persistence factor: if recent readings also show distress, amplify
    if recent_readings and len(recent_readings) >= 3:
        recent_neg = sum(1 for r in recent_readings[-5:] if r.valence < -0.2)
        persistence_ratio = recent_neg / min(len(recent_readings[-5:]), 5)
        if persistence_ratio > 0.6:
            raw_score *= 1.2
            indicators.append("persistent_pattern")

    # Clamp to EPDS range
    epds_proxy = min(30.0, max(0.0, raw_score))

    # Classify risk level
    if epds_proxy < _EPDS_LOW_THRESHOLD:
        risk_level = PPDRiskLevel.LOW
        recommendation = "Continue monitoring. Current emotional state is within normal range."
    elif epds_proxy < _EPDS_MODERATE_THRESHOLD:
        risk_level = PPDRiskLevel.MODERATE
        recommendation = (
            "Moderate risk detected. Consider discussing emotional health "
            "with your healthcare provider at your next visit."
        )
    elif epds_proxy < _EPDS_HIGH_THRESHOLD:
        risk_level = PPDRiskLevel.HIGH
        recommendation = (
            "High risk detected (EPDS proxy >= 13). Clinical screening recommended. "
            "Please contact your healthcare provider or the Postpartum Support "
            "International Helpline: 1-800-944-4773."
        )
    else:
        risk_level = PPDRiskLevel.CRITICAL
        recommendation = (
            "Critical risk detected. Immediate professional support recommended. "
            "Contact your healthcare provider, go to your nearest emergency room, "
            "or call 988 (Suicide & Crisis Lifeline)."
        )

    # Confidence: higher when more indicators present and reading is extreme
    confidence = min(1.0, len(indicators) * 0.15 + abs(reading.valence) * 0.3)

    return PPDRiskScore(
        epds_proxy_score=round(epds_proxy, 2),
        risk_level=risk_level,
        confidence=round(confidence, 4),
        indicators=indicators,
        recommendation=recommendation,
    )


def distinguish_blues_vs_ppd(
    reading: PerinatalReading,
    state: PerinatalState,
    ppd_risk: Optional[PPDRiskScore] = None,
    recent_readings: Optional[List[PerinatalReading]] = None,
) -> Dict[str, Any]:
    """Distinguish baby blues from PPD based on timing, severity, and pattern.

    Baby blues:
      - Onset: 2-3 days postpartum
      - Duration: resolves within 2 weeks
      - Severity: mild mood swings, tearfulness, anxiety
      - No functional impairment

    PPD:
      - Onset: anytime in first year, commonly 2-12 weeks
      - Duration: > 2 weeks, can last months/years without treatment
      - Severity: persistent low mood, loss of interest, guilt, hopelessness
      - Functional impairment present
    """
    result: Dict[str, Any] = {
        "classification": BluesVsPPD.NOT_APPLICABLE.value,
        "confidence": 0.0,
        "indicators": [],
        "explanation": "",
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }

    # Only applicable postpartum
    if state.weeks_postpartum is None:
        result["explanation"] = "Not yet postpartum; blues vs PPD distinction not applicable."
        return result

    # Compute PPD risk if not provided
    if ppd_risk is None:
        ppd_risk = score_ppd_risk(reading, state, recent_readings)

    severity = ppd_risk.epds_proxy_score
    weeks_pp = state.weeks_postpartum

    indicators: List[str] = []
    blues_points = 0.0
    ppd_points = 0.0

    # Timing factor
    if weeks_pp <= _BLUES_MAX_WEEKS:
        blues_points += 2.0
        indicators.append("within_blues_window")
    else:
        ppd_points += 2.0
        indicators.append("beyond_blues_window")

    # Severity factor
    if severity < _BLUES_SEVERITY_CEILING:
        blues_points += 2.0
        indicators.append("mild_severity")
    else:
        ppd_points += 3.0
        indicators.append("elevated_severity")

    # Functional impairment proxies
    if reading.fatigue_index > 0.7 and reading.arousal < 0.3:
        ppd_points += 1.5
        indicators.append("functional_impairment")
    else:
        blues_points += 1.0

    # Bonding disruption
    if reading.bonding_response < 0.3:
        ppd_points += 2.0
        indicators.append("bonding_disruption")
    elif reading.bonding_response >= 0.5:
        blues_points += 1.0
        indicators.append("bonding_intact")

    # Persistence from recent readings
    if recent_readings and len(recent_readings) >= 3:
        recent_neg = sum(1 for r in recent_readings[-5:] if r.valence < -0.2)
        if recent_neg >= 3:
            ppd_points += 2.0
            indicators.append("persistent_negative_mood")
        else:
            blues_points += 1.0
            indicators.append("fluctuating_mood")

    # Classify
    total = blues_points + ppd_points
    if total == 0:
        classification = BluesVsPPD.INSUFFICIENT_DATA
        confidence = 0.0
        explanation = "Insufficient data to distinguish blues from PPD."
    elif ppd_points > blues_points * 1.5:
        classification = BluesVsPPD.LIKELY_PPD
        confidence = min(1.0, ppd_points / total)
        explanation = (
            "Pattern suggests possible postpartum depression rather than "
            "baby blues. Clinical evaluation strongly recommended."
        )
    elif ppd_points > blues_points:
        classification = BluesVsPPD.POSSIBLE_PPD
        confidence = min(1.0, ppd_points / total)
        explanation = (
            "Some indicators suggest this may be more than baby blues. "
            "Consider clinical screening."
        )
    else:
        classification = BluesVsPPD.BABY_BLUES
        confidence = min(1.0, blues_points / total)
        explanation = (
            "Pattern is consistent with normal baby blues. Should resolve "
            "within 2 weeks. Monitor for escalation."
        )

    result["classification"] = classification.value
    result["confidence"] = round(confidence, 4)
    result["indicators"] = indicators
    result["explanation"] = explanation
    result["severity_score"] = round(severity, 2)
    result["weeks_postpartum"] = weeks_pp

    return result


def track_hormonal_emotional_pattern(
    state: PerinatalState,
    reading: PerinatalReading,
) -> Dict[str, Any]:
    """Track emotional patterns against estimated hormonal phases.

    Estrogen and progesterone dynamics:
      Pregnancy: steadily rising, peaks at term
      0-48h postpartum: dramatic drop (>90% decline)
      1-6 weeks: gradual recovery begins
      6-12 weeks: approaching pre-pregnancy levels
      12+ weeks: stabilized

    We estimate the hormonal phase from weeks postpartum and correlate
    emotional readings with expected hormonal effects.
    """
    # Determine hormonal phase
    if state.weeks_pregnant is not None:
        hormonal_phase = HormonalPhase.PREGNANCY_RISING
        expected_mood_effect = "elevated_baseline"
        vulnerability = 0.2
    elif state.weeks_postpartum is not None:
        weeks_pp = state.weeks_postpartum
        if weeks_pp < 1:
            hormonal_phase = HormonalPhase.ACUTE_DROP
            expected_mood_effect = "high_mood_vulnerability"
            vulnerability = 0.9
        elif weeks_pp < 6:
            hormonal_phase = HormonalPhase.ADJUSTMENT
            expected_mood_effect = "adjustment_phase"
            vulnerability = 0.6
        elif weeks_pp < 12:
            hormonal_phase = HormonalPhase.STABILIZING
            expected_mood_effect = "stabilizing"
            vulnerability = 0.35
        else:
            hormonal_phase = HormonalPhase.STABILIZED
            expected_mood_effect = "hormones_stabilized"
            vulnerability = 0.15
    else:
        hormonal_phase = HormonalPhase.UNKNOWN
        expected_mood_effect = "unknown"
        vulnerability = 0.5

    # Compare reading against expected hormonal effect
    mood_concern = False
    notes: List[str] = []

    if hormonal_phase == HormonalPhase.ACUTE_DROP:
        notes.append(
            "Estrogen/progesterone drop sharply postpartum. "
            "Mood lability is expected and usually transient."
        )
        if reading.valence < -0.4:
            mood_concern = True
            notes.append("Mood is significantly low during acute hormonal drop.")

    elif hormonal_phase == HormonalPhase.ADJUSTMENT:
        if reading.valence < -0.3 and reading.anxiety_index > 0.5:
            mood_concern = True
            notes.append(
                "Negative mood with anxiety during hormonal adjustment. "
                "Monitor for PPD development."
            )

    elif hormonal_phase == HormonalPhase.STABILIZING:
        if reading.valence < -0.2:
            mood_concern = True
            notes.append(
                "Mood should be improving as hormones stabilize. "
                "Persistent low mood warrants clinical attention."
            )

    elif hormonal_phase == HormonalPhase.STABILIZED:
        if reading.valence < -0.2:
            mood_concern = True
            notes.append(
                "Hormones have stabilized. Low mood at this stage "
                "is less likely hormonal and may indicate PPD."
            )

    return {
        "hormonal_phase": hormonal_phase.value,
        "expected_mood_effect": expected_mood_effect,
        "vulnerability_score": round(vulnerability, 4),
        "mood_concern": mood_concern,
        "notes": notes,
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }


def assess_bonding_readiness(
    reading: PerinatalReading,
    state: PerinatalState,
    recent_readings: Optional[List[PerinatalReading]] = None,
) -> Dict[str, Any]:
    """Assess emotional bonding readiness from baby-related stimulus response.

    Bonding readiness is estimated from:
    1. Emotional response to baby-related stimuli (bonding_response field)
    2. Overall emotional capacity (valence + arousal baseline)
    3. Consistency over recent readings (trend)

    This is a PROXY only. True bonding assessment requires clinical
    observation of parent-infant interaction.
    """
    score = reading.bonding_response

    # Modulate by emotional capacity
    if reading.valence < -0.4 and reading.arousal < 0.25:
        # Very low mood + low energy reduces bonding capacity
        score *= 0.7

    # Trend from recent readings
    trend = "stable"
    if recent_readings and len(recent_readings) >= 3:
        recent_bonding = [r.bonding_response for r in recent_readings[-5:]]
        avg_recent = sum(recent_bonding) / len(recent_bonding)
        if score > avg_recent + 0.1:
            trend = "improving"
        elif score < avg_recent - 0.1:
            trend = "declining"

    # Classify
    if score >= _BONDING_STRONG:
        level = BondingLevel.STRONG
        note = "Strong emotional responsiveness to baby-related stimuli."
    elif score >= _BONDING_DEVELOPING:
        level = BondingLevel.DEVELOPING
        note = "Bonding is developing normally. Give it time."
    elif score >= _BONDING_DELAYED:
        level = BondingLevel.DELAYED
        note = (
            "Bonding may be delayed. This is common and does not mean failure. "
            "Support and professional guidance can help."
        )
    else:
        level = BondingLevel.CONCERN
        note = (
            "Low bonding responsiveness detected. Please discuss with your "
            "healthcare provider. Many effective interventions exist."
        )

    return {
        "bonding_level": level.value,
        "bonding_score": round(score, 4),
        "trend": trend,
        "note": note,
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }


def _detect_partner_support(
    reading: PerinatalReading,
) -> Dict[str, Any]:
    """Assess whether partner's emotional state is supportive or adding stress.

    Uses partner_valence and partner_stress fields when available.
    """
    if reading.partner_valence is None and reading.partner_stress is None:
        return {
            "partner_data_available": False,
            "support_score": 0.5,
            "stress_detected": False,
            "note": "No partner emotional data available.",
        }

    support_score = 0.5
    stress_detected = False
    notes: List[str] = []

    if reading.partner_valence is not None:
        if reading.partner_valence > 0.2:
            support_score += 0.25
            notes.append("Partner shows positive emotional state.")
        elif reading.partner_valence < -0.3:
            support_score -= 0.25
            stress_detected = True
            notes.append("Partner shows negative emotional state, which may add stress.")

    if reading.partner_stress is not None:
        if reading.partner_stress > 0.6:
            support_score -= 0.20
            stress_detected = True
            notes.append("Partner stress is elevated.")
        elif reading.partner_stress < 0.3:
            support_score += 0.15
            notes.append("Partner stress is low, supportive environment.")

    support_score = max(0.0, min(1.0, support_score))

    return {
        "partner_data_available": True,
        "support_score": round(support_score, 4),
        "stress_detected": stress_detected,
        "notes": notes,
    }


def compute_perinatal_profile(
    reading: PerinatalReading,
    state: PerinatalState,
    recent_readings: Optional[List[PerinatalReading]] = None,
) -> PerinatalProfile:
    """Compute the complete perinatal emotional profile.

    Aggregates: trimester baseline, PPD risk, blues-vs-PPD distinction,
    hormonal context, bonding readiness, and partner support.
    """
    # Trimester baseline
    baseline = compute_trimester_baseline(state)

    # PPD risk
    ppd_risk = score_ppd_risk(reading, state, recent_readings)

    # Blues vs PPD
    blues_ppd_result = distinguish_blues_vs_ppd(
        reading, state, ppd_risk, recent_readings,
    )

    # Hormonal context
    hormonal_result = track_hormonal_emotional_pattern(state, reading)

    # Bonding
    bonding_result = assess_bonding_readiness(reading, state, recent_readings)

    # Partner support
    partner_result = _detect_partner_support(reading)

    return PerinatalProfile(
        state=state,
        phase=state.phase,
        trimester_baseline=baseline,
        ppd_risk=ppd_risk,
        blues_vs_ppd=BluesVsPPD(blues_ppd_result["classification"]),
        hormonal_phase=HormonalPhase(hormonal_result["hormonal_phase"]),
        hormonal_emotional_pattern=hormonal_result,
        bonding_level=BondingLevel(bonding_result["bonding_level"]),
        bonding_score=bonding_result["bonding_score"],
        partner_support_score=partner_result["support_score"],
        partner_stress_detected=partner_result["stress_detected"],
    )


def profile_to_dict(profile: PerinatalProfile) -> Dict[str, Any]:
    """Serialize a PerinatalProfile to a JSON-safe dictionary."""
    return {
        "phase": profile.phase.value,
        "trimester": profile.state.trimester,
        "weeks_pregnant": profile.state.weeks_pregnant,
        "weeks_postpartum": profile.state.weeks_postpartum,
        "trimester_baseline": profile.trimester_baseline,
        "ppd_risk": {
            "epds_proxy_score": profile.ppd_risk.epds_proxy_score,
            "risk_level": profile.ppd_risk.risk_level.value,
            "confidence": profile.ppd_risk.confidence,
            "indicators": profile.ppd_risk.indicators,
            "recommendation": profile.ppd_risk.recommendation,
        },
        "blues_vs_ppd": profile.blues_vs_ppd.value,
        "hormonal_phase": profile.hormonal_phase.value,
        "hormonal_emotional_pattern": profile.hormonal_emotional_pattern,
        "bonding": {
            "level": profile.bonding_level.value,
            "score": profile.bonding_score,
        },
        "partner_support": {
            "score": profile.partner_support_score,
            "stress_detected": profile.partner_stress_detected,
        },
        "clinical_disclaimer": profile.clinical_disclaimer,
        "crisis_resources": profile.crisis_resources,
    }
