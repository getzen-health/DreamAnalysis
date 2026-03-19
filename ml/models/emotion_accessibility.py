"""Emotion-aware accessibility layer — issue #460.

Adaptive interface recommendations based on EEG emotional state. When a user
is overwhelmed, the system suggests simplifying UI. When fatigued, it
recommends higher contrast and larger fonts. Sensory-sensitive users get
reduced animation speed and audio cue adjustments.

Accessibility profiles:
    - visual_impairment:          Larger fonts, higher contrast, reduced clutter
    - motor_difficulty:           Larger targets, slower transitions, reduced precision needs
    - cognitive_load_sensitivity: Simplified layout, fewer elements, reduced information density
    - sensory_sensitivity:        Muted colors, no animations, reduced audio, low brightness

EEG-driven adaptations are computed from valence, arousal, stress, fatigue,
and cognitive load signals — all mapped to concrete UI parameters.

Issue #460.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCESSIBILITY_PROFILES = (
    "visual_impairment",
    "motor_difficulty",
    "cognitive_load_sensitivity",
    "sensory_sensitivity",
)

# Default UI parameter ranges
_FONT_SIZE_RANGE = (12, 32)        # px
_CONTRAST_RANGE = (1.0, 4.5)       # WCAG contrast ratio multiplier
_ANIMATION_SPEED_RANGE = (0.0, 1.0)  # 0 = no animation, 1 = full speed
_INFO_DENSITY_RANGE = (0.2, 1.0)    # fraction of max info shown
_AUDIO_CUE_VOLUME_RANGE = (0.0, 1.0)

# Thresholds for EEG-based state classification
_OVERWHELM_AROUSAL_THRESHOLD = 0.75
_OVERWHELM_STRESS_THRESHOLD = 0.70
_FATIGUE_ENERGY_THRESHOLD = 0.30
_HIGH_COGNITIVE_LOAD_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EEGState:
    """Current emotional/cognitive state derived from EEG signals."""
    valence: float = 0.0       # -1..+1
    arousal: float = 0.5       # 0..1
    stress: float = 0.3        # 0..1
    fatigue: float = 0.3       # 0..1
    cognitive_load: float = 0.3  # 0..1
    timestamp: float = 0.0


@dataclass
class AccessibilityProfile:
    """User's baseline accessibility needs."""
    visual_impairment: float = 0.0        # 0..1 severity
    motor_difficulty: float = 0.0         # 0..1 severity
    cognitive_load_sensitivity: float = 0.0  # 0..1 severity
    sensory_sensitivity: float = 0.0      # 0..1 severity


@dataclass
class UIAdaptation:
    """Concrete UI adaptation recommendations."""
    font_size: int = 16                # px
    contrast_ratio: float = 1.0        # multiplier
    animation_speed: float = 1.0       # 0..1
    information_density: float = 1.0   # 0..1
    audio_cue_volume: float = 0.5      # 0..1
    color_saturation: float = 1.0      # 0..1
    target_size_multiplier: float = 1.0  # touch target size multiplier
    transition_duration_ms: int = 300  # ms for UI transitions
    reduce_motion: bool = False
    high_contrast: bool = False
    simplified_layout: bool = False
    reasons: List[str] = field(default_factory=list)


@dataclass
class AccessibilityAssessment:
    """Full accessibility assessment combining EEG state + user profile."""
    eeg_state: EEGState
    profile: AccessibilityProfile
    adaptation: UIAdaptation
    overwhelm_score: float = 0.0       # 0..1
    fatigue_score: float = 0.0         # 0..1
    needs_intervention: bool = False
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, value))


def _lerp(lo: float, hi: float, t: float) -> float:
    """Linear interpolation between lo and hi by factor t (0..1)."""
    t = _clamp(t, 0.0, 1.0)
    return lo + (hi - lo) * t


def assess_accessibility_needs(
    eeg_state: EEGState,
    profile: Optional[AccessibilityProfile] = None,
) -> AccessibilityAssessment:
    """Assess current accessibility needs from EEG state and user profile.

    Computes overwhelm and fatigue scores, determines if intervention
    is needed, and generates the initial assessment before adaptation.

    Args:
        eeg_state: Current EEG-derived emotional/cognitive state.
        profile: Optional user accessibility profile. Defaults to no impairments.

    Returns:
        AccessibilityAssessment with scores and empty adaptation (call
        recommend_adaptations to fill in the adaptation).
    """
    if profile is None:
        profile = AccessibilityProfile()

    # Overwhelm score: combination of high arousal, high stress, high cognitive load
    overwhelm = _clamp(
        0.35 * max(0.0, eeg_state.arousal - 0.4) / 0.6
        + 0.35 * eeg_state.stress
        + 0.30 * eeg_state.cognitive_load,
        0.0, 1.0,
    )

    # Fatigue score: high fatigue + low arousal + low energy proxy
    fatigue = _clamp(
        0.50 * eeg_state.fatigue
        + 0.30 * max(0.0, 0.5 - eeg_state.arousal)
        + 0.20 * max(0.0, eeg_state.cognitive_load - 0.5),
        0.0, 1.0,
    )

    needs_intervention = (
        overwhelm > 0.65
        or fatigue > 0.65
        or eeg_state.stress > _OVERWHELM_STRESS_THRESHOLD
    )

    return AccessibilityAssessment(
        eeg_state=eeg_state,
        profile=profile,
        adaptation=UIAdaptation(),
        overwhelm_score=round(overwhelm, 4),
        fatigue_score=round(fatigue, 4),
        needs_intervention=needs_intervention,
        timestamp=time.time(),
    )


def recommend_adaptations(
    assessment: AccessibilityAssessment,
) -> UIAdaptation:
    """Generate concrete UI adaptation recommendations.

    Combines the EEG-based assessment (overwhelm, fatigue) with the user's
    baseline accessibility profile to produce specific parameter values.

    Args:
        assessment: The accessibility assessment from assess_accessibility_needs.

    Returns:
        UIAdaptation with all parameters filled in.
    """
    profile = assessment.profile
    eeg = assessment.eeg_state
    overwhelm = assessment.overwhelm_score
    fatigue = assessment.fatigue_score

    reasons: List[str] = []

    # -- Font size --
    # Increase when: visual impairment, fatigue, or overwhelm
    font_factor = max(profile.visual_impairment, fatigue * 0.6, overwhelm * 0.4)
    font_size = int(_lerp(_FONT_SIZE_RANGE[0], _FONT_SIZE_RANGE[1], font_factor))
    if font_factor > 0.3:
        reasons.append("Increased font size due to "
                       + ("visual needs" if profile.visual_impairment > 0.3
                          else "fatigue" if fatigue > 0.4 else "overwhelm"))

    # -- Contrast --
    contrast_factor = max(profile.visual_impairment, fatigue * 0.5)
    contrast_ratio = round(_lerp(_CONTRAST_RANGE[0], _CONTRAST_RANGE[1], contrast_factor), 2)
    high_contrast = contrast_factor > 0.5
    if high_contrast:
        reasons.append("High contrast mode enabled")

    # -- Animation speed --
    # Reduce for sensory sensitivity, overwhelm, or motor difficulty
    suppress_factor = max(
        profile.sensory_sensitivity,
        overwhelm * 0.7,
        profile.motor_difficulty * 0.5,
    )
    animation_speed = round(1.0 - suppress_factor, 2)
    animation_speed = _clamp(animation_speed, _ANIMATION_SPEED_RANGE[0], _ANIMATION_SPEED_RANGE[1])
    reduce_motion = suppress_factor > 0.6
    if reduce_motion:
        reasons.append("Reduced motion for sensory comfort")

    # -- Information density --
    simplify_factor = max(
        profile.cognitive_load_sensitivity,
        overwhelm * 0.8,
        eeg.cognitive_load * 0.5,
    )
    info_density = round(1.0 - simplify_factor * 0.8, 2)
    info_density = _clamp(info_density, _INFO_DENSITY_RANGE[0], _INFO_DENSITY_RANGE[1])
    simplified_layout = simplify_factor > 0.5
    if simplified_layout:
        reasons.append("Simplified layout to reduce cognitive load")

    # -- Audio cue volume --
    # Lower for sensory sensitivity, slightly raise for visual impairment
    if profile.sensory_sensitivity > 0.5:
        audio_volume = round(0.5 * (1.0 - profile.sensory_sensitivity), 2)
        reasons.append("Reduced audio cues for sensory sensitivity")
    elif profile.visual_impairment > 0.5:
        audio_volume = round(min(1.0, 0.5 + profile.visual_impairment * 0.4), 2)
        reasons.append("Enhanced audio cues for visual accessibility")
    else:
        audio_volume = 0.5

    # -- Color saturation --
    color_saturation = round(1.0 - profile.sensory_sensitivity * 0.6, 2)
    color_saturation = _clamp(color_saturation, 0.2, 1.0)

    # -- Target size --
    target_multiplier = round(1.0 + profile.motor_difficulty * 1.5 + fatigue * 0.3, 2)
    if profile.motor_difficulty > 0.3:
        reasons.append("Larger touch targets for motor accessibility")

    # -- Transition duration --
    transition_ms = int(300 + profile.motor_difficulty * 500 + overwhelm * 200)
    transition_ms = min(transition_ms, 1200)

    adaptation = UIAdaptation(
        font_size=font_size,
        contrast_ratio=contrast_ratio,
        animation_speed=animation_speed,
        information_density=info_density,
        audio_cue_volume=audio_volume,
        color_saturation=color_saturation,
        target_size_multiplier=target_multiplier,
        transition_duration_ms=transition_ms,
        reduce_motion=reduce_motion,
        high_contrast=high_contrast,
        simplified_layout=simplified_layout,
        reasons=reasons,
    )

    assessment.adaptation = adaptation
    return adaptation


def compute_accessibility_profile(
    eeg_history: List[EEGState],
    known_impairments: Optional[Dict[str, float]] = None,
) -> AccessibilityProfile:
    """Compute an accessibility profile from EEG history and known impairments.

    Analyzes patterns in EEG history to detect sensitivity to cognitive load,
    sensory overload, etc. Merges with any clinically known impairments.

    Args:
        eeg_history: List of historical EEG state measurements.
        known_impairments: Optional dict mapping profile field names to severity (0..1).

    Returns:
        AccessibilityProfile with computed and/or known impairment levels.
    """
    if known_impairments is None:
        known_impairments = {}

    visual = known_impairments.get("visual_impairment", 0.0)
    motor = known_impairments.get("motor_difficulty", 0.0)

    # Compute cognitive load sensitivity from history
    cognitive_sensitivity = 0.0
    sensory_sensitivity = 0.0

    if eeg_history:
        n = len(eeg_history)
        # High cognitive load frequency
        high_load_count = sum(
            1 for s in eeg_history if s.cognitive_load > _HIGH_COGNITIVE_LOAD_THRESHOLD
        )
        cognitive_sensitivity = _clamp(high_load_count / max(n, 1), 0.0, 1.0)

        # Sensory sensitivity: frequent overwhelm (high arousal + stress)
        overwhelm_count = sum(
            1 for s in eeg_history
            if s.arousal > _OVERWHELM_AROUSAL_THRESHOLD
            and s.stress > _OVERWHELM_STRESS_THRESHOLD
        )
        sensory_sensitivity = _clamp(overwhelm_count / max(n, 1), 0.0, 1.0)

    # Override with known impairments if provided
    cognitive_sensitivity = max(
        cognitive_sensitivity,
        known_impairments.get("cognitive_load_sensitivity", 0.0),
    )
    sensory_sensitivity = max(
        sensory_sensitivity,
        known_impairments.get("sensory_sensitivity", 0.0),
    )

    return AccessibilityProfile(
        visual_impairment=_clamp(visual, 0.0, 1.0),
        motor_difficulty=_clamp(motor, 0.0, 1.0),
        cognitive_load_sensitivity=round(cognitive_sensitivity, 4),
        sensory_sensitivity=round(sensory_sensitivity, 4),
    )


def profile_to_dict(profile: AccessibilityProfile) -> Dict[str, Any]:
    """Serialize an AccessibilityProfile to a plain dict."""
    return asdict(profile)


def assessment_to_dict(assessment: AccessibilityAssessment) -> Dict[str, Any]:
    """Serialize an AccessibilityAssessment to a plain dict."""
    return {
        "eeg_state": asdict(assessment.eeg_state),
        "profile": asdict(assessment.profile),
        "adaptation": asdict(assessment.adaptation),
        "overwhelm_score": assessment.overwhelm_score,
        "fatigue_score": assessment.fatigue_score,
        "needs_intervention": assessment.needs_intervention,
        "timestamp": assessment.timestamp,
    }
