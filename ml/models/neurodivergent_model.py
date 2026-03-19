"""Neurodivergent emotion model (issue #413).

Supports alternative emotion input modalities for neurodivergent users
who may not relate to traditional valence/arousal self-report scales.
Provides color mapping, energy battery, and sensory state interfaces
that translate to standard valence/arousal space.

Includes ADHD-adapted intensity/volatility dimension with rejection
sensitivity tracking, and per-user calibration that learns individual
baselines before classifying.

References:
- Russell (1980) — Circumplex model of affect
- Barkley (2015) — ADHD and emotional dysregulation
- Dodson (2022) — Rejection Sensitive Dysphoria in ADHD
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color-to-emotion mapping
# Based on empirical colour-emotion associations (Jonauskaite et al. 2020)
# ---------------------------------------------------------------------------

# Each color maps to (valence, arousal) in [-1, 1] x [0, 1]
_COLOR_MAP: Dict[str, Tuple[float, float]] = {
    # Warm colors — positive valence, higher arousal
    "red": (0.1, 0.9),
    "orange": (0.6, 0.7),
    "yellow": (0.8, 0.6),
    "gold": (0.7, 0.55),
    # Cool colors — lower or negative valence
    "blue": (-0.2, 0.3),
    "dark_blue": (-0.5, 0.2),
    "purple": (-0.1, 0.4),
    "indigo": (-0.3, 0.3),
    # Nature / neutral
    "green": (0.4, 0.35),
    "teal": (0.2, 0.3),
    "brown": (-0.1, 0.25),
    # Extremes
    "white": (0.3, 0.2),
    "gray": (-0.3, 0.15),
    "black": (-0.6, 0.2),
    "pink": (0.5, 0.5),
}


# ---------------------------------------------------------------------------
# Energy battery mapping
# Metaphor: phone battery level + charging direction
# ---------------------------------------------------------------------------

_ENERGY_LEVELS: Dict[str, Tuple[float, float]] = {
    # label -> (valence, arousal)
    "fully_charged": (0.7, 0.85),
    "high": (0.5, 0.7),
    "moderate": (0.2, 0.5),
    "low": (-0.2, 0.3),
    "critical": (-0.5, 0.15),
    "drained": (-0.7, 0.05),
    # Charging states (recovering)
    "charging": (0.3, 0.4),
    "overcharged": (0.1, 0.95),       # hypomanic / overstimulated
}


# ---------------------------------------------------------------------------
# Sensory state mapping (for sensory processing differences)
# ---------------------------------------------------------------------------

_SENSORY_STATES: Dict[str, Tuple[float, float]] = {
    "regulated": (0.5, 0.45),
    "seeking": (0.2, 0.65),            # looking for stimulation
    "overstimulated": (-0.4, 0.85),    # too much input
    "understimulated": (-0.3, 0.15),   # not enough input, bored/flat
    "meltdown": (-0.8, 0.95),          # sensory overload crisis
    "shutdown": (-0.6, 0.05),          # withdrawn / dissociated
    "flow": (0.7, 0.6),               # optimal sensory engagement
}


# ---------------------------------------------------------------------------
# ADHD intensity model
# ---------------------------------------------------------------------------

@dataclass
class ADHDEmotionProfile:
    """ADHD-adapted emotional profile with intensity and volatility."""

    intensity: float                 # 0-1: how strongly emotions are felt
    volatility: float                # 0-1: how rapidly emotions shift
    rejection_sensitivity: float     # 0-1: RSD score
    hyperfocus_valence: float        # valence during hyperfocus episodes
    emotional_inertia: float         # 0-1: difficulty shifting out of emotion
    current_state: str               # "regulated"|"dysregulated"|"hyperfocused"|"rsd_triggered"


@dataclass
class CalibrationData:
    """Per-user calibration state that learns individual baselines."""

    user_id: str
    valence_samples: List[float] = field(default_factory=list)
    arousal_samples: List[float] = field(default_factory=list)
    valence_baseline: float = 0.0
    arousal_baseline: float = 0.5
    valence_std: float = 0.3
    arousal_std: float = 0.2
    is_calibrated: bool = False
    n_samples: int = 0
    min_samples: int = 10

    def add_sample(self, valence: float, arousal: float) -> None:
        """Add a calibration sample and update baseline stats."""
        self.valence_samples.append(valence)
        self.arousal_samples.append(arousal)
        self.n_samples = len(self.valence_samples)

        if self.n_samples >= self.min_samples:
            self.valence_baseline = float(np.mean(self.valence_samples))
            self.arousal_baseline = float(np.mean(self.arousal_samples))
            self.valence_std = max(0.05, float(np.std(self.valence_samples)))
            self.arousal_std = max(0.05, float(np.std(self.arousal_samples)))
            self.is_calibrated = True


@dataclass
class NeurodivergentProfile:
    """Full neurodivergent-adapted emotional profile."""

    valence: float                       # -1 to 1
    arousal: float                       # 0 to 1
    input_modality: str                  # "color"|"energy"|"sensory"|"direct"
    raw_input: str                       # original input value
    adhd_profile: Optional[ADHDEmotionProfile] = None
    calibration_applied: bool = False
    confidence: float = 0.5


# ---------------------------------------------------------------------------
# Core mapping functions
# ---------------------------------------------------------------------------

def map_color_to_emotion(
    color: str,
    intensity: float = 1.0,
) -> Dict[str, Any]:
    """Map a color selection to valence/arousal space.

    Parameters
    ----------
    color : str — color name (e.g. "red", "blue", "dark_blue")
    intensity : float — 0-1 multiplier for how strongly the user identifies
        with this color right now (default 1.0 = full)

    Returns
    -------
    Dict with valence, arousal, confidence, color, and raw mapping.
    """
    intensity = max(0.0, min(1.0, intensity))
    color_lower = color.lower().strip().replace(" ", "_")

    if color_lower in _COLOR_MAP:
        base_valence, base_arousal = _COLOR_MAP[color_lower]
        confidence = 0.7
    else:
        # Unknown color — neutral with low confidence
        base_valence, base_arousal = 0.0, 0.4
        confidence = 0.2
        log.warning("Unknown color '%s', using neutral mapping", color)

    valence = float(np.clip(base_valence * intensity, -1.0, 1.0))
    arousal = float(np.clip(base_arousal * intensity, 0.0, 1.0))

    return {
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "confidence": round(confidence, 2),
        "color": color_lower,
        "modality": "color",
    }


def map_energy_to_emotion(
    energy_level: str,
    trend: str = "stable",
) -> Dict[str, Any]:
    """Map an energy battery level to valence/arousal space.

    Parameters
    ----------
    energy_level : str — battery label (e.g. "high", "drained", "charging")
    trend : str — "rising", "falling", or "stable"

    Returns
    -------
    Dict with valence, arousal, confidence, energy_level, and trend adjustment.
    """
    level_lower = energy_level.lower().strip().replace(" ", "_")

    if level_lower in _ENERGY_LEVELS:
        base_valence, base_arousal = _ENERGY_LEVELS[level_lower]
        confidence = 0.75
    else:
        base_valence, base_arousal = 0.0, 0.4
        confidence = 0.2
        log.warning("Unknown energy level '%s', using neutral mapping", energy_level)

    # Trend adjustment
    trend_lower = trend.lower().strip()
    trend_valence_adj = 0.0
    trend_arousal_adj = 0.0
    if trend_lower == "rising":
        trend_valence_adj = 0.1
        trend_arousal_adj = 0.05
    elif trend_lower == "falling":
        trend_valence_adj = -0.1
        trend_arousal_adj = -0.05

    valence = float(np.clip(base_valence + trend_valence_adj, -1.0, 1.0))
    arousal = float(np.clip(base_arousal + trend_arousal_adj, 0.0, 1.0))

    return {
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "confidence": round(confidence, 2),
        "energy_level": level_lower,
        "trend": trend_lower,
        "modality": "energy",
    }


def map_sensory_state_to_emotion(
    state: str,
    sensory_domains: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Map a sensory processing state to valence/arousal space.

    Parameters
    ----------
    state : str — sensory state label (e.g. "overstimulated", "regulated")
    sensory_domains : list of str or None — which senses are affected
        (e.g. ["auditory", "visual", "tactile"])

    Returns
    -------
    Dict with valence, arousal, confidence, state, and affected domains.
    """
    state_lower = state.lower().strip().replace(" ", "_")
    domains = sensory_domains or []

    if state_lower in _SENSORY_STATES:
        base_valence, base_arousal = _SENSORY_STATES[state_lower]
        confidence = 0.7
    else:
        base_valence, base_arousal = 0.0, 0.4
        confidence = 0.2
        log.warning("Unknown sensory state '%s', using neutral mapping", state)

    # Multi-domain impact: more affected senses = more extreme
    if len(domains) > 2:
        # Scale toward extremes when many senses are affected
        scale = 1.0 + 0.1 * min(len(domains) - 2, 3)
        base_valence = float(np.clip(base_valence * scale, -1.0, 1.0))
        base_arousal = float(np.clip(base_arousal * scale, 0.0, 1.0))

    return {
        "valence": round(base_valence, 3),
        "arousal": round(base_arousal, 3),
        "confidence": round(confidence, 2),
        "sensory_state": state_lower,
        "affected_domains": domains,
        "modality": "sensory",
    }


# ---------------------------------------------------------------------------
# ADHD emotion model
# ---------------------------------------------------------------------------

def _compute_volatility(emotion_history: List[float]) -> float:
    """Compute emotional volatility from a sequence of valence readings.

    Volatility is the mean absolute change between consecutive readings,
    normalized to [0, 1].
    """
    if len(emotion_history) < 2:
        return 0.0

    diffs = np.abs(np.diff(emotion_history))
    # Max possible diff is 2.0 (from -1 to +1)
    raw = float(np.mean(diffs))
    return min(1.0, raw / 1.0)  # normalize: 1.0 diff = full volatility


def _compute_rejection_sensitivity(
    negative_spikes: int,
    total_readings: int,
    spike_magnitude_avg: float = 0.0,
) -> float:
    """Estimate rejection sensitivity from pattern of sharp negative drops.

    Parameters
    ----------
    negative_spikes : int — count of sharp negative valence drops (> 0.3 in one step)
    total_readings : int — total number of emotion readings
    spike_magnitude_avg : float — average magnitude of negative spikes

    Returns
    -------
    RSD score 0-1.
    """
    if total_readings < 5:
        return 0.0

    spike_rate = negative_spikes / total_readings
    # Weight by frequency and magnitude
    rsd = min(1.0, spike_rate * 5.0 + spike_magnitude_avg * 0.5)
    return round(rsd, 3)


def compute_neurodivergent_profile(
    valence: float,
    arousal: float,
    input_modality: str = "direct",
    raw_input: str = "",
    emotion_history: Optional[List[float]] = None,
    negative_spikes: int = 0,
    total_readings: int = 0,
    spike_magnitude_avg: float = 0.0,
    calibration: Optional[CalibrationData] = None,
) -> NeurodivergentProfile:
    """Compute a full neurodivergent-adapted emotional profile.

    Parameters
    ----------
    valence : float — current valence (-1 to 1)
    arousal : float — current arousal (0 to 1)
    input_modality : str — which input modality produced the valence/arousal
    raw_input : str — the original user input
    emotion_history : list of float or None — recent valence readings for
        ADHD volatility computation
    negative_spikes : int — count of sharp negative drops
    total_readings : int — total readings for RSD computation
    spike_magnitude_avg : float — average magnitude of negative spikes
    calibration : CalibrationData or None — per-user calibration

    Returns
    -------
    NeurodivergentProfile with all computed fields.
    """
    history = emotion_history or []

    # Apply calibration if available
    calibration_applied = False
    if calibration is not None and calibration.is_calibrated:
        # Z-score normalize against user's baseline
        valence = float(np.clip(
            (valence - calibration.valence_baseline) / calibration.valence_std,
            -1.0, 1.0,
        ))
        arousal = float(np.clip(
            (arousal - calibration.arousal_baseline) / calibration.arousal_std * 0.5 + 0.5,
            0.0, 1.0,
        ))
        calibration_applied = True

    # ADHD profile
    intensity = float(np.clip(abs(valence) + arousal * 0.5, 0.0, 1.0))
    volatility = _compute_volatility(history)
    rsd = _compute_rejection_sensitivity(
        negative_spikes, total_readings, spike_magnitude_avg,
    )

    # Determine current state
    if rsd > 0.6 and valence < -0.3:
        current_state = "rsd_triggered"
    elif intensity > 0.7 and volatility < 0.2 and valence > 0.3:
        current_state = "hyperfocused"
    elif volatility > 0.5 or intensity > 0.8:
        current_state = "dysregulated"
    else:
        current_state = "regulated"

    # Emotional inertia: high intensity + low volatility = stuck in emotion
    emotional_inertia = max(0.0, intensity - volatility * 0.5)
    emotional_inertia = min(1.0, emotional_inertia)

    adhd = ADHDEmotionProfile(
        intensity=round(intensity, 3),
        volatility=round(volatility, 3),
        rejection_sensitivity=rsd,
        hyperfocus_valence=round(valence, 3) if current_state == "hyperfocused" else 0.0,
        emotional_inertia=round(emotional_inertia, 3),
        current_state=current_state,
    )

    # Confidence based on data quality
    base_confidence = 0.5
    if calibration_applied:
        base_confidence += 0.2
    if len(history) >= 10:
        base_confidence += 0.15
    if input_modality != "direct":
        base_confidence += 0.05  # structured input slightly more reliable
    confidence = min(1.0, base_confidence)

    return NeurodivergentProfile(
        valence=round(valence, 3),
        arousal=round(arousal, 3),
        input_modality=input_modality,
        raw_input=raw_input,
        adhd_profile=adhd,
        calibration_applied=calibration_applied,
        confidence=round(confidence, 2),
    )


# ---------------------------------------------------------------------------
# Threshold adaptation
# ---------------------------------------------------------------------------

def adapt_emotion_thresholds(
    calibration: CalibrationData,
    population_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Adapt standard emotion classification thresholds to an individual user.

    Uses the user's calibrated baseline to shift thresholds so that
    "positive" and "negative" are relative to THEIR normal, not population
    average.

    Parameters
    ----------
    calibration : CalibrationData — user's learned baseline
    population_thresholds : dict or None — standard thresholds to adapt.
        Defaults to typical population values.

    Returns
    -------
    Dict of adapted thresholds.
    """
    defaults: Dict[str, float] = {
        "positive_valence": 0.2,
        "negative_valence": -0.2,
        "high_arousal": 0.65,
        "low_arousal": 0.35,
        "high_intensity": 0.7,
        "rsd_trigger": -0.3,
    }
    thresholds = population_thresholds if population_thresholds is not None else defaults

    if not calibration.is_calibrated:
        return thresholds

    adapted = {}
    for key, value in thresholds.items():
        if "valence" in key or "rsd" in key:
            # Shift valence thresholds by user's baseline
            adapted[key] = round(value + calibration.valence_baseline, 3)
        elif "arousal" in key:
            # Shift arousal thresholds by user's baseline
            adapted[key] = round(value + (calibration.arousal_baseline - 0.5), 3)
        elif "intensity" in key:
            # Scale intensity threshold by user's emotional range
            range_factor = calibration.valence_std / 0.3  # 0.3 is population average std
            adapted[key] = round(value * range_factor, 3)
        else:
            adapted[key] = value

    return adapted


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def profile_to_dict(profile: NeurodivergentProfile) -> Dict[str, Any]:
    """Serialize a NeurodivergentProfile to a JSON-safe dict."""
    result: Dict[str, Any] = {
        "valence": profile.valence,
        "arousal": profile.arousal,
        "input_modality": profile.input_modality,
        "raw_input": profile.raw_input,
        "calibration_applied": profile.calibration_applied,
        "confidence": profile.confidence,
    }

    if profile.adhd_profile is not None:
        result["adhd_profile"] = {
            "intensity": profile.adhd_profile.intensity,
            "volatility": profile.adhd_profile.volatility,
            "rejection_sensitivity": profile.adhd_profile.rejection_sensitivity,
            "hyperfocus_valence": profile.adhd_profile.hyperfocus_valence,
            "emotional_inertia": profile.adhd_profile.emotional_inertia,
            "current_state": profile.adhd_profile.current_state,
        }

    return result
