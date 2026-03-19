"""Self-report calibration model (issue #411).

Detects when users are wrong about their own emotions by measuring the
gap between self-reported emotion and objectively measured emotion (EEG,
voice biomarkers). Learns per-user bias patterns and classifies users into
reporter types.

References:
- Taylor et al. (1997) — Toronto Alexithymia Scale (TAS-20)
- Barrett et al. (2004) — Emotional granularity and interoceptive accuracy
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CalibrationObservation:
    """A single paired observation of reported vs measured emotion."""

    reported_valence: float      # user self-report (-1 to 1)
    reported_arousal: float      # user self-report (0 to 1)
    measured_valence: float      # EEG/voice measured (-1 to 1)
    measured_arousal: float      # EEG/voice measured (0 to 1)
    eeg_valence: Optional[float] = None
    voice_valence: Optional[float] = None
    channel_agreement: float = 0.0  # cosine similarity between EEG and voice
    time_h: float = 12.0            # time of day
    context: str = ""                # optional context tag


@dataclass
class GapStatistics:
    """Statistics about the emotion gap for a user."""

    mean_valence_gap: float       # avg(measured - reported) for valence
    mean_arousal_gap: float       # avg(measured - reported) for arousal
    std_valence_gap: float        # variability of valence gap
    std_arousal_gap: float        # variability of arousal gap
    abs_mean_valence_gap: float   # mean |gap| — magnitude regardless of direction
    abs_mean_arousal_gap: float
    n_observations: int


@dataclass
class BiasModel:
    """User-specific bias model: reported = alpha + beta * true."""

    alpha_valence: float    # constant bias (intercept)
    beta_valence: float     # gain/attenuation (slope)
    alpha_arousal: float
    beta_arousal: float
    r_squared_valence: float
    r_squared_arousal: float


@dataclass
class CalibrationProfile:
    """Full calibration profile for a user."""

    reporter_type: str                  # "accurate" | "suppressor" | "amplifier" | "inconsistent"
    reporter_confidence: float          # 0–1
    emotional_awareness_score: float    # 0–100
    gap_stats: GapStatistics
    bias_model: Optional[BiasModel]
    valence_blind_spots: List[str]      # e.g. ["under-reports stress", "over-reports happiness"]
    trend_direction: str                # "improving" | "stable" | "declining"
    calibrated_valence: Optional[float] # adjusted valence based on bias model
    calibrated_arousal: Optional[float]
    n_observations: int


def compute_gap_statistics(observations: List[CalibrationObservation]) -> GapStatistics:
    """Compute basic gap statistics from paired observations."""
    if not observations:
        return GapStatistics(
            mean_valence_gap=0.0, mean_arousal_gap=0.0,
            std_valence_gap=0.0, std_arousal_gap=0.0,
            abs_mean_valence_gap=0.0, abs_mean_arousal_gap=0.0,
            n_observations=0,
        )

    v_gaps = [o.measured_valence - o.reported_valence for o in observations]
    a_gaps = [o.measured_arousal - o.reported_arousal for o in observations]

    v_arr = np.array(v_gaps)
    a_arr = np.array(a_gaps)

    return GapStatistics(
        mean_valence_gap=float(np.mean(v_arr)),
        mean_arousal_gap=float(np.mean(a_arr)),
        std_valence_gap=float(np.std(v_arr)),
        std_arousal_gap=float(np.std(a_arr)),
        abs_mean_valence_gap=float(np.mean(np.abs(v_arr))),
        abs_mean_arousal_gap=float(np.mean(np.abs(a_arr))),
        n_observations=len(observations),
    )


def fit_bias_model(observations: List[CalibrationObservation]) -> Optional[BiasModel]:
    """Fit a linear bias model: reported = alpha + beta * measured.

    Requires at least 10 observations for reliable estimation.
    """
    if len(observations) < 10:
        return None

    measured_v = np.array([o.measured_valence for o in observations])
    reported_v = np.array([o.reported_valence for o in observations])
    measured_a = np.array([o.measured_arousal for o in observations])
    reported_a = np.array([o.reported_arousal for o in observations])

    def _fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """Returns (alpha, beta, r_squared)."""
        n = len(x)
        X = np.column_stack([np.ones(n), x])
        try:
            beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return 0.0, 1.0, 0.0

        y_hat = X @ beta_hat
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        return float(beta_hat[0]), float(beta_hat[1]), max(0.0, r_sq)

    av, bv, rv = _fit_linear(measured_v, reported_v)
    aa, ba, ra = _fit_linear(measured_a, reported_a)

    return BiasModel(
        alpha_valence=round(av, 4),
        beta_valence=round(bv, 4),
        alpha_arousal=round(aa, 4),
        beta_arousal=round(ba, 4),
        r_squared_valence=round(rv, 4),
        r_squared_arousal=round(ra, 4),
    )


def classify_reporter_type(
    gap_stats: GapStatistics,
) -> Tuple[str, float]:
    """Classify user into a reporter type based on gap patterns.

    Types:
    - accurate: |mean_gap| < 0.15 consistently
    - suppressor: mean_gap > 0.15 (reports better than measured)
    - amplifier: mean_gap < -0.15 (reports worse than measured)
    - inconsistent: high gap variance (std > 0.3)
    """
    mv = gap_stats.mean_valence_gap
    sv = gap_stats.std_valence_gap

    # High variance = inconsistent (possible alexithymia)
    if sv > 0.3 and gap_stats.abs_mean_valence_gap > 0.2:
        confidence = min(1.0, sv / 0.5)
        return "inconsistent", round(confidence, 3)

    # Systematic bias
    if mv > 0.15:
        # Measured > reported → user under-reports negativity → suppressor
        confidence = min(1.0, mv / 0.4)
        return "suppressor", round(confidence, 3)

    if mv < -0.15:
        # Measured < reported → user over-reports negativity → amplifier
        confidence = min(1.0, abs(mv) / 0.4)
        return "amplifier", round(confidence, 3)

    # Small gap = accurate reporter
    confidence = 1.0 - min(1.0, gap_stats.abs_mean_valence_gap / 0.15)
    return "accurate", round(max(0.1, confidence), 3)


def compute_awareness_score(gap_stats: GapStatistics) -> float:
    """Compute an Emotional Awareness Score (0–100).

    100 = perfect agreement between reported and measured
    0 = completely disconnected from own emotional state
    """
    if gap_stats.n_observations == 0:
        return 50.0  # no data = neutral

    # Combine mean absolute gap and gap variance
    # Lower gap + lower variance = higher awareness
    gap_penalty = gap_stats.abs_mean_valence_gap * 60  # max ~60 points lost from gap
    var_penalty = gap_stats.std_valence_gap * 40        # max ~40 points lost from inconsistency

    score = 100.0 - gap_penalty - var_penalty
    return round(max(0.0, min(100.0, score)), 1)


def detect_blind_spots(
    observations: List[CalibrationObservation],
) -> List[str]:
    """Identify specific emotional blind spots from observation patterns."""
    if len(observations) < 5:
        return []

    blind_spots = []

    # Check for systematic under-reporting of stress
    stress_gaps = [
        o.measured_valence - o.reported_valence
        for o in observations
        if o.measured_valence < -0.2  # measured negative state
    ]
    if len(stress_gaps) >= 3 and np.mean(stress_gaps) > 0.2:
        blind_spots.append("under-reports stress when objectively measured")

    # Check for over-reporting positive states
    happy_gaps = [
        o.reported_valence - o.measured_valence
        for o in observations
        if o.reported_valence > 0.3  # reports positive
    ]
    if len(happy_gaps) >= 3 and np.mean(happy_gaps) > 0.2:
        blind_spots.append("over-reports positive emotions relative to neural state")

    # Check for arousal mismatch
    arousal_gaps = [
        abs(o.measured_arousal - o.reported_arousal)
        for o in observations
    ]
    if len(arousal_gaps) >= 5 and np.mean(arousal_gaps) > 0.25:
        blind_spots.append("poor awareness of own arousal/energy level")

    # Check for time-of-day patterns
    afternoon_gaps = [
        o.measured_valence - o.reported_valence
        for o in observations
        if 13.0 <= o.time_h <= 17.0
    ]
    if len(afternoon_gaps) >= 3 and abs(np.mean(afternoon_gaps)) > 0.25:
        direction = "under-reports" if np.mean(afternoon_gaps) > 0 else "over-reports"
        blind_spots.append(f"{direction} afternoon emotional state")

    return blind_spots


def compute_calibration_profile(
    observations: List[CalibrationObservation],
    current_measured_valence: Optional[float] = None,
    current_measured_arousal: Optional[float] = None,
) -> CalibrationProfile:
    """Compute a full calibration profile from paired observations.

    Parameters
    ----------
    observations : list of CalibrationObservation
    current_measured_valence : optional current reading to calibrate
    current_measured_arousal : optional current reading to calibrate

    Returns
    -------
    CalibrationProfile with reporter type, awareness score, blind spots, etc.
    """
    gap_stats = compute_gap_statistics(observations)
    bias_model = fit_bias_model(observations)
    reporter_type, reporter_conf = classify_reporter_type(gap_stats)
    awareness_score = compute_awareness_score(gap_stats)
    blind_spots = detect_blind_spots(observations)

    # Determine trend (if enough data, compare first half vs second half)
    trend = "stable"
    if len(observations) >= 20:
        mid = len(observations) // 2
        first_half_gap = np.mean([
            abs(o.measured_valence - o.reported_valence)
            for o in observations[:mid]
        ])
        second_half_gap = np.mean([
            abs(o.measured_valence - o.reported_valence)
            for o in observations[mid:]
        ])
        diff = first_half_gap - second_half_gap
        if diff > 0.05:
            trend = "improving"
        elif diff < -0.05:
            trend = "declining"

    # Calibrate current reading using bias model
    cal_v = None
    cal_a = None
    if bias_model and current_measured_valence is not None:
        # Invert the bias: true ≈ (reported - alpha) / beta
        # But we want to give a "calibrated" prediction of what the user likely feels
        if abs(bias_model.beta_valence) > 0.1:
            cal_v = round(bias_model.alpha_valence + bias_model.beta_valence * current_measured_valence, 3)
    if bias_model and current_measured_arousal is not None:
        if abs(bias_model.beta_arousal) > 0.1:
            cal_a = round(bias_model.alpha_arousal + bias_model.beta_arousal * current_measured_arousal, 3)

    return CalibrationProfile(
        reporter_type=reporter_type,
        reporter_confidence=reporter_conf,
        emotional_awareness_score=awareness_score,
        gap_stats=gap_stats,
        bias_model=bias_model,
        valence_blind_spots=blind_spots,
        trend_direction=trend,
        calibrated_valence=cal_v,
        calibrated_arousal=cal_a,
        n_observations=len(observations),
    )


def profile_to_dict(profile: CalibrationProfile) -> Dict[str, Any]:
    """Serialize CalibrationProfile to a JSON-safe dict."""
    result: Dict[str, Any] = {
        "reporter_type": profile.reporter_type,
        "reporter_confidence": profile.reporter_confidence,
        "emotional_awareness_score": profile.emotional_awareness_score,
        "gap_statistics": {
            "mean_valence_gap": round(profile.gap_stats.mean_valence_gap, 4),
            "mean_arousal_gap": round(profile.gap_stats.mean_arousal_gap, 4),
            "std_valence_gap": round(profile.gap_stats.std_valence_gap, 4),
            "std_arousal_gap": round(profile.gap_stats.std_arousal_gap, 4),
            "abs_mean_valence_gap": round(profile.gap_stats.abs_mean_valence_gap, 4),
            "abs_mean_arousal_gap": round(profile.gap_stats.abs_mean_arousal_gap, 4),
            "n_observations": profile.gap_stats.n_observations,
        },
        "bias_model": None,
        "valence_blind_spots": profile.valence_blind_spots,
        "trend_direction": profile.trend_direction,
        "calibrated_valence": profile.calibrated_valence,
        "calibrated_arousal": profile.calibrated_arousal,
        "n_observations": profile.n_observations,
    }

    if profile.bias_model:
        result["bias_model"] = {
            "alpha_valence": profile.bias_model.alpha_valence,
            "beta_valence": profile.bias_model.beta_valence,
            "alpha_arousal": profile.bias_model.alpha_arousal,
            "beta_arousal": profile.bias_model.beta_arousal,
            "r_squared_valence": profile.bias_model.r_squared_valence,
            "r_squared_arousal": profile.bias_model.r_squared_arousal,
        }

    return result
