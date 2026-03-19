"""Circadian neural signature model.

Fits a personalized cosinor model to multi-modal time-stamped features
(EEG alpha/beta, valence, arousal, HRV, sleep) to discover the user's
true chronotype and circadian phase.

Reference: Refinetti et al. (2007) — Cosinor analysis procedures.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cosinor model: y(t) = M + A*cos(2*pi*(t - phi) / tau)
# Linearised as: y(t) = M + beta*cos(2*pi*t/tau) + gamma*sin(2*pi*t/tau)
# where A = sqrt(beta^2 + gamma^2), phi = atan2(gamma, beta) * tau/(2*pi)
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * math.pi
_DEFAULT_PERIOD_H = 24.0  # tau in hours


@dataclass
class CosinorFit:
    """Result of fitting a single cosinor model to one feature stream."""

    mesor: float          # M — midline estimating statistic of rhythm
    amplitude: float      # A — peak-to-trough / 2
    acrophase_h: float    # phi — time of peak in hours (0–24)
    period_h: float       # tau — fitted period (close to 24)
    r_squared: float      # goodness of fit
    p_value: float        # F-test significance (< 0.05 = significant rhythm)
    n_samples: int        # data points used


@dataclass
class CircadianProfile:
    """Full circadian profile for a user from multi-modal data."""

    chronotype: str                         # "early_bird" | "night_owl" | "intermediate"
    chronotype_confidence: float            # 0–1
    acrophase_h: float                      # weighted average peak time (hours)
    amplitude: float                        # weighted average amplitude
    period_h: float                         # weighted average period
    phase_stability: float                  # 0–1 (1 = very stable)
    current_phase: float                    # 0–1 (where in cycle right now)
    predicted_focus_window: str             # e.g. "9:30am – 12:00pm"
    predicted_slump_window: str             # e.g. "2:00pm – 3:30pm"
    phase_shift_hours: float               # shift from learned baseline (0 = normal)
    fits: Dict[str, CosinorFit] = field(default_factory=dict)
    data_days: int = 0                      # days of data used


def _fit_cosinor(
    times_h: np.ndarray,
    values: np.ndarray,
    period_h: float = _DEFAULT_PERIOD_H,
) -> CosinorFit:
    """Fit a single-component cosinor model via OLS.

    Parameters
    ----------
    times_h : array of float — time of day in fractional hours (0–24)
    values  : array of float — feature values (z-scored preferred)
    period_h: float — assumed period (default 24h)

    Returns
    -------
    CosinorFit with estimated parameters and fit statistics.
    """
    n = len(times_h)
    if n < 6:
        return CosinorFit(
            mesor=float(np.mean(values)),
            amplitude=0.0,
            acrophase_h=12.0,
            period_h=period_h,
            r_squared=0.0,
            p_value=1.0,
            n_samples=n,
        )

    omega = _TWO_PI / period_h
    cos_t = np.cos(omega * times_h)
    sin_t = np.sin(omega * times_h)

    # Design matrix: [1, cos(wt), sin(wt)]
    X = np.column_stack([np.ones(n), cos_t, sin_t])

    # OLS: beta = (X'X)^-1 X'y
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        beta_hat = XtX_inv @ (X.T @ values)
    except np.linalg.LinAlgError:
        return CosinorFit(
            mesor=float(np.mean(values)),
            amplitude=0.0,
            acrophase_h=12.0,
            period_h=period_h,
            r_squared=0.0,
            p_value=1.0,
            n_samples=n,
        )

    mesor = float(beta_hat[0])
    beta_c = float(beta_hat[1])
    gamma_s = float(beta_hat[2])

    amplitude = math.sqrt(beta_c**2 + gamma_s**2)
    acrophase_rad = math.atan2(gamma_s, beta_c)
    acrophase_h = (acrophase_rad / omega) % period_h

    # R-squared
    y_hat = X @ beta_hat
    ss_res = float(np.sum((values - y_hat) ** 2))
    ss_tot = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
    r_squared = max(0.0, r_squared)

    # F-test for rhythm significance (2 df numerator, n-3 df denominator)
    df_model = 2
    df_resid = n - 3
    if df_resid > 0 and ss_tot > 1e-10:
        ss_model = ss_tot - ss_res
        f_stat = (ss_model / df_model) / (ss_res / df_resid)
        # Approximate p-value using F-distribution survival function
        try:
            from scipy.stats import f as f_dist
            p_value = float(f_dist.sf(f_stat, df_model, df_resid))
        except ImportError:
            # Rough approximation without scipy
            p_value = 0.05 if f_stat > 3.0 else 0.5
    else:
        p_value = 1.0

    return CosinorFit(
        mesor=mesor,
        amplitude=amplitude,
        acrophase_h=acrophase_h,
        period_h=period_h,
        r_squared=r_squared,
        p_value=p_value,
        n_samples=n,
    )


def _classify_chronotype(acrophase_h: float) -> Tuple[str, float]:
    """Classify chronotype from the acrophase (peak alertness time).

    Early bird: peak before 10am
    Night owl: peak after 1pm
    Intermediate: 10am–1pm
    """
    if acrophase_h < 10.0:
        label = "early_bird"
        # confidence: how far from the boundary
        confidence = min(1.0, (10.0 - acrophase_h) / 3.0)
    elif acrophase_h > 13.0:
        label = "night_owl"
        confidence = min(1.0, (acrophase_h - 13.0) / 3.0)
    else:
        label = "intermediate"
        # middle of the range = highest confidence
        mid = 11.5
        confidence = 1.0 - min(1.0, abs(acrophase_h - mid) / 1.5)
    return label, round(confidence, 3)


def _format_time(hours: float) -> str:
    """Convert fractional hours to 'H:MMam/pm' string."""
    h = int(hours) % 24
    m = int((hours - int(hours)) * 60)
    period = "am" if h < 12 else "pm"
    display_h = h if h <= 12 else h - 12
    if display_h == 0:
        display_h = 12
    return f"{display_h}:{m:02d}{period}"


def _predict_windows(
    acrophase_h: float,
    amplitude: float,
) -> Tuple[str, str]:
    """Predict focus and slump windows from circadian peak.

    Focus window: centered on acrophase (±1.5h)
    Slump window: ~6h after peak (circadian trough)
    """
    focus_start = (acrophase_h - 1.5) % 24
    focus_end = (acrophase_h + 1.5) % 24
    focus_window = f"{_format_time(focus_start)} – {_format_time(focus_end)}"

    # Post-lunch dip / trough is roughly half-period after peak
    slump_center = (acrophase_h + 6.0) % 24
    slump_start = (slump_center - 0.75) % 24
    slump_end = (slump_center + 0.75) % 24
    slump_window = f"{_format_time(slump_start)} – {_format_time(slump_end)}"

    return focus_window, slump_window


def compute_circadian_profile(
    feature_streams: Dict[str, List[Dict[str, Any]]],
    current_hour: float = 12.0,
    baseline_acrophase: Optional[float] = None,
) -> CircadianProfile:
    """Compute a circadian profile from multiple feature streams.

    Parameters
    ----------
    feature_streams : dict mapping stream name to list of
        {"time_h": float (0-24), "value": float} entries.
        Expected streams: "alpha_beta_ratio", "valence", "arousal",
        "hrv", "stress", "focus"
    current_hour : float — current time of day (fractional hours)
    baseline_acrophase : float or None — previous learned acrophase for
        computing phase shift

    Returns
    -------
    CircadianProfile with all computed fields.
    """
    # Weights for each modality (EEG alpha is most reliable per literature)
    stream_weights = {
        "alpha_beta_ratio": 0.30,
        "valence": 0.20,
        "arousal": 0.15,
        "hrv": 0.15,
        "stress": 0.10,
        "focus": 0.10,
    }

    fits: Dict[str, CosinorFit] = {}
    weighted_acrophase_x = 0.0  # circular mean components
    weighted_acrophase_y = 0.0
    weighted_amplitude = 0.0
    total_weight = 0.0
    max_days = 0

    for stream_name, data_points in feature_streams.items():
        if not data_points or len(data_points) < 6:
            continue

        times = np.array([d["time_h"] for d in data_points], dtype=np.float64)
        values = np.array([d["value"] for d in data_points], dtype=np.float64)

        # Z-score normalize within stream
        std = np.std(values)
        if std > 1e-10:
            values = (values - np.mean(values)) / std

        fit = _fit_cosinor(times, values)
        fits[stream_name] = fit

        # Only include significant rhythms (p < 0.10 to be generous with limited data)
        if fit.p_value < 0.10 and fit.amplitude > 0.01:
            w = stream_weights.get(stream_name, 0.1)
            # Weight by R-squared quality too
            effective_w = w * fit.r_squared
            omega = _TWO_PI / 24.0
            weighted_acrophase_x += effective_w * math.cos(omega * fit.acrophase_h)
            weighted_acrophase_y += effective_w * math.sin(omega * fit.acrophase_h)
            weighted_amplitude += effective_w * fit.amplitude
            total_weight += effective_w

        # Estimate days of data from time spread
        if len(times) > 1:
            unique_days = len(set(int(t) for t in times))
            max_days = max(max_days, unique_days)

    # Compute weighted average acrophase using circular mean
    if total_weight > 1e-10:
        mean_angle = math.atan2(weighted_acrophase_y, weighted_acrophase_x)
        acrophase_h = (mean_angle / (_TWO_PI / 24.0)) % 24.0
        amplitude = weighted_amplitude / total_weight

        # Phase stability: how tightly clustered are the individual acrophases?
        r_vec = math.sqrt(weighted_acrophase_x**2 + weighted_acrophase_y**2) / total_weight
        phase_stability = min(1.0, r_vec)
    else:
        # Insufficient data — fall back to population average
        acrophase_h = 10.5  # population average peak alertness
        amplitude = 0.0
        phase_stability = 0.0

    # Chronotype classification
    chronotype, chronotype_conf = _classify_chronotype(acrophase_h)

    # Current circadian phase (0 = trough, 0.5 = peak, 1 = next trough)
    omega = _TWO_PI / 24.0
    phase_angle = omega * (current_hour - acrophase_h)
    current_phase = (math.cos(phase_angle) + 1.0) / 2.0  # normalized 0–1

    # Focus and slump windows
    focus_window, slump_window = _predict_windows(acrophase_h, amplitude)

    # Phase shift from baseline
    phase_shift = 0.0
    if baseline_acrophase is not None and total_weight > 1e-10:
        diff = acrophase_h - baseline_acrophase
        # Handle wrap-around
        if diff > 12:
            diff -= 24
        elif diff < -12:
            diff += 24
        phase_shift = round(diff, 2)

    return CircadianProfile(
        chronotype=chronotype,
        chronotype_confidence=chronotype_conf,
        acrophase_h=round(acrophase_h, 2),
        amplitude=round(amplitude, 4),
        period_h=24.0,
        phase_stability=round(phase_stability, 3),
        current_phase=round(current_phase, 3),
        predicted_focus_window=focus_window,
        predicted_slump_window=slump_window,
        phase_shift_hours=phase_shift,
        fits=fits,
        data_days=max_days,
    )


def profile_to_dict(profile: CircadianProfile) -> Dict[str, Any]:
    """Serialize a CircadianProfile to a JSON-safe dict."""
    fits_dict = {}
    for name, fit in profile.fits.items():
        fits_dict[name] = {
            "mesor": round(fit.mesor, 4),
            "amplitude": round(fit.amplitude, 4),
            "acrophase_h": round(fit.acrophase_h, 2),
            "period_h": round(fit.period_h, 2),
            "r_squared": round(fit.r_squared, 4),
            "p_value": round(fit.p_value, 6),
            "n_samples": fit.n_samples,
        }

    return {
        "chronotype": profile.chronotype,
        "chronotype_confidence": profile.chronotype_confidence,
        "acrophase_h": profile.acrophase_h,
        "amplitude": profile.amplitude,
        "period_h": profile.period_h,
        "phase_stability": profile.phase_stability,
        "current_phase": profile.current_phase,
        "predicted_focus_window": profile.predicted_focus_window,
        "predicted_slump_window": profile.predicted_slump_window,
        "phase_shift_hours": profile.phase_shift_hours,
        "fits": fits_dict,
        "data_days": profile.data_days,
    }
