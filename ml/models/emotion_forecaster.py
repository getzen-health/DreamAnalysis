"""Emotion forecasting with temporal attention.

Predicts tomorrow's mood from historical emotional patterns, using a
lightweight temporal attention mechanism inspired by Transformer architectures.
No deep learning framework required -- implemented in pure numpy.

Based on:
    JMIR Mental Health (2025): daily emotional summaries predict next-day mood
    with r=0.6-0.7 when including sleep + activity context.

Input: 7-30 days of daily emotional summaries:
    - valence (-1 to 1)
    - arousal (0 to 1)
    - stress (0 to 1)
    - sleep_quality (0 to 1)
    - activity_level (0 to 1)

Output:
    - Forecasted emotion state for next 1, 3, 7 days
    - Confidence intervals
    - Weekly pattern detection
    - Feature importance ranking
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Feature names in canonical order
_FEATURES = ["valence", "arousal", "stress", "sleep_quality", "activity_level"]

# Minimum days for meaningful forecast
_MIN_DAYS = 3

# Maximum history to retain
_MAX_DAYS = 90


def prepare_forecast_input(daily_summaries: List[Dict]) -> Dict:
    """Validate and normalize daily summaries into forecast-ready arrays.

    Each summary should contain: valence, arousal, stress, sleep_quality,
    activity_level. Missing values are imputed with rolling mean.

    Args:
        daily_summaries: List of dicts, one per day, most recent last.

    Returns:
        Dict with 'matrix' (n_days, 5), 'features', 'n_days', or 'error'.
    """
    if not daily_summaries:
        return {"error": "no_data", "detail": "need at least 3 days of history"}

    n = min(len(daily_summaries), _MAX_DAYS)
    recent = daily_summaries[-n:]

    matrix = np.zeros((n, len(_FEATURES)), dtype=float)
    for i, day in enumerate(recent):
        for j, feat in enumerate(_FEATURES):
            val = day.get(feat)
            if val is not None:
                matrix[i, j] = float(val)
            else:
                # Impute with mean of available values for this feature
                matrix[i, j] = np.nan

    # Impute NaNs with column means
    for j in range(len(_FEATURES)):
        col = matrix[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            col_mean = float(np.nanmean(col)) if not nan_mask.all() else 0.0
            col[nan_mask] = col_mean

    if n < _MIN_DAYS:
        return {
            "error": "insufficient_data",
            "detail": f"need at least {_MIN_DAYS} days, got {n}",
            "matrix": matrix,
            "features": _FEATURES,
            "n_days": n,
        }

    return {
        "matrix": matrix,
        "features": list(_FEATURES),
        "n_days": n,
    }


def forecast_emotion(
    prepared: Dict, horizon: int = 1
) -> Dict:
    """Forecast emotion state for the next N days.

    Uses temporal attention to weight recent days more heavily and
    linear trend extrapolation for multi-step forecasts.

    Args:
        prepared: Output of prepare_forecast_input().
        horizon: Number of days to forecast (1, 3, or 7).

    Returns:
        Dict with forecasted values per feature per day, or error.
    """
    if "error" in prepared:
        return prepared

    matrix = prepared["matrix"]  # (n_days, n_features)
    n_days, n_features = matrix.shape

    horizon = max(1, min(horizon, 7))

    # Temporal attention weights: exponential decay favoring recent days
    attention = _compute_temporal_attention(n_days)

    # Weighted mean (attention-weighted recent trend)
    weighted_means = np.zeros(n_features)
    for j in range(n_features):
        weighted_means[j] = float(np.sum(attention * matrix[:, j]))

    # Linear trend per feature (slope from last 7 days or all available)
    trend_window = min(n_days, 7)
    slopes = np.zeros(n_features)
    for j in range(n_features):
        slopes[j] = _linear_slope(matrix[-trend_window:, j])

    # Generate forecasts for each day in the horizon
    forecasts: List[Dict] = []
    for h in range(1, horizon + 1):
        day_forecast: Dict = {"day": h}
        for j, feat in enumerate(_FEATURES):
            # Forecast = weighted mean + slope * horizon
            val = weighted_means[j] + slopes[j] * h
            # Clamp to valid ranges
            if feat == "valence":
                val = float(np.clip(val, -1.0, 1.0))
            else:
                val = float(np.clip(val, 0.0, 1.0))
            day_forecast[feat] = round(val, 4)
        forecasts.append(day_forecast)

    return {
        "horizon_days": horizon,
        "forecasts": forecasts,
        "attention_weights": attention.tolist(),
        "trend_slopes": {
            feat: round(float(slopes[j]), 6) for j, feat in enumerate(_FEATURES)
        },
    }


def compute_forecast_confidence(prepared: Dict, horizon: int = 1) -> Dict:
    """Compute confidence intervals for emotion forecasts.

    Uses historical variance scaled by horizon distance. Wider intervals
    for longer forecasts and more variable features.

    Args:
        prepared: Output of prepare_forecast_input().
        horizon: Forecast horizon in days.

    Returns:
        Dict with confidence intervals per feature per day.
    """
    if "error" in prepared:
        return prepared

    matrix = prepared["matrix"]
    n_days, n_features = matrix.shape

    horizon = max(1, min(horizon, 7))

    # Compute per-feature standard deviation
    feature_stds = np.std(matrix, axis=0)

    # Confidence intervals widen with horizon
    intervals: List[Dict] = []
    for h in range(1, horizon + 1):
        day_ci: Dict = {"day": h}
        # Uncertainty grows with sqrt(horizon)
        scale = np.sqrt(h)
        for j, feat in enumerate(_FEATURES):
            width = float(feature_stds[j] * scale * 1.96)  # 95% CI
            day_ci[feat] = {
                "ci_width": round(width, 4),
                "confidence": round(float(np.clip(1.0 / (1.0 + width), 0.1, 0.99)), 3),
            }
        intervals.append(day_ci)

    # Overall confidence score (mean across features for day 1)
    if intervals:
        day1_confs = [
            intervals[0][feat]["confidence"]
            for feat in _FEATURES
            if feat in intervals[0]
        ]
        overall = float(np.mean(day1_confs)) if day1_confs else 0.5
    else:
        overall = 0.5

    return {
        "horizon_days": horizon,
        "intervals": intervals,
        "overall_confidence": round(overall, 3),
    }


def detect_weekly_pattern(prepared: Dict) -> Dict:
    """Detect recurring weekly patterns in emotional data.

    Identifies which day of the week tends to have higher/lower values
    for each feature. Requires at least 14 days of data.

    Args:
        prepared: Output of prepare_forecast_input().

    Returns:
        Dict with per-feature weekly pattern (day 0=Mon through 6=Sun).
    """
    if "error" in prepared:
        return prepared

    matrix = prepared["matrix"]
    n_days, n_features = matrix.shape

    if n_days < 7:
        return {
            "pattern_detected": False,
            "reason": f"need at least 7 days, got {n_days}",
        }

    # Group by day-of-week (assume most recent entry is today)
    # We just use modular indexing since we don't have actual dates
    day_means: Dict[str, List[float]] = {feat: [] for feat in _FEATURES}
    day_pattern = np.zeros((7, n_features))
    day_counts = np.zeros(7)

    for i in range(n_days):
        dow = (n_days - 1 - i) % 7  # most recent = day 0 position
        day_pattern[dow] += matrix[n_days - 1 - i]
        day_counts[dow] += 1

    # Normalize by count
    for d in range(7):
        if day_counts[d] > 0:
            day_pattern[d] /= day_counts[d]

    # Check if there's meaningful variation across days
    pattern_strength: Dict[str, float] = {}
    for j, feat in enumerate(_FEATURES):
        day_vals = day_pattern[:, j]
        if np.std(day_vals) > 0.01:
            # Ratio of between-day variance to total variance
            strength = float(np.std(day_vals) / max(np.std(matrix[:, j]), 1e-9))
            pattern_strength[feat] = round(min(strength, 1.0), 3)
        else:
            pattern_strength[feat] = 0.0

    has_pattern = any(v > 0.1 for v in pattern_strength.values())

    weekly: Dict[str, Dict] = {}
    day_names = ["day_0", "day_1", "day_2", "day_3", "day_4", "day_5", "day_6"]
    for j, feat in enumerate(_FEATURES):
        weekly[feat] = {
            day_names[d]: round(float(day_pattern[d, j]), 4) for d in range(7)
        }

    return {
        "pattern_detected": has_pattern,
        "pattern_strength": pattern_strength,
        "weekly_averages": weekly,
        "days_analyzed": int(n_days),
    }


def compute_feature_importance(prepared: Dict) -> Dict:
    """Rank which input features most influence the forecast.

    Uses correlation between each feature and next-day valence as a
    proxy for predictive importance.

    Args:
        prepared: Output of prepare_forecast_input().

    Returns:
        Dict with ranked feature importances.
    """
    if "error" in prepared:
        return prepared

    matrix = prepared["matrix"]
    n_days, n_features = matrix.shape

    if n_days < 4:
        return {
            "importances": {feat: round(1.0 / n_features, 3) for feat in _FEATURES},
            "ranking": list(_FEATURES),
            "method": "uniform_insufficient_data",
        }

    # Target: next-day valence (feature index 0)
    target = matrix[1:, 0]  # valence shifted by 1 day
    predictors = matrix[:-1]  # features from previous day

    importances: Dict[str, float] = {}
    for j, feat in enumerate(_FEATURES):
        x = predictors[:, j]
        corr = _pearson_correlation(x, target)
        importances[feat] = round(abs(corr), 4)

    # Normalize to sum to 1
    total = sum(importances.values())
    if total > 0:
        importances = {k: round(v / total, 4) for k, v in importances.items()}

    # Sort by importance
    ranking = sorted(importances, key=lambda k: importances[k], reverse=True)

    return {
        "importances": importances,
        "ranking": ranking,
        "method": "next_day_valence_correlation",
    }


def forecast_to_dict(forecast_result: Dict) -> Dict:
    """Normalize forecast result to JSON-safe dictionary.

    Args:
        forecast_result: Output of forecast_emotion() or other functions.

    Returns:
        JSON-serializable dict.
    """
    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    return _convert(forecast_result)


# -- internal helpers ---------------------------------------------------------


def _compute_temporal_attention(n_days: int, decay: float = 0.15) -> np.ndarray:
    """Compute exponential decay attention weights.

    Most recent day gets highest weight. Normalized to sum to 1.

    Args:
        n_days: Number of days.
        decay: Decay rate (higher = faster decay of older days).

    Returns:
        Array of shape (n_days,) summing to 1.0.
    """
    positions = np.arange(n_days, dtype=float)
    # Higher index = more recent = higher weight
    weights = np.exp(decay * positions)
    return weights / np.sum(weights)


def _linear_slope(values: np.ndarray) -> float:
    """Compute linear trend slope from a 1-D array."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = np.mean(x)
    y_mean = np.mean(values)
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-12:
        return 0.0
    return float(np.sum((x - x_mean) * (values - y_mean)) / denom)


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std < 1e-12 or y_std < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# -- module-level accessor ----------------------------------------------------


def get_forecaster():
    """Return a dict of forecasting functions."""
    return {
        "prepare_forecast_input": prepare_forecast_input,
        "forecast_emotion": forecast_emotion,
        "compute_forecast_confidence": compute_forecast_confidence,
        "detect_weekly_pattern": detect_weekly_pattern,
        "compute_feature_importance": compute_feature_importance,
        "forecast_to_dict": forecast_to_dict,
    }
