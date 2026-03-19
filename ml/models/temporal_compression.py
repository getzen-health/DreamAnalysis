"""Temporal emotion compression — compress months of emotional data into time-lapsed summaries.

Compresses long emotional timelines into digestible summaries by:
1. Extracting key moments: the most emotionally significant points in time
2. Detecting emotional arcs: narrative structures (rising action, climax,
   falling action, resolution)
3. Generating time-lapse data: compressed representations for visualization
4. Building compressed timelines: period-by-period emotional summaries

The compression preserves the most meaningful emotional dynamics while
discarding noise and routine-state data. Think of it as emotional video
compression — keep the keyframes, drop the filler.

Issue #462.
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

# Minimum significance score for a moment to be considered "key"
_KEY_MOMENT_THRESHOLD = 0.3
# Default number of periods for timeline compression
_DEFAULT_NUM_PERIODS = 10
# Minimum samples per period for meaningful statistics
_MIN_SAMPLES_PER_PERIOD = 2
# Arc detection: minimum number of periods to detect narrative arcs
_MIN_ARC_PERIODS = 4

# Arc phase labels
ARC_PHASES = ("baseline", "rising_action", "climax", "falling_action", "resolution")


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EmotionDataPoint:
    """A single emotional measurement in a timeline."""
    timestamp: float = 0.0     # unix epoch
    valence: float = 0.0       # -1..+1
    arousal: float = 0.5       # 0..1
    stress: float = 0.3        # 0..1
    energy: float = 0.5        # 0..1
    label: str = ""            # optional contextual label


@dataclass
class KeyMoment:
    """An emotionally significant moment extracted from the timeline."""
    timestamp: float = 0.0
    valence: float = 0.0
    arousal: float = 0.5
    significance_score: float = 0.0   # 0..1 how significant
    moment_type: str = "peak"         # "peak", "valley", "shift", "extreme"
    context: str = ""                 # human-readable description
    index: int = 0                    # position in original timeline


@dataclass
class ArcPhase:
    """A single phase of a detected emotional arc."""
    phase: str = "baseline"         # one of ARC_PHASES
    start_index: int = 0
    end_index: int = 0
    mean_valence: float = 0.0
    mean_arousal: float = 0.0
    valence_trend: float = 0.0      # slope direction (-1..+1)
    intensity: float = 0.0          # 0..1


@dataclass
class EmotionalArc:
    """A detected narrative emotional arc across the timeline."""
    phases: List[ArcPhase] = field(default_factory=list)
    arc_type: str = "unknown"       # "triumph", "tragedy", "recovery", "decline", "stable"
    overall_valence_change: float = 0.0
    peak_intensity: float = 0.0
    arc_duration_samples: int = 0


@dataclass
class TimelapsePeriod:
    """A single period in the compressed time-lapse."""
    period_index: int = 0
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    mean_valence: float = 0.0
    mean_arousal: float = 0.0
    mean_stress: float = 0.0
    mean_energy: float = 0.0
    valence_std: float = 0.0
    sample_count: int = 0
    dominant_emotion: str = "neutral"
    key_moments_count: int = 0


@dataclass
class CompressionResult:
    """Full temporal compression output."""
    key_moments: List[KeyMoment] = field(default_factory=list)
    emotional_arc: Optional[EmotionalArc] = None
    timelapse: List[TimelapsePeriod] = field(default_factory=list)
    total_samples: int = 0
    compression_ratio: float = 0.0
    timeline_duration_hours: float = 0.0
    summary: str = ""
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _safe_mean(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _emotional_intensity(dp: EmotionDataPoint) -> float:
    """Compute emotional intensity as distance from neutral state."""
    return math.sqrt(dp.valence ** 2 + (dp.arousal - 0.5) ** 2)


def _classify_period_emotion(valence: float, arousal: float) -> str:
    """Classify the dominant emotion for a period."""
    if valence > 0.3 and arousal > 0.5:
        return "excited"
    elif valence > 0.3:
        return "content"
    elif valence < -0.3 and arousal > 0.5:
        return "stressed"
    elif valence < -0.3:
        return "melancholy"
    elif arousal > 0.6:
        return "activated"
    elif arousal < 0.3:
        return "calm"
    return "neutral"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def extract_key_moments(
    data: List[EmotionDataPoint],
    max_moments: int = 20,
    threshold: float = _KEY_MOMENT_THRESHOLD,
) -> List[KeyMoment]:
    """Extract the most emotionally significant moments from a timeline.

    Key moments are detected via:
    1. Absolute intensity peaks (strong positive or negative emotion)
    2. Sudden shifts (large change from previous measurement)
    3. Extremes (all-time highs and lows)

    Args:
        data: Chronologically sorted emotional data points.
        max_moments: Maximum number of key moments to return.
        threshold: Minimum significance score to qualify.

    Returns:
        List of KeyMoment objects sorted by significance (descending).
    """
    if len(data) < 2:
        if len(data) == 1:
            return [KeyMoment(
                timestamp=data[0].timestamp,
                valence=data[0].valence,
                arousal=data[0].arousal,
                significance_score=0.5,
                moment_type="single",
                context="Only data point",
                index=0,
            )]
        return []

    moments: List[KeyMoment] = []

    # Compute baseline statistics
    valences = [d.valence for d in data]
    arousals = [d.arousal for d in data]
    mean_v = _safe_mean(valences)
    std_v = _safe_std(valences)
    mean_a = _safe_mean(arousals)
    std_a = _safe_std(arousals)

    for i, dp in enumerate(data):
        significance = 0.0
        moment_type = "peak"
        context_parts: List[str] = []

        # 1. Absolute intensity
        intensity = _emotional_intensity(dp)
        intensity_score = _clamp(intensity / 1.2, 0.0, 1.0)

        # 2. Deviation from mean (z-score-like)
        if std_v > 0.01:
            v_deviation = abs(dp.valence - mean_v) / std_v
        else:
            v_deviation = 0.0
        deviation_score = _clamp(v_deviation / 3.0, 0.0, 1.0)

        # 3. Shift from previous
        shift_score = 0.0
        if i > 0:
            prev = data[i - 1]
            v_shift = abs(dp.valence - prev.valence)
            a_shift = abs(dp.arousal - prev.arousal)
            shift_score = _clamp((v_shift + a_shift) / 1.5, 0.0, 1.0)
            if shift_score > 0.4:
                moment_type = "shift"
                context_parts.append(f"valence shift of {dp.valence - prev.valence:+.2f}")

        # Combined significance
        significance = (
            0.40 * intensity_score
            + 0.30 * deviation_score
            + 0.30 * shift_score
        )

        # Classify moment type — only mark as extreme if there is actual variation
        valence_range = max(valences) - min(valences)
        if valence_range > 0.05 and (dp.valence == max(valences) or dp.valence == min(valences)):
            moment_type = "extreme"
            significance = max(significance, 0.5)  # always notable
            context_parts.append(
                "highest valence" if dp.valence == max(valences) else "lowest valence"
            )

        if dp.valence > 0.5:
            context_parts.append("strongly positive")
        elif dp.valence < -0.5:
            context_parts.append("strongly negative")

        if significance >= threshold:
            moments.append(KeyMoment(
                timestamp=dp.timestamp,
                valence=round(dp.valence, 4),
                arousal=round(dp.arousal, 4),
                significance_score=round(significance, 4),
                moment_type=moment_type,
                context="; ".join(context_parts) if context_parts else "notable moment",
                index=i,
            ))

    # Sort by significance and limit
    moments.sort(key=lambda m: m.significance_score, reverse=True)
    return moments[:max_moments]


def detect_emotional_arc(
    data: List[EmotionDataPoint],
    num_phases: int = 5,
) -> Optional[EmotionalArc]:
    """Detect the narrative emotional arc in a timeline.

    Splits data into phases and classifies the overall trajectory as
    triumph, tragedy, recovery, decline, or stable.

    Args:
        data: Chronologically sorted emotional data points.
        num_phases: Number of phases to divide the timeline into.

    Returns:
        EmotionalArc if enough data, None otherwise.
    """
    if len(data) < _MIN_ARC_PERIODS:
        return None

    num_phases = min(num_phases, len(data))
    phase_size = len(data) // num_phases
    if phase_size < 1:
        phase_size = 1

    phases: List[ArcPhase] = []
    phase_labels = list(ARC_PHASES)

    for i in range(num_phases):
        start_idx = i * phase_size
        end_idx = min(start_idx + phase_size, len(data))
        if start_idx >= len(data):
            break

        segment = data[start_idx:end_idx]
        seg_valences = [d.valence for d in segment]
        seg_arousals = [d.arousal for d in segment]
        mean_v = _safe_mean(seg_valences)
        mean_a = _safe_mean(seg_arousals)

        # Compute trend: slope direction within the phase
        if len(seg_valences) > 1:
            trend = (seg_valences[-1] - seg_valences[0]) / max(len(seg_valences), 1)
            trend = _clamp(trend * 10, -1.0, 1.0)  # scale up for readability
        else:
            trend = 0.0

        phase_label = phase_labels[i] if i < len(phase_labels) else "extended"
        intensity = _clamp(abs(mean_v) + abs(mean_a - 0.5), 0.0, 1.0)

        phases.append(ArcPhase(
            phase=phase_label,
            start_index=start_idx,
            end_index=end_idx - 1,
            mean_valence=round(mean_v, 4),
            mean_arousal=round(mean_a, 4),
            valence_trend=round(trend, 4),
            intensity=round(intensity, 4),
        ))

    # Classify arc type based on start vs end valence
    if not phases:
        return None

    start_valence = phases[0].mean_valence
    end_valence = phases[-1].mean_valence
    overall_change = end_valence - start_valence

    # Find peak/valley
    peak_v = max(p.mean_valence for p in phases)
    valley_v = min(p.mean_valence for p in phases)
    peak_intensity = max(p.intensity for p in phases)

    if overall_change > 0.2 and valley_v < start_valence - 0.1:
        arc_type = "recovery"
    elif overall_change > 0.2:
        arc_type = "triumph"
    elif overall_change < -0.2 and peak_v > start_valence + 0.1:
        arc_type = "tragedy"
    elif overall_change < -0.2:
        arc_type = "decline"
    else:
        arc_type = "stable"

    return EmotionalArc(
        phases=phases,
        arc_type=arc_type,
        overall_valence_change=round(overall_change, 4),
        peak_intensity=round(peak_intensity, 4),
        arc_duration_samples=len(data),
    )


def compress_timeline(
    data: List[EmotionDataPoint],
    num_periods: int = _DEFAULT_NUM_PERIODS,
) -> List[TimelapsePeriod]:
    """Compress a timeline into equal-sized periods with summary statistics.

    Args:
        data: Chronologically sorted data points.
        num_periods: Number of output periods.

    Returns:
        List of TimelapsePeriod objects.
    """
    if not data:
        return []

    num_periods = max(1, min(num_periods, len(data)))
    period_size = max(1, len(data) // num_periods)

    periods: List[TimelapsePeriod] = []

    for i in range(num_periods):
        start_idx = i * period_size
        end_idx = min(start_idx + period_size, len(data))
        if i == num_periods - 1:
            end_idx = len(data)  # last period gets remainder
        if start_idx >= len(data):
            break

        segment = data[start_idx:end_idx]
        valences = [d.valence for d in segment]
        arousals = [d.arousal for d in segment]
        stresses = [d.stress for d in segment]
        energies = [d.energy for d in segment]

        mean_v = _safe_mean(valences)
        mean_a = _safe_mean(arousals)

        periods.append(TimelapsePeriod(
            period_index=i,
            start_timestamp=segment[0].timestamp,
            end_timestamp=segment[-1].timestamp,
            mean_valence=round(mean_v, 4),
            mean_arousal=round(mean_a, 4),
            mean_stress=round(_safe_mean(stresses), 4),
            mean_energy=round(_safe_mean(energies), 4),
            valence_std=round(_safe_std(valences), 4),
            sample_count=len(segment),
            dominant_emotion=_classify_period_emotion(mean_v, mean_a),
            key_moments_count=0,  # filled in by generate_timelapse
        ))

    return periods


def generate_timelapse(
    data: List[EmotionDataPoint],
    num_periods: int = _DEFAULT_NUM_PERIODS,
    max_key_moments: int = 20,
) -> CompressionResult:
    """Generate a full time-lapse compression of emotional data.

    Combines key moment extraction, arc detection, and timeline compression
    into a single result.

    Args:
        data: Chronologically sorted emotional data points.
        num_periods: Number of time-lapse periods.
        max_key_moments: Maximum key moments to extract.

    Returns:
        CompressionResult with all compression outputs.
    """
    if not data:
        return CompressionResult(
            summary="No data provided",
            timestamp=time.time(),
        )

    # Sort by timestamp
    sorted_data = sorted(data, key=lambda d: d.timestamp)

    key_moments = extract_key_moments(sorted_data, max_moments=max_key_moments)
    arc = detect_emotional_arc(sorted_data)
    periods = compress_timeline(sorted_data, num_periods=num_periods)

    # Map key moments to periods
    if periods and key_moments:
        for km in key_moments:
            for period in periods:
                if period.start_timestamp <= km.timestamp <= period.end_timestamp:
                    period.key_moments_count += 1
                    break

    # Compute metadata
    total = len(sorted_data)
    output_count = len(periods) + len(key_moments)
    compression_ratio = output_count / total if total > 0 else 0.0

    duration_hours = 0.0
    if len(sorted_data) >= 2:
        duration_sec = sorted_data[-1].timestamp - sorted_data[0].timestamp
        duration_hours = duration_sec / 3600.0

    # Build summary
    arc_desc = f"Arc: {arc.arc_type}" if arc else "No arc detected"
    summary = (
        f"Compressed {total} samples into {len(periods)} periods "
        f"with {len(key_moments)} key moments. "
        f"{arc_desc}. Duration: {duration_hours:.1f} hours."
    )

    return CompressionResult(
        key_moments=key_moments,
        emotional_arc=arc,
        timelapse=periods,
        total_samples=total,
        compression_ratio=round(compression_ratio, 4),
        timeline_duration_hours=round(duration_hours, 2),
        summary=summary,
        timestamp=time.time(),
    )


def compression_to_dict(result: CompressionResult) -> Dict[str, Any]:
    """Serialize a CompressionResult to a plain dict."""
    return {
        "key_moments": [asdict(km) for km in result.key_moments],
        "emotional_arc": asdict(result.emotional_arc) if result.emotional_arc else None,
        "timelapse": [asdict(p) for p in result.timelapse],
        "total_samples": result.total_samples,
        "compression_ratio": result.compression_ratio,
        "timeline_duration_hours": result.timeline_duration_hours,
        "summary": result.summary,
        "timestamp": result.timestamp,
    }
