"""Pilot study validation and analysis.

Provides data-quality validation and summary statistics for the 10-person
feasibility pilot study.  All functions operate on plain dicts/lists (the
Python equivalents of the JSON columns in the pilotSessions / pilotParticipants
tables) so they can be called from FastAPI routes without a DB dependency.

Key concepts:
  - Signal Quality Index (SQI): fraction of 1-second EEG epochs whose peak
    amplitude stays below a configurable threshold (default 75 uV).
  - Session completeness: a session is "complete" when it contains baseline
    EEG, task EEG, biofeedback markers, and a survey response.
  - Participant summary: completion rate, average SQI, mood stability across
    sessions.
  - Pilot statistics: aggregate metrics across all participants to decide
    whether the pilot dataset is sufficient for formal analysis.
  - Readiness report: go / no-go checklist for launching the pilot.

Functions:
  compute_session_sqi()          -- SQI for a single session's EEG
  validate_session_completeness() -- check required phases present
  compute_participant_summary()  -- per-participant aggregate stats
  compute_pilot_statistics()     -- pilot-wide aggregates
  generate_readiness_report()    -- go/no-go readiness assessment
  report_to_dict()               -- serialize a readiness report to JSON-safe dict
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────

# Default amplitude threshold (micro-volts).  Epochs exceeding this are
# considered artifact-contaminated.
DEFAULT_AMPLITUDE_THRESHOLD_UV = 75.0

# Default epoch length in seconds for SQI computation.
DEFAULT_EPOCH_LENGTH_SEC = 1.0

# Default sampling rate.
DEFAULT_FS = 256.0

# Minimum number of participants for a viable pilot.
MIN_PARTICIPANTS = 5

# Minimum per-participant completion rate.
MIN_COMPLETION_RATE = 0.6

# Minimum pilot-wide mean SQI.
MIN_MEAN_SQI = 0.5

# Required session phases for completeness.
REQUIRED_PHASES = frozenset({"baseline", "task", "biofeedback", "survey"})


# ── SQI ──────────────────────────────────────────────────────────────────────


def compute_session_sqi(
    eeg_data: Any,
    fs: float = DEFAULT_FS,
    epoch_length_sec: float = DEFAULT_EPOCH_LENGTH_SEC,
    amplitude_threshold_uv: float = DEFAULT_AMPLITUDE_THRESHOLD_UV,
) -> Dict[str, Any]:
    """Compute Signal Quality Index for one EEG recording.

    Parameters
    ----------
    eeg_data : array-like
        1-D or 2-D (channels x samples) EEG signal in micro-volts.
    fs : float
        Sampling frequency in Hz.
    epoch_length_sec : float
        Length of each epoch in seconds.
    amplitude_threshold_uv : float
        Peak-amplitude threshold for a "good" epoch.

    Returns
    -------
    dict with keys:
        sqi           -- float in [0, 1]
        total_epochs  -- int
        good_epochs   -- int
        bad_epochs    -- int
        mean_amplitude -- float (mean absolute amplitude across all samples)
    """
    arr = np.asarray(eeg_data, dtype=np.float64)
    if arr.size == 0:
        return {
            "sqi": 0.0,
            "total_epochs": 0,
            "good_epochs": 0,
            "bad_epochs": 0,
            "mean_amplitude": 0.0,
        }

    # Flatten to 1-D if multi-channel: take the max-abs across channels per sample.
    if arr.ndim == 2:
        arr = np.max(np.abs(arr), axis=0)
    else:
        arr = np.abs(arr)

    epoch_samples = max(1, int(fs * epoch_length_sec))
    n_epochs = len(arr) // epoch_samples
    if n_epochs == 0:
        # Fewer samples than one epoch -- treat entire signal as one epoch.
        peak = float(np.max(arr))
        good = 1 if peak <= amplitude_threshold_uv else 0
        return {
            "sqi": float(good),
            "total_epochs": 1,
            "good_epochs": good,
            "bad_epochs": 1 - good,
            "mean_amplitude": float(np.mean(arr)),
        }

    good = 0
    for i in range(n_epochs):
        epoch = arr[i * epoch_samples : (i + 1) * epoch_samples]
        if np.max(epoch) <= amplitude_threshold_uv:
            good += 1

    bad = n_epochs - good
    sqi = good / n_epochs

    return {
        "sqi": float(sqi),
        "total_epochs": int(n_epochs),
        "good_epochs": int(good),
        "bad_epochs": int(bad),
        "mean_amplitude": float(np.mean(arr)),
    }


# ── Session completeness ─────────────────────────────────────────────────────


def validate_session_completeness(
    session: Dict[str, Any],
    required_phases: Optional[frozenset] = None,
) -> Dict[str, Any]:
    """Check whether a pilot session has all required phases.

    A session dict is expected to contain keys mirroring the pilotSessions
    schema columns.  Phase presence is determined by non-null / non-empty
    values for the corresponding JSON columns:

        baseline    -> preEegJson  (or "pre_eeg_json")
        task        -> postEegJson (or "post_eeg_json")
        biofeedback -> eegFeaturesJson (or "eeg_features_json")
        survey      -> surveyJson  (or "survey_json")

    Parameters
    ----------
    session : dict
        Session record (camelCase or snake_case keys accepted).
    required_phases : frozenset, optional
        Override default required phases.

    Returns
    -------
    dict with keys:
        is_complete    -- bool
        present_phases -- list[str]
        missing_phases -- list[str]
        phase_details  -- dict mapping phase name to bool
    """
    if required_phases is None:
        required_phases = REQUIRED_PHASES

    # Map phase name -> list of possible key names (camelCase + snake_case).
    phase_key_map: Dict[str, list] = {
        "baseline": ["preEegJson", "pre_eeg_json"],
        "task": ["postEegJson", "post_eeg_json"],
        "biofeedback": ["eegFeaturesJson", "eeg_features_json"],
        "survey": ["surveyJson", "survey_json"],
    }

    present: list[str] = []
    missing: list[str] = []
    details: Dict[str, bool] = {}

    for phase in sorted(required_phases):
        keys = phase_key_map.get(phase, [phase])
        found = False
        for k in keys:
            val = session.get(k)
            if val is not None and val != "" and val != {} and val != []:
                found = True
                break
        details[phase] = found
        if found:
            present.append(phase)
        else:
            missing.append(phase)

    return {
        "is_complete": len(missing) == 0,
        "present_phases": sorted(present),
        "missing_phases": sorted(missing),
        "phase_details": details,
    }


# ── Participant summary ──────────────────────────────────────────────────────


def compute_participant_summary(
    participant_code: str,
    sessions: List[Dict[str, Any]],
    fs: float = DEFAULT_FS,
    amplitude_threshold_uv: float = DEFAULT_AMPLITUDE_THRESHOLD_UV,
) -> Dict[str, Any]:
    """Aggregate statistics for one pilot participant.

    Parameters
    ----------
    participant_code : str
        E.g. "P001".
    sessions : list[dict]
        Session records for this participant.
    fs : float
        Sampling frequency (used for SQI when eeg_data is provided).
    amplitude_threshold_uv : float
        Amplitude threshold for SQI.

    Returns
    -------
    dict with:
        participant_code  -- str
        total_sessions    -- int
        complete_sessions -- int
        completion_rate   -- float [0,1]
        avg_sqi           -- float [0,1] or None
        mood_stability    -- float or None (std-dev of data_quality_score)
        block_counts      -- dict mapping block_type to count
    """
    total = len(sessions)
    complete = 0
    sqi_values: list[float] = []
    quality_scores: list[float] = []
    block_counts: Dict[str, int] = {}

    for s in sessions:
        # Completeness
        comp = validate_session_completeness(s)
        if comp["is_complete"]:
            complete += 1

        # SQI -- check if raw EEG data is embedded (preEegJson or pre_eeg_json)
        eeg_raw = s.get("preEegJson") or s.get("pre_eeg_json")
        if eeg_raw is not None and isinstance(eeg_raw, (list, dict)):
            # If it's a dict with a "signals" key, extract it.
            if isinstance(eeg_raw, dict) and "signals" in eeg_raw:
                eeg_raw = eeg_raw["signals"]
            try:
                sqi_result = compute_session_sqi(
                    eeg_raw, fs=fs, amplitude_threshold_uv=amplitude_threshold_uv
                )
                sqi_values.append(sqi_result["sqi"])
            except Exception:
                pass

        # Data quality score (integer stored in the DB row).
        dqs = s.get("dataQualityScore") or s.get("data_quality_score")
        if dqs is not None:
            try:
                quality_scores.append(float(dqs))
            except (TypeError, ValueError):
                pass

        # Block type tally.
        bt = s.get("blockType") or s.get("block_type") or "unknown"
        block_counts[bt] = block_counts.get(bt, 0) + 1

    avg_sqi: Optional[float] = None
    if sqi_values:
        avg_sqi = float(np.mean(sqi_values))

    mood_stability: Optional[float] = None
    if len(quality_scores) >= 2:
        mood_stability = float(np.std(quality_scores, ddof=1))

    completion_rate = complete / total if total > 0 else 0.0

    return {
        "participant_code": participant_code,
        "total_sessions": total,
        "complete_sessions": complete,
        "completion_rate": float(completion_rate),
        "avg_sqi": avg_sqi,
        "mood_stability": mood_stability,
        "block_counts": block_counts,
    }


# ── Pilot-wide statistics ────────────────────────────────────────────────────


def compute_pilot_statistics(
    all_sessions: List[Dict[str, Any]],
    participant_codes: Optional[List[str]] = None,
    fs: float = DEFAULT_FS,
    amplitude_threshold_uv: float = DEFAULT_AMPLITUDE_THRESHOLD_UV,
) -> Dict[str, Any]:
    """Compute aggregate pilot-level statistics.

    Parameters
    ----------
    all_sessions : list[dict]
        Every session across all participants.
    participant_codes : list[str], optional
        Explicit participant list.  If None, derived from sessions.
    fs, amplitude_threshold_uv : float
        Passed through to SQI computation.

    Returns
    -------
    dict with:
        n_participants       -- int
        n_sessions           -- int
        overall_completion   -- float [0,1]
        usable_epoch_rate    -- float [0,1] (mean SQI across all sessions)
        block_distribution   -- dict[str, int]
        stress_vs_balanced   -- dict with stress count, balanced count, separation score
        per_participant      -- list of participant summaries
    """
    # Group sessions by participant code.
    by_participant: Dict[str, list] = {}
    for s in all_sessions:
        code = s.get("participantCode") or s.get("participant_code") or "unknown"
        by_participant.setdefault(code, []).append(s)

    if participant_codes is not None:
        for code in participant_codes:
            by_participant.setdefault(code, [])

    summaries: list = []
    all_sqi: list[float] = []
    total_complete = 0
    total_sessions = 0
    block_dist: Dict[str, int] = {}

    # Stress vs balanced tracking.
    stress_quality: list[float] = []
    balanced_quality: list[float] = []

    for code in sorted(by_participant):
        sessions = by_participant[code]
        summary = compute_participant_summary(
            code, sessions, fs=fs, amplitude_threshold_uv=amplitude_threshold_uv
        )
        summaries.append(summary)
        total_complete += summary["complete_sessions"]
        total_sessions += summary["total_sessions"]
        if summary["avg_sqi"] is not None:
            all_sqi.append(summary["avg_sqi"])
        for bt, count in summary["block_counts"].items():
            block_dist[bt] = block_dist.get(bt, 0) + count

    # Stress vs balanced separation.
    for s in all_sessions:
        bt = (s.get("blockType") or s.get("block_type") or "").lower()
        dqs = s.get("dataQualityScore") or s.get("data_quality_score")
        if dqs is not None:
            try:
                val = float(dqs)
            except (TypeError, ValueError):
                continue
            if bt == "stress":
                stress_quality.append(val)
            elif bt in ("food", "sleep"):
                balanced_quality.append(val)

    separation_score: Optional[float] = None
    if stress_quality and balanced_quality:
        mean_stress = float(np.mean(stress_quality))
        mean_balanced = float(np.mean(balanced_quality))
        pooled_std = float(
            np.sqrt(
                (np.var(stress_quality, ddof=0) + np.var(balanced_quality, ddof=0)) / 2
            )
        )
        if pooled_std > 0:
            separation_score = abs(mean_stress - mean_balanced) / pooled_std
        else:
            separation_score = 0.0

    overall_completion = total_complete / total_sessions if total_sessions > 0 else 0.0
    usable_epoch_rate = float(np.mean(all_sqi)) if all_sqi else 0.0

    return {
        "n_participants": len(by_participant),
        "n_sessions": total_sessions,
        "overall_completion": float(overall_completion),
        "usable_epoch_rate": float(usable_epoch_rate),
        "block_distribution": block_dist,
        "stress_vs_balanced": {
            "stress_count": len(stress_quality),
            "balanced_count": len(balanced_quality),
            "separation_score": separation_score,
        },
        "per_participant": summaries,
    }


# ── Readiness report ─────────────────────────────────────────────────────────


@dataclass
class ReadinessCheck:
    """Single item in the readiness checklist."""

    name: str
    passed: bool
    detail: str


@dataclass
class ReadinessReport:
    """Go / no-go readiness assessment for the pilot."""

    ready: bool
    checks: List[ReadinessCheck] = field(default_factory=list)
    summary: str = ""
    statistics: Dict[str, Any] = field(default_factory=dict)


def generate_readiness_report(
    all_sessions: List[Dict[str, Any]],
    participant_codes: Optional[List[str]] = None,
    min_participants: int = MIN_PARTICIPANTS,
    min_completion_rate: float = MIN_COMPLETION_RATE,
    min_mean_sqi: float = MIN_MEAN_SQI,
    fs: float = DEFAULT_FS,
    amplitude_threshold_uv: float = DEFAULT_AMPLITUDE_THRESHOLD_UV,
) -> ReadinessReport:
    """Generate a go/no-go readiness report for the pilot.

    Checks:
    1. Enough participants enrolled (>= min_participants).
    2. Overall session completion rate >= min_completion_rate.
    3. Mean SQI across sessions >= min_mean_sqi.
    4. At least one session per required block type (stress, food, sleep).
    5. Stress-vs-balanced separation score exists (> 0).

    Returns a ReadinessReport dataclass.
    """
    stats = compute_pilot_statistics(
        all_sessions,
        participant_codes=participant_codes,
        fs=fs,
        amplitude_threshold_uv=amplitude_threshold_uv,
    )

    checks: list[ReadinessCheck] = []

    # 1. Participant count.
    n_part = stats["n_participants"]
    checks.append(
        ReadinessCheck(
            name="participant_count",
            passed=n_part >= min_participants,
            detail=f"{n_part}/{min_participants} participants enrolled",
        )
    )

    # 2. Completion rate.
    comp = stats["overall_completion"]
    checks.append(
        ReadinessCheck(
            name="completion_rate",
            passed=comp >= min_completion_rate,
            detail=f"Completion rate {comp:.1%} (min {min_completion_rate:.0%})",
        )
    )

    # 3. Mean SQI.
    sqi = stats["usable_epoch_rate"]
    checks.append(
        ReadinessCheck(
            name="signal_quality",
            passed=sqi >= min_mean_sqi,
            detail=f"Mean SQI {sqi:.1%} (min {min_mean_sqi:.0%})",
        )
    )

    # 4. Block type coverage.
    required_blocks = {"stress", "food", "sleep"}
    present_blocks = set(stats["block_distribution"].keys())
    missing_blocks = required_blocks - present_blocks
    checks.append(
        ReadinessCheck(
            name="block_coverage",
            passed=len(missing_blocks) == 0,
            detail=(
                "All block types present"
                if not missing_blocks
                else f"Missing block types: {', '.join(sorted(missing_blocks))}"
            ),
        )
    )

    # 5. Stress-vs-balanced separation.
    sep = stats["stress_vs_balanced"].get("separation_score")
    has_separation = sep is not None and sep > 0
    checks.append(
        ReadinessCheck(
            name="stress_separation",
            passed=has_separation,
            detail=(
                f"Stress-vs-balanced separation d={sep:.2f}"
                if sep is not None
                else "No separation data available"
            ),
        )
    )

    all_passed = all(c.passed for c in checks)
    passed_count = sum(1 for c in checks if c.passed)
    total_count = len(checks)

    summary = (
        f"READY -- all {total_count} checks passed"
        if all_passed
        else f"NOT READY -- {passed_count}/{total_count} checks passed"
    )

    return ReadinessReport(
        ready=all_passed,
        checks=checks,
        summary=summary,
        statistics=stats,
    )


def report_to_dict(report: ReadinessReport) -> Dict[str, Any]:
    """Serialize a ReadinessReport to a JSON-safe dict."""
    return {
        "ready": report.ready,
        "summary": report.summary,
        "checks": [
            {"name": c.name, "passed": c.passed, "detail": c.detail}
            for c in report.checks
        ],
        "statistics": report.statistics,
    }
