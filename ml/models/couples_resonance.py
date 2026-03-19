"""Couples emotional resonance — consent-gated shared emotional dashboards — issue #440.

Provides relationship wellness analytics by analyzing two partners' emotional
time series for synchrony, resonance, conflict, and repair patterns.

Algorithm:
1. Both partners must explicitly opt-in (consent model).  Either can revoke
   at any time, which wipes all shared data immediately.
2. Emotional synchrony: cross-correlation of valence/arousal time series
   across two users, yielding a -1 to +1 synchrony score.
3. Resonance detection: sliding-window correlation identifies periods where
   both partners' emotions move in tandem vs. diverge.
4. Communication quality proxy: Granger-style lag correlation — does partner
   A's stress predict partner B's stress?  High lag-correlation =
   emotional contagion; low = emotional independence.
5. Conflict detection: simultaneous negative valence + high arousal in both.
6. Repair detection: after a conflict, how quickly do both partners return
   to positive emotional states?
7. Full relationship profile aggregates all of the above.

References:
    Levenson & Gottman (1983) — Marital interaction: physiological linkage
    Butler (2011) — Temporal interpersonal emotion systems
    Timmons et al. (2015) — It takes two: physiological coregulation
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EmotionSample:
    """A single timestamped emotion measurement for one partner."""
    timestamp: float   # unix epoch seconds
    valence: float     # -1.0 to 1.0
    arousal: float     # 0.0 to 1.0


@dataclass
class ConsentRecord:
    """Tracks consent state for one partner in a partnership."""
    user_id: str
    consented: bool = False
    consented_at: Optional[float] = None
    revoked_at: Optional[float] = None


@dataclass
class Partnership:
    """A bidirectional partnership between two users."""
    partner_a: ConsentRecord
    partner_b: ConsentRecord
    created_at: float = 0.0
    active: bool = False  # True only when BOTH have consented


@dataclass
class ConflictEvent:
    """A detected period of emotional conflict between partners."""
    start_ts: float
    end_ts: float
    avg_valence_a: float
    avg_valence_b: float
    avg_arousal_a: float
    avg_arousal_b: float
    duration_sec: float


@dataclass
class RepairEvent:
    """A detected emotional repair following a conflict."""
    conflict: ConflictEvent
    repair_ts: float            # timestamp when repair was achieved
    repair_latency_sec: float   # seconds from conflict end to repair
    recovered_valence_a: float
    recovered_valence_b: float


@dataclass
class RelationshipProfile:
    """Full emotional relationship profile for a partnership."""
    synchrony_score: float
    resonance_ratio: float          # fraction of time in resonance
    contagion_a_to_b: float         # how much A's stress predicts B's stress
    contagion_b_to_a: float
    conflict_count: int
    avg_conflict_duration_sec: float
    repair_count: int
    avg_repair_latency_sec: float
    repair_rate: float              # repairs / conflicts (0-1)
    overall_health: str             # "strong", "healthy", "strained", "at_risk"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Synchrony: minimum samples for meaningful computation
_MIN_SAMPLES = 5

# Resonance: sliding window size (number of samples)
_RESONANCE_WINDOW = 10

# Resonance: correlation threshold for "in sync"
_RESONANCE_THRESHOLD = 0.3

# Conflict: valence must be below this for both partners
_CONFLICT_VALENCE_THRESHOLD = -0.15

# Conflict: arousal must be above this for both partners
_CONFLICT_AROUSAL_THRESHOLD = 0.55

# Conflict: minimum consecutive samples to count as a conflict period
_CONFLICT_MIN_SAMPLES = 2

# Repair: valence must recover above this for both partners
_REPAIR_VALENCE_THRESHOLD = 0.0

# Repair: maximum seconds after conflict end to search for repair
_REPAIR_LOOKFORWARD_SEC = 3600.0  # 1 hour

# Contagion lag: maximum lag in samples for cross-correlation
_MAX_LAG_SAMPLES = 5

# Partnership store (in-memory; production would use a database)
_partnerships: Dict[str, Partnership] = {}

# Emotion data store per user (in-memory)
_user_emotions: Dict[str, List[EmotionSample]] = defaultdict(list)

# Maximum stored samples per user
_MAX_STORED_SAMPLES = 2000


# ---------------------------------------------------------------------------
# Consent management
# ---------------------------------------------------------------------------

def manage_partnership_consent(
    user_a_id: str,
    user_b_id: str,
    action: str,
    acting_user: str,
) -> Dict:
    """Manage consent for a partnership between two users.

    Args:
        user_a_id: First partner's user ID.
        user_b_id: Second partner's user ID.
        action: One of "opt_in", "revoke", "status".
        acting_user: The user performing the action (must be user_a_id or user_b_id).

    Returns:
        Dict with partnership status.
    """
    if acting_user not in (user_a_id, user_b_id):
        return {"error": "acting_user must be one of the partners"}

    if action not in ("opt_in", "revoke", "status"):
        return {"error": f"Unknown action: {action}. Must be opt_in, revoke, or status."}

    key = _partnership_key(user_a_id, user_b_id)

    if action == "status":
        if key not in _partnerships:
            return {
                "partnership_exists": False,
                "active": False,
                "partner_a": user_a_id,
                "partner_b": user_b_id,
            }
        p = _partnerships[key]
        return _partnership_to_dict(p)

    now = time.time()

    if action == "opt_in":
        if key not in _partnerships:
            _partnerships[key] = Partnership(
                partner_a=ConsentRecord(user_id=user_a_id),
                partner_b=ConsentRecord(user_id=user_b_id),
                created_at=now,
            )

        p = _partnerships[key]
        record = p.partner_a if acting_user == p.partner_a.user_id else p.partner_b
        record.consented = True
        record.consented_at = now
        record.revoked_at = None

        # Partnership becomes active when both consent
        p.active = p.partner_a.consented and p.partner_b.consented
        return _partnership_to_dict(p)

    if action == "revoke":
        if key not in _partnerships:
            return {"error": "No partnership exists to revoke"}

        p = _partnerships[key]
        record = p.partner_a if acting_user == p.partner_a.user_id else p.partner_b
        record.consented = False
        record.revoked_at = now
        p.active = False

        # Wipe shared emotion data for both partners in this partnership
        _user_emotions.pop(p.partner_a.user_id, None)
        _user_emotions.pop(p.partner_b.user_id, None)

        return _partnership_to_dict(p)

    return {"error": "Unexpected state"}


def _partnership_key(user_a: str, user_b: str) -> str:
    """Canonical key for a partnership (order-independent)."""
    return ":".join(sorted([user_a, user_b]))


def _partnership_to_dict(p: Partnership) -> Dict:
    """Serialize partnership to dict."""
    return {
        "partnership_exists": True,
        "active": p.active,
        "partner_a": {
            "user_id": p.partner_a.user_id,
            "consented": p.partner_a.consented,
            "consented_at": p.partner_a.consented_at,
            "revoked_at": p.partner_a.revoked_at,
        },
        "partner_b": {
            "user_id": p.partner_b.user_id,
            "consented": p.partner_b.consented,
            "consented_at": p.partner_b.consented_at,
            "revoked_at": p.partner_b.revoked_at,
        },
        "created_at": p.created_at,
    }


def _check_partnership_active(user_a_id: str, user_b_id: str) -> Optional[str]:
    """Return an error string if partnership is not active, else None."""
    key = _partnership_key(user_a_id, user_b_id)
    if key not in _partnerships:
        return "No partnership exists. Both users must opt in first."
    p = _partnerships[key]
    if not p.active:
        return "Partnership is not active. Both users must consent."
    return None


def store_emotion_samples(user_id: str, samples: List[EmotionSample]) -> int:
    """Store emotion samples for a user (for use in shared analysis).

    Returns number of samples stored total for this user.
    """
    _user_emotions[user_id].extend(samples)
    # Enforce cap
    if len(_user_emotions[user_id]) > _MAX_STORED_SAMPLES:
        _user_emotions[user_id] = _user_emotions[user_id][-_MAX_STORED_SAMPLES:]
    return len(_user_emotions[user_id])


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def compute_emotional_synchrony(
    samples_a: List[EmotionSample],
    samples_b: List[EmotionSample],
) -> Dict:
    """Compute emotional synchrony between two partners' time series.

    Uses cross-correlation of aligned valence and arousal series.
    Samples are aligned by nearest timestamp.

    Args:
        samples_a: Partner A's emotion samples (chronological).
        samples_b: Partner B's emotion samples (chronological).

    Returns:
        Dict with valence_synchrony, arousal_synchrony, overall_synchrony,
        n_aligned_samples, interpretation.
    """
    if len(samples_a) < _MIN_SAMPLES or len(samples_b) < _MIN_SAMPLES:
        return {
            "valence_synchrony": 0.0,
            "arousal_synchrony": 0.0,
            "overall_synchrony": 0.0,
            "n_aligned_samples": 0,
            "interpretation": "insufficient_data",
        }

    # Align by nearest timestamp
    aligned_a, aligned_b = _align_samples(samples_a, samples_b)
    n = len(aligned_a)

    if n < _MIN_SAMPLES:
        return {
            "valence_synchrony": 0.0,
            "arousal_synchrony": 0.0,
            "overall_synchrony": 0.0,
            "n_aligned_samples": n,
            "interpretation": "insufficient_data",
        }

    val_a = [s.valence for s in aligned_a]
    val_b = [s.valence for s in aligned_b]
    aro_a = [s.arousal for s in aligned_a]
    aro_b = [s.arousal for s in aligned_b]

    valence_sync = _pearson_correlation(val_a, val_b)
    arousal_sync = _pearson_correlation(aro_a, aro_b)
    overall = (valence_sync + arousal_sync) / 2.0

    interpretation = _interpret_synchrony(overall)

    return {
        "valence_synchrony": round(valence_sync, 4),
        "arousal_synchrony": round(arousal_sync, 4),
        "overall_synchrony": round(overall, 4),
        "n_aligned_samples": n,
        "interpretation": interpretation,
    }


def detect_resonance_periods(
    samples_a: List[EmotionSample],
    samples_b: List[EmotionSample],
    window: int = _RESONANCE_WINDOW,
    threshold: float = _RESONANCE_THRESHOLD,
) -> Dict:
    """Detect periods where partners' emotions move in sync vs. diverge.

    Uses a sliding window correlation on aligned valence series.

    Args:
        samples_a: Partner A's samples.
        samples_b: Partner B's samples.
        window: Sliding window size in samples.
        threshold: Minimum correlation to count as "in resonance".

    Returns:
        Dict with resonance_periods (list), divergence_periods (list),
        resonance_ratio, n_windows.
    """
    aligned_a, aligned_b = _align_samples(samples_a, samples_b)
    n = len(aligned_a)

    if n < window:
        return {
            "resonance_periods": [],
            "divergence_periods": [],
            "resonance_ratio": 0.0,
            "n_windows": 0,
        }

    val_a = [s.valence for s in aligned_a]
    val_b = [s.valence for s in aligned_b]

    resonance_periods: List[Dict] = []
    divergence_periods: List[Dict] = []
    resonance_count = 0

    n_windows = n - window + 1
    for i in range(n_windows):
        win_a = val_a[i : i + window]
        win_b = val_b[i : i + window]
        corr = _pearson_correlation(win_a, win_b)

        period_info = {
            "start_ts": aligned_a[i].timestamp,
            "end_ts": aligned_a[i + window - 1].timestamp,
            "correlation": round(corr, 4),
        }

        if corr >= threshold:
            resonance_periods.append(period_info)
            resonance_count += 1
        else:
            divergence_periods.append(period_info)

    resonance_ratio = resonance_count / n_windows if n_windows > 0 else 0.0

    return {
        "resonance_periods": resonance_periods,
        "divergence_periods": divergence_periods,
        "resonance_ratio": round(resonance_ratio, 4),
        "n_windows": n_windows,
    }


def detect_conflict(
    samples_a: List[EmotionSample],
    samples_b: List[EmotionSample],
    valence_threshold: float = _CONFLICT_VALENCE_THRESHOLD,
    arousal_threshold: float = _CONFLICT_AROUSAL_THRESHOLD,
    min_samples: int = _CONFLICT_MIN_SAMPLES,
) -> List[ConflictEvent]:
    """Detect conflict periods: both partners have negative valence + high arousal.

    Args:
        samples_a: Partner A's samples.
        samples_b: Partner B's samples.
        valence_threshold: Both must be below this valence.
        arousal_threshold: Both must be above this arousal.
        min_samples: Minimum consecutive aligned samples to count.

    Returns:
        List of ConflictEvent objects.
    """
    aligned_a, aligned_b = _align_samples(samples_a, samples_b)
    n = len(aligned_a)

    if n < min_samples:
        return []

    conflicts: List[ConflictEvent] = []
    run_start: Optional[int] = None

    for i in range(n):
        a, b = aligned_a[i], aligned_b[i]
        is_conflict = (
            a.valence < valence_threshold
            and b.valence < valence_threshold
            and a.arousal > arousal_threshold
            and b.arousal > arousal_threshold
        )

        if is_conflict:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= min_samples:
                conflicts.append(_build_conflict(aligned_a, aligned_b, run_start, i))
            run_start = None

    # Handle conflict that extends to end of data
    if run_start is not None and (n - run_start) >= min_samples:
        conflicts.append(_build_conflict(aligned_a, aligned_b, run_start, n))

    return conflicts


def detect_repair(
    conflicts: List[ConflictEvent],
    samples_a: List[EmotionSample],
    samples_b: List[EmotionSample],
    repair_valence: float = _REPAIR_VALENCE_THRESHOLD,
    lookforward_sec: float = _REPAIR_LOOKFORWARD_SEC,
) -> List[RepairEvent]:
    """Detect emotional repair after conflict events.

    Repair = both partners' valence recovers above repair_valence within
    lookforward_sec after the conflict ends.

    Args:
        conflicts: Previously detected conflicts.
        samples_a: Partner A's samples.
        samples_b: Partner B's samples.
        repair_valence: Valence threshold for "repaired" state.
        lookforward_sec: Max seconds after conflict end to search.

    Returns:
        List of RepairEvent objects.
    """
    if not conflicts:
        return []

    repairs: List[RepairEvent] = []

    for conflict in conflicts:
        # Find post-conflict samples for both partners
        post_a = [s for s in samples_a
                  if conflict.end_ts < s.timestamp <= conflict.end_ts + lookforward_sec]
        post_b = [s for s in samples_b
                  if conflict.end_ts < s.timestamp <= conflict.end_ts + lookforward_sec]

        if not post_a or not post_b:
            continue

        # Find earliest time when both are above repair threshold
        # Merge and sort all post-conflict timestamps
        all_ts = sorted(set(s.timestamp for s in post_a) | set(s.timestamp for s in post_b))

        a_vals = {s.timestamp: s.valence for s in post_a}
        b_vals = {s.timestamp: s.valence for s in post_b}

        # Track latest known valence for each partner
        latest_a_val = conflict.avg_valence_a
        latest_b_val = conflict.avg_valence_b

        for ts in all_ts:
            if ts in a_vals:
                latest_a_val = a_vals[ts]
            if ts in b_vals:
                latest_b_val = b_vals[ts]

            if latest_a_val >= repair_valence and latest_b_val >= repair_valence:
                latency = ts - conflict.end_ts
                repairs.append(RepairEvent(
                    conflict=conflict,
                    repair_ts=ts,
                    repair_latency_sec=round(latency, 2),
                    recovered_valence_a=round(latest_a_val, 4),
                    recovered_valence_b=round(latest_b_val, 4),
                ))
                break

    return repairs


def compute_relationship_profile(
    samples_a: List[EmotionSample],
    samples_b: List[EmotionSample],
) -> RelationshipProfile:
    """Compute a full relationship emotional profile.

    Aggregates synchrony, resonance, contagion, conflict, and repair metrics.

    Args:
        samples_a: Partner A's emotion samples (chronological).
        samples_b: Partner B's emotion samples (chronological).

    Returns:
        RelationshipProfile dataclass.
    """
    sync = compute_emotional_synchrony(samples_a, samples_b)
    resonance = detect_resonance_periods(samples_a, samples_b)
    conflicts = detect_conflict(samples_a, samples_b)
    repairs = detect_repair(conflicts, samples_a, samples_b)

    # Emotional contagion: lag correlation
    contagion_ab = _compute_lag_correlation(samples_a, samples_b)
    contagion_ba = _compute_lag_correlation(samples_b, samples_a)

    # Aggregate conflict stats
    conflict_count = len(conflicts)
    avg_conflict_dur = (
        sum(c.duration_sec for c in conflicts) / conflict_count
        if conflict_count > 0
        else 0.0
    )

    # Aggregate repair stats
    repair_count = len(repairs)
    avg_repair_latency = (
        sum(r.repair_latency_sec for r in repairs) / repair_count
        if repair_count > 0
        else 0.0
    )
    repair_rate = repair_count / conflict_count if conflict_count > 0 else 1.0

    # Overall health classification
    overall_health = _classify_health(
        sync["overall_synchrony"],
        resonance["resonance_ratio"],
        conflict_count,
        repair_rate,
    )

    return RelationshipProfile(
        synchrony_score=sync["overall_synchrony"],
        resonance_ratio=resonance["resonance_ratio"],
        contagion_a_to_b=round(contagion_ab, 4),
        contagion_b_to_a=round(contagion_ba, 4),
        conflict_count=conflict_count,
        avg_conflict_duration_sec=round(avg_conflict_dur, 2),
        repair_count=repair_count,
        avg_repair_latency_sec=round(avg_repair_latency, 2),
        repair_rate=round(repair_rate, 4),
        overall_health=overall_health,
    )


def profile_to_dict(profile: RelationshipProfile) -> Dict:
    """Serialize a RelationshipProfile to a plain dict for JSON responses."""
    return {
        "synchrony_score": profile.synchrony_score,
        "resonance_ratio": profile.resonance_ratio,
        "contagion_a_to_b": profile.contagion_a_to_b,
        "contagion_b_to_a": profile.contagion_b_to_a,
        "conflict_count": profile.conflict_count,
        "avg_conflict_duration_sec": profile.avg_conflict_duration_sec,
        "repair_count": profile.repair_count,
        "avg_repair_latency_sec": profile.avg_repair_latency_sec,
        "repair_rate": profile.repair_rate,
        "overall_health": profile.overall_health,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _align_samples(
    samples_a: List[EmotionSample],
    samples_b: List[EmotionSample],
    max_gap_sec: float = 120.0,
) -> Tuple[List[EmotionSample], List[EmotionSample]]:
    """Align two time series by nearest timestamp.

    For each sample in the shorter list, find the nearest sample in the
    longer list.  Pairs with time gap > max_gap_sec are discarded.

    Returns:
        Tuple of (aligned_a, aligned_b) lists of equal length.
    """
    if not samples_a or not samples_b:
        return [], []

    # Use the shorter list as the reference
    if len(samples_a) <= len(samples_b):
        ref, other = samples_a, samples_b
        ref_is_a = True
    else:
        ref, other = samples_b, samples_a
        ref_is_a = False

    aligned_ref: List[EmotionSample] = []
    aligned_other: List[EmotionSample] = []
    other_idx = 0

    for r_sample in ref:
        # Advance other_idx to find nearest
        while (
            other_idx < len(other) - 1
            and abs(other[other_idx + 1].timestamp - r_sample.timestamp)
            < abs(other[other_idx].timestamp - r_sample.timestamp)
        ):
            other_idx += 1

        gap = abs(other[other_idx].timestamp - r_sample.timestamp)
        if gap <= max_gap_sec:
            aligned_ref.append(r_sample)
            aligned_other.append(other[other_idx])

    if ref_is_a:
        return aligned_ref, aligned_other
    else:
        return aligned_other, aligned_ref


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation between two equal-length lists.

    Returns 0.0 for degenerate cases (constant series, empty).
    """
    n = len(x)
    if n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-12:
        return 0.0

    r = cov / denom
    return max(-1.0, min(1.0, r))


def _compute_lag_correlation(
    leader: List[EmotionSample],
    follower: List[EmotionSample],
    max_lag: int = _MAX_LAG_SAMPLES,
) -> float:
    """Compute maximum lag correlation (arousal) from leader to follower.

    Tests whether leader's arousal at time t predicts follower's arousal
    at time t+lag for lags 1..max_lag.  Returns the maximum correlation
    found across lags (0 if insufficient data).

    High value = leader's emotions predict follower's emotions (contagion).
    """
    aligned_a, aligned_b = _align_samples(leader, follower)
    n = len(aligned_a)

    if n < max_lag + _MIN_SAMPLES:
        return 0.0

    aro_a = [s.arousal for s in aligned_a]
    aro_b = [s.arousal for s in aligned_b]

    best_corr = 0.0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        # Leader's arousal from 0..n-lag, follower's arousal from lag..n
        leader_series = aro_a[: n - lag]
        follower_series = aro_b[lag:]
        corr = _pearson_correlation(leader_series, follower_series)
        if abs(corr) > abs(best_corr):
            best_corr = corr

    return best_corr


def _build_conflict(
    aligned_a: List[EmotionSample],
    aligned_b: List[EmotionSample],
    start_idx: int,
    end_idx: int,
) -> ConflictEvent:
    """Build a ConflictEvent from aligned samples at given indices."""
    slice_a = aligned_a[start_idx:end_idx]
    slice_b = aligned_b[start_idx:end_idx]

    avg_val_a = sum(s.valence for s in slice_a) / len(slice_a)
    avg_val_b = sum(s.valence for s in slice_b) / len(slice_b)
    avg_aro_a = sum(s.arousal for s in slice_a) / len(slice_a)
    avg_aro_b = sum(s.arousal for s in slice_b) / len(slice_b)

    start_ts = aligned_a[start_idx].timestamp
    end_ts = aligned_a[end_idx - 1].timestamp
    duration = end_ts - start_ts

    return ConflictEvent(
        start_ts=start_ts,
        end_ts=end_ts,
        avg_valence_a=round(avg_val_a, 4),
        avg_valence_b=round(avg_val_b, 4),
        avg_arousal_a=round(avg_aro_a, 4),
        avg_arousal_b=round(avg_aro_b, 4),
        duration_sec=round(duration, 2),
    )


def _interpret_synchrony(score: float) -> str:
    """Interpret an overall synchrony score."""
    if score >= 0.6:
        return "high_synchrony"
    elif score >= 0.3:
        return "moderate_synchrony"
    elif score >= 0.0:
        return "low_synchrony"
    else:
        return "anti_synchrony"


def _classify_health(
    synchrony: float,
    resonance_ratio: float,
    conflict_count: int,
    repair_rate: float,
) -> str:
    """Classify overall relationship health from aggregate metrics."""
    score = 0.0
    # Synchrony contribution (0-30 points)
    score += max(0.0, synchrony) * 30.0
    # Resonance ratio contribution (0-25 points)
    score += resonance_ratio * 25.0
    # Low conflict bonus (0-20 points): fewer conflicts = higher score
    if conflict_count == 0:
        score += 20.0
    elif conflict_count <= 2:
        score += 10.0
    # Repair rate contribution (0-25 points)
    score += repair_rate * 25.0

    if score >= 70:
        return "strong"
    elif score >= 50:
        return "healthy"
    elif score >= 30:
        return "strained"
    else:
        return "at_risk"


def _conflict_to_dict(c: ConflictEvent) -> Dict:
    """Serialize a ConflictEvent to dict."""
    return {
        "start_ts": c.start_ts,
        "end_ts": c.end_ts,
        "avg_valence_a": c.avg_valence_a,
        "avg_valence_b": c.avg_valence_b,
        "avg_arousal_a": c.avg_arousal_a,
        "avg_arousal_b": c.avg_arousal_b,
        "duration_sec": c.duration_sec,
    }


def _repair_to_dict(r: RepairEvent) -> Dict:
    """Serialize a RepairEvent to dict."""
    return {
        "conflict": _conflict_to_dict(r.conflict),
        "repair_ts": r.repair_ts,
        "repair_latency_sec": r.repair_latency_sec,
        "recovered_valence_a": r.recovered_valence_a,
        "recovered_valence_b": r.recovered_valence_b,
    }
