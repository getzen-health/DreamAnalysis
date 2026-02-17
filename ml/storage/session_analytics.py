"""Session Comparison & Trend Analytics.

Compares brain sessions, tracks progress over time, and generates
improvement metrics for the user dashboard.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from storage.session_recorder import SESSIONS_DIR


# ─── Session Comparison ───────────────────────────────────────────────────


def compare_sessions(session_id_a: str, session_id_b: str) -> Dict:
    """Compare two sessions side-by-side.

    Returns per-metric deltas, improvement indicators, and narrative summary.
    """
    a = _load_session_meta(session_id_a)
    b = _load_session_meta(session_id_b)

    if "error" in a:
        return a
    if "error" in b:
        return b

    timeline_a = a.get("analysis_timeline", [])
    timeline_b = b.get("analysis_timeline", [])

    metrics_a = _aggregate_timeline(timeline_a)
    metrics_b = _aggregate_timeline(timeline_b)

    comparison = {}
    all_keys = set(metrics_a.keys()) | set(metrics_b.keys())

    for key in sorted(all_keys):
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)
        if val_a is not None and val_b is not None:
            delta = val_b - val_a
            pct = (delta / abs(val_a) * 100) if val_a != 0 else 0
            comparison[key] = {
                "session_a": round(val_a, 4),
                "session_b": round(val_b, 4),
                "delta": round(delta, 4),
                "pct_change": round(pct, 1),
                "improved": _is_improvement(key, delta),
            }

    summary_a = a.get("summary", {})
    summary_b = b.get("summary", {})

    return {
        "session_a": {
            "id": session_id_a,
            "start_time": a.get("start_time"),
            "duration_sec": summary_a.get("duration_sec", 0),
            "n_frames": summary_a.get("n_frames", 0),
        },
        "session_b": {
            "id": session_id_b,
            "start_time": b.get("start_time"),
            "duration_sec": summary_b.get("duration_sec", 0),
            "n_frames": summary_b.get("n_frames", 0),
        },
        "metrics": comparison,
        "improvements": sum(1 for v in comparison.values() if v["improved"]),
        "regressions": sum(1 for v in comparison.values() if v["improved"] is False),
        "unchanged": sum(1 for v in comparison.values() if v["improved"] is None),
        "narrative": _build_narrative(comparison),
    }


# ─── Progress Tracking ────────────────────────────────────────────────────


def get_session_trends(
    user_id: Optional[str] = None, last_n: int = 20
) -> Dict:
    """Compute trends across recent sessions.

    Returns per-metric moving averages, best/worst sessions,
    and improvement trajectory.
    """
    sessions = _list_session_metas(user_id)
    sessions.sort(key=lambda s: s.get("start_time", 0))
    sessions = sessions[-last_n:]

    if len(sessions) < 2:
        return {
            "session_count": len(sessions),
            "message": "Need at least 2 sessions for trend analysis",
            "trends": {},
        }

    series = {}
    timestamps = []

    for sess in sessions:
        timeline = sess.get("analysis_timeline", [])
        metrics = _aggregate_timeline(timeline)
        ts = sess.get("start_time", 0)
        timestamps.append(ts)

        for key, val in metrics.items():
            series.setdefault(key, []).append(val)

    trends = {}
    for key, values in series.items():
        if len(values) < 2:
            continue
        arr = np.array(values, dtype=float)
        valid = np.isfinite(arr)
        if valid.sum() < 2:
            continue
        arr_valid = arr[valid]

        # Linear trend (slope)
        x = np.arange(len(arr_valid), dtype=float)
        slope = float(np.polyfit(x, arr_valid, 1)[0]) if len(arr_valid) >= 2 else 0

        # Moving average (last 3 vs first 3)
        n = min(3, len(arr_valid) // 2)
        if n > 0:
            early_avg = float(np.mean(arr_valid[:n]))
            recent_avg = float(np.mean(arr_valid[-n:]))
        else:
            early_avg = recent_avg = float(np.mean(arr_valid))

        trends[key] = {
            "current": round(float(arr_valid[-1]), 4),
            "mean": round(float(np.mean(arr_valid)), 4),
            "std": round(float(np.std(arr_valid)), 4),
            "min": round(float(np.min(arr_valid)), 4),
            "max": round(float(np.max(arr_valid)), 4),
            "slope": round(slope, 6),
            "direction": "improving" if _is_improvement(key, slope) else (
                "declining" if _is_improvement(key, -slope) else "stable"
            ),
            "early_avg": round(early_avg, 4),
            "recent_avg": round(recent_avg, 4),
            "data_points": len(arr_valid),
        }

    # Best/worst sessions
    best_session = None
    worst_session = None
    if "flow_score" in series:
        flow_arr = series["flow_score"]
        best_idx = int(np.argmax(flow_arr))
        worst_idx = int(np.argmin(flow_arr))
        best_session = sessions[best_idx].get("session_id")
        worst_session = sessions[worst_idx].get("session_id")

    return {
        "session_count": len(sessions),
        "time_span_days": round((timestamps[-1] - timestamps[0]) / 86400, 1) if timestamps else 0,
        "trends": trends,
        "best_session": best_session,
        "worst_session": worst_session,
    }


def get_weekly_report(user_id: Optional[str] = None) -> Dict:
    """Generate a weekly progress report.

    Compares this week's averages to last week's.
    """
    import time
    now = time.time()
    this_week_start = now - 7 * 86400
    last_week_start = now - 14 * 86400

    sessions = _list_session_metas(user_id)

    this_week = [
        s for s in sessions
        if s.get("start_time", 0) >= this_week_start
    ]
    last_week = [
        s for s in sessions
        if last_week_start <= s.get("start_time", 0) < this_week_start
    ]

    tw_metrics = _average_session_metrics(this_week)
    lw_metrics = _average_session_metrics(last_week)

    changes = {}
    for key in set(tw_metrics.keys()) | set(lw_metrics.keys()):
        tw = tw_metrics.get(key)
        lw = lw_metrics.get(key)
        if tw is not None and lw is not None and lw != 0:
            pct = (tw - lw) / abs(lw) * 100
            changes[key] = {
                "this_week": round(tw, 4),
                "last_week": round(lw, 4),
                "pct_change": round(pct, 1),
                "improved": _is_improvement(key, tw - lw),
            }
        elif tw is not None:
            changes[key] = {
                "this_week": round(tw, 4),
                "last_week": None,
                "pct_change": None,
                "improved": None,
            }

    return {
        "this_week_sessions": len(this_week),
        "last_week_sessions": len(last_week),
        "changes": changes,
        "highlights": _weekly_highlights(changes),
    }


# ─── Helpers ──────────────────────────────────────────────────────────────


def _load_session_meta(session_id: str) -> Dict:
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        return {"error": f"Session {session_id} not found"}
    with open(meta_path) as f:
        return json.load(f)


def _list_session_metas(user_id: Optional[str] = None) -> List[Dict]:
    metas = []
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            with open(f) as fp:
                meta = json.load(fp)
            if user_id and meta.get("user_id") != user_id:
                continue
            metas.append(meta)
        except Exception:
            continue
    return metas


def _aggregate_timeline(timeline: List[Dict]) -> Dict[str, float]:
    """Average all numeric fields across a session's analysis timeline."""
    if not timeline:
        return {}

    accum = {}
    counts = {}

    for frame in timeline:
        _collect_numeric(frame, "", accum, counts)

    result = {}
    for key, total in accum.items():
        if counts[key] > 0:
            result[key] = total / counts[key]

    return result


def _collect_numeric(obj, prefix, accum, counts):
    """Recursively collect numeric values from nested dicts."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("time_offset", "timestamp"):
                continue
            new_prefix = f"{prefix}.{k}" if prefix else k
            _collect_numeric(v, new_prefix, accum, counts)
    elif isinstance(obj, (int, float)) and np.isfinite(obj):
        accum[prefix] = accum.get(prefix, 0) + obj
        counts[prefix] = counts.get(prefix, 0) + 1


def _average_session_metrics(sessions: List[Dict]) -> Dict[str, float]:
    """Average metrics across multiple sessions."""
    if not sessions:
        return {}

    all_metrics = []
    for sess in sessions:
        timeline = sess.get("analysis_timeline", [])
        m = _aggregate_timeline(timeline)
        if m:
            all_metrics.append(m)

    if not all_metrics:
        return {}

    combined = {}
    counts = {}
    for m in all_metrics:
        for k, v in m.items():
            combined[k] = combined.get(k, 0) + v
            counts[k] = counts.get(k, 0) + 1

    return {k: combined[k] / counts[k] for k in combined}


# Metrics where higher = better
_HIGHER_IS_BETTER = {
    "flow_score", "creativity_score", "encoding_score",
    "will_remember_probability", "quality_score",
    "band_powers.alpha", "components.absorption",
    "components.effortlessness", "components.focus_quality",
    "valence",
}

# Metrics where lower = better
_LOWER_IS_BETTER = {
    "stress_index",
}


def _is_improvement(key: str, delta: float) -> Optional[bool]:
    """Determine if a metric change is an improvement."""
    # Check if any suffix matches
    for pattern in _HIGHER_IS_BETTER:
        if key.endswith(pattern) or key == pattern:
            return delta > 0.001 if delta > 0 else (False if delta < -0.001 else None)
    for pattern in _LOWER_IS_BETTER:
        if key.endswith(pattern) or key == pattern:
            return delta < -0.001 if delta < 0 else (False if delta > 0.001 else None)
    return None  # Unknown metric — can't determine


def _build_narrative(comparison: Dict) -> List[str]:
    """Build human-readable narrative from comparison."""
    improvements = []
    regressions = []

    for key, data in comparison.items():
        if data["improved"] is True:
            improvements.append((key, data["pct_change"]))
        elif data["improved"] is False:
            regressions.append((key, data["pct_change"]))

    narrative = []

    if improvements:
        top = sorted(improvements, key=lambda x: abs(x[1]), reverse=True)[:3]
        for key, pct in top:
            name = key.split(".")[-1].replace("_", " ")
            narrative.append(f"{name} improved by {abs(pct):.0f}%")

    if regressions:
        worst = sorted(regressions, key=lambda x: abs(x[1]), reverse=True)[:2]
        for key, pct in worst:
            name = key.split(".")[-1].replace("_", " ")
            narrative.append(f"{name} declined by {abs(pct):.0f}%")

    if not narrative:
        narrative.append("Metrics are largely unchanged between sessions")

    return narrative


def _weekly_highlights(changes: Dict) -> List[str]:
    """Generate weekly highlights from changes."""
    highlights = []

    for key, data in changes.items():
        if data["improved"] is True and data["pct_change"] and abs(data["pct_change"]) > 10:
            name = key.split(".")[-1].replace("_", " ")
            highlights.append(f"{name} up {data['pct_change']:.0f}% this week")

    for key, data in changes.items():
        if data["improved"] is False and data["pct_change"] and abs(data["pct_change"]) > 10:
            name = key.split(".")[-1].replace("_", " ")
            highlights.append(f"{name} down {abs(data['pct_change']):.0f}% — worth investigating")

    if not highlights:
        highlights.append("Brain metrics are stable this week")

    return highlights
