"""Brain-Health Correlation Engine.

Analyzes relationships between brain states (EEG) and physical health
metrics (Apple Health / Google Fit) to generate personalized insights.

Correlation types:
1. Temporal: What happened before/after brain states changed?
2. Trend: How do brain patterns evolve with health habits over weeks?
3. Predictive: Can we predict good flow/creativity days from health data?
4. Sleep-Dream: Correlate dream quality with physical health indicators.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime


DB_PATH = Path(__file__).parent.parent / "data" / "health_brain.db"


class HealthBrainDB:
    """SQLite storage for health + brain data correlation."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS health_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    value REAL,
                    timestamp REAL NOT NULL,
                    end_timestamp REAL,
                    source TEXT DEFAULT 'unknown',
                    metadata TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                );

                CREATE TABLE IF NOT EXISTS brain_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT UNIQUE,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    duration_seconds REAL,

                    -- Sleep
                    sleep_stage TEXT,
                    sleep_confidence REAL,

                    -- Emotion
                    emotion TEXT,
                    emotion_confidence REAL,
                    valence REAL,
                    arousal REAL,

                    -- Flow
                    flow_state TEXT,
                    flow_score REAL,
                    absorption REAL,
                    effortlessness REAL,

                    -- Creativity
                    creativity_state TEXT,
                    creativity_score REAL,
                    divergent_thinking REAL,
                    insight_potential REAL,

                    -- Memory
                    encoding_state TEXT,
                    encoding_score REAL,
                    will_remember_prob REAL,

                    -- Dream
                    dream_detected INTEGER,
                    dream_confidence REAL,

                    -- Raw band powers
                    alpha_power REAL,
                    beta_power REAL,
                    theta_power REAL,
                    gamma_power REAL,
                    delta_power REAL,

                    metadata TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                );

                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    correlation_strength REAL,
                    evidence_count INTEGER,
                    brain_metric TEXT,
                    health_metric TEXT,
                    generated_at REAL DEFAULT (strftime('%s', 'now'))
                );

                CREATE INDEX IF NOT EXISTS idx_health_user_time
                    ON health_samples(user_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_health_metric
                    ON health_samples(user_id, metric, timestamp);
                CREATE INDEX IF NOT EXISTS idx_brain_user_time
                    ON brain_sessions(user_id, start_time);
                CREATE INDEX IF NOT EXISTS idx_insights_user
                    ON insights(user_id, generated_at);
            """)

    def store_health_samples(self, user_id: str, metric: str, samples: List[Dict]):
        """Store health data samples."""
        with sqlite3.connect(self.db_path) as conn:
            for sample in samples:
                conn.execute(
                    """INSERT INTO health_samples
                       (user_id, metric, value, timestamp, end_timestamp, source, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (user_id, metric,
                     sample.get("value"),
                     sample.get("timestamp", time.time()),
                     sample.get("end_timestamp"),
                     sample.get("source", "unknown"),
                     json.dumps(sample.get("metadata", {})))
                )

    def store_brain_session(self, user_id: str, session: Dict):
        """Store a brain state session with all analysis results.

        Accepts both nested format (from /simulate-eeg) and flat key-value format.
        """
        # Support both nested dicts and flat key-value formats
        flow = session.get("flow_state", {})
        if isinstance(flow, str):
            flow = {"state": flow, "flow_score": session.get("flow_score")}
        creativity = session.get("creativity", {})
        if isinstance(creativity, str):
            creativity = {"state": creativity, "creativity_score": session.get("creativity_score")}
        memory = session.get("memory_encoding", {})
        if isinstance(memory, str):
            memory = {"state": memory, "will_remember_probability": session.get("memory_probability")}
        emotion = session.get("emotions", {})
        if isinstance(emotion, str):
            emotion = {"emotion": emotion, "valence": session.get("valence"), "arousal": session.get("arousal")}
        sleep = session.get("sleep_stage", {})
        if isinstance(sleep, str):
            sleep = {"stage": sleep, "confidence": session.get("sleep_confidence")}
        dream = session.get("dream_detection", {})
        if isinstance(dream, str):
            dream = {"is_dreaming": dream == "dreaming", "confidence": session.get("dream_confidence")}
        bands = emotion.get("band_powers", {}) if isinstance(emotion, dict) else {}
        flow_components = flow.get("components", {}) if isinstance(flow, dict) else {}

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO brain_sessions
                   (user_id, session_id, start_time, end_time, duration_seconds,
                    sleep_stage, sleep_confidence,
                    emotion, emotion_confidence, valence, arousal,
                    flow_state, flow_score, absorption, effortlessness,
                    creativity_state, creativity_score, divergent_thinking, insight_potential,
                    encoding_state, encoding_score, will_remember_prob,
                    dream_detected, dream_confidence,
                    alpha_power, beta_power, theta_power, gamma_power, delta_power,
                    metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id,
                 session.get("session_id"),
                 session.get("start_time", time.time()),
                 session.get("end_time"),
                 session.get("duration_seconds"),
                 sleep.get("stage"),
                 sleep.get("confidence"),
                 emotion.get("emotion"),
                 emotion.get("confidence"),
                 emotion.get("valence"),
                 emotion.get("arousal"),
                 flow.get("state"),
                 flow.get("flow_score"),
                 flow_components.get("absorption"),
                 flow_components.get("effortlessness"),
                 creativity.get("state"),
                 creativity.get("creativity_score"),
                 creativity.get("components", {}).get("divergent_thinking"),
                 creativity.get("components", {}).get("insight_potential"),
                 memory.get("state"),
                 memory.get("encoding_score"),
                 memory.get("will_remember_probability"),
                 1 if dream.get("is_dreaming") else 0,
                 dream.get("confidence"),
                 bands.get("alpha"),
                 bands.get("beta"),
                 bands.get("theta"),
                 bands.get("gamma"),
                 bands.get("delta"),
                 json.dumps(session.get("metadata", {})))
            )

    def get_daily_summary(self, user_id: str, date: str = None) -> Dict:
        """Get combined brain + health summary for a day.

        Args:
            user_id: User identifier.
            date: Date string (YYYY-MM-DD). Defaults to today.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        day_start = datetime.strptime(date, "%Y-%m-%d").timestamp()
        day_end = day_start + 86400

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Brain sessions for the day
            brain_rows = conn.execute(
                """SELECT * FROM brain_sessions
                   WHERE user_id = ? AND start_time >= ? AND start_time < ?
                   ORDER BY start_time""",
                (user_id, day_start, day_end)
            ).fetchall()

            # Health samples for the day
            health_rows = conn.execute(
                """SELECT metric, AVG(value) as avg_val, MIN(value) as min_val,
                          MAX(value) as max_val, COUNT(*) as count
                   FROM health_samples
                   WHERE user_id = ? AND timestamp >= ? AND timestamp < ?
                   GROUP BY metric""",
                (user_id, day_start, day_end)
            ).fetchall()

        # Build summary
        brain_summary = {
            "total_sessions": len(brain_rows),
            "avg_flow_score": _safe_avg([r["flow_score"] for r in brain_rows]),
            "avg_creativity_score": _safe_avg([r["creativity_score"] for r in brain_rows]),
            "avg_encoding_score": _safe_avg([r["encoding_score"] for r in brain_rows]),
            "peak_flow_state": _mode([r["flow_state"] for r in brain_rows]),
            "dominant_emotion": _mode([r["emotion"] for r in brain_rows]),
            "dreams_detected": sum(1 for r in brain_rows if r["dream_detected"]),
            "time_in_flow_minutes": sum(
                (r["duration_seconds"] or 0) / 60 for r in brain_rows
                if r["flow_state"] in ("flow", "deep_flow")
            ),
        }

        health_summary = {}
        for row in health_rows:
            health_summary[row["metric"]] = {
                "average": round(row["avg_val"], 1) if row["avg_val"] else None,
                "min": round(row["min_val"], 1) if row["min_val"] else None,
                "max": round(row["max_val"], 1) if row["max_val"] else None,
                "count": row["count"],
            }

        return {
            "date": date,
            "user_id": user_id,
            "brain": brain_summary,
            "health": health_summary,
        }

    def generate_insights(self, user_id: str, days: int = 30) -> List[Dict]:
        """Generate personalized brain-health correlation insights.

        Analyzes patterns over the specified number of days.
        """
        cutoff = time.time() - (days * 86400)
        insights = []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            brain_rows = conn.execute(
                """SELECT * FROM brain_sessions
                   WHERE user_id = ? AND start_time >= ?
                   ORDER BY start_time""",
                (user_id, cutoff)
            ).fetchall()

            health_rows = conn.execute(
                """SELECT * FROM health_samples
                   WHERE user_id = ? AND timestamp >= ?
                   ORDER BY timestamp""",
                (user_id, cutoff)
            ).fetchall()

        if not brain_rows or not health_rows:
            return [{
                "type": "info",
                "title": "Not enough data yet",
                "description": f"We need at least a few days of both brain and health data to find patterns. "
                               f"Currently: {len(brain_rows)} brain sessions, {len(health_rows)} health samples.",
                "correlation_strength": 0,
            }]

        # Group by day for daily correlations
        brain_by_day = _group_by_day(brain_rows, "start_time")
        health_by_day = _group_by_day(health_rows, "timestamp")

        # === Correlation 1: Exercise → Flow State ===
        insights.extend(_correlate_metric_to_brain(
            health_by_day, brain_by_day,
            health_metric="steps",
            brain_field="flow_score",
            title_template="Exercise and Flow",
            high_desc="You enter flow states more easily on days you walk {threshold}+ steps",
            low_desc="Low-activity days correlate with lower flow scores",
        ))

        # === Correlation 2: HRV → Creativity ===
        insights.extend(_correlate_metric_to_brain(
            health_by_day, brain_by_day,
            health_metric="hrv_sdnn",
            brain_field="creativity_score",
            title_template="HRV and Creativity",
            high_desc="Higher heart rate variability correlates with more creative brain states",
            low_desc="Low HRV days show reduced creative thinking patterns",
        ))

        # === Correlation 3: Sleep Quality → Memory Encoding ===
        insights.extend(_correlate_metric_to_brain(
            health_by_day, brain_by_day,
            health_metric="sleep_analysis",
            brain_field="encoding_score",
            title_template="Sleep and Memory",
            high_desc="Better sleep quality leads to stronger memory encoding the next day",
            low_desc="Poor sleep nights are followed by weaker memory encoding",
        ))

        # === Correlation 4: Heart Rate → Stress/Emotion ===
        insights.extend(_correlate_metric_to_brain(
            health_by_day, brain_by_day,
            health_metric="heart_rate",
            brain_field="arousal",
            title_template="Heart Rate and Emotional Arousal",
            high_desc="Higher heart rate correlates with higher emotional arousal in your brain",
            low_desc="Calm heart rate periods show more relaxed brain patterns",
        ))

        # === Correlation 5: Mindful Minutes → Flow ===
        insights.extend(_correlate_metric_to_brain(
            health_by_day, brain_by_day,
            health_metric="mindful_minutes",
            brain_field="flow_score",
            title_template="Meditation and Flow",
            high_desc="Days with meditation sessions show {pct}% higher flow scores",
            low_desc="Meditation practice appears to boost your flow capacity",
        ))

        # === Pattern: Best time of day for flow ===
        if brain_rows:
            flow_by_hour = {}
            for row in brain_rows:
                if row["flow_score"] is not None:
                    hour = datetime.fromtimestamp(row["start_time"]).hour
                    flow_by_hour.setdefault(hour, []).append(row["flow_score"])

            if flow_by_hour:
                best_hour = max(flow_by_hour, key=lambda h: sum(flow_by_hour[h]) / len(flow_by_hour[h]))
                best_avg = sum(flow_by_hour[best_hour]) / len(flow_by_hour[best_hour])
                insights.append({
                    "type": "time_pattern",
                    "title": "Your Peak Flow Hour",
                    "description": f"Your brain enters flow most easily around {best_hour}:00 "
                                   f"(avg score: {best_avg:.2f}). Schedule deep work here.",
                    "correlation_strength": 0.7,
                    "evidence_count": len(flow_by_hour[best_hour]),
                    "brain_metric": "flow_score",
                    "health_metric": "time_of_day",
                })

        # === Pattern: Dream quality over time ===
        if brain_rows:
            dream_days = [r for r in brain_rows if r["dream_detected"]]
            if dream_days:
                insights.append({
                    "type": "dream_pattern",
                    "title": "Dream Activity",
                    "description": f"Dreams detected in {len(dream_days)} of {len(brain_rows)} sessions. "
                                   f"Dream analysis helps track REM quality and emotional processing.",
                    "correlation_strength": 0.5,
                    "evidence_count": len(dream_days),
                    "brain_metric": "dream_detected",
                    "health_metric": "sleep_analysis",
                })

        # Store insights
        with sqlite3.connect(self.db_path) as conn:
            for insight in insights:
                conn.execute(
                    """INSERT INTO insights
                       (user_id, insight_type, title, description,
                        correlation_strength, evidence_count,
                        brain_metric, health_metric)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (user_id, insight.get("type", "correlation"),
                     insight["title"], insight.get("description", ""),
                     insight.get("correlation_strength", 0),
                     insight.get("evidence_count", 0),
                     insight.get("brain_metric", ""),
                     insight.get("health_metric", ""))
                )

        return insights

    def get_insights(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Retrieve stored insights for a user."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM insights
                   WHERE user_id = ?
                   ORDER BY generated_at DESC
                   LIMIT ?""",
                (user_id, limit)
            ).fetchall()

        return [dict(row) for row in rows]

    def get_brain_trends(self, user_id: str, days: int = 30) -> Dict:
        """Get brain state trends over time."""
        cutoff = time.time() - (days * 86400)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT
                    date(start_time, 'unixepoch') as day,
                    AVG(flow_score) as avg_flow,
                    AVG(creativity_score) as avg_creativity,
                    AVG(encoding_score) as avg_encoding,
                    AVG(valence) as avg_valence,
                    AVG(arousal) as avg_arousal,
                    COUNT(*) as sessions
                   FROM brain_sessions
                   WHERE user_id = ? AND start_time >= ?
                   GROUP BY day
                   ORDER BY day""",
                (user_id, cutoff)
            ).fetchall()

        return {
            "days": days,
            "data_points": len(rows),
            "trends": [dict(row) for row in rows],
        }


# === Helper Functions ===

def _safe_avg(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _mode(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)


def _group_by_day(rows, time_field):
    groups = {}
    for row in rows:
        ts = row[time_field] if isinstance(row[time_field], (int, float)) else 0
        if ts > 0:
            day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            groups.setdefault(day, []).append(row)
    return groups


def _correlate_metric_to_brain(health_by_day, brain_by_day, health_metric,
                                brain_field, title_template, high_desc, low_desc):
    """Compute correlation between a health metric and brain metric."""
    insights = []
    common_days = set(health_by_day.keys()) & set(brain_by_day.keys())

    if len(common_days) < 3:
        return insights

    health_vals = []
    brain_vals = []

    for day in common_days:
        h_samples = [s for s in health_by_day[day]
                     if (s["metric"] if isinstance(s, dict) else "") == health_metric or
                        (s["metric"] if hasattr(s, "__getitem__") and "metric" in (s.keys() if hasattr(s, "keys") else []) else "") == health_metric]

        if not h_samples:
            continue

        h_val = sum(s["value"] for s in h_samples if s["value"] is not None) / max(len(h_samples), 1)
        b_vals = [b[brain_field] for b in brain_by_day[day] if b[brain_field] is not None]

        if b_vals:
            health_vals.append(h_val)
            brain_vals.append(sum(b_vals) / len(b_vals))

    if len(health_vals) < 3:
        return insights

    # Simple correlation
    import numpy as np
    h_arr = np.array(health_vals)
    b_arr = np.array(brain_vals)

    if np.std(h_arr) > 0 and np.std(b_arr) > 0:
        correlation = float(np.corrcoef(h_arr, b_arr)[0, 1])
    else:
        correlation = 0.0

    if abs(correlation) > 0.2:
        median_h = float(np.median(h_arr))
        desc = high_desc if correlation > 0 else low_desc
        desc = desc.replace("{threshold}", str(int(median_h)))
        desc = desc.replace("{pct}", str(int(abs(correlation) * 100)))

        insights.append({
            "type": "correlation",
            "title": title_template,
            "description": desc,
            "correlation_strength": round(abs(correlation), 3),
            "evidence_count": len(health_vals),
            "brain_metric": brain_field,
            "health_metric": health_metric,
        })

    return insights
