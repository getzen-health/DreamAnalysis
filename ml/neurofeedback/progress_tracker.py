"""Cross-session neurofeedback progress tracking with dosing guidance.

Tracks per-user, per-protocol progress across sessions:
- Session count and frequency
- Resting baseline trajectory
- Score improvement curves
- Dosing compliance (sessions per week)

References:
    Gruzelier (2014): min 10 sessions for measurable effects, 20-40 for stable
    Enriquez-Geppert et al. (2019): 25-30 sessions at 2-3x/week for ADHD
    Hammond (2011): max 1 session/day, 2-3x/week optimal
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

_USER_DATA_DIR = Path(__file__).resolve().parent.parent / "user_data"


class NeurofeedbackProgressTracker:
    """Track neurofeedback training progress across sessions."""

    def __init__(self, user_id: str, protocol_type: str):
        self.user_id = user_id
        self.protocol_type = protocol_type
        self._sessions: List[Dict] = []
        self._load()

    def record_session(
        self,
        duration_minutes: float,
        avg_score: float,
        reward_rate: float,
        max_streak: int,
        baseline_value: Optional[float] = None,
    ) -> Dict:
        """Record a completed neurofeedback session."""
        entry = {
            "session_number": len(self._sessions) + 1,
            "timestamp": time.time(),
            "duration_minutes": round(duration_minutes, 1),
            "avg_score": round(avg_score, 4),
            "reward_rate": round(reward_rate, 4),
            "max_streak": max_streak,
            "baseline_value": round(baseline_value, 6) if baseline_value is not None else None,
        }
        self._sessions.append(entry)
        self._save()
        return entry

    def get_progress(self) -> Dict:
        """Return cross-session progress summary."""
        n = len(self._sessions)
        if n == 0:
            return {
                "user_id": self.user_id,
                "protocol": self.protocol_type,
                "total_sessions": 0,
                "status": "not_started",
                "message": "No sessions recorded yet. Start your first neurofeedback session.",
            }

        scores = [s["avg_score"] for s in self._sessions]
        reward_rates = [s["reward_rate"] for s in self._sessions]
        baselines = [s["baseline_value"] for s in self._sessions if s.get("baseline_value") is not None]

        # Score improvement
        first_3_avg = float(np.mean(scores[:3])) if len(scores) >= 3 else scores[0]
        last_3_avg = float(np.mean(scores[-3:])) if len(scores) >= 3 else scores[-1]
        improvement_pct = ((last_3_avg - first_3_avg) / max(abs(first_3_avg), 0.01)) * 100

        # Baseline trajectory
        baseline_change_pct = None
        if len(baselines) >= 2:
            baseline_change_pct = ((baselines[-1] - baselines[0]) / max(abs(baselines[0]), 0.01)) * 100

        # Dosing compliance
        dosing = self._compute_dosing()

        # Estimated sessions remaining (target: 30)
        target = 30
        remaining = max(0, target - n)

        # Phase
        if n < 10:
            phase = "early_learning"
            phase_msg = f"Session {n}/10 — building neural pathways. Effects typically emerge after 10 sessions."
        elif n < 20:
            phase = "consolidation"
            phase_msg = f"Session {n}/20 — consolidating gains. Continue 2-3x/week for best results."
        elif n < 30:
            phase = "mastery"
            phase_msg = f"Session {n}/30 — approaching mastery. Resting state should show training effects."
        else:
            phase = "maintenance"
            phase_msg = f"Session {n} — maintenance phase. 1x/week is sufficient to maintain gains."

        return {
            "user_id": self.user_id,
            "protocol": self.protocol_type,
            "total_sessions": n,
            "phase": phase,
            "phase_message": phase_msg,
            "score_trend": {
                "first_3_avg": round(first_3_avg, 4),
                "last_3_avg": round(last_3_avg, 4),
                "improvement_pct": round(improvement_pct, 1),
                "all_scores": [round(s, 4) for s in scores],
            },
            "reward_rate_trend": [round(r, 4) for r in reward_rates],
            "baseline_trajectory": {
                "values": [round(b, 6) for b in baselines],
                "change_pct": round(baseline_change_pct, 1) if baseline_change_pct is not None else None,
            },
            "dosing": dosing,
            "sessions_remaining": remaining,
            "target_sessions": target,
        }

    def get_dosing_alerts(self) -> List[Dict]:
        """Check dosing and return any alerts."""
        alerts = []
        if not self._sessions:
            return alerts

        last_ts = self._sessions[-1]["timestamp"]
        now = time.time()
        hours_since_last = (now - last_ts) / 3600
        days_since_last = hours_since_last / 24

        # Same-day session
        today_sessions = sum(
            1 for s in self._sessions
            if (now - s["timestamp"]) < 86400
        )
        if today_sessions >= 1:
            alerts.append({
                "level": "warning",
                "type": "same_day_session",
                "message": "You already trained today. Multiple sessions per day show no additional benefit and may cause adverse effects (Hammond 2011).",
            })

        # Gap too long
        if days_since_last > 14:
            alerts.append({
                "level": "warning",
                "type": "training_gap",
                "message": f"It has been {int(days_since_last)} days since your last session. Gaps >2 weeks cause partial skill decay. Consider resuming 2-3x/week.",
            })

        # Maintenance mode suggestion
        if len(self._sessions) >= 30:
            alerts.append({
                "level": "info",
                "type": "maintenance_mode",
                "message": "You have completed 30+ sessions. You can switch to maintenance mode (1x/week) to sustain gains.",
            })

        return alerts

    def _compute_dosing(self) -> Dict:
        """Compute dosing compliance metrics."""
        if len(self._sessions) < 2:
            return {"sessions_per_week": 0, "compliance": "insufficient_data"}

        timestamps = [s["timestamp"] for s in self._sessions]
        total_weeks = max((timestamps[-1] - timestamps[0]) / (7 * 86400), 0.5)
        sessions_per_week = len(self._sessions) / total_weeks

        # Optimal: 2-3x/week
        if 1.5 <= sessions_per_week <= 3.5:
            compliance = "optimal"
        elif sessions_per_week < 1.5:
            compliance = "under_training"
        else:
            compliance = "over_training"

        return {
            "sessions_per_week": round(sessions_per_week, 1),
            "compliance": compliance,
            "recommended": "2-3 sessions per week, spaced at least 1 day apart",
        }

    def _save(self):
        """Persist progress to disk."""
        user_dir = _USER_DATA_DIR / self.user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        path = user_dir / f"nf_progress_{self.protocol_type}.json"
        path.write_text(json.dumps(self._sessions, indent=2))

    def _load(self):
        """Load progress from disk."""
        path = _USER_DATA_DIR / self.user_id / f"nf_progress_{self.protocol_type}.json"
        if path.exists():
            try:
                self._sessions = json.loads(path.read_text())
            except Exception:
                self._sessions = []
