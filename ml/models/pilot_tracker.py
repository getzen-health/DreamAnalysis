"""Feasibility pilot session tracker (#200).

Tracks 10-person pilot study sessions: start/complete sessions,
collect self-report survey data, compute study metrics.

Each participant does 5 sessions over 2 weeks.
Per session: 2-min baseline → 5-min task → biofeedback → survey.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

_SESSIONS: Dict[str, List[dict]] = defaultdict(list)


class PilotTracker:
    """Singleton: tracks pilot study participants and sessions."""

    TARGET_PARTICIPANTS = 10
    SESSIONS_PER_PARTICIPANT = 5
    MIN_SQI_THRESHOLD = 40.0  # minimum signal quality index %
    TARGET_COMPLETION_RATE = 0.80
    TARGET_USABLE_EPOCH_RATE = 0.70

    def start_session(
        self,
        participant_id: str,
        eeg_device: str = "Muse 2",
    ) -> dict:
        """Start a new pilot session for a participant."""
        sessions = _SESSIONS[participant_id]
        session_num = len(sessions) + 1
        if session_num > self.SESSIONS_PER_PARTICIPANT:
            return {
                "participant_id": participant_id,
                "session_id": None,
                "status": "quota_reached",
                "message": f"Participant has completed all {self.SESSIONS_PER_PARTICIPANT} sessions.",
            }
        session_id = f"{participant_id}-s{session_num}-{int(time.time())}"
        record: dict = {
            "session_id": session_id,
            "participant_id": participant_id,
            "session_num": session_num,
            "eeg_device": eeg_device,
            "started_at": time.time(),
            "completed_at": None,
            "status": "active",
            "phases": {
                "baseline_done": False,
                "task_done": False,
                "biofeedback_done": False,
                "survey_done": False,
            },
            "eeg_epochs_total": 0,
            "eeg_epochs_usable": 0,
            "survey": None,
        }
        _SESSIONS[participant_id].append(record)
        return {
            "participant_id": participant_id,
            "session_id": session_id,
            "session_num": session_num,
            "status": "active",
            "instructions": [
                "Phase 1: 2-min eyes-closed baseline (do not record yet)",
                "Phase 2: 5-min cognitive task with EEG streaming",
                "Phase 3: Biofeedback breathing exercise",
                "Phase 4: Post-session survey",
            ],
        }

    def complete_session(
        self,
        participant_id: str,
        session_id: str,
        eeg_epochs_total: int,
        eeg_epochs_usable: int,
        survey: dict,
    ) -> dict:
        """Complete a session with EEG quality metrics and survey."""
        for rec in _SESSIONS[participant_id]:
            if rec["session_id"] == session_id:
                rec["completed_at"] = time.time()
                rec["status"] = "completed"
                rec["eeg_epochs_total"] = max(0, eeg_epochs_total)
                rec["eeg_epochs_usable"] = max(0, min(eeg_epochs_usable, eeg_epochs_total))
                rec["survey"] = survey
                rec["phases"]["survey_done"] = True
                rec["phases"]["baseline_done"] = True
                rec["phases"]["task_done"] = True
                rec["phases"]["biofeedback_done"] = True
                usable_rate = (
                    rec["eeg_epochs_usable"] / rec["eeg_epochs_total"]
                    if rec["eeg_epochs_total"] > 0 else 0.0
                )
                duration_min = (rec["completed_at"] - rec["started_at"]) / 60.0
                return {
                    "session_id": session_id,
                    "status": "completed",
                    "duration_min": round(duration_min, 1),
                    "usable_epoch_rate": round(usable_rate, 3),
                    "passes_sqi_threshold": usable_rate >= (self.MIN_SQI_THRESHOLD / 100),
                }
        return {"session_id": session_id, "status": "not_found"}

    def get_participant_status(self, participant_id: str) -> dict:
        """Return session count and completion status for a participant."""
        sessions = _SESSIONS[participant_id]
        completed = [s for s in sessions if s["status"] == "completed"]
        return {
            "participant_id": participant_id,
            "sessions_total": len(sessions),
            "sessions_completed": len(completed),
            "sessions_remaining": max(0, self.SESSIONS_PER_PARTICIPANT - len(completed)),
            "pilot_complete": len(completed) >= self.SESSIONS_PER_PARTICIPANT,
        }

    def get_study_metrics(self) -> dict:
        """Compute study-level feasibility metrics across all participants."""
        all_participants = list(_SESSIONS.keys())
        n = len(all_participants)
        completed_sessions: List[dict] = []
        for pid in all_participants:
            completed_sessions.extend(
                s for s in _SESSIONS[pid] if s["status"] == "completed"
            )

        total_sessions = sum(len(_SESSIONS[pid]) for pid in all_participants)
        n_completed = len(completed_sessions)
        completion_rate = n_completed / total_sessions if total_sessions > 0 else 0.0

        usable_rates = []
        for s in completed_sessions:
            if s["eeg_epochs_total"] > 0:
                usable_rates.append(s["eeg_epochs_usable"] / s["eeg_epochs_total"])

        avg_usable_epoch_rate = float(np.mean(usable_rates)) if usable_rates else 0.0
        passes_completion_target = completion_rate >= self.TARGET_COMPLETION_RATE
        passes_sqi_target = avg_usable_epoch_rate >= self.TARGET_USABLE_EPOCH_RATE

        pilots_complete = sum(
            1 for pid in all_participants
            if sum(1 for s in _SESSIONS[pid] if s["status"] == "completed") >= self.SESSIONS_PER_PARTICIPANT
        )

        return {
            "participants_enrolled": n,
            "participants_completed_pilot": pilots_complete,
            "target_participants": self.TARGET_PARTICIPANTS,
            "total_sessions": total_sessions,
            "completed_sessions": n_completed,
            "completion_rate": round(completion_rate, 3),
            "avg_usable_epoch_rate": round(avg_usable_epoch_rate, 3),
            "passes_completion_target": passes_completion_target,
            "passes_sqi_target": passes_sqi_target,
            "feasibility_verdict": (
                "go" if passes_completion_target and passes_sqi_target else "hold"
            ),
        }

    def list_participants(self) -> List[dict]:
        """List all enrolled participants with basic stats."""
        result = []
        for pid in sorted(_SESSIONS.keys()):
            sessions = _SESSIONS[pid]
            completed = [s for s in sessions if s["status"] == "completed"]
            result.append({
                "participant_id": pid,
                "sessions_completed": len(completed),
                "sessions_total": len(sessions),
            })
        return result

    def reset(self, participant_id: Optional[str] = None) -> dict:
        """Clear session data (for testing)."""
        if participant_id:
            _SESSIONS.pop(participant_id, None)
            return {"cleared": participant_id}
        _SESSIONS.clear()
        return {"cleared": "all"}


_instance: Optional[PilotTracker] = None


def get_tracker() -> PilotTracker:
    global _instance
    if _instance is None:
        _instance = PilotTracker()
    return _instance
