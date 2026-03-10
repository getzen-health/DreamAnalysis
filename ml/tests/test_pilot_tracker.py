"""Tests for pilot_tracker model and API (#200)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models.pilot_tracker import PilotTracker, get_tracker, _SESSIONS


@pytest.fixture(autouse=True)
def clear_state():
    """Reset global state before each test."""
    _SESSIONS.clear()
    yield
    _SESSIONS.clear()


@pytest.fixture
def tracker():
    return PilotTracker()


# ── start_session ─────────────────────────────────────────────────────────────

def test_start_session_returns_session_id(tracker):
    result = tracker.start_session("p01")
    assert "session_id" in result
    assert result["session_id"] is not None


def test_start_session_status_active(tracker):
    result = tracker.start_session("p01")
    assert result["status"] == "active"


def test_start_session_correct_session_num(tracker):
    result = tracker.start_session("p01")
    assert result["session_num"] == 1


def test_start_session_increments_num(tracker):
    r1 = tracker.start_session("p01")
    # complete so next session can start
    tracker.complete_session("p01", r1["session_id"], 10, 8, {})
    r2 = tracker.start_session("p01")
    assert r2["session_num"] == 2


def test_start_session_quota_reached(tracker):
    for i in range(PilotTracker.SESSIONS_PER_PARTICIPANT):
        r = tracker.start_session("p01")
        tracker.complete_session("p01", r["session_id"], 10, 8, {})
    result = tracker.start_session("p01")
    assert result["status"] == "quota_reached"
    assert result["session_id"] is None


def test_start_session_includes_instructions(tracker):
    result = tracker.start_session("p01")
    assert "instructions" in result
    assert len(result["instructions"]) == 4


def test_start_session_different_participants_independent(tracker):
    r1 = tracker.start_session("p01")
    r2 = tracker.start_session("p02")
    assert r1["session_id"] != r2["session_id"]
    assert r1["session_num"] == r2["session_num"] == 1


# ── complete_session ──────────────────────────────────────────────────────────

def test_complete_session_status(tracker):
    r = tracker.start_session("p01")
    result = tracker.complete_session("p01", r["session_id"], 20, 16, {"stress": 3})
    assert result["status"] == "completed"


def test_complete_session_usable_epoch_rate(tracker):
    r = tracker.start_session("p01")
    result = tracker.complete_session("p01", r["session_id"], 20, 16, {})
    assert abs(result["usable_epoch_rate"] - 0.8) < 0.001


def test_complete_session_passes_sqi_threshold(tracker):
    r = tracker.start_session("p01")
    result = tracker.complete_session("p01", r["session_id"], 20, 16, {})
    assert result["passes_sqi_threshold"] is True


def test_complete_session_fails_sqi_threshold(tracker):
    r = tracker.start_session("p01")
    result = tracker.complete_session("p01", r["session_id"], 20, 5, {})
    assert result["passes_sqi_threshold"] is False


def test_complete_session_not_found(tracker):
    tracker.start_session("p01")
    result = tracker.complete_session("p01", "bad-id", 10, 8, {})
    assert result["status"] == "not_found"


def test_complete_session_zero_epochs(tracker):
    r = tracker.start_session("p01")
    result = tracker.complete_session("p01", r["session_id"], 0, 0, {})
    assert result["usable_epoch_rate"] == 0.0


# ── participant_status ─────────────────────────────────────────────────────────

def test_participant_status_empty(tracker):
    result = tracker.get_participant_status("unknown")
    assert result["sessions_completed"] == 0
    assert result["pilot_complete"] is False


def test_participant_status_after_completion(tracker):
    for _ in range(PilotTracker.SESSIONS_PER_PARTICIPANT):
        r = tracker.start_session("p01")
        tracker.complete_session("p01", r["session_id"], 10, 8, {})
    status = tracker.get_participant_status("p01")
    assert status["pilot_complete"] is True
    assert status["sessions_remaining"] == 0


# ── study_metrics ─────────────────────────────────────────────────────────────

def test_study_metrics_empty(tracker):
    m = tracker.get_study_metrics()
    assert m["participants_enrolled"] == 0
    assert m["completion_rate"] == 0.0
    assert m["feasibility_verdict"] == "hold"


def test_study_metrics_verdict_go(tracker):
    # 1 participant, all 5 sessions, good SQI
    for _ in range(PilotTracker.SESSIONS_PER_PARTICIPANT):
        r = tracker.start_session("p01")
        tracker.complete_session("p01", r["session_id"], 10, 9, {})
    m = tracker.get_study_metrics()
    assert m["completion_rate"] == 1.0
    assert m["avg_usable_epoch_rate"] >= 0.7
    assert m["feasibility_verdict"] == "go"


def test_study_metrics_participants_count(tracker):
    for pid in ["p01", "p02", "p03"]:
        r = tracker.start_session(pid)
        tracker.complete_session(pid, r["session_id"], 10, 8, {})
    m = tracker.get_study_metrics()
    assert m["participants_enrolled"] == 3


# ── list_participants ─────────────────────────────────────────────────────────

def test_list_participants_empty(tracker):
    result = tracker.list_participants()
    assert result == []


def test_list_participants_sorted(tracker):
    for pid in ["p03", "p01", "p02"]:
        tracker.start_session(pid)
    result = tracker.list_participants()
    ids = [p["participant_id"] for p in result]
    assert ids == sorted(ids)


# ── reset ─────────────────────────────────────────────────────────────────────

def test_reset_all(tracker):
    tracker.start_session("p01")
    tracker.start_session("p02")
    tracker.reset()
    assert tracker.list_participants() == []


def test_reset_single_participant(tracker):
    tracker.start_session("p01")
    tracker.start_session("p02")
    tracker.reset("p01")
    ids = [p["participant_id"] for p in tracker.list_participants()]
    assert "p01" not in ids
    assert "p02" in ids


# ── singleton ─────────────────────────────────────────────────────────────────

def test_singleton_is_same_instance():
    a = get_tracker()
    b = get_tracker()
    assert a is b
