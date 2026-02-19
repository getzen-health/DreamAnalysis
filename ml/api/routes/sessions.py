"""Session recording and analytics endpoints."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ._shared import (
    _numpy_safe,
    _session_recorder,
    SessionRecorder,
    SessionStartRequest,
)

router = APIRouter()

# Import session analytics here to keep route order correct
# (analytics routes MUST be registered before /{session_id} to avoid catch-all)
from storage.session_analytics import compare_sessions, get_session_trends, get_weekly_report


@router.post("/sessions/start")
async def start_session(request: SessionStartRequest):
    """Start recording an EEG session."""
    session_id = _session_recorder.start_recording(
        user_id=request.user_id,
        session_type=request.session_type,
        metadata=request.metadata,
    )
    return {"status": "recording", "session_id": session_id}


@router.post("/sessions/stop")
async def stop_session():
    """Stop the current recording and return summary."""
    if not _session_recorder.is_recording:
        raise HTTPException(status_code=400, detail="No active recording")
    return _session_recorder.stop_recording()


@router.get("/sessions")
async def list_sessions(user_id: Optional[str] = None, session_type: Optional[str] = None):
    """List saved sessions."""
    return SessionRecorder.list_sessions(user_id, session_type)


@router.get("/sessions/trends")
async def session_trends(user_id: Optional[str] = None, last_n: int = 20):
    """Get trends across recent sessions."""
    return _numpy_safe(get_session_trends(user_id, last_n))


@router.get("/sessions/weekly-report")
async def weekly_report(user_id: Optional[str] = None):
    """Generate a weekly progress report comparing this week to last week."""
    return _numpy_safe(get_weekly_report(user_id))


@router.get("/sessions/compare/{session_a}/{session_b}")
async def compare_two_sessions(session_a: str, session_b: str):
    """Compare two sessions side-by-side with per-metric deltas."""
    result = compare_sessions(session_a, session_b)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return _numpy_safe(result)


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get full session data."""
    data = SessionRecorder.load_session(session_id)
    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])
    return data


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    deleted = SessionRecorder.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.get("/sessions/{session_id}/export")
async def export_session(session_id: str, format: str = "csv"):
    """Export session data as CSV."""
    data = SessionRecorder.export_session(session_id, format)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found or export failed")

    media_type = "text/csv" if format == "csv" else "application/octet-stream"
    filename = f"session_{session_id}.{format}"
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
