"""PHQ-9/GAD-7 mental health questionnaires for supervised training data (#25).

Provides structured endpoints for collecting validated self-report scores that
can be paired with EEG session data to create labeled datasets for supervised
learning of mental health biomarkers.

PHQ-9: Patient Health Questionnaire (depression screening, 0-27)
GAD-7: Generalized Anxiety Disorder scale (0-21)
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

router = APIRouter(prefix="/mental-health", tags=["mental-health-questionnaire"])

# ---------------------------------------------------------------------------
# PHQ-9 questions (Kroenke & Spitzer, 2002)
# ---------------------------------------------------------------------------

PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure",
    "Trouble concentrating on things (reading, TV)",
    "Moving or speaking so slowly that others noticed; or fidgety/restless",
    "Thoughts that you would be better off dead or of hurting yourself",
]

GAD7_QUESTIONS = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid, as if something awful might happen",
]

RESPONSE_SCALE = {
    0: "Not at all",
    1: "Several days",
    2: "More than half the days",
    3: "Nearly every day",
}

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PHQ9Submission(BaseModel):
    user_id: str
    responses: List[int] = Field(..., min_length=9, max_length=9,
                                  description="9 integers 0-3")
    session_id: Optional[str] = None
    eeg_session_id: Optional[str] = None   # for pairing with EEG data

    @validator("responses", each_item=True)
    def valid_response(cls, v):
        if v not in (0, 1, 2, 3):
            raise ValueError("Each response must be 0, 1, 2, or 3")
        return v


class GAD7Submission(BaseModel):
    user_id: str
    responses: List[int] = Field(..., min_length=7, max_length=7,
                                  description="7 integers 0-3")
    session_id: Optional[str] = None
    eeg_session_id: Optional[str] = None


class PHQ9Result(BaseModel):
    user_id: str
    total_score: int
    severity: str
    item_scores: List[int]
    clinical_action: str
    submitted_at: float
    eeg_session_id: Optional[str]


class GAD7Result(BaseModel):
    user_id: str
    total_score: int
    severity: str
    item_scores: List[int]
    clinical_action: str
    submitted_at: float
    eeg_session_id: Optional[str]


class TrendReport(BaseModel):
    user_id: str
    n_phq9: int
    n_gad7: int
    phq9_trend: List[int]
    gad7_trend: List[int]
    phq9_slope: Optional[float]
    gad7_slope: Optional[float]
    combined_distress_score: float
    training_data_label: str


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _phq9_severity(score: int) -> tuple:
    if score < 5:
        return "Minimal or none", "Monitor"
    elif score < 10:
        return "Mild", "Watchful waiting, repeat PHQ-9 at follow-up"
    elif score < 15:
        return "Moderate", "Treatment plan, counseling, follow-up"
    elif score < 20:
        return "Moderately severe", "Active treatment — medication and/or psychotherapy"
    else:
        return "Severe", "Immediate referral for psychiatric evaluation"


def _gad7_severity(score: int) -> tuple:
    if score < 5:
        return "Minimal", "Monitor"
    elif score < 10:
        return "Mild", "Monitor, stress management guidance"
    elif score < 15:
        return "Moderate", "Consider therapy referral"
    else:
        return "Severe", "Active treatment — consider medication and psychotherapy"


# ---------------------------------------------------------------------------
# In-memory storage
# ---------------------------------------------------------------------------

_phq9_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
_gad7_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/phq9-questions")
async def get_phq9_questions():
    """Return the PHQ-9 questionnaire items and response scale."""
    return {
        "questionnaire": "PHQ-9 (Patient Health Questionnaire)",
        "reference": "Kroenke & Spitzer (2002). Psychosomatics 43(3):175-184",
        "questions": [{"index": i, "text": q} for i, q in enumerate(PHQ9_QUESTIONS)],
        "response_scale": RESPONSE_SCALE,
        "note": "Rate how often each problem has bothered you over the LAST 2 WEEKS",
    }


@router.get("/gad7-questions")
async def get_gad7_questions():
    """Return the GAD-7 questionnaire items and response scale."""
    return {
        "questionnaire": "GAD-7 (Generalized Anxiety Disorder)",
        "reference": "Spitzer et al. (2006). Archives of Internal Medicine 166(10):1092-1097",
        "questions": [{"index": i, "text": q} for i, q in enumerate(GAD7_QUESTIONS)],
        "response_scale": RESPONSE_SCALE,
        "note": "Rate how often each problem has bothered you over the LAST 2 WEEKS",
    }


@router.post("/phq9", response_model=PHQ9Result)
async def submit_phq9(req: PHQ9Submission):
    """Submit PHQ-9 responses and receive scored result."""
    total = sum(req.responses)
    severity, action = _phq9_severity(total)

    result = PHQ9Result(
        user_id=req.user_id,
        total_score=total,
        severity=severity,
        item_scores=req.responses,
        clinical_action=action,
        submitted_at=time.time(),
        eeg_session_id=req.eeg_session_id,
    )
    _phq9_history[req.user_id].append(result.dict())
    return result


@router.post("/gad7", response_model=GAD7Result)
async def submit_gad7(req: GAD7Submission):
    """Submit GAD-7 responses and receive scored result."""
    total = sum(req.responses)
    severity, action = _gad7_severity(total)

    result = GAD7Result(
        user_id=req.user_id,
        total_score=total,
        severity=severity,
        item_scores=req.responses,
        clinical_action=action,
        submitted_at=time.time(),
        eeg_session_id=req.eeg_session_id,
    )
    _gad7_history[req.user_id].append(result.dict())
    return result


@router.get("/history/{user_id}")
async def get_questionnaire_history(user_id: str):
    """Return all PHQ-9 and GAD-7 submissions for a user."""
    return {
        "user_id": user_id,
        "phq9": list(_phq9_history[user_id]),
        "gad7": list(_gad7_history[user_id]),
    }


@router.get("/trends/{user_id}", response_model=TrendReport)
async def get_mental_health_trends(user_id: str):
    """Return trend analysis and a training data label for supervised EEG models."""
    phq9_records = list(_phq9_history[user_id])
    gad7_records = list(_gad7_history[user_id])

    phq9_scores = [r["total_score"] for r in phq9_records]
    gad7_scores = [r["total_score"] for r in gad7_records]

    phq9_slope = None
    gad7_slope = None
    if len(phq9_scores) >= 3:
        phq9_slope = float(np.polyfit(range(len(phq9_scores)), phq9_scores, 1)[0])
    if len(gad7_scores) >= 3:
        gad7_slope = float(np.polyfit(range(len(gad7_scores)), gad7_scores, 1)[0])

    latest_phq9 = phq9_scores[-1] if phq9_scores else 0
    latest_gad7 = gad7_scores[-1] if gad7_scores else 0
    combined    = float(latest_phq9 / 27.0 * 0.5 + latest_gad7 / 21.0 * 0.5)

    # Training label for supervised learning
    if combined < 0.2:
        label = "healthy"
    elif combined < 0.4:
        label = "mild_distress"
    elif combined < 0.6:
        label = "moderate_distress"
    else:
        label = "severe_distress"

    return TrendReport(
        user_id=user_id,
        n_phq9=len(phq9_scores),
        n_gad7=len(gad7_scores),
        phq9_trend=phq9_scores[-7:],
        gad7_trend=gad7_scores[-7:],
        phq9_slope=phq9_slope,
        gad7_slope=gad7_slope,
        combined_distress_score=combined,
        training_data_label=label,
    )
