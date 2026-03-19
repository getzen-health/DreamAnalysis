"""Emotion archaeology API routes — reconstruct emotional timelines from digital artifacts.

POST /archaeology/reconstruct   -- full emotional archaeology report
POST /archaeology/analyze-text  -- analyze sentiment of text/journal entries
GET  /archaeology/status        -- model availability

Issue #453: Emotion archaeology.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.emotion_archaeology import (
    analyze_text_sentiment,
    generate_archaeology_report,
    report_to_dict,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/archaeology", tags=["emotion-archaeology"])


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class TextAnalysisRequest(BaseModel):
    """Request body for text sentiment analysis."""
    texts: List[str] = Field(
        ...,
        description="List of text entries (journal entries, messages, etc.) to analyze.",
        min_length=1,
    )


class MusicEntry(BaseModel):
    """A single music listening entry."""
    genre: str = Field(default="unknown", description="Music genre")
    track: Optional[str] = Field(default=None, description="Track name")
    valence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Audio valence 0-1")
    energy: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Audio energy 0-1")


class CalendarEntry(BaseModel):
    """A single calendar event."""
    event_type: str = Field(default="default", description="Event type (meeting, vacation, etc.)")
    name: str = Field(default="", description="Event name/description")


class PhotoEntry(BaseModel):
    """Photo metadata entry."""
    event_type: str = Field(default="default", description="Photo context (wedding, travel, etc.)")
    people_count: int = Field(default=0, ge=0, description="Number of people in the photo")
    timestamp: Optional[float] = Field(default=None, description="Photo timestamp (epoch seconds)")
    location: Optional[str] = Field(default=None, description="Location name")


class ReconstructRequest(BaseModel):
    """Request body for full emotional archaeology reconstruction."""
    user_id: str = Field(..., description="User identifier")
    texts: Optional[List[str]] = Field(default=None, description="Journal entries / text artifacts")
    music: Optional[List[MusicEntry]] = Field(default=None, description="Music listening history")
    calendar: Optional[List[CalendarEntry]] = Field(default=None, description="Calendar events")
    photos: Optional[List[PhotoEntry]] = Field(default=None, description="Photo metadata")
    period_start: str = Field(default="unknown", description="Start of analysis period")
    period_end: str = Field(default="unknown", description="End of analysis period")
    period_labels: Optional[List[str]] = Field(
        default=None,
        description="Labels for sub-periods (e.g. ['Q1', 'Q2', 'Q3', 'Q4'])",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status")
async def archaeology_status() -> Dict[str, Any]:
    """Return availability status for the emotion archaeology module."""
    return {
        "available": True,
        "model": "emotion_archaeology",
        "version": "1.0.0",
        "supported_artifacts": ["texts", "music", "calendar", "photos"],
        "timestamp": time.time(),
    }


@router.post("/analyze-text")
async def analyze_text(req: TextAnalysisRequest) -> Dict[str, Any]:
    """Analyze sentiment and emotional themes in text entries.

    Returns per-entry sentiment scores, detected themes, and aggregate stats.
    """
    results = []
    total_positive = 0.0
    total_negative = 0.0
    all_themes: List[str] = []

    for text in req.texts:
        sentiment = analyze_text_sentiment(text)
        total_positive += sentiment.positive_score
        total_negative += sentiment.negative_score
        all_themes.extend(sentiment.themes)

        results.append({
            "text_preview": sentiment.text_preview,
            "positive_score": sentiment.positive_score,
            "negative_score": sentiment.negative_score,
            "net_sentiment": sentiment.net_sentiment,
            "themes": sentiment.themes,
            "word_count": sentiment.word_count,
        })

    n = len(req.texts)
    # Deduplicate themes
    unique_themes = list(set(all_themes))

    return {
        "entries_analyzed": n,
        "results": results,
        "aggregate": {
            "avg_positive_score": round(total_positive / n, 4) if n > 0 else 0.0,
            "avg_negative_score": round(total_negative / n, 4) if n > 0 else 0.0,
            "detected_themes": unique_themes,
        },
        "timestamp": time.time(),
    }


@router.post("/reconstruct")
async def reconstruct_timeline(req: ReconstructRequest) -> Dict[str, Any]:
    """Reconstruct an emotional timeline from digital artifacts.

    Accepts texts, music history, calendar events, and photo metadata.
    Returns a full emotional archaeology report with per-period breakdowns,
    overall trajectory, narrative summary, and recommendations.
    """
    # Build artifacts dict from request
    artifacts: Dict[str, Any] = {
        "texts": req.texts or [],
        "music": [m.model_dump() for m in (req.music or [])],
        "calendar": [c.model_dump() for c in (req.calendar or [])],
        "photos": [p.model_dump() for p in (req.photos or [])],
    }

    report = generate_archaeology_report(
        user_id=req.user_id,
        artifacts=artifacts,
        period_start=req.period_start,
        period_end=req.period_end,
        period_labels=req.period_labels,
    )

    return report_to_dict(report)
