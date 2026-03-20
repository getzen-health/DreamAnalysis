"""Elderly cognitive-emotional monitoring API routes (#448).

POST /elderly/assess   -- compute MCI risk assessment from sub-domain data
POST /elderly/profile  -- full monitoring profile across all domains
GET  /elderly/status   -- model availability check
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.elderly_monitoring import (
    CognitiveReading,
    EmotionRangeReading,
    EmotionalMemoryReading,
    MonitoringProfile,
    ProcessingSpeedReading,
    SocialEngagementReading,
    _CLINICAL_DISCLAIMER,
    compute_cognitive_emotional_coupling,
    compute_mci_risk_score,
    compute_monitoring_profile,
    compute_processing_speed,
    assess_emotional_memory,
    detect_emotional_flattening,
    profile_to_dict,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/elderly", tags=["elderly-monitoring"])


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class ProcessingSpeedItem(BaseModel):
    reaction_time_ms: float = Field(..., gt=0, description="Reaction time in ms")
    identification_time_ms: float = Field(..., gt=0, description="Identification time in ms")
    stimulus_type: str = Field(default="face", description="Type of stimulus")
    emotion_presented: str = Field(default="neutral", description="Emotion shown")
    correct: bool = Field(default=True, description="Was the response correct")
    timestamp: float = Field(default=0.0, description="Unix timestamp")


class EmotionRangeItem(BaseModel):
    distinct_emotions: int = Field(..., ge=0, description="Distinct emotions observed")
    dominant_emotion: str = Field(default="neutral")
    valence_range: float = Field(default=0.0, ge=0.0, le=2.0)
    arousal_range: float = Field(default=0.0, ge=0.0, le=1.0)
    window_days: int = Field(default=7, ge=1)
    timestamp: float = Field(default=0.0)


class CognitiveItem(BaseModel):
    attention_score: float = Field(default=0.5, ge=0.0, le=1.0)
    memory_score: float = Field(default=0.5, ge=0.0, le=1.0)
    executive_score: float = Field(default=0.5, ge=0.0, le=1.0)
    processing_speed_score: float = Field(default=0.5, ge=0.0, le=1.0)
    timestamp: float = Field(default=0.0)


class EmotionalMemoryItem(BaseModel):
    emotional_items_recalled: int = Field(..., ge=0)
    neutral_items_recalled: int = Field(..., ge=0)
    emotional_items_total: int = Field(default=10, ge=1)
    neutral_items_total: int = Field(default=10, ge=1)
    delay_minutes: int = Field(default=30, ge=0)
    timestamp: float = Field(default=0.0)


class SocialEngagementItem(BaseModel):
    interactions_count: int = Field(..., ge=0)
    unique_contacts: int = Field(default=0, ge=0)
    initiated_count: int = Field(default=0, ge=0)
    avg_duration_minutes: float = Field(default=0.0, ge=0.0)
    window_days: int = Field(default=7, ge=1)
    timestamp: float = Field(default=0.0)


class AssessRequest(BaseModel):
    processing_speed_readings: Optional[List[ProcessingSpeedItem]] = Field(
        default=None, description="Emotional processing speed measurements"
    )
    emotion_range_readings: Optional[List[EmotionRangeItem]] = Field(
        default=None, description="Emotional range snapshots"
    )
    cognitive_readings: Optional[List[CognitiveItem]] = Field(
        default=None, description="Cognitive performance snapshots"
    )
    emotional_memory_readings: Optional[List[EmotionalMemoryItem]] = Field(
        default=None, description="Emotional memory advantage measurements"
    )
    social_engagement_readings: Optional[List[SocialEngagementItem]] = Field(
        default=None, description="Social engagement metrics"
    )


class ProfileRequest(BaseModel):
    processing_speed_readings: Optional[List[ProcessingSpeedItem]] = None
    emotion_range_readings: Optional[List[EmotionRangeItem]] = None
    cognitive_readings: Optional[List[CognitiveItem]] = None
    emotional_memory_readings: Optional[List[EmotionalMemoryItem]] = None
    social_engagement_readings: Optional[List[SocialEngagementItem]] = None


# ---------------------------------------------------------------------------
# Helpers -- convert Pydantic items to dataclass instances
# ---------------------------------------------------------------------------


def _to_ps_readings(items: Optional[List[ProcessingSpeedItem]]) -> List[ProcessingSpeedReading]:
    if not items:
        return []
    return [
        ProcessingSpeedReading(
            reaction_time_ms=i.reaction_time_ms,
            identification_time_ms=i.identification_time_ms,
            stimulus_type=i.stimulus_type,
            emotion_presented=i.emotion_presented,
            correct=i.correct,
            timestamp=i.timestamp,
        )
        for i in items
    ]


def _to_er_readings(items: Optional[List[EmotionRangeItem]]) -> List[EmotionRangeReading]:
    if not items:
        return []
    return [
        EmotionRangeReading(
            distinct_emotions=i.distinct_emotions,
            dominant_emotion=i.dominant_emotion,
            valence_range=i.valence_range,
            arousal_range=i.arousal_range,
            window_days=i.window_days,
            timestamp=i.timestamp,
        )
        for i in items
    ]


def _to_cog_readings(items: Optional[List[CognitiveItem]]) -> List[CognitiveReading]:
    if not items:
        return []
    return [
        CognitiveReading(
            attention_score=i.attention_score,
            memory_score=i.memory_score,
            executive_score=i.executive_score,
            processing_speed_score=i.processing_speed_score,
            timestamp=i.timestamp,
        )
        for i in items
    ]


def _to_em_readings(items: Optional[List[EmotionalMemoryItem]]) -> List[EmotionalMemoryReading]:
    if not items:
        return []
    return [
        EmotionalMemoryReading(
            emotional_items_recalled=i.emotional_items_recalled,
            neutral_items_recalled=i.neutral_items_recalled,
            emotional_items_total=i.emotional_items_total,
            neutral_items_total=i.neutral_items_total,
            delay_minutes=i.delay_minutes,
            timestamp=i.timestamp,
        )
        for i in items
    ]


def _to_se_readings(items: Optional[List[SocialEngagementItem]]) -> List[SocialEngagementReading]:
    if not items:
        return []
    return [
        SocialEngagementReading(
            interactions_count=i.interactions_count,
            unique_contacts=i.unique_contacts,
            initiated_count=i.initiated_count,
            avg_duration_minutes=i.avg_duration_minutes,
            window_days=i.window_days,
            timestamp=i.timestamp,
        )
        for i in items
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/assess")
async def assess_mci_risk(req: AssessRequest) -> Dict[str, Any]:
    """Compute MCI risk assessment from cognitive-emotional indicators.

    Accepts any combination of sub-domain readings. Missing domains
    contribute 0.0 to the composite risk score.

    Wellness indicator only -- not a medical device or clinical assessment.
    """
    ps = compute_processing_speed(_to_ps_readings(req.processing_speed_readings))
    fl = detect_emotional_flattening(_to_er_readings(req.emotion_range_readings))
    co = compute_cognitive_emotional_coupling(
        _to_cog_readings(req.cognitive_readings),
        _to_er_readings(req.emotion_range_readings),
    )
    em = assess_emotional_memory(_to_em_readings(req.emotional_memory_readings))

    # Social engagement uses private helper in model
    from models.elderly_monitoring import _compute_social_engagement
    se = _compute_social_engagement(_to_se_readings(req.social_engagement_readings))

    risk = compute_mci_risk_score(ps, fl, co, em, se)

    return {
        "mci_risk_score": risk["mci_risk_score"],
        "risk_category": risk["risk_category"],
        "sub_scores": risk["sub_scores"],
        "contributing_factors": risk["contributing_factors"],
        "flags": risk["flags"],
        "disclaimer": risk["disclaimer"],
    }


@router.post("/profile")
async def full_monitoring_profile(req: ProfileRequest) -> Dict[str, Any]:
    """Compute full elderly cognitive-emotional monitoring profile.

    Returns detailed results for each sub-domain plus composite MCI risk.
    """
    profile = compute_monitoring_profile(
        processing_speed_readings=_to_ps_readings(req.processing_speed_readings),
        emotion_range_readings=_to_er_readings(req.emotion_range_readings),
        cognitive_readings=_to_cog_readings(req.cognitive_readings),
        emotional_memory_readings=_to_em_readings(req.emotional_memory_readings),
        social_engagement_readings=_to_se_readings(req.social_engagement_readings),
    )
    return profile_to_dict(profile)


@router.get("/status")
async def elderly_monitoring_status() -> Dict[str, Any]:
    """Check elderly monitoring model availability."""
    return {
        "available": True,
        "model": "elderly_cognitive_emotional_monitoring",
        "version": "1.0.0",
        "domains": [
            "processing_speed",
            "emotional_flattening",
            "cognitive_emotional_coupling",
            "emotional_memory",
            "social_engagement",
        ],
        "disclaimer": _CLINICAL_DISCLAIMER,
    }
