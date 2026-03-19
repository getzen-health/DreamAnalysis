"""Emotional constitution API routes.

Endpoints:
  POST /constitution/create              -- create a user's emotional constitution
  POST /constitution/article             -- add or amend an article
  POST /constitution/evaluate            -- evaluate a proposed action against the constitution
  GET  /constitution/{user_id}           -- get current constitution
  GET  /constitution/status              -- framework availability check

GitHub issue: #458
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.emotional_constitution import (
    DOMAIN_DESCRIPTIONS,
    ArticleDomain,
    ActionType,
    ComplianceVerdict,
    DEFAULT_PRIORITY,
    add_article,
    amend_article,
    check_compliance,
    compute_constitution_profile,
    create_constitution,
    evaluate_action,
    get_constitution,
    get_constitution_history,
    profile_to_dict,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/constitution", tags=["emotional-constitution"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class CreateConstitutionRequest(BaseModel):
    user_id: str = Field(..., description="User creating the constitution")
    name: Optional[str] = Field(None, description="Optional custom name")
    preamble: Optional[str] = Field(None, description="Optional preamble text")


class AddArticleRequest(BaseModel):
    user_id: str = Field(..., description="Owner of the constitution")
    domain: str = Field(
        ...,
        description=f"Article domain. One of: {sorted(d.value for d in ArticleDomain)}",
    )
    title: str = Field(..., description="Short human-readable title for the article")
    rule: str = Field(..., description="Plain-language description of the rule")
    action_types: Optional[List[str]] = Field(
        None,
        description=(
            "Which action types this article governs. "
            f"Valid types: {sorted(a.value for a in ActionType)}. "
            "If omitted, applies to all action types."
        ),
    )
    conditions: Optional[Dict[str, Any]] = Field(
        None,
        description="Key-value conditions that must be met for the rule to trigger",
    )
    effect: str = Field(
        "deny",
        description="Effect when rule matches: 'deny', 'allow', or 'conditional'",
    )
    priority: int = Field(
        DEFAULT_PRIORITY,
        description="Priority 0-100. Higher priority wins in conflicts.",
    )
    # Amendment fields (optional -- if article_id is set, amend instead of add)
    article_id: Optional[str] = Field(
        None,
        description="If set, amend this existing article instead of creating a new one",
    )
    reason: Optional[str] = Field(
        None,
        description="Reason for the amendment (only used when article_id is set)",
    )


class EvaluateActionRequest(BaseModel):
    user_id: str = Field(..., description="User whose constitution to check")
    action_type: str = Field(
        ...,
        description=f"Action type. One of: {sorted(a.value for a in ActionType)}",
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Key-value context describing the proposed action",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/create")
async def create_user_constitution(request: CreateConstitutionRequest):
    """Create a new emotional constitution for a user.

    A constitution is the user's sovereign ruleset governing how AI may
    interact with their emotional data.  Each user may have exactly one.
    """
    try:
        constitution = create_constitution(
            request.user_id,
            name=request.name,
            preamble=request.preamble,
        )
        return {"status": "created", "constitution": constitution}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Failed to create constitution for user %s", request.user_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/article")
async def add_or_amend_article(request: AddArticleRequest):
    """Add a new article or amend an existing one.

    If article_id is provided, the existing article is amended with the
    new values.  Otherwise a new article is created.  Amendment history
    is preserved automatically.
    """
    try:
        if request.article_id:
            # Amend existing article
            article = amend_article(
                request.user_id,
                request.article_id,
                title=request.title,
                rule=request.rule,
                action_types=request.action_types,
                conditions=request.conditions,
                effect=request.effect,
                priority=request.priority,
                reason=request.reason or "",
            )
            return {"status": "amended", "article": article}
        else:
            # Add new article
            article = add_article(
                request.user_id,
                domain=request.domain,
                title=request.title,
                rule=request.rule,
                action_types=request.action_types,
                conditions=request.conditions,
                effect=request.effect,
                priority=request.priority,
            )
            return {"status": "added", "article": article}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Failed to add/amend article for user %s", request.user_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/evaluate")
async def evaluate_action_endpoint(request: EvaluateActionRequest):
    """Evaluate a proposed action against the user's constitution.

    Returns a compliance verdict (allowed, denied, conditional) with
    the deciding article and reasoning.
    """
    try:
        result = evaluate_action(
            request.user_id,
            request.action_type,
            context=request.context,
        )
        return result
    except Exception as exc:
        log.exception("Evaluation failed for user %s", request.user_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/status")
async def get_status():
    """Check emotional constitution framework availability.

    Returns framework metadata, supported domains, action types, and
    available endpoints.
    """
    return {
        "status": "available",
        "framework": "Emotional Constitution -- User Sovereignty Framework",
        "version": "1.0.0",
        "domains": {d.value: DOMAIN_DESCRIPTIONS[d] for d in ArticleDomain},
        "action_types": sorted(a.value for a in ActionType),
        "verdict_types": sorted(v.value for v in ComplianceVerdict),
        "endpoints": [
            "POST /constitution/create",
            "POST /constitution/article",
            "POST /constitution/evaluate",
            "GET /constitution/{user_id}",
            "GET /constitution/status",
        ],
    }


@router.get("/{user_id}")
async def get_user_constitution(user_id: str):
    """Get the current constitution for a user.

    Returns the full constitution with all articles, plus the
    analytical profile and amendment history.
    """
    constitution = get_constitution(user_id)
    if constitution is None:
        raise HTTPException(
            status_code=404,
            detail=f"No constitution found for user {user_id}",
        )

    profile = compute_constitution_profile(user_id)
    history = get_constitution_history(user_id)

    return {
        "constitution": constitution,
        "profile": profile_to_dict(profile),
        "amendment_history": history,
    }
