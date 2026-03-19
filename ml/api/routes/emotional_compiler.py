"""Emotional Compiler API routes — cross-framework therapy translation.

Endpoints:
  POST /compiler/translate     — translate a concept between two frameworks
  POST /compiler/compile       — full cross-framework compilation
  GET  /compiler/frameworks    — list all supported frameworks
  GET  /compiler/status        — health check

GitHub issue: #454
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.emotional_compiler import (
    FRAMEWORKS,
    FRAMEWORK_DESCRIPTIONS,
    FRAMEWORK_VOCABULARY,
    compile_across_frameworks,
    get_framework_vocabulary,
    suggest_intervention_per_framework,
    translate_emotion,
)

router = APIRouter(tags=["Emotional Compiler"])


# ── Request models ───────────────────────────────────────────────────────────


class TranslateRequest(BaseModel):
    source_framework: str = Field(
        ..., description="Originating framework (CBT, DBT, ACT, Somatic, IFS)"
    )
    concept: str = Field(
        ..., description="Concept key within the source framework (e.g. 'catastrophizing')"
    )
    target_framework: str = Field(
        ..., description="Destination framework"
    )


class CompileRequest(BaseModel):
    source_framework: str = Field(
        ..., description="Originating framework"
    )
    concept: str = Field(
        ..., description="Concept key within the source framework"
    )


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/compiler/translate")
async def translate(body: TranslateRequest):
    """Translate a concept from one therapeutic framework to another.

    Example: CBT "catastrophizing" translated to ACT yields "cognitive_fusion".
    Returns source/target descriptions and the target framework's suggested
    intervention.
    """
    return translate_emotion(
        source_framework=body.source_framework,
        concept=body.concept,
        target_framework=body.target_framework,
    )


@router.post("/compiler/compile")
async def compile(body: CompileRequest):
    """Full cross-framework compilation of a single concept.

    Returns the equivalent concept, description, and intervention for all
    five frameworks simultaneously. Also includes per-framework intervention
    suggestions.
    """
    compilation = compile_across_frameworks(
        source_framework=body.source_framework,
        concept=body.concept,
    )
    if "error" in compilation:
        return compilation

    interventions = suggest_intervention_per_framework(
        source_framework=body.source_framework,
        concept=body.concept,
    )

    compilation["intervention_suggestions"] = interventions.get("interventions", {})
    return compilation


@router.get("/compiler/frameworks")
async def list_frameworks():
    """List all supported therapeutic frameworks with descriptions and vocabulary sizes."""
    frameworks = []
    for fw in FRAMEWORKS:
        vocab = FRAMEWORK_VOCABULARY.get(fw, {})
        frameworks.append({
            "name": fw,
            "description": FRAMEWORK_DESCRIPTIONS.get(fw, ""),
            "concept_count": len(vocab),
            "concepts": list(vocab.keys()),
        })
    return {
        "frameworks": frameworks,
        "total_frameworks": len(FRAMEWORKS),
    }


@router.get("/compiler/status")
async def status():
    """Health check for the emotional compiler module."""
    total_concepts = sum(len(v) for v in FRAMEWORK_VOCABULARY.values())
    return {
        "status": "ok",
        "frameworks_loaded": len(FRAMEWORKS),
        "total_concepts": total_concepts,
        "model_type": "rule-based",
        "llm_required": False,
    }
