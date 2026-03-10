"""Nutrition-mood recommendation API.

POST /nutrition/recommend      — get food recommendations for current emotional state
GET  /nutrition/knowledge-base — return full evidence-based nutrient database
GET  /nutrition/status         — endpoint health check
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["Nutrition"])


class EmotionState(BaseModel):
    valence: float = Field(0.0, ge=-1.0, le=1.0, description="Emotional valence (-1=negative, +1=positive)")
    arousal: float = Field(0.5, ge=0.0, le=1.0, description="Arousal level (0=calm, 1=energetic)")
    stress_index: float = Field(0.5, ge=0.0, le=1.0)
    focus_index: float = Field(0.5, ge=0.0, le=1.0)
    relaxation_index: float = Field(0.5, ge=0.0, le=1.0)
    top_k: int = Field(4, ge=1, le=10, description="Number of recommendations to return")


@router.post("/nutrition/recommend")
def get_recommendations(state: EmotionState) -> dict:
    """Return evidence-based food recommendations for the current emotional state.

    Recommendations are ranked by relevance to the detected mood dimensions
    (low valence, high stress, low focus, etc.) and evidence strength.

    Each recommendation includes:
    - nutrient and food sources
    - biological mechanism
    - evidence level (meta-analysis / RCT / observational)
    - effect size where published
    - optimal timing
    - cautions

    Sources include SMILES trial (Jacka 2017), Liao omega-3 meta-analysis (2019),
    Tarleton magnesium RCT (2017), and Giesbrecht L-theanine RCT (2010).
    """
    try:
        from models.nutrition_recommender import get_recommender

        return get_recommender().recommend(
            {
                "valence": state.valence,
                "arousal": state.arousal,
                "stress_index": state.stress_index,
                "focus_index": state.focus_index,
                "relaxation_index": state.relaxation_index,
            },
            top_k=state.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/nutrition/knowledge-base")
def knowledge_base() -> dict:
    """Return the full evidence-based nutrient-mood knowledge base.

    Contains 10 validated nutrient-mood interactions with mechanisms,
    evidence levels, effect sizes, food sources, and timing guidance.
    """
    try:
        from models.nutrition_recommender import get_recommender

        return {
            "nutrients": get_recommender().get_knowledge_base(),
            "total": 10,
            "note": "All recommendations are evidence-based — see effect_size and evidence_level fields. This is not medical advice.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/nutrition/status")
def status() -> dict:
    return {
        "status": "ready",
        "nutrients_in_db": 10,
        "evidence_sources": [
            "SMILES trial (Jacka 2017)",
            "Omega-3 meta-analysis (Liao 2019)",
            "Magnesium RCT (Tarleton 2017)",
            "L-theanine+caffeine RCT (Giesbrecht 2010)",
            "Polyphenols RCT (Pase 2015)",
            "Probiotic meta-analysis (Liang 2019)",
        ],
    }
