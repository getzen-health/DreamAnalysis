"""Food image analysis endpoint.

POST /food/analyze-image
    Accept a base64-encoded food image (or a text description) and return
    identified food items with estimated portion sizes and nutritional totals.

    Strategy (in priority order):
      1. Anthropic Claude vision  — if ANTHROPIC_API_KEY is set
      2. Nutritional DB + text description — always available as fallback
         (client must provide textDescription when no vision model is available)

Responses always include:
  - food_items     list of {name, portion, calories, protein_g, carbs_g, fat_g, fiber_g}
  - total_calories
  - total_protein_g / total_carbs_g / total_fat_g / total_fiber_g
  - dominant_macro ("protein" | "carbs" | "fat")
  - glycemic_impact ("low" | "medium" | "high")
  - confidence     0-1 float
  - analysis_method "vision_ai" | "description_lookup" | "description_ai" | "partial"
  - error          present only when analysis is incomplete
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Food Image"])

# ── Pydantic models ───────────────────────────────────────────────────────────


class FoodImageRequest(BaseModel):
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded JPEG/PNG image of the meal. Required for vision analysis.",
    )
    text_description: Optional[str] = Field(
        default=None,
        description="Text description of the meal — used when no image is provided "
        "or as a hint to improve image analysis.",
    )
    meal_type: str = Field(
        default="meal",
        description="breakfast | lunch | dinner | snack | meal",
    )


class FoodItem(BaseModel):
    name: str
    portion: str
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float


class FoodImageResponse(BaseModel):
    food_items: List[FoodItem]
    total_calories: float
    total_protein_g: float
    total_carbs_g: float
    total_fat_g: float
    total_fiber_g: float
    dominant_macro: str
    glycemic_impact: str
    confidence: float
    analysis_method: str
    summary: str
    error: Optional[str] = None


# ── Glycemic impact heuristic ─────────────────────────────────────────────────

_HIGH_GI_KEYWORDS = {
    "white bread", "white rice", "soda", "candy", "sugar", "donut", "waffle",
    "pancake", "bagel", "croissant", "corn flakes", "french fries", "potato chips",
    "cake", "cookie", "ice cream", "gummy", "muffin", "pretzel",
}
_LOW_GI_KEYWORDS = {
    "oats", "oatmeal", "quinoa", "brown rice", "lentil", "chickpea", "bean",
    "broccoli", "spinach", "kale", "vegetable", "avocado", "nuts", "almonds",
    "whole grain", "whole wheat", "sweet potato", "salmon", "chicken",
    "greek yogurt", "egg", "protein", "fiber",
}


def _glycemic_impact(food_names: List[str]) -> str:
    names_str = " ".join(n.lower() for n in food_names)
    high_hits = sum(1 for k in _HIGH_GI_KEYWORDS if k in names_str)
    low_hits = sum(1 for k in _LOW_GI_KEYWORDS if k in names_str)
    if high_hits > low_hits:
        return "high"
    if low_hits > high_hits:
        return "low"
    return "medium"


def _dominant_macro(protein: float, carbs: float, fat: float) -> str:
    # fat kcal = 9 kcal/g; protein & carbs = 4 kcal/g
    protein_cal = protein * 4
    carbs_cal = carbs * 4
    fat_cal = fat * 9
    total = protein_cal + carbs_cal + fat_cal or 1.0
    if abs(protein_cal / total - carbs_cal / total) < 0.05 and abs(protein_cal / total - fat_cal / total) < 0.05:
        return "balanced"
    best = max([("protein", protein_cal), ("carbs", carbs_cal), ("fat", fat_cal)], key=lambda x: x[1])
    return best[0]


# ── DB-based description parser ───────────────────────────────────────────────

def _parse_description_with_db(text_description: str) -> List[Dict[str, Any]]:
    """Parse a free-text meal description against the local nutrition DB.

    Splits on commas/semicolons, looks up each token, and estimates a
    default 150 g serving when no weight is mentioned.
    """
    from models.nutrition_db import lookup_with_portion, portion_size_to_grams

    items: List[Dict[str, Any]] = []
    # Split on common separators
    tokens = re.split(r"[,;]+|\band\b", text_description.lower())

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        # Try to extract a number + unit at the start: "2 scrambled eggs", "1 cup rice"
        count_match = re.match(r"^(\d+\.?\d*)\s+(.+)$", token)
        if count_match:
            count_str = count_match.group(1)
            remainder = count_match.group(2).strip()
            # Check if remainder starts with a unit word
            unit_match = re.match(r"^(cup|tablespoon|teaspoon|oz|ounce|g|gram|slice|piece)s?\s+(.+)$", remainder)
            if unit_match:
                portion_label = f"{count_str} {unit_match.group(1)}"
                food_part = unit_match.group(2).strip()
                grams = portion_size_to_grams(portion_label)
            else:
                # "2 eggs" → count * default portion
                food_part = remainder
                grams = float(count_str) * 80.0  # rough 80 g per unit
        else:
            food_part = token
            grams = 120.0  # medium serving default

        entry = lookup_with_portion(food_part, grams)
        if entry:
            items.append({
                "name": entry["name"],
                "portion": f"{grams:.0f}g",
                "calories": entry["calories"],
                "protein_g": entry["protein_g"],
                "carbs_g": entry["carbs_g"],
                "fat_g": entry["fat_g"],
                "fiber_g": entry["fiber_g"],
            })

    return items


# ── Anthropic vision analysis ─────────────────────────────────────────────────

_VISION_PROMPT = """You are a professional dietitian analyzing a food photo.
Identify all visible food items, estimate realistic portion sizes, and calculate
macronutrients.

Return ONLY valid JSON (no markdown fences, no prose) with this exact structure:
{
  "food_items": [
    {
      "name": "exact food name",
      "portion": "human-readable portion size, e.g. '1 medium (150g)'",
      "calories": <number>,
      "protein_g": <number>,
      "carbs_g": <number>,
      "fat_g": <number>,
      "fiber_g": <number>
    }
  ],
  "summary": "One plain English sentence describing the meal",
  "confidence": <0.0 to 1.0>
}

Rules:
- Only include items you can clearly see in the photo.
- If the user's description contradicts what you see, trust the photo.
- All numbers must be plausible (no 5000 cal breakfast).
- fiber_g is mandatory (estimate 0 if unknown).
- Do NOT add a "total_calories" field — the caller will sum them.
"""


async def _analyze_with_anthropic(
    image_base64: str,
    text_hint: Optional[str],
) -> Dict[str, Any]:
    """Call Claude vision API to analyze a food image."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    content: List[Dict] = []

    # Determine image media type from base64 header or assume JPEG
    media_type = "image/jpeg"
    if image_base64.startswith("data:"):
        header, image_base64 = image_base64.split(",", 1)
        if "png" in header:
            media_type = "image/png"
        elif "webp" in header:
            media_type = "image/webp"
        elif "gif" in header:
            media_type = "image/gif"

    content.append({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": image_base64,
        },
    })

    prompt = _VISION_PROMPT
    if text_hint:
        prompt += f"\n\nUser's description: {text_hint}"
    content.append({"type": "text", "text": prompt})

    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Claude returned invalid JSON: {exc}  raw={raw[:300]}")


# ── Main endpoint ─────────────────────────────────────────────────────────────


@router.post("/food/analyze-image", response_model=FoodImageResponse)
async def analyze_food_image(req: FoodImageRequest) -> FoodImageResponse:
    """Analyze a food image and return identified items with nutritional data.

    Pass ``image_base64`` for vision-based analysis (requires ANTHROPIC_API_KEY).
    Pass ``text_description`` alone for DB-based lookup (always available).
    Pass both for best results when the image is blurry or the meal is complex.

    The ``analysis_method`` field in the response indicates which path was used:
      - ``vision_ai``          — Claude vision model analyzed the image
      - ``description_ai``     — AI analyzed the text description (fallback)
      - ``description_lookup`` — local nutrition DB (no AI required)
      - ``partial``            — partial result; see ``error`` field
    """
    if not req.image_base64 and not req.text_description:
        raise HTTPException(
            status_code=422,
            detail="Provide image_base64 or text_description (or both).",
        )

    food_items_raw: List[Dict[str, Any]] = []
    confidence: float = 0.0
    method: str = "partial"
    summary: str = ""
    error_msg: Optional[str] = None

    # ── Path 1: Claude vision ──────────────────────────────────────────────────
    if req.image_base64 and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            vision_result = await _analyze_with_anthropic(req.image_base64, req.text_description)
            food_items_raw = vision_result.get("food_items", [])
            confidence = float(vision_result.get("confidence", 0.75))
            summary = vision_result.get("summary", "")
            method = "vision_ai"
            logger.info("Food image analysis via Claude vision: %d items found", len(food_items_raw))
        except Exception as exc:
            error_msg = f"Vision analysis failed: {exc}"
            logger.warning("Vision analysis failed — falling back to text: %s", exc)

    # ── Path 2: Text description ───────────────────────────────────────────────
    if not food_items_raw and req.text_description:
        try:
            food_items_raw = _parse_description_with_db(req.text_description)
            confidence = 0.55
            method = "description_lookup"
            summary = f"Meal based on your description: {req.text_description[:80]}"
            logger.info("Food analysis via DB lookup: %d items found", len(food_items_raw))
        except Exception as exc:
            error_msg = (error_msg or "") + f" DB lookup failed: {exc}"
            logger.warning("DB lookup failed: %s", exc)

    if not food_items_raw:
        raise HTTPException(
            status_code=422,
            detail=error_msg or "Could not identify any food items. Provide a clearer image or a text description.",
        )

    # ── Validate and build response ────────────────────────────────────────────
    items: List[FoodItem] = []
    for raw in food_items_raw:
        try:
            items.append(FoodItem(
                name=str(raw.get("name", "Unknown food")),
                portion=str(raw.get("portion", "1 serving")),
                calories=float(raw.get("calories", 0)),
                protein_g=float(raw.get("protein_g", 0)),
                carbs_g=float(raw.get("carbs_g", 0)),
                fat_g=float(raw.get("fat_g", 0)),
                fiber_g=float(raw.get("fiber_g", 0)),
            ))
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping malformed food item %s: %s", raw, exc)

    if not items:
        raise HTTPException(
            status_code=422,
            detail="Food items were identified but could not be parsed. Please try again.",
        )

    total_cal = round(sum(i.calories for i in items), 1)
    total_prot = round(sum(i.protein_g for i in items), 1)
    total_carbs = round(sum(i.carbs_g for i in items), 1)
    total_fat = round(sum(i.fat_g for i in items), 1)
    total_fiber = round(sum(i.fiber_g for i in items), 1)

    dom_macro = _dominant_macro(total_prot, total_carbs, total_fat)
    gi = _glycemic_impact([i.name for i in items])

    if not summary:
        names = ", ".join(i.name for i in items[:3])
        suffix = f" and {len(items) - 3} more item(s)" if len(items) > 3 else ""
        summary = f"{req.meal_type.capitalize()}: {names}{suffix} — {total_cal:.0f} kcal estimated."

    return FoodImageResponse(
        food_items=items,
        total_calories=total_cal,
        total_protein_g=total_prot,
        total_carbs_g=total_carbs,
        total_fat_g=total_fat,
        total_fiber_g=total_fiber,
        dominant_macro=dom_macro,
        glycemic_impact=gi,
        confidence=confidence,
        analysis_method=method,
        summary=summary,
        error=error_msg,
    )


@router.get("/food/analyze-image/status")
async def food_image_status() -> Dict[str, Any]:
    """Report which analysis methods are available in this deployment."""
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return {
        "vision_ai_available": has_anthropic,
        "vision_provider": "anthropic_claude_haiku" if has_anthropic else None,
        "db_lookup_available": True,
        "food_count_in_db": 200,
        "note": (
            "Vision analysis enabled — send image_base64 for best results."
            if has_anthropic
            else "No vision API key set (ANTHROPIC_API_KEY). "
            "Text description lookup is always available."
        ),
    }
