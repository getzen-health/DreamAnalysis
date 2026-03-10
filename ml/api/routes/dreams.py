"""Dream NLP endpoints.

Endpoints:
  POST /classify-dream-theme      — DreamNet-inspired theme classification
  GET  /dream-themes              — list all supported themes
  POST /analyze-dream-narrative   — full narrative analysis (emotion, archetypes, IRT)

References:
  - DreamNet 2025 (arXiv 2503.05778) — 92.1% dream text classification
  - IRT meta-analysis — d=0.85-1.24 for nightmare reduction
  - Northwestern TLR 2024 — audio cues triple lucid dream frequency
"""

from fastapi import APIRouter, Body, HTTPException

router = APIRouter()

# Lazy singletons
_classifier = None
_analyzer = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        from models.dream_theme_classifier import DreamThemeClassifier
        _classifier = DreamThemeClassifier()
    return _classifier


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from models.dream_analyzer import get_dream_analyzer
        _analyzer = get_dream_analyzer()
    return _analyzer


@router.post("/classify-dream-theme")
async def classify_dream_theme(payload: dict = Body(...)):
    """Classify dream journal text into thematic categories.

    Uses keyword scoring (+ optional TF-IDF/LightGBM if model available).
    Inspired by DreamNet (arXiv 2503.05778, 2025) which achieves 92.1% accuracy.

    Returns primary_theme, theme_scores, psychological correlations.
    """
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=422, detail="text field required")

    result = _get_classifier().classify(text)
    return {"status": "ok", "analysis": result}


@router.get("/dream-themes")
async def list_dream_themes():
    """List all supported dream themes with keywords and correlations."""
    from models.dream_theme_classifier import DREAM_THEMES, THEME_KEYWORDS, THEME_CORRELATIONS
    themes = {}
    for theme in DREAM_THEMES:
        themes[theme] = {
            "keywords": THEME_KEYWORDS.get(theme, []),
            "correlations": THEME_CORRELATIONS.get(theme, {}),
        }
    return {"themes": themes, "total": len(DREAM_THEMES)}


@router.post("/analyze-dream-narrative")
async def analyze_dream_narrative(payload: dict = Body(...)):
    """Full dream narrative analysis: emotion, archetypes, nightmare score, IRT.

    Implements Phase 1-2 of dream analysis (#287).
    Also combines theme classification when text is provided.

    Request body: { "text": str, "user_id": str (optional) }

    Returns:
      emotional_valence, emotional_arousal, emotional_intensity,
      nightmare_score, is_nightmare, archetypes, emotions_detected,
      lucid_probability, irt_recommended, irt_protocol,
      insights, morning_mood_prediction,
      theme_analysis (from DreamNet classifier)
    """
    text = payload.get("text", "")
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="text field required")

    result = _get_analyzer().analyze(text)

    # Augment with theme classification
    try:
        theme_result = _get_classifier().classify(text)
        result["theme_analysis"] = theme_result
    except Exception:
        result["theme_analysis"] = None

    return {"status": "ok", "analysis": result}
