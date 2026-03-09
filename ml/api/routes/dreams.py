"""Dream NLP endpoints: /classify-dream-theme and /dream-themes.

Inspired by DreamNet (arXiv 2503.05778, 2025) which achieves 92.1% accuracy
on dream theme classification using TF-IDF + LightGBM on the DreamBank corpus.
"""

from fastapi import APIRouter, Body, HTTPException

router = APIRouter()

# Lazy singleton — created on first request, reused thereafter
_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        from models.dream_theme_classifier import DreamThemeClassifier
        _classifier = DreamThemeClassifier()
    return _classifier


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
