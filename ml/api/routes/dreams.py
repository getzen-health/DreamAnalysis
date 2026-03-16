"""Dream NLP endpoints.

Endpoints:
  POST /classify-dream-theme      — DreamNet-inspired theme classification
  GET  /dream-themes              — list all supported themes
  POST /analyze-dream-narrative   — full narrative analysis (emotion, archetypes, IRT)
  POST /analyze-dream             — unified DreamNLP analysis (themes + sentiment + symbols + lucidity)
  POST /analyze-dream-corpus      — recurring pattern analysis across multiple entries
  GET  /dream-symbols             — list all supported dream symbols

References:
  - DreamNet 2025 (arXiv 2503.05778) — 92.1% dream text classification
  - IRT meta-analysis — d=0.85-1.24 for nightmare reduction
  - Northwestern TLR 2024 — audio cues triple lucid dream frequency
  - Hall & Van de Castle (1966) — Content Analysis of Dreams
"""

from fastapi import APIRouter, Body, HTTPException

router = APIRouter()

# Lazy singletons
_classifier = None
_analyzer = None
_dream_nlp = None


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


def _get_dream_nlp():
    global _dream_nlp
    if _dream_nlp is None:
        from models.dream_nlp import get_dream_nlp
        _dream_nlp = get_dream_nlp()
    return _dream_nlp


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


@router.post("/analyze-dream")
async def analyze_dream(payload: dict = Body(...)):
    """Unified DreamNLP analysis — themes, sentiment, symbols, and lucidity.

    Implements issue #48 (DreamNet NLP). Uses the DreamNLP class which
    applies TF-IDF keyword scoring, regex symbol extraction, and lexicon-based
    affect analysis with zero transformer dependencies.

    Request body:
      { "text": str }

    Returns:
      themes          — top detected theme names (ordered by score)
      theme_scores    — {theme: score 0-1} for all 15 themes
      primary_theme   — highest-scoring theme
      sentiment       — {valence: float, label: str, tone: str, pos/neg counts}
      symbols         — list of {name, meaning, valence, category, count, matched_text}
      lucidity        — {score: 0-1, label: str, markers_found: list}
      word_count      — int
      emotional_words — list of {word, polarity}
      insights        — up to 5 actionable observations
    """
    text = payload.get("text", "")
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="text field required")

    result = _get_dream_nlp().analyze(text)
    return {"status": "ok", "analysis": result}


@router.post("/analyze-dream-corpus")
async def analyze_dream_corpus(payload: dict = Body(...)):
    """Analyze multiple dream journal entries for recurring patterns.

    Useful for longitudinal dream journal dashboards — finds symbols and themes
    that appear repeatedly across sessions.

    Request body:
      { "entries": list[str] }   — list of dream journal texts

    Returns:
      total_entries      — int
      recurring_symbols  — symbols found in 2+ entries [{symbol, entry_count, frequency}]
      dominant_themes    — top themes across all entries [{theme, count, frequency}]
      sentiment_trend    — per-entry valence values (for time-series chart)
      mean_valence       — float -1..1
      lucidity_rate      — fraction of entries with lucidity markers (0-1)
      symbol_frequency   — {symbol: entry_count} top 10
      theme_frequency    — {theme: entry_count}
    """
    entries = payload.get("entries", [])
    if not entries or not isinstance(entries, list):
        raise HTTPException(status_code=422, detail="entries list required")
    if len(entries) > 365:
        raise HTTPException(status_code=422, detail="maximum 365 entries per request")

    result = _get_dream_nlp().analyze_corpus(entries)
    return {"status": "ok", "analysis": result}


@router.get("/dream-symbols")
async def list_dream_symbols():
    """List all supported dream symbols with meanings and categories.

    Returns the full symbol lexicon used by DreamNLP for symbol extraction.
    Symbols are grouped by category: motion, people, place, object, nature, emotional.
    """
    from models.dream_nlp import DREAM_SYMBOLS
    by_category: dict = {}
    for name, info in DREAM_SYMBOLS.items():
        cat = info["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append({
            "name": name,
            "meaning": info["meaning"],
            "valence": info["valence"],
        })
    return {
        "symbols": {name: {"meaning": v["meaning"], "valence": v["valence"], "category": v["category"]}
                    for name, v in DREAM_SYMBOLS.items()},
        "by_category": by_category,
        "total": len(DREAM_SYMBOLS),
    }
