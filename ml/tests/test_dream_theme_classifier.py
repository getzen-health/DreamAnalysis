"""Tests for DreamNet-inspired dream theme classifier."""
import pytest
from models.dream_theme_classifier import DreamThemeClassifier, DREAM_THEMES


@pytest.fixture
def clf():
    return DreamThemeClassifier()


def test_flying_dream(clf):
    text = "I was flying over the mountains, soaring through the clouds"
    result = clf.classify(text)
    assert result["primary_theme"] == "flying"
    assert result["theme_scores"]["flying"] > 0
    assert "flying" in result["keywords_found"]


def test_falling_dream(clf):
    text = "I fell off a cliff and was falling into darkness, terrified"
    result = clf.classify(text)
    assert result["primary_theme"] == "falling"
    assert result["correlated_states"].get("anxiety", 0) > 0


def test_pursuit_dream(clf):
    text = "Someone was chasing me through the forest and I had to run and escape"
    result = clf.classify(text)
    assert result["primary_theme"] == "pursuit"


def test_water_dream(clf):
    text = "I was swimming in the ocean and waves were flooding everything"
    result = clf.classify(text)
    assert result["primary_theme"] == "water"


def test_empty_text_returns_neutral(clf):
    result = clf.classify("")
    assert result["primary_theme"] == "neutral"
    assert result["confidence"] == 0.0


def test_all_themes_in_scores(clf):
    result = clf.classify("I had a strange dream about flying over water")
    for theme in DREAM_THEMES:
        assert theme in result["theme_scores"]


def test_scores_sum_to_one(clf):
    result = clf.classify("I was chasing someone through a school and fell down the stairs")
    total = sum(result["theme_scores"].values())
    assert abs(total - 1.0) < 0.01


def test_word_count_reported(clf):
    text = "short dream"
    result = clf.classify(text)
    assert result["word_count"] == 2


def test_confidence_in_range(clf):
    result = clf.classify("I had a very vivid dream about flying and exploring a new hidden room")
    assert 0.0 <= result["confidence"] <= 1.0
