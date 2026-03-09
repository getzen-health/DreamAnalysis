"""Tests for EmotionGranularityMapper and estimate_dominance."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.emotion_granularity import (
    EMOTION_VAD_MAP,
    EmotionGranularityMapper,
    estimate_dominance,
)


def test_map_output_keys():
    mapper = EmotionGranularityMapper()
    result = mapper.map(valence=0.5, arousal=0.7, dominance=0.6)
    assert "primary_emotion" in result
    assert "nuance_emotions" in result
    assert "narrative" in result
    assert "affect_zone" in result
    assert "top_emotions" in result
    assert "valence" in result
    assert "arousal" in result
    assert "dominance" in result


def test_primary_emotion_is_valid():
    mapper = EmotionGranularityMapper()
    result = mapper.map(valence=0.8, arousal=0.8, dominance=0.7)
    assert result["primary_emotion"] in EMOTION_VAD_MAP


def test_happy_region():
    mapper = EmotionGranularityMapper()
    result = mapper.map(valence=0.85, arousal=0.65, dominance=0.70)
    # Happy or elated should be top
    assert result["primary_emotion"] in {"happy", "elated", "excited", "enthusiastic"}


def test_sad_region():
    mapper = EmotionGranularityMapper()
    result = mapper.map(valence=-0.75, arousal=0.15, dominance=0.20)
    assert result["primary_emotion"] in {"sad", "melancholy", "hopeless", "lonely"}


def test_angry_region():
    mapper = EmotionGranularityMapper()
    result = mapper.map(valence=-0.70, arousal=0.85, dominance=0.75)
    assert result["primary_emotion"] in {"angry", "frustrated", "stressed"}


def test_nuance_count():
    mapper = EmotionGranularityMapper()
    result = mapper.map(valence=0.2, arousal=0.4, dominance=0.5, top_n=3)
    assert len(result["nuance_emotions"]) == 2  # top_n - 1


def test_narrative_contains_primary():
    mapper = EmotionGranularityMapper()
    result = mapper.map(valence=0.5, arousal=0.3, dominance=0.6)
    assert result["primary_emotion"] in result["narrative"]


def test_dominance_estimate_range():
    rng = np.random.default_rng(42)
    bands = {
        "alpha": float(rng.uniform(0.1, 0.4)),
        "beta": float(rng.uniform(0.1, 0.3)),
        "theta": float(rng.uniform(0.05, 0.2)),
        "high_beta": float(rng.uniform(0.02, 0.1)),
    }
    dom = estimate_dominance(bands)
    assert 0.0 <= dom <= 1.0


def test_map_from_basic():
    mapper = EmotionGranularityMapper()
    result = mapper.map_from_basic("happy", valence=0.7, arousal=0.6)
    assert "primary_emotion" in result
    assert result["primary_emotion"] in EMOTION_VAD_MAP


def test_top_emotions_sum_to_one():
    mapper = EmotionGranularityMapper()
    result = mapper.map(valence=0.0, arousal=0.5, dominance=0.5, top_n=3)
    total = sum(result["top_emotions"].values())
    assert abs(total - 1.0) < 0.01
