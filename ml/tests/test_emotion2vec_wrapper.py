"""Tests for emotion2vec wrapper — graceful degradation if not installed."""
import numpy as np
import pytest

from models.emotion2vec_wrapper import (
    E2V_LABELS,
    E2V_TO_6CLASS,
    EMOTIONS_6,
    Emotion2vecWrapper,
)


@pytest.fixture
def wrapper():
    return Emotion2vecWrapper()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_initializes_without_crash(self, wrapper):
        assert hasattr(wrapper, "available")
        assert hasattr(wrapper, "predict")

    def test_available_is_bool(self, wrapper):
        assert isinstance(wrapper.available, bool)

    def test_lazy_loading(self):
        """Model should not load until available/predict is called."""
        w = Emotion2vecWrapper()
        assert w._load_attempted is False
        _ = w.available  # triggers load
        assert w._load_attempted is True


# ---------------------------------------------------------------------------
# predict — unavailable model
# ---------------------------------------------------------------------------

class TestPredictUnavailable:
    def test_returns_none_when_unavailable(self, wrapper):
        if not wrapper.available:
            audio = np.random.randn(16000).astype(np.float32)
            result = wrapper.predict(audio)
            assert result is None

    def test_short_audio_returns_none(self, wrapper):
        short = np.random.randn(100).astype(np.float32)
        result = wrapper.predict(short)
        assert result is None

    def test_none_audio_returns_none(self, wrapper):
        result = wrapper.predict(None)
        assert result is None


# ---------------------------------------------------------------------------
# 9-class to 6-class mapping
# ---------------------------------------------------------------------------

class TestLabelMapping:
    def test_all_9class_labels_mapped(self):
        for label in E2V_LABELS:
            mapped = Emotion2vecWrapper.map_9class_to_6class(label)
            assert mapped in EMOTIONS_6, f"'{label}' mapped to '{mapped}' not in 6-class"

    def test_angry_maps_to_angry(self):
        assert Emotion2vecWrapper.map_9class_to_6class("angry") == "angry"

    def test_disgusted_merges_to_angry(self):
        assert Emotion2vecWrapper.map_9class_to_6class("disgusted") == "angry"

    def test_fearful_maps_to_fear(self):
        assert Emotion2vecWrapper.map_9class_to_6class("fearful") == "fear"

    def test_surprised_maps_to_surprise(self):
        assert Emotion2vecWrapper.map_9class_to_6class("surprised") == "surprise"

    def test_unknown_maps_to_neutral(self):
        assert Emotion2vecWrapper.map_9class_to_6class("unknown") == "neutral"

    def test_other_maps_to_neutral(self):
        assert Emotion2vecWrapper.map_9class_to_6class("other") == "neutral"

    def test_case_insensitive(self):
        assert Emotion2vecWrapper.map_9class_to_6class("HAPPY") == "happy"
        assert Emotion2vecWrapper.map_9class_to_6class("Angry") == "angry"

    def test_nonexistent_label_defaults_to_neutral(self):
        assert Emotion2vecWrapper.map_9class_to_6class("nonexistent") == "neutral"


# ---------------------------------------------------------------------------
# _parse_result
# ---------------------------------------------------------------------------

class TestParseResult:
    def test_valid_result(self, wrapper):
        raw = [{
            "labels": ["happy", "sad", "angry", "fearful", "neutral", "other", "surprised", "disgusted", "unknown"],
            "scores": [0.5, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05, 0.05],
        }]
        result = wrapper._parse_result(raw)
        assert result is not None
        assert result["emotion"] == "happy"
        assert result["model_type"] == "voice_emotion2vec"
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
        assert -1.0 <= result["valence"] <= 1.0
        assert 0.0 <= result["arousal"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0

    def test_empty_result_returns_none(self, wrapper):
        assert wrapper._parse_result([]) is None
        assert wrapper._parse_result(None) is None

    def test_missing_scores_returns_none(self, wrapper):
        raw = [{"labels": E2V_LABELS, "scores": []}]
        assert wrapper._parse_result(raw) is None

    def test_all_emotions_present_in_output(self, wrapper):
        raw = [{
            "labels": E2V_LABELS,
            "scores": [0.1, 0.1, 0.1, 0.2, 0.2, 0.05, 0.1, 0.1, 0.05],
        }]
        result = wrapper._parse_result(raw)
        assert result is not None
        for emotion in EMOTIONS_6:
            assert emotion in result["probabilities"]


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_get_model_info(self):
        info = Emotion2vecWrapper.get_model_info()
        assert info["name"] == "emotion2vec_plus_large"
        assert info["classes"] == 9
        assert info["mapped_classes"] == 6
        assert info["size_mb"] == 350
        assert "funasr" in info["requires"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_9class_labels_complete(self):
        assert len(E2V_LABELS) == 9

    def test_6class_labels_complete(self):
        assert len(EMOTIONS_6) == 6
        expected = {"happy", "sad", "angry", "fear", "surprise", "neutral"}
        assert set(EMOTIONS_6) == expected

    def test_mapping_covers_all_9class(self):
        assert set(E2V_TO_6CLASS.keys()) == set(E2V_LABELS)
