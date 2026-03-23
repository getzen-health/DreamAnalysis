"""Tests for multilingual voice emotion detection with SenseVoice."""
import numpy as np
import pytest

from models.voice_multilingual import (
    CULTURAL_VOICE_PROFILES,
    SUPPORTED_LANGUAGES,
    SenseVoiceMultilingual,
    adjust_for_culture,
    detect_language,
    get_cultural_profile,
)


# ---------------------------------------------------------------------------
# detect_language (stub)
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_returns_string(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = detect_language(audio, sr=16000)
        assert isinstance(result, str)

    def test_returns_valid_language_code(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = detect_language(audio)
        assert len(result) == 2  # ISO 639-1

    def test_default_is_english(self):
        audio = np.random.randn(16000).astype(np.float32)
        assert detect_language(audio) == "en"


# ---------------------------------------------------------------------------
# get_cultural_profile
# ---------------------------------------------------------------------------

class TestGetCulturalProfile:
    def test_japanese_is_collectivist(self):
        profile = get_cultural_profile("ja")
        assert profile["group"] == "collectivist"
        assert profile["expression_intensity"] < 1.0

    def test_english_is_individualist(self):
        profile = get_cultural_profile("en")
        assert profile["group"] == "individualist"
        assert profile["expression_intensity"] == 1.0

    def test_telugu_has_profile(self):
        profile = get_cultural_profile("te")
        assert profile["group"] == "collectivist"
        assert 0 < profile["expression_intensity"] < 1.0

    def test_hindi_has_profile(self):
        profile = get_cultural_profile("hi")
        assert profile["group"] == "collectivist"

    def test_unknown_language_returns_default(self):
        profile = get_cultural_profile("xx")
        assert profile["group"] == "unknown"
        assert profile["expression_intensity"] == 0.85

    def test_case_insensitive(self):
        profile = get_cultural_profile("JA")
        assert profile["group"] == "collectivist"

    def test_handles_locale_codes(self):
        profile = get_cultural_profile("ja-JP")
        assert profile["group"] == "collectivist"

    def test_all_profiles_have_required_keys(self):
        for lang, profile in CULTURAL_VOICE_PROFILES.items():
            assert "group" in profile, f"{lang} missing 'group'"
            assert "expression_intensity" in profile, f"{lang} missing 'expression_intensity'"
            assert "description" in profile, f"{lang} missing 'description'"
            assert 0 < profile["expression_intensity"] <= 1.0, (
                f"{lang} expression_intensity out of range"
            )


# ---------------------------------------------------------------------------
# adjust_for_culture
# ---------------------------------------------------------------------------

class TestAdjustForCulture:
    def setup_method(self):
        self.base_probs = {
            "happy": 0.2,
            "sad": 0.1,
            "angry": 0.1,
            "fear": 0.1,
            "surprise": 0.1,
            "neutral": 0.4,
        }

    def test_english_no_adjustment(self):
        """English (baseline) should not modify predictions."""
        adjusted = adjust_for_culture(self.base_probs, "en")
        for emotion in self.base_probs:
            assert adjusted[emotion] == pytest.approx(self.base_probs[emotion], abs=1e-6)

    def test_japanese_reduces_neutral(self):
        """Collectivist cultures should have lower neutral after adjustment."""
        adjusted = adjust_for_culture(self.base_probs, "ja")
        # Neutral should be relatively lower compared to English
        en_adjusted = adjust_for_culture(self.base_probs, "en")
        assert adjusted["neutral"] < en_adjusted["neutral"]

    def test_probabilities_sum_to_one(self):
        for lang in ["ja", "zh", "ko", "hi", "te", "en", "ar"]:
            adjusted = adjust_for_culture(self.base_probs, lang)
            total = sum(adjusted.values())
            assert total == pytest.approx(1.0, abs=1e-6), (
                f"Probabilities for {lang} sum to {total}, not 1.0"
            )

    def test_non_neutral_boosted_for_collectivist(self):
        """Non-neutral emotions should be boosted for suppressive cultures."""
        adjusted = adjust_for_culture(self.base_probs, "ja")
        # Happy should have a relatively larger share vs neutral
        original_ratio = self.base_probs["happy"] / self.base_probs["neutral"]
        adjusted_ratio = adjusted["happy"] / adjusted["neutral"]
        assert adjusted_ratio > original_ratio

    def test_empty_predictions(self):
        adjusted = adjust_for_culture({}, "ja")
        assert adjusted == {}

    def test_all_zero_predictions(self):
        zeros = {e: 0.0 for e in self.base_probs}
        adjusted = adjust_for_culture(zeros, "ja")
        # All zeros should remain all zeros (or handle gracefully)
        total = sum(adjusted.values())
        assert total == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# SenseVoiceMultilingual
# ---------------------------------------------------------------------------

class TestSenseVoiceMultilingual:
    def test_initializes_without_crash(self):
        model = SenseVoiceMultilingual()
        assert hasattr(model, "available")
        assert isinstance(model.available, bool)

    def test_unavailable_predict_returns_none(self):
        model = SenseVoiceMultilingual()
        if not model.available:
            audio = np.random.randn(16000).astype(np.float32)
            result = model.predict(audio)
            assert result is None

    def test_short_audio_returns_none(self):
        model = SenseVoiceMultilingual()
        if model.available:
            short_audio = np.random.randn(100).astype(np.float32)
            result = model.predict(short_audio)
            assert result is None

    def test_get_supported_languages(self):
        model = SenseVoiceMultilingual()
        langs = model.get_supported_languages()
        assert isinstance(langs, list)
        assert "en" in langs
        assert "ja" in langs
        assert "zh" in langs
        assert "te" in langs

    def test_get_cultural_context(self):
        model = SenseVoiceMultilingual()
        ctx = model.get_cultural_context("ja")
        assert ctx["language"] == "ja"
        assert ctx["culture_group"] == "collectivist"
        assert 0 < ctx["expression_intensity"] < 1.0
        assert isinstance(ctx["description"], str)

    def test_get_cultural_context_unknown_language(self):
        model = SenseVoiceMultilingual()
        ctx = model.get_cultural_context("xx")
        assert ctx["culture_group"] == "unknown"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_supported_languages_not_empty(self):
        assert len(SUPPORTED_LANGUAGES) > 10

    def test_cultural_profiles_not_empty(self):
        assert len(CULTURAL_VOICE_PROFILES) >= 5

    def test_all_profile_languages_in_supported(self):
        for lang in CULTURAL_VOICE_PROFILES:
            assert lang in SUPPORTED_LANGUAGES, (
                f"Profile language '{lang}' not in SUPPORTED_LANGUAGES"
            )
