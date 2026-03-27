"""Tests for few-shot personalization (processing module)."""
import numpy as np
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from processing.few_shot_personalization import (
    EMOTIONS,
    FewShotPersonalizer,
    LabeledSample,
    PersonalModel,
)


@pytest.fixture
def personalizer():
    return FewShotPersonalizer(user_id="test_user", blend_threshold=5)


def _make_features(emotion_idx: int, dim: int = 20, noise: float = 0.1) -> np.ndarray:
    """Create distinct feature vectors per emotion class."""
    np.random.seed(emotion_idx * 100)
    base = np.zeros(dim)
    base[emotion_idx * 3 : emotion_idx * 3 + 3] = 1.0
    return base + noise * np.random.randn(dim)


class TestAddSample:
    def test_add_sample_updates_prototype(self, personalizer):
        features = _make_features(0)
        personalizer.add_sample(features, "happy", "2026-01-01T00:00:00")
        assert "happy" in personalizer.model.prototypes
        assert personalizer.model.prototype_counts["happy"] == 1
        assert personalizer.model.total_samples == 1
        np.testing.assert_array_almost_equal(
            personalizer.model.prototypes["happy"], features
        )


class TestPredictPersonal:
    def test_returns_none_below_threshold(self, personalizer):
        for i in range(4):
            personalizer.add_sample(_make_features(0, noise=0.05), "happy")
        result = personalizer.predict_personal(_make_features(0))
        assert result is None

    def test_returns_probs_after_enough_samples(self, personalizer):
        for i in range(3):
            personalizer.add_sample(_make_features(0, noise=0.05), "happy")
            personalizer.add_sample(_make_features(1, noise=0.05), "sad")
        result = personalizer.predict_personal(_make_features(0, noise=0.05))
        assert result is not None
        assert isinstance(result, dict)
        assert set(result.keys()) == set(EMOTIONS)
        # Happy prototype should be closest to a happy-like feature
        assert result["happy"] > result["sad"]


class TestBlend:
    def test_returns_global_when_no_personal_data(self, personalizer):
        global_probs = {e: 1 / 6 for e in EMOTIONS}
        global_probs["happy"] = 0.5
        global_probs["sad"] = 0.1
        result = personalizer.blend(global_probs, _make_features(0))
        assert result == global_probs

    def test_shifts_toward_personal_with_more_samples(self, personalizer):
        global_probs = {e: 1 / 6 for e in EMOTIONS}
        features = _make_features(0, noise=0.01)

        # Add enough samples to activate personalization
        for _ in range(10):
            personalizer.add_sample(_make_features(0, noise=0.05), "happy")
            personalizer.add_sample(_make_features(1, noise=0.05), "sad")

        result = personalizer.blend(global_probs, features)
        # Personal model should push happy higher than uniform 1/6
        assert result["happy"] > 1 / 6


class TestConfidence:
    def test_confidence_grows_with_samples(self, personalizer):
        assert personalizer.model.confidence() == 0.0
        for i in range(10):
            personalizer.add_sample(_make_features(0, noise=0.1), "happy")
        conf_10 = personalizer.model.confidence()
        assert conf_10 > 0.0
        for i in range(20):
            personalizer.add_sample(_make_features(1, noise=0.1), "sad")
        conf_30 = personalizer.model.confidence()
        assert conf_30 > conf_10

    def test_confidence_caps_at_0_8(self):
        model = PersonalModel(
            user_id="cap_test", prototypes={}, prototype_counts={}, total_samples=1000
        )
        assert model.confidence() == 0.8


class TestNormalize:
    def test_zscore_against_personal_baseline(self, personalizer):
        # Add samples to establish baseline
        for _ in range(5):
            personalizer.add_sample(np.ones(10) * 2.0, "happy")
            personalizer.add_sample(np.ones(10) * 4.0, "sad")

        # Mean should be ~3.0, std should be ~1.0
        test_features = np.ones(10) * 3.0
        normalized = personalizer.normalize(test_features)
        # z-score of the mean should be near zero
        np.testing.assert_array_almost_equal(normalized, np.zeros(10), decimal=1)

    def test_no_normalization_without_samples(self, personalizer):
        features = np.array([1.0, 2.0, 3.0])
        result = personalizer.normalize(features)
        np.testing.assert_array_equal(result, features)


class TestMultipleClasses:
    def test_separate_prototypes_per_class(self, personalizer):
        personalizer.add_sample(_make_features(0), "happy")
        personalizer.add_sample(_make_features(1), "sad")
        personalizer.add_sample(_make_features(2), "angry")

        assert len(personalizer.model.prototypes) == 3
        assert "happy" in personalizer.model.prototypes
        assert "sad" in personalizer.model.prototypes
        assert "angry" in personalizer.model.prototypes
        # Prototypes should be different
        assert not np.allclose(
            personalizer.model.prototypes["happy"],
            personalizer.model.prototypes["sad"],
        )


class TestGetStatus:
    def test_returns_correct_structure(self, personalizer):
        personalizer.add_sample(_make_features(0), "happy")
        personalizer.add_sample(_make_features(1), "sad")

        status = personalizer.get_status()
        assert status["user_id"] == "test_user"
        assert status["total_samples"] == 2
        assert set(status["classes_seen"]) == {"happy", "sad"}
        assert status["samples_per_class"]["happy"] == 1
        assert status["samples_per_class"]["sad"] == 1
        assert isinstance(status["confidence"], float)
        assert isinstance(status["is_active"], bool)
        assert status["is_active"] is False  # 2 < 5 threshold


class TestRenormalization:
    def test_probs_sum_to_one(self, personalizer):
        # Build personal model
        for _ in range(10):
            personalizer.add_sample(_make_features(0, noise=0.05), "happy")
            personalizer.add_sample(_make_features(1, noise=0.05), "sad")
            personalizer.add_sample(_make_features(2, noise=0.05), "angry")

        global_probs = {
            "happy": 0.3,
            "sad": 0.2,
            "angry": 0.15,
            "fear": 0.15,
            "surprise": 0.1,
            "neutral": 0.1,
        }
        features = _make_features(0, noise=0.05)
        blended = personalizer.blend(global_probs, features)
        total = sum(blended.values())
        assert abs(total - 1.0) < 1e-6, f"Probs sum to {total}, expected ~1.0"
