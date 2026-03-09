"""Tests for few-shot personalizer."""
import numpy as np
import pytest

from models.few_shot_personalizer import FewShotPersonalizer, EMOTIONS_6


@pytest.fixture
def personalizer():
    return FewShotPersonalizer()


def _make_features(emotion_idx, noise=0.1, dim=20):
    """Create distinct feature vectors per emotion class."""
    np.random.seed(emotion_idx * 100)
    base = np.zeros(dim)
    base[emotion_idx * 3: emotion_idx * 3 + 3] = 1.0
    return base + noise * np.random.randn(dim)


class TestAddSupport:
    def test_add_single(self, personalizer):
        result = personalizer.add_support(_make_features(0), "happy")
        assert result["emotion"] == "happy"
        assert result["total_support"] == 1

    def test_shots_per_class(self, personalizer):
        personalizer.add_support(_make_features(0), "happy")
        result = personalizer.add_support(_make_features(0, noise=0.2), "happy")
        assert result["shots_per_class"]["happy"] == 2

    def test_multiple_classes(self, personalizer):
        personalizer.add_support(_make_features(0), "happy")
        result = personalizer.add_support(_make_features(1), "sad")
        assert result["total_support"] == 2
        assert result["is_adapted"] is True


class TestClassify:
    def test_no_support_returns_uniform(self, personalizer):
        result = personalizer.classify(np.random.randn(20))
        assert result["personalized"] is False
        assert result["emotion"] == "neutral"

    def test_classify_with_support(self, personalizer):
        for i in range(5):
            personalizer.add_support(_make_features(0, noise=0.05), "happy")
            personalizer.add_support(_make_features(1, noise=0.05), "sad")
        result = personalizer.classify(_make_features(0, noise=0.1))
        assert result["personalized"] is True
        assert result["emotion"] == "happy"

    def test_classify_correct_class(self, personalizer):
        # Add distinct prototypes
        for _ in range(5):
            personalizer.add_support(_make_features(0, noise=0.05), "happy")
            personalizer.add_support(_make_features(1, noise=0.05), "sad")
            personalizer.add_support(_make_features(2, noise=0.05), "angry")
        result = personalizer.classify(_make_features(1, noise=0.1))
        assert result["emotion"] == "sad"

    def test_probabilities_sum_to_one(self, personalizer):
        personalizer.add_support(_make_features(0), "happy")
        personalizer.add_support(_make_features(1), "sad")
        result = personalizer.classify(_make_features(0))
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_confidence_levels(self, personalizer):
        personalizer.add_support(_make_features(0), "happy")
        personalizer.add_support(_make_features(1), "sad")
        result = personalizer.classify(_make_features(0))
        assert result["confidence"] in ("high", "medium", "low")

    def test_best_similarity(self, personalizer):
        personalizer.add_support(_make_features(0), "happy")
        personalizer.add_support(_make_features(1), "sad")
        result = personalizer.classify(_make_features(0))
        assert -1 <= result["best_similarity"] <= 1


class TestAdaptation:
    def test_not_adapted_with_one_class(self, personalizer):
        personalizer.add_support(_make_features(0), "happy")
        assert personalizer.is_adapted() is False

    def test_adapted_with_two_classes(self, personalizer):
        personalizer.add_support(_make_features(0), "happy")
        personalizer.add_support(_make_features(1), "sad")
        assert personalizer.is_adapted() is True

    def test_progressive_improvement(self, personalizer):
        # More support → better classification
        personalizer.add_support(_make_features(0, noise=0.05), "happy")
        personalizer.add_support(_make_features(1, noise=0.05), "sad")
        result_few = personalizer.classify(_make_features(0, noise=0.1))

        for _ in range(10):
            personalizer.add_support(_make_features(0, noise=0.05), "happy")
            personalizer.add_support(_make_features(1, noise=0.05), "sad")
        result_many = personalizer.classify(_make_features(0, noise=0.1))

        # Both should classify correctly
        assert result_few["emotion"] == "happy"
        assert result_many["emotion"] == "happy"


class TestStatus:
    def test_empty_status(self, personalizer):
        status = personalizer.get_status()
        assert status["is_adapted"] is False
        assert status["total_support"] == 0

    def test_status_with_data(self, personalizer):
        personalizer.add_support(_make_features(0), "happy")
        personalizer.add_support(_make_features(1), "sad")
        status = personalizer.get_status()
        assert status["n_classes"] == 2
        assert "happy" in status["adapted_emotions"]
        assert "sad" in status["adapted_emotions"]


class TestPrototypes:
    def test_get_prototypes(self, personalizer):
        personalizer.add_support(_make_features(0), "happy")
        protos = personalizer.get_prototypes()
        assert "happy" in protos
        assert len(protos["happy"]) == 20


class TestEuclidean:
    def test_euclidean_mode(self):
        p = FewShotPersonalizer(distance_metric="euclidean")
        for _ in range(3):
            p.add_support(_make_features(0, noise=0.05), "happy")
            p.add_support(_make_features(1, noise=0.05), "sad")
        result = p.classify(_make_features(0, noise=0.1))
        assert result["emotion"] == "happy"


class TestMultiUser:
    def test_independent_users(self, personalizer):
        personalizer.add_support(_make_features(0), "happy", user_id="a")
        personalizer.add_support(_make_features(1), "sad", user_id="b")
        assert personalizer.get_status("a")["total_support"] == 1
        assert personalizer.get_status("b")["total_support"] == 1


class TestReset:
    def test_reset_clears(self, personalizer):
        personalizer.add_support(_make_features(0), "happy")
        personalizer.add_support(_make_features(1), "sad")
        personalizer.reset()
        assert personalizer.is_adapted() is False
        assert personalizer.get_status()["total_support"] == 0
