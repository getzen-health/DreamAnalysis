"""Tests for EmotionTrajectoryPredictor."""

import time

import pytest


# ── Required output keys ──────────────────────────────────────────────────────
REQUIRED_KEYS = {
    "current_valence",
    "current_arousal",
    "predicted_valence_5s",
    "predicted_arousal_5s",
    "predicted_valence_15s",
    "predicted_arousal_15s",
    "predicted_valence_30s",
    "predicted_arousal_30s",
    "valence_trend",
    "arousal_trend",
    "valence_velocity",
    "arousal_velocity",
    "predicted_emotion_5s",
    "confidence",
    "history_length",
    "model_type",
}

VALID_EMOTIONS = {"happy", "excited", "calm", "neutral", "sad", "stressed"}
VALID_TRENDS = {"rising", "falling", "stable"}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def predictor():
    from models.emotion_trajectory_predictor import EmotionTrajectoryPredictor
    return EmotionTrajectoryPredictor(history_seconds=30, fs=1.0)


@pytest.fixture
def fresh_predictor():
    """Predictor guaranteed to have no history."""
    from models.emotion_trajectory_predictor import EmotionTrajectoryPredictor
    p = EmotionTrajectoryPredictor()
    p.reset()
    return p


# ── No history ────────────────────────────────────────────────────────────────

class TestNoHistory:
    def test_predict_returns_dict_when_empty(self, fresh_predictor):
        result = fresh_predictor.predict()
        assert isinstance(result, dict)

    def test_predict_has_all_required_keys_when_empty(self, fresh_predictor):
        result = fresh_predictor.predict()
        missing = REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_predict_neutral_when_empty(self, fresh_predictor):
        result = fresh_predictor.predict()
        assert result["predicted_emotion_5s"] == "neutral"

    def test_predict_zero_valence_arousal_when_empty(self, fresh_predictor):
        result = fresh_predictor.predict()
        assert result["current_valence"] == 0.0
        assert result["current_arousal"] == 0.0

    def test_predict_zero_confidence_when_empty(self, fresh_predictor):
        result = fresh_predictor.predict()
        assert result["confidence"] == 0.0

    def test_predict_history_length_zero_when_empty(self, fresh_predictor):
        result = fresh_predictor.predict()
        assert result["history_length"] == 0

    def test_predict_stable_trend_when_empty(self, fresh_predictor):
        result = fresh_predictor.predict()
        assert result["valence_trend"] == "stable"
        assert result["arousal_trend"] == "stable"


# ── Single update ─────────────────────────────────────────────────────────────

class TestSingleUpdate:
    def test_update_then_predict_cycle(self, predictor):
        predictor.update(0.5, 0.6)
        result = predictor.predict()
        assert isinstance(result, dict)
        assert result["history_length"] == 1

    def test_single_update_returns_all_keys(self, predictor):
        predictor.update(0.5, 0.6)
        result = predictor.predict()
        missing = REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_single_update_current_values_correct(self, predictor):
        predictor.update(0.5, 0.6)
        result = predictor.predict()
        assert abs(result["current_valence"] - 0.5) < 1e-3
        assert abs(result["current_arousal"] - 0.6) < 1e-3


# ── Output value range checks ─────────────────────────────────────────────────

class TestOutputRanges:
    def setup_method(self):
        from models.emotion_trajectory_predictor import EmotionTrajectoryPredictor
        self.predictor = EmotionTrajectoryPredictor()
        for i in range(10):
            self.predictor.update(0.4 + i * 0.01, 0.5 + i * 0.01, timestamp=float(i))

    def test_predicted_valence_5s_in_valid_range(self):
        result = self.predictor.predict()
        assert -1.0 <= result["predicted_valence_5s"] <= 1.0

    def test_predicted_valence_15s_in_valid_range(self):
        result = self.predictor.predict()
        assert -1.0 <= result["predicted_valence_15s"] <= 1.0

    def test_predicted_valence_30s_in_valid_range(self):
        result = self.predictor.predict()
        assert -1.0 <= result["predicted_valence_30s"] <= 1.0

    def test_predicted_arousal_5s_in_valid_range(self):
        result = self.predictor.predict()
        assert 0.0 <= result["predicted_arousal_5s"] <= 1.0

    def test_predicted_arousal_15s_in_valid_range(self):
        result = self.predictor.predict()
        assert 0.0 <= result["predicted_arousal_15s"] <= 1.0

    def test_predicted_arousal_30s_in_valid_range(self):
        result = self.predictor.predict()
        assert 0.0 <= result["predicted_arousal_30s"] <= 1.0

    def test_confidence_in_valid_range(self):
        result = self.predictor.predict()
        assert 0.0 <= result["confidence"] <= 1.0

    def test_valence_trend_valid_value(self):
        result = self.predictor.predict()
        assert result["valence_trend"] in VALID_TRENDS

    def test_arousal_trend_valid_value(self):
        result = self.predictor.predict()
        assert result["arousal_trend"] in VALID_TRENDS

    def test_predicted_emotion_5s_valid_label(self):
        result = self.predictor.predict()
        assert result["predicted_emotion_5s"] in VALID_EMOTIONS

    def test_model_type_is_holt_winters(self):
        result = self.predictor.predict()
        assert result["model_type"] == "holt_winters"


# ── Rising valence trend predicts positive future valence ─────────────────────

class TestRisingValenceTrend:
    def test_rising_valence_predicts_positive_future(self):
        """When valence is consistently rising, predicted 5s valence > current."""
        from models.emotion_trajectory_predictor import EmotionTrajectoryPredictor
        p = EmotionTrajectoryPredictor(alpha=0.5, beta=0.3)
        # Feed a clearly rising valence sequence
        start_ts = 1000.0
        for i in range(20):
            p.update(valence=-0.5 + i * 0.05, arousal=0.5, timestamp=start_ts + i)
        result = p.predict()
        assert result["valence_trend"] == "rising", (
            f"Expected 'rising', got '{result['valence_trend']}'; "
            f"velocity={result['valence_velocity']}"
        )
        assert result["predicted_valence_5s"] >= result["current_valence"] - 0.01, (
            "Rising trend should predict valence >= current (allowing tiny rounding)"
        )

    def test_falling_valence_trend_detected(self):
        """When valence is consistently falling, trend is 'falling'."""
        from models.emotion_trajectory_predictor import EmotionTrajectoryPredictor
        p = EmotionTrajectoryPredictor(alpha=0.5, beta=0.3)
        start_ts = 2000.0
        for i in range(20):
            p.update(valence=0.5 - i * 0.05, arousal=0.5, timestamp=start_ts + i)
        result = p.predict()
        assert result["valence_trend"] == "falling", (
            f"Expected 'falling', got '{result['valence_trend']}'; "
            f"velocity={result['valence_velocity']}"
        )

    def test_stable_valence_trend_detected(self):
        """When valence is flat, trend is 'stable'."""
        from models.emotion_trajectory_predictor import EmotionTrajectoryPredictor
        p = EmotionTrajectoryPredictor(alpha=0.3, beta=0.1)
        start_ts = 3000.0
        for i in range(20):
            p.update(valence=0.3, arousal=0.5, timestamp=start_ts + i)
        result = p.predict()
        assert result["valence_trend"] == "stable", (
            f"Expected 'stable', got '{result['valence_trend']}'"
        )


# ── Emotion mapping ───────────────────────────────────────────────────────────

class TestEmotionMapping:
    """Test that valence/arousal coordinates map to expected emotions."""

    def _predict_with_fixed_state(self, valence, arousal):
        from models.emotion_trajectory_predictor import EmotionTrajectoryPredictor
        p = EmotionTrajectoryPredictor()
        # Feed identical readings so HW level/trend converges to the target
        for i in range(15):
            p.update(valence, arousal, timestamp=float(i))
        return p.predict()

    def test_happy_mapping(self):
        result = self._predict_with_fixed_state(0.5, 0.6)
        assert result["predicted_emotion_5s"] in {"happy", "excited"}

    def test_calm_mapping(self):
        result = self._predict_with_fixed_state(0.2, 0.2)
        assert result["predicted_emotion_5s"] == "calm"

    def test_sad_mapping(self):
        result = self._predict_with_fixed_state(-0.4, 0.3)
        assert result["predicted_emotion_5s"] == "sad"

    def test_stressed_mapping(self):
        result = self._predict_with_fixed_state(-0.3, 0.8)
        assert result["predicted_emotion_5s"] == "stressed"


# ── Reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_history(self, predictor):
        for i in range(5):
            predictor.update(0.5, 0.5, timestamp=float(i))
        predictor.reset()
        result = predictor.predict()
        assert result["history_length"] == 0

    def test_reset_then_update_works(self, predictor):
        for i in range(5):
            predictor.update(0.5, 0.5, timestamp=float(i))
        predictor.reset()
        predictor.update(0.3, 0.4)
        result = predictor.predict()
        assert result["history_length"] == 1


# ── get_trajectory ────────────────────────────────────────────────────────────

class TestGetTrajectory:
    def test_trajectory_has_required_keys(self, predictor):
        predictor.update(0.3, 0.4)
        traj = predictor.get_trajectory()
        for key in ("valence_history", "arousal_history", "history_length", "history_seconds"):
            assert key in traj, f"Missing key: {key}"

    def test_trajectory_history_length_matches(self, predictor):
        for i in range(5):
            predictor.update(0.3, 0.4, timestamp=float(i))
        traj = predictor.get_trajectory()
        assert traj["history_length"] == 5
        assert len(traj["valence_history"]) == 5
        assert len(traj["arousal_history"]) == 5


# ── Per-user singleton ────────────────────────────────────────────────────────

class TestPerUserSingleton:
    def test_different_users_get_different_instances(self):
        from models.emotion_trajectory_predictor import get_trajectory_predictor
        p1 = get_trajectory_predictor("user_a")
        p2 = get_trajectory_predictor("user_b")
        assert p1 is not p2

    def test_same_user_gets_same_instance(self):
        from models.emotion_trajectory_predictor import get_trajectory_predictor
        p1 = get_trajectory_predictor("user_singleton_test")
        p2 = get_trajectory_predictor("user_singleton_test")
        assert p1 is p2

    def test_singleton_persists_history(self):
        from models.emotion_trajectory_predictor import get_trajectory_predictor
        user = "user_history_persist"
        p = get_trajectory_predictor(user)
        p.reset()
        p.update(0.5, 0.5)
        p2 = get_trajectory_predictor(user)
        assert p2.predict()["history_length"] == 1


# ── Input clamping ────────────────────────────────────────────────────────────

class TestInputClamping:
    def test_valence_clamped_to_minus_one(self, predictor):
        predictor.update(-5.0, 0.5)
        result = predictor.predict()
        assert result["current_valence"] >= -1.0

    def test_valence_clamped_to_plus_one(self, predictor):
        predictor.update(5.0, 0.5)
        result = predictor.predict()
        assert result["current_valence"] <= 1.0

    def test_arousal_clamped_to_zero(self, predictor):
        predictor.update(0.0, -0.5)
        result = predictor.predict()
        assert result["current_arousal"] >= 0.0

    def test_arousal_clamped_to_one(self, predictor):
        predictor.update(0.0, 5.0)
        result = predictor.predict()
        assert result["current_arousal"] <= 1.0
