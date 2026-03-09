"""Tests for EmotionTrajectoryTracker — valence-arousal dynamics."""
import pytest

from models.emotion_trajectory import EmotionTrajectoryTracker


@pytest.fixture
def tracker():
    return EmotionTrajectoryTracker()


class TestBasicTracking:
    def test_first_update(self, tracker):
        result = tracker.update(0.5, 0.7, "happy")
        assert result["valence"] == 0.5
        assert result["arousal"] == 0.7
        assert result["n_samples"] == 1

    def test_quadrant_high_positive(self, tracker):
        assert tracker.update(0.5, 0.8)["quadrant"] == "high_positive"

    def test_quadrant_low_positive(self, tracker):
        assert tracker.update(0.5, 0.2)["quadrant"] == "low_positive"

    def test_quadrant_high_negative(self, tracker):
        assert tracker.update(-0.5, 0.8)["quadrant"] == "high_negative"

    def test_quadrant_low_negative(self, tracker):
        assert tracker.update(-0.5, 0.2)["quadrant"] == "low_negative"

    def test_clamping(self, tracker):
        result = tracker.update(5.0, -3.0)
        assert result["valence"] == 1.0
        assert result["arousal"] == 0.0


class TestVelocity:
    def test_zero_velocity_first(self, tracker):
        result = tracker.update(0.5, 0.5)
        assert result["velocity_valence"] == 0
        assert result["speed"] == 0

    def test_velocity_computed(self, tracker):
        tracker.update(0.0, 0.5)
        result = tracker.update(0.3, 0.7)
        assert result["velocity_valence"] == pytest.approx(0.3, abs=0.01)

    def test_speed_magnitude(self, tracker):
        tracker.update(0.0, 0.0)
        result = tracker.update(0.3, 0.4)
        assert result["speed"] == pytest.approx(0.5, abs=0.01)


class TestTransitions:
    def test_no_transition(self, tracker):
        tracker.update(0.5, 0.8)
        result = tracker.update(0.6, 0.7)
        assert result["transition_detected"] is False

    def test_transition_detected(self, tracker):
        tracker.update(0.5, 0.8)
        result = tracker.update(-0.5, 0.8)
        assert result["transition_detected"] is True


class TestPrediction:
    def test_prediction_follows_momentum(self, tracker):
        tracker.update(0.0, 0.5)
        tracker.update(0.2, 0.5)
        result = tracker.update(0.4, 0.5)
        assert result["predicted_valence"] > 0.4


class TestTrajectory:
    def test_get_trajectory(self, tracker):
        tracker.update(0.1, 0.5)
        tracker.update(0.2, 0.6)
        traj = tracker.get_trajectory()
        assert len(traj) == 2

    def test_get_trajectory_last_n(self, tracker):
        for i in range(10):
            tracker.update(i * 0.1, 0.5)
        assert len(tracker.get_trajectory(last_n=3)) == 3

    def test_history_cap(self, tracker):
        t = EmotionTrajectoryTracker(max_history=50)
        for _ in range(60):
            t.update(0.1, 0.5)
        assert t.update(0.1, 0.5)["n_samples"] == 50


class TestSummary:
    def test_empty_summary(self, tracker):
        assert tracker.get_summary()["n_samples"] == 0

    def test_summary_stats(self, tracker):
        for _ in range(5):
            tracker.update(0.5, 0.7, "happy")
        s = tracker.get_summary()
        assert s["n_samples"] == 5
        assert s["dominant_quadrant"] == "high_positive"

    def test_transition_count(self, tracker):
        tracker.update(0.5, 0.8)
        tracker.update(-0.5, 0.8)
        tracker.update(0.5, 0.2)
        assert tracker.get_summary()["quadrant_transitions"] == 2


class TestMultiUser:
    def test_independent_users(self, tracker):
        tracker.update(0.5, 0.5, user_id="a")
        tracker.update(-0.5, 0.5, user_id="b")
        assert tracker.get_summary("a")["mean_valence"] > 0
        assert tracker.get_summary("b")["mean_valence"] < 0


class TestReset:
    def test_reset(self, tracker):
        tracker.update(0.5, 0.5)
        tracker.reset()
        assert tracker.get_summary()["n_samples"] == 0


class TestOutputKeys:
    def test_all_keys(self, tracker):
        tracker.update(0.1, 0.5)
        result = tracker.update(0.5, 0.7)
        expected = {"valence", "arousal", "emotion", "quadrant",
                    "velocity_valence", "velocity_arousal", "speed",
                    "emotional_inertia", "stability_score",
                    "transition_detected", "predicted_valence",
                    "predicted_arousal", "n_samples"}
        assert expected.issubset(set(result.keys()))
