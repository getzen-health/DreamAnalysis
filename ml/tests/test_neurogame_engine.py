"""Tests for neurogame engine."""
import numpy as np
import pytest

from models.neurogame_engine import NeurogameEngine, GAME_COMMANDS, DIFFICULTY_LEVELS


@pytest.fixture
def engine():
    return NeurogameEngine()


class TestCalibration:
    def test_calibrate_returns_thresholds(self, engine):
        result = engine.calibrate(focus_beta_theta=2.0, relax_alpha_beta=1.8)
        assert result["calibrated"] is True
        assert "focus_threshold" in result
        assert "relax_threshold" in result

    def test_calibration_sets_difficulty(self, engine):
        engine.calibrate(2.0, 1.8)
        assert engine._difficulty["default"] == 0.5

    def test_calibration_per_user(self, engine):
        engine.calibrate(2.0, 1.8, user_id="alice")
        engine.calibrate(3.0, 2.5, user_id="bob")
        assert engine._calibration["alice"]["focus_mean"] == 2.0
        assert engine._calibration["bob"]["focus_mean"] == 3.0


class TestGameCommand:
    def test_output_keys(self, engine):
        result = engine.get_command(0.3, 0.4, 0.5)
        expected = {"command", "intensity", "engagement_level",
                    "difficulty_adjustment", "difficulty_value",
                    "difficulty_level", "focus_score", "relax_score", "calibrated"}
        assert expected.issubset(set(result.keys()))

    def test_command_is_valid(self, engine):
        result = engine.get_command(0.3, 0.4, 0.5)
        assert result["command"] in GAME_COMMANDS

    def test_focus_command(self, engine):
        # High beta, low theta → focus_boost
        result = engine.get_command(theta_power=0.1, alpha_power=0.2, beta_power=0.8)
        assert result["command"] == "focus_boost"

    def test_relax_command(self, engine):
        # High alpha, low beta → relax_action
        result = engine.get_command(theta_power=0.1, alpha_power=0.8, beta_power=0.1)
        assert result["command"] == "relax_action"

    def test_idle_command(self, engine):
        # Balanced powers → idle
        result = engine.get_command(theta_power=0.4, alpha_power=0.4, beta_power=0.4)
        assert result["command"] == "idle"

    def test_intensity_range(self, engine):
        result = engine.get_command(0.1, 0.2, 0.9)
        assert 0 <= result["intensity"] <= 1

    def test_engagement_range(self, engine):
        result = engine.get_command(0.3, 0.3, 0.5)
        assert 0 <= result["engagement_level"] <= 1

    def test_calibrated_flag_without_cal(self, engine):
        result = engine.get_command(0.3, 0.3, 0.5)
        assert result["calibrated"] is False

    def test_calibrated_flag_with_cal(self, engine):
        engine.calibrate(2.0, 1.5)
        result = engine.get_command(0.3, 0.3, 0.5)
        assert result["calibrated"] is True


class TestAdaptiveDifficulty:
    def test_difficulty_label_valid(self, engine):
        result = engine.get_command(0.3, 0.3, 0.5)
        assert result["difficulty_level"] in DIFFICULTY_LEVELS

    def test_difficulty_increases_with_engagement(self, engine):
        # High engagement should increase difficulty over time
        engine.calibrate(1.0, 1.0)
        for _ in range(20):
            result = engine.get_command(theta_power=0.1, alpha_power=0.1, beta_power=0.8)
        assert result["difficulty_value"] > 0.5

    def test_difficulty_decreases_with_boredom(self, engine):
        engine.calibrate(1.0, 1.0)
        # Start at 0.5 difficulty
        # High alpha + low beta = bored
        for _ in range(20):
            result = engine.get_command(theta_power=0.1, alpha_power=0.8, beta_power=0.1)
        assert result["difficulty_value"] < 0.5

    def test_difficulty_bounded(self, engine):
        for _ in range(100):
            result = engine.get_command(0.1, 0.1, 0.9)
        assert 0 <= result["difficulty_value"] <= 1


class TestSessionStats:
    def test_empty_stats(self, engine):
        stats = engine.get_session_stats()
        assert stats["total_commands"] == 0

    def test_stats_with_data(self, engine):
        for _ in range(5):
            engine.get_command(0.3, 0.3, 0.5)
        stats = engine.get_session_stats()
        assert stats["total_commands"] == 5
        assert "mean_engagement" in stats
        assert "command_distribution" in stats

    def test_command_distribution(self, engine):
        engine.get_command(0.1, 0.2, 0.9)  # focus
        engine.get_command(0.1, 0.8, 0.1)  # relax
        stats = engine.get_session_stats()
        assert "focus_boost" in stats["command_distribution"]
        assert "relax_action" in stats["command_distribution"]


class TestHistory:
    def test_empty_history(self, engine):
        assert engine.get_history() == []

    def test_history_grows(self, engine):
        engine.get_command(0.3, 0.3, 0.5)
        engine.get_command(0.3, 0.3, 0.5)
        assert len(engine.get_history()) == 2

    def test_history_last_n(self, engine):
        for _ in range(10):
            engine.get_command(0.3, 0.3, 0.5)
        assert len(engine.get_history(last_n=3)) == 3

    def test_history_cap(self):
        eng = NeurogameEngine(max_history=50)
        for _ in range(60):
            eng.get_command(0.3, 0.3, 0.5)
        assert len(eng.get_history()) == 50


class TestMultiUser:
    def test_independent_users(self, engine):
        engine.get_command(0.1, 0.1, 0.9, user_id="alice")
        engine.get_command(0.1, 0.8, 0.1, user_id="bob")
        alice_stats = engine.get_session_stats("alice")
        bob_stats = engine.get_session_stats("bob")
        assert alice_stats["total_commands"] == 1
        assert bob_stats["total_commands"] == 1


class TestReset:
    def test_reset_clears(self, engine):
        engine.calibrate(2.0, 1.5)
        engine.get_command(0.3, 0.3, 0.5)
        engine.reset()
        assert engine.get_session_stats()["total_commands"] == 0
        assert "default" not in engine._calibration
