"""Tests for binaural beat neurofeedback controller."""
import pytest
from models.binaural_feedback import BinauralFeedbackController, ENTRAINMENT_TARGETS


def test_start_session():
    ctrl = BinauralFeedbackController()
    result = ctrl.start_session("focus", volume=0.5)
    assert result["session_active"] is True
    assert result["target_state"] == "focus"
    assert result["beat_frequency_hz"] == 15.0
    assert result["volume"] == 0.5


def test_all_presets_valid():
    ctrl = BinauralFeedbackController()
    for state in ENTRAINMENT_TARGETS:
        result = ctrl.start_session(state)
        assert result["session_active"] is True
        assert result["beat_frequency_hz"] > 0


def test_invalid_state_falls_back_to_relax():
    ctrl = BinauralFeedbackController()
    result = ctrl.start_session("nonexistent_state")
    assert result["target_state"] == "relax"


def test_update_from_eeg():
    ctrl = BinauralFeedbackController()
    ctrl.start_session("relax")
    # EEG with low alpha (user not in target band yet)
    features = {"alpha": 0.1, "beta": 0.6, "theta": 0.2, "delta": 0.1}
    result = ctrl.update_from_eeg(features)
    assert "beat_frequency_hz" in result
    assert "entrainment_score" in result
    assert 0.0 <= result["entrainment_score"] <= 1.0


def test_stop_session():
    ctrl = BinauralFeedbackController()
    ctrl.start_session("focus")
    result = ctrl.stop_session()
    assert result["status"] == "stopped"
    assert ctrl.session_active is False


def test_beat_frequency_in_valid_range():
    ctrl = BinauralFeedbackController()
    ctrl.start_session("meditation")
    for _ in range(20):
        features = {"alpha": 0.2, "beta": 0.5, "theta": 0.2, "delta": 0.1}
        result = ctrl.update_from_eeg(features)
        assert 0.5 <= result["beat_frequency_hz"] <= 40.0
