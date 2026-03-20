"""Tests for evidence-based validation of creativity and flow state detectors.

Issue #473: Creativity (99.18%) and flow (62.86%) models need evidence-based validation.

Tests cover:
1. Creativity experimental labeling in output
2. Flow binary mode
3. Flow calibration enforcement
4. Flow epoch length validation
"""

import warnings
import numpy as np
import pytest

from models.creativity_detector import CreativityDetector
from models.flow_state_detector import (
    FlowStateDetector,
    FLOW_STATES,
    FLOW_STATES_BINARY,
    MIN_EPOCH_SECONDS,
    FLOW_MODEL_ACCURACY,
    FLOW_ACCURACY_NOTE,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def creativity_detector():
    """Uncalibrated creativity detector (no saved model)."""
    return CreativityDetector()


@pytest.fixture
def flow_detector():
    """Uncalibrated flow state detector (no saved model)."""
    return FlowStateDetector()


@pytest.fixture
def calibrated_flow_detector():
    """Flow state detector calibrated with resting EEG."""
    det = FlowStateDetector()
    resting = np.random.randn(256 * 60) * 20  # 60 seconds at 256 Hz
    det.calibrate(resting, fs=256.0)
    return det


@pytest.fixture
def short_eeg():
    """4 seconds of EEG at 256 Hz — below MIN_EPOCH_SECONDS."""
    return np.random.randn(256 * 4) * 20


@pytest.fixture
def long_eeg():
    """60 seconds of EEG at 256 Hz — above MIN_EPOCH_SECONDS."""
    return np.random.randn(256 * 60) * 20


@pytest.fixture
def multichannel_long_eeg():
    """4 channels x 60 seconds of EEG at 256 Hz."""
    return np.random.randn(4, 256 * 60) * 20


# ─── Creativity: Experimental Labeling ────────────────────────────────────────


class TestCreativityExperimentalLabeling:
    """Verify that the creativity detector marks all output as experimental."""

    def test_class_level_experimental_flag(self, creativity_detector):
        """CreativityDetector.EXPERIMENTAL must be True."""
        assert CreativityDetector.EXPERIMENTAL is True

    def test_class_level_confidence_note(self, creativity_detector):
        """CreativityDetector.CONFIDENCE_NOTE must mention overfit and sample size."""
        note = CreativityDetector.CONFIDENCE_NOTE
        assert "experimental" in note.lower()
        assert "850" in note
        assert "overfit" in note.lower()

    def test_predict_output_contains_experimental_flag(self, creativity_detector, short_eeg):
        """predict() output must include experimental: True."""
        result = creativity_detector.predict(short_eeg, fs=256.0)
        assert "experimental" in result
        assert result["experimental"] is True

    def test_predict_output_contains_confidence_note(self, creativity_detector, short_eeg):
        """predict() output must include a confidence_note string."""
        result = creativity_detector.predict(short_eeg, fs=256.0)
        assert "confidence_note" in result
        assert isinstance(result["confidence_note"], str)
        assert len(result["confidence_note"]) > 10

    def test_predict_still_returns_normal_fields(self, creativity_detector, short_eeg):
        """Experimental labeling must not break existing output fields."""
        result = creativity_detector.predict(short_eeg, fs=256.0)
        assert "state" in result
        assert "creativity_score" in result
        assert "confidence" in result
        assert "components" in result
        assert result["state"] in CreativityDetector.STATES

    def test_multichannel_predict_contains_experimental(self, creativity_detector):
        """Multichannel input should also produce experimental fields."""
        eeg = np.random.randn(4, 1024) * 20
        result = creativity_detector.predict(eeg, fs=256.0)
        assert result["experimental"] is True
        assert "confidence_note" in result


# ─── Flow: Binary Mode ───────────────────────────────────────────────────────


class TestFlowBinaryMode:
    """Verify binary flow/no-flow mode works correctly."""

    def test_binary_mode_returns_two_states(self, calibrated_flow_detector, long_eeg):
        """In binary mode, state must be 'flow' or 'no_flow'."""
        result = calibrated_flow_detector.predict(long_eeg, fs=256.0, binary=True)
        assert result["state"] in FLOW_STATES_BINARY

    def test_binary_mode_flag_in_output(self, calibrated_flow_detector, long_eeg):
        """binary_mode key must be True when binary=True."""
        result = calibrated_flow_detector.predict(long_eeg, fs=256.0, binary=True)
        assert result["binary_mode"] is True

    def test_non_binary_mode_flag_in_output(self, calibrated_flow_detector, long_eeg):
        """binary_mode key must be False when binary=False (default)."""
        result = calibrated_flow_detector.predict(long_eeg, fs=256.0, binary=False)
        assert result["binary_mode"] is False

    def test_default_is_non_binary(self, calibrated_flow_detector, long_eeg):
        """Default predict() should use 4-state mode."""
        result = calibrated_flow_detector.predict(long_eeg, fs=256.0)
        assert result["binary_mode"] is False
        assert result["state"] in FLOW_STATES

    def test_binary_state_index_range(self, calibrated_flow_detector, long_eeg):
        """In binary mode, state_index should be 0 or 1."""
        result = calibrated_flow_detector.predict(long_eeg, fs=256.0, binary=True)
        assert result["state_index"] in (0, 1)

    def test_binary_threshold_at_045(self, calibrated_flow_detector, long_eeg):
        """Binary mode uses 0.45 as threshold (moderate/shallow boundary)."""
        result = calibrated_flow_detector.predict(long_eeg, fs=256.0, binary=True)
        if result["flow_score"] >= 0.45:
            assert result["state"] == "flow"
        else:
            assert result["state"] == "no_flow"


# ─── Flow: Calibration Enforcement ───────────────────────────────────────────


class TestFlowCalibrationEnforcement:
    """Verify calibration is tracked and warnings are issued."""

    def test_uncalibrated_by_default(self, flow_detector):
        """New detector must not be calibrated."""
        assert flow_detector.is_calibrated is False

    def test_calibrated_after_calibrate(self, flow_detector, long_eeg):
        """After calibrate(), is_calibrated must be True."""
        flow_detector.calibrate(long_eeg, fs=256.0)
        assert flow_detector.is_calibrated is True

    def test_uncalibrated_predict_has_warning(self, flow_detector, long_eeg):
        """Predicting without calibration must include calibration_warning."""
        result = flow_detector.predict(long_eeg, fs=256.0)
        assert "calibration_warning" in result
        assert "calibrate" in result["calibration_warning"].lower()

    def test_calibrated_predict_no_warning(self, calibrated_flow_detector, long_eeg):
        """Predicting after calibration must not include calibration_warning."""
        result = calibrated_flow_detector.predict(long_eeg, fs=256.0)
        assert "calibration_warning" not in result

    def test_uncalibrated_predict_emits_python_warning(self, flow_detector, long_eeg):
        """Predicting without calibration must emit a Python warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            flow_detector.predict(long_eeg, fs=256.0)
            cal_warnings = [x for x in w if "calibrate" in str(x.message).lower()]
            assert len(cal_warnings) >= 1

    def test_calibrated_predict_still_returns_results(self, calibrated_flow_detector, long_eeg):
        """Calibration enforcement must not break normal output."""
        result = calibrated_flow_detector.predict(long_eeg, fs=256.0)
        assert "flow_score" in result
        assert "state" in result
        assert "components" in result


# ─── Flow: Epoch Length Validation ────────────────────────────────────────────


class TestFlowEpochLengthValidation:
    """Verify minimum epoch length enforcement."""

    def test_short_epoch_has_warning(self, calibrated_flow_detector, short_eeg):
        """Epochs shorter than MIN_EPOCH_SECONDS must include epoch_length_warning."""
        result = calibrated_flow_detector.predict(short_eeg, fs=256.0)
        assert "epoch_length_warning" in result
        assert "30" in result["epoch_length_warning"]

    def test_long_epoch_no_warning(self, calibrated_flow_detector, long_eeg):
        """Epochs at or above MIN_EPOCH_SECONDS must not include epoch_length_warning."""
        result = calibrated_flow_detector.predict(long_eeg, fs=256.0)
        assert "epoch_length_warning" not in result

    def test_short_epoch_emits_python_warning(self, calibrated_flow_detector, short_eeg):
        """Short epochs must emit a Python warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calibrated_flow_detector.predict(short_eeg, fs=256.0)
            epoch_warnings = [x for x in w if "epoch" in str(x.message).lower()]
            assert len(epoch_warnings) >= 1

    def test_short_epoch_still_returns_results(self, calibrated_flow_detector, short_eeg):
        """Short epochs should still produce results (with warning)."""
        result = calibrated_flow_detector.predict(short_eeg, fs=256.0)
        assert "flow_score" in result
        assert "state" in result

    def test_min_epoch_constant(self):
        """MIN_EPOCH_SECONDS must be 30 (per flow state research)."""
        assert MIN_EPOCH_SECONDS == 30.0


# ─── Flow: Accuracy Metadata ─────────────────────────────────────────────────


class TestFlowAccuracyMetadata:
    """Verify accuracy information is surfaced in output."""

    def test_model_accuracy_in_output(self, calibrated_flow_detector, long_eeg):
        """predict() must include model_accuracy field."""
        result = calibrated_flow_detector.predict(long_eeg, fs=256.0)
        assert "model_accuracy" in result
        assert result["model_accuracy"] == 62.86

    def test_accuracy_note_in_output(self, calibrated_flow_detector, long_eeg):
        """predict() must include accuracy_note field."""
        result = calibrated_flow_detector.predict(long_eeg, fs=256.0)
        assert "accuracy_note" in result
        assert "62.86%" in result["accuracy_note"]
        assert "binary" in result["accuracy_note"].lower()

    def test_module_level_constants(self):
        """Module-level accuracy constants must exist."""
        assert FLOW_MODEL_ACCURACY == 62.86
        assert isinstance(FLOW_ACCURACY_NOTE, str)
        assert len(FLOW_ACCURACY_NOTE) > 10
